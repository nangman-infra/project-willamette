//! Right-pane performance dashboard for the TUI.
//!
//! Pure render functions — given a [`SysSnapshot`] and a
//! [`DashboardState`] (engine-side info), produce a `Vec<Line>` the
//! TUI's render loop can hand to ratatui's `Paragraph` widget.
//!
//! Layout sections (top to bottom):
//!
//! 1. **HARDWARE** (static) — CPU brand, arch, cores, active SIMD kernel
//! 2. **CPU** (live, 1 Hz) — overall %, our process %, per-core
//! 3. **MEMORY** (live, 1 Hz) — KV cache, RSS, system total/used
//! 4. **INFERENCE** (live, per-token) — layer N/total, tok/s, tokens, elapsed
//! 5. **SAMPLING** (static per turn) — temperature, top-k, top-p, seed, sys prompt
//!
//! Each section is self-contained — if the data isn't available
//! (e.g., not generating right now, sysinfo not warmed up yet) the
//! section degrades gracefully to a neutral string.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::sysmon::SysSnapshot;

/// Engine-side info that the dashboard renders alongside system stats.
/// Populated by the worker thread from the `ChatEngine` and shared
/// with the UI thread via atomics + a snapshot copy.
#[derive(Debug, Clone)]
pub struct DashboardState {
    // ── model (static) ──
    /// Not yet rendered, kept for the upcoming preprocessor stage
    /// (Phase IV) where we'll show which .gguf file is loaded.
    #[allow(dead_code)]
    pub model_path: String,
    pub architecture: String,
    pub n_layers: u32,
    pub n_embd: u32,
    pub vocab_size: u32,
    pub model_file_bytes: u64,
    pub quant_label: String, // "I2_S (1.58b)"

    // ── kernel (static) ──
    /// "aarch64 NEON", "x86_64 scalar", etc.
    pub active_kernel: String,
    /// e.g. ("dotprod", true). Listed under kernel as ● / ○.
    pub kernel_features: Vec<(&'static str, bool)>,

    // ── sampling (per-turn) ──
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub system_prompt: Option<String>,

    // ── context budget (live) ──
    pub token_position: u32,
    pub max_seq_len: usize,

    // ── inference live (only meaningful while generating) ──
    /// Some(N) when generating, where N is the layer index the
    /// forward is currently inside (0..n_layers). None when idle.
    pub current_layer: Option<u32>,
    /// Tokens emitted in the in-flight turn (0 when idle).
    pub turn_tokens_emitted: u32,
    /// Cap for this turn (max_new_tokens). 0 when idle.
    pub turn_tokens_cap: u32,
    /// Seconds since the in-flight turn started. 0.0 when idle.
    pub turn_elapsed_secs: f64,
    /// Rolling-average tok/s for the in-flight turn. 0.0 when idle
    /// or fewer than 2 tokens emitted.
    pub turn_tok_per_sec: f64,
    /// `true` while a generation is in flight.
    pub generating: bool,

    // ── memory live ──
    /// Bytes used by the KV cache right now.
    pub kv_cache_bytes: u64,
}

impl DashboardState {
    /// Build the multi-line styled output for the right pane.
    pub fn render_lines(&self, sys: &SysSnapshot, width: u16) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = Vec::new();

        // Section: HARDWARE
        push_section_header(&mut lines, "HARDWARE");
        push_kv(
            &mut lines,
            "cpu",
            truncate(&sys.cpu_brand, width.saturating_sub(7) as usize),
        );
        push_kv(
            &mut lines,
            "arch",
            format!(
                "{} · {} cores ({} phys)",
                sys.arch, sys.logical_cores, sys.physical_cores
            ),
        );
        push_kv(&mut lines, "kernel", self.active_kernel.clone());
        for (name, active) in &self.kernel_features {
            let dot = if *active {
                Span::styled("●", Style::default().fg(Color::Green))
            } else {
                Span::styled("○", Style::default().fg(Color::DarkGray))
            };
            lines.push(Line::from(vec![
                Span::raw("  ".to_string()),
                dot,
                Span::raw(format!(" {}", name)),
            ]));
        }
        lines.push(Line::from(""));

        // Section: CPU live
        push_section_header(&mut lines, "CPU");
        lines.push(gauge_line(
            "overall",
            sys.overall_pct as f64,
            100.0,
            width.saturating_sub(2),
        ));
        lines.push(gauge_line(
            "proc",
            sys.process_pct_normalized as f64,
            100.0,
            width.saturating_sub(2),
        ));
        // Per-core mini-gauges (compact).
        for (i, pct) in sys.per_core_pct.iter().enumerate() {
            // Only show first 12 cores explicitly; collapse the rest.
            if i >= 12 {
                let remaining = sys.per_core_pct.len() - i;
                let max_rest = sys.per_core_pct[i..]
                    .iter()
                    .cloned()
                    .fold(0.0_f32, f32::max);
                lines.push(Line::from(format!(
                    "  c{}-{}  max {:.0}%",
                    i,
                    sys.per_core_pct.len() - 1,
                    max_rest
                )));
                let _ = remaining;
                break;
            }
            lines.push(per_core_line(i, *pct as f64, width.saturating_sub(2)));
        }
        lines.push(Line::from(""));

        // Section: MEMORY
        push_section_header(&mut lines, "MEMORY");
        push_kv(
            &mut lines,
            "KV",
            human_bytes(self.kv_cache_bytes),
        );
        push_kv(&mut lines, "RSS", human_bytes(sys.process_rss_bytes));
        if sys.total_mem_bytes > 0 {
            let used = sys.used_mem_bytes;
            let total = sys.total_mem_bytes;
            push_kv(
                &mut lines,
                "sys",
                format!("{} / {}", human_bytes(used), human_bytes(total)),
            );
        }
        let ctx_pct = if self.max_seq_len > 0 {
            self.token_position as f64 / self.max_seq_len as f64 * 100.0
        } else {
            0.0
        };
        lines.push(gauge_line("ctx", ctx_pct, 100.0, width.saturating_sub(2)));
        lines.push(Line::from(""));

        // Section: INFERENCE live
        push_section_header(&mut lines, "INFERENCE");
        if self.generating {
            if let Some(layer) = self.current_layer {
                push_kv(
                    &mut lines,
                    "layer",
                    format!("{:>2} / {}", layer + 1, self.n_layers),
                );
            } else {
                push_kv(&mut lines, "layer", "(starting…)".to_string());
            }
            push_kv(&mut lines, "tok/s", format!("{:.2}", self.turn_tok_per_sec));
            push_kv(
                &mut lines,
                "tokens",
                format!("{} / {}", self.turn_tokens_emitted, self.turn_tokens_cap),
            );
            push_kv(
                &mut lines,
                "elapsed",
                format!("{:.1} s", self.turn_elapsed_secs),
            );
            // ETA from rolling tok/s + remaining tokens
            let remaining = self
                .turn_tokens_cap
                .saturating_sub(self.turn_tokens_emitted);
            if self.turn_tok_per_sec > 0.0 && remaining > 0 {
                let eta = remaining as f64 / self.turn_tok_per_sec;
                push_kv(&mut lines, "eta", format!("~{:.0} s", eta));
            }
            lines.push(Line::from(Span::styled(
                "  (Esc to cancel)",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            lines.push(Line::from(Span::styled(
                "  (idle)",
                Style::default().fg(Color::DarkGray),
            )));
        }
        lines.push(Line::from(""));

        // Section: SAMPLING
        push_section_header(&mut lines, "SAMPLING");
        push_kv(&mut lines, "temp", format!("{:.2}", self.temperature));
        push_kv(&mut lines, "top-k", self.top_k.to_string());
        push_kv(&mut lines, "top-p", format!("{:.2}", self.top_p));
        push_kv(
            &mut lines,
            "rep-pen",
            format!("{:.2}", self.repetition_penalty),
        );
        push_kv(&mut lines, "seed", format!("0x{:x}", self.seed));
        push_kv(
            &mut lines,
            "sys",
            self.system_prompt
                .as_deref()
                .map(|s| truncate(s, width.saturating_sub(7) as usize))
                .unwrap_or_else(|| "(none)".to_string()),
        );
        lines.push(Line::from(""));

        // Section: MODEL
        push_section_header(&mut lines, "MODEL");
        push_kv(&mut lines, "arch", self.architecture.clone());
        push_kv(&mut lines, "layers", self.n_layers.to_string());
        push_kv(&mut lines, "embd", self.n_embd.to_string());
        push_kv(
            &mut lines,
            "vocab",
            format_compact_thousands(self.vocab_size as u64),
        );
        push_kv(&mut lines, "quant", self.quant_label.clone());
        push_kv(&mut lines, "file", human_bytes(self.model_file_bytes));

        lines
    }
}

// ── helpers ──────────────────────────────────────────────────────────

fn push_section_header(lines: &mut Vec<Line<'static>>, label: &str) {
    lines.push(Line::from(Span::styled(
        format!("── {} ──", label),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Cyan),
    )));
}

fn push_kv(lines: &mut Vec<Line<'static>>, key: &str, value: impl Into<String>) {
    let key_str = format!("  {:<7}", key);
    let val_str: String = value.into();
    lines.push(Line::from(vec![
        Span::styled(key_str, Style::default().fg(Color::DarkGray)),
        Span::raw(val_str),
    ]));
}

/// Render a gauge: "  label   ▓▓▓▓▓░░░░░░ 41%".
fn gauge_line(label: &str, value: f64, max: f64, width: u16) -> Line<'static> {
    let pct = (value / max).clamp(0.0, 1.0);
    let bar_w = 10_usize;
    let filled = (pct * bar_w as f64).round() as usize;
    let bar: String = "▓".repeat(filled) + &"░".repeat(bar_w - filled);
    let color = if pct < 0.6 {
        Color::Green
    } else if pct < 0.85 {
        Color::Yellow
    } else {
        Color::Red
    };
    let _ = width;
    Line::from(vec![
        Span::styled(
            format!("  {:<7}", label),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(bar, Style::default().fg(color)),
        Span::raw(format!(" {:>3.0}%", value)),
    ])
}

fn per_core_line(index: usize, pct: f64, width: u16) -> Line<'static> {
    let bar_w = 8_usize;
    let filled = ((pct / 100.0).clamp(0.0, 1.0) * bar_w as f64).round() as usize;
    let bar: String = "▓".repeat(filled) + &"░".repeat(bar_w - filled);
    let color = if pct < 60.0 {
        Color::Green
    } else if pct < 85.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    let _ = width;
    Line::from(vec![
        Span::styled(
            format!("  c{:<2}    ", index),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(bar, Style::default().fg(color)),
        Span::raw(format!(" {:>3.0}%", pct)),
    ])
}

fn human_bytes(b: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    if b >= GIB {
        format!("{:.2} GB", b as f64 / GIB as f64)
    } else if b >= MIB {
        format!("{:.1} MB", b as f64 / MIB as f64)
    } else if b >= KIB {
        format!("{:.1} KB", b as f64 / KIB as f64)
    } else {
        format!("{} B", b)
    }
}

fn format_compact_thousands(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut t: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    t.push('…');
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> DashboardState {
        DashboardState {
            model_path: "model.gguf".into(),
            architecture: "bitnet-b1.58".into(),
            n_layers: 30,
            n_embd: 2560,
            vocab_size: 128256,
            model_file_bytes: 1_100_000_000,
            quant_label: "I2_S (1.58b)".into(),
            active_kernel: "aarch64 NEON".into(),
            kernel_features: vec![("dotprod", true), ("SVE", false)],
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: 0xabad_1dea,
            system_prompt: None,
            token_position: 245,
            max_seq_len: 2048,
            current_layer: Some(17),
            turn_tokens_emitted: 23,
            turn_tokens_cap: 256,
            turn_elapsed_secs: 2.9,
            turn_tok_per_sec: 7.81,
            generating: true,
            kv_cache_bytes: 23 * 1024 * 1024,
        }
    }

    fn sample_snap() -> SysSnapshot {
        SysSnapshot {
            cpu_brand: "Apple M4".into(),
            arch: "aarch64",
            logical_cores: 10,
            physical_cores: 10,
            per_core_pct: vec![78.0, 72.0, 66.0, 54.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0],
            overall_pct: 41.0,
            process_pct_normalized: 18.0,
            process_rss_bytes: 1_180_000_000,
            total_mem_bytes: 16_000_000_000,
            used_mem_bytes: 3_400_000_000,
        }
    }

    #[test]
    fn render_includes_all_sections() {
        let lines = sample_state().render_lines(&sample_snap(), 40);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("");
        for header in &[
            "HARDWARE",
            "CPU",
            "MEMORY",
            "INFERENCE",
            "SAMPLING",
            "MODEL",
        ] {
            assert!(
                joined.contains(header),
                "expected section '{}' in rendered output; got:\n{}",
                header,
                joined
            );
        }
    }

    #[test]
    fn idle_state_says_idle() {
        let mut st = sample_state();
        st.generating = false;
        st.current_layer = None;
        let lines = st.render_lines(&sample_snap(), 40);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("");
        assert!(joined.contains("(idle)"));
        // Live-only fields shouldn't appear when idle. We match the
        // exact aligned key form (8-char left-pad "  layer  ") so
        // we don't false-positive on "layers" in the MODEL section.
        assert!(
            !joined.contains("  layer  "),
            "live 'layer' row should be hidden when idle"
        );
        assert!(
            !joined.contains("  tok/s  "),
            "live 'tok/s' row should be hidden when idle"
        );
        assert!(
            !joined.contains("  elapsed"),
            "live 'elapsed' row should be hidden when idle"
        );
    }

    #[test]
    fn generating_state_shows_eta_when_tok_per_sec_known() {
        let lines = sample_state().render_lines(&sample_snap(), 40);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("");
        assert!(joined.contains("eta"));
        // (256 - 23) / 7.81 ≈ 29.8 sec
        assert!(joined.contains("~30 s") || joined.contains("~29 s"));
    }

    #[test]
    fn human_bytes_buckets() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(500), "500 B");
        assert_eq!(human_bytes(2048), "2.0 KB");
        assert_eq!(human_bytes(5 * 1024 * 1024), "5.0 MB");
        assert_eq!(human_bytes(1_100_000_000), "1.02 GB");
    }

    #[test]
    fn truncate_appends_ellipsis() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hell…");
    }

    #[test]
    fn collapses_per_core_beyond_twelve() {
        let mut snap = sample_snap();
        snap.logical_cores = 32;
        snap.per_core_pct = (0..32).map(|i| (100 - i) as f32).collect();
        let lines = sample_state().render_lines(&snap, 40);
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("");
        // Should mention "c12-31" or similar collapse line.
        assert!(
            joined.contains("c12-"),
            "expected per-core collapse line for cores ≥ 13; got:\n{}",
            joined
        );
    }
}
