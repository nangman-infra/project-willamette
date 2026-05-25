//! Stage 9-E / v0.3.0 — ratatui chat TUI with operator-grade dashboard.
//!
//! Layout:
//!
//! ```text
//! ┌─ Project Willamette ─────────── ctx N/M (X%) ───────────────────┐
//! │                                                │ ── HARDWARE ── │
//! │  USR  ...                                      │ cpu ...        │
//! │                                                │ arch ...       │
//! │  BOT  ...                                      │                │
//! │                                                │ ── CPU ──      │
//! │                            chat history pane   │ overall ▓▓▓░░  │
//! │                            (markdown rendered) │ ...            │
//! │                                                │ dashboard pane │
//! │                                                │ (live perf)    │
//! ├────────────────────────────────────────────────┴────────────────┤
//! │ status: idle  ·  /help · ↑↓ history · Ctrl-R search · F1 help  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ > _                                                             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Threading:
//! * **Worker thread** owns the `ChatEngine` and runs blocking
//!   `send_user_message` calls. Sends `TokenEvent`s back to UI.
//! * **Sysmon thread** polls sysinfo at 1 Hz and pushes `SysSnapshot`s.
//! * **UI thread** redraws at ~30 fps, drains both event channels,
//!   handles keyboard / mouse. Reads `WorkerProgress` atomics for
//!   high-frequency layer-by-layer updates without flooding the
//!   token channel.
//!
//! v0.3.0 additions over v0.2.3:
//! * Right-pane perf dashboard (CPU, memory, inference live, model).
//! * Readline-grade input editing via `InputEditor` — arrows, Home/End,
//!   Ctrl-W/U/K/A/E, Up/Down history, Ctrl-R reverse search.
//! * Mouse wheel scroll, bracketed paste.
//! * Real terminal cursor via `Frame::set_cursor_position`.
//! * Esc as mid-turn cancel (was: no-op flash).
//! * F1 help overlay popup.
//! * Tab completion for slash commands; typo suggestion via
//!   Levenshtein distance.
//! * Persisted input history at `~/.config/willamette/history`.

use std::io::Stdout;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Terminal;

use super::dashboard::DashboardState;
use super::engine::WorkerProgress;
use super::input_editor::InputEditor;
use super::sysmon::{snapshot_now, spawn_sysmon, SysSnapshot};
use super::ChatEngine;

// ── messages between threads ─────────────────────────────────────────

enum UserCmd {
    Send(String),
    Reset,
    SetSys(Option<String>),
    Shutdown,
}

enum TokenEvent {
    Chunk(String),
    Done { secs: f64, chars: usize },
    Failed(String),
    StateChanged { token_position: u32 },
}

// ── chat-log display message ────────────────────────────────────────

#[derive(Clone)]
struct DisplayMsg {
    role: Role,
    content: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Bot,
    System,
}

impl Role {
    fn label(self) -> &'static str {
        match self {
            Role::User => "USR",
            Role::Bot => "BOT",
            Role::System => "SYS",
        }
    }
    fn color(self) -> Color {
        match self {
            Role::User => Color::Cyan,
            Role::Bot => Color::Green,
            Role::System => Color::Magenta,
        }
    }
}

// ── ui state ─────────────────────────────────────────────────────────

/// Modal overlay drawn on top of everything. Closed by any keypress
/// (for `Help`) or by submit/cancel keys (for `Search`).
enum Overlay {
    Help,
}

struct UiState {
    chat_log: Vec<DisplayMsg>,
    /// In-flight assistant response (not yet pushed to chat_log).
    streaming: String,
    /// Readline-grade input editor (cursor, history, search).
    input: InputEditor,
    /// Status text shown above input.
    status: String,
    generating: bool,
    token_position: u32,
    max_seq_len: usize,
    /// Most recent CPU/memory snapshot from sysmon.
    sys: SysSnapshot,
    /// Static + live model/sampling info for the right pane.
    dashboard: DashboardState,
    /// Shared atomics so render can read layer / tok progress.
    progress: Arc<WorkerProgress>,
    /// When the current turn started (for elapsed / tok/s computation).
    turn_start: Option<Instant>,
    /// Wrapped-line offset on the chat pane, measured from the top.
    /// Larger value = viewport shifted further down. ratatui's
    /// `Paragraph::scroll((n, 0))` uses the same convention: "skip
    /// the first n lines."
    scroll_offset: u16,
    /// True = the renderer pins the viewport to the last line every
    /// frame, so newly arriving tokens stay in view. Flipped to false
    /// the moment the user scrolls up; flipped back to true on End /
    /// Ctrl-End or when they scroll down past the last line.
    follow_bottom: bool,
    /// Modal overlay state.
    overlay: Option<Overlay>,
    /// Transient flash message (cleared after timeout).
    transient: Option<(String, Instant)>,
    /// Path to the persisted history file.
    history_path: Option<std::path::PathBuf>,
    /// Last assistant response — used for Ctrl-Y "yank".
    last_bot_text: Option<String>,
}

impl UiState {
    fn new(
        max_seq_len: usize,
        dashboard: DashboardState,
        progress: Arc<WorkerProgress>,
        history_path: Option<std::path::PathBuf>,
        loaded_history: Vec<String>,
    ) -> Self {
        Self {
            chat_log: Vec::new(),
            streaming: String::new(),
            input: InputEditor::with_history(loaded_history),
            status: "idle · type, then Enter".to_string(),
            generating: false,
            token_position: 0,
            max_seq_len,
            sys: SysSnapshot::placeholder(),
            dashboard,
            progress,
            turn_start: None,
            scroll_offset: 0,
            follow_bottom: true,
            overlay: None,
            transient: None,
            history_path,
            last_bot_text: None,
        }
    }

    fn flash(&mut self, msg: impl Into<String>) {
        self.transient = Some((msg.into(), Instant::now()));
    }
}

// ── entry point ──────────────────────────────────────────────────────

/// Run the TUI. Takes ownership of the engine; returns when the user
/// exits cleanly (Esc / Ctrl-C / `/quit`).
pub fn run_tui(mut engine: ChatEngine<'_, '_>, max_new_tokens: usize) -> Result<()> {
    // Initial dashboard data from the engine.
    let dashboard = initial_dashboard_state(&engine, max_new_tokens);
    let max_seq = engine.max_seq_len();

    // Shared progress + cancel state — wires engine forward callbacks
    // to the UI thread.
    let progress = Arc::new(WorkerProgress::new());
    engine.set_worker_progress(Arc::clone(&progress));

    // Load persisted history (best-effort).
    let history_path = persisted_history_path();
    let loaded_history = history_path
        .as_ref()
        .map(|p| load_history_file(p))
        .unwrap_or_default();

    // Channels.
    let (cmd_tx, cmd_rx) = mpsc::channel::<UserCmd>();
    let (evt_tx, evt_rx) = mpsc::channel::<TokenEvent>();
    let (sys_tx, sys_rx) = mpsc::channel::<SysSnapshot>();

    // Sysmon polling thread (daemon — its sender stops when receiver
    // drops at the end of this function).
    let sysmon_stop = spawn_sysmon(Duration::from_secs(1), sys_tx);

    // Terminal setup.
    enable_raw_mode().map_err(|e| anyhow::anyhow!("enable raw mode: {}", e))?;
    let mut stdout = std::io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    )
    .map_err(|e| anyhow::anyhow!("enter alt screen: {}", e))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal =
        Terminal::new(backend).map_err(|e| anyhow::anyhow!("ratatui terminal: {}", e))?;

    let mut ui = UiState::new(
        max_seq,
        dashboard,
        Arc::clone(&progress),
        history_path,
        loaded_history,
    );
    ui.token_position = engine.token_position();

    // Try to seed the dashboard with a real first snapshot synchronously
    // so the very first frame isn't all-zero. ~250ms cost, acceptable.
    ui.sys = snapshot_now();

    let run_result = thread::scope(|s| -> Result<()> {
        // Worker thread.
        let worker_evt_tx = evt_tx.clone();
        s.spawn(move || worker_loop(&mut engine, max_new_tokens, cmd_rx, worker_evt_tx));

        // Main UI loop.
        ui_loop(&mut terminal, &mut ui, cmd_tx, evt_rx, sys_rx)
    });

    // Terminal teardown.
    sysmon_stop.store(true, Ordering::Relaxed);
    let _ = disable_raw_mode();
    let _ = execute!(
        terminal.backend_mut(),
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    let _ = terminal.show_cursor();

    run_result
}

fn initial_dashboard_state(engine: &ChatEngine<'_, '_>, max_new_tokens: usize) -> DashboardState {
    // Single source of truth for both the dashboard label and the
    // matvec dispatch — see `src/model/dispatch.rs`. Whatever runs
    // on the matvec hot path is what gets displayed here.
    let active_kernel = crate::model::dispatch::active_kernel().label().to_string();
    let features = crate::model::dispatch::detected_features();

    let sampler = engine.sampler();
    let sp = sampler.params_clone();

    let _ = max_new_tokens;
    DashboardState {
        model_path: "(mmap)".into(),
        architecture: engine.config_architecture().to_string(),
        n_layers: engine.config_n_layers(),
        n_embd: engine.config_n_embd(),
        vocab_size: engine.config_vocab_size(),
        model_file_bytes: 0, // populated by caller if known
        quant_label: "I2_S (1.58b)".into(),
        active_kernel,
        kernel_features: features,
        temperature: sp.temperature,
        top_k: sp.top_k.unwrap_or(0),
        top_p: sp.top_p.unwrap_or(0.0),
        repetition_penalty: sp.repetition_penalty.unwrap_or(1.0),
        seed: sp.seed,
        system_prompt: engine.system_prompt().map(str::to_string),
        token_position: engine.token_position(),
        max_seq_len: engine.max_seq_len(),
        current_layer: None,
        turn_tokens_emitted: 0,
        turn_tokens_cap: 0,
        turn_elapsed_secs: 0.0,
        turn_tok_per_sec: 0.0,
        generating: false,
        kv_cache_bytes: 0,
    }
}

// ── worker thread ────────────────────────────────────────────────────

fn worker_loop<'g, 'a>(
    engine: &mut ChatEngine<'g, 'a>,
    max_new_tokens: usize,
    cmd_rx: Receiver<UserCmd>,
    evt_tx: Sender<TokenEvent>,
) {
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            UserCmd::Send(text) => {
                let start = Instant::now();
                let tx = evt_tx.clone();
                let result = engine.send_user_message(&text, max_new_tokens, |chunk| {
                    let _ = tx.send(TokenEvent::Chunk(chunk.to_string()));
                });
                let secs = start.elapsed().as_secs_f64();
                let _ = evt_tx.send(TokenEvent::StateChanged {
                    token_position: engine.token_position(),
                });
                match result {
                    Ok(response) => {
                        let _ = evt_tx.send(TokenEvent::Done {
                            secs,
                            chars: response.chars().count(),
                        });
                    }
                    Err(e) => {
                        let _ = evt_tx.send(TokenEvent::Failed(e.to_string()));
                    }
                }
            }
            UserCmd::Reset => {
                engine.reset();
                let _ = evt_tx.send(TokenEvent::StateChanged {
                    token_position: engine.token_position(),
                });
            }
            UserCmd::SetSys(s) => {
                engine.set_system_prompt(s);
            }
            UserCmd::Shutdown => break,
        }
    }
}

// ── UI loop ──────────────────────────────────────────────────────────

fn ui_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ui: &mut UiState,
    cmd_tx: Sender<UserCmd>,
    evt_rx: Receiver<TokenEvent>,
    sys_rx: Receiver<SysSnapshot>,
) -> Result<()> {
    let tick_period = Duration::from_millis(33);
    let mut last_tick = Instant::now();

    loop {
        // Pull any pending sys snapshots.
        while let Ok(snap) = sys_rx.try_recv() {
            ui.sys = snap;
        }

        // Update dashboard live fields from atomics + UiState.
        refresh_dashboard_live_fields(ui);

        terminal
            .draw(|f| render(f, ui))
            .map_err(|e| anyhow::anyhow!("draw: {}", e))?;
        // ^ `render` takes &mut: it may clamp scroll_offset to the
        // measured maximum (so PageDown spam can't strand the view
        // past the last line) and toggle follow_bottom on/off.

        // Drain token events.
        if drain_token_events(ui, &evt_rx) {
            return Ok(());
        }
        clear_transient_if_old(ui);

        let timeout = tick_period.saturating_sub(last_tick.elapsed());
        if event::poll(timeout).map_err(|e| anyhow::anyhow!("poll: {}", e))? {
            let evt = event::read().map_err(|e| anyhow::anyhow!("read: {}", e))?;
            if dispatch_event(ui, evt, &cmd_tx)? {
                let _ = cmd_tx.send(UserCmd::Shutdown);
                return Ok(());
            }
        }
        if last_tick.elapsed() >= tick_period {
            last_tick = Instant::now();
        }
    }
}

/// Dispatch one crossterm `Event` to the right handler.
/// Returns `true` if the user requested quit.
fn dispatch_event(ui: &mut UiState, evt: Event, cmd_tx: &Sender<UserCmd>) -> Result<bool> {
    match evt {
        Event::Key(key) => handle_key(ui, key, cmd_tx),
        Event::Mouse(m) => {
            handle_mouse(ui, m);
            Ok(false)
        }
        Event::Paste(text) => {
            ui.input.insert_str(&text);
            Ok(false)
        }
        _ => Ok(false),
    }
}

fn refresh_dashboard_live_fields(ui: &mut UiState) {
    let p = &ui.progress;
    let layer = p.current_layer.load(Ordering::Relaxed);
    ui.dashboard.current_layer = if layer == u32::MAX { None } else { Some(layer) };
    ui.dashboard.turn_tokens_emitted = p.tokens_emitted.load(Ordering::Relaxed);
    ui.dashboard.turn_tokens_cap = p.tokens_cap.load(Ordering::Relaxed);
    ui.dashboard.kv_cache_bytes = p.kv_cache_bytes.load(Ordering::Relaxed);
    ui.dashboard.token_position = ui.token_position;
    ui.dashboard.generating = ui.generating;

    if let Some(start) = ui.turn_start {
        let elapsed = start.elapsed().as_secs_f64();
        ui.dashboard.turn_elapsed_secs = elapsed;
        if elapsed > 0.05 && ui.dashboard.turn_tokens_emitted > 0 {
            ui.dashboard.turn_tok_per_sec = ui.dashboard.turn_tokens_emitted as f64 / elapsed;
        }
    } else {
        ui.dashboard.turn_elapsed_secs = 0.0;
        ui.dashboard.turn_tok_per_sec = 0.0;
    }
}

fn drain_token_events(ui: &mut UiState, evt_rx: &Receiver<TokenEvent>) -> bool {
    loop {
        match evt_rx.try_recv() {
            Ok(evt) => apply_token_event(ui, evt),
            Err(mpsc::TryRecvError::Empty) => return false,
            Err(mpsc::TryRecvError::Disconnected) => return true,
        }
    }
}

fn apply_token_event(ui: &mut UiState, evt: TokenEvent) {
    match evt {
        TokenEvent::Chunk(c) => ui.streaming.push_str(&c),
        TokenEvent::Done { secs, chars } => finish_bot_turn(ui, secs, chars),
        TokenEvent::Failed(msg) => fail_bot_turn(ui, msg),
        TokenEvent::StateChanged { token_position } => ui.token_position = token_position,
    }
    // Auto-scroll happens in render_chat_pane: when follow_bottom is
    // true, it overrides scroll_offset with the maximum value so the
    // newest line is always visible. We don't touch scroll_offset
    // here — the renderer needs the area width / line count to know
    // the right max, and that info isn't available until render time.
}

fn finish_bot_turn(ui: &mut UiState, secs: f64, chars: usize) {
    let resp = std::mem::take(&mut ui.streaming);
    if !resp.is_empty() {
        ui.last_bot_text = Some(resp.clone());
    }
    ui.chat_log.push(DisplayMsg {
        role: Role::Bot,
        content: resp,
    });
    ui.generating = false;
    ui.turn_start = None;
    let cps = if secs > 0.0 { chars as f64 / secs } else { 0.0 };
    ui.status = format!("idle · last turn {:.1}s ({:.1} chars/s)", secs, cps);
}

fn fail_bot_turn(ui: &mut UiState, msg: String) {
    let partial = std::mem::take(&mut ui.streaming);
    if !partial.is_empty() {
        ui.chat_log.push(DisplayMsg {
            role: Role::Bot,
            content: partial,
        });
    }
    ui.generating = false;
    ui.turn_start = None;
    ui.status = format!("error: {}", msg);
}

fn clear_transient_if_old(ui: &mut UiState) {
    if let Some((_, when)) = ui.transient {
        if when.elapsed() > Duration::from_millis(2500) {
            ui.transient = None;
        }
    }
}

// ── input handling ───────────────────────────────────────────────────

/// Returns `true` to request quit.
fn handle_key(ui: &mut UiState, key: KeyEvent, cmd_tx: &Sender<UserCmd>) -> Result<bool> {
    // Help overlay swallows any key.
    if matches!(ui.overlay, Some(Overlay::Help)) {
        ui.overlay = None;
        return Ok(false);
    }

    // Global: Ctrl-C quits.
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Ok(true);
    }

    // Reverse-search mode handles keys specially.
    if ui.input.search().is_some() {
        return handle_search_key(ui, key);
    }

    if ui.generating {
        return handle_key_while_generating(ui, key);
    }

    handle_key_normal(ui, key, cmd_tx)
}

fn handle_key_while_generating(ui: &mut UiState, key: KeyEvent) -> Result<bool> {
    match key.code {
        KeyCode::Esc => {
            // Real mid-turn cancel.
            ui.progress.cancel_requested.store(true, Ordering::Relaxed);
            ui.status = "cancelling…".to_string();
            ui.flash("cancel requested");
        }
        // Scroll while generating is fine.
        KeyCode::PageUp => scroll_up_by(ui, 10),
        KeyCode::PageDown => scroll_down_by(ui, 10),
        _ => {}
    }
    Ok(false)
}

fn handle_search_key(ui: &mut UiState, key: KeyEvent) -> Result<bool> {
    match key.code {
        KeyCode::Esc => ui.input.search_cancel(),
        KeyCode::Enter => ui.input.search_accept(),
        KeyCode::Backspace => ui.input.search_backspace(),
        KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            ui.input.begin_search() // step to next older match
        }
        KeyCode::Char(c) => ui.input.search_input(c),
        _ => {}
    }
    Ok(false)
}

fn handle_key_normal(ui: &mut UiState, key: KeyEvent, cmd_tx: &Sender<UserCmd>) -> Result<bool> {
    if key.code == KeyCode::Esc {
        // Esc while idle: quit (matches stdio chat convention).
        return Ok(true);
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && handle_ctrl_key(ui, key.code) {
        return Ok(false);
    }

    match key.code {
        KeyCode::F(1) => ui.overlay = Some(Overlay::Help),
        KeyCode::Left => ui.input.move_left(),
        KeyCode::Right => ui.input.move_right(),
        KeyCode::Home => ui.input.move_home(),
        KeyCode::End => ui.input.move_end(),
        KeyCode::Backspace => ui.input.backspace(),
        KeyCode::Delete => ui.input.delete(),
        KeyCode::Up => ui.input.history_prev(),
        KeyCode::Down => ui.input.history_next(),
        KeyCode::PageUp => scroll_up_by(ui, 10),
        KeyCode::PageDown => scroll_down_by(ui, 10),
        KeyCode::Tab => try_tab_complete_slash(ui),
        KeyCode::Enter => return handle_enter(ui, cmd_tx),
        KeyCode::Char(c) => ui.input.insert_char(c),
        _ => {}
    }
    Ok(false)
}

/// Returns true if the Ctrl-key combo was consumed.
fn handle_ctrl_key(ui: &mut UiState, code: KeyCode) -> bool {
    match code {
        KeyCode::Char('w') => ui.input.delete_word_back(),
        KeyCode::Char('u') => ui.input.delete_to_start(),
        KeyCode::Char('k') => ui.input.delete_to_end(),
        KeyCode::Char('a') => ui.input.move_home(),
        KeyCode::Char('e') => ui.input.move_end(),
        KeyCode::Char('r') => ui.input.begin_search(),
        KeyCode::Char('l') => ui.chat_log.clear(),
        KeyCode::Char('y') => yank_last_bot_response(ui),
        KeyCode::Home => scroll_to_top(ui),
        KeyCode::End => scroll_to_bottom(ui),
        _ => return false,
    }
    true
}

fn handle_enter(ui: &mut UiState, cmd_tx: &Sender<UserCmd>) -> Result<bool> {
    let line = ui.input.submit();
    persist_history_append(ui, &line);
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    if let Some(rest) = trimmed.strip_prefix('/') {
        return handle_slash(ui, rest, cmd_tx);
    }
    ui.chat_log.push(DisplayMsg {
        role: Role::User,
        content: trimmed.to_string(),
    });
    ui.streaming.clear();
    ui.generating = true;
    ui.turn_start = Some(Instant::now());
    ui.status = "generating…".to_string();
    // New turn — pin to bottom so the user sees the response stream in.
    scroll_to_bottom(ui);
    cmd_tx
        .send(UserCmd::Send(trimmed.to_string()))
        .map_err(|e| anyhow::anyhow!("send: {}", e))?;
    Ok(false)
}

fn handle_mouse(ui: &mut UiState, m: MouseEvent) {
    match m.kind {
        MouseEventKind::ScrollUp => scroll_up_by(ui, 3),
        MouseEventKind::ScrollDown => scroll_down_by(ui, 3),
        _ => {}
    }
}

/// Scroll the chat pane up by `n` wrapped lines. Detaches the viewport
/// from the bottom so newly arriving tokens won't yank the user back.
fn scroll_up_by(ui: &mut UiState, n: u16) {
    ui.follow_bottom = false;
    ui.scroll_offset = ui.scroll_offset.saturating_sub(n);
}

/// Scroll down by `n` wrapped lines. If the user goes past the last
/// line, render_chat_pane will promote them to follow_bottom = true
/// — we don't know `max_scroll` from here (no area info).
fn scroll_down_by(ui: &mut UiState, n: u16) {
    ui.follow_bottom = false;
    ui.scroll_offset = ui.scroll_offset.saturating_add(n);
}

fn scroll_to_top(ui: &mut UiState) {
    ui.follow_bottom = false;
    ui.scroll_offset = 0;
}

fn scroll_to_bottom(ui: &mut UiState) {
    ui.follow_bottom = true;
    // scroll_offset gets clamped to max_scroll inside render_chat_pane.
}

// ── slash commands (with tab completion + typo suggestion) ──────────

const KNOWN_SLASH_COMMANDS: &[&str] = &[
    "help", "quit", "exit", "q", "reset", "clear", "sys", "history", "save", "stats", "retry",
];

fn try_tab_complete_slash(ui: &mut UiState) {
    let buf = ui.input.buffer();
    if !buf.starts_with('/') {
        return;
    }
    // Split into "command" + optional "rest"
    let body = &buf[1..];
    let (partial_cmd, rest) = body.split_once(' ').unwrap_or((body, ""));
    let matches: Vec<&&str> = KNOWN_SLASH_COMMANDS
        .iter()
        .filter(|c| c.starts_with(partial_cmd))
        .collect();
    if matches.len() == 1 {
        let full = matches[0];
        let new_buf = if rest.is_empty() {
            format!("/{} ", full)
        } else {
            format!("/{} {}", full, rest)
        };
        ui.input.clear();
        ui.input.insert_str(&new_buf);
    } else if matches.len() > 1 {
        let suggestions: Vec<String> = matches.iter().take(6).map(|s| format!("/{}", s)).collect();
        ui.flash(suggestions.join("  "));
    } else {
        ui.flash(format!("no completion for /{}", partial_cmd));
    }
}

fn handle_slash(ui: &mut UiState, cmd: &str, cmd_tx: &Sender<UserCmd>) -> Result<bool> {
    let (head, rest) = cmd
        .split_once(char::is_whitespace)
        .map(|(h, r)| (h, r.trim()))
        .unwrap_or((cmd, ""));
    match head {
        "quit" | "exit" | "q" => Ok(true),
        "help" | "?" => {
            ui.overlay = Some(Overlay::Help);
            Ok(false)
        }
        "reset" => {
            cmd_tx
                .send(UserCmd::Reset)
                .map_err(|e| anyhow::anyhow!("reset: {}", e))?;
            ui.chat_log.push(DisplayMsg {
                role: Role::System,
                content: "[history cleared; KV cache reset]".to_string(),
            });
            Ok(false)
        }
        "clear" => {
            ui.chat_log.clear();
            Ok(false)
        }
        "sys" => handle_slash_sys(ui, rest, cmd_tx),
        other => {
            handle_unknown_slash(ui, other);
            Ok(false)
        }
    }
}

fn handle_slash_sys(ui: &mut UiState, rest: &str, cmd_tx: &Sender<UserCmd>) -> Result<bool> {
    if rest.is_empty() {
        ui.flash("usage: /sys <text>  or  /sys off");
        return Ok(false);
    }
    let payload = (rest != "off").then(|| rest.to_string());
    cmd_tx
        .send(UserCmd::SetSys(payload.clone()))
        .map_err(|e| anyhow::anyhow!("sys: {}", e))?;
    ui.dashboard.system_prompt = payload.clone();
    let content = match &payload {
        Some(_) => format!(
            "[system prompt set ({} chars) — takes effect after next /reset]",
            rest.len()
        ),
        None => "[system prompt cleared]".to_string(),
    };
    ui.chat_log.push(DisplayMsg {
        role: Role::System,
        content,
    });
    Ok(false)
}

fn handle_unknown_slash(ui: &mut UiState, other: &str) {
    let suggestion = nearest_slash_command(other);
    match suggestion {
        Some(sugg) => ui.flash(format!("unknown /{} — did you mean /{}?", other, sugg)),
        None => ui.flash(format!("unknown /{} (try /help)", other)),
    }
}

fn nearest_slash_command(input: &str) -> Option<&'static str> {
    KNOWN_SLASH_COMMANDS
        .iter()
        .map(|known| (*known, levenshtein(input, known)))
        .filter(|(_, d)| *d <= 2)
        .min_by_key(|(_, d)| *d)
        .map(|(name, _)| name)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0_usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

// ── copy / yank ─────────────────────────────────────────────────────

fn yank_last_bot_response(ui: &mut UiState) {
    let Some(text) = &ui.last_bot_text else {
        ui.flash("no bot response to copy yet");
        return;
    };
    // OSC52 escape sequence — most modern terminals (iTerm2, Kitty,
    // Alacritty, wezterm, mintty, recent xterm) support it.
    use base64_simple::encode;
    let b64 = encode(text.as_bytes());
    print!("\x1b]52;c;{}\x07", b64);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    ui.flash(format!("copied {} chars to clipboard (OSC52)", text.len()));
}

// Tiny inline base64 encoder — avoids adding a `base64` crate dep
// for this single use.
mod base64_simple {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    pub fn encode(input: &[u8]) -> String {
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        for chunk in input.chunks(3) {
            let b0 = chunk[0];
            let b1 = chunk.get(1).copied().unwrap_or(0);
            let b2 = chunk.get(2).copied().unwrap_or(0);
            out.push(ALPHABET[(b0 >> 2) as usize] as char);
            out.push(ALPHABET[((b0 & 0b11) << 4 | b1 >> 4) as usize] as char);
            if chunk.len() > 1 {
                out.push(ALPHABET[((b1 & 0b1111) << 2 | b2 >> 6) as usize] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(ALPHABET[(b2 & 0b111111) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }
}

// ── history file ────────────────────────────────────────────────────

fn persisted_history_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(".config");
    p.push("willamette");
    let _ = std::fs::create_dir_all(&p);
    p.push("history");
    Some(p)
}

fn load_history_file(path: &std::path::Path) -> Vec<String> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    // File is newest-last (append-on-submit); reverse so newest is first.
    let mut lines: Vec<String> = content
        .lines()
        .map(|l| l.to_string())
        .filter(|l| !l.is_empty())
        .collect();
    lines.reverse();
    lines.truncate(super::input_editor::HISTORY_CAP);
    lines
}

fn persist_history_append(ui: &UiState, line: &str) {
    if line.trim().is_empty() {
        return;
    }
    let Some(path) = &ui.history_path else {
        return;
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        use std::io::Write;
        let _ = writeln!(f, "{}", line);
    }
}

// ── rendering ───────────────────────────────────────────────────────

fn render(f: &mut ratatui::Frame, ui: &mut UiState) {
    let area = f.area();

    // Outer vertical: main area + status (1) + input (3).
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(1), // keybinding hint
        ])
        .split(area);

    // Main area horizontal split: chat (left) / dashboard (right).
    // Dashboard ~32 cols; if terminal is narrow, omit dashboard.
    let main_chunks = if outer[0].width >= 72 {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(40), Constraint::Length(34)])
            .split(outer[0])
    } else {
        // Single-pane fallback.
        std::rc::Rc::from(vec![outer[0]])
    };

    render_chat_pane(f, ui, main_chunks[0]);
    let dash_area_opt = (main_chunks.len() > 1).then(|| main_chunks[1]);

    // Chat pane is the only renderer that needs &mut; everything below
    // is read-only.
    let ui: &UiState = ui;
    if let Some(da) = dash_area_opt {
        render_dashboard_pane(f, ui, da);
    }
    render_status_bar(f, ui, outer[1]);
    render_input_box(f, ui, outer[2]);
    render_keyhint(f, ui, outer[3]);

    if let Some(Overlay::Help) = ui.overlay {
        render_help_overlay(f, area);
    } else if let Some(s) = ui.input.search() {
        render_search_overlay(f, ui, area, s);
    }
}

fn render_chat_pane(f: &mut ratatui::Frame, ui: &mut UiState, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for msg in &ui.chat_log {
        append_message_lines(&mut lines, msg.role, &msg.content, false);
        lines.push(Line::from(""));
    }
    if !ui.streaming.is_empty() || ui.generating {
        append_message_lines(&mut lines, Role::Bot, &ui.streaming, true);
    }

    let pct = if ui.max_seq_len > 0 {
        100.0 * ui.token_position as f64 / ui.max_seq_len as f64
    } else {
        0.0
    };
    let title = format!(
        " Project Willamette  ·  ctx {}/{}  ({:.0}%) ",
        ui.token_position, ui.max_seq_len, pct
    );
    let para = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .title_style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .wrap(Wrap { trim: false });

    // ratatui's Paragraph.scroll((n, 0)) means "skip the first n
    // wrapped lines". So the bottom-pinning offset is
    // (total_wrapped_lines - viewport_height), saturated at zero.
    let inner_w = area.width.saturating_sub(2); // borders eat 2 cells
    let inner_h = area.height.saturating_sub(2);
    let total_lines = para.line_count(inner_w) as u16;
    let max_scroll = total_lines.saturating_sub(inner_h);

    // Reconcile follow_bottom ↔ scroll_offset:
    //   * follow_bottom = true  → renderer always shows the last line
    //   * follow_bottom = false → user is reading older content
    // If a key/wheel handler bumped scroll_offset past max_scroll
    // (e.g. PageDown spam), promote them back to follow mode.
    if ui.follow_bottom || ui.scroll_offset >= max_scroll {
        ui.follow_bottom = true;
        ui.scroll_offset = max_scroll;
    } else {
        ui.scroll_offset = ui.scroll_offset.min(max_scroll);
    }

    f.render_widget(para.scroll((ui.scroll_offset, 0)), area);
}

fn render_dashboard_pane(f: &mut ratatui::Frame, ui: &UiState, area: Rect) {
    let lines = ui
        .dashboard
        .render_lines(&ui.sys, area.width.saturating_sub(2));
    let para = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Dashboard ")
                .title_style(
                    Style::default()
                        .add_modifier(Modifier::BOLD)
                        .fg(Color::Cyan),
                ),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn render_status_bar(f: &mut ratatui::Frame, ui: &UiState, area: Rect) {
    let status_text = if let Some((msg, _)) = &ui.transient {
        format!("  {}", msg)
    } else {
        format!("  {}", ui.status)
    };
    let status_color = if ui.generating {
        Color::Yellow
    } else {
        Color::Gray
    };
    let p = Paragraph::new(Line::from(Span::styled(
        status_text,
        Style::default().fg(status_color),
    )));
    f.render_widget(p, area);
}

fn render_input_box(f: &mut ratatui::Frame, ui: &UiState, area: Rect) {
    let prompt_prefix = if ui.generating {
        "  (busy) "
    } else if ui.input.search().is_some() {
        "  (search) "
    } else {
        "  > "
    };
    let display: String = if ui.input.search().is_some() {
        // Show search needle, not the buffer.
        let s = ui.input.search().unwrap();
        let matched = s
            .match_idx
            .map(|i| ui.input.history().get(i).cloned().unwrap_or_default())
            .unwrap_or_default();
        format!("`{}`  →  {}", s.needle, matched)
    } else {
        ui.input.buffer().to_string()
    };
    let input_line = format!("{}{}", prompt_prefix, display);
    let para = Paragraph::new(input_line).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" input ")
            .title_style(Style::default().add_modifier(Modifier::BOLD)),
    );
    f.render_widget(para, area);

    // Real terminal cursor at the right position. Both the prompt
    // prefix and the buffer-up-to-cursor are measured in *display
    // columns* (terminal cells), so Korean / CJK / emoji glyphs that
    // occupy 2 cells each don't make the cursor land mid-glyph and
    // cause subsequent input to overlap the previous character.
    if !ui.generating && ui.input.search().is_none() {
        use unicode_width::UnicodeWidthStr;
        let prefix_w = UnicodeWidthStr::width(prompt_prefix) as u16;
        let cursor_x = area.x + 1 + prefix_w + ui.input.cursor_display_col() as u16;
        let cursor_y = area.y + 1;
        f.set_cursor_position((cursor_x, cursor_y));
    }
}

fn render_keyhint(f: &mut ratatui::Frame, _ui: &UiState, area: Rect) {
    let hint =
        "  F1 help · ↑↓ history · Ctrl-R search · Ctrl-L clear · Ctrl-Y copy · Esc cancel/quit";
    let p = Paragraph::new(Line::from(Span::styled(
        hint,
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(p, area);
}

fn render_help_overlay(f: &mut ratatui::Frame, area: Rect) {
    let width = 60.min(area.width.saturating_sub(4));
    let height = 22.min(area.height.saturating_sub(4));
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    let popup = Rect {
        x,
        y,
        width,
        height,
    };
    f.render_widget(Clear, popup);

    let help_lines: Vec<Line<'static>> = vec![
        Line::from(Span::styled(
            "  Project Willamette — Key bindings",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  Input editing"),
        Line::from("    ← →     move cursor"),
        Line::from("    Home/End  start/end of line  (Ctrl-A / Ctrl-E)"),
        Line::from("    Ctrl-W   delete word"),
        Line::from("    Ctrl-U   delete to start"),
        Line::from("    Ctrl-K   delete to end"),
        Line::from("    Tab      complete slash command"),
        Line::from(""),
        Line::from("  History"),
        Line::from("    ↑ ↓     scroll input history"),
        Line::from("    Ctrl-R   reverse search"),
        Line::from(""),
        Line::from("  Chat log"),
        Line::from("    PgUp/PgDn   scroll  (also mouse wheel)"),
        Line::from("    Ctrl-Home/End   jump top/bottom"),
        Line::from("    Ctrl-L   clear visible log"),
        Line::from("    Ctrl-Y   copy last bot response (OSC52)"),
        Line::from(""),
        Line::from("  Generation"),
        Line::from("    Esc      cancel current turn  (then quit)"),
        Line::from("    Ctrl-C   force quit"),
        Line::from(""),
        Line::from(Span::styled(
            "  any key to dismiss",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    let p = Paragraph::new(help_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Help (F1) ")
            .title_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::Cyan),
            ),
    );
    f.render_widget(p, popup);
}

fn render_search_overlay(
    f: &mut ratatui::Frame,
    _ui: &UiState,
    area: Rect,
    s: &super::input_editor::SearchState,
) {
    let width = 60.min(area.width.saturating_sub(4));
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + area.height / 3,
        width,
        height: 5,
    };
    f.render_widget(Clear, popup);
    let body = vec![
        Line::from(format!("  needle: {}", s.needle)),
        Line::from(""),
        Line::from(Span::styled(
            "  Enter accept · Esc cancel · Ctrl-R next match",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    let p = Paragraph::new(body).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" reverse-i-search ")
            .title_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::Yellow),
            ),
    );
    f.render_widget(p, popup);
}

/// Append one chat message to `lines`, with BOT messages going through
/// the markdown renderer and USR/SYS messages rendered as plain text.
/// First line gets the role badge prefixed; continuation lines get
/// aligned under the body. When `with_cursor`, append a green ▌ to
/// the last line (live streaming indicator).
fn append_message_lines(
    lines: &mut Vec<Line<'static>>,
    role: Role,
    content: &str,
    with_cursor: bool,
) {
    let role_span = Span::styled(
        format!(" {} ", role.label()),
        Style::default()
            .fg(Color::Black)
            .bg(role.color())
            .add_modifier(Modifier::BOLD),
    );

    let mut body_lines: Vec<Line<'static>> = if matches!(role, Role::Bot) {
        super::markdown::render_markdown_lines(content)
    } else {
        vec![Line::from(vec![Span::raw(content.to_string())])]
    };
    if body_lines.is_empty() {
        body_lines.push(Line::from(""));
    }

    let first = body_lines.remove(0);
    let mut first_spans: Vec<Span<'static>> = vec![role_span, Span::raw(" ".to_string())];
    first_spans.extend(first.spans);
    lines.push(Line::from(first_spans));

    for body_line in body_lines {
        let mut spans: Vec<Span<'static>> = vec![Span::raw("      ".to_string())];
        spans.extend(body_line.spans);
        lines.push(Line::from(spans));
    }

    if with_cursor {
        if let Some(last) = lines.last_mut() {
            last.spans
                .push(Span::styled(" ▌", Style::default().fg(Color::Green)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_distance() {
        assert_eq!(levenshtein("reset", "reset"), 0);
        assert_eq!(levenshtein("rese", "reset"), 1);
        assert_eq!(levenshtein("resert", "reset"), 1);
        // "retset" vs "reset": delete the extra 't' = 1 edit.
        assert_eq!(levenshtein("retset", "reset"), 1);
        // 2-edit case: swap-adjacent ('re' vs 'er') = 2 substitutions.
        assert_eq!(levenshtein("ersit", "reset"), 3);
        assert_eq!(levenshtein("hi", "reset"), 5);
    }

    #[test]
    fn base64_encodes_correctly() {
        assert_eq!(base64_simple::encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_simple::encode(b""), "");
        assert_eq!(base64_simple::encode(b"f"), "Zg==");
        assert_eq!(base64_simple::encode(b"fo"), "Zm8=");
        assert_eq!(base64_simple::encode(b"foo"), "Zm9v");
        assert_eq!(base64_simple::encode(b"foob"), "Zm9vYg==");
    }
}
