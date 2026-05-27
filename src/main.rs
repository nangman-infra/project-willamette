//! Project Willamette CLI.
//!
//! Stage 1 implements only the `inspect` subcommand. `tokenize`, `run`, and any
//! generation/inference paths intentionally do **not** exist yet — calling them
//! would mean producing fake output.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use project_willamette::chat::ChatEngine;
use project_willamette::gguf::reader::{GgufFile, GgufValue, GGUF_MAGIC};
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::bitlinear::bitlinear_i2s_matvec_f32;
use project_willamette::model::cached_forward::forward_with_cache;
use project_willamette::model::forward::forward_single_token_position_zero;
use project_willamette::model::generate::generate_with_cache_and_sampler;
use project_willamette::model::kv_cache::KVCache;
use project_willamette::model::lm_head::{argmax, compute_logits_from_graph, top_k};
use project_willamette::model::multi_forward::multi_token_forward;
use project_willamette::model::primitives::embedding_gather_f16;
use project_willamette::model::sampler::{Sampler, SamplingParams};
use project_willamette::model::ModelGraph;
use project_willamette::tokenizer::Tokenizer;

#[derive(Parser)]
#[command(
    name = "willamette",
    about = "Project Willamette — Rust-native BitNet 1.58-bit GGUF inference runtime",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Shared argument group for the `chat` and `tui` subcommands. Both
/// load one model, keep one KV cache, and accept the same sampling
/// knobs — keeping the flag surface DRY also makes them
/// configuration-equivalent: any answer one mode gives, the other can
/// reproduce with the same CLI.
#[derive(Args, Debug, Clone)]
struct ChatArgs {
    /// Path to the .gguf model file.
    #[arg(long)]
    model: PathBuf,
    /// Token budget for the KV cache (prompt + all turns).
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,
    /// Cap on new tokens per assistant turn.
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
    /// Optional system prompt prepended to the very first turn.
    #[arg(long)]
    system: Option<String>,
    /// Sampling temperature. 0 = greedy.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    /// Keep only the K highest-probability tokens before sampling.
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    /// Nucleus: keep tokens up to cumulative probability `p`.
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    /// Repetition penalty (HF convention; > 1.0 enables).
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    /// PRNG seed.
    #[arg(long, default_value_t = 0xabad_1dea_u64)]
    seed: u64,
}

fn build_sampling_params(args: &ChatArgs) -> SamplingParams {
    SamplingParams {
        temperature: args.temperature,
        top_k: if args.top_k == 0 {
            None
        } else {
            Some(args.top_k)
        },
        top_p: if args.top_p >= 1.0 || args.top_p <= 0.0 {
            None
        } else {
            Some(args.top_p)
        },
        repetition_penalty: if args.repetition_penalty <= 1.0 {
            None
        } else {
            Some(args.repetition_penalty)
        },
        seed: args.seed,
    }
}

#[derive(Subcommand)]
enum Command {
    /// Stage 1: parse a real GGUF file and print header, metadata, and tensor directory.
    Inspect {
        /// Path to the .gguf model file.
        #[arg(long)]
        model: PathBuf,
    },
    /// Stage 2: encode + decode text using the model's GGUF tokenizer metadata.
    ///
    /// Prints token IDs, decoded text, and roundtrip status. Refuses to run if
    /// the metadata does not describe a supported tokenizer — no fake fallback.
    Tokenize {
        /// Path to the .gguf model file.
        #[arg(long)]
        model: PathBuf,
        /// Text to encode.
        #[arg(long)]
        text: String,
        /// Override metadata's add_bos_token: do NOT prepend BOS.
        #[arg(long, default_value_t = false)]
        no_bos: bool,
        /// Append EOS token at end (regardless of metadata default).
        #[arg(long, default_value_t = false)]
        add_eos: bool,
    },
    /// Stage 5-C/D: real causal forward with KV cache + greedy or
    /// sampled decoding.
    ///
    /// Default is deterministic greedy. Sampling activates when any of
    /// `--temperature`, `--top-k`, `--top-p`, or
    /// `--repetition-penalty` is supplied. With sampling, `--seed`
    /// makes the run reproducible.
    Run {
        /// Path to the .gguf model file.
        #[arg(long)]
        model: PathBuf,
        /// Prompt text.
        #[arg(long)]
        prompt: String,
        /// Number of new tokens to generate (must be >= 1).
        #[arg(long, default_value_t = 1)]
        max_new_tokens: usize,
        /// Suppress BOS even if `tokenizer.ggml.add_bos_token` is set.
        #[arg(long, default_value_t = false)]
        no_bos: bool,
        /// Sampling temperature. 0 = greedy / argmax.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Keep only the K highest-probability tokens before sampling.
        #[arg(long)]
        top_k: Option<usize>,
        /// Nucleus: keep tokens up to cumulative probability `p`.
        #[arg(long)]
        top_p: Option<f32>,
        /// Repetition penalty (HF convention: logits of recently-seen
        /// tokens are divided by this when > 1.0).
        #[arg(long)]
        repetition_penalty: Option<f32>,
        /// PRNG seed (only used when sampling).
        #[arg(long, default_value_t = 0xabad_1dea_u64)]
        seed: u64,
        /// Additional stop token ids beyond `eos_token_id` (may be
        /// repeated). Match → generation stops.
        #[arg(long)]
        stop_id: Vec<u32>,
    },
    /// Stage 5-E: dump top-k logits for a prompt — used to compare
    /// against bitnet.cpp reference. Includes the full prompt token id
    /// list and the argmax/top-k from the forward over the prompt (the
    /// distribution that predicts the FIRST new token).
    Logits {
        /// Path to the .gguf model file.
        #[arg(long)]
        model: PathBuf,
        /// Prompt text.
        #[arg(long)]
        prompt: String,
        /// How many top logits to print.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Suppress BOS even if `tokenizer.ggml.add_bos_token` is set.
        #[arg(long, default_value_t = false)]
        no_bos: bool,
    },
    /// Stage 9-E: ratatui-based full-screen chat TUI.
    ///
    /// Same engine as `chat` but with a proper history view, input
    /// box, status bar, and slash commands. Quit with `/quit` or Ctrl-C.
    Tui {
        #[command(flatten)]
        chat: ChatArgs,
    },
    /// Stage 9: multi-turn interactive chat (stdin/stdout).
    ///
    /// Loads the model once, keeps a KV cache across turns, streams
    /// each assistant response in UTF-8-safe chunks. Type `/quit` or
    /// press Ctrl-D to exit. Other slash commands arrive in Stage 9-D.
    Chat {
        #[command(flatten)]
        chat: ChatArgs,
    },
    /// Stage 6-A: scalar-reference baseline benchmarks.
    ///
    /// Times one I2_S BitLinear matvec, one single-token (no cache)
    /// forward, and one decode step with KV cache. Reports
    /// milliseconds and rough tokens/sec. SIMD comparison happens in
    /// Stage 6-B / 6-C — this is the "before".
    Bench {
        /// Path to the .gguf model file.
        #[arg(long)]
        model: PathBuf,
        /// Number of decode-step samples averaged for the cache bench
        /// (the prefill is also timed separately).
        #[arg(long, default_value_t = 3)]
        decode_steps: usize,
    },
    /// Build a synthetic BitNet b1.58 GGUF file for throughput
    /// benchmarking on humble hardware. No tokenizer; random ternary
    /// weights (Preset::Small / Medium) or all-zero
    /// (Preset::Tiny). Output is NOT useful for `run` / `chat` /
    /// `tui` — only for `inspect` and `bench`. See `src/synth.rs`.
    SynthGguf {
        /// Where to write the GGUF file.
        #[arg(long)]
        output: PathBuf,
        /// Size preset. `medium` ≈ 110 M params (TinyLlama scale,
        /// same class as EXO's Pentium-II demo model).
        #[arg(long, default_value_t = SynthPreset::Medium)]
        preset: SynthPreset,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SynthPreset {
    Tiny,
    Small,
    Medium,
}

impl std::fmt::Display for SynthPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SynthPreset::Tiny => "tiny",
            SynthPreset::Small => "small",
            SynthPreset::Medium => "medium",
        })
    }
}

impl From<SynthPreset> for project_willamette::synth::Preset {
    fn from(p: SynthPreset) -> Self {
        match p {
            SynthPreset::Tiny => project_willamette::synth::Preset::Tiny,
            SynthPreset::Small => project_willamette::synth::Preset::Small,
            SynthPreset::Medium => project_willamette::synth::Preset::Medium,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { model } => cmd_inspect(&model),
        Command::Tokenize {
            model,
            text,
            no_bos,
            add_eos,
        } => cmd_tokenize(&model, &text, no_bos, add_eos),
        Command::Run {
            model,
            prompt,
            max_new_tokens,
            no_bos,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            stop_id,
        } => cmd_run(
            &model,
            &prompt,
            max_new_tokens,
            no_bos,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            &stop_id,
        ),
        Command::Bench {
            model,
            decode_steps,
        } => cmd_bench(&model, decode_steps),
        Command::Logits {
            model,
            prompt,
            top_k,
            no_bos,
        } => cmd_logits(&model, &prompt, top_k, no_bos),
        Command::Chat { chat } => cmd_chat(&chat),
        Command::Tui { chat } => cmd_tui(&chat),
        Command::SynthGguf { output, preset } => cmd_synth_gguf(&output, preset.into()),
    }
}

fn cmd_inspect(path: &Path) -> Result<()> {
    let mmap =
        ModelMmap::open(path).with_context(|| format!("opening model file: {}", path.display()))?;
    let bytes = mmap.as_bytes();
    let file = GgufFile::parse(bytes).map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;

    let gib = bytes.len() as f64 / (1024.0_f64.powi(3));

    println!("==================================================");
    println!("GGUF Inspection: {}", path.display());
    println!("==================================================");
    println!("File size:    {} bytes ({:.3} GiB)", bytes.len(), gib);
    println!("Magic:        0x{:08X} (\"GGUF\")", GGUF_MAGIC);
    println!("Version:      {}", file.version);
    println!("Tensor count: {}", file.tensor_count);
    println!("Metadata kv:  {}", file.metadata.len());
    println!("Alignment:    {} bytes", file.alignment);
    println!();

    let mut general_kv: Vec<(&String, &GgufValue)> = Vec::new();
    let mut tokenizer_kv: Vec<(&String, &GgufValue)> = Vec::new();
    let mut other_kv: Vec<(&String, &GgufValue)> = Vec::new();
    for (k, v) in &file.metadata {
        if k.starts_with("general.") {
            general_kv.push((k, v));
        } else if k.starts_with("tokenizer.") {
            tokenizer_kv.push((k, v));
        } else {
            other_kv.push((k, v));
        }
    }
    general_kv.sort_by(|a, b| a.0.cmp(b.0));
    tokenizer_kv.sort_by(|a, b| a.0.cmp(b.0));
    other_kv.sort_by(|a, b| a.0.cmp(b.0));

    print_section("general.* metadata", &general_kv);
    print_section("tokenizer.* metadata", &tokenizer_kv);
    print_section("other metadata", &other_kv);

    println!("--- tensor directory ({} tensors) ---", file.tensors.len());
    println!(
        "{:>4}  {:>8}  {:<18}  {:<50}  {:<26}  {:>14}  {:>16}",
        "idx", "raw_u32", "resolved_dtype", "name", "shape", "offset", "byte_len"
    );
    println!("{}", "-".repeat(146));
    for (i, t) in file.tensors.iter().enumerate() {
        let shape_str: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        let shape_field = format!("[{}]", shape_str.join(", "));
        println!(
            "{:>4}  {:>8}  {:<18}  {:<50}  {:<26}  0x{:>12X}  {:>16}",
            i,
            t.ggml_type.to_raw(),
            t.ggml_type.name(),
            truncate(&t.name, 50),
            truncate(&shape_field, 26),
            t.offset,
            t.byte_len,
        );
    }

    Ok(())
}

fn print_section(title: &str, kv: &[(&String, &GgufValue)]) {
    println!("--- {} ({}) ---", title, kv.len());
    if kv.is_empty() {
        println!("  (none)");
    } else {
        for (k, v) in kv {
            println!("  {:<40} = {}", k, format_value(v));
        }
    }
    println!();
}

fn cmd_tokenize(path: &Path, text: &str, no_bos: bool, force_eos: bool) -> Result<()> {
    let mmap =
        ModelMmap::open(path).with_context(|| format!("opening model file: {}", path.display()))?;
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;

    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))?;

    let mut opts = tokenizer.default_encode_options();
    if no_bos {
        opts.add_bos = false;
    }
    if force_eos {
        opts.add_eos = true;
    }

    let ids = tokenizer
        .encode(text, opts)
        .map_err(|e| anyhow::anyhow!("encode failed: {}", e))?;
    let decoded = tokenizer
        .decode(&ids)
        .map_err(|e| anyhow::anyhow!("decode failed: {}", e))?;

    // Build the expected decoded string given the options used.
    let expected = {
        let mut s = String::new();
        if opts.add_bos {
            if let Some(bos) = tokenizer.bos_id {
                if let Some(t) = tokenizer.token_str(bos) {
                    s.push_str(t);
                }
            }
        }
        s.push_str(text);
        if opts.add_eos {
            if let Some(eos) = tokenizer.eos_id {
                if let Some(t) = tokenizer.token_str(eos) {
                    s.push_str(t);
                }
            }
        }
        s
    };
    let roundtrip_ok = decoded == expected;

    let token_strings: Vec<String> = ids
        .iter()
        .map(|&id| tokenizer.token_str(id).unwrap_or("<?>").to_string())
        .collect();

    println!("==================================================");
    println!("Tokenize: {}", path.display());
    println!("==================================================");
    println!("Tokenizer model: {}", tokenizer.model_type);
    println!("Vocab size:      {}", tokenizer.vocab_size());
    println!(
        "Special IDs:     BOS={:?}, EOS={:?}, PAD={:?}",
        tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id
    );
    println!(
        "Options:         add_bos={}, add_eos={}  (metadata default add_bos={})",
        opts.add_bos, opts.add_eos, tokenizer.default_add_bos
    );
    println!();
    println!("Input:           {:?}", text);
    println!("Token count:     {}", ids.len());
    println!("Token IDs:       {:?}", ids);
    println!("Token strings:   {:?}", token_strings);
    println!("Decoded:         {:?}", decoded);
    println!(
        "Roundtrip:       {} (expected {:?})",
        if roundtrip_ok { "OK" } else { "FAIL" },
        expected
    );

    if !roundtrip_ok {
        anyhow::bail!("roundtrip mismatch: encode/decode did not reproduce the input");
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(
    path: &Path,
    prompt: &str,
    max_new_tokens: usize,
    no_bos: bool,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: u64,
    stop_ids: &[u32],
) -> Result<()> {
    if max_new_tokens == 0 {
        anyhow::bail!("--max-new-tokens must be >= 1");
    }

    let mmap =
        ModelMmap::open(path).with_context(|| format!("opening model file: {}", path.display()))?;
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))?;
    let graph = ModelGraph::from_gguf(&gguf)
        .map_err(|e| anyhow::anyhow!("model graph load failed: {}", e))?;

    let mut opts = tokenizer.default_encode_options();
    if no_bos {
        opts.add_bos = false;
    }

    let prompt_ids = tokenizer
        .encode(prompt, opts)
        .map_err(|e| anyhow::anyhow!("encode failed: {}", e))?;
    if prompt_ids.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens — cannot run forward");
    }

    let sp = SamplingParams {
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        seed,
    };
    let mode = if sp.is_greedy() {
        "greedy (argmax)"
    } else {
        "sampled"
    };
    let mut sampler = Sampler::new(sp);

    println!("==================================================");
    println!("Run: {}", path.display());
    println!("==================================================");
    println!("Architecture: {}", graph.config.architecture);
    println!("Block count:  {}", graph.config.block_count);
    println!("Vocab size:   {}", graph.config.vocab_size);
    println!();
    println!("Prompt:           {:?}", prompt);
    println!("Prompt tokens:    {} ({:?})", prompt_ids.len(), prompt_ids);
    println!("Decode policy:    Stage 5-C — single-token forward + per-layer KV cache (causal)");
    println!("Decode mode:      {}", mode);
    println!(
        "Max new tokens:   {}  EOS id: {:?}  PAD id: {:?}  extra stop: {:?}",
        max_new_tokens, tokenizer.eos_id, tokenizer.pad_id, stop_ids
    );
    println!();
    use std::io::Write;
    print!("Generating: ");
    std::io::stdout().flush().ok();

    let max_seq_len = (prompt_ids.len() + max_new_tokens + 16).max(64);
    // Streaming UTF-8 boundary-aware printer: a multi-byte Korean /
    // CJK / emoji codepoint is often split across two or three BPE
    // tokens. We accumulate raw bytes and only print up to the last
    // complete UTF-8 boundary on each tick — the partial suffix
    // waits for the next token.
    let mut pending: Vec<u8> = Vec::new();
    let mut printed_up_to: usize = 0;
    let generated = generate_with_cache_and_sampler(
        &graph,
        &prompt_ids,
        max_new_tokens,
        tokenizer.eos_id,
        stop_ids,
        max_seq_len,
        &mut sampler,
        |_step, _next_pos, tok_id| {
            if let Ok(more) = tokenizer.decode_to_bytes(&[tok_id]) {
                pending.extend_from_slice(&more);
                let valid_end = match std::str::from_utf8(&pending) {
                    Ok(_) => pending.len(),
                    Err(e) => e.valid_up_to(),
                };
                if valid_end > printed_up_to {
                    // SAFETY: bytes[..valid_end] is valid UTF-8 by
                    // definition of `Utf8Error::valid_up_to`.
                    let chunk = unsafe {
                        std::str::from_utf8_unchecked(&pending[printed_up_to..valid_end])
                    };
                    print!("{}", chunk);
                    std::io::stdout().flush().ok();
                    printed_up_to = valid_end;
                }
            }
        },
    )
    .map_err(|e| anyhow::anyhow!("generation failed: {}", e))?;
    // Flush any leftover incomplete suffix as U+FFFD so the user sees
    // it was there (rather than silently dropping it).
    if printed_up_to < pending.len() {
        print!("\u{FFFD}");
        std::io::stdout().flush().ok();
    }
    println!();
    println!();
    println!("Generated {} token(s): {:?}", generated.len(), generated);
    // Use lossy decode for the final summary so a truncated trailing
    // multi-byte character doesn't crash the run.
    let generated_text = tokenizer
        .decode_lossy(&generated)
        .map_err(|e| anyhow::anyhow!("decode failed: {}", e))?;
    println!("Generated text:   {:?}", generated_text);
    let full_text = format!("{}{}", prompt, generated_text);
    println!("Full text:        {:?}", full_text);
    Ok(())
}

fn cmd_chat(args: &ChatArgs) -> Result<()> {
    use std::io::{BufRead, Write};

    let load_start = std::time::Instant::now();
    let mmap = ModelMmap::open(&args.model)
        .with_context(|| format!("opening model file: {}", args.model.display()))?;
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))?;
    let graph = ModelGraph::from_gguf(&gguf)
        .map_err(|e| anyhow::anyhow!("model graph load failed: {}", e))?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let mut engine = ChatEngine::new(
        &graph,
        tokenizer,
        build_sampling_params(args),
        args.max_seq_len,
    );
    if let Some(sys) = args.system.as_deref() {
        engine.set_system_prompt(Some(sys.to_string()));
    }

    print_chat_banner(args, &graph, load_ms);

    let stdin = std::io::stdin();
    let mut input_line = String::new();
    let mut stdout = std::io::stdout();
    loop {
        print!("You: ");
        stdout.flush().ok();
        input_line.clear();
        let n = stdin
            .lock()
            .read_line(&mut input_line)
            .map_err(|e| anyhow::anyhow!("stdin read failed: {}", e))?;
        if n == 0 {
            println!();
            break;
        }
        let user_text = input_line.trim_end_matches(['\n', '\r']).trim();
        if user_text.is_empty() {
            continue;
        }
        if let Some(stripped) = user_text.strip_prefix('/') {
            match handle_slash_command(stripped, &mut engine)? {
                SlashOutcome::Quit => break,
                SlashOutcome::Continue => {
                    println!();
                    continue;
                }
            }
        }

        run_one_chat_turn(&mut engine, user_text, args.max_new_tokens)?;
    }
    println!("Goodbye.");
    Ok(())
}

fn print_chat_banner(args: &ChatArgs, graph: &ModelGraph<'_>, load_ms: f64) {
    println!("====================================================");
    println!("Project Willamette — chat (Stage 9-A)");
    println!("====================================================");
    println!("Model:       {}", args.model.display());
    println!(
        "Architecture: {}  layers: {}  vocab: {}",
        graph.config.architecture, graph.config.block_count, graph.config.vocab_size
    );
    println!("Loaded in:   {:.1} ms", load_ms);
    println!(
        "Sampling:    temp={}  top_k={}  top_p={}  rep_pen={}  seed=0x{:x}",
        args.temperature, args.top_k, args.top_p, args.repetition_penalty, args.seed
    );
    println!(
        "Budget:      max_seq_len={}  max_new_tokens_per_turn={}",
        args.max_seq_len, args.max_new_tokens
    );
    println!("Slash commands: /help /reset /history /save <file> /sys [text|off] /quit (Ctrl-D)");
    println!();
}

fn run_one_chat_turn(
    engine: &mut ChatEngine<'_, '_>,
    user_text: &str,
    max_new_tokens: usize,
) -> Result<()> {
    use std::io::Write;
    print!("Bot: ");
    std::io::stdout().flush().ok();
    let turn_start = std::time::Instant::now();
    let result = engine.send_user_message(user_text, max_new_tokens, |chunk| {
        print!("{}", chunk);
        std::io::stdout().flush().ok();
    });
    let turn_ms = turn_start.elapsed().as_secs_f64() * 1000.0;
    println!();
    report_turn_outcome(engine, result, turn_ms);
    println!();
    Ok(())
}

fn report_turn_outcome(
    engine: &ChatEngine<'_, '_>,
    result: Result<String, project_willamette::error::WillametteError>,
    turn_ms: f64,
) {
    match result {
        Ok(response) => {
            let toks = response.chars().count() as f64;
            let tps = toks / (turn_ms / 1000.0).max(1e-6);
            println!(
                "      [turn took {:.1} s  ~{:.1} chars/s  ctx {}/{} tokens]",
                turn_ms / 1000.0,
                tps,
                engine.token_position(),
                engine.max_seq_len()
            );
        }
        Err(e) => {
            eprintln!("[chat error] {}", e);
            eprintln!("[hint] type /quit to exit");
        }
    }
}

fn cmd_tui(args: &ChatArgs) -> Result<()> {
    let mmap = ModelMmap::open(&args.model)
        .with_context(|| format!("opening model file: {}", args.model.display()))?;
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))?;
    let graph = ModelGraph::from_gguf(&gguf)
        .map_err(|e| anyhow::anyhow!("model graph load failed: {}", e))?;

    let mut engine = ChatEngine::new(
        &graph,
        tokenizer,
        build_sampling_params(args),
        args.max_seq_len,
    );
    if let Some(sys) = args.system.as_deref() {
        engine.set_system_prompt(Some(sys.to_string()));
    }
    project_willamette::chat::run_tui(engine, args.max_new_tokens)
}

enum SlashOutcome {
    Continue,
    Quit,
}

fn handle_slash_command(cmd_line: &str, engine: &mut ChatEngine<'_, '_>) -> Result<SlashOutcome> {
    let (cmd, rest) = match cmd_line.split_once(char::is_whitespace) {
        Some((c, r)) => (c, r.trim()),
        None => (cmd_line, ""),
    };
    match cmd {
        "quit" | "exit" | "q" => Ok(SlashOutcome::Quit),
        "help" | "?" => {
            print_slash_help();
            Ok(SlashOutcome::Continue)
        }
        "reset" => {
            engine.reset();
            println!("[history cleared; KV cache reset]");
            Ok(SlashOutcome::Continue)
        }
        "history" => {
            print_slash_history(engine);
            Ok(SlashOutcome::Continue)
        }
        "save" => handle_slash_save(rest, engine),
        "sys" => handle_slash_sys(rest, engine),
        "stats" => {
            print_slash_stats(engine);
            Ok(SlashOutcome::Continue)
        }
        other => {
            println!("[unknown command: /{} — try /help]", other);
            Ok(SlashOutcome::Continue)
        }
    }
}

fn print_slash_help() {
    println!("Commands:");
    println!("  /help              — this message");
    println!("  /quit              — exit (alias /exit, /q, or Ctrl-D)");
    println!("  /reset             — clear conversation history + KV cache");
    println!("  /history           — print the current turn-by-turn history");
    println!("  /save <path>       — write history as JSON lines to <path>");
    println!("  /sys <text>        — set or replace the system prompt");
    println!("  /sys off           — clear the system prompt");
    println!("  /stats             — show token-position + budget usage");
}

fn print_slash_history(engine: &ChatEngine<'_, '_>) {
    if engine.history().is_empty() {
        println!("[history is empty]");
        return;
    }
    for (i, msg) in engine.history().iter().enumerate() {
        let role = match msg.role {
            project_willamette::chat::Role::System => "SYS",
            project_willamette::chat::Role::User => "USR",
            project_willamette::chat::Role::Assistant => "BOT",
        };
        println!("  [{:>2}] {} | {}", i, role, msg.content);
    }
}

fn print_slash_stats(engine: &ChatEngine<'_, '_>) {
    let pct = 100.0 * (engine.token_position() as f64) / (engine.max_seq_len() as f64);
    println!(
        "  position: {}/{} tokens  ({:.1}% used)",
        engine.token_position(),
        engine.max_seq_len(),
        pct
    );
    println!("  turns:    {} messages", engine.history().len());
}

fn handle_slash_save(rest: &str, engine: &ChatEngine<'_, '_>) -> Result<SlashOutcome> {
    if rest.is_empty() {
        println!("[usage: /save <path>]");
        return Ok(SlashOutcome::Continue);
    }
    let path = PathBuf::from(rest);
    save_history_jsonl(&path, engine.history())?;
    println!(
        "[wrote {} message(s) to {}]",
        engine.history().len(),
        path.display()
    );
    Ok(SlashOutcome::Continue)
}

fn handle_slash_sys(rest: &str, engine: &mut ChatEngine<'_, '_>) -> Result<SlashOutcome> {
    if rest.is_empty() {
        println!("[usage: /sys <text>  or  /sys off]");
        return Ok(SlashOutcome::Continue);
    }
    if rest == "off" {
        engine.set_system_prompt(None);
        println!("[system prompt cleared]");
    } else {
        engine.set_system_prompt(Some(rest.to_string()));
        println!(
            "[system prompt set ({} chars) — takes effect from next /reset]",
            rest.len()
        );
    }
    Ok(SlashOutcome::Continue)
}

fn save_history_jsonl(
    path: &Path,
    history: &[project_willamette::chat::ChatMessage],
) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)
        .with_context(|| format!("create save file: {}", path.display()))?;
    for msg in history {
        let role = match msg.role {
            project_willamette::chat::Role::System => "system",
            project_willamette::chat::Role::User => "user",
            project_willamette::chat::Role::Assistant => "assistant",
        };
        let content_escaped = msg
            .content
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        writeln!(
            f,
            "{{\"role\":\"{}\",\"content\":\"{}\"}}",
            role, content_escaped
        )?;
    }
    f.flush()?;
    Ok(())
}

fn cmd_logits(path: &Path, prompt: &str, top_k_n: usize, no_bos: bool) -> Result<()> {
    let mmap =
        ModelMmap::open(path).with_context(|| format!("opening model file: {}", path.display()))?;
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)
        .map_err(|e| anyhow::anyhow!("tokenizer load failed: {}", e))?;
    let graph = ModelGraph::from_gguf(&gguf)
        .map_err(|e| anyhow::anyhow!("model graph load failed: {}", e))?;

    let mut opts = tokenizer.default_encode_options();
    if no_bos {
        opts.add_bos = false;
    }
    let prompt_ids = tokenizer
        .encode(prompt, opts)
        .map_err(|e| anyhow::anyhow!("encode failed: {}", e))?;
    if prompt_ids.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }

    let hidden = multi_token_forward(&graph, &prompt_ids)
        .map_err(|e| anyhow::anyhow!("forward failed: {}", e))?;
    let logits = compute_logits_from_graph(&hidden, &graph)
        .map_err(|e| anyhow::anyhow!("logits failed: {}", e))?;
    let am = argmax(&logits).unwrap_or(0);
    let top = top_k(&logits, top_k_n);

    println!("# willamette logits dump");
    println!("model:      {}", path.display());
    println!("prompt:     {:?}", prompt);
    println!("add_bos:    {}", opts.add_bos);
    println!("ntoks:      {}", prompt_ids.len());
    println!("tokens:     {:?}", prompt_ids);
    println!(
        "tokens_str: {:?}",
        prompt_ids
            .iter()
            .map(|&t| tokenizer.token_str(t).unwrap_or("<?>").to_string())
            .collect::<Vec<_>>()
    );
    println!();
    println!("argmax_id:  {}", am);
    println!("argmax_str: {:?}", tokenizer.token_str(am).unwrap_or("<?>"));
    println!();
    println!("# rank | id    | logit          | token_str");
    for (rank, (id, l)) in top.iter().enumerate() {
        println!(
            "{:4}   {:6}   {:14.6}   {:?}",
            rank,
            id,
            l,
            tokenizer.token_str(*id).unwrap_or("<?>")
        );
    }
    Ok(())
}

fn cmd_bench(path: &Path, decode_steps: usize) -> Result<()> {
    use std::time::Instant;

    let mmap =
        ModelMmap::open(path).with_context(|| format!("opening model file: {}", path.display()))?;
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).map_err(|e| anyhow::anyhow!("GGUF parse error: {}", e))?;
    let graph = ModelGraph::from_gguf(&gguf)
        .map_err(|e| anyhow::anyhow!("model graph load failed: {}", e))?;

    let n_embd = graph.config.embedding_length as usize;
    // Both fields come from the dispatch module so the bench banner
    // can never disagree with what the matvec loop actually calls.
    let host_arch = std::env::consts::ARCH;
    let backend = project_willamette::model::dispatch::active_kernel().label();

    println!("==================================================");
    println!("Bench (Stage 6 — BitLinear matvec timings)");
    println!("==================================================");
    println!("Host arch:        {}", host_arch);
    println!("Matvec backend:   {}", backend);
    println!("Model:            {}", path.display());
    println!("Block count:      {}", graph.config.block_count);
    println!("Vocab size:       {}", graph.config.vocab_size);
    println!();

    // Hard-coded token id 15339 ("Hello" in Llama-3 tokenizer) is fine
    // for the real BitNet 2B (vocab 128256) and the synthetic Medium
    // preset (vocab 32000), but tiny / small / future-small presets
    // have smaller vocabs. Clamp into range — every position in the
    // embedding table produces a valid f32 vector regardless of which
    // row we pick. Throughput numbers are unaffected.
    let bench_token = if graph.config.vocab_size > 15339 {
        15339
    } else {
        0
    };

    // ── 1) Single BitLinear I2_S matvec (attn_q in layer 0) ──
    let mut x = vec![0.0_f32; n_embd];
    embedding_gather_f16(graph.token_embd, bench_token, &mut x)
        .map_err(|e| anyhow::anyhow!("embed: {}", e))?;
    let attn_q = graph.layers[0].attn_q;
    let mv_in = attn_q.shape[0] as usize;
    let mv_out = attn_q.shape[1] as usize;
    let mut q = vec![0.0_f32; mv_out];

    // Warm-up run.
    bitlinear_i2s_matvec_f32(attn_q, &x, &mut q)
        .map_err(|e| anyhow::anyhow!("warm-up matvec: {}", e))?;
    let t = Instant::now();
    bitlinear_i2s_matvec_f32(attn_q, &x, &mut q).map_err(|e| anyhow::anyhow!("matvec: {}", e))?;
    let matvec_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "BitLinear matvec (attn_q, in_dim={}, out_dim={}):",
        mv_in, mv_out
    );
    println!("  Time:           {:.3} ms", matvec_ms);
    println!(
        "  Throughput:     {:.2} M elements / sec",
        (mv_in as f64 * mv_out as f64) / (matvec_ms / 1000.0) / 1.0e6
    );
    println!();

    // ── 2) Single-token full forward (no cache) ──
    let t = Instant::now();
    let _hidden = forward_single_token_position_zero(&graph, bench_token)
        .map_err(|e| anyhow::anyhow!("forward: {}", e))?;
    let forward_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Single-token forward ({} layers, no cache):",
        graph.config.block_count
    );
    println!("  Time:           {:.1} ms", forward_ms);
    println!("  Throughput:     {:.2} tokens/sec", 1000.0 / forward_ms);
    println!();

    // ── 3) Decode-step with KV cache ──
    let kv_dim = graph.config.kv_dim as usize;
    let mut cache = KVCache::new(graph.layers.len(), kv_dim, decode_steps + 8);
    // Warm-up: prefill 1 token.
    let _ = forward_with_cache(&graph, &mut cache, bench_token, 0)
        .map_err(|e| anyhow::anyhow!("prefill: {}", e))?;

    let mut decode_total = 0.0_f64;
    let mut samples = 0usize;
    for step in 0..decode_steps {
        let t = Instant::now();
        let _ = forward_with_cache(&graph, &mut cache, bench_token, (step + 1) as u32)
            .map_err(|e| anyhow::anyhow!("decode step: {}", e))?;
        decode_total += t.elapsed().as_secs_f64();
        samples += 1;
    }
    let decode_avg_ms = if samples > 0 {
        (decode_total / samples as f64) * 1000.0
    } else {
        0.0
    };
    println!(
        "Decode-step forward (with KV cache, average of {} runs):",
        samples
    );
    println!("  Time:           {:.1} ms", decode_avg_ms);
    println!("  Throughput:     {:.2} tokens/sec", 1000.0 / decode_avg_ms);
    println!();
    println!("Tolerance vs scalar reference is documented in");
    println!("tests/bitlinear_simd.rs (max abs diff < 1e-2 across the");
    println!("210 BitLinear weights of layer 0). Re-run with");
    println!("RUST_LOG=info cargo test --release bitlinear_simd to see");
    println!("the per-tensor numbers.");
    Ok(())
}

fn format_value(v: &GgufValue) -> String {
    match v {
        GgufValue::Uint8(x) => format!("u8   {}", x),
        GgufValue::Int8(x) => format!("i8   {}", x),
        GgufValue::Uint16(x) => format!("u16  {}", x),
        GgufValue::Int16(x) => format!("i16  {}", x),
        GgufValue::Uint32(x) => format!("u32  {}", x),
        GgufValue::Int32(x) => format!("i32  {}", x),
        GgufValue::Float32(x) => format!("f32  {}", x),
        GgufValue::Bool(x) => format!("bool {}", x),
        GgufValue::Uint64(x) => format!("u64  {}", x),
        GgufValue::Int64(x) => format!("i64  {}", x),
        GgufValue::Float64(x) => format!("f64  {}", x),
        GgufValue::Str(s) => {
            let body = safe_truncate(s, 240);
            if body.len() < s.len() {
                format!("str  \"{}…\" ({} bytes total)", body, s.len())
            } else {
                format!("str  \"{}\"", body)
            }
        }
        GgufValue::Array(arr) => {
            let preview_n = 5.min(arr.len());
            let previews: Vec<String> = arr
                .iter()
                .take(preview_n)
                .map(format_value_compact)
                .collect();
            let ellipsis = if arr.len() > preview_n { ", …" } else { "" };
            format!(
                "array(len={}) [{}{}]",
                arr.len(),
                previews.join(", "),
                ellipsis
            )
        }
    }
}

fn format_value_compact(v: &GgufValue) -> String {
    match v {
        GgufValue::Str(s) => {
            let body = safe_truncate(s, 48);
            if body.len() < s.len() {
                format!("\"{}…\"", body)
            } else {
                format!("\"{}\"", body)
            }
        }
        GgufValue::Uint8(x) => x.to_string(),
        GgufValue::Int8(x) => x.to_string(),
        GgufValue::Uint16(x) => x.to_string(),
        GgufValue::Int16(x) => x.to_string(),
        GgufValue::Uint32(x) => x.to_string(),
        GgufValue::Int32(x) => x.to_string(),
        GgufValue::Uint64(x) => x.to_string(),
        GgufValue::Int64(x) => x.to_string(),
        GgufValue::Float32(x) => x.to_string(),
        GgufValue::Float64(x) => x.to_string(),
        GgufValue::Bool(x) => x.to_string(),
        GgufValue::Array(arr) => format!("array(len={})", arr.len()),
    }
}

fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let body: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{}…", body)
    }
}

fn cmd_synth_gguf(output: &Path, preset: project_willamette::synth::Preset) -> Result<()> {
    let cfg = preset.config();
    let est = project_willamette::synth::estimated_params(&cfg);

    println!("==================================================");
    println!("Synthetic BitNet b1.58 GGUF builder");
    println!("==================================================");
    println!("Preset:           {}", preset.name());
    println!("Estimated params: {} ({:.1} M)", est, est as f64 / 1e6);
    println!(
        "Layers:           {}  n_embd: {}  n_ff: {}  heads: {}  vocab: {}",
        cfg.n_layers, cfg.n_embd, cfg.n_ff, cfg.head_count, cfg.vocab_size
    );

    // Tiny preset keeps the all-zero weights so the existing test
    // suite's numerical assertions stay valid. Small / Medium are
    // explicitly for throughput measurement and need real ternary
    // distribution to exercise the matvec data path.
    let random_weights = !matches!(preset, project_willamette::synth::Preset::Tiny);
    let bytes = project_willamette::synth::build_gguf(preset, random_weights);
    std::fs::write(output, &bytes)
        .with_context(|| format!("write synthetic GGUF to {}", output.display()))?;
    println!(
        "Output:           {} ({:.1} MB)",
        output.display(),
        bytes.len() as f64 / 1e6
    );
    println!(
        "Note:             No tokenizer in this file. `inspect` and `bench` work; \
         `run`/`chat`/`tui` will not — random ternary weights produce garbage tokens \
         by construction (see src/synth.rs)."
    );
    Ok(())
}
