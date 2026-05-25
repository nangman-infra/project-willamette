# Changelog

All notable changes to Project Willamette are recorded here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
project version increments follow [SemVer](https://semver.org/) — the
`-mvp` suffix marks releases that still treat the runtime as an MVP
rather than a stabilised library.

### Versioning policy (carried forward)

| Change | Bumps |
| ------ | ----- |
| Bug fix (same intent, corrected result) | `patch` |
| Public CLI / API addition (new subcommand, new public function) | `minor` |
| Public CLI / API change or removal (breaking) | `major` |
| Internal-only (CI, refactor, docs, clippy cleanup) | _no bump; only `[Unreleased]` notes_ |
| Model compatibility breakage (new ggml_type required, new tokenizer pre mandatory) | `major` or `minor`, sized by user impact |

The `-mvp` suffix is kept while iterating in `v0.1.x` and `v0.2.x`.
It will be dropped on the first release we feel comfortable advertising
as a stable library — at which point the next tag becomes `v0.3.0`
(or `v1.0.0` if there is also a public API guarantee).

## [Unreleased]

_No changes yet._

## [v0.3.0-mvp] — 2026-05-25

Minor release: operator-grade chat TUI.

The v0.2.x TUI was usable but missing the editing baseline every
modern terminal AI tool provides — arrow keys, history recall,
search, mid-turn cancel, paste, real cursor — and gave no
visibility into engine / system state. v0.3.0 fills both gaps
in one cycle.

### Added

#### Right-pane live perf dashboard
* New `src/chat/sysmon.rs` — 1 Hz polling of `sysinfo` over a
  daemon thread, normalised process CPU %, per-core %, memory.
* New `src/chat/dashboard.rs` — pure render fn producing six
  sections: HARDWARE, CPU, MEMORY, INFERENCE, SAMPLING, MODEL.
  Gauges with traffic-light coloring (green ≤ 60 %, yellow ≤ 85 %,
  red > 85 %). Per-core display collapses cores ≥ 13.
* Dashboard lives at terminal width ≥ 72; narrower widths fall
  back to single-pane.

#### Readline-grade input editor
* New `src/chat/input_editor.rs` — pure data structure, fully
  unit-tested (19 inline tests).
* Cursor movement: ← →, Home, End, Ctrl-A / Ctrl-E.
* Deletion: Backspace, Delete, Ctrl-W (word), Ctrl-U (to start),
  Ctrl-K (to end).
* History: ↑ / ↓ to recall previous prompts (ring buffer cap 1000).
* Reverse search: Ctrl-R opens overlay; type to filter newest-
  first; Enter loads, Esc cancels, Ctrl-R steps to next older
  match.
* UTF-8 atomic: cursor moves snap to char boundaries, multi-byte
  CJK / emoji codepoints never split.
* Persisted history at `~/.config/willamette/history`,
  cap 1000 entries, append-on-submit, oldest evicted.

#### Inference observability
* Layer-progress display: `forward_with_cache_progress` calls an
  optional `on_layer(layer_idx)` callback after each transformer
  block; the TUI shows "layer 17 / 30" updating live.
* Live tok/s: rolling token count + turn elapsed in atomics;
  dashboard reads them every frame.
* ETA estimation: remaining tokens ÷ live tok/s.

#### Mid-turn cancel
* Esc during generation sets `WorkerProgress.cancel_requested`
  atomic; engine checks at each iteration and exits cleanly.
  History truncates to whatever was emitted so KV cache stays
  consistent for the next turn.

#### Discoverability + polish
* F1 / `/help` opens a Help overlay popup with the full keybinding
  cheatsheet.
* Tab completes slash commands; ambiguous prefix shows candidates.
* Unknown slash command suggestions via Levenshtein distance ≤ 2:
  "unknown /reser — did you mean /reset?".
* Real terminal cursor in input area via
  `Frame::set_cursor_position` (was: rendered `_` glyph).
* Mouse wheel scrolling on chat log (3 lines per tick).
* Bracketed paste support: multi-line paste arrives as a single
  `Event::Paste(String)`, inserted at cursor.
* Ctrl-Y "yanks" the last bot response to system clipboard via
  OSC52 (works in iTerm2 / Kitty / Alacritty / wezterm / recent
  xterm). Inline 30-LoC base64 encoder — no new dep.
* Ctrl-L clears visible chat log without dropping history.
* Ctrl-Home / Ctrl-End jump to top/bottom of chat scrollback.

### Changed
* `ChatEngine` gains `set_worker_progress` + nine read-only
  accessors (config\_*, sampler, system_prompt,
  estimate_kv_cache_bytes). Existing call sites unaffected.
* `Sampler` exposes `params()` and `params_clone()` for the
  dashboard.
* Stdio `willamette chat` is unchanged — TUI features don't
  affect the simpler stdio surface.

### Dependencies
* `sysinfo = "0.32"` (~50 KB compiled). Cross-platform CPU/memory
  sampling. The only new dep this cycle.

### Tests
* 272 total (v0.2.3 had 242 — 30 new).
  * 19 new in `input_editor::tests`
  * 3 new in `sysmon::tests`
  * 6 new in `dashboard::tests`
  * 2 new in `tui::tests` (Levenshtein + base64)

### Notes
* No public API removal. No numeric inference change.
* Live verification still requires a real TTY — headless tests
  cover everything but the actual ratatui draw + mouse + paste.
  See `_internal/VISION.md` for the planned QEMU bench harness
  (Stage 27) that will let humble-hardware UX be measured too.
* This release closes "what every terminal AI tool already
  provides" gap. Next likely cycle: Phase III generalisation
  (multi-architecture model support — Llama / Mistral / Phi).

## [v0.2.3-mvp] — 2026-05-25

Patch release: three TUI / chat readability fixes after real-session
feedback on v0.2.2.

### Fixed
* **Emoji clutter in chat output.** The base model often writes
  trailing pictograph clusters (`😊👍🏻💬📚✨` etc.) lifted from
  social-media training data. They show up everywhere and don't
  add information. `is_emoji_char(char)` covers the major emoji
  blocks (plus ZWJ + variation selectors); `strip_emoji_chars`
  filters them from both the live tick stream and the recorded
  history. The model still emits the underlying tokens so the KV
  cache stays in sync with what it thinks it said; we just hide
  the visual noise.
* **`User:` leak on screen.** v0.2.2's stop-sequence detection
  worked at the *history* level — next turn's prefill saw clean
  text — but the bytes that triggered detection (`User:`,
  `BITNETAssistant:`, …) had already streamed to the caller's
  `tick` callback and were visible on screen for the truncated
  turn. v0.2.3 holds back a 24-byte look-ahead buffer, sliding
  it forward each step; if a stop sequence appears in the
  buffered tail, the tail is discarded before it can be ticked.
  Trade-off: ~24 bytes (3-6 words) of streaming latency. The
  surface output is now genuinely clean.

### Added
* **Markdown rendering in the TUI.** `src/chat/markdown.rs` —
  new ~160-LoC inline renderer maps `**bold**`,
  `` `inline code` ``, `# heading`, `- bullet`, `1. numbered`,
  with leading whitespace preserved, onto `ratatui::Line`/`Span`
  styled output. BOT messages and the live streaming response
  both go through it; USR / SYS bubbles stay plain text.
* `append_message_lines` helper in `src/chat/tui.rs` — the role
  badge is prepended to the first body line; continuation lines
  align under the badge; the streaming path appends a green `▌`
  cursor so the user can see generation is live.
* `floor_char_boundary` inline helper in `src/chat/engine.rs` —
  the unstable `str::floor_char_boundary` is unavailable on
  stable Rust, so we provide our own. Used by the look-ahead
  buffer to snap the safe-emit boundary onto a UTF-8 char so we
  never tick a half-codepoint.

### Tests
* Total: **242** (v0.2.2 had 221 — 21 new this cycle).
  * 14 new in `src/chat/markdown.rs::tests` — bold, inline code,
    bullet (`-` and `*`), numbered, multidigit, heading,
    indented bullet, indent preservation, plain pass-through,
    unclosed-bold-stays-literal, period-in-version-not-list,
    multiple lines, realistic Korean-history multi-line.
  * 7 new in `src/chat/engine.rs::stop_sequence_tests` —
    `is_emoji_char` for pictographs and CJK preservation,
    `strip_emoji_chars` for trailing clutter / Korean
    interleave / ZWJ sequences, `floor_char_boundary` on ASCII
    and Korean.
* Coverage: SonarQube `new_coverage` stays at 100 % on v0.2.x
  new code; the new modules are unit-tested in-CI without the
  real model file.

### Notes
* No public-API change. No CLI flag change. No numeric inference
  change. Reference parity with pinned bitnet.cpp on Stage 5-E
  prompts is preserved.
* `--no-emoji` / `--no-markdown` CLI flags intentionally not
  added — the cleaner output is the right default for a chat
  surface. If a future use-case ever needs them they're easy to
  graft onto `ChatArgs`.

## [v0.2.2-mvp] — 2026-05-25

Patch release: chat usability + honest in-CI coverage.

The user-facing v0.2.1 chat was leaking model self-talk into the
visible output (the base model writes its own follow-up
`User:` / `BITNETAssistant:` turns past the answer). This release
detects those patterns and truncates. It also replaces the
v0.2.0 cycle's Sonar `coverage.exclusions`-shaped fix with actual
in-CI tests covering 59 new lines that were previously at 0 %.

### Fixed
* **`ChatEngine` runaway**: `stream_assistant_response` now checks
  the accumulated response text after each token for hallucinated
  turn-boundary phrases — `BITNETAssistant:`, `User:`, `Human:`,
  `Human (`, `User (`, `AI Assistant:`, `Assistant:`, `Question:`,
  and 3 more variants observed in real-model output — and breaks
  out of the generation loop on the first match. The recorded
  `history` is truncated at the boundary so subsequent turns see
  a clean transcript. Empirical: a single `"how are you?"` turn
  that used to spill 543 tokens of fake hash-function tutorial now
  stops cleanly at 51 tokens.
* **`UnsupportedTokenizer` out-of-range messaging**: now consistent
  across the `encode_with_specials` synthetic test (was implicit
  before; tests in `tests/tokenizer_synthetic.rs` lock it in).

### Added
* `find_chat_stop_sequence(&str) -> Option<usize>` and
  `truncate_at_chat_stop_sequence(&mut String) -> bool` as
  `pub(crate)` helpers — pure functions, 15 inline unit tests
  cover Unicode safety, false-positive guards (`Humans`/`Users`),
  earliest-match selection, and the verbatim v0.2.1 TUI failure
  string.
* `tests/tokenizer_synthetic.rs` extended with 7 new tests that
  build a valid in-memory tokenizer GGUF (256 byte-level glyphs
  + BOS + EOS, no merges) and exercise `encode_with_specials`,
  `encode`, and the BOS/EOS plumbing without needing the 1.1 GiB
  model file.
* `tests/synthetic_model.rs` — new ~280-LoC test file that builds
  a complete in-memory BitNet b1.58 GGUF (≈73 KB; 2 layers,
  n_embd 128, vocab 4, all-1.0 norms, all-0 BitLinear, all-1.0
  embeddings) and exercises `ModelGraph::from_gguf`,
  `forward_single_token_position_zero`, `forward_with_cache`,
  and `multi_token_forward`. 6 new tests cover:
  full GGUF parse, norm-cache pre-decode invariant, no-NaN
  forward output, KV cache continuity across 2 positions, and
  cache-vs-no-cache parity at position 0.
* `Quality Gate breakdown` step in `.github/workflows/sonar.yml`
  — queries the SonarQube REST API on every scan and prints the
  per-condition pass/fail with actual values, so future Quality
  Gate failures are debuggable from the GitHub Actions log
  without dashboard access. `continue-on-error: true` so a
  transient API hiccup never blocks the official gate-action.
* `ChatArgs` shared clap argument group + `build_sampling_params`
  helper in `src/main.rs` — DRY for the `chat` and `tui`
  subcommands.
* `print_slash_help`, `print_slash_history`, `print_slash_stats`,
  `handle_slash_save`, `handle_slash_sys` — per-command helpers
  in `src/main.rs`.
* `build_chat_fragment`, `prefill_prompt_tokens`,
  `stream_assistant_response`, `emit_token_bytes` — helper
  methods on `ChatEngine` extracted from the previously-monolithic
  `send_user_message`.
* `drain_token_events`, `apply_token_event`, `finish_bot_turn`,
  `fail_bot_turn`, `clear_transient_if_old`, `poll_one_input` —
  helpers extracted from `ui_loop` in `src/chat/tui.rs`.

### Changed
* **Sonar action bumped** `SonarSource/sonarqube-scan-action@v6`
  → `@v7.2.1` — v6 is a Node 20 action that GitHub now force-runs
  on Node 24 with a deprecation warning. v7.2.1 is the last v7
  release before v8.0.0 flipped `skipSignatureVerification` to
  `false` (breaking change); v7.2.1 is drop-in compatible with v6.
* **Sonar `coverage.exclusions`** pruned back to only what truly
  cannot run in CI: `src/main.rs` (interactive CLI),
  `src/chat/**` (TTY-dependent), `src/model/bitlinear_neon.rs`
  (aarch64-only kernel; x86 CI runner compiles it out via cfg),
  `scripts/**`, `.github/**`. Everything else (model load,
  forward path, tokenizer module) now has real in-CI coverage
  via the new synthetic-GGUF tests above.
* **Cognitive Complexity** (rust:S3776) fixed on the four
  functions Sonar flagged: `cmd_chat` (was 20 → ≤15),
  `handle_slash_command` (19 → ≤15), `send_user_message`
  (23 → ≤15), `ui_loop` (31 → ≤15). All four broken out into
  named helpers.
* **`Tokenizer::encode_with_specials` test** rewritten with
  `ids.contains(&eot_id)` instead of `iter().any(|&id| id == eot_id)`
  to satisfy Rust 1.95.0's new `clippy::manual_contains` lint.

### Tests
* Total: **221** (v0.2.1 had 193 — 28 new this cycle).
* Coverage: SonarQube `new_coverage = 100.0 %` on the v0.2.x new
  code, replacing the previous 0 % (which was hidden by a blanket
  exclusion).
* Quality Gate: ✔ OK on all 3 conditions
  (`new_violations` 0, `new_duplicated_lines_density` 1.34 %,
  `new_coverage` 100 %).

### Notes
* No public-API change to any of the existing subcommands. The
  `chat` / `tui` CLI flag surface is unchanged (clap's
  `#[command(flatten)]` preserves the argument layout).
* No numeric inference change. Reference parity with the pinned
  bitnet.cpp on Stage 5-E prompts is preserved.

## [v0.2.1-mvp] — 2026-05-25

Patch release: chat-template choice tuned for the base model.

Empirical testing of v0.2.0's chat surface showed two failure
modes — every response was prefixed with a hallucinated tag
(`PowerShell>`, `Vietnamese>`, `French>`, …) and the model would
not honour even trivial instructions like "tell me only english."

Investigation:

* `microsoft/bitnet-b1.58-2B-4T-gguf` is a **base/foundation
  model**, not instruct-tuned. The GGUF self-description is plain
  `general.name = "bitnet2b"` (no Instruct tag); the upstream
  `microsoft/BitNet` README:245 documents `-cnv, --conversation` as
  being "for instruct models" and lists eligible repos — this one
  is not in that list. The model was trained on 4 T tokens of web
  text without SFT or RLHF; expecting it to follow instructions is
  out of scope for a base model.
* The GGUF includes a `tokenizer.chat_template` Jinja string of the
  shape `Human: <content>\n\nBITNETAssistant: <eos_token>`, but
  that template was inserted unconditionally by the conversion
  script (`utils/convert-ms-to-gguf-bitnet.py:1324`) regardless of
  whether the model itself was trained on that pattern. The
  `eos_token` variable was therefore never grounded in any specific
  inference-time token id during training.

What v0.2.0 did wrong: it injected `<|eot_id|>` (128009) between
turns, interpreting Jinja `eos_token` as the LLaMA-3 turn
boundary. Empirically this pushed the model into the
"language-prefix" failure mode above.

### Fixed
* `ChatEngine::send_user_message` now uses a plain text bridge
  (`\n\nHuman: <content>\n\nBITNETAssistant: `) between turns
  instead of injecting a Jinja-template-derived turn marker. The
  same prompt that produced `"PowerShell> Hello!"` in v0.2.0 now
  produces `"Hello! How can I assist you today?"` in v0.2.1.
  Reference parity with bitnet.cpp greedy decode (Stage 5-E
  prompts) is unchanged.

### Unchanged
* `Tokenizer::encode_with_specials(&[PromptPart])` (Stage 9-B) and
  `PromptPart::{Text, Special}` remain in the public API. They were
  needed for the template-faithful approach we just reverted and
  may still be useful for future instruct-tuned BitNet variants
  (e.g. Falcon3-Instruct-1.58bit) that *were* trained with explicit
  turn markers.
* All 193 tests still pass.
* Performance unchanged (this is a chat-template choice, not a
  kernel change).

## [v0.2.0-mvp] — 2026-05-25

Minor release: first-class chat experience + ~5× decode-step speedup.

The inference path's numeric semantics are unchanged from v0.1.x —
greedy decode on the Stage 5-E reference prompts still produces
byte-identical tokens to the pinned bitnet.cpp reference. What's new
is the *runtime surface*: a real chat engine, a full TUI, a launcher,
and a parallelised matvec.

### Added
* `willamette chat` — stdin/stdout multi-turn dialogue subcommand with
  KV-cache reuse across turns, UTF-8-safe streaming output, EOS auto-
  stop, slash commands (`/help`, `/reset`, `/history`, `/save`,
  `/sys`, `/stats`, `/quit`).
* `willamette tui` — ratatui full-screen chat TUI over the same engine
  (history pane, input box, status bar, PgUp/PgDn scrolling).
* `Tokenizer::encode_with_specials(&[PromptPart])` for mid-prompt
  token-id injection — required to render the BitNet chat template's
  `<|end_of_text|>` boundary verbatim instead of byte-level-BPEing it
  into 7 tokens.
* `PromptPart::{Text, Special}` enum.
* `src/chat/engine.rs::ChatEngine` — turn-streaming chat runner.
* `src/chat/tui.rs::run_tui` — terminal UI driver with a worker
  thread + mpsc channels.
* `scripts/willamette` — all-in-one launcher: SHA256-verifies the
  model, optionally downloads it from Hugging Face, rebuilds the
  binary if stale, then launches the requested mode (default TUI).
* `bitlinear_i2s_matvec_f32_neon_i8` — int8-activation NEON kernel
  (Stage 10-D). Code present but **not the default**: see "Changed"
  for why.

### Changed
* **BitLinear matvec is now multi-threaded** via `rayon::par_chunks_mut`
  with chunks of 32 output rows, each chunk owning a thread-local i8
  scratch buffer (Stage 10-C + 10-B). On Apple M1 the decode-step
  improves from `~656 ms / ~1.5 tok/s` (v0.1.1) to
  `~126 ms / ~7.9 tok/s` (v0.2.0) — a 5.2× speedup. The matvec itself
  drops from 1.87 ms to 0.64 ms (2.94×). ISA-neutral: the rayon
  parallelism also helps the scalar fallback on multi-core x86 hosts
  once the SSE2 kernel lands.
* Norm weights (`attn_norm`, `attn_sub_norm`, `ffn_norm`,
  `ffn_sub_norm` per layer, plus `output_norm`) are now pre-decoded
  into `Vec<f32>` at `ModelGraph::from_gguf` time (Stage 10-A). The
  forward path reads them directly — 121 fewer per-token
  allocations.
* `ChatEngine::send_user_message` always forwards the just-emitted
  token into the KV cache (unlike one-shot `generate_with_cache_and_sampler`,
  which skipped the final step). Continuity across turns now matches
  the canonical training-time pattern.
* Stage 10-D int8-activation path investigated and benched. On stable
  Rust the `vdotq_s32` SDOT intrinsic is gated behind the unstable
  `stdarch_neon_dotprod` feature, so the kernel falls back to
  `vmull_s8`-based widening dot. Measured at 7.82 tok/s vs the f32-
  input NEON path's 7.91 tok/s on Apple M1 (20-sample average) — a
  small regression, not a win. The int8 kernel is therefore present
  but gated behind
  `RUSTFLAGS="--cfg willamette_i8_activations"`. Default stays on the
  f32-input NEON path. We'll switch over when `stdarch_neon_dotprod`
  stabilises.

### Dependencies
* `rayon = "1.10"` — for Stage 10-C row parallelism.
* `ratatui = "0.29"` and `crossterm = "0.28"` — for Stage 9-E TUI.

### Tests
* `encode_with_specials` parity (text-only path equals plain
  `encode`), special-id injection, out-of-range rejection, BOS-via-
  `Special` prefix.
* All 189 v0.1.1 tests still pass; total at v0.2.0 is 193 (4 new).

### Performance (Apple M1, NEON, release profile, 20-run avg)

| Metric | v0.1.1 | v0.2.0 | Change |
| ------ | -----: | -----: | -----: |
| BitLinear matvec (attn_q, 2560×2560) | 1.87 ms | 0.64 ms | **2.94× faster** |
| Decode-step with KV cache | 656 ms | 126 ms | **5.19× faster** |
| Tokens / second (decode) | 1.52 | 7.91 | **5.19×** |

## [v0.1.1-mvp] — 2026-05-24

Patch release: bug fix in generation + SonarQube validation lane.
No public-API or behaviour change to the inference path itself
(token-id parity vs. bitnet.cpp from v0.1.0-mvp is preserved).

### Fixed
* `willamette run` no longer panics with `decoded bytes are not valid
  UTF-8` when the generated stream is cut off in the middle of a
  multi-byte character (Korean / CJK / emoji). Generation now ends
  cleanly and the truncated suffix is shown as `U+FFFD`.
* The streaming token-by-token printer in `willamette run` now buffers
  bytes across tokens and emits only up to the last valid UTF-8
  boundary on each tick, so multi-byte characters split across two or
  three BPE tokens no longer silently disappear from the live output.

### Added
* `Tokenizer::decode_to_bytes(ids) -> Vec<u8>` — raw byte stream, no
  UTF-8 validation.
* `Tokenizer::decode_lossy(ids) -> String` — replaces a trailing
  incomplete UTF-8 suffix with `U+FFFD`; keeps internal multi-byte
  characters intact.
* `sonar-project.properties` and
  `.github/workflows/sonar.yml` matching the
  `nangman-crypto-research` SonarQube pattern (quality + sonar
  pipeline, `cargo-llvm-cov` lcov, Quality Gate).

### Changed
* `.github/workflows/ci.yml` removed; its checks (fmt, test, clippy)
  are now performed by the `quality` job in the new SonarQube
  workflow.
* Clippy now passes `cargo clippy --all-targets -- -D warnings`
  cleanly: 23 fixes (6× `manual_is_multiple_of`, 12× kernel-loop
  `needless_range_loop` allowed at module level with rationale,
  1× `too_many_arguments`, 1× `missing_safety_doc`,
  1× `doc_lazy_continuation`, plus 2 test cosmetic fixes).

## [v0.1.0-mvp] — 2026-05-24

Initial MVP. Reads the official `microsoft/bitnet-b1.58-2B-4T-gguf`
GGUF, runs the full BitNet b1.58 forward in Rust, produces text that
matches the pinned bitnet.cpp reference byte-for-byte on four
reference prompts, and gets ~1.5 tokens / second on a single Apple
Silicon core.

### Added — Stage 1: GGUF inspect
* `willamette inspect` CLI subcommand.
* `src/gguf/reader.rs` parser for GGUF v2/v3 metadata + tensor
  directory.
* `src/gguf/tensor.rs` `TensorView` (zero-copy slice into the mmap'd
  file).
* `src/gguf/types.rs` `GgmlType` enum (including the BitNet-fork
  values `I2_S=36`, `I8_S=37`, `TL1=38`, `TL2=39`).
* `src/memory/mmap.rs` `ModelMmap` wrapper.

### Added — Stage 2: Tokenizer
* `willamette tokenize` CLI subcommand.
* `src/tokenizer/byte_unicode.rs` GPT-2 byte↔unicode bijection.
* `src/tokenizer/pretokenize.rs` `LLAMA_VOCAB_PRE_TYPE_DEFAULT`
  3-regex pre-tokeniser (Stage 5-E fix).
* `src/tokenizer/bpe.rs` rank-priority BPE merger.
* `src/tokenizer/mod.rs` `Tokenizer::from_gguf_metadata` factory,
  `EncodeOptions`, byte-level BPE encode/decode.

### Added — Stage 3: I2_S layout
* [`docs/I2_S_LAYOUT.md`](docs/I2_S_LAYOUT.md) — pinned-source
  citations for the 2-bit packing, code → ternary map, scale offset.
* `TensorView::I2S_*` constants and `i2s_scale()` helper.
* `tests/i2s_layout.rs` — 210-I2S-tensor parity tests against the
  real model.

### Added — Stage 4-A: ModelConfig / ModelGraph
* `src/model/config.rs` `BitNetConfig` (loaded purely from GGUF
  metadata).
* `src/model/graph.rs` `ModelGraph` + `LayerWeights` (332 TensorView
  references, shape/dtype-checked at construction).

### Added — Stage 4-B: f32 primitives
* `src/model/primitives.rs` — `f16_to_f32`, `embedding_gather_f16`,
  `rms_norm_f32`, `apply_rope_f32` (NEOX), GQA shape helpers,
  attention scale, causal mask.

### Added — Stage 4-C: BitLinear scalar matvec
* `src/model/bitlinear.rs` `bitlinear_i2s_matvec_f32_scalar`
  (two-accumulator form, no full dequant, packed-only).
* [`docs/BITLINEAR_I2S_MATVEC.md`](docs/BITLINEAR_I2S_MATVEC.md) —
  the function contract, with `file:line` citations.

### Added — Stage 4-D: forward pass
* `src/model/attention.rs` single-token GQA attention at position 0.
* `src/model/ffn.rs` parallel-gated `ReLU²` FFN (per
  `LLM_FFN_RELU_SQR` + `LLM_FFN_PAR`).
* `src/model/block.rs` single transformer block with both residuals.
* `src/model/forward.rs` 30-layer single-token forward.
* `src/model/lm_head.rs` logits from tied `token_embd.weight` (F16).

### Added — Stage 5: Generation
* `willamette run` CLI subcommand with greedy + sampled decoding.
* `willamette logits` CLI subcommand (top-k dump).
* `src/model/multi_forward.rs` no-cache multi-token causal forward
  (Stage 5-B).
* `src/model/kv_cache.rs` + `src/model/cached_forward.rs` KV cache +
  incremental forward (Stage 5-C).
* `src/model/generate.rs` `greedy_generate_no_cache`,
  `greedy_generate_with_cache`, `generate_with_cache_and_sampler`.
* `src/model/sampler.rs` `Sampler` with temperature, top-k, top-p,
  repetition penalty, seedable xorshift PRNG (Stage 5-D).

### Added — Stage 5-E: Reference compatibility
* [`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md)
  with the four-prompt comparison.
* `scripts/run_willamette_reference.sh`,
  `scripts/run_bitnet_reference.sh`,
  `scripts/compare_reference.sh`.
* **Fix**: rewritten pre-tokeniser to apply the
  `LLAMA_VOCAB_PRE_TYPE_DEFAULT` 3-regex set (was a single GPT-2
  regex). Without this, `"1 + 1 ="` tokenised differently than
  bitnet.cpp.

### Added — Stage 6-A: scalar baseline benchmark
* `willamette bench` CLI subcommand.

### Added — Stage 6-C: Apple Silicon NEON
* `src/model/bitlinear_neon.rs` — 16-element NEON dot product with 4
  parallel `float32x4_t` accumulators.
* Runtime dispatch in `src/model/bitlinear.rs` via
  `is_aarch64_feature_detected!("neon")`.
* `tests/bitlinear_simd.rs` — scalar↔NEON tolerance equivalence on
  every layer-0 BitLinear weight.
* 7.5× end-to-end speed-up vs. scalar.

### Added — Stage 7-A: Release hardening
* `README.md`, `LIMITATIONS.md`, `REPRODUCIBILITY.md`,
  `GOLDEN_TESTS.md`, this `CHANGELOG.md`, `ARCHITECTURE.md`.
* `.github/workflows/ci.yml` — fmt + clippy + model-less tests on
  linux-x86_64 and macos-aarch64.
* `.gitignore` — models, generated outputs, editor noise.

### Deferred
* **Stage 6-B** (x86 AVX2 / SSE2) — pending an x86 host on which the
  produced SIMD can be validated against the scalar fallback. No
  unverified SIMD merges per the project rules.
* GPU backends (Metal / CUDA / Vulkan).
* Multi-threaded BitLinear matvec.
* Long-context KV cache eviction.

### Known limitations
See [`LIMITATIONS.md`](LIMITATIONS.md). Highlights:
* Only `microsoft/bitnet-b1.58-2B-4T-gguf` with `I2_S` quant is
  supported.
* Only `tokenizer.ggml.model = "gpt2"` is supported.
* On x86 hosts the matvec runs the scalar fallback (correctness, not
  speed).
