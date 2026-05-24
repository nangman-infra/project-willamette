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
