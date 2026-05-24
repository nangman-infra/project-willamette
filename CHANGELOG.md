# Changelog

All notable changes to Project Willamette are recorded here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
project version increments follow [SemVer](https://semver.org/) — the
`-mvp` suffix marks the first release that hits the v0.1.0 MVP bar
defined in [`README.md`](README.md).

## [Unreleased]

_No changes yet._

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
