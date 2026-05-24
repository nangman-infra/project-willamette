# Limitations — Project Willamette v0.1.0 MVP

This document is the honest counter-balance to [`README.md`](README.md).
Read this **before** treating the project as a general LLM runtime.

## 1. Scope

Project Willamette is a **BitNet-specific runtime**, not a general
GGUF / LLM inference framework. It targets exactly:

| Model | Quant | Tokenizer |
| ----- | ----- | --------- |
| `microsoft/BitNet-b1.58-2B-4T` (GGUF distribution `microsoft/bitnet-b1.58-2B-4T-gguf`) | `I2_S` (raw ggml_type = 36) | `gpt2` byte-level BPE with `LLAMA_VOCAB_PRE_TYPE_DEFAULT` pre-tokeniser |

Anything outside this combination returns a typed error
(`UnsupportedArchitecture`, `UnsupportedTensorType`,
`UnsupportedTokenizer`, `NotImplemented`) — by design.

### Not supported

* **Other architectures.** Only `general.architecture = "bitnet-b1.58"`
  is accepted. Plain `bitnet` (24/26 layer Microsoft BitNet) and
  `bitnet-25` use the same forward graph upstream but Willamette has
  not been validated on them; constructing a `BitNetConfig` from one
  will reject the architecture string.
* **Other GGUF quantisations.** F32, F16, Q4_0, Q4_K, Q8_0, IQ4_XS,
  TL1, TL2, … will parse via `willamette inspect` (the GGUF reader
  enumerates them and labels them by raw u32) but the BitLinear matvec
  refuses to operate on anything except `BitNetI2S`. There is no
  Q-something matmul kernel.
* **Other tokenizer models.** `tokenizer.ggml.model = "llama"` (i.e.
  SentencePiece-Unigram) is rejected. The tokenizer factory in
  `src/tokenizer/mod.rs` returns `UnsupportedTokenizer` for any model
  type other than `"gpt2"`.
* **Other pre-tokeniser hints.** If a future GGUF arrives with
  `tokenizer.ggml.pre = "llama-bpe"` (instead of the missing-key
  default our reference file has), the LLaMA 3 regex set in
  `llama-vocab.cpp:373..381` is NOT yet implemented in Willamette —
  we use only the DEFAULT 3-regex set.

## 2. Performance

The performance numbers in [`README.md`](README.md) are **single-host,
single-thread** measurements on an Apple Silicon M-series Mac with our
reference scalar and NEON kernels. They are not portable promises.

| Concern | Status |
| ------- | ------ |
| Apple Silicon NEON | implemented, validated against scalar (Stage 6-C) |
| **x86 AVX2** | **not yet implemented** — needs an x86 host to validate against scalar; Project Willamette principle is "no unverified SIMD merge" |
| **x86 SSE2** | same |
| Apple Silicon with `+dotprod` (i8×i8 dot product) | not used; current NEON path keeps f32 activations for parity with scalar |
| Multi-threading | not implemented; one BitLinear matvec runs on one core |
| GPU (CUDA / Metal / Vulkan / ROCm) | not implemented |
| Batched / multi-token-per-step decoding | the multi-token path exists for prompt prefill, but per-step decode is single-token |
| Memory-pinned KV cache | the KV cache lives in normal heap memory and grows linearly with context length; no swap / eviction |

On x86 hosts Willamette currently falls back to the scalar reference,
which clocks roughly **0.2 tokens/sec on a 2.4 B parameter model**.
This is correctness-first, not throughput-first.

## 3. Numerical equivalence

NEON-vs-scalar matvec results differ by `~1e-3` absolute per element
(documented in `tests/bitlinear_simd.rs`), which is small enough that
greedy / sampling argmax matches scalar for all four reference
prompts — but it is NOT bit-identical. Anyone diffing intermediate
hidden states across backends should expect small float deltas.

Reference parity vs. bitnet.cpp is verified at the **byte level for
generated text** and at the **token-id level for prompt tokens**.
Internal hidden states are not compared (the upstream binary doesn't
dump them by default).

## 4. Error surfaces

The following errors are real and intentional, not "should never
happen" guards:

* `UnsupportedArchitecture("xxx")` — `general.architecture` is not
  `"bitnet-b1.58"`.
* `UnsupportedTensorType(N)` — any tensor whose raw `u32` ggml_type is
  not one of the small set we recognise. `inspect` will print the raw
  number; if that number is genuinely a new BitNet type, upgrade
  [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md) and `src/gguf/types.rs`
  together.
* `UnsupportedTokenizer("…")` — described above.
* `NotImplemented("…")` — Stage-specific features that haven't shipped.
* `InvalidMagic`, `UnsupportedVersion`, `TensorOutOfBounds`,
  `MetadataTypeMismatch`, `MissingMetadata`, `StringOverflow` —
  GGUF parse-time integrity errors. None of them ever silently
  proceed to inference.

## 5. What this project is NOT

* It is **not** a drop-in replacement for `llama-cli` or
  `llama-server`. There is no OpenAI-compatible HTTP server, no chat
  templating engine, no LoRA loading, no multi-model orchestration.
* It is **not** a production runtime. There is no graceful
  out-of-memory handling for the KV cache, no streaming protocol, no
  cancellation token, no rate-limiting.
* It is **not** a benchmark suite. The `willamette bench` subcommand
  measures three primitives; it is not Criterion, not MLPerf, and not
  a substitute for either.

## 6. What the project IS aimed at

* A small (~3000 LoC), source-pinned, auditable Rust runtime for one
  specific BitNet 1.58-bit GGUF file.
* A reproducible reference against which other implementations can
  diff their I2_S BitLinear semantics.
* An honest baseline for further BitNet-on-CPU work (Stage 8 x86,
  potential Stage 9 thread pool, etc.).

If the README claim doesn't appear in this file's "supported" column,
treat it as **not validated**. If you need a guarantee for any of the
gaps above, please file an issue with the exact use case before
relying on the code.
