# Limitations — Project Willamette v0.9.0-mvp

*Last revised 2026-05-30 (mbp2012 Ivy Bridge measurement cycle).*

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

* **Architectures outside the BitNet family.** Willamette accepts the
  BitNet family (`bitnet-b1.58`, `bitnet-25`, `bitnet`) through the
  `ModelArchitecture` registry (see `src/model/architecture/`).
  `bitnet-25` was end-to-end verified on antix1 against
  [`jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF`](https://huggingface.co/jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF)
  and
  [`Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf`](https://huggingface.co/Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf).
  The bare `bitnet` string (paper-era 24/26-layer variants) is
  accepted on the assumption that its metadata prefix matches the
  arch string — that branch will be confirmed the first time such a
  GGUF is in hand. Llama / Mistral / Phi / Gemma remain rejected;
  the design path for them is
  [`docs/PHASE_III_ARCHITECTURE_RFC.md`](docs/PHASE_III_ARCHITECTURE_RFC.md)
  § 5.4 (Phase III-B).
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

The performance numbers in [`README.md`](README.md) and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
are **single-host** measurements on the Apple M4 dev box and the
antiX Pentium-M humble-validation host. They are not portable
promises — other CPUs in the same ISA family will land somewhere on
the scaling curve, not at the exact numbers.

| Concern | Status |
| ------- | ------ |
| Apple Silicon NEON | implemented, validated against scalar (Stage 6-C). Measured on Apple M4. |
| **x86 / i686 SSE2 (int8 activation)** | **default since v0.5.0 / v0.7.0** — validated on antiX Pentium-M 2 GHz. 2.2× over the f32 SSE2 path, ~5.4× over scalar. Byte-identical greedy output to f32 on the real 2B model. |
| x86 / i686 SSE2 (f32 mask-add) | kept as the numerical reference behind `--cfg willamette_sse2_f32`. |
| **x86 AVX2 / AVX-512** | not yet implemented — gain target for modern x86 hosts (Haswell+ AVX2, Skylake-X+ AVX-512). |
| **LUT kernel (scalar)** | **landed on main 2026-05-30 as the default on sub-SSSE3 x86 hosts** via `Kernel::X86Sse2ScalarLut` dispatch arm (`9f95f4d`). Stage 5-E greedy is byte-identical across NEON / SSE2 i8 / scalar LUT paths. **End-to-end tok/s on antix1 did not move** (0.41 → 0.40, within noise): matvec is ≈ 10 % of the decode-step budget, so the matvec-level 5× cut does not become an end-to-end win. Kept for fidelity + a pure-Rust path (vs SSE2 i8's unsafe intrinsics); not a performance claim. See [`docs/LUT_KERNEL_RFC.md`](docs/LUT_KERNEL_RFC.md) § 5 step-3 outcome + `docs/BENCHMARKS.md` 2026-05-30 § "Step-3 end-to-end measurement". |
| **Sparsity-aware skipping** | prototype shipped (`src/model/bitlinear_sparse.rs`), benched, but on antix1 it ties with dense i8 (1.01× slower — irregular access cancels the 42% skip). Documented; not default. Likely a win on sub-SSE2 hardware. |
| Apple Silicon with `+dotprod` / FEAT_DotProd | hardware present on the M4 dev host; the stable-Rust `vdotq_s32` intrinsic remains unused (kernel keeps `vmull_s8`-style widening for parity). Switching is an `RUSTFLAGS="--cfg willamette_i8_activations"` flag away. |
| Apple Silicon with FEAT_I8MM / SME / SME2 | hardware present on M4; intrinsics not in stable Rust → unused. |
| Multi-threading | `rayon` per-row BitLinear matvec parallelism (Stage 10-C). 1-thread and 4-thread runs on mbp2012 (Ivy Bridge, 4 HT threads) measure within run-to-run noise — the matvec is **memory-bandwidth bound** (≈ 6.45 GB/s of i8 weight traffic per token vs DDR3-1600 ~25 GB/s peak), so adding cores does not help on this workload. antix1's 1-core "no rayon gain" is the same constraint surfacing differently. See `docs/BENCHMARKS.md` 2026-05-30. |
| bitnet.cpp same-machine comparison on sub-AVX2 hosts | bitnet.cpp's x86 production CPU path (both the default `ggml-bitnet-mad` scalar fallback and the `BITNET_X86_TL2` LUT path) **effectively assumes AVX2**. On Ivy Bridge (no AVX2): the default build crashes with `SIGILL`, the `GGML_AVX2=OFF` build emits garbage (`!!!!!`), and the LUT build fails to compile. Willamette's hand-written SSE2 i8 kernel produces byte-identical Stage 5-E output on the same machine — see `docs/BENCHMARKS.md` 2026-05-30 § "bitnet.cpp head-to-head". The reference comparison in `docs/REFERENCE_COMPATIBILITY.md` therefore stays on AVX2-capable hosts. |
| GPU (CUDA / Metal / Vulkan / ROCm) | not implemented (out of scope by thesis). |
| Batched / multi-token-per-step decoding | the multi-token path exists for prompt prefill, but per-step decode is single-token. |
| KV cache memory | **per-token absmax i8** since v0.9.0 — 3.97× smaller than the prior f32 layout (37.7 KB/token vs 150 KB/token on BitNet 2B). Lifts the practical chat-history ceiling on antix1 from ~3 K to ~13 K tokens. See [`docs/KV_CACHE_QUANT.md`](docs/KV_CACHE_QUANT.md). Lives in normal heap memory; no swap / eviction. |

On x86 hosts Willamette currently falls back to the scalar reference,
which clocks roughly **0.2 tokens/sec on a 2.4 B parameter model**.
This is correctness-first, not throughput-first.

## 3. Numerical equivalence

NEON-vs-scalar matvec results differ by `~1e-3` absolute per element
(documented in `tests/bitlinear_simd.rs`), which is small enough that
greedy / sampling argmax matches scalar for all four reference
prompts — but it is NOT bit-identical. Anyone diffing intermediate
hidden states across backends should expect small float deltas.

Since v0.9.0 the KV cache stores i8 per-token absmax quantised K and
V tensors, so the *cached* forward path is also no longer bit-equal
to the no-cache reference (per-element drift on the order of
`absmax / 254`). The contract is now **cosine ≥ 0.999 on the
post-`output_norm` hidden** plus **byte-identical greedy
token-id sequences**; both are enforced by `tests/kv_cache.rs`.
The Stage 5-E reference prompt "The capital of France is" produces
`[12366, 13, 12366] = " Paris. Paris"` byte-identical on Apple M4
NEON and antix1 i686 SSE2 i8 paths — i8 KV did not flip any argmax
on the reference set.

Reference parity vs. bitnet.cpp is verified at the **byte level for
generated text** and at the **token-id level for prompt tokens**.
Internal hidden states are not compared (the upstream binary doesn't
dump them by default).

## 4. Error surfaces

The following errors are real and intentional, not "should never
happen" guards:

* `UnsupportedArchitecture("xxx")` — `general.architecture` is not
  claimed by any impl in the
  [`crate::model::architecture::registry`] (today: the BitNet
  family — `bitnet-b1.58`, `bitnet-25`, `bitnet`).
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
