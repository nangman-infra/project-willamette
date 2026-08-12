# Limitations — Project Willamette v0.12.0-mvp

*Last revised 2026-08-12 (classic Llama F16/Q4_0/Q8_0 vertical slice).*

This document is the honest counter-balance to [`README.md`](README.md).
Read this **before** treating the project as a general LLM runtime.

## 1. Scope

Project Willamette is not a general GGUF / LLM inference framework. It targets
three narrow, source-pinned model surfaces:

| Model | Quant | Tokenizer |
| ----- | ----- | --------- |
| `microsoft/BitNet-b1.58-2B-4T` (GGUF distribution `microsoft/bitnet-b1.58-2B-4T-gguf`) | `I2_S` (raw ggml_type = 36) | `gpt2` byte-level BPE with `LLAMA_VOCAB_PRE_TYPE_DEFAULT` pre-tokeniser |
| Classic Llama/Llama-2 (`shibatch/stories-converted` acceptance artifacts) | F16 transformer, embedding, and lm-head weights; F32 RMSNorm | `llama` SentencePiece BPE with byte fallback |
| `SmolLM-135M-Instruct` acceptance artifacts | F16, Q4_0, or Q8_0 transformer linears; F16/Q4_0/Q8_0 embedding and tied lm-head; F32 RMSNorm | `gpt2` BPE with `LLAMA_VOCAB_PRE_TYPE_SMOLLM` |

Anything outside this combination returns a typed error
(`UnsupportedArchitecture`, `UnsupportedTensorType`,
`UnsupportedTokenizer`, `NotImplemented`) — by design.

### Not supported

* **Architectures outside the BitNet family and classic Llama subset.** Willamette accepts the
  BitNet family (`bitnet-b1.58`, `bitnet-25`, `bitnet`) through the
  `ModelArchitecture` registry (see `src/model/architecture/`).
  `bitnet-25` was end-to-end verified on antix1 against
  [`jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF`](https://huggingface.co/jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF)
  and
  [`Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf`](https://huggingface.co/Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf).
  The bare `bitnet` string (paper-era 24/26-layer variants) is
  accepted on the assumption that its metadata prefix matches the
  arch string — that branch will be confirmed the first time such a
  GGUF is in hand. Architecture implementations now own their layer
  tensor roles and forward variant. Classic `general.architecture = "llama"`
  runs only with F16, Q4_0, or Q8_0 linears, standard unscaled full-head RoPE, RMSNorm,
  SiLU/SwiGLU, and no biases. Llama 3 scaling/tokenizer variants, Mistral,
  Phi, and Gemma remain rejected; the design path is
  [`docs/PHASE_III_ARCHITECTURE_RFC.md`](docs/PHASE_III_ARCHITECTURE_RFC.md)
  § 5.4 (Phase III-B).
* **Other BitLinear and standard quantisations.** The GGUF reader labels layouts
  such as F32, F16, Q4_0, Q4_K, Q8_0, and Q6_K, but rejects types whose byte
  layout is not implemented instead of guessing their tensor boundaries. The
  BitLinear matvec refuses to operate on anything except `BitNetI2S`. The tied
  embedding and lm-head additionally support standard Q6_K; this is not a
  general Q6_K matrix-multiplication kernel. Q8_0 matvec, embedding gather, and
  lm-head are scalar and accepted only through the implemented classic Llama
  graph surface. Q4_0 has the same scalar consumers; Q4_K/Q5 families and
  Q4_0/Q8_0 SIMD kernels remain unsupported.
* **Other tokenizer models.** `gpt2` byte BPE supports the default and `smollm`
  pre-tokenizers; classic `llama` SentencePiece BPE is also supported. Unigram,
  WordPiece, and other architecture-specific BPE normalizers remain rejected.
  SmolLM's published vocabulary intentionally omits a source-pinned set of
  control and rare byte symbols, so unlike the BitNet/default GPT-2 vocabulary
  it does not promise arbitrary Unicode byte-level roundtrips; ordinary English
  instruction prompts are covered.
* **Llama chat/TUI and BitNet tooling.** Llama F16/Q4_0/Q8_0 is enabled for `run`,
  `tokenize`, `logits`, and `perplexity`. `chat`, `tui`, `bench`, and `analyze`
  remain explicitly BitNet-only until their prompt/template and reporting
  surfaces are generalized. `run --chatml` supports one standard ChatML user
  turn by inserting `<|im_start|>` and `<|im_end|>` as special token ids; generic
  Jinja templates and multi-turn Llama chat remain unsupported.
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
| Multi-threading | `rayon` per-row BitLinear matvec parallelism (Stage 10-C). 1-thread and 4-thread transformer-forward runs on mbp2012 remain within noise because that path is memory-bandwidth bound. The tied F16 lm-head is different: parallel vocabulary rows measured 838 ms → 446 ms and improved complete steady-state throughput 0.86 → 1.29 tok/s on mbp2012. antix1 remains single-core. See `docs/BENCHMARKS.md` 2026-08-09. |
| **Q6_K tied embedding** | Supported by scalar gather, the `embedding-q6-k` artifact-linker profile, and runtime SSE2 lm-head dot products on x86/i686. The linker can relocate a transformed tensor from any physical slot and recomputes alignment, but this remains the only production transform profile. The 0.745 GiB artifact plus SSE2 reaches 0.24 tok/s on antix1 and 2.31 tok/s on mbp2012. A 1,024-transition WikiText-2 prefix measured 14.273354 perplexity versus F16 14.266282 (+0.0496%); same-host scalar/SSE2 output matches on the reference prompts. |
| **Llama Q8_0 scalar path** | Row-local 32-element/34-byte block validation, embedding gather, transformer matvec, and tied/separate lm-head projection are implemented. SmolLM-135M Q8_0 is 46.5% smaller than F16, measured 1.35x to 3.05x faster across the four validation hosts, and regressed perplexity by 0.363% on one 1,024-transition WikiText-2 prefix. No Q8_0 SIMD kernel or broader quality evaluation exists yet. |
| **Llama Q4_0 scalar path** | Row-local 32-element/18-byte validation and scalar embedding/matvec/lm-head consumers are implemented. The pinned mixed Q4_0/Q8_0 SmolLM artifact is 66.1% smaller than F16 and uses 103.9 MiB RSS on antix1. It is slower than Q8_0 on all four hosts and regresses bounded perplexity by 19.27%, so it is a low-memory option rather than the recommended default. |
| bitnet.cpp same-machine comparison on sub-AVX2 hosts | bitnet.cpp's x86 production CPU path (both the default `ggml-bitnet-mad` scalar fallback and the `BITNET_X86_TL2` LUT path) **effectively assumes AVX2**. On Ivy Bridge (no AVX2): the default build crashes with `SIGILL`, the `GGML_AVX2=OFF` build emits garbage (`!!!!!`), and the LUT build fails to compile. Willamette's hand-written SSE2 i8 kernel produces byte-identical Stage 5-E output on the same machine — see `docs/BENCHMARKS.md` 2026-05-30 § "bitnet.cpp head-to-head". The reference comparison in `docs/REFERENCE_COMPATIBILITY.md` therefore stays on AVX2-capable hosts. |
| GPU (CUDA / Metal / Vulkan / ROCm) | not implemented (out of scope by thesis). |
| Batched / multi-token-per-step decoding | the multi-token path exists for prompt prefill, but per-step decode is single-token. |
| KV cache memory | **per-token absmax i8** since v0.9.0 — 3.97× smaller than the prior f32 layout (37.7 KB/token vs 150 KB/token on BitNet 2B). 2026-05-30 long-context measurement on antix1 (800-token greedy): VmHWM growth was **0.45 KB/token** (Vec::with_capacity pre-commits + Linux page allocator's lazy commit behaviour swallow the per-token cost). The practical chat-length ceiling on antix1 is the `max_seq_len` startup argument + the model's own 4 096-position cap, not a runtime KV cliff. See [`docs/KV_CACHE_QUANT.md`](docs/KV_CACHE_QUANT.md) § "Measured long-context behaviour". Lives in normal heap memory; no swap / eviction. |

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

The Q6_K embedding is lossy and therefore does not preserve F16 logits
bit-for-bit. Its current quality gate combines byte-identical five-token greedy
output on the four Stage 5-E prompts with a 1,024-transition WikiText-2 prefix.
That prefix is enough for the narrow artifact decision, not a substitute for a
full benchmark-suite evaluation across domains and context lengths.

## 4. Error surfaces

The following errors are real and intentional, not "should never
happen" guards:

* `UnsupportedArchitecture("xxx")` — `general.architecture` is not
  claimed by any impl in the
  [`crate::model::architecture::registry`] (today: the BitNet
  family — `bitnet-b1.58`, `bitnet-25`, `bitnet` — and `llama`).
* `UnsupportedTensorType(N)` — any tensor whose raw `u32` ggml_type is
  not one of the small set we recognise. If that number is genuinely a new
  BitNet type, upgrade
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

* A small, source-pinned, auditable Rust runtime for BitNet I2_S and a narrow
  classic Llama F16/Q4_0/Q8_0 subset.
* A reproducible reference against which other implementations can
  diff their I2_S BitLinear semantics.
* An honest baseline for further BitNet-on-CPU work (Stage 8 x86,
  potential Stage 9 thread pool, etc.).

If the README claim doesn't appear in this file's "supported" column,
treat it as **not validated**. If you need a guarantee for any of the
gaps above, please file an issue with the exact use case before
relying on the code.
