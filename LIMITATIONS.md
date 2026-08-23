# Limitations — Project Willamette v0.15.0-mvp

*Last revised 2026-08-23 (Qwen batched prefill and quality evaluation).*

This document is the honest counter-balance to [`README.md`](README.md).
Read this **before** treating the project as a general LLM runtime.

## 1. Scope

Project Willamette is not a general GGUF / LLM inference framework. It targets
the following narrow, source-pinned model surfaces:

| Model | Quant | Tokenizer |
| ----- | ----- | --------- |
| `microsoft/BitNet-b1.58-2B-4T` (GGUF distribution `microsoft/bitnet-b1.58-2B-4T-gguf`) | `I2_S` (raw ggml_type = 36) | `gpt2` byte-level BPE with `LLAMA_VOCAB_PRE_TYPE_DEFAULT` pre-tokeniser |
| Classic Llama/Llama-2 (`shibatch/stories-converted` acceptance artifacts) | F16 transformer, embedding, and lm-head weights; F32 RMSNorm | `llama` SentencePiece BPE with byte fallback |
| `SmolLM-135M-Instruct` acceptance artifacts | F16, Q4_0, or Q8_0 transformer linears; F16/Q4_0/Q8_0 embedding and tied lm-head; F32 RMSNorm | `gpt2` BPE with `LLAMA_VOCAB_PRE_TYPE_SMOLLM` |
| `SmolLM2-360M-Instruct` acceptance artifact | Q8_0 transformer linears, embedding, and tied lm-head; F32 RMSNorm | `gpt2` BPE with `LLAMA_VOCAB_PRE_TYPE_SMOLLM` and incremental multi-turn ChatML encoding |
| `SmolLM2-1.7B-Instruct` acceptance artifact | Mixed Q4_K/Q6_K transformer linears, embedding, and tied lm-head; F32 RMSNorm | `gpt2` BPE with `LLAMA_VOCAB_PRE_TYPE_SMOLLM` and incremental multi-turn ChatML encoding |
| `Qwen2.5-3B-Instruct` acceptance artifact | Mixed Q4_K/Q6_K transformer linears, tied lm-head, F32 RMSNorm and Q/K/V biases | `gpt2` BPE with `LLAMA_VOCAB_PRE_TYPE_QWEN2` and incremental ChatML encoding |

Anything outside this combination returns a typed error
(`UnsupportedArchitecture`, `UnsupportedTensorType`,
`UnsupportedTokenizer`, `NotImplemented`) — by design.

### Not supported

* **Architectures outside the BitNet family, classic Llama subset, and pinned Qwen2 subset.** Willamette accepts the
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
   runs only with F16, Q4_0, Q4_K, Q6_K, or Q8_0 linears, standard unscaled full-head RoPE, RMSNorm,
  SiLU/SwiGLU, and no biases. Llama 3 scaling/tokenizer variants, Mistral,
  Phi, and Gemma remain rejected; the design path is
  [`docs/PHASE_III_ARCHITECTURE_RFC.md`](docs/PHASE_III_ARCHITECTURE_RFC.md)
  § 5.4 (Phase III-B).
* **Other BitLinear and standard quantisations.** The GGUF reader rejects types
   whose byte layout is not implemented instead of guessing tensor boundaries.
   BitLinear matvec refuses to operate on anything except `BitNetI2S`. Classic
   Llama embedding, transformer, and lm-head consumers support Q4_K and Q6_K,
   including mixed Q4_K_M artifacts. Q8_0 matvec and lm-head row dots
   dispatch to NEON, AVX2, or SSE2 where available, with scalar fallback;
   Q4_K row dots dispatch to AVX2 or SSE2 on x86 with scalar fallback. Embedding
   gather and Q4_0 consumers remain scalar. Q5_K, IQ families, Q4_K NEON, and
   Q4_0 SIMD kernels remain unsupported.
* **Other tokenizer models.** `gpt2` byte BPE supports the default and `smollm`
  pre-tokenizers; classic `llama` SentencePiece BPE is also supported. Unigram,
  WordPiece, and other architecture-specific BPE normalizers remain rejected.
  SmolLM's published vocabulary intentionally omits a source-pinned set of
  control and rare byte symbols, so unlike the BitNet/default GPT-2 vocabulary
  it does not promise arbitrary Unicode byte-level roundtrips; ordinary English
  instruction prompts are covered.
* **Llama chat/TUI and BitNet tooling.** Llama F16/Q4_0/Q4_K/Q6_K/Q8_0 is enabled for
  `run`, `tokenize`, `logits`, `perplexity`, and machine-readable
  `bench --format json`. `chat` and `tui` support Llama-family instruct models
  whose vocabulary provides the standard ChatML `<|im_start|>` and
  `<|im_end|>` markers; classic TinyStories completion models therefore remain
  unsuitable for those interactive commands. The human `bench` report and
  `analyze` remain BitNet-only. Generic Jinja templates and other chat formats
  are unsupported.
* **Other pre-tokeniser hints.** If a future GGUF arrives with
  `tokenizer.ggml.pre = "llama-bpe"` (instead of the missing-key
  default our reference file has), the LLaMA 3 regex set in
  `llama-vocab.cpp:373..381` is NOT yet implemented in Willamette —
  we use only the DEFAULT 3-regex set.

## 2. Performance

The performance numbers in [`README.md`](README.md) and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
are host-specific measurements across Apple M4, antix1, mbp2012, and HP
ProBook 430 G6, depending on the section. They are not portable promises;
each table's host, thread count, model, and metric boundary are authoritative.

| Concern | Status |
| ------- | ------ |
| Apple Silicon NEON | implemented, validated against scalar (Stage 6-C). Measured on Apple M4. |
| **x86 / i686 SSE2 (int8 activation)** | **default since v0.5.0 / v0.7.0** — validated on antiX Pentium-M 2 GHz. 2.2× over the f32 SSE2 path, ~5.4× over scalar. Byte-identical greedy output to f32 on the real 2B model. |
| x86 / i686 SSE2 (f32 mask-add) | kept as the numerical reference behind `--cfg willamette_sse2_f32`. |
| **BitLinear x86 AVX2 / AVX-512** | not yet implemented. This does not apply to classic Llama Q8_0, whose AVX2 row dot is validated on HP ProBook. |
| **LUT kernel (scalar)** | **shipped in v0.10.0-mvp as the default on SSE2-without-SSSE3 x86 hosts** via `Kernel::X86Sse2ScalarLut`. Stage 5-E greedy is byte-identical across NEON / SSE2 i8 / scalar LUT paths. The historical 0.41 → 0.40 cached-forward result was within noise. Its original explanation that matvec was only ~10% of the step was wrong: later stage instrumentation found all seven BitLinear matvecs consume about 99.4% of antix1 cached-forward time. That percentage excludes lm-head and argmax. The original single-`attn_q` 5.29× comparison was not representative enough to predict complete-token performance. |
| **Sparsity-aware skipping** | prototype shipped (`src/model/bitlinear_sparse.rs`), benched, but on antix1 it ties with dense i8 (1.01× slower — irregular access cancels the 42% skip). Documented; not default. Likely a win on sub-SSE2 hardware. |
| Apple Silicon with `+dotprod` / FEAT_DotProd | hardware present on the M4 dev host; the stable-Rust `vdotq_s32` intrinsic remains unused (kernel keeps `vmull_s8`-style widening for parity). Switching is an `RUSTFLAGS="--cfg willamette_i8_activations"` flag away. |
| Apple Silicon with FEAT_I8MM / SME / SME2 | hardware present on M4; intrinsics not in stable Rust → unused. |
| Multi-threading | `rayon` per-row BitLinear matvec parallelism (Stage 10-C). 1-thread and 4-thread transformer-forward runs on mbp2012 remain within noise because that path is memory-bandwidth bound. The tied F16 lm-head is different: parallel vocabulary rows measured 838 ms → 446 ms and improved complete steady-state throughput 0.86 → 1.29 tok/s on mbp2012. antix1 remains single-core. See `docs/BENCHMARKS.md` 2026-08-09. |
| **Q6_K tied embedding** | Supported by scalar gather, the `embedding-q6-k` artifact-linker profile, and runtime SSE2 lm-head dot products on x86/i686. The linker can relocate a transformed tensor from any physical slot and recomputes alignment, but this remains the only production transform profile. The 0.745 GiB artifact plus SSE2 reaches 0.24 tok/s on antix1 and 2.31 tok/s on mbp2012. A 1,024-transition WikiText-2 prefix measured 14.273354 perplexity versus F16 14.266282 (+0.0496%); same-host scalar/SSE2 output matches on the reference prompts. |
| **Llama Q8_0 path** | Row-local 32-element/34-byte block validation, scalar embedding gather, and SIMD-dispatched transformer/lm-head row dots are implemented. NEON, AVX2, and SSE2 improve complete-token profiles by 2.10-3.04x over the same-host scalar control across M4, HP ProBook, mbp2012, and antix1. The pinned greedy golden and SIMD-vs-scalar multi-block tolerance pass; broader quality evaluation remains limited to the documented prompts and one 1,024-transition WikiText-2 prefix (+0.363% perplexity versus F16). |
| **Llama Q4_0 scalar path** | Row-local 32-element/18-byte validation and scalar embedding/matvec/lm-head consumers are implemented. The pinned mixed Q4_0/Q8_0 SmolLM artifact is 66.1% smaller than F16 and uses 103.9 MiB RSS on antix1. It is slower than Q8_0 on all four hosts and regresses bounded perplexity by 19.27%, so it is a low-memory option rather than the recommended default. |
| **Llama Q4_K_M path** | Canonical 256-element/144-byte Q4_K and 256-element/210-byte Q6_K rows are validated and supported across embedding, transformer linears, and tied lm-head. Q4_K uses AVX2/SSE2 on x86; Q6_K uses SSE2. The pinned 1.7B Paris and two-turn goldens pass. HP Korean comparison showed a response truncated at the common 50-token cap; broader quality evaluation remains pending. |
| bitnet.cpp same-machine comparison on sub-AVX2 hosts | bitnet.cpp's x86 production CPU path (both the default `ggml-bitnet-mad` scalar fallback and the `BITNET_X86_TL2` LUT path) **effectively assumes AVX2**. On Ivy Bridge (no AVX2): the default build crashes with `SIGILL`, the `GGML_AVX2=OFF` build emits garbage (`!!!!!`), and the LUT build fails to compile. Willamette's hand-written SSE2 i8 kernel produces byte-identical Stage 5-E output on the same machine — see `docs/BENCHMARKS.md` 2026-05-30 § "bitnet.cpp head-to-head". The reference comparison in `docs/REFERENCE_COMPATIBILITY.md` therefore stays on AVX2-capable hosts. |
| GPU (CUDA / Metal / Vulkan / ROCm) | not implemented (out of scope by thesis). |
| Batched / multi-token-per-step decoding | Prompt prefill is layer-major and batches Q4_K projections with tiled AVX2/SSE2 row dots. Per-step decode remains single-token. |
| ARMv7 acceptance | `armv7-unknown-linux-musleabihf` remains a release build target and compiles in CI. No physical ARMv7 device was available for this release, so runtime, memory, and instruction-dispatch acceptance remain unverified on hardware. |
| KV cache memory | **per-token absmax i8** since v0.9.0 — 3.97× smaller than the prior f32 layout (37.7 KB/token vs 150 KB/token on BitNet 2B). 2026-05-30 long-context measurement on antix1 (800-token greedy): VmHWM growth was **0.45 KB/token** (Vec::with_capacity pre-commits + Linux page allocator's lazy commit behaviour swallow the per-token cost). The practical chat-length ceiling on antix1 is the `max_seq_len` startup argument + the model's own 4 096-position cap, not a runtime KV cliff. See [`docs/KV_CACHE_QUANT.md`](docs/KV_CACHE_QUANT.md) § "Measured long-context behaviour". Lives in normal heap memory; no swap / eviction. |

For BitLinear I2_S on x86/x86_64, Willamette uses SSE2 i8 on SSSE3+ hosts and
the scalar LUT on SSE2-only hosts. Q8_0 instead uses AVX2 when available and
SSE2 otherwise; Q4_K uses the same AVX2-then-SSE2 selection. CPUs without a compiled and detected optimized path use the
generic scalar reference; throughput depends on model, host, and metric
boundary.

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

Q8_0 SIMD changes floating-point reduction order and is therefore not promised
to be bit-identical to the scalar row dot. Multi-block unit coverage enforces a
bounded absolute delta, while the pinned 135M arithmetic and 360M Paris goldens
guard selected output-level behavior. The longer 360M sky prompt diverged
between scalar and SIMD at its first generated token and near the end between
AVX2 and SSE2. On M4, scalar and NEON retain the same two leading first-step
candidates but reverse their order; the external-model diagnostic gates that
bounded candidate/logit/margin envelope. Greedy parity is therefore a
prompt-specific acceptance gate, not a universal cross-kernel promise.

The Qwen quality profile is not a general structured-data guarantee. The pinned
six-field report, one-sentence summary, and four-turn recall pass, but the
expanded suite also records a strict line-count failure and a missing-field
detection failure. See [`GOLDEN_TESTS.md`](GOLDEN_TESTS.md).

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
   classic Llama F16/Q4_0/Q4_K/Q6_K/Q8_0 subset.
* A reproducible reference against which other implementations can
  diff their I2_S BitLinear semantics.
* An honest baseline for further BitNet-on-CPU work (Stage 8 x86,
  potential Stage 9 thread pool, etc.).

If the README claim doesn't appear in this file's "supported" column,
treat it as **not validated**. If you need a guarantee for any of the
gaps above, please file an issue with the exact use case before
relying on the code.
