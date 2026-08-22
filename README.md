# Project Willamette

**Thesis:** mid-sized publicly-released LLMs run on **CPU-only
humble hardware** — older laptops, low-RAM thin clients, retro x86,
Raspberry-Pi-class ARM — without a GPU. The proof is two binaries:
an offline **`willamette-prep`** that bakes a model down to a
hardware-aware form, and an online **`willamette`** runtime that
just executes the baked form. The runtime is Rust, uses zero-copy
`mmap`, and ships Linux `x86_64`, `i686`, `aarch64`, and `armv7`
plus macOS `aarch64` and `x86_64` builds. Real-hardware inference is
validated on Apple aarch64 and Linux x86_64/i686; Linux ARM remains
build-supported but not yet benchmarked on-device.

> **Sweet spot is hardware-dependent.** On Pentium-M-class SSE2
> hardware (the verified floor at 2026-05-27) the measured ceiling
> is roughly **100 M params at ≥ 5 cached-forward steps/s**, **500 M
> for "slow but usable" (≥ 1 tok/s)**, **5 B for demonstration
> (≥ 0.1 tok/s)**. These historical thresholds exclude lm-head and
> token-selection cost, so they are not complete-generation guarantees.
> Full scaling table and the EXO Pentium-II
> comparison: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

Starting point: [microsoft/BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
in its `ggml-model-i2_s.gguf` form (1.58-bit ternary weights) — the
first model proven end-to-end. A classic Llama/Llama-2 F16 vertical slice is
also working against pinned TinyStories GGUFs. Destination: a runtime
that, given any preprocessed mid-sized GGUF, runs it on the same
humble-hardware envelope. **BitNet is how the runtime got proven;
it is not the only model we will ever support.**

Engineering rules every change is held to (full list in
[§ Project rules](#project-rules-carried-forward-to-every-contribution)):

* **No fake weights, no fake logits, no synthetic inference paths.**
* **Zero-copy mmap** — packed weights stay in their on-disk blocks.
* **Source-pinned semantics** — every layout / dtype constant cites a
  pinned upstream commit (see [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md)).
* **No unverified SIMD merges** — runtime feature detection only; no
  silent `target-cpu=native`.

## Two-piece architecture

```text
┌─ heavy / one-time, beefy machine ──┐         ┌─ light / per-inference, humble machine ──┐
│                                    │         │                                          │
│   public model (HF, GGUF, etc.)    │         │   willamette-prep'd model artifact       │
│            │                       │         │            │                             │
│            ▼                       │         │            ▼                             │
│   willamette-prep                  │ ──────▶ │   willamette  (this binary, today)       │
│   ── analyze activations           │         │   ── mmap, run, chat                     │
│   ── quantise + re-layout          │         │   ── CPU only, no model conversion       │
│   ── windowing / sparse tables     │         │                                          │
│   ── target-ISA aware blocking     │         │                                          │
└────────────────────────────────────┘         └──────────────────────────────────────────┘
 NARROW Q6_K PATH WORKING TODAY                         WORKING ON MAIN
```

The split is the same pattern TensorFlow Lite / Core ML / ONNX
Runtime / `bitnet.cpp`'s `quantize` use: the expensive once-per-model
work runs where compute is cheap, and the on-device runtime stays
small. `willamette-prep` now has a format-correct GGUF linker and one production
profile: it plans every aligned tensor offset, converts the tied F16 embedding
to Q6_K, and preserves every transformer I2_S slot. Additional transform
profiles, architecture conversion, and target-aware blocking remain roadmap
work.

## Status: v0.13.0-mvp

What works **today**, on the path toward the thesis:

| Property | Value |
| -------- | ----- |
| Working reference model | `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` (1.1 GiB ternary) |
| Model SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| BitNet-family fine-tunes accepted | ✅ `bitnet-b1.58`, `bitnet-25`, `bitnet` GGUF strings load through `model::architecture::registry`. End-to-end greedy decode verified on antix1 against [`jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF`](https://huggingface.co/jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF) (French) and [`Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf`](https://huggingface.co/Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf) (Solana coding). See [`docs/PHASE_III_ARCHITECTURE_RFC.md`](docs/PHASE_III_ARCHITECTURE_RFC.md). |
| Classic Llama F16 | ✅ unscaled full-head RoPE, GQA/MHA, F16 linears, SiLU/SwiGLU, separate or tied F16 lm-head, and `llama` SentencePiece BPE. Pinned 260K and 15M TinyStories prompt IDs and greedy outputs match llama.cpp exactly; the 260K golden also passes on antix1 i686/Pentium-M. |
| SmolLM-135M-Instruct F16 | ✅ `smollm` GPT-2 pre-tokenizer, 30-layer GQA graph, tied lm-head, plain completion prompts, and single-turn `run --chatml`. Pinned tokenizer IDs and `2 + 2 → 4` greedy output match llama.cpp; antix1 reaches about 0.765 steady-state tok/s at 274.7 MiB peak RSS. |
| SmolLM-135M-Instruct Q8_0 | ✅ Q8_0 embedding, transformer linears, and tied lm-head with runtime NEON / AVX2 / SSE2 row-dot dispatch and scalar fallback. The pinned arithmetic golden matches llama.cpp. SIMD improves measured complete-token throughput by 2.10-3.04x across four hosts; current 120-token greedy IDs also match across HP ProBook, mbp2012, and antix1. Its first 1,024 WikiText-2 transitions regress perplexity by 0.363% versus F16. |
| SmolLM-135M-Instruct Q4_0 | ⚠️ supported low-memory scalar path, but not recommended as the default. The pinned mixed artifact uses Q4_0 transformer linears and a Q8_0 tied embedding/lm-head. It is 87.5 MiB and reaches 0.873 tok/s at 103.9 MiB peak RSS on antix1, but is slower than Q8_0 on all four hosts and regresses 1,024-transition perplexity by 19.27% versus F16. |
| SmolLM2-360M-Instruct Q8_0 | ✅ official 386,404,992-byte artifact loads unchanged. Explicit ChatML system/user prompt IDs and the greedy `The capital of France is Paris.` output match llama.cpp exactly. The same 120 sampled token IDs were reproduced on Apple M4, HP ProBook 430 G6, mbp2012, and the 996 MiB antix1 host. The recorded 16.77 / 5.72 / 2.93 / 0.254 tok/s product-path baseline predates the Q8_0 SIMD kernel and is retained for provenance. |
| Reference parity (bitnet.cpp) | ✅ byte-identical generated text on Stage 5-E prompts |
| Reference build | `microsoft/BitNet @ 01eb4157…` (see [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md)) |
| Apple Silicon NEON kernel | ✅ implemented + validated (Apple M4 dev host) |
| **x86 SSE2 kernels** | ✅ SSE2 i8 is selected on SSSE3+ x86; the pure-Rust scalar LUT is selected on SSE2-only x86 such as antix1. Both preserve verified greedy output. |
| **Q8_0 SIMD kernels** | ✅ NEON on Apple M4, AVX2 on HP ProBook, and SSE2 on mbp2012/antix1; all measured on-device against the scalar control. |
| Runtime CPU dispatch | ✅ BitLinear selects NEON / SSE2-i8 / SSE2-only scalar LUT / scalar; Q8_0 independently selects NEON / AVX2 / SSE2 / scalar. |
| **Prebuilt static binaries** | ✅ the release workflow builds 6 targets — `x86_64`, `i686`, `aarch64`, `armv7` Linux musl + `aarch64`, `x86_64` macOS. See [Releases](https://github.com/nangman-infra/project-willamette/releases). |
| Multi-core CPU parallelism | ✅ `rayon` per-row BitLinear matvec |
| Norm-weight + scratch caching | ✅ Stage 10-A / 10-B |
| **KV cache i8 quantisation** | ✅ **per-token absmax i8 since v0.9.0** — ~3.97× memory shrink (150 KB → 37.7 KB per token on BitNet 2B). Greedy output byte-identical to the f32 reference on Stage 5-E prompts (Apple M4 NEON + antix1 i686 SSE2). See [`docs/KV_CACHE_QUANT.md`](docs/KV_CACHE_QUANT.md). |
| Chat + TUI surfaces | ✅ `willamette chat` (stdio) + `willamette tui` (ratatui full-screen), using the BitNet text bridge or incremental SmolLM ChatML |
| Synthetic GGUF builder | ✅ `willamette synth-gguf --preset {tiny\|small\|medium}` (humble-HW throughput benchmarks) |
| Ternary weight distribution | ✅ `willamette analyze` (-1 / 0 / +1 fractions across BitLinear tensors) |
| `willamette-prep` artifact linker / Q6_K tied embedding | ✅ explicit plan + full GGUF relocation + dry-run; standalone prep binary and compatible runtime subcommand produce the same 0.745 GiB artifact; scalar gather + runtime SSE2 lm-head on x86 |
| Architecture graph seam | ✅ registered families declare layer tensor roles and a forward variant; BitNet and classic Llama F16/Q4_0/Q8_0 are implemented |
| All-in-one launcher | ✅ `scripts/willamette` (SHA verify + HF download + build + run) |
| Tests | Default and explicitly ignored external-model suites pass on the documented validation hosts; exact counts vary by target architecture. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). |
| SonarQube Quality Gate | ✅ OK across the v0.x release cycle |
| Historical `llama2.c` comparison | ⚠️ the 110M antix1 comparison used generated tok/s for `llama2.c` but cached-forward-only tok/s for Willamette. It is retained as forward-path evidence, not a complete-generation speedup claim. |

What does **not** work yet but is on the roadmap toward the thesis:

| Property | Value |
| -------- | ----- |
| Model coverage beyond the implemented classic Llama F16/Q4_0/Q8_0 subset (Llama 3 / Mistral / Phi / Gemma) | ❌ scaled/partial RoPE, other linear formats, architecture-specific biases, and additional tokenizer families remain unsupported — see [`docs/PHASE_III_ARCHITECTURE_RFC.md`](docs/PHASE_III_ARCHITECTURE_RFC.md) |
| Standard GGUF quant types beyond the narrow Llama Q4_0/Q8_0 slice (Q4_K, Q5_K, …) | ❌ BitLinear remains `I2_S`-only; tied embedding additionally supports Q6_K |
| Additional `willamette-prep` transform profiles beyond the tied Q6_K embedding | ❌ architecture conversion, activation analysis, sparse tables, and target-ISA blocking not started; generic GGUF relocation mechanics are implemented |
| BitLinear I2_S AVX2 / AVX-512 kernel | ❌ not started; the new Q8_0 AVX2 row-dot kernel is a separate, shipped path validated on HP ProBook. |
| Additional LUT kernels | ⚠️ the project-specific scalar LUT shipped in v0.10.0-mvp for SSE2-only x86. SSSE3 `pshufb`, upstream-compatible TL1/TL2, NEON, and AVX LUT variants remain unimplemented; see [`docs/LUT_KERNEL_RFC.md`](docs/LUT_KERNEL_RFC.md). |
| MMX-era / sub-SSE2 kernel | ❌ not started |
| KV cache int8 quantisation | ✅ landed in v0.9.0 (see Status table above) |
| LLM-in-a-Flash style mmap windowing | ❌ |
| Emulator-based humble-hardware benchmark pipeline (QEMU / 86Box) | ❌ |
| Generic scalar fallback on build-supported targets | ✅ correctness-only; real-hardware inference is not yet documented for every target |
| GPU | ⛔ explicitly out of scope by thesis (CPU only) |

## Quick start

You have **two install paths** — picking the lighter one matters on
humble hardware:

### Option A — Prebuilt static binary (recommended for low-end hosts)

No toolchain, no compile time. Pick the tarball matching your host:

```bash
TAG=v0.13.0-mvp
TARGET=i686-unknown-linux-musl   # also: x86_64-unknown-linux-musl,
                                 #       aarch64-unknown-linux-musl,
                                 #       armv7-unknown-linux-musleabihf,
                                 #       aarch64-apple-darwin,
                                 #       x86_64-apple-darwin
curl -LO https://github.com/nangman-infra/project-willamette/releases/download/$TAG/willamette-$TAG-$TARGET.tar.gz
curl -LO https://github.com/nangman-infra/project-willamette/releases/download/$TAG/willamette-$TAG-$TARGET.tar.gz.sha256
sha256sum -c willamette-$TAG-$TARGET.tar.gz.sha256
tar -xzf willamette-$TAG-$TARGET.tar.gz
./willamette-$TAG-$TARGET/willamette --version
```

The Linux binaries are **musl-static** (no glibc dependency) — the
same artifact runs on antiX Pentium-M (glibc 2.36), Raspberry Pi OS,
and modern Ubuntu. i686 build is ≈ **2.5 MB** stripped.

### Option B — Build from source

* Rust 1.94 (`rust-toolchain.toml` pins this).
* macOS / Linux on aarch64 or x86_64 / i686. Apple Silicon gets the
  NEON path; x86 / i686 gets the **SSE2 int8 kernel by default**
  (validated on antiX Pentium-M, see
  [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

### 2. Download the model

We do **not** ship the GGUF in this repo (1.1 GiB and not ours to
redistribute). Use the official Hugging Face CLI:

```bash
hf download microsoft/bitnet-b1.58-2B-4T-gguf \
    ggml-model-i2_s.gguf \
    --local-dir ./models/bitnet-b1.58-2B-4T-gguf
```

Verify the file integrity:

```bash
shasum -a 256 ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
# expected:
# 4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162  ...
```

If the SHA256 differs, the file is corrupt or a different revision —
the layout pins documented in [`docs/I2_S_LAYOUT.md`](docs/I2_S_LAYOUT.md)
are only guaranteed against this one byte stream.

For memory-constrained hosts, derive the pinned Q6_K-embedding artifact without
changing any transformer I2_S tensor:

```bash
cargo run --release --bin willamette-prep -- \
  --model ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --output ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s-embed-q6_k.gguf \
  --profile embedding-q6-k
shasum -a 256 ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s-embed-q6_k.gguf
# 492e4d2a8db2eefc5f8c86acd08eea6707294de67ce871b5d732e9bfcb468376
```

The derived file is 800,468,160 bytes (0.745 GiB). Existing output paths are
never overwritten. Release tarballs include `willamette-prep`; source builds
also retain `cargo run --release -- repack-embedding-q6k -- ...` as a
byte-identical compatibility interface.

Inspect the validated architecture, tensor change, sizes, and relocated offsets
without creating a file:

```bash
cargo run --release --bin willamette-prep -- \
  --model SOURCE.gguf --output DEST.gguf \
  --profile embedding-q6-k --dry-run
```

### 3. Build

```bash
cargo build --release
```

The release profile uses `lto = "fat"`, `panic = "abort"`, `strip = true`
and runtime feature detection (NEON on aarch64). No `target-cpu=native`
default — produced binaries work on any aarch64 / x86_64 of the same
generation as the build host.

### 4. Smoke test

```bash
./target/release/project-willamette run \
    --model ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --prompt "The capital of France is" \
    --max-new-tokens 3
```

Expected output (Apple M-series, < 3 s on M4). On antix1, complete
generation is materially slower than the historical 8 s cached-forward
figure; see the complete-token measurements in `docs/BENCHMARKS.md`:

```
Generating:  Paris. Paris
Generated 3 token(s): [12366, 13, 12366]
Generated text:   " Paris. Paris"
Full text:        "The capital of France is Paris. Paris"
```

For a model whose GGUF vocabulary uses standard ChatML markers, encode one
user turn with special token ids rather than passing the marker strings through
BPE:

```bash
willamette run --model SmolLM-135M-Instruct-Q8_0.gguf \
  --prompt "Explain why the sky looks blue." --chatml \
  --max-new-tokens 120 --temperature 0.7 --top-k 40 --top-p 0.9
```

`run` reports end-to-end inference wall time and generated tokens per second,
including prompt prefill. `--chatml` is intentionally single-turn; generic GGUF
Jinja templates remain unsupported.

## CLI subcommands

```text
willamette-prep --model PATH --output PATH
                 [--profile embedding-q6-k] [--dry-run]

willamette inspect    --model PATH
willamette repack-embedding-q6k --model PATH --output PATH
willamette analyze    --model PATH
willamette tokenize   --model PATH --text TEXT [--no-bos] [--add-eos]
willamette logits     --model PATH --prompt TEXT [--top-k N] [--no-bos]
willamette perplexity --model PATH --file UTF8_PATH [--max-tokens N] [--no-bos]
willamette run        --model PATH --prompt TEXT
                      [--max-new-tokens N]
                      [--no-bos] [--chatml] [--system TEXT]
                      [--temperature F] [--top-k K] [--top-p P]
                      [--repetition-penalty R] [--seed S]
                      [--stop-id ID]...
willamette bench      --model PATH [--decode-steps N] [--format human|json]
willamette chat       --model PATH [--max-seq-len N] [--max-new-tokens N]
                      [--system TEXT]
                      [--temperature F] [--top-k K] [--top-p P]
                      [--repetition-penalty R] [--seed S]
willamette tui        --model PATH [--max-seq-len N] [--max-new-tokens N]
                      [--system TEXT]
                      [--temperature F] [--top-k K] [--top-p P]
                      [--repetition-penalty R] [--seed S]
willamette synth-gguf --output PATH --preset {tiny|small|medium}
willamette --version
```

* `inspect` — Stage 1. Dumps every metadata key + every tensor's raw
  ggml_type, shape, offset, and byte length. No inference.
* `willamette-prep` / `repack-embedding-q6k` — The standalone offline linker
  validates a profile, computes every aligned output tensor offset, patches the
  GGUF directory, and streams copied or transformed physical tensor slots. The
  default `embedding-q6-k` profile converts the tied F16 embedding through the
  standard Q6_K reference quantiser. `--dry-run` prints the plan without an
  output file. The runtime compatibility command produces identical bytes; all
  transformer tensors remain byte-identical I2_S.
* `perplexity` — Scores contiguous next-token transitions with the cached
  autoregressive path and stable f64 log-sum-exp. Defaults to 256 transitions
  and refuses to exceed the model context.
* `analyze` — Counts -1 / 0 / +1 across every BitLinear (I2_S) tensor.
  Reports the zero fraction (the upper bound on what sparsity-aware
  skipping could save). Real 2B: 28.9 / 42.2 / 28.9 %.
* `tokenize` — Stage 2. Runs GGUF-bundled GPT-2 byte BPE with the default or
  SmolLM pre-tokenizer, or classic Llama SentencePiece BPE. Refuses to run on
  tokenizer models and pre-tokenizers we do not support.
* `logits` — Stage 4-D5. Runs the full 30-layer forward and prints the
  top-K next-token logits. Useful for comparing against bitnet.cpp.
* `run` — Stage 5. Real BitLinear forward + greedy or sampled
  generation, with KV cache.
* `bench` — Times one matvec, one no-cache forward, cached transformer
  forward, lm-head projection, argmax, and their complete steady-state token
  total. `--format json` supports BitNet and classic Llama F16/Q4_0/Q8_0 and
  reports the measured linear backend. The default human report remains
  BitNet-specific and also compares the sparse prototype against dense
  `attn_q`.
* `chat` — Stage 9. Multi-turn stdio chat (line-based) for BitNet and
  ChatML-compatible SmolLM models. `/quit`,
  `/reset`, `/sys [text|off]`, `/history`, `/save <file>`.
* `tui` — Stage 9-E. Full-screen ratatui chat — left chat pane + right
  live dashboard (per-core CPU %, KV cache size, **tok/s**, current
  layer, RSS, sampling params, active SIMD kernel). Keys: type+Enter,
  ↑↓ history, Ctrl-R reverse search, Ctrl-L clear log, Ctrl-Y yank
  last bot reply (OSC52), Esc cancel mid-generation, F1 help,
  `/quit`. Needs a terminal ≥ 72 columns for the 2-column layout.
* `synth-gguf` — Builds a synthetic BitNet b1.58 GGUF (random ternary
  weights) for throughput benchmarking on humble hosts. `tiny`
  ≈ 73 KB, `small` ≈ 10 M params, `medium` ≈ 110 M params (same scale
  class as TinyLlama). No tokenizer included → `inspect` and `bench`
  work, `run` / `chat` / `tui` will reject the file (random weights →
  garbage tokens — see [[feedback-no-fake]]).

### Running the TUI

```bash
./willamette tui --model ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
scripts/willamette --profile smollm-135m tui
scripts/willamette --profile smollm2-360m tui
```

The launcher verifies each profile's pinned SHA256 before starting. On the
Pentium-M antiX demo host, `scripts/demo_antix.sh` presents the 360M quality
demo first, the faster but limited 135M comparison, the historical BitNet TUI,
the Paris golden, and llama2.c side by side.

Needs a real terminal (not the Claude-Code embedded chat). Over SSH
use `ssh -t` to force a pseudo-tty when launching one-shot:

```bash
ssh -t user@host '~/bin/willamette tui --model ~/models/ggml-model-i2_s.gguf'
```

Expect very slow generation on humble HW. The historical antix1
`~0.4 tok/s` figure measured cached transformer forward only; complete
steady-state generation measured about 0.08 tok/s with the original F16
tied embedding and 0.24 tok/s with the prepared Q6_K embedding plus SSE2
lm-head. Use **Esc** to cancel a long answer.

## Performance

Historical cached-forward-only numbers (real BitNet 2B model, `cargo
--release`). They exclude lm-head projection and token selection. Full
table including complete-token measurements and the synthetic 110M / 7M points,
EXO Pentium-II comparison, and llama2.c head-to-head live in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| Host | Historical kernel | cached-forward tok/s |
| --- | --- | ---: |
| **Apple M4** (Mac16,10, dev box) | aarch64 NEON | **7.9** |
| **mbp2012** Mid-2012 MBP Ivy Bridge i7-3520M (sub-AVX2 host) | x86_64 SSE2 (i8) | **2.65** |
| **antiX Pentium-M 2 GHz** (humble validation host) | i686 SSE2 (i8) | **0.41** |
| antiX Pentium-M 2 GHz | i686 scalar (v0.4.1) | 0.05 |

Historical same-hardware, same-model cached-forward progression:

| antiX Pentium-M progression | cached-forward tok/s | speed-up |
| --- | ---: | ---: |
| scalar reference | 0.05 | — |
| SSE2 f32 mask-add (v0.4.x f32 path) | 0.19 | 2.49× over scalar |
| **SSE2 i8 (v0.5.0+ default)** | **0.41** | **2.15× over f32 / 5.4× over scalar** |

Historical same-machine comparison vs `llama2.c` at 110M scale. The
metric boundary differs: `llama2.c` reports generation while the cited
Willamette value excludes lm-head and token selection, so this is not a
complete-generation speedup claim:

| Build | tok/s |
| --- | ---: |
| `llama2.c` `stories110M` (vanilla f32) | 2.51 |
| `willamette` synth 110M (BitNet b1.58 + SSE2 i8) | **4.96 (1.97× faster)** |

Current SmolLM-135M Q8_0 complete-token profiles, measured with the same
stage-instrumented binary and a scalar control on each host:

| Host | Q8_0 SIMD | Scalar | SIMD | Speed-up |
| --- | --- | ---: | ---: | ---: |
| Apple M4 | NEON | 24.92 ms | 8.19 ms | **3.04x** |
| HP ProBook 430 G6 | AVX2 | 52.08 ms | 22.08 ms | **2.36x** |
| mbp2012 | SSE2 | 82.91 ms | 39.47 ms | **2.10x** |
| antix1 | SSE2 | 967.51 ms | 399.22 ms | **2.42x** |

These are steady-state complete-token profiles, not prompt-inclusive `run`
throughput. Full stage boundaries and caveats are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

For SmolLM-family instruction use, the recommendation remains quality and
latency dependent. The three Linux laptop rows use the post-SIMD greedy rerun;
the M4 row retains the earlier sampled quality sweep until it is rerun:

| Host | Recommended model | Reason |
| --- | --- | --- |
| Apple M4 | SmolLM2-360M-Instruct Q8_0 | Earlier sampled quality sweep: 16.77 tok/s. The larger model gave the materially better answer; post-SIMD product-path rerun is pending. |
| mbp2012 | SmolLM2-360M-Instruct Q8_0 | Post-SIMD greedy rerun: 7.04 tok/s and 405 MiB peak RSS. |
| antix1 | SmolLM-135M-Instruct Q8_0 | Post-SIMD greedy rerun: 1.87 tok/s versus 0.698 tok/s for 360M. Use 360M for quality-first unattended generation. |
| HP ProBook 430 G6 | SmolLM2-360M-Instruct Q8_0 | Post-SIMD greedy rerun: 15.44 tok/s and 438 MiB peak RSS. |

These are one-run, prompt-inclusive measurements rather than the short
steady-state decode figures above. The M4 row used sampled generation while
the three post-SIMD laptop rows use greedy generation, so compare within each
documented sweep rather than ranking those rows directly. Full prompts and
caveats are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

The runtime is "correctness + memory floor + portability floor" first
— `llama.cpp` will likely win raw speed on modern x86. We win the
**lowest hardware floor a real medium LLM can be run on**.

## Reference compatibility (bitnet.cpp)

We verify Willamette against the pinned `microsoft/BitNet` build on
the four reference prompts (`hello`, `안녕하세요`,
`The capital of France is`, `1 + 1 =`).

| Surface | Result |
| ------- | ------ |
| Tokenizer (prompt → ids) | ✅ exact match (after Stage 5-E pre-tokenizer fix) |
| Greedy generated bytes (5 tokens × 4 prompts) | ✅ byte-identical |
| Token-id sequences | 3/4 byte-identical; 1/4 BPE-segmentation-equivalent (same bytes, different valid tokenisation) |

Reproduce yourself:

```bash
./scripts/run_willamette_reference.sh
./scripts/run_bitnet_reference.sh   # needs the upstream build, see docs
./scripts/compare_reference.sh
```

Full procedure in [`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md).

## Documentation map

| File | Purpose |
| ---- | ------- |
| [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md) | Exact upstream SHA, file/line references, model SHA256 |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | All benchmark numbers, scaling sweep, llama2.c head-to-head, EXO Pentium-II comparison |
| [`docs/BENCHMARK_SCHEMA.md`](docs/BENCHMARK_SCHEMA.md) | Versioned machine-readable benchmark contract and remote matrix runner |
| [`REFERENCE_COMMIT.md`](REFERENCE_COMMIT.md) | Stage 1 GGUF inspection log + verification table |
| [`docs/I2_S_LAYOUT.md`](docs/I2_S_LAYOUT.md) | Pinned-source citation for the I2_S byte/scale layout |
| [`docs/BITLINEAR_I2S_MATVEC.md`](docs/BITLINEAR_I2S_MATVEC.md) | BitLinear matvec contract & code → ternary map |
| [`docs/BITNET_FORWARD_PLAN.md`](docs/BITNET_FORWARD_PLAN.md) | Stage-by-stage forward-pass plan & status |
| [`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md) | Willamette ↔ bitnet.cpp comparison procedure & result |
| [`LIMITATIONS.md`](LIMITATIONS.md) | What's supported, what isn't, what won't be |
| [`docs/KV_CACHE_QUANT.md`](docs/KV_CACHE_QUANT.md) | v0.9.0 KV i8 design + memory math + fidelity contract + measured-negative i4 prototypes |
| [`docs/LUT_KERNEL_RFC.md`](docs/LUT_KERNEL_RFC.md) | Design and outcome record for the shipped scalar LUT; SSSE3 follow-up remains measurement-gated |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Exact env to reproduce every number above |
| [`GOLDEN_TESTS.md`](GOLDEN_TESTS.md) | Reference prompts, token ids, expected outputs |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

## Project rules (carried forward to every contribution)

1. **No fake weights.** Every weight tensor read from the real GGUF
   bytes. No random/pseudo/procedural placeholders.
2. **No fake tokenizer.** Vocabulary and merges come from
   `tokenizer.ggml.*` metadata; no hand-written Korean vocab or
   ASCII-only fallback.
3. **No fake logits.** If a forward step is not implemented, the
   relevant code returns a typed error (`NotImplemented`,
   `UnsupportedTensorType`, `UnsupportedTokenizer`, …) — it does not
   synthesise output.
4. **No unverified SIMD.** `target-cpu=native` is not the default;
   every SIMD kernel ships only after on-target validation against the scalar
   reference. BitLinear SSE2 is validated on antiX Pentium-M; Q8_0 NEON,
   AVX2, and SSE2 are validated on M4, HP ProBook, mbp2012, and antix1. The
   BitLinear AVX2/AVX-512 and additional LUT variants remain unmerged.
5. **No model files in this repo.** GGUFs are downloaded at use time.
6. **Source-pinned changes.** Any modification of a constant
   (`GGML_TYPE_*`, RoPE type, regex set, scale offset, …) must cite
   the upstream `file:line` it derives from.

See [`LIMITATIONS.md`](LIMITATIONS.md) for what those rules currently
exclude.

## License

Licensed under either of

* Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms
or conditions.

This project consumes packed weights from
[`microsoft/BitNet-b1.58-2B-4T`](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
under that model's separate license; see Microsoft's repository for
upstream model terms. We do not redistribute the model file.
