# Project Willamette

**Thesis:** mid-sized publicly-released LLMs run on **CPU-only
humble hardware** — older laptops, low-RAM thin clients, retro x86,
Raspberry-Pi-class ARM — without a GPU. The proof is two binaries:
an offline **`willamette-prep`** that bakes a model down to a
hardware-aware form, and an online **`willamette`** runtime that
just executes the baked form. The runtime is Rust, uses zero-copy
`mmap`, and targets ARM + x86_64 + i686 (eventually MMX-era),
validated on real hardware (antiX on Pentium-M today) and on
emulators (QEMU / 86Box).

> **Sweet spot is hardware-dependent.** On Pentium-M-class SSE2
> hardware (the verified floor at 2026-05-27) the measured ceiling
> is roughly **100 M params for chat speed (≥ 5 tok/s)**, **500 M
> for "slow but usable" (≥ 1 tok/s)**, **5 B for demonstration
> (≥ 0.1 tok/s)**. Modern AVX2 / multi-core moves every threshold an
> order of magnitude up. Full scaling table and the EXO Pentium-II
> comparison: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

Starting point: [microsoft/BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
in its `ggml-model-i2_s.gguf` form (1.58-bit ternary weights) — the
one model fully working end-to-end today. Destination: a runtime
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
       NOT BUILT YET                                      WORKING TODAY (v0.9.0-mvp)
```

The split is the same pattern TensorFlow Lite / Core ML / ONNX
Runtime / `bitnet.cpp`'s `quantize` use: the expensive once-per-model
work runs where compute is cheap, and the on-device runtime stays
small. `willamette-prep` is the next major piece of work; what
exists today is the runtime side, hardcoded to BitNet b1.58 2B.

## Status: v0.9.0-mvp

What works **today**, on the path toward the thesis:

| Property | Value |
| -------- | ----- |
| Working reference model | `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` (1.1 GiB ternary) |
| Model SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| BitNet-family fine-tunes accepted | ✅ `bitnet-b1.58`, `bitnet-25`, `bitnet` GGUF strings load through `model::architecture::registry`. End-to-end greedy decode verified on antix1 against [`jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF`](https://huggingface.co/jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF) (French) and [`Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf`](https://huggingface.co/Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf) (Solana coding). See [`docs/PHASE_III_ARCHITECTURE_RFC.md`](docs/PHASE_III_ARCHITECTURE_RFC.md). |
| Reference parity (bitnet.cpp) | ✅ byte-identical generated text on Stage 5-E prompts |
| Reference build | `microsoft/BitNet @ 01eb4157…` (see [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md)) |
| Apple Silicon NEON kernel | ✅ implemented + validated (Apple M4 dev host) |
| **x86 SSE2 i8 kernel (default)** | ✅ **Stage 6-B landed** — validated on antix1 (Pentium-M 2 GHz, i686). 2.2× over f32 SSE2, ~5.4× over scalar; byte-identical greedy output |
| Runtime CPU dispatch | ✅ NEON / SSE2-i8 / SSE2-f32 / scalar selected at runtime ([`src/model/dispatch.rs`](src/model/dispatch.rs)) |
| **Prebuilt static binaries** | ✅ 6 targets per release — `x86_64`, `i686`, `aarch64`, `armv7` Linux musl + `aarch64`, `x86_64` macOS. See [Releases](https://github.com/nangman-infra/project-willamette/releases). |
| Multi-core CPU parallelism | ✅ `rayon` per-row BitLinear matvec |
| Norm-weight + scratch caching | ✅ Stage 10-A / 10-B |
| **KV cache i8 quantisation** | ✅ **per-token absmax i8 since v0.9.0** — ~3.97× memory shrink (150 KB → 37.7 KB per token on BitNet 2B). Greedy output byte-identical to the f32 reference on Stage 5-E prompts (Apple M4 NEON + antix1 i686 SSE2). See [`docs/KV_CACHE_QUANT.md`](docs/KV_CACHE_QUANT.md). |
| Chat + TUI surfaces | ✅ `willamette chat` (stdio) + `willamette tui` (ratatui full-screen) |
| Synthetic GGUF builder | ✅ `willamette synth-gguf --preset {tiny\|small\|medium}` (humble-HW throughput benchmarks) |
| Ternary weight distribution | ✅ `willamette analyze` (-1 / 0 / +1 fractions across BitLinear tensors) |
| All-in-one launcher | ✅ `scripts/willamette` (SHA verify + HF download + build + run) |
| Tests | **301** passing (Mac aarch64), 305 (x86 with SSE2 paths), 0 warnings, `cargo test --release` |
| SonarQube Quality Gate | ✅ OK across the v0.x release cycle |
| Beat vanilla Llama 2 same-machine | ✅ 110M head-to-head on antix1: BitNet+SSE2 **1.97× faster** than `llama2.c` |

What does **not** work yet but is on the roadmap toward the thesis:

| Property | Value |
| -------- | ----- |
| Model coverage beyond the BitNet family (Llama / Mistral / Phi / Gemma) | ❌ BitNet family (`bitnet-b1.58` / `bitnet-25` / `bitnet`) accepted today; non-BitNet architectures pending Phase III-B — see [`docs/PHASE_III_ARCHITECTURE_RFC.md`](docs/PHASE_III_ARCHITECTURE_RFC.md) |
| Standard GGUF quant types (Q4_0, Q4_K, Q5_K, Q8_0, …) | ❌ only `I2_S` |
| `willamette-prep` (offline preprocessor) | ❌ not started — thesis's missing half |
| AVX2 / AVX-512 SIMD kernel | ❌ not started — Pentium-M doesn't have it; gain target for modern x86 |
| LUT (TL1/TL2) kernel | ❌ needs SSSE3+ (`pshufb`) → not for Pentium-M; for SSSE3+ hosts later |
| MMX-era / sub-SSE2 kernel | ❌ not started |
| KV cache int8 quantisation | ✅ landed in v0.9.0 (see Status table above) |
| LLM-in-a-Flash style mmap windowing | ❌ |
| Emulator-based humble-hardware benchmark pipeline (QEMU / 86Box) | ❌ |
| Generic scalar fallback (every supported ISA) | ✅ correctness-only; ports cleanly |
| GPU | ⛔ explicitly out of scope by thesis (CPU only) |

## Quick start

You have **two install paths** — picking the lighter one matters on
humble hardware:

### Option A — Prebuilt static binary (recommended for low-end hosts)

No toolchain, no compile time. Pick the tarball matching your host:

```bash
TAG=v0.9.0-mvp
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

Expected output (Apple M-series, < 3 s on M4; antiX Pentium-M ≈ 8 s
per token after the prefill, so plan for ~60 s end-to-end for 3
tokens including model mmap):

```
Generating:  Paris. Paris
Generated 3 token(s): [12366, 13, 12366]
Generated text:   " Paris. Paris"
Full text:        "The capital of France is Paris. Paris"
```

## CLI subcommands

```text
willamette inspect    --model PATH
willamette analyze    --model PATH
willamette tokenize   --model PATH --text TEXT [--no-bos] [--add-eos]
willamette logits     --model PATH --prompt TEXT [--top-k N] [--no-bos]
willamette run        --model PATH --prompt TEXT
                      [--max-new-tokens N]
                      [--no-bos]
                      [--temperature F] [--top-k K] [--top-p P]
                      [--repetition-penalty R] [--seed S]
                      [--stop-id ID]...
willamette bench      --model PATH [--decode-steps N]
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
* `analyze` — Counts -1 / 0 / +1 across every BitLinear (I2_S) tensor.
  Reports the zero fraction (the upper bound on what sparsity-aware
  skipping could save). Real 2B: 28.9 / 42.2 / 28.9 %.
* `tokenize` — Stage 2. Runs the GGUF-bundled GPT-2 byte-level BPE
  tokenizer (with the `LLAMA_VOCAB_PRE_TYPE_DEFAULT` 3-regex
  pre-tokenization). Refuses to run on tokenizer models we don't
  support.
* `logits` — Stage 4-D5. Runs the full 30-layer forward and prints the
  top-K next-token logits. Useful for comparing against bitnet.cpp.
* `run` — Stage 5. Real BitLinear forward + greedy or sampled
  generation, with KV cache.
* `bench` — Stage 6-A. Times one matvec, one no-cache forward, and one
  cached decode step. Reports the **active backend label** (e.g.
  `i686 SSE2 (i8)`, `aarch64 NEON`) — also runs a sparse-prototype
  comparison against the dense kernel on `attn_q`.
* `chat` — Stage 9. Multi-turn stdio chat (line-based). `/quit`,
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
```

Needs a real terminal (not the Claude-Code embedded chat). Over SSH
use `ssh -t` to force a pseudo-tty when launching one-shot:

```bash
ssh -t user@host '~/bin/willamette tui --model ~/models/ggml-model-i2_s.gguf'
```

Expect very slow generation on humble HW — on antix1 Pentium-M the
real 2B model runs at ~0.4 tok/s (i8 SSE2 default). Use **Esc** to
cancel a long answer.

## Performance

Headline numbers (real BitNet 2B model, decode step, `cargo
--release`). Full table including the synthetic 110M / 7M points,
EXO Pentium-II comparison, and llama2.c head-to-head live in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| Host | Kernel | tok/s |
| --- | --- | ---: |
| **Apple M4** (Mac16,10, dev box) | aarch64 NEON | **7.9** |
| **mbp2012** Mid-2012 MBP Ivy Bridge i7-3520M (sub-AVX2 host) | x86_64 SSE2 (i8) | **2.65** |
| **antiX Pentium-M 2 GHz** (humble validation host) | i686 SSE2 (i8) | **0.41** |
| antiX Pentium-M 2 GHz | i686 scalar (v0.4.1) | 0.05 |

Same hardware, same model, kernel only:

| antiX Pentium-M progression | tok/s | speed-up |
| --- | ---: | ---: |
| scalar reference | 0.05 | — |
| SSE2 f32 mask-add (v0.4.x f32 path) | 0.19 | 2.49× over scalar |
| **SSE2 i8 (v0.5.0+ default)** | **0.41** | **2.15× over f32 / 5.4× over scalar** |

Same-machine head-to-head vs `llama2.c` (vanilla Llama 2 f32) on
antix1 at **110M scale** — both single-thread, both SSE2:

| Build | tok/s |
| --- | ---: |
| `llama2.c` `stories110M` (vanilla f32) | 2.51 |
| `willamette` synth 110M (BitNet b1.58 + SSE2 i8) | **4.96 (1.97× faster)** |

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
| [`REFERENCE_COMMIT.md`](REFERENCE_COMMIT.md) | Stage 1 GGUF inspection log + verification table |
| [`docs/I2_S_LAYOUT.md`](docs/I2_S_LAYOUT.md) | Pinned-source citation for the I2_S byte/scale layout |
| [`docs/BITLINEAR_I2S_MATVEC.md`](docs/BITLINEAR_I2S_MATVEC.md) | BitLinear matvec contract & code → ternary map |
| [`docs/BITNET_FORWARD_PLAN.md`](docs/BITNET_FORWARD_PLAN.md) | Stage-by-stage forward-pass plan & status |
| [`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md) | Willamette ↔ bitnet.cpp comparison procedure & result |
| [`LIMITATIONS.md`](LIMITATIONS.md) | What's supported, what isn't, what won't be |
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
   every SIMD kernel ships only after on-target validation against
   the scalar reference. SSE2 (i8) is validated on antiX Pentium-M;
   AVX2 / AVX-512 / LUT (SSSE3+) remain unmerged until a host is in
   hand to test them.
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
