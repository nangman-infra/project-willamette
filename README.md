# Project Willamette

**Thesis:** medium-sized publicly-released LLMs (1B – 13B parameters)
run on **CPU-only humble hardware** — older laptops, low-RAM thin
clients, retro x86, Raspberry-Pi-class ARM — without a GPU. The
proof is two binaries: an offline **`willamette-prep`** that bakes
a model down to a hardware-aware form, and an online
**`willamette`** runtime that just executes the baked form. The
runtime is Rust, uses zero-copy `mmap`, and targets ARM + x86_64 +
i686 (eventually MMX-era), validated on emulators.

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
       NOT BUILT YET                                      WORKING TODAY (v0.2.3)
```

The split is the same pattern TensorFlow Lite / Core ML / ONNX
Runtime / `bitnet.cpp`'s `quantize` use: the expensive once-per-model
work runs where compute is cheap, and the on-device runtime stays
small. `willamette-prep` is the next major piece of work; what
exists today is the runtime side, hardcoded to BitNet b1.58 2B.

## Status: v0.2.3-mvp

What works **today**, on the path toward the thesis:

| Property | Value |
| -------- | ----- |
| Working model | `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` (1.1 GiB ternary) |
| Model SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| Reference parity (bitnet.cpp) | ✅ byte-identical generated text on Stage 5-E prompts |
| Reference build | `microsoft/BitNet @ 01eb4157…` (see [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md)) |
| Apple Silicon NEON kernel | ✅ implemented + validated |
| Multi-core CPU parallelism | ✅ `rayon` per-row BitLinear matvec |
| Norm-weight + scratch caching | ✅ Stage 10-A / 10-B |
| Chat + TUI surfaces | ✅ `willamette chat` (stdio) + `willamette tui` (ratatui) |
| All-in-one launcher | ✅ `scripts/willamette` (SHA verify + HF download + build + run) |
| Tests | **242** passing, 0 warnings (`cargo test --release`) |
| SonarQube Quality Gate | ✅ OK — `new_coverage` 100 %, `new_violations` 0 |

What does **not** work yet but is on the roadmap toward the thesis:

| Property | Value |
| -------- | ----- |
| Model coverage beyond BitNet b1.58 (Llama / Mistral / Phi / …) | ❌ runtime hardcoded to BitNet b1.58 |
| Standard GGUF quant types (Q4_0, Q4_K, Q5_K, Q8_0, …) | ❌ only `I2_S` |
| `willamette-prep` (offline preprocessor) | ❌ not started |
| x86_64 AVX2 / SSE2 SIMD kernel | ⏳ Stage 6-B pending — needs x86 host validation |
| i686 / MMX kernel | ❌ not started |
| KV cache int8 quantisation | ❌ — biggest immediately-available memory win |
| LLM-in-a-Flash style mmap windowing | ❌ |
| Emulator-based humble-hardware benchmark pipeline (QEMU / 86Box) | ❌ |
| Generic scalar fallback (every supported ISA) | ✅ correctness-only; ports cleanly |
| GPU | ⛔ explicitly out of scope by thesis (CPU only) |

## Quick start

### 1. Toolchain

* Rust 1.94 or newer (`rustup install stable`).
* macOS / Linux on aarch64 or x86_64. Apple Silicon gets the NEON path
  for free; x86 currently runs the scalar fallback (see
  [`LIMITATIONS.md`](LIMITATIONS.md) for the AVX2/SSE2 roadmap).

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

Expected output (Apple Silicon, ~5 s):

```
Generating:  Paris. Paris
Generated 3 token(s): [12366, 13, 12366]
Generated text:   " Paris. Paris"
Full text:        "The capital of France is Paris. Paris"
```

## CLI subcommands

```text
willamette inspect    --model PATH
willamette tokenize   --model PATH --text TEXT [--no-bos] [--add-eos]
willamette logits     --model PATH --prompt TEXT [--top-k N] [--no-bos]
willamette run        --model PATH --prompt TEXT
                      [--max-new-tokens N]
                      [--no-bos]
                      [--temperature F] [--top-k K] [--top-p P]
                      [--repetition-penalty R] [--seed S]
                      [--stop-id ID]...
willamette bench      --model PATH [--decode-steps N]
willamette --version
```

* `inspect` — Stage 1. Dumps every metadata key + every tensor's raw
  ggml_type, shape, offset, and byte length. No inference.
* `tokenize` — Stage 2. Runs the GGUF-bundled GPT-2 byte-level BPE
  tokenizer (with the `LLAMA_VOCAB_PRE_TYPE_DEFAULT` 3-regex
  pre-tokenization, matching upstream when `tokenizer.ggml.pre` is
  absent). Refuses to run on tokenizer models we don't support.
* `logits` — Stage 4-D5. Runs the full 30-layer forward and prints the
  top-K next-token logits. Useful for comparing against bitnet.cpp.
* `run` — Stage 5. Real BitLinear forward + greedy or sampled
  generation, with KV cache.
* `bench` — Stage 6-A. Times one matvec, one no-cache forward, and one
  cached decode step. Reports which BitLinear backend (NEON or scalar)
  is active.

## Performance

Numbers from Apple Silicon (M1+, aarch64, single core, default cargo
release profile, our scalar reference vs. our NEON backend):

| Operation | Scalar (Stage 6-A) | NEON (Stage 6-C) | Speed-up |
| --------- | ----------------: | ---------------: | -------: |
| One I2_S BitLinear matvec (2560×2560 ternary) | 13.7 ms | **1.9 ms** | **7.3×** |
| Single-token forward (30 layers, no cache) | 5104 ms | **669 ms** | **7.6×** |
| Decode step (with KV cache, avg 3 runs) | 5034 ms | **656 ms** | **7.7×** |
| Throughput (decode, tokens/sec) | 0.20 | **1.52** | **7.5×** |

These are correctness-first numbers. There is no SIMD activation
quantisation, no thread pool, no graph fusion. Future Stage 8 work on
x86 AVX2/SSE2 and possible BitNet-paper i8 activation quant could move
this further.

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
   AVX2/SSE2 stays unimplemented until an x86 host is available to
   validate it against the scalar fallback (Stage 6-B).
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
