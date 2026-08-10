# Reproducibility — Project Willamette v0.10.0

*Last revised 2026-08-05.*

This file pins every external value that the numbers in
[`README.md`](README.md), [`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md),
and [`GOLDEN_TESTS.md`](GOLDEN_TESTS.md) depend on. If you cannot
reproduce a result, check this file first.

## 1. Toolchain

| Tool | Version |
| ---- | ------- |
| Rust toolchain | `rustc 1.94.0` — pinned by `rust-toolchain.toml` |
| Cargo | `cargo 1.94.0` (matches Rust) |
| Apple `clang` for C++ side (bitnet.cpp build only) | `clang version 21` (Xcode CommandLineTools 1267) |
| CMake (bitnet.cpp build only) | `4.3.2` (Homebrew) |
| Python (only for bitnet.cpp's LUT codegen, not needed for Willamette itself) | `python3 ≥ 3.10` |

`cargo --version` and `rustc --version` should both return `1.94.0` or
newer; older versions may compile but were not exercised in CI.

## 2. Host

* **Reference host**: Apple Silicon Mac, `aarch64-apple-darwin`,
  Darwin kernel `25.5.0` or newer.
* `uname -m` → `arm64`
* `rustc -vV | grep host` → `host: aarch64-apple-darwin`

All numbers in this repo were generated on this host class. Other
hosts can run the project — Stage 6-A scalar fallback is portable —
but the NEON timings and the 7.5× speed-up will not transfer.

## 3. Model file

| Property | Value |
| -------- | ----- |
| HuggingFace repo | `microsoft/bitnet-b1.58-2B-4T-gguf` |
| File name | `ggml-model-i2_s.gguf` |
| Local path (default) | `./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` |
| Size | `1,187,801,280` bytes (1.106 GiB) |
| SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| Architecture (in metadata) | `bitnet-b1.58` |
| `general.file_type` | `40` (= `LLAMA_FTYPE_MOSTLY_I2_S`) |
| Tokenizer model | `gpt2` (byte-level BPE) |
| Vocab size | `128256` |
| Block count | `30` |
| Embedding length | `2560` |
| FFN length | `6912` |
| Head count | `20` |
| KV head count | `5` (GQA 4:1) |
| Head dim | `128` |
| RoPE freq base | `500000` |
| Context length | `4096` |

Verify the SHA256 before doing anything:

```bash
shasum -a 256 ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
```

If the value differs, your downloaded file does not match the layout
pins in [`docs/I2_S_LAYOUT.md`](docs/I2_S_LAYOUT.md).

### Derived Q6_K embedding artifact

The runtime can derive an additional low-memory artifact from the exact source
above. Only `token_embd.weight` changes from F16 to Q6_K; every transformer
I2_S tensor and tokenizer byte is copied unchanged.

| Property | Value |
| -------- | ----- |
| File name | `ggml-model-i2_s-embed-q6_k.gguf` |
| Size | `800,468,160` bytes (0.745 GiB) |
| SHA256 | `492e4d2a8db2eefc5f8c86acd08eea6707294de67ce871b5d732e9bfcb468376` |
| Source SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| Changed tensor | `token_embd.weight`: F16 656,670,720 bytes → Q6_K 269,337,600 bytes |

```bash
cargo run --release --bin willamette-prep -- \
  --model ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --output ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s-embed-q6_k.gguf \
  --profile embedding-q6-k
```

Use `--dry-run` with the same arguments to validate the source graph and print
the complete size/offset plan without creating the output path. The linker
recomputes every aligned tensor offset, so the transformed embedding need not
be the first physical tensor and its size reduction need not be an alignment
multiple.

The compatibility interface
`cargo run --release -- repack-embedding-q6k --model SOURCE --output DEST`
calls the same repacker and produces identical bytes.

When the pinned external model is present, reproduce both the output size and
SHA256 regression gate with:

```bash
cargo test --test artifact_linker -- --ignored
```

Quality comparison uses the same WikiText-2 raw test prefix for both artifacts:

| Property | Value |
| -------- | ----- |
| Archive URL | `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip` |
| Archive SHA256 | `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11` |
| `wiki.test.raw` SHA256 | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| Scored transitions | first 1,024, one metadata-default BOS, no implicit EOS |
| F16 perplexity | `14.266282121` |
| Q6_K perplexity | `14.273353951` (`+0.0496%`) |

```bash
willamette perplexity --model MODEL.gguf \
  --file wikitext-2-raw/wiki.test.raw --max-tokens 1024
```

This is a relative artifact-quality gate over an identical runtime and token
sequence, not a claim that the number is directly comparable to another
framework's chunking or cache policy.

## 4. Pinned upstream

See [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md) for the canonical table; the
short version:

| Repo | Branch | Commit |
| ---- | ------ | ------ |
| `microsoft/BitNet` | `main` | `01eb415772c342d9f20dc42772f1583ae1e5b102` |
| `Eddie-Wang1120/llama.cpp` (submodule `3rdparty/llama.cpp`) | _detached_ | `1f86f058de0c3f4098dedae2ae8653c335c868a1` |

`GGML_TYPE_I2_S = 36` is defined at
`3rdparty/llama.cpp/ggml/include/ggml.h:393` of the pinned submodule
revision; every other source citation in our docs uses the same
revision.

## 5. Reproducing the build

```bash
git clone <THIS REPO> project-willamette
cd project-willamette

hf download microsoft/bitnet-b1.58-2B-4T-gguf \
    ggml-model-i2_s.gguf \
    --local-dir ./models/bitnet-b1.58-2B-4T-gguf

cargo build --release
cargo test --release
```

The default suite does not require the external model. Tests that read the
official GGUF use Rust's explicit `#[ignore]` marker, so the test summary
reports them as ignored rather than passing after an early return. After the
model is downloaded and its checksum is verified, run them with:

```bash
cargo test --release --tests -- --ignored
```

Rust's standard test harness has no runtime "skipped" result. Consequently,
an explicitly selected real-model test fails with a model-not-found message
instead of returning successfully when the file is absent.

Failure modes:

* `cargo test` reports every real-GGUF test as ignored; unit tests,
  synthetic GGUF tests, and model-independent kernel fixtures still run.
* `tests/bitlinear_simd.rs` is `#![cfg(target_arch = "aarch64")]` —
  on x86 hosts it compiles to zero tests. Its x86 counterparts are
  `tests/bitlinear_sse2.rs` and `tests/bitlinear_sse2_i8.rs`
  (`#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]`).
  Their real-GGUF cases follow the same explicit-ignore policy.
* Matvec backend on x86 is **SSE2 int8** by default since v0.5.0 /
  v0.7.0; fall back to f32 mask-add with `RUSTFLAGS="--cfg
  willamette_sse2_f32"`. Pure scalar runs only on architectures with
  no SIMD kernel compiled in (or when no SIMD feature is detected at
  runtime).
* Exact test counts vary by target architecture because NEON and x86 kernel
  crates are compile-time gated. Treat the passed/ignored/failed summary as
  authoritative rather than comparing against a stale fixed count.

## 6. Reproducing the reference comparison

The reference comparison (`docs/REFERENCE_COMPATIBILITY.md`) requires
the bitnet.cpp build. Procedure:

```bash
brew install cmake                              # one-time

# Clone microsoft/BitNet at the pinned SHA.
git clone https://github.com/microsoft/BitNet.git /tmp/bitnet-upstream
cd /tmp/bitnet-upstream
git checkout 01eb415772c342d9f20dc42772f1583ae1e5b102
git submodule update --init --recursive

# Generate the (model-specific) LUT kernel header. NOT used by the
# I2_S CPU path, but the build expects the file to exist.
python3 utils/codegen_tl1.py --model bitnet_b1_58-3B \
    --BM 160,320,320 --BK 64,128,64 --bm 32,64,32

# Configure WITHOUT BITNET_ARM_TL1 (the LUT path file
# ggml-bitnet-lut.cpp takes ~60 min of clang template instantiation
# when TL1 is on — we skip it because we only need the I2_S MAD path).
cmake -B build -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build --target llama-cli llama-tokenize -j 4
# Expected wall-clock: ~7 minutes on Apple M-series.

# Back in the Willamette repo:
cd <THIS REPO>
./scripts/run_willamette_reference.sh
./scripts/run_bitnet_reference.sh           # uses /tmp/bitnet-upstream/build/bin/*
./scripts/compare_reference.sh              # writes compat_report.md
```

Expected `compat_report.md` (full content tracked in
[`GOLDEN_TESTS.md`](GOLDEN_TESTS.md)):

| Prompt | Tokenizer match | Generated-bytes match |
| ------ | :-------------: | :--------------------: |
| `hello` | ✅ | ✅ |
| `안녕하세요` | ✅ | ✅ |
| `The capital of France is` | ✅ | ✅ |
| `1 + 1 =` | ✅ | ✅ |

## 7. Reproducing the benchmark

```bash
./target/release/project-willamette bench \
    --model ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --decode-steps 3
```

Expected on Apple Silicon M-series (NEON dispatch active):

```text
Host arch:        aarch64 (Apple Silicon / ARM64)
Matvec backend:   aarch64 NEON (Stage 6-C)
...
BitLinear matvec (attn_q, 2560×2560 ternary): ~1.9 ms / ~3500 M elem/s
Single-token forward (30 layers, no cache):   ~670 ms / ~1.5 tok/s
Decode-step forward (with KV cache, avg 3):    ~660 ms / ~1.5 tok/s
```

Variance: ±10 % run-to-run is normal (no warm-up beyond a single
matvec). Numbers will be 5–7× slower on the same hardware if you
hot-patch `bitlinear_i2s_matvec_f32` to call the scalar path.

## 8. Reporting an unreproducible result

If you cannot reproduce a number with the above pins, please include:

1. `rustc -vV` output
2. `uname -a` output
3. `shasum -a 256` of `ggml-model-i2_s.gguf`
4. The exact `cargo test --release` output (or the failing
   subset, e.g. `cargo test --release --test bitlinear_simd`)
5. For the bitnet.cpp comparison: `cd /tmp/bitnet-upstream && git rev-parse HEAD`
   plus the submodule SHA.
