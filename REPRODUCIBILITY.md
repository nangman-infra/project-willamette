# Reproducibility — Project Willamette v0.7.1-mvp

*Last revised 2026-05-27.*

This file pins every external value that the numbers in
[`README.md`](README.md), [`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md),
and [`GOLDEN_TESTS.md`](GOLDEN_TESTS.md) depend on. If you cannot
reproduce a result, check this file first.

## 1. Toolchain

| Tool | Version |
| ---- | ------- |
| Rust toolchain | `rustc 1.94.0` (stable) — see `rust-toolchain.toml` (none currently — uses `rustup default stable`) |
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
cargo test --release       # expect 189 passing, 0 warnings, 0 failures
```

Failure modes:

* `cargo test` SKIPs every test that needs the real GGUF if the file
  is missing — but everything else (unit tests, synthetic GGUF parser
  tests, NEON unit fixtures) still runs.
* `tests/bitlinear_simd.rs` is `#![cfg(target_arch = "aarch64")]` —
  on x86 hosts it compiles to zero tests. Its x86 counterparts are
  `tests/bitlinear_sse2.rs` and `tests/bitlinear_sse2_i8.rs`
  (`#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]`).
  When the real GGUF isn't present the integration tests SKIP at
  runtime.
* Matvec backend on x86 is **SSE2 int8** by default since v0.5.0 /
  v0.7.0; fall back to f32 mask-add with `RUSTFLAGS="--cfg
  willamette_sse2_f32"`. Pure scalar runs only on architectures with
  no SIMD kernel compiled in (or when no SIMD feature is detected at
  runtime).
* Test counts (v0.7.1-mvp): **291** on Mac aarch64 (default cfg),
  **295** on x86 with the real model present (SSE2 + SSE2-i8
  integration tests run), **287** on x86 without the model (those
  integration tests SKIP but unit tests still run).

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
