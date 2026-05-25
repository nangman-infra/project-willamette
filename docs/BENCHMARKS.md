# Benchmarks

Reproducible CPU-only inference numbers, captured against the official
`microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` model
(SHA-256 `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`,
1.106 GiB on disk, 332 tensors, 30 transformer blocks).

These numbers exist to make speed-up claims **falsifiable**. Every
section names the host, the willamette tag, the dispatch backend, and
the command that produced the timing. Re-running the same command on
the same host with the same tag must reproduce the reported figure to
within ±10 % (warm-cache decode-step variance).

## Hosts

### Mac M1 — Apple Silicon NEON reference

| | |
| --- | --- |
| CPU | Apple M1 (8 cores: 4 P + 4 E) |
| RAM | 16 GiB |
| OS | macOS (Sequoia / Sonoma — equivalent for these numbers) |
| Toolchain | rustc 1.94.0 (rust-toolchain.toml pin) |

### antix1 — Pentium-M humble-hardware host

| | |
| --- | --- |
| CPU | Intel Pentium M 2.00 GHz (Banias/Dothan, family 6 model 13) |
| Cores | 1 (no SMT) |
| SIMD ceiling | SSE2 (no SSE3 / SSSE3 / SSE4 / AVX) |
| RAM | 2 GiB |
| OS | Debian 12 bookworm + antiX kernel `5.10.224-antix.1-486-smp` |
| Toolchain | i686-unknown-linux-musl, cross-built on the CI runner |

## 2026-05-25 — v0.4.1-mvp baselines

### Apple M1 — NEON (`Kernel::AArch64Neon`)

Measured on the v0.2.0-mvp release cycle; the matvec kernel and
attention path have not changed shape since, so the figure carries
forward (re-bench when a structural change lands).

* `cargo run --release -- bench --model …/ggml-model-i2_s.gguf --decode-steps 20`
* **Decode-step throughput**: ≈ **7.9 tok/s** (warm KV cache,
  averaged over 20 samples, Stage 10 perf set: pre-decoded norm
  weights + rayon row-parallel matvec + f32-input NEON kernel).

### antix1 — Pentium-M scalar (`Kernel::Scalar`)

Prebuilt static binary from the
[v0.4.1-mvp release](https://github.com/nangman-infra/project-willamette/releases/tag/v0.4.1-mvp)
(`willamette-v0.4.1-mvp-i686-unknown-linux-musl.tar.gz`, 2.5 MB stripped):

```
./willamette bench --model ~/models/ggml-model-i2_s.gguf --decode-steps 3
```

| Measurement | Value |
| --- | --- |
| `dispatch::active_kernel().label()` | `i686 scalar` |
| `Host arch` (from `std::env::consts::ARCH`) | `x86` |
| BitLinear matvec (attn_q, 2560 × 2560 ternary) | **60.5 ms** |
| BitLinear matvec throughput | 108 M elements / sec |
| Single-token forward (30 layers, no cache) | **21.72 s** |
| Decode-step forward (with KV cache, avg of 3) | **21.65 s** |
| Decode-step throughput | **0.05 tok/s** |

End-to-end wall time of the bench command: 1 min 48 s (5.2 s of which
is GGUF parse + tensor directory build over the 332 tensors via mmap).

#### What this number does and does not prove

1. **It proves the runtime works on Pentium-class hardware.** A 1.1 GiB
   GGUF maps into a 2 GiB-RAM machine, the tokenizer constructs, and
   the BitLinear forward produces finite hidden states with KV cache
   maintained. The "humble CPU runs medium LLMs" half of the thesis
   is verified end-to-end on a 21-year-old CPU class.

2. **It does not make this configuration usable for chat.** 0.05 tok/s
   ≈ 21 seconds per token; a 50-token reply takes ~18 minutes. The
   chat / TUI subcommands run but the bottleneck is the matvec kernel,
   not I/O or attention.

3. **It gives Stage 6-B a concrete "before" number.** Any SSE2 kernel
   added under `src/model/bitlinear_sse2.rs` must:
   * produce the same matvec output as the scalar reference within
     the tolerance already documented in `tests/bitlinear_simd.rs`
     (max-abs-diff `< 1e-2` per BitLinear column);
   * report a `matvec ms` lower than 60.5 ms on antix1 to justify
     dispatch picking it.

   Until both conditions hold, `dispatch::select_kernel` keeps
   returning `Kernel::Scalar` on x86 — the `Kernel::X86Sse2` slot is
   present for the dashboard and detection arrays but does not route
   any traffic.

#### M1 NEON ÷ Pentium-M scalar

The two hosts are different in four independent dimensions (clock,
IPC, SIMD width, memory bandwidth), so a single ratio understates the
SIMD contribution. For the record:

* **Decode-step ratio**: 7.9 / 0.05 ≈ **158× faster on M1**.
* **BitLinear matvec ratio**: M1's per-matvec time is roughly 5 ms
  (back-calculated from the 7.9 tok/s figure across 30 layers × ~6
  matvecs/layer of similar shape), versus 60.5 ms scalar → **≈ 12×**
  on the matvec alone. The remaining factor of ~13× comes from clock
  (2.0 GHz → ~3.2 GHz P-core), IPC (in-order P-M vs out-of-order
  Firestorm), memory bandwidth, and rayon multi-core scheduling on
  the M1 against antix1's single core.

A theoretical SSE2 kernel that processes 16 × i8 elements per cycle
sits at ~8 × the scalar's per-cycle work; once memory-bandwidth
limits kick in, the realised speed-up on BitLinear matvecs is
typically 4–8 ×. Anything claiming materially more on this host
warrants verification.

## How to reproduce

### Pentium-M host (antix1 or equivalent)

1. Download the prebuilt static binary for your tag:
   ```
   curl -L -o willamette.tar.gz \
     https://github.com/nangman-infra/project-willamette/releases/download/v0.4.1-mvp/willamette-v0.4.1-mvp-i686-unknown-linux-musl.tar.gz
   tar -xzf willamette.tar.gz
   ```
2. Fetch the model from HuggingFace and verify SHA-256:
   ```
   curl -L -o ggml-model-i2_s.gguf \
     https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true
   echo '4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162  ggml-model-i2_s.gguf' | sha256sum -c -
   ```
3. Run the bench (3 decode samples is enough; variance < 5 %):
   ```
   ./willamette-…/willamette bench --model ./ggml-model-i2_s.gguf --decode-steps 3
   ```

### Apple Silicon host

1. Clone the repo and let `rust-toolchain.toml` pin the compiler:
   ```
   git clone https://github.com/nangman-infra/project-willamette
   cd project-willamette
   cargo run --release -- bench --model ./ggml-model-i2_s.gguf --decode-steps 20
   ```

Either path prints a banner with `Host arch:` + `Matvec backend:` that
matches the `Kernel` your dispatch picked.
