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

### Mac M4 — Apple Silicon NEON reference

| | |
| --- | --- |
| CPU | Apple M4 (Mac16,10) |
| Cores | 10 (4 P-core + 6 E-core) |
| L2 cache | 16 MB (P-core) / 4 MB (E-core) |
| RAM | 24 GB |
| OS | macOS (current — equivalent for these numbers) |
| Toolchain | rustc 1.94.0 (rust-toolchain.toml pin) |
| ISA extensions detected | NEON, FEAT_DotProd, FEAT_I8MM, FEAT_BF16, FEAT_FP16, SME, SME2 |

> Earlier revisions of this document labeled the host as "M1" by
> mistake — corrected to M4 on 2026-05-27. Every Mac NEON measurement
> in this file was taken on M4; the numbers themselves are correct,
> only the chip label was wrong. M4 ≈ 1.5–2× faster than M1 in
> single-thread, so reading our Mac figures as "M1" overstates what
> an M1 user would observe.

### antix1 — Pentium-M humble-hardware host

| | |
| --- | --- |
| CPU | Intel Pentium M 2.00 GHz (Banias/Dothan, family 6 model 13) |
| Cores | 1 (no SMT) |
| SIMD ceiling | SSE2 (no SSE3 / SSSE3 / SSE4 / AVX) |
| RAM | 2 GiB |
| OS | Debian 12 bookworm + antiX kernel `5.10.224-antix.1-486-smp` |
| Toolchain | i686-unknown-linux-musl, cross-built on the CI runner |

## 2026-05-27 — Sparsity experiment (negative result, kept on purpose)

BitNet ternary weights are ~42% zero, and a zero contributes nothing
to the dot product, so skipping zeros *seems* like free speed. We
tested it. It isn't — at least not on antix1 with a scalar sparse
kernel.

### Ternary distribution (`willamette analyze`, real 2B)

| value | count | fraction |
| --- | ---: | ---: |
| -1 | 602,163,685 | 28.89% |
| **0** | 879,693,294 | **42.21%** |
| +1 | 602,187,821 | 28.90% |

42% zeros = the theoretical ceiling on what skipping could save.

### Dense i8 vs CSR-sparse scalar matvec (attn_q, 50.4% non-zero)

| host | dense | sparse | result |
| --- | ---: | ---: | --- |
| Mac M4 (NEON) | 0.82 ms | 2.92 ms | sparse **3.55× slower** |
| antix1 (SSE2 i8) | 15.54 ms | 15.75 ms | sparse **1.01× slower (tie)** |

The dense kernel processes 100% of elements but 16-wide SIMD and
regularly; the sparse kernel processes ~50% but one-at-a-time scalar
with irregular gather. On antix1 the "half the work" win and the
"scalar + irregular" loss almost exactly cancel → a tie. On the M4's
fast NEON, dense wins outright.

### What it tells us

* **Skipping zeros does not help on antix1** with a scalar sparse
  kernel — net zero. Not worth the added format complexity here.
* **But the trend is real**: Mac 3.55× → antix1 1.01×. The slower /
  simpler the CPU's SIMD, the more sparse closes the gap. On a CPU
  *below* antix1 (Pentium II, 486, no SIMD), sparse would likely
  *win* — the dense SIMD advantage that beats it here would be gone.
* So sparse isn't dead; it's a **"lowest-tier hardware" optimization**
  that needs a host below antix1 to pay off. Revisit when a Pentium-II
  / SIMD-less machine is in hand (2nd-tier hardware track).

i8 (the dense default, scalar→i8 ≈ 5.4×) stays the antix1 optimum.

## 2026-05-27 — i8 activation kernel (now the x86 default)

profiling (below) showed BitLinear matvec is **96.35%** of decode-step
runtime on antix1, and a chunk of that was the f32 kernel's
per-element `i8 → i32 → f32` sign-extend + convert. The i8 activation
kernel removes that: quantise the activation to int8 once, run the dot
product in integer lanes (16 i8/instr vs 4 f32/instr), no f32 convert
in the inner loop.

### Speed (antix1, Pentium-M, same session)

| Model | f32 SSE2 | i8 SSE2 | speed-up |
| --- | ---: | ---: | ---: |
| synth 110M — matvec | 1.456 ms | 0.668 ms | 2.18× |
| synth 110M — decode | 4.60 tok/s | **10.1 tok/s** | **2.2×** |
| real 2B — matvec | 15.27 ms | 7.19 ms | 2.12× |
| real 2B — decode | 0.19 tok/s | **0.41 tok/s** | **2.15×** |

Cumulative: `scalar → f32 SSE2 (2.49×) → i8 SSE2 (2.2×)` ≈ **5.4×**
over the scalar reference.

### Fidelity — greedy decode is byte-identical

int8 activation is lossy, so the question is whether it changes the
*output*. It doesn't (at least here). Real 2B, prompt
`"The capital of France is"`, 20 tokens, temperature 0:

```
f32: [12366,13,12366,374,264,3363,430,374,3967,369,
      1202,9257,3925,11,7829,11,323,18112,13,1102]
i8:  [12366,13,12366,374,264,3363,430,374,3967,369,
      1202,9257,3925,11,7829,11,323,18112,13,1102]   ← identical
```

Both decode to *"Paris. Paris is a city that is known for its rich
history, culture, and architecture. It"*. The int8 quantisation never
flipped an argmax over 20 steps. The unit test
`tests/bitlinear_sse2_i8.rs` backs this at the matvec level
(cosine > 0.999, max-rel < 5% vs scalar). Caveat: one prompt — not a
perplexity sweep.

### Decision

i8 is now the **x86 default** (`bitlinear.rs` X86Sse2 arm). Unlike
NEON — where i8 was slightly *slower* so f32 stays default — x86 i8
wins on both speed and fidelity. The f32 mask-add kernel stays behind
`--cfg willamette_sse2_f32` for numerical reference. Every prebuilt
x86 binary (x86_64 + i686 musl) now ships the 2.2× kernel.

Effect on the sweet spot: chat speed (≥ 5 tok/s) ceiling on Pentium-M
moves from ~100M to **~220M params**.

## 2026-05-27 — Head-to-head vs llama2.c on the SAME machine

The earlier EXO comparison normalised across two different CPUs
(Pentium II 350 MHz vs our Pentium-M 2 GHz) with a calculated
hardware-correction factor. This section removes that estimate
entirely: Karpathy's `llama2.c` (the engine EXO's demo is built on)
is a single C file, so it compiles and runs on **antix1 itself**.
Same CPU, same SSE2 (gcc `-O3 -march=native` → `__SSE2__` confirmed),
same model size class. Pure architecture + quantization difference.

### Setup

* `llama2.c` @ `karpathy/llama2.c`, `gcc -O3 -march=native -o run run.c -lm`.
* Models from `karpathy/tinyllamas`: `stories15M` (58 MB f32),
  `stories42M` (160 MB), `stories110M` (419 MB).
* `./run <model>.bin -n 256 -i "Once upon a time"`, reading the
  reported `achieved tok/s`.
* Our side: `willamette synth-gguf --preset {small|medium}` then
  `willamette bench --decode-steps 3`, reading decode-step tok/s.

### Result (antix1, Pentium-M, SSE2 both sides)

| Model size | llama2.c (vanilla Llama 2, f32) | willamette (BitNet b1.58, ternary) | willamette advantage |
| ---: | ---: | ---: | ---: |
| 7 M (ours) / 15 M (theirs) | 17.4 tok/s @ 15 M | 103.6 tok/s @ 7 M | params·tok/s: 2.8× |
| 42 M | 6.50 tok/s | — | — |
| **110 M** | **2.51 tok/s** | **4.96 tok/s** | **1.97× faster** |

The clean number is the 110 M row — closest size match, both
measured directly: **BitNet b1.58 + our SSE2 kernel is ~2× faster
than vanilla Llama 2 f32 + gcc-autovectorized SSE2 on the same
21-year-old-class CPU.** The earlier hardware-normalised estimate
(2.6×) was in the right ballpark; the direct measurement lands at
1.97×.

### Why — and a corrected earlier claim

An earlier revision of this doc speculated that "110 M is below the
BitNet sweet spot, so vanilla might win there". **That was wrong, and
the measurement says so.** At 110 M:

* `stories110M.bin` (f32) is 419 MB. Our packed BitNet 110 M is
  70.6 MB — 6× smaller on the bus.
* Both blow past antix1's 2 MB L2, so both are memory-bandwidth
  bound on the decode step — and the 6× smaller weight stream is
  exactly where ternary packing pays off. The BitNet memory
  advantage is already active at 110 M, not only at 2 B.

### Honest caveats

* Our synthetic 110 M has **random ternary weights** — it cannot
  write the coherent TinyStories text `stories110M` produces. We
  compare **throughput only**; tok/s is independent of weight
  *values* (compute is fixed by architecture + size). We make no
  quality claim — see [[feedback-no-fake]].
* The architectures aren't identical: our BitNet b1.58 has the
  extra `attn_sub_norm` / `ffn_sub_norm` RMSNorms vanilla Llama 2
  lacks, so our forward does slightly *more* norm work per layer.
  The 1.97× is achieved despite that, not because of a lighter
  graph.
* Both are single-threaded here (antix1 is 1 core). On a
  multi-core humble host `llama2.c` has OpenMP and we have rayon;
  that comparison is future work.

## 2026-05-27 — Scaling sweep across 4 model sizes

How throughput scales with model size on the same hardware. Built via
`willamette synth-gguf --preset {tiny|small|medium}` (random ternary
weights — see [`src/synth.rs`](../src/synth.rs)). The real 2 B point
is reproduced from the v0.4.1 / v0.5.0 measurements on the official
GGUF.

### Pentium-M antix1 (SSE2, `Kernel::X86Sse2`)

| Preset | Params | Model on disk | matvec | matvec throughput | Decode-step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 0.23 M | 0.1 MB | 0.042 ms (128 × 128) | 390 M e/s | **1576 tok/s** |
| Small | 7.0 M | 7.2 MB | 0.160 ms (256 × 256) | 409 M e/s | **103.6 tok/s** |
| Medium | 110 M | 70.6 MB | 1.47 ms (768 × 768) | 402 M e/s | **4.96 tok/s** |
| Real 2 B | 2 000 M | 1106 MB | 24.3 ms (2560 × 2560) | 269 M e/s | **0.12 tok/s** |

Two structural facts:

1. **matvec throughput is constant (≈ 400 M elements / sec) for
   tiny → medium**, then drops 33 % on the real 2 B model. The drop is
   *not* in our kernel — it's main-memory bandwidth taking over once
   the weight tensors stop fitting in the Pentium-M's 2 MB L2.
2. **`params × tok/s` is constant** at ≈ 500 M params · tok / sec
   right through the sweep. So the BitLinear-dominated forward time
   scales linearly with parameter count on this host. Doubling the
   model exactly halves the tok/s.

### Mac M4 NEON (`Kernel::AArch64Neon`), same model files

| Preset | Params | matvec | Decode-step |
| --- | ---: | ---: | ---: |
| Small | 7.0 M | 0.020 ms (256 × 256) | **916 tok/s** |
| Medium | 110 M | 0.057 ms (768 × 768) | **131 tok/s** |
| Real 2 B | 2 000 M | ≈ 0.6 ms (2560 × 2560) | **7.9 tok/s** |

### Cross-host speed-up (Mac M4 ÷ antix1)

| Preset | Ratio | Comment |
| --- | ---: | --- |
| Small | 8.8× | both fit in L2 — clock + IPC + SIMD width dominate |
| Medium | 26.4× | Mac still cache-fit; antix1 hitting DDR2 bandwidth |
| Real 2 B | 65.8× | Mac's unified LPDDR5 vs antix1's DDR2-533 main memory |

The ratio grows monotonically with model size: the bigger the model,
the more cache hierarchy beats raw compute. This is the **structural
reason** for the "humble hardware × medium LLM" sweet spot to be
narrower than it sounds.

### Sweet-spot redefinition

For "usable chat speed ≥ 5 tok/s on Pentium-M-class SSE2 hardware":
the scaling line `params × tok/s ≈ 500 M` puts the upper bound at
**~ 100 M parameters**. Going larger is not impossible — Real 2 B
works end-to-end — but at 0.12 tok/s it's a demonstration, not a
chat.

For ≥ 1 tok/s: about **500 M parameters** is the ceiling on this
host.

For ≥ 0.1 tok/s ("the user is willing to wait 10 s per token"):
~ 5 B parameters.

This is more *honest* than the original "1 B – 7 B" formulation. On
Pentium-M-class hardware, BitNet 1.58 + SSE2 gets you a 100 M model
at chat speed, a 500 M model at "slow but usable", and a 5 B model
at "demonstration". Newer SIMD (AVX2 / AVX-512) or multi-core
(Pentium 4 HT / Atom dual / RPi 4) shifts every threshold roughly an
order of magnitude up.

### Where BitNet 1.58 actually pulls ahead vs. vanilla Llama 2

EXO Labs' Pentium II 350 MHz @ 260 K params = 35.9 tok/s. That gives
us their **vanilla-Llama-2 efficiency**: 260 K × 35.9 ≈ 9.3 M params
· tok/sec. Our Pentium-M (SSE2) Medium runs at 110 M × 4.96 ≈ 544 M
params · tok/sec. Correcting for hardware (Pentium-M is roughly 22.8×
the Pentium II raw work / sec: 5.7× clock × 4-wide SSE2), our
**BitNet 1.58 + SSE2 stack is ~ 2.6× more efficient than vanilla
Llama 2 + no-SIMD on the same params per cycle**.

Concretely: on a Pentium II 350 MHz, our 110 M BitNet path would
predict 0.22 tok/s, against EXO's vanilla 110 M extrapolation of
~ 0.085 tok/s. Same hardware, 2.6× tokens per second from
architecture + SIMD.

That 2.6× factor is small in absolute terms but it's the right unit
to measure ourselves in: every architectural change should move it,
not the raw tok/s figure.

## 2026-05-25 — v0.5.0-mvp SSE2 kernel landed

### antix1 — Pentium-M SSE2 (`Kernel::X86Sse2`)

First Stage 6-B measurement. Same host (antix1, Pentium-M 2.0 GHz),
same model file, same bench command. Built locally via
`cargo build --release` (7m 13s on 2 GB RAM, no OOM); the published
`v0.5.0-mvp` cross-compiled artifact matches this binary.

```
./target/release/project-willamette bench --model ~/models/ggml-model-i2_s.gguf --decode-steps 3
```

| Measurement | Value | vs. scalar |
| --- | --- | --- |
| `dispatch::active_kernel().label()` | `i686 SSE2` | (was `i686 scalar`) |
| BitLinear matvec (attn_q, 2560 × 2560 ternary) | **24.3 ms** | 2.49× faster |
| BitLinear matvec throughput | 269 M elements / sec | 2.49× |
| Single-token forward (30 layers, no cache) | **8.87 s** | 2.45× faster |
| Decode-step forward (KV cache, avg of 3) | **8.15 s** | 2.66× faster |
| Decode-step throughput | **0.12 tok/s** | 2.4× |

Parity: `cargo test --test bitlinear_sse2` runs the 8 layer-0
BitLinear weights end-to-end against the scalar reference;
`max |Δ| < 1e-2` holds across all of them (same tolerance the NEON
test uses). Verified on antix1, 8/8 pass.

#### Why 2.5×, not 8× (the SIMD width)?

The pure-SSE2 i8 → i32 → f32 sign-extension sequence
(`unpacklo_epi8` + `srai_epi16` + `unpacklo_epi16` + `srai_epi32`
+ `cvtepi32_ps`) costs four μops per 4-element chunk; the actual
mask-add is one μop. The kernel is also memory-bandwidth bound
on the Pentium-M's modest 533 MT/s DDR2 — 269 M f32 reads per
second × 4 bytes = 1.08 GB/s sustained, which is in the right
ballpark for a single in-order issue port pulling f32 input plus
ternary weight bytes through L1.

The next obvious step is an **i8 activation path**: quantize `x`
once per matvec call into `i8`, then use the `pmullw` /
`pmaddwd` pattern to compute the dot product in 8-wide i16 lanes,
producing i32 partial sums. That mirrors the NEON `vmull_s8`
kernel and roughly halves the L1 traffic for the activations. We
don't ship it in v0.5.0 — first cut keeps the same numerical
shape as scalar and proves the dispatch route works.

### antix1 — Pentium-M scalar (`Kernel::Scalar`, v0.4.1-mvp)

Kept as the "before" reference for any future kernel.

## 2026-05-25 — v0.4.1-mvp baselines

### Apple M4 — NEON (`Kernel::AArch64Neon`)

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

#### M4 NEON ÷ Pentium-M scalar

The two hosts are different in four independent dimensions (clock,
IPC, SIMD width, memory bandwidth), so a single ratio understates the
SIMD contribution. For the record:

* **Decode-step ratio**: 7.9 / 0.05 ≈ **158× faster on M4**.
* **BitLinear matvec ratio**: M4's per-matvec time is roughly 5 ms
  (back-calculated from the 7.9 tok/s figure across 30 layers × ~6
  matvecs/layer of similar shape), versus 60.5 ms scalar → **≈ 12×**
  on the matvec alone. The remaining factor of ~13× comes from clock
  (2.0 GHz → ~3.2 GHz P-core), IPC (in-order P-M vs out-of-order
  Firestorm), memory bandwidth, and rayon multi-core scheduling on
  the M4 against antix1's single core.

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
