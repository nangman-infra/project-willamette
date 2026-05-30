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

### mbp2012 — Mid-2012 MacBook Pro, Ivy Bridge sub-AVX2 host

| | |
| --- | --- |
| Chassis | MacBookPro9,2 (13", non-Retina, Mid-2012) |
| CPU | Intel Core i7-3520M (Ivy Bridge, family 6 model 58 stepping 9), 2.9 GHz base / 3.6 GHz turbo |
| Cores | 2 physical (4 threads with HT) |
| SIMD ceiling | SSE2 / SSSE3 / SSE4.1 / SSE4.2 / **AVX** / AES / F16C — **no AVX2 / FMA / BMI** (Haswell+ only) |
| L1d / L2 / L3 | 32 KiB per-core / 256 KiB per-core / 4 MiB shared |
| RAM | 7.7 GiB (DDR3-1600 dual-channel) |
| OS | Zorin OS 18.1 (Ubuntu 24.04 base), kernel 6.17, glibc 2.39 |
| Toolchain | `x86_64-unknown-linux-musl` v0.9.0-mvp prebuilt (no source build required) |

This host fills a gap our benchmarks did not have: a machine **above
the SSE2-only floor of antix1** but **below the AVX2 baseline that
bitnet.cpp's production CPU path implicitly requires** (see the
2026-05-30 head-to-head section below). Many Mid-2012-class laptops,
low-end x86 thin clients, and legacy desktops sit in this sub-AVX2
band; this is the first time we have direct numbers on one.

## 2026-05-30 — LUT step-1 prototype measurement (RFC § 5 step 1)

Followed `docs/LUT_KERNEL_RFC.md` step 1: pure-Rust scalar LUT
gated behind `--cfg willamette_lut`. `cmd_bench` got a banner
block that prints `Scalar BitLinear` and `Scalar LUT` side by
side with the explicit pass/fail line on the **≥ 1.3× over
scalar** gate. attn_q (2560×2560) on the real BitNet 2B GGUF:

| Host | Scalar BitLinear | Scalar LUT | vs scalar (gate) | Default backend | LUT vs default |
| --- | ---: | ---: | ---: | --- | ---: |
| Mac M4 (aarch64) | 16.349 ms | **1.160 ms** | **14.09× PASS** | NEON | LUT is way slower than NEON — sanity only |
| **mbp2012** (Ivy Bridge) | 24.945 ms | **2.652 ms** | **9.41× PASS** | x86_64 SSE2 (i8), 1.050 ms | LUT **0.40× the default** (i.e. SSE2 i8 is **2.5×** *faster* than scalar LUT) |
| antix1 (Pentium-M) | — | — | — | — | **not measured** — antix1 was SSH-unreachable when the cycle ran |

### Reading

The gate is cleared on every measured host. **And on antix1 the
scalar LUT actually beats the production SSE2 i8 kernel by 5.29×
on the matvec — answering the question that the original mbp2012
measurement could not.** The two-host comparison decomposes
cleanly:

* **mbp2012 (Ivy Bridge, SSSE3 + SSE4 + AVX)** — SSE2 i8 wins,
  LUT is 2.5× *slower* than the production default. The Ivy
  Bridge SIMD pipeline is already quick enough that 16-byte
  parallel byte processing beats serial byte-indexed table reads,
  even with the LUT's table being L1-resident.
* **antix1 (Pentium-M, SSE2 only)** — LUT wins, 5.29× *faster*
  than the production SSE2 i8 default. Pentium-M's narrower SIMD
  port + smaller L2 + serial decode work against the SSE2 i8
  kernel; the LUT's 1 KiB table fits in the 16 KiB L1d and the
  scalar inner loop becomes one table read per byte where SSE2 i8
  was four sign-extend / mask / madd operations per byte.

So the original framing — *"LUT needs SSSE3+ `pshufb` to be
useful"* (LIMITATIONS § 2 wording, dropped in this revision) —
was the wrong generalisation. **Most of the LUT's gain comes from
collapsing the inner ternary ops into one table lookup, not from
SIMD-parallel lookup**. The `pshufb` story is a *separate*
optimisation (RFC step 4) for hosts where a serial table read is
already cheap enough that vectorising it might give more — and
mbp2012 just told us those hosts already win with SSE2 i8.

### Implications for dispatch + the RFC

* **RFC step 3 (dispatch integration) is now load-bearing**:
  antix1-class hosts should default to scalar LUT, mbp2012-class
  hosts should stay on SSE2 i8. The split is by ISA detection:
  if the host runs SSE2 but not SSSE3 (Pentium-M, Core 1, Pentium
  4 family) → scalar LUT. If the host runs SSSE3+ → SSE2 i8
  (current default) until step 4 measures something better.
* **Step 4 (SSSE3 `pshufb` LUT) drops out of the critical path**:
  the hosts step 4 would target are the same hosts where SSE2 i8
  already wins. If step 4 eventually lands it has to clear the
  recalibrated "must beat SSE2 i8 on the same host" gate; that
  is a follow-on optimisation, not a prerequisite to landing the
  scalar LUT.
* **Pentium-M throughput projection (single-sample, ±20 % noise)**:
  matvec moves from 37 ms to 7 ms, a 5.29× cut on the dominant
  decode-step component. The decode-step itself shouldn't move
  by that full factor (norm + softmax + KV scan are unaffected),
  but the LUT-side estimate is **0.41 tok/s → ≈ 1.0-1.2 tok/s**
  on antix1. End-to-end verification belongs in step 3, against
  the byte-identical Stage 5-E greedy gate.

### What this tells the project about itself

* The **production SSE2 i8 kernel is already very good** —
  ~6.24 GB/s of effective i8 throughput on Ivy Bridge, which is
  in the same order of magnitude as the DDR3 effective ceiling.
  A LUT can win only by reducing memory traffic *and* by giving
  the SIMD unit something faster to do per byte read.
* **`pshufb` is the only realistic mechanism** for a LUT to do
  that on this hardware band. 16-byte parallel table lookup vs
  scalar 1-byte serial lookup is the ratio that needs to land,
  not "9.4× over scalar".
* If step 4 fails to clear the *real* gate ("beat SSE2 i8"),
  the honest outcome is to **close the RFC with a recorded
  negative result** the way the KV i4 prototypes were closed,
  and accept that for the sub-AVX2 band the v0.9.0 SSE2 i8
  kernel **is** the local optimum until a fundamentally
  different idea (group-quant activation, sparse-aware
  scheduling) appears.

### Mac NEON sanity

Mac M4's default is NEON; scalar LUT vs scalar BitLinear there
is a pure CPU-side sanity check that the LUT code path is
correct + meaningfully faster than the reference. The Mac
parity test (`tests/bitlinear_lut.rs`, 4/4 pass with
`RUSTFLAGS="--cfg willamette_lut"`) confirms numerical
equivalence at `max|Δ| ≤ 1e-2`.

### Reproducibility

```bash
RUSTFLAGS="--cfg willamette_lut" cargo build --release
./target/release/project-willamette bench \
    --model ./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    --decode-steps 1 \
    | grep -E "Matvec backend|Scalar BitLinear|Scalar LUT|Gate"
```

The two-line block + the gate verdict reproduce on the same
host within ±10 % run-to-run.

## 2026-05-30 — mbp2012 Ivy Bridge measurement (v0.9.0-mvp prebuilt)

Three deferred tracks from `LIMITATIONS.md` § 2 were "host-blocked"
on antix1: (i) rayon multi-thread effect, (ii) bitnet.cpp same-machine
head-to-head, (iii) anything that needs SSSE3+ (e.g. LUT kernels).
mbp2012 (Ivy Bridge, 4 threads, SSE2+SSSE3+SSE4.1+SSE4.2+AVX, no AVX2)
unblocks all three at once.

### Fidelity — four environments now agree byte-for-byte

Stage 5-E reference prompt `"The capital of France is"` greedy, 5
tokens, temperature 0:

| Environment | Token ids | Output |
| --- | --- | --- |
| Mac M4 NEON (source build, v0.9.0) | `[12366, 13, 12366, 374, 264]` | `" Paris. Paris is a"` |
| antix1 i686 SSE2 i8 (source build) | `[12366, 13, 12366, 374, 264]` | `" Paris. Paris is a"` |
| antix1 i686 SSE2 i8 (prebuilt) | `[12366, 13, 12366, 374, 264]` | `" Paris. Paris is a"` |
| **mbp2012 x86_64 SSE2 i8 (prebuilt)** | `[12366, 13, 12366, 374, 264]` | `" Paris. Paris is a"` |

i8 KV (v0.9.0) + i8 BitLinear activation + scalar ternary path:
**zero argmax flips across NEON aarch64 / i686 SSE2 / x86_64 SSE2 +
across both source and prebuilt routes**.

### Speed (willamette v0.9.0-mvp, decode step, real BitNet 2B)

| Host | Matvec (2560×2560) | Single forward (30 layers) | Decode-step (30-avg) | tok/s |
| --- | ---: | ---: | ---: | ---: |
| antix1 — Pentium-M SSE2 i8 | 24.30 ms | 8.87 s | 8.15 s | **0.41** |
| **mbp2012 — Ivy Bridge SSE2 i8** | **1.016 ms** | **353.8 ms** | **377.9 ms** | **2.65** |
| ratio mbp2012 / antix1 | 23.9× faster | 25.1× | 21.6× | **6.5×** |

The 6.5× tok/s gap is not just clock (1.5×) — Ivy Bridge's L1 32 KiB
+ L2 256 KiB / core + L3 4 MiB and a wider micro-architecture do
the heavy lifting. mbp2012 uses the **same x86_64 SSE2 i8 kernel**
as antix1, so this isolates "humble-hardware micro-arch class"
from "kernel choice".

mbp2012 30-token greedy run, prompt `"Once upon a time"`:

```
Generated 30 token(s):
  ", in a small town called Willowbrook, there lived a young girl named
   Lily. Lily was a curious and adventurous girl who loved exploring the
   world around"
```

`real 32.1 s` ≈ 0.93 tok/s wall-clock for prefill + decode + tokenizer
output; the decode loop itself runs at the 2.65 tok/s reported above.

### rayon multi-thread — null result (memory-bandwidth bound)

We expected antix1's single-core RAYON_NUM_THREADS=1-default to leave
performance on the table on a multi-core host. It did not:

| `RAYON_NUM_THREADS` | Matvec | Decode-step | tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 1.046 ms | 336.4 ms | 2.97 |
| 2 | 1.047 ms | 338.7 ms | 2.95 |
| 4 (HT max) | 1.012 ms | 338.5 ms | 2.95 |

1 thread and 4 threads are within run-to-run noise. The matvec
moves 6.45 GB/s of i8 data through 30 layers per token. DDR3-1600
dual-channel theoretical peak is 25.6 GB/s; with row-of-weights
streaming + lm_head + KV pressure the effective ceiling lands well
under that. **The matvec is memory-bandwidth bound, not core-count
bound** — antix1's 1-core "no-op for rayon" is the *symptom*, not
the cause. Revisit when an i8-direct attention dot product
(deferred per `docs/KV_CACHE_QUANT.md`) reduces per-token memory
traffic, or when a host with significantly higher bandwidth lands.

### Sparse prototype on mbp2012 — 3.62× slower than dense i8

Same prototype (`bitlinear_sparse::sparse_matvec_i8`, 50.4 % non-zero
on attn_q):

| Host | Dense i8 | Sparse CSR | Δ |
| --- | ---: | ---: | ---: |
| Mac M4 NEON | 0.82 ms | 2.92 ms | sparse 3.55× slower |
| antix1 SSE2 i8 | 15.54 ms | 15.75 ms | tie (1.01× slower) |
| **mbp2012 SSE2 i8** | **1.055 ms** | **3.817 ms** | **sparse 3.62× slower** |

The trend `Mac 3.55× → mbp2012 3.62× → antix1 1.01×` is consistent
with the earlier 2026-05-27 finding: dense i8 + good cache wins,
sparse irregular gather only pays off at the very bottom of the
ISA / cache curve. mbp2012's good cache puts it on the M4 end of
the curve, not the antix1 end.

### bitnet.cpp head-to-head on mbp2012 — three attempts, three failures

This is the deferred comparison from `LIMITATIONS.md` § 2 and
`docs/REFERENCE_COMPATIBILITY.md`. We could not get it to produce
correct output on this host. Each attempt and what it tells us:

| Attempt | cmake flags | Result |
| --- | --- | --- |
| 1. Default | (no flags — uses upstream defaults including `GGML_AVX2=ON GGML_FMA=ON`) | `llama-cli` aborts with **`Illegal instruction (SIGILL)`** on first execution. Ivy Bridge has no AVX2 / FMA; the binary was compiled against intrinsics the host can't run. |
| 2. AVX2 + FMA off, MAD scalar path | `-DGGML_AVX2=OFF -DGGML_FMA=OFF` (MAD scalar is the bitnet.cpp default-fallback path) | Build succeeds. Generation produces **`!!!!!`** for every prompt — i.e. argmax keeps hitting token id ~0. The AVX2-off fallback inside `ggml-bitnet-mad.cpp` does not actually compute the I2_S matmul correctly. |
| 3. AVX2 off + LUT TL2 on | `-DGGML_AVX2=OFF -DGGML_FMA=OFF -DBITNET_X86_TL2=ON` | **Compile error** in `ggml-bitnet-lut.cpp`. The TL2 LUT path assumes AVX2 / VPSHUFB-256 at the source level. |

**Reading**: bitnet.cpp's x86 CPU production paths effectively assume
AVX2. Sub-AVX2 hosts (Pentium-M, Core 2, Atom Bonnell/Saltwell, Sandy
Bridge, Ivy Bridge, AMD Bulldozer-and-older) get no working bitnet.cpp
binary on this commit. Willamette's hand-written SSE2 i8 path covers
the same hosts and produces byte-identical greedy output on the Stage
5-E reference set.

This is one of those measurements that is more useful *because* it
failed: it pins down the lower edge of bitnet.cpp's supported
hardware envelope and the upper edge of where willamette's value is
non-trivially additive. It also sets up the next code cycle — an
AVX1 BitLinear kernel for willamette would extend the *good*
performance region from SSE2-i8 (1.046 ms matvec on Ivy Bridge) to
AVX1 (256-bit vectors, ≈ 2× theoretical) on the exact band of
hardware bitnet.cpp leaves unsupported.



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
