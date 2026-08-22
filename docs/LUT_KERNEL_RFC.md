# LUT BitLinear Kernel — Design RFC

*Status: partially implemented. The scalar LUT and SSE2-only runtime
dispatch shipped in v0.10.0-mvp; the SSSE3 `pshufb` follow-up remains
deferred and measurement-gated.*

This document is the design + acceptance contract for adding a
table-lookup BitLinear matvec kernel to Willamette. It exists to
keep the implementation honest: every claim made here is testable,
and the implementation either matches the contract or the RFC gets
revised, not the code silently.

## 1. Why this exists

The 2026-05-30 measurement cycle on mbp2012 (Ivy Bridge i7-3520M)
established three facts that together demand a LUT path:

1. **Memory + cache pressure dominates, not raw FLOPs.** rayon
   1-thread and 4-thread runs are within run-to-run noise on a
   4-thread CPU. The matvec moves ~6.45 GB/s of i8 traffic per
   token, ~40 % of the loaded DDR3-1600 effective ceiling. Adding
   cores does not help; cutting *work per byte read* does.
2. **bitnet.cpp does not cover the sub-AVX2 band.** Three build
   attempts on Ivy Bridge (default `GGML_AVX2=ON` → SIGILL, AVX2
   off MAD → garbage `!!!!!`, AVX2 off TL2 → compile error) all
   failed. Every host below Haswell (and antix1's Pentium-M is far
   below that) has no working bitnet.cpp CPU binary. A Willamette
   LUT kernel that compiles on SSE2 hosts therefore opens the
   *first* LUT-accelerated BitNet path on this hardware band.
3. **A table-lookup *replaces* per-element multiply-add with a
   single index + load.** Since BitNet weights are ternary
   (`-1 / 0 / +1`) and packed 2-bit, the matvec inner loop is
   already addition / sign-flip; an LUT collapses several such
   ternary contributions into one pre-summed scalar. Whether that
   wins on real hosts is the empirical question this RFC commits
   to *measuring*, not assuming.

Read together: the LUT kernel is justified iff measurement (step 1
of the migration below) shows ≥ 1.3× matvec speed-up on at least
one of {antix1 SSE2, mbp2012 SSE2}. Anything below that is a
negative result, recorded the same way the KV i4 prototypes were,
and the RFC closes without merging the kernel.

## 2. What this covers

* A new BitLinear matvec implementation: `src/model/bitlinear_lut.rs`
* Runtime dispatch entry from `src/model/bitlinear.rs` and a
  `Kernel::Lut*` variant in `src/model/dispatch.rs`.
* Two planned LUT variants:
  * **Scalar LUT** — pure Rust, no SIMD. Portable; runs on
    Pentium-M antix1 (SSE2 only) without `pshufb`. First target.
  * **SSSE3 LUT** — `pshufb` accelerated. Runs on mbp2012 and
    everything from Core 2 Conroe onward. Second target, only
    landed if (a) scalar LUT clears the speed-up gate and (b)
    `pshufb` measurably extends the win.
* Acceptance numbers measured on the same machines benchmarks
  currently live on: antix1 (Pentium-M) + mbp2012 (Ivy Bridge) +
  Mac M4 NEON (reference).

## 3. What this does *not* cover

* **NEON LUT.** Apple Silicon already runs the f32-input NEON
  kernel at 7.9 tok/s; the bandwidth ratio on M4 is different,
  so adding a NEON LUT is a separate measurement task. Out of
  scope for this RFC.
* **AVX2 LUT (`vpshufb` 256-bit).** The hosts that benefit from
  AVX2 are precisely the hosts bitnet.cpp already covers. Not
  the gap we are filling.
* **Replacing the i8 activation kernel.** The i8 BitLinear stays
  the default until LUT proves a measurement-supported win. Even
  then it stays as a fallback for hosts where LUT loses (some
  almost-certainly exist — small caches, wide vectors).
* **Per-group / Q4-style activation quantisation.** Independent
  question.

## 4. Design

### 4.1 LUT shape

The activation tile size `K_TILE` and the ternary group size `T_GROUP`
together fix the table:

```
table[v][s] = Σ_{i=0..T_GROUP-1}  decode(s, i) * v[i]
```

where:
* `v` is one `T_GROUP`-wide slice of the i8-quantised activation
  vector (one cell of the table per slice);
* `s` is the packed 2-bit ternary code of `T_GROUP` weight elements
  (`0..3^T_GROUP - 1` after we mask out the I2_S "unused" code);
* `decode(s, i)` returns `-1 / 0 / +1` per the I2_S code → ternary
  map in `docs/I2_S_LAYOUT.md`.

Concretely, with `T_GROUP = 5`:
* `3^5 = 243` table rows
* Each row is one i32 (sum of up to 5 i8 activations, fits well)
* Per activation slice: 243 × 4 B = **972 B → L1-resident** even
  on Pentium-M's 16 KiB L1d.

A whole `BitLinear` row of length 2560 → 2560 / 5 = 512 activation
slices, each producing one i32 table that is consumed 30 layers ×
2560 / 5 weight-slice times before being thrown away. The expected
trade is "build the table once per activation slice, amortise the
work across all weight rows in this slice".

`T_GROUP = 5` is *not* fixed — it is a free parameter that the
benchmark in step 2 will sweep over {4, 5, 8}. The RFC's
acceptance gate is "best `T_GROUP` ≥ 1.3× over i8 dense"; the
exact winning value is whatever the measurement says.

### 4.2 I2_S decode

The 2-bit code → ternary value map (`docs/I2_S_LAYOUT.md` § 3) is
the only piece of upstream byte semantics this kernel touches.
Building the table from the I2_S code requires that map exactly;
re-deriving it inside the LUT module would be a duplication that
[[feedback-no-fake]] cancels. Implementation pulls the helper
out of `src/gguf/tensor.rs` (or wherever the canonical decode
lives at implementation time).

### 4.3 Sub-norm + scale

BitLinear has two scalars per row: `i2_scale` (the model-wide
fp16 → f32 scale of the weight tensor) and the per-row sub-norm
(F32 per-element). Neither belongs in the LUT — they multiply the
accumulated table sum *after* the matvec. The LUT module's API
mirrors `bitlinear_i2s_matvec_f32` exactly so the call site does
not change.

### 4.4 Dispatch surface

```rust
// dispatch.rs
pub enum Kernel {
    Scalar,
    AArch64Neon,
    X86Sse2,
    X86Sse2LutScalar,   // new; chosen only when feature flag is set + measured
    X86Ssse3Lut,        // new; chosen only when `pshufb` is detected
    // …
}
```

The scalar LUT is compiled on x86/x86_64 and selected at runtime for
SSE2 hosts without SSSE3. SSSE3+ hosts retain SSE2 i8. The SSSE3 LUT
remains unimplemented; `willamette_lut` is historical prototype cfg
terminology rather than the production dispatch switch.

## 5. Migration steps

Five steps. Steps 4–5 only run if step 1 measurement says the
kernel earns its keep.

### Step 1 — Scalar LUT prototype + measurement *(executed 2026-05-30)*

Add `bitlinear_lut.rs` with a pure-Rust scalar LUT for one
`T_GROUP` choice. Wire it behind `--cfg willamette_lut` to a
separate test binary; do **not** touch `dispatch.rs` yet. Run
the same `cargo bench` matvec measurement we already use on
antix1 + mbp2012.

**Gate**: scalar LUT matvec ≥ 1.3× faster than `bitlinear_i2s_matvec_f32_scalar`
on at least one of antix1 / mbp2012, with byte-identical Stage 5-E
greedy output. Below 1.3× → record as a negative result in
`BENCHMARKS.md` (the way the KV i4 prototypes were recorded) and
close this RFC without merging.

**Outcome (2026-05-30, `BENCHMARKS.md` § "2026-05-30 — LUT step-1
prototype measurement")**: gate **PASSES** on every measured
host:

| Host | scalar LUT vs scalar BitLinear | scalar LUT vs production default |
| --- | ---: | ---: |
| Mac M4 (NEON default)   | 14.09× | (sanity only — NEON is default) |
| mbp2012 (SSE2 i8 default 1.05 ms) | 9.41× | **0.40×** (SSE2 i8 wins by 2.5×) |
| **antix1 (SSE2 i8 default 37.08 ms)** | **8.54×** | **5.29×** (LUT wins decisively) |

The sampled antix1 `attn_q` result justified testing dispatch, but it
did not establish a whole-forward or complete-token speedup. The
single-shape comparison was later found insufficiently representative;
mbp2012 still loses with LUT because its Ivy Bridge SIMD handles the
dense path efficiently.

**Step-4 gate recalibration (kept from the original step-1 doc
write-up, still correct)**: any SSSE3 `pshufb` LUT has to beat
SSE2 i8 on the same host, which means ≥ 2.5× over scalar LUT
on mbp2012 (1.05 ms / 2.65 ms). The original "≥ 1.5× over scalar
LUT" wording is dropped because it would have let step 4 land a
regression.

Parity (`tests/bitlinear_lut.rs`, `max|Δ| ≤ 1e-2`): **4/4 pass**
on Mac aarch64. antix1 + mbp2012 inherit the same parity from
the algorithmic equivalence; integration parity on those hosts
lands with step 3 (the byte-identical Stage 5-E gate).

### Step 2 — `T_GROUP` sweep

Run scalar LUT with `T_GROUP ∈ {4, 5, 8}` on the same hosts.
Pick the smallest `T_GROUP` whose matvec time is within 5 % of
the fastest. This minimises L1 pressure on antix1 (16 KiB L1d)
while keeping the win.

### Step 3 — Dispatch integration

**Outcome:** `Kernel::X86Sse2ScalarLut` shipped in v0.10.0-mvp.
SSE2-without-SSSE3 selects scalar LUT; SSSE3+ selects SSE2 i8. No CLI
opt-in is required. The bench banner reports the selected kernel, and
Stage 5-E greedy parity is preserved.

### Step 4 — SSSE3 `pshufb` LUT *(only on hosts that detect SSSE3)*

`pshufb` does a parallel 16-byte table lookup; this collapses
the inner table read of step 1 to one instruction per 16
activations. Same correctness gate.

**Speed gate — revised after the step-1 measurement, 2026-05-30**:
**SSSE3 LUT must beat the existing SSE2 i8 production kernel on
the same host (mbp2012)**, not the scalar LUT. Concretely:
matvec must come in under ~1.05 ms on mbp2012 (the v0.9.0
production SSE2 i8 timing), which works out to ≥ 2.5× over
scalar LUT given the 2.652 ms step-1 number. Below that, the
honest move is to **close this RFC with a recorded negative
result** the way the KV i4 prototypes were closed; sub-AVX2
hosts keep SSE2 i8 as their local optimum and the project
moves on to a different track.

The original "≥ 1.5× over scalar LUT" gate is preserved here
only as a historical marker — it was calibrated against an
imagined scalar baseline, not against the production kernel
that already exists.

### Step 5 — Documentation and release *(completed without a speed claim)*

The historical cached-forward measurement on antix1 showed no
measurable gain. Its original explanation that BitLinear was only
~10% of the step was later disproved: stage timing found all seven
BitLinear matvecs consume about 99% of cached-forward time. That
instrumented scope excludes lm-head and argmax. The prototype's cold,
single-`attn_q` 5.29× comparison was not representative of all matrix
shapes or steady complete-token execution.

Per [[feedback-no-fake]], v0.10.0-mvp did not claim a 5×
Pentium-M speedup. The dispatch integration shipped as a correctness
and consistency change (the pure-Rust
LUT path is easier to maintain than the SSE2 i8 intrinsics
path; both produce byte-identical Stage 5-E greedy output;
parity tests cover the LUT path on every x86 build).

SSSE3 `pshufb` remains deferred. It must beat SSE2 i8 on
representative FFN shapes and improve complete-token throughput before
landing; the historical single-`attn_q` result is not sufficient.

The RFC itself stays in `docs/` as the durable record of why
this kernel exists, what it promised, and where the
measurement disagreed with the projection.

## 6. Tests

* `tests/bitlinear_lut.rs` — same shape as `tests/bitlinear_sse2_i8.rs`:
  byte-parity vs scalar BitLinear on every layer-0 weight, cosine
  ≥ 0.999 numeric tolerance, skip gracefully if the real model
  isn't present.
* `src/model/bitlinear_lut.rs::tests` — table build correctness
  (every `(slice, code) → expected sum` for a small hand-checked
  fixture), boundary cases (`T_GROUP` not dividing the row length,
  all-zero slice, all-`+1` slice, all-`-1` slice).
* No new model file required — both tests reuse the real BitNet
  2B GGUF for the integration check.

## 7. Risks (and what would falsify the design)

* **Table fits L1 only with very small `T_GROUP`.** If `T_GROUP = 5`
  + table rebuild cost > i8 matvec savings, step 1 fails. The RFC
  treats this as a negative measurement and stops. Not a bug.
* **`pshufb` LUT loses on Ivy Bridge cache.** 16-byte parallel
  lookup is fast but only if the table cache-line residency holds
  across 16 lookups. Measurement-gated in step 4.
* **Bandwidth-bound stays bandwidth-bound.** If matvec on mbp2012
  is genuinely DDR3-pinned, even a perfect 0-FLOP LUT can't
  exceed ~25 GB/s ÷ 1.6 MB / token / BitLinear = ~16 BitLinears /
  sec / dim — i.e. a hard cap not very far above what i8 already
  reaches. The 1.3× gate is calibrated to reject "won by a
  rounding error" wins for this reason.

## 8. Out of scope (intentional)

* Per-activation-token absmax quantisation *combined* with LUT.
  Possible but doubles the design space. Not part of this RFC.
* Replacing the activation i8 quantisation. Stays untouched.
* Changing the BitLinear matvec contract (input / output shape /
  scale handling). Stays exactly as `bitlinear.rs` documents.
* `bitnet.cpp` LUT compatibility / interop. We do not need
  bitnet.cpp's TL1/TL2 binary layout; we need a working LUT for
  the sub-AVX2 band, and bitnet.cpp doesn't ship one. If the
  TL2 layout happens to be a useful reference, fine; if it
  conflicts with our shape, we ignore it.

## 9. Disposition

The scalar prototype and SSE2-only dispatch shipped in v0.10.0-mvp.
The SSSE3 step remains deferred under the representative-shape and
complete-token gates above. This document is retained as the design
and corrected outcome record.
