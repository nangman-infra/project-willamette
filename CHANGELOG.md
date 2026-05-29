# Changelog

All notable changes to Project Willamette are recorded here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
project version increments follow [SemVer](https://semver.org/) — the
`-mvp` suffix marks releases that still treat the runtime as an MVP
rather than a stabilised library.

### Versioning policy (carried forward)

| Change | Bumps |
| ------ | ----- |
| Bug fix (same intent, corrected result) | `patch` |
| Public CLI / API addition (new subcommand, new public function) | `minor` |
| Public CLI / API change or removal (breaking) | `major` |
| Internal-only (CI, refactor, docs, clippy cleanup) | _no bump; only `[Unreleased]` notes_ |
| Model compatibility breakage (new ggml_type required, new tokenizer pre mandatory) | `major` or `minor`, sized by user impact |

The `-mvp` suffix is kept while iterating in `v0.1.x` and `v0.2.x`.
It will be dropped on the first release we feel comfortable advertising
as a stable library — at which point the next tag becomes `v0.3.0`
(or `v1.0.0` if there is also a public API guarantee).

## [Unreleased]

_No changes yet._

## [v0.9.0-mvp] — 2026-05-29

Minor release. **KV cache moves from f32 to per-token absmax i8**,
shrinking the dominant piece of dynamic memory by ~3.97×. The
prior f32 cache cost 150 KB per token on BitNet 2B; the new i8 cache
costs 37.7 KB. At full 4096-token context the resident KV cache
drops from ~614 MB to ~154 MB — about 460 MB freed on every host,
which translates on antix1 (Pentium-M, 2 GB) to a practical chat-
history ceiling lift from ~3 K to ~13 K tokens (past the model's
own 4096-position embedding limit). Design + measurements:
[`docs/KV_CACHE_QUANT.md`](docs/KV_CACHE_QUANT.md).

### Added

* `KVCache::read_into(layer_idx, &mut Vec<f32>, &mut Vec<f32>)` —
  caller-managed dequant buffers. Production caller
  (`cached_forward::forward_one_layer`) allocates one pair per
  forward call and reuses across the 30 transformer blocks.
* `KVCache::resident_bytes()` — actual i8 + scale bytes resident,
  per cache instance. Used by `ChatEngine::estimate_kv_cache_bytes`.
* `docs/KV_CACHE_QUANT.md` — full design doc covering the memory
  math, the API change, the fidelity contract, and the out-of-scope
  schemes (i8-direct attention dot, Q4 group quant, per-head
  scales) with the reason each was deferred.

### Changed

* `src/model/kv_cache.rs` — internal storage now `Vec<i8>` +
  `Vec<f32>` (scales) for K and V independently. `append()` does
  per-token absmax quantisation; the worst-case per-element error
  is `absmax / 254`. Zero vectors round-trip exactly.
* `src/model/cached_forward.rs::forward_one_layer` — replaces the
  `let (cached_k, cached_v) = cache.read(...)` borrow with
  `cache.read_into(layer_idx, scratch_k, scratch_v)`. Signature
  grows by two `&mut Vec<f32>` parameters; the surrounding
  `forward_with_cache_progress` allocates the scratch pair once
  per call.
* `src/chat/engine.rs::estimate_kv_cache_bytes` — now reads
  `self.cache.resident_bytes()` instead of the old
  `layers · kv_dim · pos · 4 · 2` formula (which is wrong by 4×
  under the new layout).

### Removed

* `KVCache::read(layer_idx) -> (&[f32], &[f32])`. There is no
  contiguous f32 slice in the new layout to borrow; the replacement
  is `read_into`. Production had one call site, all migrated.

### Fidelity

The cached forward is no longer bit-equal to the no-cache
reference — i8 round-trip drifts the hidden state on the order of
`absmax / 254`. The new contract is **cosine ≥ 0.999** on the
post-`output_norm` hidden (enforced by `tests/kv_cache.rs::cache_*`)
plus **byte-identical greedy token-id sequence** vs the no-cache
path (enforced by
`tests/kv_cache.rs::greedy_with_cache_matches_greedy_no_cache_for_2_steps`).
The Stage 5-E reference prompt `"The capital of France is"`
generates `[12366, 13, 12366] = " Paris. Paris"` — byte-identical
on Apple M4 (NEON) and antix1 (i686 SSE2 i8). i8 KV did not flip
any argmax across the reference set.

[[feedback-no-fake]]: bit-equality vs the no-cache path is *not*
claimed any more. The fidelity contract above is what is measured
and enforced.

### Tests

* Suite total: **301** (was 299) — `+8` in
  `model::kv_cache::tests`:
  * `new_cache_has_zero_position` (updated for new `read_into` API)
  * `append_then_dequantise_round_trips_within_absmax_tol`
  * `zero_vector_round_trips_exactly`
  * `append_to_capacity_then_errors` (updated)
  * `append_rejects_wrong_length`, `append_rejects_invalid_layer_idx`
  * `reset_clears_position_but_keeps_capacity` (updated)
  * `resident_bytes_matches_layout`
* `tests/kv_cache.rs` integration cases moved from bit-equal asserts
  to cosine-fidelity asserts on `multi_token_forward` vs
  `forward_with_cache`. The greedy-equivalence case is unchanged.

### Compatibility

* **Public API break** within 0.x: `KVCache::read` removed.
  Consistent with the 0.x minor-bump convention (see
  `CHANGELOG.md` preamble). Only one production caller existed; all
  call sites updated atomically.
* Microsoft 2B reference greedy output is byte-identical on the
  Stage 5-E prompts (Mac NEON, antix1 SSE2 i8). The BitNet-family
  fine-tunes (Aramis French, Bifrost Solana coding) verified in
  v0.8.0 also continue to run end-to-end on antix1.

## [v0.8.0-mvp] — 2026-05-29

Minor release. **Phase III step 2** — generic
`ModelArchitecture` trait + registry so the GGUF loader accepts the
BitNet *family* (`bitnet-b1.58`, `bitnet-25`, `bitnet`) instead of
just the canonical Microsoft 2B string. Community fine-tunes
(Aramis French, Bifrost Solana coding) load + greedy-decode
end-to-end on the antix1 Pentium-M validation host. Design
document: [`docs/PHASE_III_ARCHITECTURE_RFC.md`](docs/PHASE_III_ARCHITECTURE_RFC.md).

### Added

* `src/model/architecture/` module — `ModelArchitecture` trait,
  global `OnceLock` registry, `resolve(arch_string) -> Option<&'static dyn …>`.
  Trait methods: `architecture_strings()`, `metadata_prefix()`,
  `config_from_meta()`. Object-safe (`Send + Sync + 'static`).
* `BitNetArchitecture` impl claiming three aliases
  (`bitnet-b1.58` / `bitnet-25` / `bitnet`); all share the same
  forward graph + tensor name layout. Equivalence verified by
  inspecting Aramis + Bifrost — 332 tensors, identical
  `blk.N.{role}.weight` names (including `attn_sub_norm` +
  `ffn_sub_norm`), identical hyperparameter values, identical
  packed byte size.
* `BitNetConfig::from_gguf_metadata_with_prefix(arch_string, prefix, meta)`
  — the entry point each architecture impl calls; the old
  hard-coded `"bitnet-b1.58."` key prefix is gone, the prefix is
  supplied per arch.
* `docs/PHASE_III_ARCHITECTURE_RFC.md` (330 lines) — design doc
  for this work and the upcoming non-BitNet architectures
  (Llama 2, Phi-3, Gemma). The five-step migration plan's steps
  3–5 are deferred: building empty `LayerTensorRole` /
  `ForwardVariant` machinery without a second forward graph to
  exercise them would be the empty-cathedral shape that
  [[feedback-principled-design]] warns against.

### Changed

* `BitNetConfig::from_gguf_metadata()` consults the registry via
  `crate::model::architecture::resolve(arch_string)` instead of
  hard-coding the canonical string. Unknown
  `general.architecture` values still raise
  `UnsupportedArchitecture` (failure surface unchanged).
* `BitNetConfig::ARCHITECTURE` const kept — `src/synth.rs`
  still uses it as the canonical writer-side string.

### Tests

* Suite total: **299** (was 291) — `+8`:
  * `architecture::tests` — 3 (canonical resolve, alias resolve,
    unknown returns None).
  * `architecture::bitnet::tests` — 2 (three aliases claimed,
    prefix equals arch string for each).
  * `architecture::registry::tests` — 3 (non-empty, OnceLock
    stability across calls, all aliases share the same impl).
* End-to-end verified on antix1 Pentium-M (i686 SSE2 i8 default):
  * Aramis — `"La capitale de la France est"` →
    `" Paris. La capitale de la Chine est Paris. La capitale"`
    (greedy, 15 tokens, temp 0). French grammar + first answer
    correct; factual drift beyond is the underlying 2B base
    model's limit.
  * Bifrost — `"fn main() {"` →
    `"  // Your code here  }  {  }  return 0"` (greedy, 15
    tokens, temp 0). Code-domain follow-on.

### Compatibility

* No API removal. New public surface: `model::architecture`
  module (trait + `resolve()` + `BitNetArchitecture`). Existing
  callers of `BitNetConfig::from_gguf_metadata` see no signature
  change.
* Microsoft 2B reference path is byte-identical (load + forward).
  All 291 pre-existing tests still pass on Mac aarch64.

## [v0.7.1-mvp] — 2026-05-27

Patch release. v0.7.0-mvp's prebuilt binaries are fine (i8 kernel
works; greedy output verified), but the v0.7.0 tag commit's CI
Quality Checks were red: the `matvec_column_stride_32_mapping` unit
test broke when i8 became the x86 default. This patch carries the
fix so the release tag points at a green commit.

### Fixed

* `model::bitlinear::tests::matvec_column_stride_32_mapping` now calls
  `bitlinear_i2s_matvec_f32_scalar` directly instead of the dispatch
  entry point. The test mixes activation magnitudes 1.0 … 1000.0 to
  read out each ternary code's contribution exactly; with i8 the
  default x86 kernel, absmax int8 quantisation rounds the 1.0 entry
  to 0 next to the 1000.0 entry — correct i8 behaviour, but it
  defeats a *mapping*-correctness check. The check is kernel-agnostic
  (column-stride-32 unpacking), so scalar is the right reference. The
  i8 path's numerical fidelity stays covered by
  `tests/bitlinear_sse2_i8.rs`. Only on x86 (Mac/aarch64 kept NEON
  f32 as default, so it passed there).

### Unchanged from v0.7.0

* i8 activation kernel is the x86 default (2.2× over f32 on
  Pentium-M, byte-identical greedy output). No kernel or dispatch
  change in this patch — test-only.

## [v0.7.0-mvp] — 2026-05-27

Minor release: SSE2 int8 activation kernel, now the x86 default —
2.2× faster on Pentium-M with byte-identical greedy output.

### Why

Profiling on antix1 (Pentium-M) showed BitLinear matvec is **96.35%**
of decode-step runtime, and a large part of that was the f32 kernel's
per-element `i8 → i32 → f32` sign-extend + convert in the inner loop.

### Added

* `bitlinear_sse2::bitlinear_i2s_matvec_f32_sse2_i8` — int8 activation
  SSE2 BitLinear kernel. Quantises the activation to int8 once
  (absmax-per-vector, same as the NEON i8 path), then runs the dot
  product in integer lanes: ternary weights need no multiply (product
  is `+x / -x / 0` via `cmpeq_epi8` masks + `sub_epi8` negate), i8
  products sign-extend to i16 and fold to i32 via `madd_epi16`. 16 i8
  lanes/instruction vs the f32 kernel's 4, no f32 convert in the loop.
* `tests/bitlinear_sse2_i8.rs` — parity vs scalar at cosine > 0.999,
  max-relative-error < 5% (looser than the f32 kernel's 1e-2 absolute
  because int8 activation is lossy — the same step bitnet.cpp's
  production CPU path takes).

### Changed

* **i8 is now the x86 default kernel.** `bitlinear.rs` X86Sse2 arm
  routes to the i8 kernel by default; the f32 mask-add kernel stays
  behind `--cfg willamette_sse2_f32` for numerical reference. Unlike
  NEON (where i8 was slightly slower, so f32 stays default), x86 i8
  wins on both speed and fidelity.
* `dispatch::Kernel::X86Sse2` label now shows the active variant —
  `"i686 SSE2 (i8)"` vs `"... (f32)"` — so the bench banner / dashboard
  report which kernel ran.
* Every prebuilt x86 binary (x86_64 + i686 musl) now ships the 2.2×
  kernel automatically.

### Performance (antix1, Pentium-M, measured)

| Model | f32 SSE2 | i8 SSE2 | speed-up |
| --- | ---: | ---: | ---: |
| synth 110M decode | 4.60 tok/s | 10.1 tok/s | 2.2× |
| real 2B decode | 0.19 tok/s | 0.41 tok/s | 2.15× |

Cumulative over scalar: ≈ 5.4×. Chat-speed (≥ 5 tok/s) ceiling on
Pentium-M moves from ~100M to ~220M params.

### Fidelity

Greedy decode on the real 2B model (`"The capital of France is"`, 20
tokens, temp 0) produced **byte-identical token sequences** for f32
and i8 — int8 quantisation never flipped an argmax. Full numbers in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### Tests

* Suite: **288** (was 284) — `+4` in `tests/bitlinear_sse2_i8.rs`
  (x86-gated; skip without the real model file).

### Compatibility

* No API change. aarch64 NEON / scalar paths untouched.
* x86 users get the speed-up by upgrading; output unchanged on the
  verified prompt. `--cfg willamette_sse2_f32` restores the old f32
  kernel if a bit-close-to-scalar reference is needed.

## [v0.6.0-mvp] — 2026-05-27

Minor release: synthetic GGUF builder for humble-hardware throughput
benchmarks + scaling data + thesis sweet-spot redefinition.

### Why this release exists

Microsoft only published the 2 B variant of BitNet b1.58. Every
community reproduction at 70 M – 200 M (e.g. `nijil-k/Bitnet-1.58b-
Nous-Llama2-70M`, `Chris4K/bitnet-gpt2-1.58bit`) is a Llama 2 / GPT-2
architecture + BitLinear finetune in `f32` safetensors — none of them
parse as BitNet b1.58 GGUF, so we can't compare ourselves to them
directly. To measure throughput in the same scale band as
TinyLlama / TinyStories 110 M (Karpathy's model the EXO Labs Pentium
II 350 MHz demo runs), we need to build a BitNet b1.58 GGUF at that
size ourselves. This release ships the builder.

### Added

#### `willamette synth-gguf` CLI subcommand
* New `src/synth.rs` library module — `Preset::{Tiny, Small, Medium}`
  + `build_gguf(preset, random_weights)` writing a complete BitNet
  b1.58 GGUF byte buffer.
* `Tiny` ≈ 73 KB (the existing in-CI synthetic, unchanged behaviour:
  all-zero ternary weights so `tests/synthetic_model.rs`'s numerical
  assertions still hold).
* `Small` ≈ 7 M params (256-d embedding, 6 layers, 12 000 vocab).
* `Medium` ≈ 110 M params (768-d embedding, 12 layers, 32 000 vocab)
  — same scale class as `tinyllamas/stories110M.bin`.
* `Small` / `Medium` use random ternary weights drawn from an inline
  64-bit xorshift PRNG seeded by the preset config. Same seed → bit-
  identical GGUF across hosts. No new external dependencies.
* The synthetic GGUF has NO tokenizer metadata, so `inspect` and
  `bench` work against it, but `run` / `chat` / `tui` will reject
  it. This is by design — random ternary weights produce garbage
  tokens. Keeping the tool inside [[feedback-no-fake]]: we never
  claim *quality*, only *throughput*.

#### 4-point scaling table
* [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) extended with
  Pentium-M antix1 and Mac M4 NEON measurements at the three synth
  preset sizes plus the real 2 B model. New conclusions:
  * On antix1, `params × tok/s ≈ 500 M` is constant across the four
    points — clean linear scaling, BitLinear matvec dominates.
  * The Mac M4 ÷ antix1 ratio grows 8.8× → 26.4× → 65.8× with model
    size, because the cache hierarchy diverges once weights stop
    fitting in antix1's 2 MB L2.
  * Direct cross-architecture comparison vs EXO Labs' Pentium II
    demonstration: same-cycle efficiency advantage of **2.6×** for
    BitNet 1.58 + SSE2 over vanilla Llama 2 + no-SIMD.

#### Sweet-spot redefinition
* README and `_internal/VISION.md` updated. The old "medium 1 B – 13 B
  on humble hardware" formulation was tier-blind. The 2026-05-27
  measurement makes the coupling explicit: on Pentium-M-class SSE2
  hardware the practical ceilings are **~100 M params for chat
  speed**, **~500 M for slow-but-usable**, **~5 B for
  demonstration**. Modern AVX2 / multi-core hosts shift every
  threshold ~1 order of magnitude up, restoring the aspirational
  range.

### Fixed

* `cmd_bench` no longer hard-codes `in_dim=2560, out_dim=2560` in
  the matvec banner — reads from the actual `attn_q.shape` so the
  Throughput / Time numbers are consistent on any model size.
* `cmd_bench` no longer hard-codes token id 15339 (Llama-3
  "Hello") as the probe — that crashed on Tiny preset's vocab=4.
  Now clamps to 0 when vocab ≤ 15339. Doesn't change throughput on
  the real BitNet 2B (any embedding row is fine).
* `cmd_bench` banner reads `graph.config.block_count` instead of
  the literal "30 layers" string.

### Tests

* Suite total: **284** (was 279). `+5` in `src/synth::tests` cover
  preset dimensions, parameter-count estimation accuracy for
  Medium, byte-stability across builds, and PRNG legal-code range.

### Compatibility

* No API removals. Existing `inspect` / `run` / `bench` / `chat` /
  `tui` against the real BitNet 2B model are unchanged.
* New CLI surface (`synth-gguf`) — that's the minor-version
  trigger. No flag changes elsewhere.
* aarch64 NEON and x86 SSE2 dispatch paths identical to v0.5.0.

## [v0.5.0-mvp] — 2026-05-25

Minor release: Stage 6-B SSE2 BitLinear kernel lands. First time
`dispatch::Kernel::X86Sse2` actually routes traffic — the v0.4.0
slot is now filled with a working implementation, verified for
both parity and speed on a real Pentium-M host.

### Added

#### `src/model/bitlinear_sse2.rs` — SSE2 BitLinear matvec

* Numerically equivalent to `bitlinear_i2s_matvec_f32_scalar`
  within the same `max |Δ| < 1e-2` tolerance the NEON parity test
  already enforces. Validated on antix1 (Pentium-M Banias/Dothan,
  family 6 model 13, i686 / SSE2 ceiling): all 8 layer-0
  BitLinear weights (attn_q/k/v/output, ffn_gate/up/down,
  zero-input check) pass.
* Same two-accumulator shape as scalar
  (`out[j] = scale · (Σ_pos x[i] − Σ_neg x[i])`) — no
  multiplication by ±1.0, only mask-add. SIMD strategy:
    1. Per 128-element block, unpack 32 packed bytes into a
       stack-resident `[i8; 128]` using the column-stride-32 map
       (`c0→gp`, `c1→32+gp`, `c2→64+gp`, `c3→96+gp`).
    2. Walk in 4-float chunks. Sign-extend each ternary `i8`
       to `i32` via the pure-SSE2 sequence
       `unpacklo_epi8`+`srai_epi16`+`unpacklo_epi16`+`srai_epi32`
       (no SSE4 `_mm_cvtepi8_epi32` — Pentium-M doesn't have it),
       then `cvtepi32_ps` to `f32`.
    3. `_mm_cmpeq_epi32` builds the +1 / −1 masks;
       `_mm_and_ps(x, mask)` does the conditional add into a
       4-lane positive / negative accumulator.
    4. Horizontal-sum (pure SSE2, no `_mm_hadd_ps`) and combine
       at the end of each row.
* `#[target_feature(enable = "sse2")]` on the kernel; sound
  because `dispatch::select_kernel` gates the call on
  `is_x86_feature_detected!("sse2")`.

#### `tests/bitlinear_sse2.rs` — parity contract

* Mirror of `tests/bitlinear_simd.rs` (NEON). cfg-gated on
  `target_arch = "x86"` / `"x86_64"`.
* SKIPs gracefully if the real GGUF isn't present or if the host
  doesn't advertise SSE2 (so the suite still passes on synthetic
  test runners that lack the model file).

#### Performance — antix1 measurement (v0.5.0 vs v0.4.1)

| Measurement | scalar | SSE2 | speed-up |
| --- | ---: | ---: | ---: |
| BitLinear matvec (2560 × 2560) | 60.5 ms | **24.3 ms** | **2.49×** |
| Matvec throughput | 108 M e/s | 269 M e/s | 2.49× |
| Single-token forward (30 layers) | 21.7 s | **8.87 s** | **2.45×** |
| Decode-step (KV cache, avg of 3) | 21.65 s | **8.15 s** | **2.66×** |
| tok/s | 0.05 | **0.12** | **2.4×** |

Full numbers + reproduction recipe in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Still slow in
absolute terms (0.12 tok/s = ~8 s/token on a 21-year-old CPU),
but the dispatch path is real, parity is enforced, and there is
a documented next-step (i8 activation path) for further gains.

### Changed

* `dispatch::select_kernel` returns `Kernel::X86Sse2` on x86 /
  x86_64 hosts that report SSE2 (previously always fell through
  to `Scalar`).
* `bitlinear::bitlinear_i2s_matvec_f32` has a new `X86Sse2` arm
  that routes to the unsafe kernel; aarch64 NEON arm and scalar
  fallback are untouched.

### Tests

* Suite total: **279 lib + 8 SSE2 integration = 287** (was 279).
  The 8 SSE2 cases only run on x86 hosts with the model file
  present; the lib suite is unchanged.

### Compatibility

* No ABI / API changes. Existing v0.4.1-mvp users on x86 / x86_64
  get the 2.4× speed-up automatically by upgrading — no flags, no
  recompile knobs.
* aarch64 / Apple Silicon path identical to v0.4.1.
* Pre-built binaries are still produced by `.github/workflows/release.yml`
  for the same 6 targets.

## [v0.4.1-mvp] — 2026-05-25

Patch release. The v0.4.0-mvp `release.yml` workflow built 5/6
targets successfully but the `armv7-unknown-linux-musleabihf` job
failed at the build step with `error: variable does not need to be
mutable` on `src/model/dispatch.rs:97`. On armv7 / armv7-musleabihf
none of the cfg arms inside `detected_features()` activate, so the
`let mut out` is genuinely unused-mut on that target — and the
workflow's `RUSTFLAGS=-D warnings` promoted that to an error.

That blocked the `Publish GitHub Release` job (`needs: build`)
across the board, so v0.4.0-mvp ended up with a git tag but no
artifacts on GitHub. This patch fixes the build and re-runs the
distribution pipeline.

### Fixed

* `src/model/dispatch.rs:97` — `#[allow(unused_mut)]` on the
  `out` vec inside `detected_features()`. On targets that have
  SIMD slots compiled in (aarch64 / x86 / x86_64) the `mut` is
  used; on armv7 and generic targets it isn't, and the cfg arms
  that would have used it expand to nothing. The allow narrows
  the exception to one variable; everywhere else still benefits
  from `-D unused-mut`.

### Distribution

* This is the first release with `release.yml` actually publishing
  to GitHub Releases. v0.4.0-mvp's tag is left in place as a
  historical marker (no artifacts attached); future users should
  pull v0.4.1-mvp or later.

## [v0.4.0-mvp] — 2026-05-25

Minor release: humble-hardware friendly distribution.

Two pieces of the original thesis ("medium-sized public LLMs on
CPU-only humble hardware") land here. (1) The runtime now picks its
own BitLinear kernel based on the host CPU, with a single source of
truth — the dashboard, bench banner, and the matvec dispatcher can
no longer drift apart. (2) Every tag push produces cross-compiled
static binaries for six targets, so a Pentium-M antiX user no
longer needs gcc, make, rustup, and 4 minutes of compile time —
they download a 5-ish MB tarball and run.

### Added

#### Pre-built release binaries (`.github/workflows/release.yml`)

* Triggered on any `v*-mvp` tag push.
* Six build targets:
  * `x86_64-unknown-linux-musl` — modern Linux desktops, CI, dev
    servers.
  * `i686-unknown-linux-musl` — Pentium-M / antiX class (the Stage
    6-B validation host).
  * `aarch64-unknown-linux-musl` — RPi 4 64-bit, AWS Graviton,
    ARM VPS.
  * `armv7-unknown-linux-musleabihf` — RPi 3, BeagleBone, Pi
    Zero 2.
  * `aarch64-apple-darwin` — Apple Silicon native.
  * `x86_64-apple-darwin` — Intel Macs (cross-compiled on the
    M-class runner).
* Linux builds go through `cargo-zigbuild` with Zig 0.13, producing
  musl-static binaries. One artifact runs on antiX (glibc 2.36),
  Raspberry Pi OS (glibc 2.31), and Ubuntu 24.04 (glibc 2.39)
  without an LD\_LIBRARY\_PATH dance. Stripped after build.
* Each archive is `willamette-<tag>-<target>.tar.gz` — the binary
  is renamed `willamette` inside the tarball (crate name remains
  `project-willamette` for cargo) so the user types `./willamette`.
* Each artifact ships with a SHA-256 sum, plus README + license +
  CHANGELOG so the tarball is self-contained.
* A second job pulls every artifact, slices the matching CHANGELOG
  section as the release notes, and either creates the release or
  `--clobber`-uploads to one already created by the manual 8-step
  flow.

#### `src/model/dispatch.rs` — runtime CPU dispatch module

* `Kernel` enum with three variants (`Scalar`, `AArch64Neon`,
  `X86Sse2`).
* `active_kernel()` — `OnceLock`-cached, single CPU-ID read.
* `Kernel::label()` returns the same string used in the TUI
  dashboard, the bench banner, and (future) log lines.
* `detected_features()` is the source of the dashboard's per-SIMD
  ●/○ list (currently `neon` + `dotprod` on aarch64; `sse2` +
  `sse4.1` + `avx2` on x86 / x86_64).

### Changed

* `bitlinear::bitlinear_i2s_matvec_f32` now branches on
  `dispatch::active_kernel()`. The aarch64 NEON path is unchanged
  numerically — byte-parity tests (`kv_cache`, `multi_token`,
  `forward`) all green.
* `chat/tui.rs::initial_dashboard_state` and `main.rs::cmd_bench`
  both consume `dispatch::active_kernel().label()` /
  `dispatch::detected_features()`. The old per-call-site arch
  detection in those two files is gone (~80 lines removed).
* `src/model/mod.rs` exports the new `dispatch` module.

### Tests

* Suite total: **279** (was 276).
* `+3` in `model::dispatch::tests`:
  * `active_kernel_is_stable_across_calls` — OnceLock correctness.
  * `label_is_non_empty` — every variant has a non-empty label so
    the dashboard never renders a blank line.
  * Plus an aarch64-gated case that confirms `Kernel::AArch64Neon`
    is picked when the host has NEON (it always does on Apple
    Silicon / Cortex-A57+), and an x86-gated case that confirms
    `sse2` shows up in `detected_features()`.

### What's intentionally NOT in this release

* **Stage 6-B SSE2 kernel itself.** `Kernel::X86Sse2` is defined,
  the detection slot is in place, and dispatch falls through to
  Scalar with a clear comment for the next contributor. The actual
  intrinsic implementation (`pmaddubsw` / `pmaddwd`) is Phase 3 —
  separate work that needs benchmark numbers from the antiX host
  before / after to be honest about the speedup claim.

## [v0.3.1-mvp] — 2026-05-25

Patch release. Three user-reported usability bugs in the v0.3.0 chat
TUI, plus a CI / refactor cleanup pass against the Sonar Quality Gate.

### Fixed

* **Korean / CJK input no longer overlaps the previous character.**
  The screen cursor was placed at `prefix + cursor_char()`, but
  `cursor_char()` counts codepoints while ratatui draws each Hangul
  / CJK / emoji glyph in two terminal cells. Subsequent input landed
  mid-glyph and visually overlapped. Replaced with a new
  `InputEditor::cursor_display_col()` backed by the `unicode-width`
  crate; the prompt prefix is now measured the same way for
  symmetry. (`src/chat/input_editor.rs`, `src/chat/tui.rs`.)
* **Auto-scroll: the chat pane now sticks to the bottom while
  streaming.** Once the wrapped-line count exceeded the viewport
  height the user only ever saw the first lines — newly streamed
  tokens scrolled off-screen. Added a `follow_bottom` flag on
  `UiState`; the renderer pins `scroll_offset` to
  `total_lines - viewport_h` every frame when it's set, using
  `Paragraph::line_count` (ratatui's `unstable-rendered-line-info`
  feature). Scrolling up turns it off; Ctrl-End or scrolling down
  past the last line turns it back on.
* **Scroll keys + wheel were inverted.** `Paragraph::scroll((n, 0))`
  is a top-skip count, not a "lines from bottom" offset, so PageUp /
  wheel-up moved the view down and vice versa; Ctrl-Home pinned to
  the bottom; Ctrl-End pinned to the top. Renamed
  `UiState::scroll_back` → `scroll_offset` to match ratatui's
  convention and routed all four entry points (PageUp/Down,
  Ctrl-Home/End, wheel ↑/↓, mid-stream PageUp/Down) through helpers
  (`scroll_up_by`, `scroll_down_by`, `scroll_to_top`,
  `scroll_to_bottom`) that also flip `follow_bottom` correctly.

### Changed

#### CI hygiene
* `rust-toolchain.toml` pins channel `1.94.0` so CI's `stable` (which
  follows the latest release) stops drifting away from local fmt /
  clippy output. The v0.3.0 cycle alone cost four "Rust 1.95 fmt /
  clippy drift" CI fixes — the pin removes that cost permanently.
* Quality Gate breakdown step now uses the CE-task → analysisId →
  project_status flow with `Authorization: Bearer …` (the previous
  `-u TOKEN:` basic-auth call silently returned the SonarQube SPA
  HTML on the API endpoints, so gate failures only surfaced as an
  opaque `FAILED` exit). Diagnostic output now lists each failed
  condition by metric + threshold.

#### Refactor (no behaviour change)
* Five functions split to clear `rust:S3776` Cognitive Complexity
  ≤ 15: `chat/dashboard.rs::render_lines` (18 → six section
  helpers), `chat/engine.rs::stream_assistant_response` (19 → emit
  / cancel / finalize helpers + free-fn `flush_safe_window`),
  `chat/tui.rs::handle_key_normal` (18 → `handle_ctrl_key` +
  `handle_enter`), `chat/tui.rs::handle_slash` (18 →
  `handle_slash_sys` + `handle_unknown_slash` +
  `nearest_slash_command`), and `model/cached_forward.rs::
  forward_with_cache_progress` (26 → `validate_cache_inputs`,
  `forward_one_layer`, `scaled_dot_product_attention`,
  `apply_ffn_block`, `check_finite_hidden` + a private `LayerCtx`
  struct for per-token scalars).
* Byte-parity vs `bitnet.cpp` preserved on the hot path: verified
  by `tests/kv_cache.rs` (3), `tests/multi_token.rs` (5),
  `tests/forward.rs` (3).
* `is_aarch64_feature_detected!` macro now triple-cfg-gated for
  aarch64 / x86_64 / other so x86 CI hosts build the active_kernel
  string without referencing the macro.

### Tests

* Suite total: **276** (was 272 at v0.3.0-mvp).
* `+4` new in `chat::input_editor::tests` covering
  `cursor_display_col` on ASCII, Hangul, mixed, and per-codepoint
  advancement.

### Dependencies

* `unicode-width = "0.2"` — explicit dep (was transitive via
  ratatui); used by both `chat/input_editor.rs` and `chat/tui.rs`.
* `ratatui` now opts into the `unstable-rendered-line-info` feature
  for `Paragraph::line_count`.

## [v0.3.0-mvp] — 2026-05-25

Minor release: operator-grade chat TUI.

The v0.2.x TUI was usable but missing the editing baseline every
modern terminal AI tool provides — arrow keys, history recall,
search, mid-turn cancel, paste, real cursor — and gave no
visibility into engine / system state. v0.3.0 fills both gaps
in one cycle.

### Added

#### Right-pane live perf dashboard
* New `src/chat/sysmon.rs` — 1 Hz polling of `sysinfo` over a
  daemon thread, normalised process CPU %, per-core %, memory.
* New `src/chat/dashboard.rs` — pure render fn producing six
  sections: HARDWARE, CPU, MEMORY, INFERENCE, SAMPLING, MODEL.
  Gauges with traffic-light coloring (green ≤ 60 %, yellow ≤ 85 %,
  red > 85 %). Per-core display collapses cores ≥ 13.
* Dashboard lives at terminal width ≥ 72; narrower widths fall
  back to single-pane.

#### Readline-grade input editor
* New `src/chat/input_editor.rs` — pure data structure, fully
  unit-tested (19 inline tests).
* Cursor movement: ← →, Home, End, Ctrl-A / Ctrl-E.
* Deletion: Backspace, Delete, Ctrl-W (word), Ctrl-U (to start),
  Ctrl-K (to end).
* History: ↑ / ↓ to recall previous prompts (ring buffer cap 1000).
* Reverse search: Ctrl-R opens overlay; type to filter newest-
  first; Enter loads, Esc cancels, Ctrl-R steps to next older
  match.
* UTF-8 atomic: cursor moves snap to char boundaries, multi-byte
  CJK / emoji codepoints never split.
* Persisted history at `~/.config/willamette/history`,
  cap 1000 entries, append-on-submit, oldest evicted.

#### Inference observability
* Layer-progress display: `forward_with_cache_progress` calls an
  optional `on_layer(layer_idx)` callback after each transformer
  block; the TUI shows "layer 17 / 30" updating live.
* Live tok/s: rolling token count + turn elapsed in atomics;
  dashboard reads them every frame.
* ETA estimation: remaining tokens ÷ live tok/s.

#### Mid-turn cancel
* Esc during generation sets `WorkerProgress.cancel_requested`
  atomic; engine checks at each iteration and exits cleanly.
  History truncates to whatever was emitted so KV cache stays
  consistent for the next turn.

#### Discoverability + polish
* F1 / `/help` opens a Help overlay popup with the full keybinding
  cheatsheet.
* Tab completes slash commands; ambiguous prefix shows candidates.
* Unknown slash command suggestions via Levenshtein distance ≤ 2:
  "unknown /reser — did you mean /reset?".
* Real terminal cursor in input area via
  `Frame::set_cursor_position` (was: rendered `_` glyph).
* Mouse wheel scrolling on chat log (3 lines per tick).
* Bracketed paste support: multi-line paste arrives as a single
  `Event::Paste(String)`, inserted at cursor.
* Ctrl-Y "yanks" the last bot response to system clipboard via
  OSC52 (works in iTerm2 / Kitty / Alacritty / wezterm / recent
  xterm). Inline 30-LoC base64 encoder — no new dep.
* Ctrl-L clears visible chat log without dropping history.
* Ctrl-Home / Ctrl-End jump to top/bottom of chat scrollback.

### Changed
* `ChatEngine` gains `set_worker_progress` + nine read-only
  accessors (config\_*, sampler, system_prompt,
  estimate_kv_cache_bytes). Existing call sites unaffected.
* `Sampler` exposes `params()` and `params_clone()` for the
  dashboard.
* Stdio `willamette chat` is unchanged — TUI features don't
  affect the simpler stdio surface.

### Dependencies
* `sysinfo = "0.32"` (~50 KB compiled). Cross-platform CPU/memory
  sampling. The only new dep this cycle.

### Tests
* 272 total (v0.2.3 had 242 — 30 new).
  * 19 new in `input_editor::tests`
  * 3 new in `sysmon::tests`
  * 6 new in `dashboard::tests`
  * 2 new in `tui::tests` (Levenshtein + base64)

### Notes
* No public API removal. No numeric inference change.
* Live verification still requires a real TTY — headless tests
  cover everything but the actual ratatui draw + mouse + paste.
  See `_internal/VISION.md` for the planned QEMU bench harness
  (Stage 27) that will let humble-hardware UX be measured too.
* This release closes "what every terminal AI tool already
  provides" gap. Next likely cycle: Phase III generalisation
  (multi-architecture model support — Llama / Mistral / Phi).

## [v0.2.3-mvp] — 2026-05-25

Patch release: three TUI / chat readability fixes after real-session
feedback on v0.2.2.

### Fixed
* **Emoji clutter in chat output.** The base model often writes
  trailing pictograph clusters (`😊👍🏻💬📚✨` etc.) lifted from
  social-media training data. They show up everywhere and don't
  add information. `is_emoji_char(char)` covers the major emoji
  blocks (plus ZWJ + variation selectors); `strip_emoji_chars`
  filters them from both the live tick stream and the recorded
  history. The model still emits the underlying tokens so the KV
  cache stays in sync with what it thinks it said; we just hide
  the visual noise.
* **`User:` leak on screen.** v0.2.2's stop-sequence detection
  worked at the *history* level — next turn's prefill saw clean
  text — but the bytes that triggered detection (`User:`,
  `BITNETAssistant:`, …) had already streamed to the caller's
  `tick` callback and were visible on screen for the truncated
  turn. v0.2.3 holds back a 24-byte look-ahead buffer, sliding
  it forward each step; if a stop sequence appears in the
  buffered tail, the tail is discarded before it can be ticked.
  Trade-off: ~24 bytes (3-6 words) of streaming latency. The
  surface output is now genuinely clean.

### Added
* **Markdown rendering in the TUI.** `src/chat/markdown.rs` —
  new ~160-LoC inline renderer maps `**bold**`,
  `` `inline code` ``, `# heading`, `- bullet`, `1. numbered`,
  with leading whitespace preserved, onto `ratatui::Line`/`Span`
  styled output. BOT messages and the live streaming response
  both go through it; USR / SYS bubbles stay plain text.
* `append_message_lines` helper in `src/chat/tui.rs` — the role
  badge is prepended to the first body line; continuation lines
  align under the badge; the streaming path appends a green `▌`
  cursor so the user can see generation is live.
* `floor_char_boundary` inline helper in `src/chat/engine.rs` —
  the unstable `str::floor_char_boundary` is unavailable on
  stable Rust, so we provide our own. Used by the look-ahead
  buffer to snap the safe-emit boundary onto a UTF-8 char so we
  never tick a half-codepoint.

### Tests
* Total: **242** (v0.2.2 had 221 — 21 new this cycle).
  * 14 new in `src/chat/markdown.rs::tests` — bold, inline code,
    bullet (`-` and `*`), numbered, multidigit, heading,
    indented bullet, indent preservation, plain pass-through,
    unclosed-bold-stays-literal, period-in-version-not-list,
    multiple lines, realistic Korean-history multi-line.
  * 7 new in `src/chat/engine.rs::stop_sequence_tests` —
    `is_emoji_char` for pictographs and CJK preservation,
    `strip_emoji_chars` for trailing clutter / Korean
    interleave / ZWJ sequences, `floor_char_boundary` on ASCII
    and Korean.
* Coverage: SonarQube `new_coverage` stays at 100 % on v0.2.x
  new code; the new modules are unit-tested in-CI without the
  real model file.

### Notes
* No public-API change. No CLI flag change. No numeric inference
  change. Reference parity with pinned bitnet.cpp on Stage 5-E
  prompts is preserved.
* `--no-emoji` / `--no-markdown` CLI flags intentionally not
  added — the cleaner output is the right default for a chat
  surface. If a future use-case ever needs them they're easy to
  graft onto `ChatArgs`.

## [v0.2.2-mvp] — 2026-05-25

Patch release: chat usability + honest in-CI coverage.

The user-facing v0.2.1 chat was leaking model self-talk into the
visible output (the base model writes its own follow-up
`User:` / `BITNETAssistant:` turns past the answer). This release
detects those patterns and truncates. It also replaces the
v0.2.0 cycle's Sonar `coverage.exclusions`-shaped fix with actual
in-CI tests covering 59 new lines that were previously at 0 %.

### Fixed
* **`ChatEngine` runaway**: `stream_assistant_response` now checks
  the accumulated response text after each token for hallucinated
  turn-boundary phrases — `BITNETAssistant:`, `User:`, `Human:`,
  `Human (`, `User (`, `AI Assistant:`, `Assistant:`, `Question:`,
  and 3 more variants observed in real-model output — and breaks
  out of the generation loop on the first match. The recorded
  `history` is truncated at the boundary so subsequent turns see
  a clean transcript. Empirical: a single `"how are you?"` turn
  that used to spill 543 tokens of fake hash-function tutorial now
  stops cleanly at 51 tokens.
* **`UnsupportedTokenizer` out-of-range messaging**: now consistent
  across the `encode_with_specials` synthetic test (was implicit
  before; tests in `tests/tokenizer_synthetic.rs` lock it in).

### Added
* `find_chat_stop_sequence(&str) -> Option<usize>` and
  `truncate_at_chat_stop_sequence(&mut String) -> bool` as
  `pub(crate)` helpers — pure functions, 15 inline unit tests
  cover Unicode safety, false-positive guards (`Humans`/`Users`),
  earliest-match selection, and the verbatim v0.2.1 TUI failure
  string.
* `tests/tokenizer_synthetic.rs` extended with 7 new tests that
  build a valid in-memory tokenizer GGUF (256 byte-level glyphs
  + BOS + EOS, no merges) and exercise `encode_with_specials`,
  `encode`, and the BOS/EOS plumbing without needing the 1.1 GiB
  model file.
* `tests/synthetic_model.rs` — new ~280-LoC test file that builds
  a complete in-memory BitNet b1.58 GGUF (≈73 KB; 2 layers,
  n_embd 128, vocab 4, all-1.0 norms, all-0 BitLinear, all-1.0
  embeddings) and exercises `ModelGraph::from_gguf`,
  `forward_single_token_position_zero`, `forward_with_cache`,
  and `multi_token_forward`. 6 new tests cover:
  full GGUF parse, norm-cache pre-decode invariant, no-NaN
  forward output, KV cache continuity across 2 positions, and
  cache-vs-no-cache parity at position 0.
* `Quality Gate breakdown` step in `.github/workflows/sonar.yml`
  — queries the SonarQube REST API on every scan and prints the
  per-condition pass/fail with actual values, so future Quality
  Gate failures are debuggable from the GitHub Actions log
  without dashboard access. `continue-on-error: true` so a
  transient API hiccup never blocks the official gate-action.
* `ChatArgs` shared clap argument group + `build_sampling_params`
  helper in `src/main.rs` — DRY for the `chat` and `tui`
  subcommands.
* `print_slash_help`, `print_slash_history`, `print_slash_stats`,
  `handle_slash_save`, `handle_slash_sys` — per-command helpers
  in `src/main.rs`.
* `build_chat_fragment`, `prefill_prompt_tokens`,
  `stream_assistant_response`, `emit_token_bytes` — helper
  methods on `ChatEngine` extracted from the previously-monolithic
  `send_user_message`.
* `drain_token_events`, `apply_token_event`, `finish_bot_turn`,
  `fail_bot_turn`, `clear_transient_if_old`, `poll_one_input` —
  helpers extracted from `ui_loop` in `src/chat/tui.rs`.

### Changed
* **Sonar action bumped** `SonarSource/sonarqube-scan-action@v6`
  → `@v7.2.1` — v6 is a Node 20 action that GitHub now force-runs
  on Node 24 with a deprecation warning. v7.2.1 is the last v7
  release before v8.0.0 flipped `skipSignatureVerification` to
  `false` (breaking change); v7.2.1 is drop-in compatible with v6.
* **Sonar `coverage.exclusions`** pruned back to only what truly
  cannot run in CI: `src/main.rs` (interactive CLI),
  `src/chat/**` (TTY-dependent), `src/model/bitlinear_neon.rs`
  (aarch64-only kernel; x86 CI runner compiles it out via cfg),
  `scripts/**`, `.github/**`. Everything else (model load,
  forward path, tokenizer module) now has real in-CI coverage
  via the new synthetic-GGUF tests above.
* **Cognitive Complexity** (rust:S3776) fixed on the four
  functions Sonar flagged: `cmd_chat` (was 20 → ≤15),
  `handle_slash_command` (19 → ≤15), `send_user_message`
  (23 → ≤15), `ui_loop` (31 → ≤15). All four broken out into
  named helpers.
* **`Tokenizer::encode_with_specials` test** rewritten with
  `ids.contains(&eot_id)` instead of `iter().any(|&id| id == eot_id)`
  to satisfy Rust 1.95.0's new `clippy::manual_contains` lint.

### Tests
* Total: **221** (v0.2.1 had 193 — 28 new this cycle).
* Coverage: SonarQube `new_coverage = 100.0 %` on the v0.2.x new
  code, replacing the previous 0 % (which was hidden by a blanket
  exclusion).
* Quality Gate: ✔ OK on all 3 conditions
  (`new_violations` 0, `new_duplicated_lines_density` 1.34 %,
  `new_coverage` 100 %).

### Notes
* No public-API change to any of the existing subcommands. The
  `chat` / `tui` CLI flag surface is unchanged (clap's
  `#[command(flatten)]` preserves the argument layout).
* No numeric inference change. Reference parity with the pinned
  bitnet.cpp on Stage 5-E prompts is preserved.

## [v0.2.1-mvp] — 2026-05-25

Patch release: chat-template choice tuned for the base model.

Empirical testing of v0.2.0's chat surface showed two failure
modes — every response was prefixed with a hallucinated tag
(`PowerShell>`, `Vietnamese>`, `French>`, …) and the model would
not honour even trivial instructions like "tell me only english."

Investigation:

* `microsoft/bitnet-b1.58-2B-4T-gguf` is a **base/foundation
  model**, not instruct-tuned. The GGUF self-description is plain
  `general.name = "bitnet2b"` (no Instruct tag); the upstream
  `microsoft/BitNet` README:245 documents `-cnv, --conversation` as
  being "for instruct models" and lists eligible repos — this one
  is not in that list. The model was trained on 4 T tokens of web
  text without SFT or RLHF; expecting it to follow instructions is
  out of scope for a base model.
* The GGUF includes a `tokenizer.chat_template` Jinja string of the
  shape `Human: <content>\n\nBITNETAssistant: <eos_token>`, but
  that template was inserted unconditionally by the conversion
  script (`utils/convert-ms-to-gguf-bitnet.py:1324`) regardless of
  whether the model itself was trained on that pattern. The
  `eos_token` variable was therefore never grounded in any specific
  inference-time token id during training.

What v0.2.0 did wrong: it injected `<|eot_id|>` (128009) between
turns, interpreting Jinja `eos_token` as the LLaMA-3 turn
boundary. Empirically this pushed the model into the
"language-prefix" failure mode above.

### Fixed
* `ChatEngine::send_user_message` now uses a plain text bridge
  (`\n\nHuman: <content>\n\nBITNETAssistant: `) between turns
  instead of injecting a Jinja-template-derived turn marker. The
  same prompt that produced `"PowerShell> Hello!"` in v0.2.0 now
  produces `"Hello! How can I assist you today?"` in v0.2.1.
  Reference parity with bitnet.cpp greedy decode (Stage 5-E
  prompts) is unchanged.

### Unchanged
* `Tokenizer::encode_with_specials(&[PromptPart])` (Stage 9-B) and
  `PromptPart::{Text, Special}` remain in the public API. They were
  needed for the template-faithful approach we just reverted and
  may still be useful for future instruct-tuned BitNet variants
  (e.g. Falcon3-Instruct-1.58bit) that *were* trained with explicit
  turn markers.
* All 193 tests still pass.
* Performance unchanged (this is a chat-template choice, not a
  kernel change).

## [v0.2.0-mvp] — 2026-05-25

Minor release: first-class chat experience + ~5× decode-step speedup.

The inference path's numeric semantics are unchanged from v0.1.x —
greedy decode on the Stage 5-E reference prompts still produces
byte-identical tokens to the pinned bitnet.cpp reference. What's new
is the *runtime surface*: a real chat engine, a full TUI, a launcher,
and a parallelised matvec.

### Added
* `willamette chat` — stdin/stdout multi-turn dialogue subcommand with
  KV-cache reuse across turns, UTF-8-safe streaming output, EOS auto-
  stop, slash commands (`/help`, `/reset`, `/history`, `/save`,
  `/sys`, `/stats`, `/quit`).
* `willamette tui` — ratatui full-screen chat TUI over the same engine
  (history pane, input box, status bar, PgUp/PgDn scrolling).
* `Tokenizer::encode_with_specials(&[PromptPart])` for mid-prompt
  token-id injection — required to render the BitNet chat template's
  `<|end_of_text|>` boundary verbatim instead of byte-level-BPEing it
  into 7 tokens.
* `PromptPart::{Text, Special}` enum.
* `src/chat/engine.rs::ChatEngine` — turn-streaming chat runner.
* `src/chat/tui.rs::run_tui` — terminal UI driver with a worker
  thread + mpsc channels.
* `scripts/willamette` — all-in-one launcher: SHA256-verifies the
  model, optionally downloads it from Hugging Face, rebuilds the
  binary if stale, then launches the requested mode (default TUI).
* `bitlinear_i2s_matvec_f32_neon_i8` — int8-activation NEON kernel
  (Stage 10-D). Code present but **not the default**: see "Changed"
  for why.

### Changed
* **BitLinear matvec is now multi-threaded** via `rayon::par_chunks_mut`
  with chunks of 32 output rows, each chunk owning a thread-local i8
  scratch buffer (Stage 10-C + 10-B). On Apple M4 the decode-step
  improves from `~656 ms / ~1.5 tok/s` (v0.1.1) to
  `~126 ms / ~7.9 tok/s` (v0.2.0) — a 5.2× speedup. The matvec itself
  drops from 1.87 ms to 0.64 ms (2.94×). ISA-neutral: the rayon
  parallelism also helps the scalar fallback on multi-core x86 hosts
  once the SSE2 kernel lands.
* Norm weights (`attn_norm`, `attn_sub_norm`, `ffn_norm`,
  `ffn_sub_norm` per layer, plus `output_norm`) are now pre-decoded
  into `Vec<f32>` at `ModelGraph::from_gguf` time (Stage 10-A). The
  forward path reads them directly — 121 fewer per-token
  allocations.
* `ChatEngine::send_user_message` always forwards the just-emitted
  token into the KV cache (unlike one-shot `generate_with_cache_and_sampler`,
  which skipped the final step). Continuity across turns now matches
  the canonical training-time pattern.
* Stage 10-D int8-activation path investigated and benched. On stable
  Rust the `vdotq_s32` SDOT intrinsic is gated behind the unstable
  `stdarch_neon_dotprod` feature, so the kernel falls back to
  `vmull_s8`-based widening dot. Measured at 7.82 tok/s vs the f32-
  input NEON path's 7.91 tok/s on Apple M4 (20-sample average) — a
  small regression, not a win. The int8 kernel is therefore present
  but gated behind
  `RUSTFLAGS="--cfg willamette_i8_activations"`. Default stays on the
  f32-input NEON path. We'll switch over when `stdarch_neon_dotprod`
  stabilises.

### Dependencies
* `rayon = "1.10"` — for Stage 10-C row parallelism.
* `ratatui = "0.29"` and `crossterm = "0.28"` — for Stage 9-E TUI.

### Tests
* `encode_with_specials` parity (text-only path equals plain
  `encode`), special-id injection, out-of-range rejection, BOS-via-
  `Special` prefix.
* All 189 v0.1.1 tests still pass; total at v0.2.0 is 193 (4 new).

### Performance (Apple M4, NEON, release profile, 20-run avg)

| Metric | v0.1.1 | v0.2.0 | Change |
| ------ | -----: | -----: | -----: |
| BitLinear matvec (attn_q, 2560×2560) | 1.87 ms | 0.64 ms | **2.94× faster** |
| Decode-step with KV cache | 656 ms | 126 ms | **5.19× faster** |
| Tokens / second (decode) | 1.52 | 7.91 | **5.19×** |

## [v0.1.1-mvp] — 2026-05-24

Patch release: bug fix in generation + SonarQube validation lane.
No public-API or behaviour change to the inference path itself
(token-id parity vs. bitnet.cpp from v0.1.0-mvp is preserved).

### Fixed
* `willamette run` no longer panics with `decoded bytes are not valid
  UTF-8` when the generated stream is cut off in the middle of a
  multi-byte character (Korean / CJK / emoji). Generation now ends
  cleanly and the truncated suffix is shown as `U+FFFD`.
* The streaming token-by-token printer in `willamette run` now buffers
  bytes across tokens and emits only up to the last valid UTF-8
  boundary on each tick, so multi-byte characters split across two or
  three BPE tokens no longer silently disappear from the live output.

### Added
* `Tokenizer::decode_to_bytes(ids) -> Vec<u8>` — raw byte stream, no
  UTF-8 validation.
* `Tokenizer::decode_lossy(ids) -> String` — replaces a trailing
  incomplete UTF-8 suffix with `U+FFFD`; keeps internal multi-byte
  characters intact.
* `sonar-project.properties` and
  `.github/workflows/sonar.yml` matching the
  `nangman-crypto-research` SonarQube pattern (quality + sonar
  pipeline, `cargo-llvm-cov` lcov, Quality Gate).

### Changed
* `.github/workflows/ci.yml` removed; its checks (fmt, test, clippy)
  are now performed by the `quality` job in the new SonarQube
  workflow.
* Clippy now passes `cargo clippy --all-targets -- -D warnings`
  cleanly: 23 fixes (6× `manual_is_multiple_of`, 12× kernel-loop
  `needless_range_loop` allowed at module level with rationale,
  1× `too_many_arguments`, 1× `missing_safety_doc`,
  1× `doc_lazy_continuation`, plus 2 test cosmetic fixes).

## [v0.1.0-mvp] — 2026-05-24

Initial MVP. Reads the official `microsoft/bitnet-b1.58-2B-4T-gguf`
GGUF, runs the full BitNet b1.58 forward in Rust, produces text that
matches the pinned bitnet.cpp reference byte-for-byte on four
reference prompts, and gets ~1.5 tokens / second on a single Apple
Silicon core.

### Added — Stage 1: GGUF inspect
* `willamette inspect` CLI subcommand.
* `src/gguf/reader.rs` parser for GGUF v2/v3 metadata + tensor
  directory.
* `src/gguf/tensor.rs` `TensorView` (zero-copy slice into the mmap'd
  file).
* `src/gguf/types.rs` `GgmlType` enum (including the BitNet-fork
  values `I2_S=36`, `I8_S=37`, `TL1=38`, `TL2=39`).
* `src/memory/mmap.rs` `ModelMmap` wrapper.

### Added — Stage 2: Tokenizer
* `willamette tokenize` CLI subcommand.
* `src/tokenizer/byte_unicode.rs` GPT-2 byte↔unicode bijection.
* `src/tokenizer/pretokenize.rs` `LLAMA_VOCAB_PRE_TYPE_DEFAULT`
  3-regex pre-tokeniser (Stage 5-E fix).
* `src/tokenizer/bpe.rs` rank-priority BPE merger.
* `src/tokenizer/mod.rs` `Tokenizer::from_gguf_metadata` factory,
  `EncodeOptions`, byte-level BPE encode/decode.

### Added — Stage 3: I2_S layout
* [`docs/I2_S_LAYOUT.md`](docs/I2_S_LAYOUT.md) — pinned-source
  citations for the 2-bit packing, code → ternary map, scale offset.
* `TensorView::I2S_*` constants and `i2s_scale()` helper.
* `tests/i2s_layout.rs` — 210-I2S-tensor parity tests against the
  real model.

### Added — Stage 4-A: ModelConfig / ModelGraph
* `src/model/config.rs` `BitNetConfig` (loaded purely from GGUF
  metadata).
* `src/model/graph.rs` `ModelGraph` + `LayerWeights` (332 TensorView
  references, shape/dtype-checked at construction).

### Added — Stage 4-B: f32 primitives
* `src/model/primitives.rs` — `f16_to_f32`, `embedding_gather_f16`,
  `rms_norm_f32`, `apply_rope_f32` (NEOX), GQA shape helpers,
  attention scale, causal mask.

### Added — Stage 4-C: BitLinear scalar matvec
* `src/model/bitlinear.rs` `bitlinear_i2s_matvec_f32_scalar`
  (two-accumulator form, no full dequant, packed-only).
* [`docs/BITLINEAR_I2S_MATVEC.md`](docs/BITLINEAR_I2S_MATVEC.md) —
  the function contract, with `file:line` citations.

### Added — Stage 4-D: forward pass
* `src/model/attention.rs` single-token GQA attention at position 0.
* `src/model/ffn.rs` parallel-gated `ReLU²` FFN (per
  `LLM_FFN_RELU_SQR` + `LLM_FFN_PAR`).
* `src/model/block.rs` single transformer block with both residuals.
* `src/model/forward.rs` 30-layer single-token forward.
* `src/model/lm_head.rs` logits from tied `token_embd.weight` (F16).

### Added — Stage 5: Generation
* `willamette run` CLI subcommand with greedy + sampled decoding.
* `willamette logits` CLI subcommand (top-k dump).
* `src/model/multi_forward.rs` no-cache multi-token causal forward
  (Stage 5-B).
* `src/model/kv_cache.rs` + `src/model/cached_forward.rs` KV cache +
  incremental forward (Stage 5-C).
* `src/model/generate.rs` `greedy_generate_no_cache`,
  `greedy_generate_with_cache`, `generate_with_cache_and_sampler`.
* `src/model/sampler.rs` `Sampler` with temperature, top-k, top-p,
  repetition penalty, seedable xorshift PRNG (Stage 5-D).

### Added — Stage 5-E: Reference compatibility
* [`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md)
  with the four-prompt comparison.
* `scripts/run_willamette_reference.sh`,
  `scripts/run_bitnet_reference.sh`,
  `scripts/compare_reference.sh`.
* **Fix**: rewritten pre-tokeniser to apply the
  `LLAMA_VOCAB_PRE_TYPE_DEFAULT` 3-regex set (was a single GPT-2
  regex). Without this, `"1 + 1 ="` tokenised differently than
  bitnet.cpp.

### Added — Stage 6-A: scalar baseline benchmark
* `willamette bench` CLI subcommand.

### Added — Stage 6-C: Apple Silicon NEON
* `src/model/bitlinear_neon.rs` — 16-element NEON dot product with 4
  parallel `float32x4_t` accumulators.
* Runtime dispatch in `src/model/bitlinear.rs` via
  `is_aarch64_feature_detected!("neon")`.
* `tests/bitlinear_simd.rs` — scalar↔NEON tolerance equivalence on
  every layer-0 BitLinear weight.
* 7.5× end-to-end speed-up vs. scalar.

### Added — Stage 7-A: Release hardening
* `README.md`, `LIMITATIONS.md`, `REPRODUCIBILITY.md`,
  `GOLDEN_TESTS.md`, this `CHANGELOG.md`, `ARCHITECTURE.md`.
* `.github/workflows/ci.yml` — fmt + clippy + model-less tests on
  linux-x86_64 and macos-aarch64.
* `.gitignore` — models, generated outputs, editor noise.

### Deferred
* **Stage 6-B** (x86 AVX2 / SSE2) — pending an x86 host on which the
  produced SIMD can be validated against the scalar fallback. No
  unverified SIMD merges per the project rules.
* GPU backends (Metal / CUDA / Vulkan).
* Multi-threaded BitLinear matvec.
* Long-context KV cache eviction.

### Known limitations
See [`LIMITATIONS.md`](LIMITATIONS.md). Highlights:
* Only `microsoft/bitnet-b1.58-2B-4T-gguf` with `I2_S` quant is
  supported.
* Only `tokenizer.ggml.model = "gpt2"` is supported.
* On x86 hosts the matvec runs the scalar fallback (correctness, not
  speed).
