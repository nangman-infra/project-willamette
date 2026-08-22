# Golden Tests — Project Willamette v0.13.0-mvp

*Last revised 2026-08-22.*

Reference outputs that future code changes must preserve. If anything
in this file regresses, the change is wrong by construction — either
the model file, the upstream pin, the tokenizer fix, or this document
must change first (with an explicit "verified against new file" entry
in [`UPSTREAM_PIN.md`](UPSTREAM_PIN.md)).

Reproduce all of these in one go:

```bash
./scripts/run_willamette_reference.sh   # ~5 minutes (NEON)
./scripts/run_bitnet_reference.sh        # needs the upstream build (see REPRODUCIBILITY.md §6)
./scripts/compare_reference.sh           # writes compat_report.md
```

## Prompt 1 — `hello`

* Tokenizer (Willamette **and** bitnet.cpp): `[128000, 15339]`
  * `128000` → `<|begin_of_text|>` (BOS)
  * `15339` → `hello`
* Greedy first-token argmax: `1917` (`" world"`), logit ≈ `11.59`
* Greedy 5-token generation (Willamette ids): `[1917, 198, 262, 3270, 262]`
* Greedy 5-token text (both): `" world\n    \"\"\"\n   "` (18 bytes)
* Note: bitnet.cpp's 5 actual generated token ids are not directly
  observable from `llama-cli` (no logit dump); re-tokenising its
  generated text yields a 6-token canonical sequence
  `[1917, 198, 257, 12885, 198, 262]` which encodes the **same 18
  bytes** — see [`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md) §5
  for the BPE-segmentation explanation.

## Prompt 2 — `안녕하세요`

* Tokenizer (both): `[128000, 101193, 124409]`
  * `101193` → `ìķĪ` (the byte-unicode form of `안`)
  * `124409` → `ëħķíķĺìĦ¸ìļĶ` (`녕하세요`)
* Greedy first-token argmax: `11` (`","`)
* Greedy 5-token generation (Willamette ids): `[11, 19668, 62, 17, 13]`
* Greedy 5-token text (both): `", NAME_2."`

## Prompt 3 — `The capital of France is`

* Tokenizer (both): `[128000, 791, 6864, 315, 9822, 374]`
  * `791` → `The`
  * `6864` → `Ġcapital`
  * `315` → `Ġof`
  * `9822` → `ĠFrance`
  * `374` → `Ġis`
* Greedy first-token argmax: `12366` (`" Paris"`), logit ≈ `18.95`
  (with a 4.99-logit margin over the runner-up `" not"` at 13.96 —
  Willamette ranks `" Paris"` first with overwhelming confidence)
* Greedy 5-token generation (Willamette ids): `[12366, 13, 12366, 374, 264]`
* Greedy 5-token text (both): `" Paris. Paris is a"`

## Prompt 4 — `1 + 1 =`

* Tokenizer (both, **after the Stage 5-E pre-tokenizer fix**):
  `[128000, 16, 220, 10, 220, 16, 220, 28]`
  * `16` → `1`, `220` → `Ġ` (space), `10` → `+`, `28` → `=`
* This prompt is the one that surfaced the
  `LLAMA_VOCAB_PRE_TYPE_DEFAULT` 3-regex pre-tokeniser mismatch. The
  pre-fix Willamette tokenisation
  `[128000, 16, 489, 220, 16, 284]` (with merged `" +"` / `" ="`
  tokens) is incorrect and must NOT come back.
* Greedy first-token argmax: `220` (`" "` — space), logit ≈ `15.75`
* Greedy 5-token generation (Willamette ids): `[220, 17, 198, 17, 220]`
* Greedy 5-token text (both): `" 2\n2 "` — i.e. `" "`, `"2"`, `"\n"`,
  `"2"`, `" "`.

## Numerical sanity invariants

Even without running the full forward you can verify (Stage 4-D5):

* `argmax(logits(forward("The capital of France is")))` must be `12366`.
* Top-5 logits for that prompt must include both `" Paris"` and `"Paris"`
  (one with leading space, one without) — the model's certainty about
  Paris should be unambiguous.
* `argmax(logits(forward("1 + 1 =")))` must be `220` (space). If it
  ever becomes `17` (`"2"`) directly, either the pre-tokeniser
  regressed (back to merging `" ="` into one token) or the forward
  itself diverged.

## Classic Llama F16

Pinned llama.cpp `704485942ab54bbbbf1f241b3550ffba35f5f37e` and Willamette
produce identical prompt and greedy token IDs:

* `stories260K.F16.gguf`, `"One day"`: prompt `[1, 385, 328]`,
  generation `[432, 261, 376, 298, 315]`, text `", a little gir"`.
* `stories15M.F16.gguf`, `"One day, Timmy went to"`: prompt
  `[1, 3118, 2462, 29892, 7870, 1357, 3512, 304]`, generation
  `[278, 14089, 411, 670, 16823, 29889, 940, 4446, 263, 4802]`, text
  `" the park with his mom. He saw a big"`.

These gates cover SentencePiece BPE, separate `output.weight`, F16 Linear,
normal RoPE, GQA/MHA, SiLU/SwiGLU, and cached autoregressive generation.
Run them with `cargo test --test llama_f16 -- --ignored` after downloading and
verifying the artifacts in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
The 260K tokenizer and five-token generation golden also passed on the antix1
i686/Pentium-M host on 2026-08-11 using the musl-static release build.

### SmolLM-135M-Instruct F16, Q4_0, and Q8_0

Pinned llama.cpp and Willamette produce identical plain-completion inputs and
greedy output:

* `"Question: What is 84 * 3 / 2?"` tokenizes to
  `[17872, 42, 1812, 314, 216, 40, 36, 1672, 216, 35, 2272, 216, 34, 47]`.
* `"Question: What is 2 + 2? Answer:"` tokenizes to
  `[17872, 42, 1812, 314, 216, 34, 1232, 216, 34, 47, 19842, 42]`.
* Greedy generation is `[216, 36]`, text `" 4"`, followed by EOS for all three
  artifacts. The Q4_0 file keeps its tied embedding/lm-head at Q8_0 and uses
  Q4_0 for all transformer linears.

The same result passed on antix1 i686/Pentium-M. Additional same-host quality
smokes returned `"The capital of France is Paris."` and a coherent explanation
of why the sky is blue. F16, Q4_0, and Q8_0 also produced the identical sky-prompt IDs
`[378, 6376, 314, 4461, 975, 282, 260, 24484, 282, 1420]` on Apple M4,
HP ProBook 430 G6, mbp2012, and antix1. These are behavior examples, not broad
quality-equivalence claims for a 135M-parameter model.

### SmolLM2-360M-Instruct Q8_0

The pinned ChatML prompt `What is the capital of France? Answer in one
sentence.` uses vocabulary-resolved system/user markers and generates
`[504, 3575, 282, 4649, 314, 7042, 30]`, or `"The capital of France is
Paris."`. This exact golden passes with both the default SIMD kernel and the
`willamette_q8_scalar` control.

Q8_0 SIMD changes reduction order and does not promise universal cross-kernel
greedy identity. The longer sky prompt reverses the first two candidates
between M4 NEON and scalar. `tests/q8_simd_parity.rs` therefore gates the pinned
first-step candidate, logit, and margin envelope and retains a 120-step trace
for diagnosis.

## Backend equivalence (Apple Silicon NEON)

For each of the seven BitLinear weights in `blk.0` (with realistic
embedding → RMSNorm input), `bitlinear_i2s_matvec_f32_neon` and
`bitlinear_i2s_matvec_f32_scalar` agree to at least
`max_abs_diff ≤ 1e-2` per output element. The hard cap is enforced
by `tests/bitlinear_simd.rs`; if it's tripped, do not "loosen the
tolerance" — investigate the kernel.

## How this file ages

This file is short and concrete on purpose. When the upstream model
or upstream tokenizer changes (which would change the SHA256 in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)), this entire file becomes
stale at once and must be regenerated by re-running the comparison
scripts. We deliberately do NOT hard-code any token id without
either showing it in [`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md)
or pairing it with a citation.
