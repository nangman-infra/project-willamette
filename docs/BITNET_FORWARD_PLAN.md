# BitNet b1.58 Forward Pass — Plan & Tensor Map

This document records the model topology Stage 4-A relies on, every fact
cited against the [pinned upstream](../UPSTREAM_PIN.md) (`microsoft/BitNet @ 01eb415772c342d9f20dc42772f1583ae1e5b102`, submodule
`Eddie-Wang1120/llama.cpp @ 1f86f058de0c3f4098dedae2ae8653c335c868a1`).

Stage 4-A scope: tensor registry + ModelConfig only. **No** forward
implementation, **no** matmul, **no** I2_S dequant, **no** sampling.

## 1. Architecture identifier

| `general.architecture` (our file) | `"bitnet-b1.58"` |
| --------------------------------- | ---------------- |
| llama.cpp `LLM_ARCH_BITNET_B158` mapping | `src/llama.cpp:263` |
| Forward-graph dispatch | `src/llama.cpp:16850..16854` → `build_bitnet_158()` |
| Layer-count → model size table | `src/llama.cpp:6117..6126` (n_layer 30 → `MODEL_2B`) |

(Note: there are three closely related architectures in this fork:
`LLM_ARCH_BITNET`, `LLM_ARCH_BITNET_25`, and `LLM_ARCH_BITNET_B158`. The
official 2B-4T file is `LLM_ARCH_BITNET_B158`; `BITNET_25` shares the
*same* `build_bitnet_158()` graph, while plain `BITNET` uses a different
`build_bitnet()` builder we do not target.)

## 2. ModelConfig — metadata keys consumed

All keys verified present in our `inspect.log`:

| metadata key | type | our value | symbol in code |
| ------------ | ---- | --------: | -------------- |
| `general.architecture` | string | `"bitnet-b1.58"` | (gate) |
| `bitnet-b1.58.block_count` | u32 | 30 | `n_layer` |
| `bitnet-b1.58.embedding_length` | u32 | 2560 | `n_embd` |
| `bitnet-b1.58.feed_forward_length` | u32 | 6912 | `n_ff` |
| `bitnet-b1.58.context_length` | u32 | 4096 | `n_ctx_train` |
| `bitnet-b1.58.attention.head_count` | u32 | 20 | `n_head` |
| `bitnet-b1.58.attention.head_count_kv` | u32 | 5 | `n_head_kv` |
| `bitnet-b1.58.attention.layer_norm_rms_epsilon` | f32 | 1e-5 | `f_norm_rms_eps` |
| `bitnet-b1.58.rope.dimension_count` | u32 | 128 | `n_rot` |
| `bitnet-b1.58.rope.freq_base` | f32 | 500000 | `freq_base` |
| `bitnet-b1.58.vocab_size` | u32 | 128256 | `n_vocab` |

Derived:
* `head_dim = embedding_length / head_count = 2560 / 20 = 128` (matches `rope.dimension_count`)
* `kv_dim   = head_dim * head_count_kv     = 128  * 5  = 640` (GQA, 4:1 ratio)
* `n_ff     = 6912` (asymmetric — FFN expansion ratio 2.7×, not the usual 4×)

## 3. Tensor name table for LLM_ARCH_BITNET_B158

`src/llama.cpp:1382..1408`:

```cpp
{ LLM_ARCH_BITNET_B158, {
    { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
    { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
    { LLM_TENSOR_OUTPUT,          "output" },
    { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
    { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
    { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
    { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
    { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
    { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
    { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
    { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
    { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
    { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
    { LLM_TENSOR_ATTN_SUB_NORM,   "blk.%d.attn_sub_norm" },
    { LLM_TENSOR_FFN_SUB_NORM,    "blk.%d.ffn_sub_norm" },
    /* MoE/expert/bias entries — not used by our model */
}, },
```

## 4. Tensor allocation policy & weight tying

`src/llama.cpp:8717..8750` (LLM_ARCH_BITNET_B158 / LLM_ARCH_BITNET_25):

```cpp
model.tok_embd = ml.create_tensor(ctx_input,        tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

// output
{
    model.output_norm = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
    model.output      = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab},
                                          llama_model_loader::TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (model.output == NULL) {
        model.output = ml.create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab},
                                         llama_model_loader::TENSOR_DUPLICATED);
    }
}
```

### Weight-tying conclusion

* `output.weight` is **optional** for `LLM_ARCH_BITNET_B158` (the
  `TENSOR_NOT_REQUIRED` flag).
* If absent (our case), `model.output` is aliased to `token_embd.weight`
  via `TENSOR_DUPLICATED` (alias, not copy).
* **Critically**, the forward graph `build_bitnet_158()` does NOT read
  `model.output` at all — it reads `model.tok_embd` directly. See §6
  for the exact line.

So **for any file produced by this loader, the lm_head projection uses
`token_embd.weight`** regardless of whether a separate `output.weight`
tensor exists in the file. Our file has none, so the policy is uniform.

Per-layer allocations (same source block, lines 8736..8760):

| field | tensor name | shape |
| ----- | ----------- | ----- |
| `layer.attn_norm` | `blk.N.attn_norm.weight` | `{n_embd}` |
| `layer.attn_sub_norm` | `blk.N.attn_sub_norm.weight` | `{n_embd}` |
| `layer.ffn_sub_norm` | `blk.N.ffn_sub_norm.weight` | `{n_ff}` |
| `layer.wq` | `blk.N.attn_q.weight` | `{n_embd, n_embd_head_k * n_head}` = `{2560, 2560}` |
| `layer.wk` | `blk.N.attn_k.weight` | `{n_embd, n_embd_k_gqa}` = `{2560, 640}` |
| `layer.wv` | `blk.N.attn_v.weight` | `{n_embd, n_embd_v_gqa}` = `{2560, 640}` |
| `layer.wo` | `blk.N.attn_output.weight` | `{n_embd_head_k * n_head, n_embd}` = `{2560, 2560}` |
| `layer.ffn_norm` | `blk.N.ffn_norm.weight` | `{n_embd}` |
| `layer.ffn_gate` | `blk.N.ffn_gate.weight` | `{n_embd, n_ff}` = `{2560, 6912}` |
| `layer.ffn_up`   | `blk.N.ffn_up.weight`   | `{n_embd, n_ff}` = `{2560, 6912}` |
| `layer.ffn_down` | `blk.N.ffn_down.weight` | `{n_ff, n_embd}` = `{6912, 2560}` |

(Optional bias `bq`/`bk`/`bv`/`bo`/`ffn_gate_b`/`ffn_down_b`/`ffn_up_b`
and `rope_freqs.weight` are flagged `TENSOR_NOT_REQUIRED`. Our file
does NOT include any of them — verified by inspect.log: 332 = 1 + 30×11 + 1
with no extras.)

Total tensor count: `1 (token_embd) + 30 × 11 + 1 (output_norm) = 332` ✓

## 5. Per-tensor dtype expectations (from `LLAMA_FTYPE_MOSTLY_I2_S = 40`, "except 1d tensors")

| Shape rank | Example | Expected dtype |
| ---------- | ------- | -------------- |
| 1 | `attn_norm.weight` `[2560]` | **F32** (raw 0) |
| 1 | `attn_sub_norm.weight` `[2560]` | F32 |
| 1 | `ffn_norm.weight` `[2560]` | F32 |
| 1 | `ffn_sub_norm.weight` `[6912]` | F32 |
| 1 | `output_norm.weight` `[2560]` | F32 |
| 2 | `attn_q/k/v/output.weight` | **I2_S** (raw 36) |
| 2 | `ffn_gate/up/down.weight` | I2_S |
| 2 (special) | `token_embd.weight` `[2560, 128256]` | **F16** (raw 1) — embeddings are kept in F16 even under `MOSTLY_I2_S` |

Verified by `inspect.log` tensor table: counts 210 I2_S / 121 F32 / 1 F16.

## 6. Forward pass — operation order

`src/llama.cpp:15389..15532` (`build_bitnet_158`). Reproduced here in plain
form so Stage 4-B / Stage 4-C can implement each primitive in isolation.

```
inpL = embed(input_ids) using model.tok_embd
inp_pos = positional indices
KQ_mask = causal attention mask

for il in 0..n_layer:
    inpSA = inpL

    // attention block
    cur = RMSNorm(inpL, attn_norm[il], rms_eps)
    Qcur = matmul(wq[il], cur)
    Kcur = matmul(wk[il], cur)
    Vcur = matmul(wv[il], cur)
    Qcur = reshape(Qcur, [head_dim, n_head,    n_tokens])
    Kcur = reshape(Kcur, [head_dim, n_head_kv, n_tokens])
    Qcur = RoPE(Qcur, inp_pos, freq_base=500000, n_rot=128)
    Kcur = RoPE(Kcur, inp_pos, freq_base=500000, n_rot=128)
    cur  = scaled_dot_product_attention(Qcur, Kcur, Vcur, KQ_mask,
                                         kv_cache,
                                         scale = 1/sqrt(head_dim))
    cur  = RMSNorm(cur, attn_sub_norm[il], rms_eps)     // ★ post-attention sub_norm
    cur  = matmul(wo[il], cur)

    ffn_inp = cur + inpSA                                // residual #1

    // ffn block
    cur = RMSNorm(ffn_inp, ffn_norm[il], rms_eps)
    cur = FFN_RELU_SQR_parallel(                         // ReLU² gated, LLM_FFN_PAR
              up   = matmul(ffn_up[il],   cur),
              gate = matmul(ffn_gate[il], cur)
          )
    cur = RMSNorm(cur, ffn_sub_norm[il], rms_eps)        // ★ post-up/gate sub_norm
    cur = matmul(ffn_down[il], cur)

    inpL = cur + ffn_inp                                 // residual #2

cur    = RMSNorm(inpL, output_norm, rms_eps)
logits = matmul(model.tok_embd, cur)                     // ★ lm_head = tok_embd
```

Critical pinned-source citations for the steps that diverge from "vanilla
LLaMA":

* **`attn_sub_norm` placement** — applied to the attention OUTPUT before the
  `wo` projection. `src/llama.cpp:15469` (line "attn_sub_norm" callback).
* **`ffn_sub_norm` placement** — applied to the FFN hidden state (after
  the gated up-projection and activation, before `wo`-equivalent `ffn_down`).
  `src/llama.cpp:15506..15510`.
* **FFN activation = ReLU² ("ReLU-squared")** — `LLM_FFN_RELU_SQR`.
  `src/llama.cpp:15502`. This is the BitNet b1.58 paper's choice; SiLU
  (the LLaMA default) is NOT what this model expects.
* **FFN topology = parallel gated** — `LLM_FFN_PAR`.
  `src/llama.cpp:15502`. So the formula is `down(ffn_up(x) ⊙ act(ffn_gate(x)))`.
* **lm_head = `model.tok_embd`** — `src/llama.cpp:15527`:
  ```cpp
  cur = llm_build_lora_mm(lctx, ctx0, model.tok_embd, cur);
  ```
  Even though `model.output` was allocated (tied to tok_embd when missing),
  the build always reads `model.tok_embd` directly. **Weight tying is
  unconditional for this architecture.**
* **RoPE: full head_dim rotated** — `n_rot = n_embd_head = 128`, no
  partial rotation.
  `src/llama.cpp:15396..15397` (`GGML_ASSERT(n_embd_head == hparams.n_rot)`).

## 7. Stage 4-B status (2026-05-24)

Implemented in [`src/model/primitives.rs`](../src/model/primitives.rs):

| Primitive | Signature highlight | Status |
| --------- | ------------------- | ------ |
| `f16_to_f32(u16) -> f32` | IEEE 754 binary16 → binary32, includes subnormal / ±inf / NaN | ✅ |
| `embedding_gather_f16(tensor, token_id, &mut out)` | Reads one F16 row from `token_embd.weight`, decodes to f32 | ✅ |
| `rms_norm_f32(x, w, eps, &mut out)` | Standard RMSNorm; ε from `BitNetConfig::layer_norm_rms_epsilon` | ✅ |
| `apply_rope_f32(x, head_dim, n_rot, pos, base, RopeType)` | Rotation in place; both `Norm` and `Neox` pairings supported | ✅ |
| `AttentionShape::from_config(n_heads, n_kv_heads, head_dim)` | Derives `group_size`, `q_per_token_dim`, `kv_per_token_dim` | ✅ |
| `attention_scale(head_dim) -> f32` | `1/√head_dim` | ✅ |
| `gqa_group_size`, `kv_head_for_q_head` | GQA mapping helpers | ✅ |
| `causal_mask_value(q_pos, k_pos) -> f32` | `0.0` or `-∞` (added to logits) | ✅ |

### Source-pinned design notes for Stage 4-B

**RoPE type for `LLM_ARCH_BITNET_B158` is `LLAMA_ROPE_TYPE_NEOX`**, NOT
the LLaMA-style `NORM`. Citation: `src/llama.cpp:20107..20117` of the
pinned commit lists `LLM_ARCH_BITNET_B158` under the NEOX block with the
comment `"the pairs of head values are offset by n_rot/2"`. Concretely:

```
for j in 0..n_rot/2:
    θ_j = pos × freq_base^(-2j / n_rot)
    (x[j], x[j + n_rot/2]) ← rot2(θ_j, x[j], x[j + n_rot/2])
```

If anything in Stage 4-D produces wildly wrong logits, **first check
that Q/K were rotated with `RopeType::Neox`, not `Norm`** — the failure
mode is silent (Norm rotates legal pairs that just aren't the ones the
model expects).

**RMSNorm epsilon** is a single value (`bitnet-b1.58.attention.layer_norm_rms_epsilon`,
`1e-5` for our file) reused for every `LLM_NORM_RMS` callsite in
`build_bitnet_158` (`src/llama.cpp:15414`, `15467`, `15495`, `15510`,
`15524`). There is no per-tensor epsilon override.

**GQA mapping** (`group_size = n_heads / n_kv_heads = 4`): every block
of 4 consecutive Q heads attends against one shared K/V head pair. The
mapping `q_head → kv_head` is `q_head / 4`. Verified in
`tests/primitives.rs::gqa_mapping_covers_all_q_heads`.

### Stage 4-B verification (against the real GGUF)

[`tests/primitives.rs`](../tests/primitives.rs) — 13 tests, all green:

* `embedding_gather_f16` produces 2560-dim finite f32 rows for token ids
  used by the Stage 2 CLI (`hello = 15339`, Korean prefix = `101193`,
  Korean suffix = `124409`); two different token ids produce
  substantially different rows (>1000 differing dims).
* `rms_norm_f32` operates correctly against real `attn_norm`,
  `output_norm`, and the asymmetric `ffn_sub_norm` (width `n_ff = 6912`).
* `apply_rope_f32` preserves head_dim length, is identity at position 0,
  preserves per-pair norm, and produces **different** outputs from
  `RopeType::Norm` than `RopeType::Neox` (so the wrong choice is
  detectable, not silently equal).
* `AttentionShape::from_config(20, 5, 128)` returns the expected
  `(group_size=4, q_per_token_dim=2560, kv_per_token_dim=640)`.
* `attention_scale(128) = 1/√128`.
* Causal mask is lower-triangular (`0.0` on/below diagonal, `-∞` above).

## 8. Operations NOT implemented in Stage 4-A

This list is the **explicit non-goal surface area** — everything below
will be split across Stage 4-B (shape-safe primitives), 4-C (BitLinear
matmul that consumes I2_S + sub_norm + scale), and 4-D (full forward).

| Op | Where used | Status |
| --- | --- | --- |
| Token embedding lookup (F16 → f32 row gather) | start | not implemented |
| RMSNorm (f32) | every layer × 4 + final | not implemented |
| RoPE (freq_base=500000, full head_dim) | per layer in Q,K | not implemented |
| Scaled dot-product attention with causal mask + KV cache + GQA | per layer | not implemented |
| BitLinear matmul: f32 activations · I2_S weight (read 2-bit codes, apply per-tensor i2_scale, optionally apply sub_norm) | every layer ×7 + lm_head | not implemented |
| ReLU² activation | per layer | not implemented |
| Parallel-gated FFN (`up * relu²(gate)` then `ffn_down`) | per layer | not implemented |
| lm_head logits (BitLinear over tied tok_embd transpose) | end | not implemented |
| Sampling (greedy / temperature / top-k / top-p) | post-logits | not implemented |
| KV cache datastructure | attention | not implemented |
| Causal mask construction | attention | not implemented |

For Stage 4-B we will start with the safest leaves: f32-only RMSNorm and
F16→f32 embedding row gather. RoPE follows. Then a shape-only stub for
attention. BitLinear matmul (the I2_S consumer) is the last thing before
the end-to-end forward in 4-D.

## 9. Stage 4-A deliverables

* `src/model/config.rs` — `BitNetConfig`, loaded purely from GGUF metadata.
* `src/model/graph.rs` — `ModelGraph` + `LayerWeights`, holding TensorView
  references for all 332 weights. Builds shape + dtype checks at
  construction; never mutates tensors.
* `tests/model_graph.rs` — verifies, against the real GGUF:
  * Config values match this document.
  * 30 layers built, no missing tensors.
  * Per-layer dtypes: 7 × I2_S, 4 × F32.
  * `token_embd.weight` is F16, shape `[n_embd, n_vocab]`.
  * `output_norm.weight` is F32, shape `[n_embd]`.
  * `output.weight` is absent in our file.
  * `lm_head == token_embd` (tying realized).
  * Shapes match the table in §4.

## 10. Stage 4-B deliverables

* `src/model/primitives.rs` — see §7 table above.
* `tests/primitives.rs` — 13 integration tests against the real GGUF.

## 11. Open questions deferred to later stages

* **Per-tensor I2_S scale extraction** — the f32 lies at
  `tensor.offset + tensor.byte_len`, but the helper to read it lives in
  Stage 4-C alongside the BitLinear kernel, not here.
* **Whether `sub_norm` weight should be multiplied with i2_scale at load
  time** — possibility raised in the BitNet paper. Stage 4-C decision,
  with cross-check against `llm_build_lora_mm`'s actual scale application.
* **Optional `*_scale` tensors** referenced by `build_bitnet_158`
  (`wo_scale`, `ffn_up_scale`, `ffn_gate_scale`, `ffn_down_scale`).
  Absent from our file. Confirm at Stage 4-C that the BitLinear math is
  closed without them.

Every line of this document is reversible: if Stage 4-B/C/D finds a
contradiction, the relevant cell here must change in the same commit
that introduces the conflicting code, with a one-line "verified against
new file" entry added to `UPSTREAM_PIN.md`.

## 12. Forward primitive checklist for Stage 4-C / 4-D

| Op | Stage | Notes |
| --- | --- | --- |
| F16 embedding row gather | ✅ 4-B | done |
| RMSNorm (f32) | ✅ 4-B | done |
| RoPE (NEOX, full head_dim) | ✅ 4-B | done |
| GQA mapping + attention scale + causal mask | ✅ 4-B | done (shape primitives) |
| BitLinear matmul `f32 ⊗ I2_S` (read 2-bit codes, apply per-tensor scale) | ✅ **4-C** | scalar reference in `src/model/bitlinear.rs`; semantics in `docs/BITLINEAR_I2S_MATVEC.md` |
| KV cache structure + insert/read | ⏳ 4-D | grows with token position |
| Scaled dot-product attention softmax + value aggregation | ⏳ 4-D | per layer |
| ReLU² activation | ⏳ 4-D | trivial after primitives exist |
| Parallel-gated FFN composition | ⏳ 4-D | `down(up(x) ⊙ relu²(gate(x)))` |
| End-to-end forward + greedy token sampling | ⏳ 4-D | first wired-up generation |

## 13a-1. Stage 6-C — aarch64 NEON BitLinear kernel (2026-05-24)

* `src/model/bitlinear_neon.rs` — NEON kernel: per-row scalar unpack
  into an `i8` scratch buffer (column-stride-32 layout per
  `docs/I2_S_LAYOUT.md` §4) followed by 16-elements-per-iteration
  `float32x4_t` dot product with 4 parallel accumulators.
* `src/model/bitlinear.rs` — public `bitlinear_i2s_matvec_f32`
  dispatches via `is_aarch64_feature_detected!("neon")` to the NEON
  path on aarch64, or to `bitlinear_i2s_matvec_f32_scalar` otherwise.
* `tests/bitlinear_simd.rs` — 8 tests, one per BitLinear role in
  layer 0, comparing scalar vs NEON outputs against the real GGUF
  with realistic embedding+RMSNorm input. Tolerance: `max|Δ| < 1e-2`
  per element (observed `max|Δ| ≈ 1e-3` in practice on Apple M-series; measured on M4).
* Stage 5-E reference parity preserved: re-running
  `scripts/run_willamette_reference.sh` with the NEON path produces
  byte-identical generated text and identical generated token-id
  sequences as the scalar path for all four reference prompts; the
  argmax is robust to the sub-`1e-2` accumulator difference.

### Bench result (Apple Silicon aarch64, Stage 6-A scalar vs Stage 6-C NEON)

| Operation | Stage 6-A scalar | Stage 6-C NEON | Speed-up |
| --------- | ---------------: | -------------: | -------: |
| One BitLinear matvec (attn_q, 2560×2560 ternary) | 13.7 ms | **1.9 ms** | **7.3×** |
| Single-token forward (30 layers, no cache) | 5104 ms | **669 ms** | **7.6×** |
| Decode-step (with KV cache, avg of 3) | 5034 ms | **656 ms** | **7.7×** |
| Throughput (tokens/sec, decode) | 0.20 | **1.52** | **7.5×** |

End-to-end CLI smoke (3 new tokens off `hello`): ~4 s wall clock,
roughly 1 s/token with prefill + decode combined.

## 13a-2. Stages 4-D through 6-A — earlier completion roll-up (2026-05-24)

All stages from this roll-up keep the project rules: real GGUF + real
tokenizer only, no fake weights, no synthetic inference paths, packed
I2_S as the inference path.

| Stage | Deliverable | Tests |
| ----- | ----------- | ----- |
| 4-D1 | `src/model/attention.rs` — softmax / multi-head RoPE / single-token attention / `attention_block_forward_position_zero` | `tests/attention.rs` (4) + 11 unit |
| 4-D2 | `src/model/ffn.rs` — `relu_square`, parallel-gated `ffn_block_forward` | `tests/ffn.rs` (4) + 4 unit |
| 4-D3 | `src/model/block.rs` — `transformer_block_forward_position_zero` with residual #1 + #2 | `tests/block.rs` (5) |
| 4-D4 | `src/model/forward.rs` — `forward_single_token_position_zero` (30 layers + output_norm) | `tests/forward.rs` (3) |
| 4-D5 | `src/model/lm_head.rs` — `compute_logits_from_graph` against tied F16 token_embd, `argmax`, `top_k` | `tests/lm_head.rs` (5) + 4 unit |
| 5-A | CLI `Run`, `greedy_next_token_from_single_position_zero` | `tests/run_pipeline.rs` (2) |
| 5-B | `src/model/multi_forward.rs` — `multi_token_forward` (no cache, causal); `greedy_generate_no_cache` | `tests/multi_token.rs` (5) |
| 5-C | `src/model/kv_cache.rs` + `src/model/cached_forward.rs` — `KVCache` + `forward_with_cache`; `greedy_generate_with_cache` | `tests/kv_cache.rs` (3) + 6 unit (bit-exact equivalence to no-cache) |
| 5-D | `src/model/sampler.rs` — temperature, top-k, top-p, repetition penalty, seed; `generate_with_cache_and_sampler` | 7 unit (seed determinism verified) |
| 6-A | CLI `Bench` — `std::time` scalar-reference baseline (matvec / forward / decode-step) | runnable on real GGUF |

CLI smoke-test recap (Apple Silicon aarch64, `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`):

* `run --prompt "hello" --max-new-tokens 2` (greedy):
  `[1917, 198]` → `" world\n"` → full text `"hello world\n"`.
* `run --prompt "hello" --max-new-tokens 2 --temperature 0.8 --top-p 0.9 --seed 42`:
  `[0, 1268]` → `"! how"` → full text `"hello! how"`.
* Same flags, second run with the same seed: byte-identical output (`[0, 1268]`).

Scalar reference performance (Stage 6-A bench on this host, aarch64):

| Operation | Time | Note |
| --------- | ---: | ---- |
| One BitLinear matvec `attn_q` (2560×2560 ternary) | 13.7 ms | 477 M elements/sec |
| Single-token forward (30 layers, no cache) | 5.1 s | 0.2 tokens/sec |
| Decode-step (with KV cache, avg of 3) | 5.0 s | 0.2 tokens/sec |

The cache wins compound only at longer contexts — at small `position`
the BitLinear matvecs dominate and attention is a small fraction. The
~5 s/token is the **scalar reference baseline** that SIMD kernels (Stage 6-B/6-C)
must clear with measurable wins and bit-for-bit (or documented-tolerance)
correctness.

## 13. Stage 4-C deliverables (2026-05-24)

* [`docs/BITLINEAR_I2S_MATVEC.md`](BITLINEAR_I2S_MATVEC.md) — pinned BitLinear / I2_S
  semantics: tensor orientation, byte layout, 2-bit code mapping, scale
  read rule, sub_norm relationship, function contract.
* `src/gguf/tensor.rs` — extended `TensorView` with
  `scale_data: Option<&[u8]>` (32-byte trailing block for I2_S); new
  `i2s_scale() -> f32` reader.
* `src/gguf/reader.rs` — populates `scale_data` for every I2_S tensor,
  with explicit bounds check.
* `src/model/bitlinear.rs` — `bitlinear_i2s_matvec_f32(weight, input,
  output)` scalar reference + `i2s_unpack_row_to_i8` debug helper +
  `ternary_from_code` mapping. No SIMD, no full dequant, no activation
  quantization.
* `tests/bitlinear_i2s.rs` — 13 integration tests; lib unit tests cover
  synthetic-fixture matvec correctness (all +1 / all -1 / all 0 / mixed
  patterns / column-stride-32 mapping).
