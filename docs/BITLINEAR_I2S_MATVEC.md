# BitLinear / I2_S matvec — semantics

Pinned source: see [`../UPSTREAM_PIN.md`](../UPSTREAM_PIN.md).

* `microsoft/BitNet @ 01eb415772c342d9f20dc42772f1583ae1e5b102`
* submodule `Eddie-Wang1120/llama.cpp @ 1f86f058de0c3f4098dedae2ae8653c335c868a1`

This document fixes the contract that the Stage 4-C scalar reference
implementation must obey. It does not introduce SIMD, KV cache, attention,
or generation.

## 1. Tensor orientation — `[in_dim, out_dim]`

In ggml convention, `tensor.ne[0]` is the innermost (fastest-varying)
dimension. For a 2-D BitLinear weight `W` used as `y = W · x`:

* `shape[0]` = `ne[0]` = `in_dim` (column count, the dimension the matvec
  reduces along)
* `shape[1]` = `ne[1]` = `out_dim` (row count, the dimension the matvec
  produces)

Allocator citation (`src/llama.cpp:8739..8752`) — every BitLinear weight
declares its shape with `n_embd`-side first, `out`-side second:

| field | declared `create_tensor` shape | (in_dim, out_dim) |
| ----- | ------------------------------ | ----------------- |
| `wq`        | `{n_embd, n_embd_head_k * n_head}` | `(2560, 2560)` |
| `wk`        | `{n_embd, n_embd_k_gqa}`           | `(2560, 640)`  |
| `wv`        | `{n_embd, n_embd_v_gqa}`           | `(2560, 640)`  |
| `wo`        | `{n_embd_head_k * n_head, n_embd}` | `(2560, 2560)` |
| `ffn_gate`  | `{n_embd, n_ff}`                   | `(2560, 6912)` |
| `ffn_up`    | `{n_embd, n_ff}`                   | `(2560, 6912)` |
| `ffn_down`  | `{n_ff, n_embd}`                   | `(6912, 2560)` |

`llm_build_lora_mm(ctx, w, cur)` (`src/llama.cpp:9443`) calls
`ggml_mul_mat(ctx0, w, cur)`. The matmul produces `out` whose dimensions
are `(w.ne[1], cur.ne[1])`, i.e. `out_dim` rows times whatever batch
dimension `cur` carries. For a single-token forward path, `cur` is
effectively a vector of length `in_dim` and the result is a vector of
length `out_dim`. Thus all seven BitLinear roles can share **one matvec
function** with the signature `(weight, x[in_dim], out[out_dim])`.

## 2. On-disk byte layout per row

For every I2_S tensor:

* Total disk footprint = `n_elements / 4 + 32 bytes`. The trailing 32
  bytes are the scale block (§4). They live OUTSIDE `TensorView::data`
  but inside `TensorView::scale_data` after the Stage 4-C extension.
* Packed-codes area = `n_elements / 4 = (in_dim * out_dim) / 4` bytes,
  organized as `out_dim` rows of `in_dim / 4` bytes each, in row-major
  order.
* Each row of `in_dim` elements is split into `in_dim / 128` "blocks";
  each block is `QK_I2_S = 128` elements packed into exactly 32 bytes
  (§3 below).
* The packed-area length is the value of `TensorView::byte_len` produced
  by `compute_tensor_byte_len(shape, BitNetI2S)` in our reader.

`in_dim` is therefore required to be a positive multiple of 128 — the
official model satisfies this for every BitLinear weight (smallest case:
`attn_k` with `in_dim = 2560 = 20 × 128`).

## 3. Inside one 128-element / 32-byte block

`3rdparty/llama.cpp/ggml/src/ggml-quants.c:3897..3927`
(`dequantize_row_i2_s`) and `src/ggml-bitnet-mad.cpp:65..107`
(`quantize_i2_s`) together fix the layout:

* Byte index `gp ∈ 0..32` within the block holds **four** 2-bit codes:

```
   bit position 7  6  5  4  3  2  1  0
                 \__/  \__/  \__/  \__/
                  c0    c1    c2    c3
```

  * `c0 = (byte >> 6) & 0x3`
  * `c1 = (byte >> 4) & 0x3`
  * `c2 = (byte >> 2) & 0x3`
  * `c3 = (byte >> 0) & 0x3`

* The four codes go to **column-stride-32** positions inside the block
  (0-based within the block, length 128):

  | code | position in block |
  | ---- | ----------------- |
  | `c0` | `0  + gp` |
  | `c1` | `32 + gp` |
  | `c2` | `64 + gp` |
  | `c3` | `96 + gp` |

  i.e. byte `gp` contributes one element to each of the four
  contiguous 32-element sub-rows.

* Block `bk` (within a row of `in_dim / 128` blocks) starts at byte
  offset `bk * 32` within that row.

* Code-to-ternary map (`ggml-quants.c:3898`,
  `static const float map2bit[4] = { -1.0f, 0.0f, +1.0f, 0.0f }`):

  | 2-bit code | ternary value |
  | :--------: | :-----------: |
  | `00` (0) | **−1** |
  | `01` (1) | **0** |
  | `10` (2) | **+1** |
  | `11` (3) | **0** *(degenerate; quantizer at `ggml-bitnet-mad.cpp:65..72` never produces it)* |

## 4. Scale read rule

`src/ggml-bitnet-mad.cpp:142..145` (active CPU `quantize_i2_s` ACT_PARALLEL=0 path):

```c
// store scale at the end of quantized data (same location pattern as quantize_i2_s)
float* scale_ptr = (float*)((char*)out + n / 4);
scale_ptr[0] = (float)i2_scale;
// ...
return nrow * row_size / 4 + 32;
```

and `ggml.c:3485..3492` (`ggml_nbytes` override) confirms that the
on-disk size includes the extra 32 bytes.

**Rule:** for every I2_S tensor `T`, the per-tensor scale is one
little-endian `f32` at file offset `T.offset + T.byte_len` (= the start
of the 32-byte trailing block). The remaining 28 bytes of that block
are padding and MUST NOT be interpreted as data.

* **There is no per-block scale.** All `out_dim` rows of the same tensor
  share one f32.
* **There is no per-row scale.** Same as above.

The scale is applied in dequant exactly once per element
(`y = i2_scale * map2bit[code]` in `dequantize_row_i2_s:3917`). Therefore
for the matvec we can factor it out of the reduction:

```
out[j] = i2_scale * Σᵢ map2bit[code(W[j,i])] * x[i]
```

## 5. Activation type

Production CPU path: activations are int8-quantized via
`quantize_row_i8_s` (`ggml.c:13170`) before being fed to
`ggml_vec_dot_i2_i8_s` (`ggml.c:1175`). That path exists for SIMD speed.

**Reference Stage 4-C path: keep activations as f32.** Mathematically
equivalent up to rounding because ternary `{-1, 0, +1}` multiplication
distributes over either int8 or f32 inputs; for ternary weights the
quantization of activations only buys speed, not correctness. Stage 4-C
chooses correctness clarity over speed.

Concretely:

```rust
let mut pos = 0.0_f32;     // Σ x[i] where code == +1
let mut neg = 0.0_f32;     // Σ x[i] where code == -1
for i in 0..in_dim {
    match map2bit_code(W[j,i]) {
        0 => neg += x[i],   // code 00 → −1
        2 => pos += x[i],   // code 10 → +1
        _ => {}             // code 01 → 0, code 11 → degenerate 0
    }
}
out[j] = i2_scale * (pos - neg);
```

`pos − neg` is exact (no `× ±1.0` multiplications). The Stage 4-C
reference uses this two-accumulator form for numerical stability.

## 6. Relationship to `*_sub_norm.weight`

`build_bitnet_158` (`src/llama.cpp:15389..15532`) places the BitLinear
calls and the sub-norm calls as separate operations:

```
                                            uses sub_norm directly before?
  wq      = matmul(W_q,    attn_norm(x))                NO
  wk      = matmul(W_k,    attn_norm(x))                NO
  wv      = matmul(W_v,    attn_norm(x))                NO
  ...attention computation...
  wo      = matmul(W_o,    attn_sub_norm(att_out))      YES — attn_sub_norm before W_o
  ...residual...
  gate    = matmul(W_g,    ffn_norm(x))                 NO
  up      = matmul(W_up,   ffn_norm(x))                 NO
  ...FFN composition...
  down    = matmul(W_down, ffn_sub_norm(ffn_hidden))    YES — ffn_sub_norm before W_down
```

So `sub_norm` is **not part of BitLinear**. It is an ordinary RMSNorm
that some callsites apply just before the matvec call. The matvec
function therefore must NOT bake sub_norm into its body — the caller
chooses whether to RMSNorm the input first.

The Stage 4-B `rms_norm_f32` primitive already handles that
preprocessing for any caller that needs it.

## 7. The function signature for Stage 4-C

```rust
/// out_dim-length output = i2_scale * (W_ternary · in_dim-length input)
pub fn bitlinear_i2s_matvec_f32(
    weight: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), WillametteError>;
```

Rules enforced at the boundary:

1. `weight.ggml_type == GgmlType::BitNetI2S` (else `UnsupportedTensorType`).
2. `weight.shape.len() == 2` (else parse error).
3. `input.len() == weight.shape[0] as usize` (= in_dim).
4. `output.len() == weight.shape[1] as usize` (= out_dim).
5. `in_dim % 128 == 0` (QK_I2_S alignment).
6. `weight.data.len() == in_dim * out_dim / 4` (packed-area sanity).
7. `weight.scale_data` is `Some(...)` and at least 4 bytes long (scale
   present and readable).
8. The scale must be a finite (non-NaN, non-inf) f32. (If a future file
   ships an `Infinity` scale we want to know about it loudly.)

If any precondition fails, return a `WillametteError` — never silently
write zeros to `output`, never fake an answer.

## 8. What this function does NOT do (Stage 4-C non-goals)

* No full-tensor dequant — never expand all `in_dim × out_dim` to f32.
* No KV cache, no attention math, no softmax.
* No ReLU² activation, no FFN composition (`up * relu²(gate)`).
* No lm_head logits, no sampling, no generation.
* No SIMD (AVX2 / SSE2 / NEON). Stage 6 territory.
* No activation int8 quantization (only the production path needs that;
  reference keeps f32).
* No application of optional `wo_scale` / `ffn_*_scale` tensors. Our file
  doesn't ship them and the reference matvec uses only the intrinsic
  per-tensor `i2_scale` from the trailing block.

## 9. Open items deferred to later stages

* Whether the BitNet b1.58 paper's "AbsMax activation quantization"
  matters for end-to-end perplexity vs. the f32 reference path. Stage
  4-D end-to-end correctness test may surface this; if so, implement
  `quantize_row_i8_s` + integer accumulator and benchmark.
* Per-tensor `i2_scale` source provenance — is it the trained scalar
  from `i2_scale = max(|W|)` (per `ggml-bitnet-mad.cpp:60..63`), or has
  the upstream model converter applied an extra normalisation? Cross-
  check by reading a few scales out of the official file and comparing
  against the absmax of the dequantised ternary weights × scale.

## 10. Test plan

* Synthetic single-row tensor with all +1, all −1, all 0 patterns;
  trivially-checkable scale; expected matvec result derived by hand.
* Real-file integration: for each of the seven BitLinear roles in layer
  0, run the matvec on the result of `embedding_gather_f16 ∘ rms_norm_f32`
  and confirm the output is finite, non-zero, and deterministic across
  repeated runs.
