# I2_S Tensor Layout — Stage 3 Investigation

Pinned source: see [`../UPSTREAM_PIN.md`](../UPSTREAM_PIN.md).

* `microsoft/BitNet @ 01eb415772c342d9f20dc42772f1583ae1e5b102`
* submodule `Eddie-Wang1120/llama.cpp @ 1f86f058de0c3f4098dedae2ae8653c335c868a1`

All file paths below are relative to a clone of `microsoft/BitNet` at the
pinned SHA. Stage 3 does NOT implement unpack or dequant — this document
records what the source says so Stage 4+ can act on facts rather than guesses.

## 1. ggml_type values

`3rdparty/llama.cpp/ggml/include/ggml.h:357..396`

| Symbol | Value |
| ------ | ----: |
| `GGML_TYPE_F32` | 0 |
| `GGML_TYPE_F16` | 1 |
| `GGML_TYPE_I2_S` | 36 |
| `GGML_TYPE_I8_S` | 37 |
| `GGML_TYPE_TL1` | 38 |
| `GGML_TYPE_TL2` | 39 |

`willamette inspect` against the official file reports raw `u32 = 36` for every
2-D BitLinear weight, confirming the enum match.

## 2. `LLAMA_FTYPE_MOSTLY_I2_S = 40`

`3rdparty/llama.cpp/include/llama.h:183`

```c
LLAMA_FTYPE_MOSTLY_I2_S          = 40, // except 1d tensors
```

Display name (`3rdparty/llama.cpp/src/llama.cpp:5371`):

```c
case LLAMA_FTYPE_MOSTLY_I2_S:    return "I2_S - 2 bpw ternary";
```

The "except 1d tensors" comment matches our inspect: 1-D `*_norm.weight`
tensors are kept as F32 (raw=0); only 2-D weight matrices are I2_S (raw=36).
Our file's `general.file_type = 40` is therefore exactly `LLAMA_FTYPE_MOSTLY_I2_S`.

## 3. Block size: `QK_I2_S`

`src/ggml-bitnet-mad.cpp:12..16`

```c
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#define QK_I2_S 128
#elif defined(__ARM_NEON)
#define QK_I2_S 64
#endif
```

`dequantize_row_i2_s` in `3rdparty/llama.cpp/ggml/src/ggml-quants.c:3897..3927`
**hard-codes 128 elements per on-disk block** (`blk_e = ... 128`, 4 sub-rows of
32, advance `x += 32` after each 128 elements). The ARM `QK_I2_S = 64` is the
SIMD inner-loop block, not the disk layout.

On-disk: **128 elements per 32-byte block, regardless of host architecture.**

## 4. The packed area — byte layout inside one block

`3rdparty/llama.cpp/ggml/src/ggml-quants.c:3897..3927`

```c
void dequantize_row_i2_s(const uint8_t * x, float * y, int64_t n, const float i2_scale) {
    static const float map2bit[4] = { -1.0f, 0.0f, +1.0f, 0.0f };

    int64_t done = 0;
    while (done < n) {
        const int64_t blk_e = (n - done >= 128) ? 128 : (n - done);
        // ...
        for (int gp = 0; gp < 32; ++gp) {
            const uint8_t b = x[gp];
            const uint8_t c0 = (b >> 6) & 0x3;
            const uint8_t c1 = (b >> 4) & 0x3;
            const uint8_t c2 = (b >> 2) & 0x3;
            const uint8_t c3 = (b >> 0) & 0x3;
            if (gp < cols0) y[done + 0*32 + gp] = i2_scale * map2bit[c0];
            if (gp < cols1) y[done + 1*32 + gp] = i2_scale * map2bit[c1];
            if (gp < cols2) y[done + 2*32 + gp] = i2_scale * map2bit[c2];
            if (gp < cols3) y[done + 3*32 + gp] = i2_scale * map2bit[c3];
        }
        x    += 32;
        done += blk_e;
    }
}
```

So inside a block of 128 elements (32 bytes), byte `gp` (0..32) encodes four
2-bit codes:

```
byte layout (MSB ─→ LSB):
    [ c0_high c0_low | c1_high c1_low | c2_high c2_low | c3_high c3_low ]
        (>>6)             (>>4)             (>>2)             (>>0)
```

These four codes go to **stride-32 positions** within the 128-element block:

| byte index `gp` | contributes to element positions |
| --------------- | -------------------------------- |
| 0  | 0,  32,  64,  96 |
| 1  | 1,  33,  65,  97 |
| 2  | 2,  34,  66,  98 |
| …  | … |
| 31 | 31, 63, 95, 127 |

Equivalently: think of the 128-element block as a 4×32 matrix laid out
**column-major**; byte `gp` holds column `gp` (4 elements top-to-bottom).

The matching quantizer is in `src/ggml-bitnet-mad.cpp:51..107` (active path):

```c
// inside the i-th block of QK_I2_S=128 elements:
for (int j = 0; j < QK_I2_S; j++) {
    int group_idx = j / 32;       // 0..3
    int group_pos = j % 32;       // 0..31
    uint8_t temp = (q8[i*QK_I2_S + j] << (6 - 2 * group_idx));
    i2_weight[i*32 + group_pos] |= temp;
}
```

i.e. element `j` of the block packs into bits `[6 - 2*(j/32), 7 - 2*(j/32)]`
of byte `j%32` — confirming the dequant layout exactly.

## 5. Code → ternary mapping

`3rdparty/llama.cpp/ggml/src/ggml-quants.c:3898`

```c
static const float map2bit[4] = { -1.0f, 0.0f, +1.0f, 0.0f };
```

| 2-bit code | ternary value |
| ---------: | ------------- |
| `00` (0)   | **−1.0** |
| `01` (1)   | **0.0** |
| `10` (2)   | **+1.0** |
| `11` (3)   | **0.0**  *(degenerate — never produced by the quantizer; the table makes it a defined no-op rather than UB)* |

The quantizer (`src/ggml-bitnet-mad.cpp:65..72`) confirms it never writes
`11`:

```c
if (fabs((double)(src[i])) < 1e-6) { q8[i] = 1; continue; }    // 0
q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;                  // +1 or -1
```

So the encoder produces exactly one of `{0, 1, 2}` for every element.

## 6. Scale storage — where is `i2_scale`?

This was the Stage 1 open question. The answer comes from three sources:

**(a) `quantize_i2_s` writes the scale immediately after the packed area**
(`src/ggml-bitnet-mad.cpp:142..149`):

```c
// store scale at the end of quantized data (same location pattern as quantize_i2_s)
float* scale_ptr = (float*)((char*)out + n / 4);
scale_ptr[0] = (float)i2_scale;
// return size (keep same formula as quantize_i2_s)
return nrow * row_size / 4 + 32;
```

**(b) `ggml_nbytes` overrides the on-disk size for I2_S**
(`3rdparty/llama.cpp/ggml/src/ggml.c:3485..3492`):

```c
if (tensor->type == GGML_TYPE_I2_S || tensor->type == GGML_TYPE_TL1) {
    nbytes = nbytes / 4 + 32;
}
```

**(c) The quantize wrapper applies the same offset**
(`3rdparty/llama.cpp/ggml/src/ggml.c:22692..22696`):

```c
if (type == GGML_TYPE_I2_S) {
    result = nrows * row_size / 4 + 32;
} else {
    GGML_ASSERT(result == nrows * row_size);
}
```

### Conclusion — I2_S tensor on-disk footprint

For an I2_S tensor with `n_elements = prod(shape)`:

```
total_disk_bytes = packed_bytes + trailing_scale_block
                 = (n_elements / 4) + 32

  packed_bytes        = n_elements / 4
                      = (n_elements / 128) * 32          // QK_I2_S=128 blocks, 32B each
  trailing_scale_block = 32 bytes
      └─ first 4 bytes  : `i2_scale` as f32 (per-tensor scalar)
      └─ next  28 bytes : padding for 32-byte alignment of the next tensor
```

**Per-block scale: there is none.** I2_S uses a **single per-tensor scale**
stored once at the end of each tensor's data area.

### Verified against the official GGUF file

For every consecutive pair of tensors in
`ggml-model-i2_s.gguf`, the byte gap between offsets equals our derivation:

| Type | Tensors | Gap pattern | Trailing bytes |
| ---- | ------: | ----------- | -------------- |
| I2_S | 210 | `gap == n_elements/4 + 32` | always `32` |
| F32  | 120 | `gap == n_elements * 4` | always `0` |

(The 121st F32 tensor `output_norm.weight` is last in the file and has no
"next" gap to compare.)

## 7. Relationship to `*_sub_norm.weight`

`bitnet-b1.58.attention.*` plus the per-block tensor inventory shows two
F32 normalization tensors PER BitLinear (`attn_sub_norm`, `ffn_sub_norm`).

These are **NOT the same thing as `i2_scale`**:

* `i2_scale` is a **scalar (one f32)** stored in the I2_S tensor's trailing 32 bytes; it converts the ternary codes to per-tensor floats.
* `*_sub_norm.weight` is a **per-channel RMSNorm gain vector** (shape `[2560]` for attn paths, `[6912]` for ffn_down's input). It is applied *inside the BitLinear forward pass* — the canonical BitNet b1.58 paper calls it the pre-quantization sub-LN.

The two compose, but neither stands in for the other. Both are required by
Stage 4 forward-pass code.

## 8. Block struct in `ggml-common.h` (GPU-only, not the disk layout)

`3rdparty/llama.cpp/ggml/src/ggml-common.h:271..273`:

```c
typedef struct {
    uint8_t qs[QK_K/4];      // quants
} block_i2_s;
static_assert(sizeof(block_i2_s) == QK_K/4, "wrong gpu i2_s block size/padding");
```

Here `QK_K = 256` (line 72), so `sizeof(block_i2_s) = 64` bytes. The comment
`"wrong gpu i2_s block size/padding"` and the absence of a scale field
indicate this struct is for the GPU (CUDA/HIP) kernel that re-packs the
data into 256-element K-blocks at runtime. **It does not describe the
on-disk layout** — the actual file uses 128-element / 32-byte CPU blocks as
per §4. We can ignore `block_i2_s` for CPU inference work.

## 9. `block_size = 1` in `type_traits[I2_S]`

`3rdparty/llama.cpp/ggml/src/ggml.c:1172..1186`:

```c
[GGML_TYPE_I2_S] = {
    .type_name = "i2_s",
    .blck_size = 1,
    .type_size = sizeof(int8_t),
    .is_quantized = true,
    .to_float = (ggml_to_float_t) dequantize_row_i2_s,
    ...
}
```

`blck_size = 1` and `type_size = 1` here are **intentional placeholders**.
The real size is computed by the override in `ggml_nbytes` (§6b), not by the
generic `n_elements * type_size` formula. Anyone reading this enum to derive
sizes without also reading the override will get the wrong answer.

## 10. `output.weight` absence — Stage 4 open question

Our inspect found no `output.weight` tensor — only `token_embd.weight`
(F16) at the top of the directory and `output_norm.weight` (F32) at the
end. The conventional LLaMA reading is that the lm_head is weight-tied to
the input embedding when `output.weight` is absent, but `llama.cpp/src/llama.cpp`
contains many architecture-specific branches and we have not yet read the
`bitnet-b1.58` model-loading path to confirm.

**Stage 4 will:**

1. Find the BitNet b1.58 model loading branch in `llama.cpp` (search for the
   architecture string `"bitnet-b1.58"`).
2. Read whether it ties `lm_head` to `token_embd.weight` or expects a
   separate output tensor.
3. Record the answer here.

Until that is done, do not assume tying.

## 11. Effect on `src/gguf/reader.rs` (current state)

`compute_tensor_byte_len(shape, BitNetI2S)` in our reader returns
`n_elements / 128 * 32 = n_elements / 4`. That is the **packed-area size
only**; the trailing 32-byte scale block is NOT included in
`tensor.byte_len` or in `tensor.data`.

This is **intentionally conservative** for Stage 3: the field
`tensor.data` currently exposes the 2-bit codes alone, with no risk of a
consumer mis-reading the scale region as more codes. When Stage 4 needs the
scale, it should use the new helper `TensorView::i2s_scale_offset()` (added
alongside this document) rather than re-deriving the layout.

The actual on-disk footprint of each I2_S tensor is `n_elements / 4 + 32`,
which is verified by tests in `tests/i2s_layout.rs`.

## 12. Citation table — quick reference

| Topic | File : line |
| ----- | ----------- |
| `GGML_TYPE_I2_S = 36` | `3rdparty/llama.cpp/ggml/include/ggml.h:393` |
| `LLAMA_FTYPE_MOSTLY_I2_S = 40` | `3rdparty/llama.cpp/include/llama.h:183` |
| I2_S display name `"I2_S - 2 bpw ternary"` | `3rdparty/llama.cpp/src/llama.cpp:5371` |
| `QK_I2_S = 128` (x86) / `64` (NEON) | `src/ggml-bitnet-mad.cpp:12..16` |
| Disk block: 128 elements / 32 bytes | `3rdparty/llama.cpp/ggml/src/ggml-quants.c:3897..3927` |
| Column-stride-32 byte→element mapping | same |
| Code → ternary table `{-1, 0, +1, 0}` | `3rdparty/llama.cpp/ggml/src/ggml-quants.c:3898` |
| Scale write: `(char*)out + n/4` | `src/ggml-bitnet-mad.cpp:142..143` |
| `nbytes = n/4 + 32` override | `3rdparty/llama.cpp/ggml/src/ggml.c:3485..3492` |
| `quantize_i2_s` size return | `3rdparty/llama.cpp/ggml/src/ggml.c:22692..22696` |
| `type_traits[I2_S]` (placeholder sizes) | `3rdparty/llama.cpp/ggml/src/ggml.c:1172..1186` |
| GPU-only `block_i2_s` struct | `3rdparty/llama.cpp/ggml/src/ggml-common.h:271..273` |
| BitNet runtime extras struct | `include/ggml-bitnet.h:18..24` |
