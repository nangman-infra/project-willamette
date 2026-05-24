# Upstream Pin — microsoft/BitNet

This file records the exact upstream sources that justify the values currently
hard-coded in Project Willamette (`ggml_type` enum numbers, expected tensor
layouts, expected metadata keys). Pair this with `REFERENCE_COMMIT.md`, which
holds the Stage 1 inspect verification against the official GGUF file.

If any upstream value changes (renames, new ftype constants, packing
re-layout), this pin is updated **before** the codebase is updated, and the two
must be changed in a single commit that cites the new SHA.

## Pinned commits

| Repo | Branch | SHA | Resolved on |
| ---- | ------ | --- | ----------- |
| `microsoft/BitNet` | `main` | `01eb415772c342d9f20dc42772f1583ae1e5b102` | 2026-05-24 |
| `Eddie-Wang1120/llama.cpp` (submodule `3rdparty/llama.cpp`) | _detached_ | `1f86f058de0c3f4098dedae2ae8653c335c868a1` | 2026-05-24 |

Reproduce:

```bash
git clone --depth=1 --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/BitNet.git /tmp/bitnet-upstream
cd /tmp/bitnet-upstream && git rev-parse HEAD
cd 3rdparty/llama.cpp && git rev-parse HEAD
```

Both SHAs above should match exactly.

## Pinned GGUF model file

| Field | Value |
| ----- | ----- |
| HuggingFace repo | `microsoft/bitnet-b1.58-2B-4T-gguf` |
| File | `ggml-model-i2_s.gguf` |
| Size | 1,187,801,280 bytes (1.106 GiB) |
| SHA256 | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| Verified | 2026-05-23 via `shasum -a 256` |

## ggml_type enum — source of truth

File: `3rdparty/llama.cpp/ggml/include/ggml.h` (lines 357..396 at the pinned
submodule SHA).

| Symbol | Value | File / line |
| ------ | ----: | ----------- |
| `GGML_TYPE_F32`  | `0`  | ggml.h:357 |
| `GGML_TYPE_F16`  | `1`  | ggml.h:358 |
| `GGML_TYPE_I2_S` | `36` | ggml.h:393 |
| `GGML_TYPE_I8_S` | `37` | ggml.h:394 |
| `GGML_TYPE_TL1`  | `38` | ggml.h:395 |
| `GGML_TYPE_TL2`  | `39` | ggml.h:396 |

`src/gguf/types.rs` in this repo MUST agree with these values. Any new ggml
type added upstream that we want to support requires updating both this table
and `from_raw`/`to_raw` in the same commit.

## I2_S / TL1 / TL2 implementation candidates

These are the files we will read (not copy) during Stage 3 when we reverse the
exact tensor layout for packed BitNet matmul. **No code from these files has
been ported into the Rust runtime yet** — they are listed here so that when
Stage 3 begins, the investigation has a known starting point.

### Block size constants

* `src/ggml-bitnet-mad.cpp:12..16` — `QK_I2_S = 128` for x86 (AVX/AVX2/AVX512/SSSE3), `QK_I2_S = 64` for ARM NEON.

### I2_S quantize / dequantize

* `3rdparty/llama.cpp/ggml/src/ggml-quants.c:3552` — `quantize_i2_s(...)` (commented prototype near here)
* `3rdparty/llama.cpp/ggml/src/ggml-quants.c:3897` — `dequantize_row_i2_s(const uint8_t * x, float * y, int64_t n, const float i2_scale)`
* `src/ggml-bitnet-mad.cpp:51`  — `quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights)`
* `src/ggml-bitnet-mad.cpp:142` (comment) — "store scale at the end of quantized data (same location pattern as `quantize_i2_s`)" — strong hint that I2_S layout includes trailing scale bytes; **do not act on this hint until Stage 3 reads the actual code path**.

### TL1 / TL2 (LUT-based) kernels

* `src/ggml-bitnet-lut.cpp` — runtime LUT initialization for ARM TL1 / x86 TL2
* `include/ggml-bitnet.h:43..46` — `ggml_qgemm_lut` declarations conditional on `GGML_BITNET_ARM_TL1` / `GGML_BITNET_X86_TL2`
* Generated header: `src/bitnet-lut-kernels.h` (built by the upstream build system; not in repo source)

### Runtime tensor extras

* `include/ggml-bitnet.h:18..24` — `struct bitnet_tensor_extra { int lut_scales_size; int BK; int n_tile_num; uint8_t * qweights; bitnet_float_type * scales; }`
* `include/ggml-bitnet.h:38` — `ggml_bitnet_transform_tensor(struct ggml_tensor * tensor)` — preprocesses a BitNet weight tensor before inference

These reveal that the BitNet runtime maintains **separate per-tensor scale
buffers** beyond what is stored in the GGUF file. Whether the scales are
recomputed from the ternary distribution at load time or read from
`*_sub_norm.weight` tensors (or both) is still an **open question — Stage 3**.

## Reference-binary build (Stage 5-E)

Building the pinned bitnet.cpp `llama-cli` from this checkout reproduces
the upstream reference inference we compare against. One-time
prerequisites and steps live in
[`docs/REFERENCE_COMPATIBILITY.md`](docs/REFERENCE_COMPATIBILITY.md) §1.
Note: `ggml-bitnet-lut.cpp` is the expensive compilation unit (heavily
templated; 20–40 minutes on Apple Silicon in `-O3 -DNDEBUG`).

## inspect.log captures

Each row is one captured run of `willamette inspect` against the pinned GGUF.
Earlier captures may show outdated cosmetic details (e.g. printed magic value);
prefer the latest row.

| Captured | binary | inspect.log SHA256 | lines | Notes |
| -------- | ------ | ------------------ | ----: | ----- |
| 2026-05-23 | first Stage 1 build | _not retained_ | 375 | printed `Magic: 0x46475547` (display typo; comparison logic was always correct) |
| 2026-05-24 | post error.rs/main.rs magic typo fix | `d85f8b11a743e7d339785449d098c55efdb56d2a009fcb876f1cdba60cb34088` | 375 | printed `Magic: 0x46554747` (matches `GGUF_MAGIC` constant). Tensor directory unchanged from previous capture. |

## Stage 1 inspect summary (cross-reference)

The pinned values above were sanity-checked against the actual model file:

| Verified hypothesis | Evidence | Result |
| ------------------- | -------- | ------ |
| `I2_S == 36` upstream symbol matches actual raw u32 in the GGUF | inspect.log shows 210 BitLinear tensors with `raw_u32 = 36` | ✅ |
| `general.architecture = "bitnet-b1.58"` | inspect.log line 12 | ✅ |
| `tokenizer.ggml.model = "gpt2"` (byte-level BPE) | inspect.log line 23 | ✅ |
| GGUF magic `0x46475547`, version `3`, alignment `32` | inspect.log lines 5-9 | ✅ |
| 332 tensors, byte-len math (`n_elements / 4` for I2_S) matches file exactly | exit 0, no `TensorOutOfBounds` | ✅ |
| `general.file_type = 40` decodes to a BitNet ftype | _open_ | ⏸ Stage 3 |
| I2_S per-tensor scale storage location | _open_ — upstream comment suggests trailing bytes after each row, but our byte-len matched without them | ⏸ Stage 3 |
| lm_head weight tying with `token_embd.weight` | no separate `output.weight` in tensor directory | ⏸ Stage 4 |

Inspect run date: 2026-05-23. CLI revision: first commit after fake-code purge.

## Update procedure

When advancing the pin (e.g. upstream renames a type or adds I3_S):

1. Run `git ls-remote https://github.com/microsoft/BitNet HEAD` and the
   submodule equivalent.
2. Update the SHAs in the table above with `Resolved on` set to today.
3. Re-verify the ggml_type enum table by greppting `GGML_TYPE_(I2_S|I8_S|TL1|TL2)` in the new `ggml.h`.
4. Update `src/gguf/types.rs` only if the values changed.
5. Run `willamette inspect` against the official GGUF, append a fresh row to
   the verification table in `REFERENCE_COMMIT.md`.
