# BitNet Reference Commit

Project Willamette interprets GGUF files in accordance with a specific commit of
the upstream BitNet llama.cpp / ggml fork. The values below are the source of
truth for:

* `ggml_type` enum numbers (most importantly `I2_S=36`, `I8_S=37`, `TL1=38`, `TL2=39`)
* BitNet tensor packing layouts (block size, per-tensor / per-block scales)
* tokenizer metadata layout

## Reference GGUF model (verified)

* HuggingFace repo: `microsoft/bitnet-b1.58-2B-4T-gguf`
* File: `ggml-model-i2_s.gguf`
* File size: `1,187,801,280` bytes (1.106 GiB)
* SHA256: `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
* Download date: 2026-05-23

## Upstream source

Upstream commits, file paths, and the ggml_type → number table are recorded in
[UPSTREAM_PIN.md](./UPSTREAM_PIN.md). Summary:

* `microsoft/BitNet` @ `01eb415772c342d9f20dc42772f1583ae1e5b102` (main, pinned 2026-05-24)
* Submodule `Eddie-Wang1120/llama.cpp` @ `1f86f058de0c3f4098dedae2ae8653c335c868a1`
* `GGML_TYPE_I2_S = 36` defined at `3rdparty/llama.cpp/ggml/include/ggml.h:393`

## Stage 1 verification log (2026-05-23)

`willamette inspect` was run against the verified GGUF file. Findings:

### File header

| Field | Value |
| ----- | ----- |
| Magic | `0x46475547` ("GGUF") |
| Version | `3` |
| Tensor count | `332` |
| Metadata KV count | `24` |
| Alignment | `32` bytes |

### Architecture metadata (`general.*`)

| Key | Value | Notes |
| --- | ----- | ----- |
| `general.architecture` | `"bitnet-b1.58"` | First-class BitNet arch tag, NOT `"llama"` — branching is straightforward |
| `general.name` | `"bitnet2b"` | |
| `general.file_type` | `40` (u32) | BitNet fork ftype constant. **Not yet decoded against upstream `enum llama_ftype`** — record as raw |
| `general.quantization_version` | `2` | |

### Architecture-specific metadata (`bitnet-b1.58.*`)

| Key | Value |
| --- | ----- |
| `bitnet-b1.58.block_count` | 30 |
| `bitnet-b1.58.embedding_length` | 2560 |
| `bitnet-b1.58.feed_forward_length` | 6912 |
| `bitnet-b1.58.context_length` | 4096 |
| `bitnet-b1.58.attention.head_count` | 20 |
| `bitnet-b1.58.attention.head_count_kv` | 5 (GQA 4:1, head_dim = 128) |
| `bitnet-b1.58.attention.layer_norm_rms_epsilon` | 1e-5 |
| `bitnet-b1.58.rope.dimension_count` | 128 |
| `bitnet-b1.58.rope.freq_base` | 500000 (LLaMA 3 style) |
| `bitnet-b1.58.vocab_size` | 128256 |

### Tokenizer metadata (`tokenizer.*`)

| Key | Value | Notes |
| --- | ----- | ----- |
| `tokenizer.ggml.model` | `"gpt2"` | **byte-level BPE** (LLaMA 3 family), not SentencePiece |
| `tokenizer.ggml.tokens` | array len `128256` | full vocab present |
| `tokenizer.ggml.merges` | array len `280147` | BPE merges complete |
| `tokenizer.ggml.scores` | array len `128256` | all zero (BPE doesn't use scores) |
| `tokenizer.ggml.token_type` | array len `128256` | mostly `1` (normal tokens) |
| `tokenizer.ggml.bos_token_id` | `128000` | |
| `tokenizer.ggml.eos_token_id` | `128001` | |
| `tokenizer.ggml.padding_token_id` | `128001` | |
| `tokenizer.ggml.add_bos_token` | `true` | |
| `tokenizer.chat_template` | Jinja template | uses `Human: ` / `BITNETAssistant: ` markers |

### Tensor dtype distribution

Across all 332 tensors:

| Raw u32 | Resolved | Count | Notes |
| ------- | -------- | ----: | ----- |
| `36` | `I2_S (BitNet)` | 210 | all BitLinear weights (attn_q/k/v/output, ffn_gate/up/down) |
| `0`  | `F32` | 121 | 30×4 norm layers + 1 final `output_norm.weight` |
| `1`  | `F16` | 1 | `token_embd.weight` only |
| _other_ | _none_ | 0 | **zero `Unknown(N)` tensors — every type was recognized** |

### Per-block tensor inventory (30 blocks × 11 = 330 + 2 = 332)

For each `blk.N.*` (verified identical for N = 0..29):

| Tensor name | dtype | shape |
| ----------- | ----- | ----- |
| `blk.N.attn_norm.weight` | F32 | `[2560]` |
| `blk.N.attn_sub_norm.weight` | F32 | `[2560]` |
| `blk.N.attn_q.weight` | I2_S | `[2560, 2560]` |
| `blk.N.attn_k.weight` | I2_S | `[2560, 640]` |
| `blk.N.attn_v.weight` | I2_S | `[2560, 640]` |
| `blk.N.attn_output.weight` | I2_S | `[2560, 2560]` |
| `blk.N.ffn_norm.weight` | F32 | `[2560]` |
| `blk.N.ffn_sub_norm.weight` | F32 | `[6912]` |
| `blk.N.ffn_gate.weight` | I2_S | `[2560, 6912]` |
| `blk.N.ffn_up.weight` | I2_S | `[2560, 6912]` |
| `blk.N.ffn_down.weight` | I2_S | `[6912, 2560]` |

Top-level:
* `token_embd.weight` — F16, `[2560, 128256]`
* `output_norm.weight` — F32, `[2560]`
* **No separate `output.weight` / lm_head** — assumed weight-tied with `token_embd.weight` (to be re-verified at Stage 4)

### I2_S byte-length analysis

For every I2_S tensor, our reader's formula `byte_len = n_elements / 128 * 32`
(equivalent to `n_elements / 4` = 2 bits per element) matched the file's
declared `byte_len` exactly, and all tensor offsets fell within file bounds
(no `TensorOutOfBounds` error). Samples:

| Tensor | Shape | n_elements | byte_len (file) | bits/elem |
| ------ | ----- | ---------: | --------------: | --------: |
| `attn_k` | [2560, 640] | 1,638,400 | 409,600 | 2.000 |
| `attn_q` | [2560, 2560] | 6,553,600 | 1,638,400 | 2.000 |
| `ffn_down` | [6912, 2560] | 17,694,720 | 4,423,680 | 2.000 |

**Implication for Stage 3:** I2_S tensor bytes hold *only* the packed 2-bit
ternary indices. No per-block scales are co-located in the tensor data. The
per-tensor scale (γ in the BitNet b1.58 formulation) is therefore either
absorbed into the F32 `*_sub_norm` tensors or recomputed at load time. This
**must be confirmed against the upstream `ggml-bitnet-*.cpp` source** before
Stage 3 implements packed I2_S matmul.

## Verification table (Stage 1)

| Hypothesis | Result | Evidence |
| ---------- | ------ | -------- |
| BitNet GGUF parses cleanly with version 2/3 reader | ✅ verified | version=3, exit 0, all tensors in-bounds |
| `general.architecture == "bitnet-b1.58"` (not `"llama"`) | ✅ verified | inspect.log line 12 |
| `tokenizer.ggml.model == "gpt2"` (byte-level BPE) | ✅ verified | inspect.log line 23 |
| I2_S raw u32 == 36 | ✅ verified | 210 BitLinear tensors all resolve to 36 |
| I8_S raw u32 == 37 | ⏸ not observed in this file | activation type, not present in weight file |
| TL1 raw u32 == 38 | ⏸ not observed in this file | separate TL1 GGUF variant needed |
| TL2 raw u32 == 39 | ⏸ not observed in this file | separate TL2 GGUF variant needed |
| Standard types (F32=0, F16=1) unchanged from upstream GGML | ✅ verified | norm + embed tensors |
| Per-block I2_S size = n_elements/4 (no scale bytes in tensor) | ✅ verified | exact match across 210 tensors |
| Per-tensor weight scale location | ❓ open | likely in `*_sub_norm` F32; confirm in Stage 3 |
| `general.file_type = 40` meaning | ❓ open | decode against upstream `enum llama_ftype` when SHA is pinned |
| `output.weight` (lm_head) presence | ❌ absent | weight-tied with `token_embd.weight`; reconfirm at Stage 4 |

## Discrepancy procedure

If `willamette inspect` ever reports an `Unknown(N)` ggml_type for a tensor in
the official BitNet model:

1. Do **not** silently extend the enum to map `N` onto an existing BitNet type.
2. Read the matching `ggml-common.h` / `ggml-bitnet-*.cpp` in the pinned commit
   and confirm which type tag `N` corresponds to.
3. Update `src/gguf/types.rs` and this file together, in a single commit, that
   cites the source line.
