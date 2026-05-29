# KV Cache Quantisation

*Landed in v0.9.0-mvp. Previous releases stored K and V in f32.*

## Why this exists

The KV cache is the largest piece of *dynamic* memory the runtime
allocates. Everything else — model weights, embeddings, scratch
buffers — is bounded by the model file itself. The KV cache, by
contrast, grows linearly with the number of tokens already processed
in this conversation. On humble hardware that growth is the dominant
limit on how long a chat can run.

On antix1 (Pentium-M, 2 GB RAM) the working budget looks like this:

| Item | Size |
| --- | --- |
| Model mmap (`ggml-model-i2_s.gguf`) | 1.13 GB (zero-copy, shared with OS page cache) |
| OS + everything else | ≈ 0.3-0.4 GB |
| Available for KV cache | **≈ 0.5 GB** |

Pre-v0.9 the f32 K and V tensors cost 150 KB per token (see math
below), so the available budget capped chat history at roughly
**3.3 K tokens** before allocation pressure started causing trouble.
After v0.9 the per-token cost is 37.7 KB, raising the same ceiling
to **≈ 13 K tokens** — well past the model's 4096-token positional
embedding limit, i.e. the runtime is no longer the bottleneck on
this host.

## What changed

`KVCache` (`src/model/kv_cache.rs`) used to hold

```rust
struct LayerKV { k: Vec<f32>, v: Vec<f32> }
```

with one `f32` per element. v0.9 replaces that with

```rust
struct LayerKV {
    k_quant:  Vec<i8>,    // 1 byte / element
    k_scales: Vec<f32>,   // 1 f32 / token
    v_quant:  Vec<i8>,
    v_scales: Vec<f32>,
}
```

Each token's `kv_dim`-long K vector is **per-token absmax** quantised
to i8:

```text
scale = absmax(x) / 127
q[i]  = round(x[i] / scale).clamp(-127, 127) as i8
```

V uses the same scheme with its own absmax. Zero vectors round-trip
exactly (the `absmax == 0` branch in `append_quantised` writes
all-zero quant and a zero scale). Dequantisation is the obvious
`f32 = (q as f32) * scale`.

This is the same scheme the BitLinear i8 activation kernel
(`bitlinear_sse2_i8`, `bitlinear_neon_i8`) already applies once per
matvec. The decode data path is now end-to-end i8 (activations,
BitLinear product, KV) with f32 reappearing only at norm layers and
softmax — a useful consistency property when reasoning about where
precision can leak in.

## API change

The old `KVCache::read(layer_idx) -> (&[f32], &[f32])` is gone — the
storage is i8 now, so there is no contiguous f32 slice to borrow.
The replacement is:

```rust
pub fn read_into(
    &self,
    layer_idx: usize,
    out_k: &mut Vec<f32>,
    out_v: &mut Vec<f32>,
) -> Result<(), WillametteError>;
```

The caller provides reusable dequant buffers. The only production
call site (`cached_forward::forward_one_layer`) allocates one pair
per `forward_with_cache_progress` invocation and reuses them across
all 30 transformer blocks; the buffer capacity stabilises at
`(position + 1) × kv_dim` after the first growth.

`KVCache::resident_bytes()` was added so `ChatEngine::estimate_kv_cache_bytes`
can read the actual cache footprint instead of computing a formula
that assumed the old f32 layout.

## Memory math

For BitNet b1.58 2B: `block_count = 30`, `kv_dim = 640` (5 KV heads ×
128 head_dim).

| Per token, all layers | f32 KV | i8 KV |
| --- | ---: | ---: |
| K bytes  | `30 × 640 × 4 = 76,800` | `30 × 640 × 1 = 19,200` |
| V bytes  | `30 × 640 × 4 = 76,800` | `30 × 640 × 1 = 19,200` |
| K scales | — | `30 × 4 = 120` |
| V scales | — | `30 × 4 = 120` |
| **Total** | **153,600 (= 150 KB)** | **38,640 (= 37.7 KB)** |
| Ratio    | 1.00× | **0.251× (≈ 3.97× smaller)** |

At full 4096-token context: ~614 MB f32 → ~154 MB i8. **460 MB
saved** — about 92 % of antix1's available KV budget if it had
ever been allocated.

The `resident_bytes_matches_layout` unit test
(`src/model/kv_cache.rs::tests`) asserts that the actual code path
matches this formula exactly across a configurable
`(n_layers, kv_dim, n_tokens)` shape.

## Fidelity

Per-token absmax i8 quantisation introduces a per-element
worst-case error of `scale / 2 = absmax / 254`. That drift is no
longer bit-equal to the no-cache reference, but the *property
users see* — the greedy token-id sequence — stays unchanged.

| Test | Reference | Behaviour at v0.9 |
| --- | --- | --- |
| `cache_single_token_matches_no_cache_single_token` (`tests/kv_cache.rs`) | Bit-equal hidden state vs `multi_token_forward` | **Cosine ≥ 0.999** on the post-`output_norm` hidden. Bit-equal dropped. |
| `cache_two_token_sequence_matches_no_cache` | Bit-equal | **Cosine ≥ 0.999** |
| `greedy_with_cache_matches_greedy_no_cache_for_2_steps` | Token-id sequence exact-equal | **Unchanged — still exact-equal**. argmax collapses the per-element drift before the user can see it. |
| Stage 5-E reference "The capital of France is" (Mac NEON + antix1 SSE2 i8) | `[12366, 13, 12366]` = `" Paris. Paris"` (README L184) | **Identical** at v0.9 — i8 KV did not flip any argmax across the reference set. Verified on Apple M4 (NEON) and antix1 (i686 SSE2 i8). |

So in plain terms: at decoding time the model writes exactly the
same bytes it wrote in v0.8 on the verified prompts, but it costs
~4 × less memory to keep the cache around.

## Out of scope (intentional)

* **i8-direct attention dot product.** The current implementation
  dequantises the whole cache window once per layer into a caller-
  managed f32 scratch. The attention dot products then run in f32.
  An i8 K-and-Q dot (similar to the i8 BitLinear path) would save
  the dequantisation cost; whether that is worth the code added is
  a measurement question on a host where decode time is dominated
  by KV scan, not BitLinear matvec — which is not where antix1 sits
  today (96.35 % BitLinear per `perf` on the v0.5 era). Revisit if
  that ratio shifts.
* **More aggressive schemes (Q4_0, Q4_K-style group quant).** Would
  cut another ~2 × off the cache. The trade-off is more code (group
  size, two-pass scale/zero-point) and a wider fidelity gap that
  needs perplexity-style measurement to characterise. v0.9 establishes
  the cleanest possible baseline (per-token absmax i8) so a future
  switch has a documented reference.
* **Per-head or per-group K/V scales.** Would slightly improve
  fidelity around RoPE-rotated K dimensions whose absmax varies by
  head. Same trade-off as above: only worth doing if a measured
  prompt actually breaks the current per-token scheme.

## Reverting

There is no `--cfg willamette_kv_f32` flag. If a future bug shows
that i8 KV breaks for a specific prompt, the right response is to
add the appropriate test, document the failure mode, and either
narrow the scheme (per-head) or expand it (Q8_0-style). Quietly
re-introducing the f32 path would mask the failure rather than
characterise it.
