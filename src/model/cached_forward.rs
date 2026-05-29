#![allow(clippy::needless_range_loop)]
// Token-position + head-index indexing reads more naturally than
// iterator chains over multiple parallel arrays.

//! Stage 5-C — single-token forward with a `KVCache`.
//!
//! Designed to be called once per token, in position order
//! (`0, 1, 2, …`). For each layer it:
//!
//!   1. RMSNorms the hidden state with `attn_norm`.
//!   2. Computes Q/K/V via BitLinear matvecs.
//!   3. Applies NEOX RoPE at the supplied position.
//!   4. Appends the new (K, V) to the cache for this layer.
//!   5. Runs scaled-dot-product attention from this Q against the
//!      cached `[K, V]` window (positions `0..=position`).
//!   6. Applies `attn_sub_norm`, the output BitLinear, and the first
//!      residual.
//!   7. Runs the FFN half (Stage 4-D2) and the second residual.
//!
//! After all layers, applies `output_norm` and returns the final
//! hidden — the same shape Stage 4-D5 hands to `compute_logits`.
//!
//! Numerical equivalence with the no-cache path (`multi_token_forward`)
//! is verified by `tests/kv_cache_forward.rs`.

use crate::error::WillametteError;
use crate::model::attention::{apply_rope_multi_head, softmax_inplace};
use crate::model::bitlinear::bitlinear_i2s_matvec_f32;
use crate::model::ffn::{elementwise_mul, relu_square};
use crate::model::graph::{LayerWeights, ModelGraph};
use crate::model::kv_cache::KVCache;
use crate::model::primitives::{
    attention_scale, embedding_gather_f16, kv_head_for_q_head, rms_norm_f32, AttentionShape,
    RopeType,
};

/// Per-token constants pulled from `ModelConfig` — packaged so the
/// inner per-layer helper doesn't take a dozen scalar arguments.
struct LayerCtx {
    n_embd: usize,
    kv_dim: usize,
    n_ff: usize,
    head_dim: usize,
    n_rot: usize,
    freq_base: f32,
    eps: f32,
    n_heads: usize,
    n_heads_kv: u32,
    shape: AttentionShape,
    scale: f32,
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Single-token forward at the given position, reading/writing `cache`.
///
/// Returns the post-`output_norm` hidden state (length `n_embd`). The
/// `cache.position()` must equal `position` on entry — i.e. tokens must
/// be processed strictly in order.
pub fn forward_with_cache(
    graph: &ModelGraph<'_>,
    cache: &mut KVCache,
    token_id: u32,
    position: u32,
) -> Result<Vec<f32>, WillametteError> {
    forward_with_cache_progress(graph, cache, token_id, position, |_| {})
}

/// Same as [`forward_with_cache`] but calls `on_layer(layer_idx)`
/// after each transformer block finishes. Used by the TUI to update
/// the layer-progress indicator in the dashboard. The overhead is
/// one closure call per layer (≤ 30 calls/token for BitNet b1.58 2B)
/// — well below the matvec cost.
pub fn forward_with_cache_progress<F: FnMut(u32)>(
    graph: &ModelGraph<'_>,
    cache: &mut KVCache,
    token_id: u32,
    position: u32,
    mut on_layer: F,
) -> Result<Vec<f32>, WillametteError> {
    let cfg = &graph.config;
    let ctx = LayerCtx {
        n_embd: cfg.embedding_length as usize,
        kv_dim: cfg.kv_dim as usize,
        n_ff: cfg.feed_forward_length as usize,
        head_dim: cfg.head_dim as usize,
        n_rot: cfg.rope_dimension_count as usize,
        freq_base: cfg.rope_freq_base,
        eps: cfg.layer_norm_rms_epsilon,
        n_heads: cfg.head_count as usize,
        n_heads_kv: cfg.head_count_kv,
        shape: AttentionShape::from_config(cfg.head_count, cfg.head_count_kv, cfg.head_dim)?,
        scale: attention_scale(cfg.head_dim as usize),
    };

    validate_cache_inputs(graph, cache, token_id, position, ctx.kv_dim)?;

    let mut hidden = vec![0.0_f32; ctx.n_embd];
    embedding_gather_f16(graph.token_embd, token_id, &mut hidden)?;

    // Dequant scratch reused across layers — capacity stabilises at
    // (position + 1) × kv_dim after the first growth.
    let mut scratch_k: Vec<f32> = Vec::new();
    let mut scratch_v: Vec<f32> = Vec::new();

    for layer in &graph.layers {
        forward_one_layer(
            layer,
            cache,
            &mut hidden,
            &mut scratch_k,
            &mut scratch_v,
            &ctx,
            position,
        )?;
        on_layer(layer.index);
    }

    let mut final_hidden = vec![0.0_f32; ctx.n_embd];
    rms_norm_f32(&hidden, &graph.output_norm_f32, ctx.eps, &mut final_hidden)?;
    Ok(final_hidden)
}

fn validate_cache_inputs(
    graph: &ModelGraph<'_>,
    cache: &KVCache,
    token_id: u32,
    position: u32,
    kv_dim: usize,
) -> Result<(), WillametteError> {
    if cache.kv_dim != kv_dim {
        return Err(WillametteError::GgufParse(format!(
            "forward_with_cache: cache.kv_dim={} != model kv_dim={}",
            cache.kv_dim, kv_dim
        )));
    }
    if cache.n_layers != graph.layers.len() {
        return Err(WillametteError::GgufParse(format!(
            "forward_with_cache: cache.n_layers={} != model layers={}",
            cache.n_layers,
            graph.layers.len()
        )));
    }
    if cache.position() as u32 != position {
        return Err(WillametteError::GgufParse(format!(
            "forward_with_cache: cache.position()={} != position={}; tokens must be processed in order",
            cache.position(),
            position
        )));
    }
    if token_id >= graph.config.vocab_size {
        return Err(WillametteError::GgufParse(format!(
            "forward_with_cache: token_id {} out of vocab range {}",
            token_id, graph.config.vocab_size
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_one_layer(
    layer: &LayerWeights<'_>,
    cache: &mut KVCache,
    hidden: &mut [f32],
    scratch_k: &mut Vec<f32>,
    scratch_v: &mut Vec<f32>,
    ctx: &LayerCtx,
    position: u32,
) -> Result<(), WillametteError> {
    let mut x_norm = vec![0.0_f32; ctx.n_embd];
    rms_norm_f32(hidden, &layer.attn_norm_f32, ctx.eps, &mut x_norm)?;

    let mut q = vec![0.0_f32; ctx.n_embd];
    let mut k = vec![0.0_f32; ctx.kv_dim];
    let mut v = vec![0.0_f32; ctx.kv_dim];
    bitlinear_i2s_matvec_f32(layer.attn_q, &x_norm, &mut q)?;
    bitlinear_i2s_matvec_f32(layer.attn_k, &x_norm, &mut k)?;
    bitlinear_i2s_matvec_f32(layer.attn_v, &x_norm, &mut v)?;

    apply_rope_multi_head(
        &mut q,
        ctx.n_heads as u32,
        ctx.head_dim,
        ctx.n_rot,
        position,
        ctx.freq_base,
        RopeType::Neox,
    )?;
    apply_rope_multi_head(
        &mut k,
        ctx.n_heads_kv,
        ctx.head_dim,
        ctx.n_rot,
        position,
        ctx.freq_base,
        RopeType::Neox,
    )?;

    let layer_idx = layer.index as usize;
    cache.append(layer_idx, &k, &v)?;
    cache.read_into(layer_idx, scratch_k, scratch_v)?;
    let n_past = scratch_k.len() / ctx.kv_dim;

    let attn_out = scaled_dot_product_attention(&q, scratch_k, scratch_v, n_past, ctx);

    let mut sub_normed = vec![0.0_f32; ctx.n_embd];
    rms_norm_f32(
        &attn_out,
        &layer.attn_sub_norm_f32,
        ctx.eps,
        &mut sub_normed,
    )?;
    let mut wo_out = vec![0.0_f32; ctx.n_embd];
    bitlinear_i2s_matvec_f32(layer.attn_output, &sub_normed, &mut wo_out)?;
    for d in 0..ctx.n_embd {
        hidden[d] += wo_out[d];
    }

    apply_ffn_block(layer, hidden, ctx)?;
    check_finite_hidden(hidden, layer.index)?;
    Ok(())
}

fn scaled_dot_product_attention(
    q: &[f32],
    cached_k: &[f32],
    cached_v: &[f32],
    n_past: usize,
    ctx: &LayerCtx,
) -> Vec<f32> {
    let mut attn_out = vec![0.0_f32; ctx.n_embd];
    for h in 0..ctx.n_heads {
        let kv_h = kv_head_for_q_head(h as u32, ctx.shape.group_size) as usize;
        let q_h = &q[h * ctx.head_dim..(h + 1) * ctx.head_dim];
        let mut scores = Vec::with_capacity(n_past);
        for p in 0..n_past {
            let base = p * ctx.kv_dim + kv_h * ctx.head_dim;
            let k_h = &cached_k[base..base + ctx.head_dim];
            scores.push(dot_f32(q_h, k_h) * ctx.scale);
        }
        softmax_inplace(&mut scores);

        let out_h = &mut attn_out[h * ctx.head_dim..(h + 1) * ctx.head_dim];
        for p in 0..n_past {
            let base = p * ctx.kv_dim + kv_h * ctx.head_dim;
            let v_h = &cached_v[base..base + ctx.head_dim];
            let w = scores[p];
            for d in 0..ctx.head_dim {
                out_h[d] += w * v_h[d];
            }
        }
    }
    attn_out
}

fn apply_ffn_block(
    layer: &LayerWeights<'_>,
    hidden: &mut [f32],
    ctx: &LayerCtx,
) -> Result<(), WillametteError> {
    let mut x_norm_ffn = vec![0.0_f32; ctx.n_embd];
    rms_norm_f32(hidden, &layer.ffn_norm_f32, ctx.eps, &mut x_norm_ffn)?;
    let mut gate = vec![0.0_f32; ctx.n_ff];
    let mut up = vec![0.0_f32; ctx.n_ff];
    bitlinear_i2s_matvec_f32(layer.ffn_gate, &x_norm_ffn, &mut gate)?;
    bitlinear_i2s_matvec_f32(layer.ffn_up, &x_norm_ffn, &mut up)?;
    relu_square(&mut gate);
    let mut fused = vec![0.0_f32; ctx.n_ff];
    elementwise_mul(&gate, &up, &mut fused)?;
    let mut fused_norm = vec![0.0_f32; ctx.n_ff];
    rms_norm_f32(&fused, &layer.ffn_sub_norm_f32, ctx.eps, &mut fused_norm)?;
    let mut down = vec![0.0_f32; ctx.n_embd];
    bitlinear_i2s_matvec_f32(layer.ffn_down, &fused_norm, &mut down)?;
    for d in 0..ctx.n_embd {
        hidden[d] += down[d];
    }
    Ok(())
}

fn check_finite_hidden(hidden: &[f32], layer_idx: u32) -> Result<(), WillametteError> {
    for v in hidden {
        if !v.is_finite() {
            return Err(WillametteError::GgufParse(format!(
                "forward_with_cache: non-finite hidden after layer {}",
                layer_idx
            )));
        }
    }
    Ok(())
}
