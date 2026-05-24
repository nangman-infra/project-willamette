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
use crate::model::graph::ModelGraph;
use crate::model::kv_cache::KVCache;
use crate::model::primitives::{
    attention_scale, embedding_gather_f16, f32_tensor_to_vec, kv_head_for_q_head, rms_norm_f32,
    AttentionShape, RopeType,
};

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
    let cfg = &graph.config;
    let n_embd = cfg.embedding_length as usize;
    let kv_dim = cfg.kv_dim as usize;
    let n_ff = cfg.feed_forward_length as usize;
    let head_dim = cfg.head_dim as usize;
    let n_rot = cfg.rope_dimension_count as usize;
    let freq_base = cfg.rope_freq_base;
    let eps = cfg.layer_norm_rms_epsilon;
    let n_heads = cfg.head_count as usize;
    let shape = AttentionShape::from_config(cfg.head_count, cfg.head_count_kv, cfg.head_dim)?;
    let scale = attention_scale(head_dim);

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
    if token_id >= cfg.vocab_size {
        return Err(WillametteError::GgufParse(format!(
            "forward_with_cache: token_id {} out of vocab range {}",
            token_id, cfg.vocab_size
        )));
    }

    let mut hidden = vec![0.0_f32; n_embd];
    embedding_gather_f16(graph.token_embd, token_id, &mut hidden)?;

    for layer in &graph.layers {
        let attn_norm_w = f32_tensor_to_vec(layer.attn_norm)?;
        let attn_sub_norm_w = f32_tensor_to_vec(layer.attn_sub_norm)?;
        let ffn_norm_w = f32_tensor_to_vec(layer.ffn_norm)?;
        let ffn_sub_norm_w = f32_tensor_to_vec(layer.ffn_sub_norm)?;

        let mut x_norm = vec![0.0_f32; n_embd];
        rms_norm_f32(&hidden, &attn_norm_w, eps, &mut x_norm)?;

        let mut q = vec![0.0_f32; n_embd];
        let mut k = vec![0.0_f32; kv_dim];
        let mut v = vec![0.0_f32; kv_dim];
        bitlinear_i2s_matvec_f32(layer.attn_q, &x_norm, &mut q)?;
        bitlinear_i2s_matvec_f32(layer.attn_k, &x_norm, &mut k)?;
        bitlinear_i2s_matvec_f32(layer.attn_v, &x_norm, &mut v)?;

        apply_rope_multi_head(
            &mut q,
            cfg.head_count,
            head_dim,
            n_rot,
            position,
            freq_base,
            RopeType::Neox,
        )?;
        apply_rope_multi_head(
            &mut k,
            cfg.head_count_kv,
            head_dim,
            n_rot,
            position,
            freq_base,
            RopeType::Neox,
        )?;

        // Append THIS token's K/V to the cache before attending.
        let layer_idx = layer.index as usize;
        cache.append(layer_idx, &k, &v)?;
        let (cached_k, cached_v) = cache.read(layer_idx)?;
        let n_past = cached_k.len() / kv_dim; // = position + 1

        // Scaled dot-product attention against the full cache window.
        let mut attn_out = vec![0.0_f32; n_embd];
        for h in 0..n_heads {
            let kv_h = kv_head_for_q_head(h as u32, shape.group_size) as usize;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::with_capacity(n_past);
            for p in 0..n_past {
                let base = p * kv_dim + kv_h * head_dim;
                let k_h = &cached_k[base..base + head_dim];
                scores.push(dot_f32(q_h, k_h) * scale);
            }
            softmax_inplace(&mut scores);

            let out_h = &mut attn_out[h * head_dim..(h + 1) * head_dim];
            for p in 0..n_past {
                let base = p * kv_dim + kv_h * head_dim;
                let v_h = &cached_v[base..base + head_dim];
                let w = scores[p];
                for d in 0..head_dim {
                    out_h[d] += w * v_h[d];
                }
            }
        }

        let mut sub_normed = vec![0.0_f32; n_embd];
        rms_norm_f32(&attn_out, &attn_sub_norm_w, eps, &mut sub_normed)?;
        let mut wo_out = vec![0.0_f32; n_embd];
        bitlinear_i2s_matvec_f32(layer.attn_output, &sub_normed, &mut wo_out)?;
        for d in 0..n_embd {
            hidden[d] += wo_out[d];
        }

        // FFN half.
        let mut x_norm_ffn = vec![0.0_f32; n_embd];
        rms_norm_f32(&hidden, &ffn_norm_w, eps, &mut x_norm_ffn)?;
        let mut gate = vec![0.0_f32; n_ff];
        let mut up = vec![0.0_f32; n_ff];
        bitlinear_i2s_matvec_f32(layer.ffn_gate, &x_norm_ffn, &mut gate)?;
        bitlinear_i2s_matvec_f32(layer.ffn_up, &x_norm_ffn, &mut up)?;
        relu_square(&mut gate);
        let mut fused = vec![0.0_f32; n_ff];
        elementwise_mul(&gate, &up, &mut fused)?;
        let mut fused_norm = vec![0.0_f32; n_ff];
        rms_norm_f32(&fused, &ffn_sub_norm_w, eps, &mut fused_norm)?;
        let mut down = vec![0.0_f32; n_embd];
        bitlinear_i2s_matvec_f32(layer.ffn_down, &fused_norm, &mut down)?;
        for d in 0..n_embd {
            hidden[d] += down[d];
        }
        for v in &hidden {
            if !v.is_finite() {
                return Err(WillametteError::GgufParse(format!(
                    "forward_with_cache: non-finite hidden after layer {}",
                    layer.index
                )));
            }
        }
    }

    let on_w = f32_tensor_to_vec(graph.output_norm)?;
    let mut final_hidden = vec![0.0_f32; n_embd];
    rms_norm_f32(&hidden, &on_w, eps, &mut final_hidden)?;
    Ok(final_hidden)
}
