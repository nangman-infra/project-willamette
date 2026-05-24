//! Stage 5-B — multi-token reference forward without KV cache.
//!
//! Given a sequence of `M` token ids, run them all the way through the
//! transformer with causal attention and return the final hidden state
//! for **the last token** (the one used to predict the next token).
//!
//! No cache: every generation step recomputes the whole context from
//! scratch. Cost grows as O(M × per_token_cost) per generation step, so
//! this path is intentionally slow — Stage 5-C will add a KV cache that
//! drops the marginal cost of each new token to a single
//! single-token-equivalent pass.
//!
//! The implementation matches the operation order pinned in
//! `docs/BITNET_FORWARD_PLAN.md` §6 and §7; the only Stage 5-B
//! generalisation over Stage 4-D4 is that attention reads from all
//! `0..=t` (K, V) pairs (with NEOX RoPE applied per actual position)
//! instead of the single position-0 pair.

use crate::error::WillametteError;
use crate::model::attention::{apply_rope_multi_head, softmax_inplace};
use crate::model::bitlinear::bitlinear_i2s_matvec_f32;
use crate::model::ffn::{elementwise_mul, relu_square};
use crate::model::graph::ModelGraph;
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

/// Multi-token forward through all 30 layers with causal attention.
/// Returns the post-`output_norm` hidden state for the **last** input
/// token (used for next-token prediction).
///
/// `token_ids` must be non-empty.
pub fn multi_token_forward(
    graph: &ModelGraph<'_>,
    token_ids: &[u32],
) -> Result<Vec<f32>, WillametteError> {
    if token_ids.is_empty() {
        return Err(WillametteError::GgufParse(
            "multi_token_forward: token_ids must not be empty".to_string(),
        ));
    }
    let cfg = &graph.config;
    let n_embd = cfg.embedding_length as usize;
    let n_ff = cfg.feed_forward_length as usize;
    let kv_dim = cfg.kv_dim as usize;
    let head_dim = cfg.head_dim as usize;
    let n_rot = cfg.rope_dimension_count as usize;
    let freq_base = cfg.rope_freq_base;
    let eps = cfg.layer_norm_rms_epsilon;
    let n_heads = cfg.head_count as usize;
    let n_kv_heads = cfg.head_count_kv as usize;
    let shape = AttentionShape::from_config(cfg.head_count, cfg.head_count_kv, cfg.head_dim)?;
    let scale = attention_scale(head_dim);
    let m = token_ids.len();

    if m as u64 > u32::MAX as u64 {
        return Err(WillametteError::GgufParse(format!(
            "multi_token_forward: too many tokens ({})",
            m
        )));
    }

    // Initial: gather embeddings for every token.
    let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(m);
    for &tid in token_ids {
        let mut e = vec![0.0_f32; n_embd];
        embedding_gather_f16(graph.token_embd, tid, &mut e)?;
        hidden.push(e);
    }

    // Per-layer transformer.
    for layer in &graph.layers {
        let attn_norm_w = f32_tensor_to_vec(layer.attn_norm)?;
        let attn_sub_norm_w = f32_tensor_to_vec(layer.attn_sub_norm)?;
        let ffn_norm_w = f32_tensor_to_vec(layer.ffn_norm)?;
        let ffn_sub_norm_w = f32_tensor_to_vec(layer.ffn_sub_norm)?;

        // ── attention half ──
        // x_norm per token.
        let mut x_norms: Vec<Vec<f32>> = Vec::with_capacity(m);
        for t in 0..m {
            let mut xn = vec![0.0_f32; n_embd];
            rms_norm_f32(&hidden[t], &attn_norm_w, eps, &mut xn)?;
            x_norms.push(xn);
        }

        // Q, K, V per token with NEOX RoPE at actual position t.
        let mut qs: Vec<Vec<f32>> = Vec::with_capacity(m);
        let mut ks: Vec<Vec<f32>> = Vec::with_capacity(m);
        let mut vs: Vec<Vec<f32>> = Vec::with_capacity(m);
        for t in 0..m {
            let mut q = vec![0.0_f32; n_embd];
            let mut k = vec![0.0_f32; kv_dim];
            let mut v = vec![0.0_f32; kv_dim];
            bitlinear_i2s_matvec_f32(layer.attn_q, &x_norms[t], &mut q)?;
            bitlinear_i2s_matvec_f32(layer.attn_k, &x_norms[t], &mut k)?;
            bitlinear_i2s_matvec_f32(layer.attn_v, &x_norms[t], &mut v)?;
            apply_rope_multi_head(
                &mut q,
                cfg.head_count,
                head_dim,
                n_rot,
                t as u32,
                freq_base,
                RopeType::Neox,
            )?;
            apply_rope_multi_head(
                &mut k,
                cfg.head_count_kv,
                head_dim,
                n_rot,
                t as u32,
                freq_base,
                RopeType::Neox,
            )?;
            qs.push(q);
            ks.push(k);
            vs.push(v);
        }

        // Causal scaled dot-product attention.
        let mut attn_outs: Vec<Vec<f32>> = Vec::with_capacity(m);
        for t in 0..m {
            let mut attn_out = vec![0.0_f32; n_embd];
            for h in 0..n_heads {
                let kv_h = kv_head_for_q_head(h as u32, shape.group_size) as usize;
                debug_assert!(kv_h < n_kv_heads);
                let q_h = &qs[t][h * head_dim..(h + 1) * head_dim];

                let mut scores = Vec::with_capacity(t + 1);
                for k_idx in 0..=t {
                    let k_h = &ks[k_idx][kv_h * head_dim..(kv_h + 1) * head_dim];
                    scores.push(dot_f32(q_h, k_h) * scale);
                }
                softmax_inplace(&mut scores);

                let out_h = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                for k_idx in 0..=t {
                    let v_h = &vs[k_idx][kv_h * head_dim..(kv_h + 1) * head_dim];
                    let w = scores[k_idx];
                    for d in 0..head_dim {
                        out_h[d] += w * v_h[d];
                    }
                }
            }
            attn_outs.push(attn_out);
        }

        // attn_sub_norm + Wo + residual #1 (in place on hidden[t]).
        for t in 0..m {
            let mut sub_normed = vec![0.0_f32; n_embd];
            rms_norm_f32(&attn_outs[t], &attn_sub_norm_w, eps, &mut sub_normed)?;
            let mut wo_out = vec![0.0_f32; n_embd];
            bitlinear_i2s_matvec_f32(layer.attn_output, &sub_normed, &mut wo_out)?;
            for d in 0..n_embd {
                hidden[t][d] += wo_out[d];
            }
            for v in &hidden[t] {
                if !v.is_finite() {
                    return Err(WillametteError::GgufParse(format!(
                        "multi_token_forward: non-finite hidden after attn (layer {}, token {})",
                        layer.index, t
                    )));
                }
            }
        }

        // ── FFN half (parallel-gated ReLU²) + residual #2 ──
        for t in 0..m {
            // x_norm = RMSNorm(hidden[t], ffn_norm)
            let mut x_norm = vec![0.0_f32; n_embd];
            rms_norm_f32(&hidden[t], &ffn_norm_w, eps, &mut x_norm)?;

            let mut gate = vec![0.0_f32; n_ff];
            let mut up = vec![0.0_f32; n_ff];
            bitlinear_i2s_matvec_f32(layer.ffn_gate, &x_norm, &mut gate)?;
            bitlinear_i2s_matvec_f32(layer.ffn_up, &x_norm, &mut up)?;
            relu_square(&mut gate);
            let mut fused = vec![0.0_f32; n_ff];
            elementwise_mul(&gate, &up, &mut fused)?;
            let mut fused_norm = vec![0.0_f32; n_ff];
            rms_norm_f32(&fused, &ffn_sub_norm_w, eps, &mut fused_norm)?;
            let mut down = vec![0.0_f32; n_embd];
            bitlinear_i2s_matvec_f32(layer.ffn_down, &fused_norm, &mut down)?;
            for d in 0..n_embd {
                hidden[t][d] += down[d];
            }
            for v in &hidden[t] {
                if !v.is_finite() {
                    return Err(WillametteError::GgufParse(format!(
                        "multi_token_forward: non-finite hidden after ffn (layer {}, token {})",
                        layer.index, t
                    )));
                }
            }
        }
    }

    // output_norm on the LAST token only — that's the only hidden we
    // need for next-token prediction.
    let on_w = f32_tensor_to_vec(graph.output_norm)?;
    let mut final_hidden = vec![0.0_f32; n_embd];
    rms_norm_f32(&hidden[m - 1], &on_w, eps, &mut final_hidden)?;
    Ok(final_hidden)
}
