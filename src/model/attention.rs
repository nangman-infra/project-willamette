//! Stage 4-D1 — single-token attention path for BitNet b1.58.
//!
//! Composes the Stage 4-B/4-C primitives into the attention half of one
//! transformer block (RMSNorm → Q/K/V projections → RoPE → scaled dot-product
//! attention → sub_norm → output projection). Stage 4-D1 covers the
//! position-0 single-token case only; KV cache and multi-token attention
//! are deferred to Stage 5-B / 5-C.
//!
//! Pinned forward order from `build_bitnet_158` (`src/llama.cpp:15411..15479`
//! of the pinned commit):
//!
//! ```text
//!   cur     = RMSNorm(inpL, attn_norm)
//!   Qcur    = matmul(Wq, cur)               // BitLinear
//!   Kcur    = matmul(Wk, cur)               // BitLinear
//!   Vcur    = matmul(Wv, cur)               // BitLinear
//!   Qcur    = RoPE(Qcur)
//!   Kcur    = RoPE(Kcur)
//!   cur     = SDPA(Q, K, V) with causal mask, KV cache, GQA
//!   cur     = RMSNorm(cur, attn_sub_norm)
//!   cur     = matmul(Wo, cur)               // BitLinear
//! ```
//!
//! Stage 4-D1 omits the KV-cache write/read and assumes there is exactly
//! one (K, V) pair (the current token at position 0). Softmax over a
//! single key trivially yields `[1.0]`.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::model::bitlinear::bitlinear_i2s_matvec_f32;
use crate::model::config::BitNetConfig;
use crate::model::primitives::{
    apply_rope_f32, attention_scale, kv_head_for_q_head, rms_norm_f32, AttentionShape, RopeType,
};

// ──────────────────────────────────────────────────────────────────────────
// Numerical primitives
// ──────────────────────────────────────────────────────────────────────────

/// In-place numerically-stable softmax (max-subtract, exp, normalise).
///
/// * Empty slice: no-op.
/// * All `-inf` slice (fully masked row): becomes all zeros, never NaN.
/// * Single-element slice: becomes `[1.0]` exactly.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        // Either all -inf (fully masked) or NaN snuck in. Fail safe: zeros.
        for v in x.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut sum: f32 = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
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

// ──────────────────────────────────────────────────────────────────────────
// Multi-head RoPE
// ──────────────────────────────────────────────────────────────────────────

/// Apply RoPE in place to each of `n_heads` `head_dim`-length sub-vectors
/// of `x`. Memory layout matches ggml's `reshape_3d(_, head_dim, n_head,
/// n_tokens)` convention with `n_tokens = 1`: head 0 occupies the first
/// `head_dim` floats, head 1 the next `head_dim`, etc.
pub fn apply_rope_multi_head(
    x: &mut [f32],
    n_heads: u32,
    head_dim: usize,
    n_rot: usize,
    position: u32,
    freq_base: f32,
    rope_type: RopeType,
) -> Result<(), WillametteError> {
    let expected = (n_heads as usize) * head_dim;
    if x.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "apply_rope_multi_head: x.len()={} != n_heads*head_dim={}*{}={}",
            x.len(),
            n_heads,
            head_dim,
            expected
        )));
    }
    for h in 0..n_heads as usize {
        let chunk = &mut x[h * head_dim..(h + 1) * head_dim];
        apply_rope_f32(chunk, head_dim, n_rot, position, freq_base, rope_type)?;
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Single-token scaled-dot-product attention (position 0, no past)
// ──────────────────────────────────────────────────────────────────────────

/// Compute attention output for a single token at position 0 with no past
/// KV. Q, K, V are pre-projected and RoPE'd. Output length equals
/// `n_heads × head_dim`.
///
/// For each Q head:
///   * `score = (Q_h · K_kv(h)) / sqrt(head_dim)`
///   * `softmax([score]) = [1.0]` (single element)
///   * `out_h = 1.0 × V_kv(h)`
///
/// So the per-head output is the V vector for the corresponding KV head,
/// independent of the score value (which is still computed to exercise
/// the softmax path).
pub fn single_token_attention_position_zero(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    output: &mut [f32],
) -> Result<(), WillametteError> {
    let head_dim = shape.head_dim as usize;
    let n_heads = shape.n_heads as usize;
    let n_kv_heads = shape.n_kv_heads as usize;

    if q.len() != n_heads * head_dim {
        return Err(WillametteError::GgufParse(format!(
            "attention: q.len()={} != n_heads*head_dim={}",
            q.len(),
            n_heads * head_dim
        )));
    }
    if k.len() != n_kv_heads * head_dim {
        return Err(WillametteError::GgufParse(format!(
            "attention: k.len()={} != n_kv_heads*head_dim={}",
            k.len(),
            n_kv_heads * head_dim
        )));
    }
    if v.len() != n_kv_heads * head_dim {
        return Err(WillametteError::GgufParse(format!(
            "attention: v.len()={} != n_kv_heads*head_dim={}",
            v.len(),
            n_kv_heads * head_dim
        )));
    }
    if output.len() != q.len() {
        return Err(WillametteError::GgufParse(format!(
            "attention: output.len()={} != q.len()={}",
            output.len(),
            q.len()
        )));
    }

    let scale = attention_scale(head_dim);

    for h in 0..n_heads {
        let kv_h = kv_head_for_q_head(h as u32, shape.group_size) as usize;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let k_h = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
        let v_h = &v[kv_h * head_dim..(kv_h + 1) * head_dim];

        // Compute and softmax even though [single score] always softmaxes
        // to [1.0]. Keeps the math identical to the multi-token path that
        // Stage 5-B will write.
        let mut scores = vec![dot_f32(q_h, k_h) * scale];
        softmax_inplace(&mut scores);
        let w = scores[0];

        let out_h = &mut output[h * head_dim..(h + 1) * head_dim];
        for i in 0..head_dim {
            out_h[i] = w * v_h[i];
        }
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Attention block forward — single token, position 0
// ──────────────────────────────────────────────────────────────────────────

/// Compose RMSNorm → Q/K/V BitLinear → RoPE → SDPA → sub_norm → output
/// BitLinear into one call. This is the attention half of one
/// transformer block for a single token at position 0.
///
/// Residual addition is intentionally NOT applied here — `block.rs`
/// (Stage 4-D3) will sum `x + attention_block_forward(...)` separately.
#[allow(clippy::too_many_arguments)]
pub fn attention_block_forward_position_zero(
    x: &[f32],
    attn_norm_weight: &[f32],
    wq: &TensorView<'_>,
    wk: &TensorView<'_>,
    wv: &TensorView<'_>,
    wo: &TensorView<'_>,
    attn_sub_norm_weight: &[f32],
    config: &BitNetConfig,
    output: &mut [f32],
) -> Result<(), WillametteError> {
    let n_embd = config.embedding_length as usize;
    let kv_dim = config.kv_dim as usize;
    let head_dim = config.head_dim as usize;
    let n_rot = config.rope_dimension_count as usize;
    let freq_base = config.rope_freq_base;
    let eps = config.layer_norm_rms_epsilon;
    let shape =
        AttentionShape::from_config(config.head_count, config.head_count_kv, config.head_dim)?;

    if x.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "attention_block_forward: x.len()={} != n_embd={}",
            x.len(),
            n_embd
        )));
    }
    if attn_norm_weight.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "attention_block_forward: attn_norm_weight.len()={} != n_embd={}",
            attn_norm_weight.len(),
            n_embd
        )));
    }
    if attn_sub_norm_weight.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "attention_block_forward: attn_sub_norm_weight.len()={} != n_embd={}",
            attn_sub_norm_weight.len(),
            n_embd
        )));
    }
    if output.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "attention_block_forward: output.len()={} != n_embd={}",
            output.len(),
            n_embd
        )));
    }

    let mut x_norm = vec![0.0_f32; n_embd];
    rms_norm_f32(x, attn_norm_weight, eps, &mut x_norm)?;

    let mut q = vec![0.0_f32; n_embd];
    let mut k = vec![0.0_f32; kv_dim];
    let mut v = vec![0.0_f32; kv_dim];
    bitlinear_i2s_matvec_f32(wq, &x_norm, &mut q)?;
    bitlinear_i2s_matvec_f32(wk, &x_norm, &mut k)?;
    bitlinear_i2s_matvec_f32(wv, &x_norm, &mut v)?;

    apply_rope_multi_head(
        &mut q,
        config.head_count,
        head_dim,
        n_rot,
        0,
        freq_base,
        RopeType::Neox,
    )?;
    apply_rope_multi_head(
        &mut k,
        config.head_count_kv,
        head_dim,
        n_rot,
        0,
        freq_base,
        RopeType::Neox,
    )?;

    let mut attn_out = vec![0.0_f32; n_embd];
    single_token_attention_position_zero(&q, &k, &v, shape, &mut attn_out)?;

    let mut attn_out_normed = vec![0.0_f32; n_embd];
    rms_norm_f32(&attn_out, attn_sub_norm_weight, eps, &mut attn_out_normed)?;

    bitlinear_i2s_matvec_f32(wo, &attn_out_normed, output)?;
    Ok(())
}

/// `out = a + b` element-wise. Length-checked.
pub fn residual_add(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), WillametteError> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(WillametteError::GgufParse(format!(
            "residual_add: length mismatch a={} b={} out={}",
            a.len(),
            b.len(),
            out.len()
        )));
    }
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_single_element_is_one() {
        let mut x = vec![1.25];
        softmax_inplace(&mut x);
        assert_eq!(x, vec![1.0]);
    }

    #[test]
    fn softmax_uniform_input() {
        let mut x = vec![0.0, 0.0, 0.0, 0.0];
        softmax_inplace(&mut x);
        for v in &x {
            assert!((v - 0.25).abs() < 1e-6);
        }
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_with_neg_inf_masks() {
        let mut x = vec![f32::NEG_INFINITY, 0.0];
        softmax_inplace(&mut x);
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one_for_typical_input() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Should be monotonically increasing — largest input → largest weight
        for i in 1..x.len() {
            assert!(x[i] > x[i - 1]);
        }
    }

    #[test]
    fn softmax_all_neg_inf_gives_zeros() {
        let mut x = vec![f32::NEG_INFINITY; 4];
        softmax_inplace(&mut x);
        for &v in &x {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn rope_multi_head_at_position_zero_is_identity() {
        let mut q = vec![0.5_f32; 20 * 128];
        let orig = q.clone();
        apply_rope_multi_head(&mut q, 20, 128, 128, 0, 500_000.0, RopeType::Neox).unwrap();
        for i in 0..q.len() {
            assert!(
                (q[i] - orig[i]).abs() < 1e-6,
                "position 0 should be identity at index {}",
                i
            );
        }
    }

    #[test]
    fn rope_multi_head_each_head_independent() {
        let mut q = vec![0.0_f32; 2 * 128];
        // Head 0 has values, head 1 is all zero
        for i in 0..128 {
            q[i] = (i as f32) * 0.01;
        }
        let head1_pre = q[128..].to_vec();
        apply_rope_multi_head(&mut q, 2, 128, 128, 7, 500_000.0, RopeType::Neox).unwrap();
        // Head 1 should still be all zeros (rotation of zeros is zeros)
        for (i, &v) in q[128..].iter().enumerate() {
            assert_eq!(v, head1_pre[i], "head 1 dim {} should still be 0", i);
        }
    }

    #[test]
    fn attention_position_zero_returns_v_per_kv_head() {
        // 4 Q heads, 2 KV heads, head_dim = 2.
        let shape = AttentionShape::from_config(4, 2, 2).unwrap();
        assert_eq!(shape.group_size, 2);

        // Arbitrary Q and K; we just need to confirm output equals V for
        // each Q head's mapped KV head.
        let q = vec![1.0; 8]; // 4 heads × 2 = 8
        let k = vec![0.1; 4]; // 2 kv_heads × 2 = 4
        let v = vec![5.0, 6.0, 70.0, 80.0]; // kv_head 0 = [5,6], kv_head 1 = [70,80]
        let mut out = vec![0.0; 8];
        single_token_attention_position_zero(&q, &k, &v, shape, &mut out).unwrap();
        // Q heads 0,1 → kv head 0; Q heads 2,3 → kv head 1.
        assert_eq!(out[0..2], [5.0, 6.0]);
        assert_eq!(out[2..4], [5.0, 6.0]);
        assert_eq!(out[4..6], [70.0, 80.0]);
        assert_eq!(out[6..8], [70.0, 80.0]);
    }

    #[test]
    fn attention_rejects_length_mismatch() {
        let shape = AttentionShape::from_config(4, 2, 2).unwrap();
        let q = vec![0.0; 7]; // wrong (should be 8)
        let k = vec![0.0; 4];
        let v = vec![0.0; 4];
        let mut out = vec![0.0; 8];
        assert!(single_token_attention_position_zero(&q, &k, &v, shape, &mut out).is_err());
    }

    #[test]
    fn residual_add_works() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![10.0_f32, 20.0, 30.0];
        let mut out = vec![0.0_f32; 3];
        residual_add(&a, &b, &mut out).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn residual_add_length_mismatch_errors() {
        let a = vec![1.0; 3];
        let b = vec![2.0; 4];
        let mut out = vec![0.0; 3];
        assert!(residual_add(&a, &b, &mut out).is_err());
    }
}
