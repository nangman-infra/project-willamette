//! Stage 4-B — pure float / shape-safe primitives for BitNet b1.58 forward.
//!
//! Every function here is:
//!   * deterministic, no global state,
//!   * f32 in / f32 out (no quantised dtypes touched yet),
//!   * shape-checked at the boundary and otherwise allocation-free where
//!     the caller provides an output slice,
//!   * pinned against `docs/BITNET_FORWARD_PLAN.md` for source citations.
//!
//! Operations explicitly NOT in this module (Stage 4-C / 4-D):
//!   * I2_S matmul / unpack / dequant
//!   * attention matmul + softmax + value aggregation
//!   * KV cache datastructure
//!   * sampling, logits, generation

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;

// ──────────────────────────────────────────────────────────────────────────
// F32 tensor reader (for norm weights consumed by Stage 4-D)
// ──────────────────────────────────────────────────────────────────────────

/// Decode an F32 tensor's little-endian bytes into a fresh `Vec<f32>` of
/// length `n_elements`. Errors if the tensor is not F32 or if its
/// `data.len()` does not match `n_elements × 4`.
pub fn f32_tensor_to_vec(t: &TensorView<'_>) -> Result<Vec<f32>, WillametteError> {
    if t.ggml_type != GgmlType::F32 {
        return Err(WillametteError::GgufParse(format!(
            "f32_tensor_to_vec: tensor {:?} is {} (raw {}), not F32",
            t.name,
            t.ggml_type.name(),
            t.ggml_type.to_raw()
        )));
    }
    let n = t.n_elements() as usize;
    if t.data.len() != n * 4 {
        return Err(WillametteError::GgufParse(format!(
            "f32_tensor_to_vec: data.len()={} != n_elements*4={}",
            t.data.len(),
            n * 4
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &t.data[4 * i..4 * i + 4];
        out.push(f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
    }
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────
// f16 → f32 conversion
// ──────────────────────────────────────────────────────────────────────────

/// IEEE 754 binary16 → binary32 conversion.
///
/// Hand-rolled (no `half` crate dep) so the path is auditable. Handles
/// subnormals, ±0, ±inf, and NaN per IEEE 754-2008.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;

    if exp == 0 {
        if mantissa == 0 {
            // ±0
            return f32::from_bits((sign as u32) << 31);
        }
        // Subnormal: value = (-1)^sign × mantissa × 2^(-24)
        let m = mantissa as f32;
        let val = m * (1.0_f32 / (1u32 << 24) as f32);
        return if sign == 1 { -val } else { val };
    }

    if exp == 0x1F {
        // ±inf or NaN
        let f32_bits = ((sign as u32) << 31) | (0xFFu32 << 23) | ((mantissa as u32) << 13);
        return f32::from_bits(f32_bits);
    }

    // Normal: rebias exponent (15 → 127) and shift mantissa (10 → 23 bits)
    let exp_f32 = (exp as u32 + (127 - 15)) << 23;
    let mantissa_f32 = (mantissa as u32) << 13;
    let sign_f32 = (sign as u32) << 31;
    f32::from_bits(sign_f32 | exp_f32 | mantissa_f32)
}

// ──────────────────────────────────────────────────────────────────────────
// Embedding row gather
// ──────────────────────────────────────────────────────────────────────────

/// Gather one row of an F16 embedding table into an f32 buffer.
///
/// `token_embd` must be a 2-D F16 tensor with shape
/// `[embedding_length, vocab_size]` (GGUF innermost-first; `n_embd` is the
/// fast axis, `n_vocab` is the slow axis — every vocab entry's hidden
/// vector is contiguous in memory).
///
/// `out.len()` must equal `embedding_length`. `token_id` must be `<
/// vocab_size`.
pub fn embedding_gather_f16(
    token_embd: &TensorView<'_>,
    token_id: u32,
    out: &mut [f32],
) -> Result<(), WillametteError> {
    if token_embd.ggml_type != GgmlType::F16 {
        return Err(WillametteError::GgufParse(format!(
            "embedding_gather_f16: tensor {:?} is {} (raw {}), expected F16",
            token_embd.name,
            token_embd.ggml_type.name(),
            token_embd.ggml_type.to_raw()
        )));
    }
    if token_embd.shape.len() != 2 {
        return Err(WillametteError::GgufParse(format!(
            "embedding_gather_f16: tensor {:?} is not 2-D (shape={:?})",
            token_embd.name, token_embd.shape
        )));
    }
    let n_embd = token_embd.shape[0] as usize;
    let n_vocab = token_embd.shape[1] as usize;

    if out.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "embedding_gather_f16: out.len()={} != n_embd={}",
            out.len(),
            n_embd
        )));
    }
    if token_id as usize >= n_vocab {
        return Err(WillametteError::GgufParse(format!(
            "embedding_gather_f16: token_id {} out of range (vocab_size={})",
            token_id, n_vocab
        )));
    }

    let row_bytes = n_embd * 2;
    let byte_offset = (token_id as usize) * row_bytes;
    let end = byte_offset + row_bytes;
    if end > token_embd.data.len() {
        return Err(WillametteError::GgufParse(format!(
            "embedding_gather_f16: row bytes [{}..{}) exceed tensor data len {}",
            byte_offset,
            end,
            token_embd.data.len()
        )));
    }
    let row = &token_embd.data[byte_offset..end];

    for i in 0..n_embd {
        let lo = row[2 * i] as u16;
        let hi = row[2 * i + 1] as u16;
        let bits = lo | (hi << 8);
        out[i] = f16_to_f32(bits);
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// RMSNorm
// ──────────────────────────────────────────────────────────────────────────

/// RMSNorm: `out[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]`
///
/// `eps` must come from the model's
/// `bitnet-b1.58.attention.layer_norm_rms_epsilon` (loaded into
/// `BitNetConfig::layer_norm_rms_epsilon`). The pinned source uses this
/// single epsilon for every RMSNorm in `build_bitnet_158`:
/// `attn_norm`, `attn_sub_norm`, `ffn_norm`, `ffn_sub_norm`, `output_norm`
/// — all four per-layer norms and the final `output_norm`.
/// See `src/llama.cpp:6118` (`get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,
/// hparams.f_norm_rms_eps)`) and `src/llama.cpp:15414/15495/15510` (the
/// `llm_build_norm(..., LLM_NORM_RMS, ...)` callsites).
pub fn rms_norm_f32(
    x: &[f32],
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
) -> Result<(), WillametteError> {
    if x.len() != weight.len() || x.len() != out.len() {
        return Err(WillametteError::GgufParse(format!(
            "rms_norm_f32: length mismatch x={} weight={} out={}",
            x.len(),
            weight.len(),
            out.len()
        )));
    }
    if x.is_empty() {
        return Ok(());
    }
    let n = x.len();
    let mut sum_sq: f32 = 0.0;
    for &xi in x.iter() {
        sum_sq += xi * xi;
    }
    let mean = sum_sq / (n as f32);
    let rsqrt = 1.0 / (mean + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * rsqrt * weight[i];
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// RoPE
// ──────────────────────────────────────────────────────────────────────────

/// RoPE encoding flavour. Pinned per architecture in
/// `src/llama.cpp:20072..20139` (`llama_rope_type()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// "Normal" RoPE: rotate pairs of consecutive head values
    /// `(x[2j], x[2j+1])`. Used by LLM_ARCH_LLAMA and friends.
    Norm,
    /// GPT-NeoX RoPE: rotate pairs offset by `n_rot/2`, i.e.
    /// `(x[j], x[j + n_rot/2])` for `j in 0..n_rot/2`.
    /// **This is what LLM_ARCH_BITNET_B158 uses**
    /// (`src/llama.cpp:20117` of the pinned commit).
    Neox,
}

/// Apply RoPE rotation in place to one head's worth of values.
///
/// * `x.len()` must equal `head_dim`.
/// * `n_rot` is the number of rotated dimensions (must be even and
///   ≤ `head_dim`). For BitNet b1.58 we hold `n_rot == head_dim == 128`,
///   so full-head rotation is the canonical case.
/// * `position` is the 0-based token position.
/// * `freq_base` is the RoPE base θ₀ (500_000 for BitNet b1.58).
/// * `rope_type` selects pairing layout. Pass [`RopeType::Neox`] for
///   BitNet b1.58.
pub fn apply_rope_f32(
    x: &mut [f32],
    head_dim: usize,
    n_rot: usize,
    position: u32,
    freq_base: f32,
    rope_type: RopeType,
) -> Result<(), WillametteError> {
    if x.len() != head_dim {
        return Err(WillametteError::GgufParse(format!(
            "apply_rope_f32: x.len()={} != head_dim={}",
            x.len(),
            head_dim
        )));
    }
    if n_rot == 0 || !n_rot.is_multiple_of(2) {
        return Err(WillametteError::GgufParse(format!(
            "apply_rope_f32: n_rot={} must be a positive even number",
            n_rot
        )));
    }
    if n_rot > head_dim {
        return Err(WillametteError::GgufParse(format!(
            "apply_rope_f32: n_rot={} > head_dim={}",
            n_rot, head_dim
        )));
    }
    let half = n_rot / 2;
    let pos = position as f32;
    // θ_j = pos × freq_base^(-2j / n_rot) for j in 0..half
    let n_rot_f = n_rot as f32;

    match rope_type {
        RopeType::Norm => {
            // Pairs (x[2j], x[2j+1])
            for j in 0..half {
                let exponent = -2.0 * (j as f32) / n_rot_f;
                let theta = pos * freq_base.powf(exponent);
                let (sin_t, cos_t) = theta.sin_cos();
                let a = x[2 * j];
                let b = x[2 * j + 1];
                x[2 * j] = a * cos_t - b * sin_t;
                x[2 * j + 1] = a * sin_t + b * cos_t;
            }
        }
        RopeType::Neox => {
            // Pairs (x[j], x[j + half])
            for j in 0..half {
                let exponent = -2.0 * (j as f32) / n_rot_f;
                let theta = pos * freq_base.powf(exponent);
                let (sin_t, cos_t) = theta.sin_cos();
                let a = x[j];
                let b = x[j + half];
                x[j] = a * cos_t - b * sin_t;
                x[j + half] = a * sin_t + b * cos_t;
            }
        }
    }
    // Dimensions in [n_rot..head_dim) are NOT rotated (no-op).
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Attention shape helpers (no matmul, no softmax)
// ──────────────────────────────────────────────────────────────────────────

/// Standard scaled-dot-product attention scale: `1 / sqrt(head_dim)`.
/// Matches `1.0f/sqrtf(float(n_embd_head))` in `src/llama.cpp:15466`.
#[inline]
pub fn attention_scale(head_dim: usize) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

/// GQA group size: how many Q heads share one K/V head. Must be exact.
pub fn gqa_group_size(n_heads: u32, n_kv_heads: u32) -> Result<u32, WillametteError> {
    if n_kv_heads == 0 {
        return Err(WillametteError::GgufParse(
            "gqa_group_size: n_kv_heads must be > 0".to_string(),
        ));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(WillametteError::GgufParse(format!(
            "gqa_group_size: n_heads ({}) not divisible by n_kv_heads ({})",
            n_heads, n_kv_heads
        )));
    }
    Ok(n_heads / n_kv_heads)
}

/// Given a Q head index and the GQA group size, return the K/V head it
/// should attend against.
#[inline]
pub fn kv_head_for_q_head(q_head: u32, group_size: u32) -> u32 {
    q_head / group_size
}

/// Causal mask value for one (query position, key position) pair, in the
/// "bias added to logits before softmax" convention.
///
/// Returns `0.0` when the key is allowed (k_pos ≤ q_pos) and
/// `f32::NEG_INFINITY` otherwise.
#[inline]
pub fn causal_mask_value(q_pos: u32, k_pos: u32) -> f32 {
    if k_pos <= q_pos {
        0.0
    } else {
        f32::NEG_INFINITY
    }
}

/// Logical attention shapes computed once from a `BitNetConfig`.
///
/// `Q` will be a `[n_heads, head_dim]` block per token,
/// `K` and `V` will each be a `[n_kv_heads, head_dim]` block per token.
/// During attention each Q head reads the K/V head selected by
/// `kv_head_for_q_head`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionShape {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub group_size: u32,
    pub q_per_token_dim: u32,
    pub kv_per_token_dim: u32,
}

impl AttentionShape {
    pub fn from_config(
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<Self, WillametteError> {
        let group_size = gqa_group_size(n_heads, n_kv_heads)?;
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            group_size,
            q_per_token_dim: n_heads * head_dim,
            kv_per_token_dim: n_kv_heads * head_dim,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        // +inf
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00) > 0.0);
        // -inf
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00) < 0.0);
        // NaN
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn f16_smallest_subnormal_is_finite_and_positive() {
        let v = f16_to_f32(0x0001);
        assert!(v.is_sign_positive());
        assert!(v.is_finite());
        assert!(v > 0.0);
        // 2^-24 ≈ 5.96e-8
        assert!((v - (1.0 / (1u32 << 24) as f32)).abs() < 1e-12);
    }

    #[test]
    fn rms_norm_preserves_length_and_handles_unit_weight() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let w = vec![1.0_f32; 4];
        let mut out = vec![0.0_f32; 4];
        rms_norm_f32(&x, &w, 1e-5, &mut out).unwrap();
        // mean(x²) = (1+4+9+16)/4 = 7.5; rsqrt = 1/sqrt(7.5 + 1e-5)
        let denom = (7.5_f32 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = x[i] / denom;
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "i={} out={} expected={}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn rms_norm_length_mismatch_errors() {
        let x = vec![1.0_f32; 4];
        let w = vec![1.0_f32; 3];
        let mut out = vec![0.0_f32; 4];
        assert!(rms_norm_f32(&x, &w, 1e-5, &mut out).is_err());
    }

    #[test]
    fn rope_zero_position_is_identity() {
        let mut x = vec![0.1, 0.2, 0.3, 0.4_f32];
        let orig = x.clone();
        apply_rope_f32(&mut x, 4, 4, 0, 10000.0, RopeType::Neox).unwrap();
        for i in 0..4 {
            assert!(
                (x[i] - orig[i]).abs() < 1e-6,
                "pos=0 should be identity at i={}",
                i
            );
        }
    }

    #[test]
    fn rope_preserves_per_pair_norm() {
        // Each rotated pair is a 2D rotation, so its L2 norm is invariant.
        let mut x = vec![0.4_f32, 0.7, 0.1, -0.5, 0.2, 0.3, -0.6, 0.05];
        let orig = x.clone();
        apply_rope_f32(&mut x, 8, 8, 3, 500_000.0, RopeType::Neox).unwrap();
        // NEOX pairs: (i, i+4) for i in 0..4
        for j in 0..4 {
            let pre = orig[j].powi(2) + orig[j + 4].powi(2);
            let post = x[j].powi(2) + x[j + 4].powi(2);
            assert!(
                (pre - post).abs() < 1e-5,
                "NEOX pair {}/{} norm changed: {} -> {}",
                j,
                j + 4,
                pre,
                post
            );
        }
    }

    #[test]
    fn rope_norm_and_neox_produce_different_outputs() {
        let mut a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let mut b = a.clone();
        apply_rope_f32(&mut a, 4, 4, 1, 10000.0, RopeType::Norm).unwrap();
        apply_rope_f32(&mut b, 4, 4, 1, 10000.0, RopeType::Neox).unwrap();
        // For non-zero position with this input, the two pairings must
        // produce different outputs — confirming that picking the wrong
        // type would be silently incorrect, not silently equal.
        let mut same = true;
        for i in 0..4 {
            if (a[i] - b[i]).abs() > 1e-6 {
                same = false;
                break;
            }
        }
        assert!(!same, "Norm and NEOX must differ for this input");
    }

    #[test]
    fn rope_invalid_n_rot_errors() {
        let mut x = vec![0.0_f32; 4];
        assert!(apply_rope_f32(&mut x, 4, 3, 0, 10000.0, RopeType::Neox).is_err()); // odd
        assert!(apply_rope_f32(&mut x, 4, 8, 0, 10000.0, RopeType::Neox).is_err());
        // > head_dim
    }

    #[test]
    fn attention_scale_is_inverse_sqrt() {
        let s = attention_scale(128);
        let expected = 1.0 / (128.0_f32).sqrt();
        assert!((s - expected).abs() < 1e-7);
    }

    #[test]
    fn gqa_group_size_known_value() {
        assert_eq!(gqa_group_size(20, 5).unwrap(), 4);
        assert_eq!(kv_head_for_q_head(0, 4), 0);
        assert_eq!(kv_head_for_q_head(3, 4), 0);
        assert_eq!(kv_head_for_q_head(4, 4), 1);
        assert_eq!(kv_head_for_q_head(19, 4), 4);
    }

    #[test]
    fn gqa_group_size_non_divisible_errors() {
        assert!(gqa_group_size(20, 6).is_err());
        assert!(gqa_group_size(20, 0).is_err());
    }

    #[test]
    fn causal_mask_basic() {
        assert_eq!(causal_mask_value(5, 0), 0.0);
        assert_eq!(causal_mask_value(5, 5), 0.0);
        assert_eq!(causal_mask_value(5, 6), f32::NEG_INFINITY);
        assert_eq!(causal_mask_value(0, 1), f32::NEG_INFINITY);
    }

    #[test]
    fn attention_shape_dimensions_match() {
        let s = AttentionShape::from_config(20, 5, 128).unwrap();
        assert_eq!(s.n_heads, 20);
        assert_eq!(s.n_kv_heads, 5);
        assert_eq!(s.head_dim, 128);
        assert_eq!(s.group_size, 4);
        assert_eq!(s.q_per_token_dim, 2560);
        assert_eq!(s.kv_per_token_dim, 640);
    }
}
