//! Stage 4-D2 — single-token FFN path for BitNet b1.58.
//!
//! Composes Stage 4-B/4-C primitives into the FFN half of one transformer
//! block. The activation is **ReLU²** (per `LLM_FFN_RELU_SQR` at
//! `src/llama.cpp:9605..9613` of the pinned commit), NOT SiLU/GeLU. The
//! topology is **parallel-gated** (`LLM_FFN_PAR` at
//! `src/llama.cpp:9564..9568` and `9628..9631`), meaning `gate` and `up`
//! both consume the same `x_norm` input, then `relu²(gate) * up` is fed
//! into `ffn_sub_norm` and the final BitLinear `ffn_down`.
//!
//! Pinned sequence from `build_bitnet_158` (`src/llama.cpp:15495..15518`):
//!
//! ```text
//!   cur = RMSNorm(ffn_inp, ffn_norm)
//!   tmp = matmul(ffn_up,   cur)         // both reads same x_norm
//!   cur = matmul(ffn_gate, cur)
//!   cur = ggml_relu(cur); cur = ggml_sqr(cur)  // = ReLU²
//!   cur = cur * tmp                      // PAR multiply
//!   cur = RMSNorm(cur, ffn_sub_norm)     // ffn_sub_norm width = n_ff
//!   cur = matmul(ffn_down, cur)
//! ```
//!
//! Note: `llm_build_ffn` is called with `down=NULL` so it stops after the
//! gated multiply; `build_bitnet_158` then runs the post-norm and
//! `ffn_down` BitLinear separately. This file collapses both into a
//! single function for ergonomic Stage 4-D2 testing, matching the
//! eventual block-level forward.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::model::bitlinear::bitlinear_i2s_matvec_f32;
use crate::model::config::BitNetConfig;
use crate::model::primitives::rms_norm_f32;

/// In-place `relu(x)² ` — negative entries clamp to 0, non-negative
/// entries get squared.
pub fn relu_square(x: &mut [f32]) {
    for v in x.iter_mut() {
        let r = if *v > 0.0 { *v } else { 0.0 };
        *v = r * r;
    }
}

/// Element-wise multiply `out[i] = a[i] * b[i]`. Length-checked.
pub fn elementwise_mul(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), WillametteError> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(WillametteError::GgufParse(format!(
            "elementwise_mul: length mismatch a={} b={} out={}",
            a.len(),
            b.len(),
            out.len()
        )));
    }
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
    Ok(())
}

/// FFN block forward for a single token.
///
/// Sequence: `ffn_norm` → (`ffn_gate`, `ffn_up`) BitLinear → ReLU² on
/// gate → multiply by up → `ffn_sub_norm` (width = `n_ff`) → `ffn_down`
/// BitLinear. Residual addition is NOT applied here.
#[allow(clippy::too_many_arguments)]
pub fn ffn_block_forward(
    x: &[f32],
    ffn_norm_weight: &[f32],
    ffn_gate: &TensorView<'_>,
    ffn_up: &TensorView<'_>,
    ffn_down: &TensorView<'_>,
    ffn_sub_norm_weight: &[f32],
    config: &BitNetConfig,
    output: &mut [f32],
) -> Result<(), WillametteError> {
    let n_embd = config.embedding_length as usize;
    let n_ff = config.feed_forward_length as usize;
    let eps = config.layer_norm_rms_epsilon;

    if x.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "ffn_block_forward: x.len()={} != n_embd={}",
            x.len(),
            n_embd
        )));
    }
    if ffn_norm_weight.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "ffn_block_forward: ffn_norm_weight.len()={} != n_embd={}",
            ffn_norm_weight.len(),
            n_embd
        )));
    }
    if ffn_sub_norm_weight.len() != n_ff {
        return Err(WillametteError::GgufParse(format!(
            "ffn_block_forward: ffn_sub_norm_weight.len()={} != n_ff={}",
            ffn_sub_norm_weight.len(),
            n_ff
        )));
    }
    if output.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "ffn_block_forward: output.len()={} != n_embd={}",
            output.len(),
            n_embd
        )));
    }

    let mut x_norm = vec![0.0_f32; n_embd];
    rms_norm_f32(x, ffn_norm_weight, eps, &mut x_norm)?;

    let mut gate = vec![0.0_f32; n_ff];
    let mut up = vec![0.0_f32; n_ff];
    bitlinear_i2s_matvec_f32(ffn_gate, &x_norm, &mut gate)?;
    bitlinear_i2s_matvec_f32(ffn_up, &x_norm, &mut up)?;

    relu_square(&mut gate);

    // `gate` is now ReLU²(gate). Fuse with up element-wise into a scratch
    // buffer so we don't lose the unfused `up` values.
    let mut fused = vec![0.0_f32; n_ff];
    elementwise_mul(&gate, &up, &mut fused)?;

    let mut fused_norm = vec![0.0_f32; n_ff];
    rms_norm_f32(&fused, ffn_sub_norm_weight, eps, &mut fused_norm)?;

    bitlinear_i2s_matvec_f32(ffn_down, &fused_norm, output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_square_clamps_negatives_and_squares() {
        let mut x = vec![-2.0_f32, -0.5, 0.0, 0.5, 2.0, 3.0];
        relu_square(&mut x);
        assert_eq!(x, vec![0.0, 0.0, 0.0, 0.25, 4.0, 9.0]);
    }

    #[test]
    fn relu_square_handles_neg_inf_and_inf() {
        let mut x = vec![f32::NEG_INFINITY, f32::INFINITY];
        relu_square(&mut x);
        assert_eq!(x[0], 0.0);
        assert!(x[1].is_infinite() && x[1].is_sign_positive());
    }

    #[test]
    fn elementwise_mul_basic() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let mut out = vec![0.0_f32; 3];
        elementwise_mul(&a, &b, &mut out).unwrap();
        assert_eq!(out, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn elementwise_mul_length_mismatch_errors() {
        let a = vec![1.0; 3];
        let b = vec![1.0; 4];
        let mut out = vec![0.0; 3];
        assert!(elementwise_mul(&a, &b, &mut out).is_err());
    }
}
