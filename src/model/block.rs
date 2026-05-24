//! Stage 4-D3 — single transformer block forward for BitNet b1.58.
//!
//! Composes the Stage 4-D1 attention path and the Stage 4-D2 FFN path
//! with the two residual additions:
//!
//! ```text
//!   x_mid = x + attention_block(x, attn_norm, Wq, Wk, Wv, Wo, attn_sub_norm)
//!   x_out = x_mid + ffn_block(x_mid, ffn_norm, Wg, Wu, Wd, ffn_sub_norm)
//! ```
//!
//! Pinned against `build_bitnet_158` (`src/llama.cpp:15412..15523`):
//!
//!   * `inpSA = inpL`
//!   * attention block produces `cur`
//!   * `ffn_inp = ggml_add(cur, inpSA)`   ← residual #1
//!   * FFN block on `ffn_inp` produces `cur`
//!   * `cur = ggml_add(cur, ffn_inp)`     ← residual #2
//!   * `inpL = cur` (input for next layer)
//!
//! Stage 4-D3 covers the single-token, position-0 case only — it
//! delegates to the position-0 attention path. Stage 5-B/5-C will
//! generalise.

use crate::error::WillametteError;
use crate::model::attention::{attention_block_forward_position_zero, residual_add};
use crate::model::config::BitNetConfig;
use crate::model::ffn::ffn_block_forward;
use crate::model::graph::LayerWeights;

/// Run one transformer block on a single token at position 0.
///
/// Length checks: `x.len() == output.len() == config.embedding_length`.
pub fn transformer_block_forward_position_zero(
    x: &[f32],
    layer: &LayerWeights<'_>,
    config: &BitNetConfig,
    output: &mut [f32],
) -> Result<(), WillametteError> {
    let n_embd = config.embedding_length as usize;
    if x.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "transformer_block_forward: x.len()={} != n_embd={}",
            x.len(),
            n_embd
        )));
    }
    if output.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "transformer_block_forward: output.len()={} != n_embd={}",
            output.len(),
            n_embd
        )));
    }

    // Stage 10-A: norm weights are pre-decoded in ModelGraph::from_gguf
    // — no per-call allocation needed.
    let attn_norm_w = &layer.attn_norm_f32;
    let attn_sub_norm_w = &layer.attn_sub_norm_f32;
    let ffn_norm_w = &layer.ffn_norm_f32;
    let ffn_sub_norm_w = &layer.ffn_sub_norm_f32;

    // Attention half.
    let mut attn_out = vec![0.0_f32; n_embd];
    attention_block_forward_position_zero(
        x,
        attn_norm_w,
        layer.attn_q,
        layer.attn_k,
        layer.attn_v,
        layer.attn_output,
        attn_sub_norm_w,
        config,
        &mut attn_out,
    )?;

    // Residual #1: x_mid = x + attn_out.
    let mut x_mid = vec![0.0_f32; n_embd];
    residual_add(x, &attn_out, &mut x_mid)?;

    // FFN half on the residual'd state.
    let mut ffn_out = vec![0.0_f32; n_embd];
    ffn_block_forward(
        &x_mid,
        ffn_norm_w,
        layer.ffn_gate,
        layer.ffn_up,
        layer.ffn_down,
        ffn_sub_norm_w,
        config,
        &mut ffn_out,
    )?;

    // Residual #2: output = x_mid + ffn_out.
    residual_add(&x_mid, &ffn_out, output)?;
    Ok(())
}
