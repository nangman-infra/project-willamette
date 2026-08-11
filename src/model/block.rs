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
use crate::model::architecture::ForwardVariant;
use crate::model::attention::{
    attention_block_forward_position_zero, residual_add, single_token_attention_position_zero,
};
use crate::model::config::ModelConfig;
use crate::model::ffn::{elementwise_mul, ffn_block_forward, silu};
use crate::model::graph::LayerWeights;
use crate::model::linear::linear_matvec_f32;
use crate::model::primitives::{rms_norm_f32, AttentionShape};

/// Run one transformer block on a single token at position 0.
///
/// Length checks: `x.len() == output.len() == config.embedding_length`.
pub fn transformer_block_forward_position_zero(
    x: &[f32],
    layer: &LayerWeights<'_>,
    config: &ModelConfig,
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
    let attn_sub_norm_w = layer.attn_sub_norm_f32.as_deref().ok_or_else(|| {
        WillametteError::GgufParse("BitNet layer is missing attn_sub_norm".to_string())
    })?;
    let ffn_norm_w = &layer.ffn_norm_f32;
    let ffn_sub_norm_w = layer.ffn_sub_norm_f32.as_deref().ok_or_else(|| {
        WillametteError::GgufParse("BitNet layer is missing ffn_sub_norm".to_string())
    })?;

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

/// Architecture-dispatched position-zero transformer block.
pub fn transformer_block_forward_position_zero_variant(
    x: &[f32],
    layer: &LayerWeights<'_>,
    config: &ModelConfig,
    variant: ForwardVariant,
    output: &mut [f32],
) -> Result<(), WillametteError> {
    match variant {
        ForwardVariant::BitNetSubNorm => {
            transformer_block_forward_position_zero(x, layer, config, output)
        }
        ForwardVariant::VanillaLlama => llama_block_forward_position_zero(x, layer, config, output),
    }
}

fn llama_block_forward_position_zero(
    x: &[f32],
    layer: &LayerWeights<'_>,
    config: &ModelConfig,
    output: &mut [f32],
) -> Result<(), WillametteError> {
    let n_embd = config.embedding_length as usize;
    let kv_dim = config.kv_dim as usize;
    let n_ff = config.feed_forward_length as usize;
    if x.len() != n_embd || output.len() != n_embd {
        return Err(WillametteError::GgufParse(
            "Llama transformer block hidden length mismatch".to_string(),
        ));
    }

    let mut normed = vec![0.0; n_embd];
    rms_norm_f32(
        x,
        &layer.attn_norm_f32,
        config.layer_norm_rms_epsilon,
        &mut normed,
    )?;
    let mut q = vec![0.0; n_embd];
    let mut k = vec![0.0; kv_dim];
    let mut v = vec![0.0; kv_dim];
    linear_matvec_f32(layer.attn_q, &normed, &mut q)?;
    linear_matvec_f32(layer.attn_k, &normed, &mut k)?;
    linear_matvec_f32(layer.attn_v, &normed, &mut v)?;
    // Both RoPE layouts are identity at position zero.
    let shape =
        AttentionShape::from_config(config.head_count, config.head_count_kv, config.head_dim)?;
    let mut attention = vec![0.0; n_embd];
    single_token_attention_position_zero(&q, &k, &v, shape, &mut attention)?;
    let mut projected = vec![0.0; n_embd];
    linear_matvec_f32(layer.attn_output, &attention, &mut projected)?;
    let mut residual = vec![0.0; n_embd];
    residual_add(x, &projected, &mut residual)?;

    rms_norm_f32(
        &residual,
        &layer.ffn_norm_f32,
        config.layer_norm_rms_epsilon,
        &mut normed,
    )?;
    let mut gate = vec![0.0; n_ff];
    let mut up = vec![0.0; n_ff];
    linear_matvec_f32(layer.ffn_gate, &normed, &mut gate)?;
    linear_matvec_f32(layer.ffn_up, &normed, &mut up)?;
    silu(&mut gate);
    let mut fused = vec![0.0; n_ff];
    elementwise_mul(&gate, &up, &mut fused)?;
    let mut down = vec![0.0; n_embd];
    linear_matvec_f32(layer.ffn_down, &fused, &mut down)?;
    residual_add(&residual, &down, output)
}
