//! Stage 4-D4 — 30-layer single-token forward for BitNet b1.58.
//!
//! Returns the final hidden state (length `n_embd`) AFTER `output_norm`
//! is applied. Logits and sampling are Stage 4-D5 / Stage 5; this module
//! deliberately stops at the final hidden state.
//!
//! Sequence (pinned against `build_bitnet_158` final lines,
//! `src/llama.cpp:15524..15528`):
//!
//! ```text
//!   hidden = embed(token_id)
//!   for il in 0..n_layer:
//!       hidden = transformer_block(hidden, layer[il])
//!   hidden = RMSNorm(hidden, output_norm)
//!   return hidden
//! ```
//!
//! Stage 4-D4 is **single-token, position-0 only** — no KV cache, no
//! attention to past tokens. Multi-token forward (with or without cache)
//! is Stage 5-B / 5-C.

use crate::error::WillametteError;
use crate::model::block::transformer_block_forward_position_zero;
use crate::model::graph::ModelGraph;
use crate::model::primitives::{embedding_gather_f16, f32_tensor_to_vec, rms_norm_f32};

/// Single-token forward at position 0. Returns the post-`output_norm`
/// hidden state (length `n_embd`). Does NOT compute logits.
pub fn forward_single_token_position_zero(
    graph: &ModelGraph<'_>,
    token_id: u32,
) -> Result<Vec<f32>, WillametteError> {
    let n_embd = graph.config.embedding_length as usize;

    let mut hidden_a = vec![0.0_f32; n_embd];
    embedding_gather_f16(graph.token_embd, token_id, &mut hidden_a)?;

    let mut hidden_b = vec![0.0_f32; n_embd];
    for layer in &graph.layers {
        transformer_block_forward_position_zero(&hidden_a, layer, &graph.config, &mut hidden_b)?;
        std::mem::swap(&mut hidden_a, &mut hidden_b);
        // Defensive: if any layer produced non-finite values we want to
        // fail loudly at the boundary, not 29 layers later.
        if let Some((dim, v)) = hidden_a.iter().enumerate().find(|(_, &v)| !v.is_finite()) {
            return Err(WillametteError::GgufParse(format!(
                "forward_single_token: non-finite value {} at hidden dim {} after layer {}",
                v, dim, layer.index
            )));
        }
    }

    // Final output_norm.
    let on_w = f32_tensor_to_vec(graph.output_norm)?;
    let mut final_hidden = vec![0.0_f32; n_embd];
    rms_norm_f32(
        &hidden_a,
        &on_w,
        graph.config.layer_norm_rms_epsilon,
        &mut final_hidden,
    )?;
    Ok(final_hidden)
}
