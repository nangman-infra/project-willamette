//! Greedy generation against the real model.
//!
//! Two entry points, in order of correctness:
//!
//!   * Stage 5-A — `greedy_next_token_from_single_position_zero`:
//!     forwards a single token at position 0. Used only when the prompt
//!     is one token long; otherwise produces a degenerate prediction.
//!   * Stage 5-B — `greedy_generate_no_cache`: multi-token causal
//!     forward (Stage 5-B/`multi_token_forward`) recomputed from
//!     scratch each step. Slow but correct. EOS-aware.
//!
//! No sampling, no temperature, no repetition penalty in this module —
//! pure greedy argmax. Sampling lives in Stage 5-D.

use crate::error::WillametteError;
use crate::model::cached_forward::forward_with_cache;
use crate::model::forward::forward_single_token_position_zero;
use crate::model::graph::ModelGraph;
use crate::model::kv_cache::KVCache;
use crate::model::lm_head::{argmax, compute_logits_from_graph};
use crate::model::multi_forward::multi_token_forward;
use crate::model::sampler::Sampler;

/// Run one greedy decode step from a single token at position 0:
///
/// `last_token_id → forward (30 layers) → output_norm → tied lm_head logits → argmax`.
///
/// Returns the predicted next-token id.
pub fn greedy_next_token_from_single_position_zero(
    graph: &ModelGraph<'_>,
    last_token_id: u32,
) -> Result<u32, WillametteError> {
    if last_token_id >= graph.config.vocab_size {
        return Err(WillametteError::GgufParse(format!(
            "greedy_next_token: token id {} out of vocab range (size {})",
            last_token_id, graph.config.vocab_size
        )));
    }
    let hidden = forward_single_token_position_zero(graph, last_token_id)?;
    let logits = compute_logits_from_graph(&hidden, graph)?;
    argmax(&logits)
        .ok_or_else(|| WillametteError::GgufParse("greedy_next_token: empty logits vector".into()))
}

/// Greedy autoregressive decode without a KV cache. Each step
/// recomputes the entire context (prompt + previously generated tokens)
/// from scratch via `multi_token_forward`.
///
/// Stops early when `eos_id == Some(generated_token)` (if `eos_id` is
/// supplied). Returns only the **newly generated** ids (the prompt is
/// not included).
///
/// Optional `tick` callback fires once per generated token with
/// `(step, total_ctx_len, new_token_id)` so callers can stream output.
pub fn greedy_generate_no_cache<F>(
    graph: &ModelGraph<'_>,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_id: Option<u32>,
    mut tick: F,
) -> Result<Vec<u32>, WillametteError>
where
    F: FnMut(usize, usize, u32),
{
    if prompt_ids.is_empty() {
        return Err(WillametteError::GgufParse(
            "greedy_generate_no_cache: prompt_ids must not be empty".to_string(),
        ));
    }
    for (i, &tid) in prompt_ids.iter().enumerate() {
        if tid >= graph.config.vocab_size {
            return Err(WillametteError::GgufParse(format!(
                "greedy_generate_no_cache: prompt token {} (idx {}) out of vocab range {}",
                tid, i, graph.config.vocab_size
            )));
        }
    }

    let mut context: Vec<u32> = prompt_ids.to_vec();
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);

    for step in 0..max_new_tokens {
        let final_hidden = multi_token_forward(graph, &context)?;
        let logits = compute_logits_from_graph(&final_hidden, graph)?;
        let next = argmax(&logits).ok_or_else(|| {
            WillametteError::GgufParse("greedy_generate: empty logits".to_string())
        })?;
        tick(step, context.len(), next);
        if Some(next) == eos_id {
            break;
        }
        context.push(next);
        generated.push(next);
    }
    Ok(generated)
}

/// Greedy autoregressive decode with a `KVCache`. Marginal cost per
/// generated token is one single-token-equivalent forward — far better
/// than `greedy_generate_no_cache` which re-runs the whole context.
///
/// `max_seq_len` is the cache capacity; choose `prompt_len + max_new_tokens
/// + slack`.
pub fn greedy_generate_with_cache<F>(
    graph: &ModelGraph<'_>,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_id: Option<u32>,
    max_seq_len: usize,
    mut tick: F,
) -> Result<Vec<u32>, WillametteError>
where
    F: FnMut(usize, usize, u32),
{
    if prompt_ids.is_empty() {
        return Err(WillametteError::GgufParse(
            "greedy_generate_with_cache: prompt_ids must not be empty".to_string(),
        ));
    }
    for (i, &tid) in prompt_ids.iter().enumerate() {
        if tid >= graph.config.vocab_size {
            return Err(WillametteError::GgufParse(format!(
                "greedy_generate_with_cache: prompt token {} (idx {}) out of vocab range {}",
                tid, i, graph.config.vocab_size
            )));
        }
    }
    let needed = prompt_ids.len() + max_new_tokens;
    if needed > max_seq_len {
        return Err(WillametteError::GgufParse(format!(
            "greedy_generate_with_cache: prompt({}) + max_new_tokens({}) = {} exceeds max_seq_len={}",
            prompt_ids.len(),
            max_new_tokens,
            needed,
            max_seq_len
        )));
    }

    let kv_dim = graph.config.kv_dim as usize;
    let n_layers = graph.layers.len();
    let mut cache = KVCache::new(n_layers, kv_dim, max_seq_len);

    // Prefill: process every prompt token in order. Retain only the
    // final hidden — that's what predicts the first new token.
    let mut last_hidden: Vec<f32> = Vec::new();
    for (i, &tid) in prompt_ids.iter().enumerate() {
        last_hidden = forward_with_cache(graph, &mut cache, tid, i as u32)?;
    }

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut next_pos = prompt_ids.len() as u32;

    for step in 0..max_new_tokens {
        let logits = compute_logits_from_graph(&last_hidden, graph)?;
        let next = argmax(&logits).ok_or_else(|| {
            WillametteError::GgufParse("greedy_generate_with_cache: empty logits".to_string())
        })?;
        tick(step, next_pos as usize, next);
        if Some(next) == eos_id {
            break;
        }
        generated.push(next);
        // Don't forward unnecessarily after the final accepted token.
        if step + 1 < max_new_tokens {
            last_hidden = forward_with_cache(graph, &mut cache, next, next_pos)?;
            next_pos += 1;
        }
    }
    Ok(generated)
}

/// Generate with a user-supplied `Sampler` (temperature / top-k /
/// top-p / repetition penalty). Defaults to greedy when the sampler
/// has no knobs set. Stops on `eos_id` OR on any id in `stop_ids`.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_cache_and_sampler<F>(
    graph: &ModelGraph<'_>,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_id: Option<u32>,
    stop_ids: &[u32],
    max_seq_len: usize,
    sampler: &mut Sampler,
    mut tick: F,
) -> Result<Vec<u32>, WillametteError>
where
    F: FnMut(usize, usize, u32),
{
    if prompt_ids.is_empty() {
        return Err(WillametteError::GgufParse(
            "generate_with_cache_and_sampler: prompt_ids must not be empty".to_string(),
        ));
    }
    for (i, &tid) in prompt_ids.iter().enumerate() {
        if tid >= graph.config.vocab_size {
            return Err(WillametteError::GgufParse(format!(
                "generate_with_cache_and_sampler: prompt token {} (idx {}) out of vocab range {}",
                tid, i, graph.config.vocab_size
            )));
        }
    }
    let needed = prompt_ids.len() + max_new_tokens;
    if needed > max_seq_len {
        return Err(WillametteError::GgufParse(format!(
            "generate_with_cache_and_sampler: prompt({}) + max_new_tokens({}) = {} exceeds max_seq_len={}",
            prompt_ids.len(),
            max_new_tokens,
            needed,
            max_seq_len
        )));
    }

    // Seed sampler history with the prompt tokens so repetition penalty
    // includes the user-supplied context, not just the generated tail.
    for &tid in prompt_ids {
        sampler.observe(tid);
    }

    let kv_dim = graph.config.kv_dim as usize;
    let n_layers = graph.layers.len();
    let mut cache = KVCache::new(n_layers, kv_dim, max_seq_len);

    let mut last_hidden: Vec<f32> = Vec::new();
    for (i, &tid) in prompt_ids.iter().enumerate() {
        last_hidden = forward_with_cache(graph, &mut cache, tid, i as u32)?;
    }

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut next_pos = prompt_ids.len() as u32;

    for step in 0..max_new_tokens {
        let logits = compute_logits_from_graph(&last_hidden, graph)?;
        let next = sampler.sample(&logits)?;
        tick(step, next_pos as usize, next);
        if Some(next) == eos_id || stop_ids.contains(&next) {
            break;
        }
        generated.push(next);
        sampler.observe(next);
        if step + 1 < max_new_tokens {
            last_hidden = forward_with_cache(graph, &mut cache, next, next_pos)?;
            next_pos += 1;
        }
    }
    Ok(generated)
}
