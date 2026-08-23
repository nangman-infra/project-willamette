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

use std::time::{Duration, Instant};

use crate::error::WillametteError;
use crate::model::cached_forward::{
    forward_with_cache_into, prefill_with_cache_into, ForwardWorkspace, PrefillWorkspace,
};
use crate::model::forward::forward_single_token_position_zero;
use crate::model::graph::ModelGraph;
use crate::model::kv_cache::KVCache;
use crate::model::lm_head::{argmax, compute_logits_from_graph};
use crate::model::multi_forward::multi_token_forward;
use crate::model::sampler::Sampler;

/// Inference work split between processing the prompt and producing output.
///
/// `decode_tokens` counts accepted output tokens. A sampled EOS or custom stop
/// token is not returned and is therefore not included. Callback execution
/// time is excluded from both durations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenerationStats {
    pub prefill_tokens: usize,
    pub prefill_duration: Duration,
    pub decode_tokens: usize,
    pub decode_duration: Duration,
}

impl GenerationStats {
    pub fn total_tokens(&self) -> usize {
        self.prefill_tokens + self.decode_tokens
    }

    pub fn total_duration(&self) -> Duration {
        self.prefill_duration + self.decode_duration
    }
}

/// Generated token ids together with phase-specific inference statistics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationResult {
    pub generated_tokens: Vec<u32>,
    pub stats: GenerationStats,
}

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
    validate_generation_budget(
        "greedy_generate_no_cache",
        prompt_ids.len(),
        max_new_tokens,
        None,
        graph.config.context_length as usize,
    )?;

    let mut context: Vec<u32> = prompt_ids.to_vec();
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);

    for step in 0..max_new_tokens {
        let final_hidden = multi_token_forward(graph, &context)?;
        let logits = compute_logits_from_graph(&final_hidden, graph)?;
        let next = argmax(&logits).ok_or_else(|| {
            WillametteError::GgufParse("greedy_generate: empty logits".to_string())
        })?;
        if Some(next) == eos_id {
            break;
        }
        tick(step, context.len(), next);
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
    validate_generation_budget(
        "greedy_generate_with_cache",
        prompt_ids.len(),
        max_new_tokens,
        Some(max_seq_len),
        graph.config.context_length as usize,
    )?;

    let kv_dim = graph.config.kv_dim as usize;
    let n_layers = graph.layers.len();
    let mut cache = KVCache::try_new(n_layers, kv_dim, max_seq_len)?;
    let mut workspace = ForwardWorkspace::new(graph);
    let mut prefill_workspace = PrefillWorkspace::new(graph);

    // Layer-major prefill retains only the final hidden, which predicts the
    // first new token.
    let mut last_hidden: Vec<f32> = Vec::new();
    prefill_with_cache_into(
        graph,
        &mut cache,
        &mut prefill_workspace,
        prompt_ids,
        0,
        &mut last_hidden,
    )?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut next_pos = prompt_ids.len() as u32;

    for step in 0..max_new_tokens {
        let logits = compute_logits_from_graph(&last_hidden, graph)?;
        let next = argmax(&logits).ok_or_else(|| {
            WillametteError::GgufParse("greedy_generate_with_cache: empty logits".to_string())
        })?;
        if Some(next) == eos_id {
            break;
        }
        tick(step, next_pos as usize, next);
        generated.push(next);
        // Don't forward unnecessarily after the final accepted token.
        if step + 1 < max_new_tokens {
            forward_with_cache_into(
                graph,
                &mut cache,
                &mut workspace,
                next,
                next_pos,
                &mut last_hidden,
            )?;
            next_pos += 1;
        }
    }
    Ok(generated)
}

/// Generate with a user-supplied `Sampler` (temperature / top-k /
/// top-p / repetition penalty). Defaults to greedy when the sampler
/// has no knobs set. Stops on `eos_id` OR on any id in `stop_ids`.
///
/// The callback runs once for each accepted output token, after sampling and
/// stop-token checks but before that token is observed by the sampler or
/// forwarded to predict the following token.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_cache_and_sampler<F>(
    graph: &ModelGraph<'_>,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_id: Option<u32>,
    stop_ids: &[u32],
    max_seq_len: usize,
    sampler: &mut Sampler,
    tick: F,
) -> Result<Vec<u32>, WillametteError>
where
    F: FnMut(usize, usize, u32),
{
    generate_with_cache_and_sampler_with_stats(
        graph,
        prompt_ids,
        max_new_tokens,
        eos_id,
        stop_ids,
        max_seq_len,
        sampler,
        tick,
    )
    .map(|result| result.generated_tokens)
}

/// Stats-returning companion to [`generate_with_cache_and_sampler`].
///
/// Prefill time covers forwarding all prompt tokens. Decode time covers
/// logits, sampling, sampler observation, and generated-token forwards. The
/// callback has the same timing semantics as the legacy API, but time spent in
/// it is excluded from the phase durations.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_cache_and_sampler_with_stats<F>(
    graph: &ModelGraph<'_>,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_id: Option<u32>,
    stop_ids: &[u32],
    max_seq_len: usize,
    sampler: &mut Sampler,
    mut tick: F,
) -> Result<GenerationResult, WillametteError>
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
    validate_generation_budget(
        "generate_with_cache_and_sampler",
        prompt_ids.len(),
        max_new_tokens,
        Some(max_seq_len),
        graph.config.context_length as usize,
    )?;

    // Seed sampler history with the prompt tokens so repetition penalty
    // includes the user-supplied context, not just the generated tail.
    for &tid in prompt_ids {
        sampler.observe(tid);
    }

    let kv_dim = graph.config.kv_dim as usize;
    let n_layers = graph.layers.len();
    let mut cache = KVCache::try_new(n_layers, kv_dim, max_seq_len)?;
    let mut workspace = ForwardWorkspace::new(graph);
    let mut prefill_workspace = PrefillWorkspace::new(graph);

    let mut last_hidden: Vec<f32> = Vec::new();
    let prefill_start = Instant::now();
    prefill_with_cache_into(
        graph,
        &mut cache,
        &mut prefill_workspace,
        prompt_ids,
        0,
        &mut last_hidden,
    )?;
    let prefill_duration = prefill_start.elapsed();

    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut next_pos = prompt_ids.len() as u32;
    let mut decode_duration = Duration::ZERO;

    for step in 0..max_new_tokens {
        let decode_start = Instant::now();
        let logits = compute_logits_from_graph(&last_hidden, graph)?;
        let next = sampler.sample(&logits)?;
        decode_duration += decode_start.elapsed();
        if Some(next) == eos_id || stop_ids.contains(&next) {
            break;
        }
        tick(step, next_pos as usize, next);
        generated.push(next);
        let decode_start = Instant::now();
        sampler.observe(next);
        if step + 1 < max_new_tokens {
            forward_with_cache_into(
                graph,
                &mut cache,
                &mut workspace,
                next,
                next_pos,
                &mut last_hidden,
            )?;
            next_pos += 1;
        }
        decode_duration += decode_start.elapsed();
    }
    Ok(GenerationResult {
        stats: GenerationStats {
            prefill_tokens: prompt_ids.len(),
            prefill_duration,
            decode_tokens: generated.len(),
            decode_duration,
        },
        generated_tokens: generated,
    })
}

fn validate_generation_budget(
    operation: &str,
    prompt_len: usize,
    max_new_tokens: usize,
    max_seq_len: Option<usize>,
    context_length: usize,
) -> Result<(), WillametteError> {
    if let Some(max_seq_len) = max_seq_len {
        if max_seq_len > context_length {
            return Err(WillametteError::GgufParse(format!(
                "{}: max_seq_len={} exceeds model context_length={}",
                operation, max_seq_len, context_length
            )));
        }
    }

    // The final sampled token is returned to the caller but is not forwarded;
    // only earlier generated tokens consume model positions/cache entries.
    let generated_positions = max_new_tokens.saturating_sub(1);
    let needed = prompt_len.checked_add(generated_positions).ok_or_else(|| {
        WillametteError::GgufParse(format!(
            "{}: prompt({}) + generated_positions({}) overflows usize",
            operation, prompt_len, generated_positions
        ))
    })?;
    if needed > context_length {
        return Err(WillametteError::GgufParse(format!(
            "{}: prompt({}) + max_new_tokens({}) = {} exceeds model context_length={}",
            operation, prompt_len, max_new_tokens, needed, context_length
        )));
    }
    if let Some(max_seq_len) = max_seq_len {
        if needed > max_seq_len {
            return Err(WillametteError::GgufParse(format!(
                "{}: prompt({}) + max_new_tokens({}) = {} exceeds max_seq_len={}",
                operation, prompt_len, max_new_tokens, needed, max_seq_len
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_generation_budget;

    #[test]
    fn generation_budget_rejects_cache_larger_than_model_context() {
        let err = validate_generation_budget("generate", 1, 1, Some(17), 16).unwrap_err();
        assert!(err
            .to_string()
            .contains("max_seq_len=17 exceeds model context_length=16"));
    }

    #[test]
    fn generation_budget_rejects_prompt_plus_generation_overflow() {
        let err = validate_generation_budget("generate", 2, usize::MAX, Some(16), 16).unwrap_err();
        assert!(err.to_string().contains("overflows usize"));
    }

    #[test]
    fn generation_budget_rejects_model_context_overrun() {
        let err = validate_generation_budget("generate", 13, 5, Some(16), 16).unwrap_err();
        assert!(err.to_string().contains("exceeds model context_length=16"));
    }

    #[test]
    fn generation_budget_allows_one_prediction_at_full_context() {
        validate_generation_budget("generate", 16, 1, Some(16), 16).unwrap();
    }
}
