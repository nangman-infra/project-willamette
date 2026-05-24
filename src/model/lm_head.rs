//! Stage 4-D5 — lm_head logits via tied `token_embd.weight`.
//!
//! For BitNet b1.58 the final projection is unconditionally tied to the
//! input embedding (`src/llama.cpp:15527`,
//! `cur = llm_build_lora_mm(lctx, ctx0, model.tok_embd, cur);`). So:
//!
//! ```text
//!   logits[v] = Σᵢ token_embd[v, i] * final_hidden[i]
//! ```
//!
//! `token_embd.weight` is F16 with shape `[embedding_length, vocab_size]`,
//! i.e. row `v` (slow axis) holds the embedding for vocab id `v` as
//! `n_embd` little-endian f16s.
//!
//! We decode each row on the fly to avoid a 1.2 GiB f32 cache.

use crate::error::WillametteError;
use crate::gguf::tensor::TensorView;
use crate::gguf::types::GgmlType;
use crate::model::graph::ModelGraph;
use crate::model::primitives::f16_to_f32;

/// Compute the full vocab-size logit vector by dotting `final_hidden`
/// against each row of an F16 `token_embd` table.
pub fn compute_logits(
    final_hidden: &[f32],
    token_embd: &TensorView<'_>,
    embedding_length: u32,
    vocab_size: u32,
) -> Result<Vec<f32>, WillametteError> {
    if token_embd.ggml_type != GgmlType::F16 {
        return Err(WillametteError::GgufParse(format!(
            "compute_logits: token_embd is {} (raw {}), expected F16",
            token_embd.ggml_type.name(),
            token_embd.ggml_type.to_raw()
        )));
    }
    if token_embd.shape != vec![embedding_length as u64, vocab_size as u64] {
        return Err(WillametteError::GgufParse(format!(
            "compute_logits: token_embd shape {:?} != [{}, {}]",
            token_embd.shape, embedding_length, vocab_size
        )));
    }
    let n_embd = embedding_length as usize;
    if final_hidden.len() != n_embd {
        return Err(WillametteError::GgufParse(format!(
            "compute_logits: final_hidden.len()={} != embedding_length={}",
            final_hidden.len(),
            n_embd
        )));
    }
    let row_bytes = n_embd * 2;
    let expected = row_bytes * (vocab_size as usize);
    if token_embd.data.len() != expected {
        return Err(WillametteError::GgufParse(format!(
            "compute_logits: token_embd.data.len()={} != expected {}",
            token_embd.data.len(),
            expected
        )));
    }

    let mut logits = vec![0.0_f32; vocab_size as usize];
    for v in 0..vocab_size as usize {
        let row = &token_embd.data[v * row_bytes..(v + 1) * row_bytes];
        let mut s = 0.0_f32;
        for i in 0..n_embd {
            let lo = row[2 * i] as u16;
            let hi = row[2 * i + 1] as u16;
            let bits = lo | (hi << 8);
            let w = f16_to_f32(bits);
            s += w * final_hidden[i];
        }
        logits[v] = s;
    }
    Ok(logits)
}

/// Convenience: compute_logits with all four shape arguments pulled from
/// a `ModelGraph`.
pub fn compute_logits_from_graph(
    final_hidden: &[f32],
    graph: &ModelGraph<'_>,
) -> Result<Vec<f32>, WillametteError> {
    compute_logits(
        final_hidden,
        graph.lm_head, // already tied to token_embd by ModelGraph
        graph.config.embedding_length,
        graph.config.vocab_size,
    )
}

/// Return the vocab id with the highest logit. Returns `None` only if
/// `logits` is empty.
pub fn argmax(logits: &[f32]) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    let mut best_idx = 0u32;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_idx = i as u32;
        }
    }
    Some(best_idx)
}

/// Return the top-`k` `(id, logit)` pairs sorted by descending logit.
/// `k` is clamped to `logits.len()`.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_returns_index_of_largest() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3]), Some(1));
        assert_eq!(argmax(&[]), None);
        assert_eq!(argmax(&[0.5]), Some(0));
        assert_eq!(argmax(&[1.0, 1.0]), Some(0)); // ties → first
    }

    #[test]
    fn top_k_basic() {
        let logits = vec![0.1_f32, 0.5, 0.3, 0.9, 0.2];
        let top = top_k(&logits, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0], (3, 0.9));
        assert_eq!(top[1], (1, 0.5));
        assert_eq!(top[2], (2, 0.3));
    }

    #[test]
    fn top_k_clamps_to_len() {
        let logits = vec![0.1_f32, 0.5];
        let top = top_k(&logits, 10);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let logits = vec![0.1_f32, 0.5];
        let top = top_k(&logits, 0);
        assert!(top.is_empty());
    }
}
