//! Stage 5-C integration tests — KV cache fidelity vs the no-cache
//! `multi_token_forward` path.
//!
//! Since the v0.9.0 i8 KV quantisation, the cached forward is no
//! longer bit-identical to the no-cache reference (i8 quant introduces
//! per-element error on the order of `absmax / 254`). What stays
//! invariant is (a) cosine similarity ≥ 0.999 on the post-`output_norm`
//! hidden state and (b) byte-identical greedy token-id sequences —
//! the property that actually matters for users. The third test below
//! enforces (b).

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::cached_forward::forward_with_cache;
use project_willamette::model::generate::{greedy_generate_no_cache, greedy_generate_with_cache};
use project_willamette::model::kv_cache::KVCache;
use project_willamette::model::multi_forward::multi_token_forward;
use project_willamette::model::ModelGraph;
use project_willamette::tokenizer::Tokenizer;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

fn skip_if_missing() -> Option<ModelMmap> {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 5-C tests require it",
            MODEL_PATH
        );
        return None;
    }
    Some(ModelMmap::open(MODEL_PATH).expect("open"))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn cache_single_token_matches_no_cache_single_token() {
    let Some(mmap) = skip_if_missing() else {
        return;
    };
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    // No-cache reference.
    let h_no_cache = multi_token_forward(&graph, &[15339]).expect("no-cache");

    // With cache: same one token at position 0.
    let mut cache = KVCache::new(graph.layers.len(), graph.config.kv_dim as usize, 16);
    let h_cache = forward_with_cache(&graph, &mut cache, 15339, 0).expect("cache");

    let cos = cosine_similarity(&h_no_cache, &h_cache);
    let m_abs = max_abs_diff(&h_no_cache, &h_cache);
    assert!(
        cos > 0.999,
        "M=1 i8 KV cache fidelity vs no-cache: cos={} (need > 0.999); max|Δ|={}",
        cos,
        m_abs
    );
    assert_eq!(cache.position(), 1);
}

#[test]
fn cache_two_token_sequence_matches_no_cache() {
    let Some(mmap) = skip_if_missing() else {
        return;
    };
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let ctx = [128000_u32, 15339];

    let h_no_cache = multi_token_forward(&graph, &ctx).expect("no-cache");

    let mut cache = KVCache::new(graph.layers.len(), graph.config.kv_dim as usize, 16);
    let mut h_cache = Vec::new();
    for (i, &tid) in ctx.iter().enumerate() {
        h_cache = forward_with_cache(&graph, &mut cache, tid, i as u32).expect("cache");
    }
    assert_eq!(cache.position(), ctx.len());
    let cos = cosine_similarity(&h_no_cache, &h_cache);
    let m_abs = max_abs_diff(&h_no_cache, &h_cache);
    assert!(
        cos > 0.999,
        "M=2 i8 KV cache fidelity vs no-cache: cos={} (need > 0.999); max|Δ|={}",
        cos,
        m_abs
    );
}

#[test]
fn greedy_with_cache_matches_greedy_no_cache_for_2_steps() {
    // For prompt "hello" (BOS + "hello"), generating 2 tokens should
    // produce identical token IDs whether or not we use a cache.
    let Some(mmap) = skip_if_missing() else {
        return;
    };
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("tokenizer");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let prompt_ids = tokenizer
        .encode("hello", tokenizer.default_encode_options())
        .expect("encode");

    let no_cache = greedy_generate_no_cache(&graph, &prompt_ids, 2, tokenizer.eos_id, |_, _, _| {})
        .expect("no-cache");

    let with_cache =
        greedy_generate_with_cache(&graph, &prompt_ids, 2, tokenizer.eos_id, 64, |_, _, _| {})
            .expect("cache");

    assert_eq!(no_cache, with_cache);
}
