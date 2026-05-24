//! Stage 5-C integration tests — KV cache numerical equivalence with
//! the no-cache `multi_token_forward` path.

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

    assert_eq!(
        h_no_cache, h_cache,
        "M=1 cache and no-cache must match exactly"
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
    assert_eq!(
        h_no_cache, h_cache,
        "two-token cache vs no-cache must match exactly"
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
