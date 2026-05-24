//! Stage 5-B integration tests — multi-token causal forward + greedy
//! generation without a KV cache.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::forward::forward_single_token_position_zero;
use project_willamette::model::generate::greedy_generate_no_cache;
use project_willamette::model::multi_forward::multi_token_forward;
use project_willamette::model::ModelGraph;
use project_willamette::tokenizer::Tokenizer;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

fn maybe_open() -> Option<(ModelMmap,)> {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 5-B tests require it",
            MODEL_PATH
        );
        return None;
    }
    Some((ModelMmap::open(MODEL_PATH).expect("open"),))
}

#[test]
fn multi_token_with_one_token_matches_single_token_forward() {
    let Some((mmap,)) = maybe_open() else {
        return;
    };
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let h_single = forward_single_token_position_zero(&graph, 15339).expect("single");
    let h_multi = multi_token_forward(&graph, &[15339]).expect("multi");
    // The single-token reference and the M=1 multi-token path use the
    // same primitives in the same order, so the outputs should match
    // bit-for-bit.
    assert_eq!(h_single.len(), h_multi.len());
    assert_eq!(h_single, h_multi);
}

#[test]
fn multi_token_two_token_produces_finite_hidden() {
    let Some((mmap,)) = maybe_open() else {
        return;
    };
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let h = multi_token_forward(&graph, &[128000, 15339]).expect("two-token");
    assert_eq!(h.len(), graph.config.embedding_length as usize);
    for (i, &v) in h.iter().enumerate() {
        assert!(v.is_finite(), "non-finite hidden at dim {}: {}", i, v);
    }
}

#[test]
fn multi_token_is_deterministic() {
    let Some((mmap,)) = maybe_open() else {
        return;
    };
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");
    let a = multi_token_forward(&graph, &[128000, 15339]).expect("a");
    let b = multi_token_forward(&graph, &[128000, 15339]).expect("b");
    assert_eq!(a, b);
}

#[test]
fn greedy_generate_2_tokens_produces_in_range_ids() {
    let Some((mmap,)) = maybe_open() else {
        return;
    };
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("tokenizer");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let prompt_ids = tokenizer
        .encode("hello", tokenizer.default_encode_options())
        .unwrap();
    let mut ticks = 0usize;
    let generated =
        greedy_generate_no_cache(&graph, &prompt_ids, 2, tokenizer.eos_id, |_, _, _| {
            ticks += 1;
        })
        .expect("generate");
    // We asked for up to 2; EOS may have shortened it.
    assert!(generated.len() <= 2);
    assert!(ticks >= generated.len());
    for &id in &generated {
        assert!(id < graph.config.vocab_size);
    }
}

#[test]
fn greedy_generate_eos_short_circuit() {
    // Force the EOS id to be whatever argmax actually returns at step 0
    // — this is a tautological-looking but useful test: it confirms the
    // EOS path stops the loop immediately and returns zero generated
    // tokens. We avoid hardcoding a particular argmax value.
    let Some((mmap,)) = maybe_open() else {
        return;
    };
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse");
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata).expect("tokenizer");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");
    let prompt_ids = tokenizer
        .encode("hello", tokenizer.default_encode_options())
        .unwrap();

    // First find what the argmax will be on this context.
    let h = multi_token_forward(&graph, &prompt_ids).unwrap();
    let logits = project_willamette::model::lm_head::compute_logits_from_graph(&h, &graph).unwrap();
    let pretend_eos = project_willamette::model::lm_head::argmax(&logits).unwrap();

    let generated =
        greedy_generate_no_cache(&graph, &prompt_ids, 10, Some(pretend_eos), |_, _, _| {})
            .expect("generate");
    assert!(
        generated.is_empty(),
        "EOS at first step should produce no generated tokens, got {:?}",
        generated
    );
}
