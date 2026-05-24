//! Stage 4-D5 integration tests — lm_head logits against real GGUF.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::forward::forward_single_token_position_zero;
use project_willamette::model::lm_head::{argmax, compute_logits_from_graph, top_k};
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 4-D5 tests require it",
            MODEL_PATH
        );
        return;
    }
    let mmap = ModelMmap::open(MODEL_PATH).expect("open model");
    let bytes = mmap.as_bytes();
    let gguf = GgufFile::parse(bytes).expect("parse model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build model graph");
    f(&graph);
}

#[test]
fn logits_have_vocab_size_length_and_are_finite() {
    with_real_graph(|g| {
        let hidden = forward_single_token_position_zero(g, 15339).unwrap();
        let logits = compute_logits_from_graph(&hidden, g).unwrap();
        assert_eq!(logits.len(), g.config.vocab_size as usize);
        assert_eq!(logits.len(), 128256);
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "non-finite logit at vocab id {}: {}", i, v);
        }
    });
}

#[test]
fn logits_are_not_all_zero() {
    with_real_graph(|g| {
        let hidden = forward_single_token_position_zero(g, 15339).unwrap();
        let logits = compute_logits_from_graph(&hidden, g).unwrap();
        let nz = logits.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nz > logits.len() / 2,
            "expected most logits non-zero; got {} / {}",
            nz,
            logits.len()
        );
    });
}

#[test]
fn logits_are_deterministic() {
    with_real_graph(|g| {
        let hidden = forward_single_token_position_zero(g, 15339).unwrap();
        let a = compute_logits_from_graph(&hidden, g).unwrap();
        let b = compute_logits_from_graph(&hidden, g).unwrap();
        assert_eq!(a, b, "logits must be deterministic for the same hidden");
    });
}

#[test]
fn argmax_returns_in_range_token_id() {
    with_real_graph(|g| {
        let hidden = forward_single_token_position_zero(g, 15339).unwrap();
        let logits = compute_logits_from_graph(&hidden, g).unwrap();
        let id = argmax(&logits).expect("argmax of non-empty logits");
        assert!(
            id < g.config.vocab_size,
            "argmax id {} >= vocab_size {}",
            id,
            g.config.vocab_size
        );
    });
}

#[test]
fn top_k_returns_in_range_token_ids() {
    with_real_graph(|g| {
        let hidden = forward_single_token_position_zero(g, 15339).unwrap();
        let logits = compute_logits_from_graph(&hidden, g).unwrap();
        let top = top_k(&logits, 5);
        assert_eq!(top.len(), 5);
        for (id, _) in &top {
            assert!(*id < g.config.vocab_size);
        }
        // Top-1 should equal argmax.
        assert_eq!(top[0].0, argmax(&logits).unwrap());
        // Logits must be in non-increasing order.
        for w in top.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    });
}
