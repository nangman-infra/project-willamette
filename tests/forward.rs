//! Stage 4-D4 integration tests — 30-layer single-token forward on real GGUF.
//!
//! These tests exercise the full transformer stack and therefore take
//! noticeably longer than earlier stages — each call runs 30 layers × 7
//! BitLinear matvecs (the biggest of which is 6912 × 2560 ternary
//! elements). Expect a few seconds per `forward_single_token_position_zero`
//! call on commodity hardware.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::forward::forward_single_token_position_zero;
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 4-D4 tests require it",
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
fn full_forward_returns_n_embd_finite_hidden() {
    with_real_graph(|g| {
        let h = forward_single_token_position_zero(g, 15339).expect("forward");
        assert_eq!(h.len(), 2560);
        assert_eq!(h.len(), g.config.embedding_length as usize);
        let nz = h.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nz > h.len() / 2,
            "more than half of final hidden should be non-zero, got {}",
            nz
        );
        for (i, &v) in h.iter().enumerate() {
            assert!(v.is_finite(), "non-finite hidden at dim {}: {}", i, v);
        }
    });
}

#[test]
fn full_forward_is_deterministic() {
    with_real_graph(|g| {
        let a = forward_single_token_position_zero(g, 15339).expect("forward a");
        let b = forward_single_token_position_zero(g, 15339).expect("forward b");
        assert_eq!(a, b, "full forward must be deterministic");
    });
}

#[test]
fn full_forward_different_tokens_differ() {
    with_real_graph(|g| {
        let h_a = forward_single_token_position_zero(g, 15339).expect("forward 15339");
        let h_b = forward_single_token_position_zero(g, 101193).expect("forward 101193");
        let mut differ = 0usize;
        for i in 0..h_a.len() {
            if (h_a[i] - h_b[i]).abs() > 1e-6 {
                differ += 1;
            }
        }
        assert!(
            differ > h_a.len() / 2,
            "different tokens should produce different final hiddens; only {} differ",
            differ
        );
    });
}
