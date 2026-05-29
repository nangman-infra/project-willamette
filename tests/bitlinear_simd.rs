//! Stage 6-C — Scalar vs NEON BitLinear matvec equivalence.
//!
//! On aarch64 hosts, this test runs both the scalar reference and the
//! NEON kernel on every BitLinear weight in layer 0 of the real GGUF
//! and verifies that per-element absolute error stays within a
//! documented tolerance. The tolerance bound is derived from
//! `O(in_dim · ε · max|input|)` where `ε ≈ 1e-7` is the f32 unit
//! roundoff; for `in_dim = 6912` (the largest BitLinear input
//! dimension in this model, used by `ffn_down`) and typical
//! pre-quant-norm inputs we observe |Δ| < ~5e-3 with a 1.0 scale.
//!
//! On non-aarch64 hosts this entire file compiles to no tests.

#![cfg(target_arch = "aarch64")]

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::gguf::tensor::TensorView;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::bitlinear::bitlinear_i2s_matvec_f32_scalar;
use project_willamette::model::bitlinear_neon::bitlinear_i2s_matvec_f32_neon;
use project_willamette::model::primitives::{
    embedding_gather_f16, f32_tensor_to_vec, rms_norm_f32,
};
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

/// Observed tolerance per output dim, with a comfortable safety margin
/// above what we actually see on Apple M-series (current host: M4) (typical < 1e-3 absolute).
const TOL: f32 = 1e-2;

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 6-C SIMD tests require it",
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

fn realistic_input(graph: &ModelGraph<'_>) -> Vec<f32> {
    let n_embd = graph.config.embedding_length as usize;
    let mut x = vec![0.0_f32; n_embd];
    embedding_gather_f16(graph.token_embd, 15339, &mut x).unwrap();
    let w = f32_tensor_to_vec(graph.layers[0].attn_norm).unwrap();
    let mut normed = vec![0.0_f32; n_embd];
    rms_norm_f32(&x, &w, graph.config.layer_norm_rms_epsilon, &mut normed).unwrap();
    normed
}

fn compare(weight: &TensorView<'_>, input: &[f32], out_dim: usize, label: &str) {
    let mut a = vec![0.0_f32; out_dim];
    let mut b = vec![0.0_f32; out_dim];
    bitlinear_i2s_matvec_f32_scalar(weight, input, &mut a).expect("scalar");
    unsafe {
        bitlinear_i2s_matvec_f32_neon(weight, input, &mut b).expect("neon");
    }

    let mut max_abs_diff = 0.0_f32;
    let mut max_rel_diff = 0.0_f32;
    let mut mean_abs_diff = 0.0_f64;
    for i in 0..out_dim {
        assert!(
            a[i].is_finite(),
            "{} scalar non-finite at {}: {}",
            label,
            i,
            a[i]
        );
        assert!(
            b[i].is_finite(),
            "{} neon non-finite at {}: {}",
            label,
            i,
            b[i]
        );
        let d = (a[i] - b[i]).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        let denom = a[i].abs().max(1e-6);
        let r = d / denom;
        if r > max_rel_diff {
            max_rel_diff = r;
        }
        mean_abs_diff += d as f64;
    }
    mean_abs_diff /= out_dim as f64;

    eprintln!(
        "[{}] max|Δ| = {:.3e}  mean|Δ| = {:.3e}  max rel = {:.3e}  (out_dim = {})",
        label, max_abs_diff, mean_abs_diff, max_rel_diff, out_dim
    );

    assert!(
        max_abs_diff <= TOL,
        "{}: max|Δ| = {} exceeds tolerance {}",
        label,
        max_abs_diff,
        TOL
    );
}

#[test]
fn scalar_vs_neon_attn_q() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.embedding_length as usize;
        compare(g.layers[0].attn_q, &input, out_dim, "attn_q");
    });
}

#[test]
fn scalar_vs_neon_attn_k() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.kv_dim as usize;
        compare(g.layers[0].attn_k, &input, out_dim, "attn_k");
    });
}

#[test]
fn scalar_vs_neon_attn_v() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.kv_dim as usize;
        compare(g.layers[0].attn_v, &input, out_dim, "attn_v");
    });
}

#[test]
fn scalar_vs_neon_attn_output() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.embedding_length as usize;
        compare(g.layers[0].attn_output, &input, out_dim, "attn_output");
    });
}

#[test]
fn scalar_vs_neon_ffn_gate() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.feed_forward_length as usize;
        compare(g.layers[0].ffn_gate, &input, out_dim, "ffn_gate");
    });
}

#[test]
fn scalar_vs_neon_ffn_up() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.feed_forward_length as usize;
        compare(g.layers[0].ffn_up, &input, out_dim, "ffn_up");
    });
}

#[test]
fn scalar_vs_neon_ffn_down() {
    // ffn_down takes n_ff input, so we need to fabricate one of that size.
    with_real_graph(|g| {
        let n_ff = g.config.feed_forward_length as usize;
        let n_embd = g.config.embedding_length as usize;
        let input: Vec<f32> = (0..n_ff)
            .map(|i| ((i as f32) / (n_ff as f32)) * 0.5)
            .collect();
        compare(g.layers[0].ffn_down, &input, n_embd, "ffn_down");
    });
}

#[test]
fn neon_zero_input_matches_scalar_zero_output() {
    with_real_graph(|g| {
        let input = vec![0.0_f32; g.config.embedding_length as usize];
        let out_dim = g.config.embedding_length as usize;
        let mut a = vec![1.0_f32; out_dim];
        let mut b = vec![1.0_f32; out_dim];
        bitlinear_i2s_matvec_f32_scalar(g.layers[0].attn_q, &input, &mut a).unwrap();
        unsafe {
            bitlinear_i2s_matvec_f32_neon(g.layers[0].attn_q, &input, &mut b).unwrap();
        }
        for i in 0..out_dim {
            assert_eq!(a[i], 0.0);
            assert_eq!(b[i], 0.0);
        }
    });
}
