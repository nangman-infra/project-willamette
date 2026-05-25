//! Stage 6-B — Scalar vs SSE2 BitLinear matvec equivalence.
//!
//! Mirror of `tests/bitlinear_simd.rs` (which targets aarch64 NEON).
//! On any x86 / x86_64 host that reports SSE2 (universally true since
//! Pentium 4 / 2003 for x86_64; antiX Pentium-M for i686), this test
//! runs both the scalar reference and the SSE2 kernel on every
//! BitLinear weight in layer 0 of the real GGUF and verifies that
//! per-element absolute error stays within the same tolerance the
//! NEON tests already use.
//!
//! On non-x86 hosts this file compiles to no tests.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::gguf::tensor::TensorView;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::bitlinear::bitlinear_i2s_matvec_f32_scalar;
use project_willamette::model::bitlinear_sse2::bitlinear_i2s_matvec_f32_sse2;
use project_willamette::model::primitives::{
    embedding_gather_f16, f32_tensor_to_vec, rms_norm_f32,
};
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

/// Same tolerance as the NEON test (`tests/bitlinear_simd.rs`).
const TOL: f32 = 1e-2;

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 6-B SSE2 tests require it",
            MODEL_PATH
        );
        return;
    }
    if !std::arch::is_x86_feature_detected!("sse2") {
        eprintln!("SKIP: host does not advertise SSE2");
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
        bitlinear_i2s_matvec_f32_sse2(weight, input, &mut b).expect("sse2");
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
            "{} sse2   non-finite at {}: {}",
            label,
            i,
            b[i]
        );
        let d = (a[i] - b[i]).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        let denom = a[i].abs().max(1e-12);
        let r = d / denom;
        if r > max_rel_diff {
            max_rel_diff = r;
        }
        mean_abs_diff += d as f64;
    }
    mean_abs_diff /= out_dim as f64;
    eprintln!(
        "[{}] max|Δ|={:.3e}  mean|Δ|={:.3e}  max rel={:.3e}  out_dim={}",
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
fn scalar_vs_sse2_attn_q() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.embedding_length as usize;
        compare(g.layers[0].attn_q, &input, out_dim, "attn_q");
    });
}

#[test]
fn scalar_vs_sse2_attn_k() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.kv_dim as usize;
        compare(g.layers[0].attn_k, &input, out_dim, "attn_k");
    });
}

#[test]
fn scalar_vs_sse2_attn_v() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.kv_dim as usize;
        compare(g.layers[0].attn_v, &input, out_dim, "attn_v");
    });
}

#[test]
fn scalar_vs_sse2_attn_output() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.embedding_length as usize;
        compare(g.layers[0].attn_output, &input, out_dim, "attn_output");
    });
}

#[test]
fn scalar_vs_sse2_ffn_gate() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.feed_forward_length as usize;
        compare(g.layers[0].ffn_gate, &input, out_dim, "ffn_gate");
    });
}

#[test]
fn scalar_vs_sse2_ffn_up() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.feed_forward_length as usize;
        compare(g.layers[0].ffn_up, &input, out_dim, "ffn_up");
    });
}

#[test]
fn scalar_vs_sse2_ffn_down() {
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
fn sse2_zero_input_matches_scalar_zero_output() {
    with_real_graph(|g| {
        let input = vec![0.0_f32; g.config.embedding_length as usize];
        let out_dim = g.config.embedding_length as usize;
        let mut a = vec![1.0_f32; out_dim];
        let mut b = vec![1.0_f32; out_dim];
        bitlinear_i2s_matvec_f32_scalar(g.layers[0].attn_q, &input, &mut a).unwrap();
        unsafe {
            bitlinear_i2s_matvec_f32_sse2(g.layers[0].attn_q, &input, &mut b).unwrap();
        }
        for i in 0..out_dim {
            assert_eq!(a[i], 0.0, "scalar zero-input produced non-zero at {}", i);
            assert_eq!(b[i], 0.0, "sse2 zero-input produced non-zero at {}", i);
        }
    });
}
