//! LUT BitLinear parity test — see `docs/LUT_KERNEL_RFC.md`.
//!
//! Numerical-equivalence test, scalar LUT vs scalar reference BitLinear,
//! on every layer-0 weight of the real GGUF. Tolerance matches the
//! NEON / SSE2 i8 tests' `max|Δ| ≤ 1e-2`. f32 reassociation may shift
//! per-element error by a few ULP; bit-equality is not claimed.
//!
//! Compiled on x86 only — the LUT kernel lives in `bitlinear_lut.rs`
//! which is itself x86-gated.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::gguf::tensor::TensorView;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::bitlinear::bitlinear_i2s_matvec_f32_scalar;
use project_willamette::model::bitlinear_lut::bitlinear_i2s_matvec_f32_lut_scalar;
use project_willamette::model::primitives::{
    embedding_gather_f16, f32_tensor_to_vec, rms_norm_f32,
};
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";
const TOL: f32 = 1e-2;

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — LUT step-1 tests need it",
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
    bitlinear_i2s_matvec_f32_lut_scalar(weight, input, &mut b).expect("lut");

    let mut max_abs = 0.0_f32;
    for i in 0..out_dim {
        assert!(
            a[i].is_finite() && b[i].is_finite(),
            "{label} non-finite at {i}"
        );
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    eprintln!("[{}] max|Δ|={:.3e}  out_dim={}", label, max_abs, out_dim);
    assert!(max_abs <= TOL, "{label}: max|Δ| = {max_abs} > {TOL}");
}

#[test]
fn scalar_vs_lut_attn_q() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.embedding_length as usize;
        compare(g.layers[0].attn_q, &input, out_dim, "attn_q");
    });
}

#[test]
fn scalar_vs_lut_attn_k() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.kv_dim as usize;
        compare(g.layers[0].attn_k, &input, out_dim, "attn_k");
    });
}

#[test]
fn scalar_vs_lut_attn_v() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.kv_dim as usize;
        compare(g.layers[0].attn_v, &input, out_dim, "attn_v");
    });
}

#[test]
fn scalar_vs_lut_ffn_gate() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.feed_forward_length as usize;
        compare(g.layers[0].ffn_gate, &input, out_dim, "ffn_gate");
    });
}
