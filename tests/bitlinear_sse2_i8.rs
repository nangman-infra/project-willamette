//! Stage 6-B follow-up — scalar vs SSE2-i8 BitLinear equivalence.
//!
//! Unlike the f32 SSE2 kernel (`tests/bitlinear_sse2.rs`, tolerance
//! 1e-2), the i8-activation kernel quantises the activation to int8
//! before the dot product — the same lossy step the production
//! bitnet.cpp CPU path takes. So the tolerance here is **looser** and
//! is expressed relative to the output magnitude, not as a tiny
//! absolute bound. The point of this test is not bit-equivalence; it's
//! to confirm the kernel is *correct* (no garbage / NaN / transposed
//! indexing) and tracks the scalar reference closely enough that
//! greedy decoding would pick the same tokens.
//!
//! On non-x86 hosts this file compiles to no tests.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::gguf::tensor::TensorView;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::bitlinear::bitlinear_i2s_matvec_f32_scalar;
use project_willamette::model::bitlinear_sse2::bitlinear_i2s_matvec_f32_sse2_i8;
use project_willamette::model::primitives::{
    embedding_gather_f16, f32_tensor_to_vec, rms_norm_f32,
};
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

/// Relative tolerance: int8 activation quantisation introduces
/// ~`1/127` relative error per element, partly averaging out across
/// the reduction. We assert the per-output relative error stays under
/// 5 % and the cosine similarity stays very high — that's the bar for
/// "same argmax in practice".
const REL_TOL: f32 = 0.05;

fn with_real_graph<F: FnOnce(&ModelGraph<'_>)>(f: F) {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: real GGUF not found at {}", MODEL_PATH);
        return;
    }
    if !std::arch::is_x86_feature_detected!("sse2") {
        eprintln!("SKIP: host has no SSE2");
        return;
    }
    let mmap = ModelMmap::open(MODEL_PATH).expect("open model");
    let gguf = GgufFile::parse(mmap.as_bytes()).expect("parse model");
    let graph = ModelGraph::from_gguf(&gguf).expect("build graph");
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
        bitlinear_i2s_matvec_f32_sse2_i8(weight, input, &mut b).expect("sse2-i8");
    }

    // Cosine similarity + max relative error against the scalar ref.
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    let mut max_rel = 0.0_f32;
    let mut ref_absmax = 0.0_f32;
    for i in 0..out_dim {
        assert!(b[i].is_finite(), "{} sse2-i8 non-finite at {}", label, i);
        dot += (a[i] * b[i]) as f64;
        na += (a[i] * a[i]) as f64;
        nb += (b[i] * b[i]) as f64;
        ref_absmax = ref_absmax.max(a[i].abs());
    }
    // Relative error normalised by the reference's magnitude scale,
    // so a few near-zero outputs don't blow up the metric.
    let denom = ref_absmax.max(1e-6);
    for i in 0..out_dim {
        let r = (a[i] - b[i]).abs() / denom;
        if r > max_rel {
            max_rel = r;
        }
    }
    let cosine = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    eprintln!(
        "[{}] cosine={:.6}  max_rel(vs absmax)={:.4}  out_dim={}",
        label, cosine, max_rel, out_dim
    );
    assert!(
        cosine > 0.999,
        "{}: cosine {} too low — kernel likely wrong, not just quantisation",
        label,
        cosine
    );
    assert!(
        max_rel <= REL_TOL,
        "{}: max relative error {} exceeds {}",
        label,
        max_rel,
        REL_TOL
    );
}

#[test]
fn scalar_vs_sse2_i8_attn_q() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.embedding_length as usize;
        compare(g.layers[0].attn_q, &input, out_dim, "attn_q");
    });
}

#[test]
fn scalar_vs_sse2_i8_ffn_gate() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.feed_forward_length as usize;
        compare(g.layers[0].ffn_gate, &input, out_dim, "ffn_gate");
    });
}

#[test]
fn scalar_vs_sse2_i8_ffn_down() {
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
fn sse2_i8_zero_input_is_zero() {
    with_real_graph(|g| {
        let input = vec![0.0_f32; g.config.embedding_length as usize];
        let out_dim = g.config.embedding_length as usize;
        let mut b = vec![1.0_f32; out_dim];
        unsafe {
            bitlinear_i2s_matvec_f32_sse2_i8(g.layers[0].attn_q, &input, &mut b).unwrap();
        }
        for (i, v) in b.iter().enumerate() {
            assert_eq!(*v, 0.0, "sse2-i8 zero-input nonzero at {}", i);
        }
    });
}
