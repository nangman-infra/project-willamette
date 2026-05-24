//! Stage 4-C integration tests — scalar I2_S matvec over the real GGUF.
//!
//! These tests run the [`bitlinear_i2s_matvec_f32`] reference against
//! every BitLinear role in layer 0 of `ggml-model-i2_s.gguf`, fed with
//! actual embedding + RMSNorm output. They check:
//!
//!   * shape / dtype / scale preconditions are satisfied by the real file,
//!   * the matvec produces a finite, non-zero, deterministic output,
//!   * the scale block is within file bounds (re-confirms Stage 3),
//!   * the per-tensor scale itself is a finite f32.
//!
//! They do NOT assert "correct logits" — Stage 4-D's job is to put the
//! pieces together end-to-end.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::gguf::tensor::TensorView;
use project_willamette::gguf::types::GgmlType;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::bitlinear::bitlinear_i2s_matvec_f32;
use project_willamette::model::primitives::{embedding_gather_f16, rms_norm_f32};
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 4-C tests require it",
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

/// Read an F32 tensor's raw bytes into an owned `Vec<f32>` (no Stage 4-C
/// helper exposed for this since reading F32 weights is a one-liner).
fn f32_tensor_to_vec(t: &TensorView<'_>) -> Vec<f32> {
    assert_eq!(t.ggml_type, GgmlType::F32);
    let n = t.n_elements() as usize;
    assert_eq!(t.data.len(), n * 4);
    (0..n)
        .map(|i| {
            let b = &t.data[4 * i..4 * i + 4];
            f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        })
        .collect()
}

/// Produce a realistic f32 input vector for the BitLinear layer being
/// tested. Mimics the start of the forward path: embed token 15339, then
/// apply layer-0 `attn_norm` so the input distribution looks like what
/// the real model would receive.
fn realistic_input(graph: &ModelGraph<'_>) -> Vec<f32> {
    let n_embd = graph.config.embedding_length as usize;
    let mut x = vec![0.0_f32; n_embd];
    embedding_gather_f16(graph.token_embd, 15339, &mut x).expect("embed token 15339");
    let w = f32_tensor_to_vec(graph.layers[0].attn_norm);
    let mut normed = vec![0.0_f32; n_embd];
    rms_norm_f32(&x, &w, graph.config.layer_norm_rms_epsilon, &mut normed).expect("rms norm");
    normed
}

// ── role-by-role shape checks ─────────────────────────────────────────

#[test]
fn attn_q_shape_matches_n_embd_to_n_embd() {
    with_real_graph(|g| {
        let w = g.layers[0].attn_q;
        assert_eq!(w.ggml_type, GgmlType::BitNetI2S);
        assert_eq!(
            w.shape,
            vec![
                g.config.embedding_length as u64,
                g.config.embedding_length as u64
            ]
        );
        assert_eq!(w.shape[0], 2560);
        assert_eq!(w.shape[1], 2560);
    });
}

#[test]
fn attn_k_and_attn_v_share_gqa_kv_dim() {
    with_real_graph(|g| {
        for w in [g.layers[0].attn_k, g.layers[0].attn_v] {
            assert_eq!(w.ggml_type, GgmlType::BitNetI2S);
            assert_eq!(
                w.shape,
                vec![g.config.embedding_length as u64, g.config.kv_dim as u64]
            );
            assert_eq!(w.shape[0], 2560);
            assert_eq!(w.shape[1], 640);
        }
    });
}

#[test]
fn attn_output_collapses_back_to_n_embd() {
    with_real_graph(|g| {
        let w = g.layers[0].attn_output;
        assert_eq!(
            w.shape,
            vec![
                g.config.embedding_length as u64,
                g.config.embedding_length as u64
            ]
        );
    });
}

#[test]
fn ffn_gate_up_down_match_n_embd_n_ff_pattern() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as u64;
        let n_ff = g.config.feed_forward_length as u64;
        assert_eq!(g.layers[0].ffn_gate.shape, vec![n_embd, n_ff]);
        assert_eq!(g.layers[0].ffn_up.shape, vec![n_embd, n_ff]);
        assert_eq!(g.layers[0].ffn_down.shape, vec![n_ff, n_embd]);
    });
}

// ── scale block sanity ────────────────────────────────────────────────

#[test]
fn every_i2s_tensor_has_finite_scale() {
    with_real_graph(|g| {
        for layer in &g.layers {
            for (label, t) in [
                ("attn_q", layer.attn_q),
                ("attn_k", layer.attn_k),
                ("attn_v", layer.attn_v),
                ("attn_output", layer.attn_output),
                ("ffn_gate", layer.ffn_gate),
                ("ffn_up", layer.ffn_up),
                ("ffn_down", layer.ffn_down),
            ] {
                let scale = t.i2s_scale().unwrap_or_else(|e| {
                    panic!("layer {} {}: scale read failed: {}", layer.index, label, e)
                });
                assert!(
                    scale.is_finite(),
                    "layer {} {}: non-finite scale {}",
                    layer.index,
                    label,
                    scale
                );
                // Trained scales are positive (i2_scale = max(|W|)).
                assert!(
                    scale > 0.0,
                    "layer {} {}: non-positive scale {}",
                    layer.index,
                    label,
                    scale
                );
            }
        }
    });
}

#[test]
fn scale_data_is_present_for_every_i2s_tensor() {
    with_real_graph(|g| {
        let mut count = 0;
        for t in g.layers.iter().flat_map(|l| {
            [
                l.attn_q,
                l.attn_k,
                l.attn_v,
                l.attn_output,
                l.ffn_gate,
                l.ffn_up,
                l.ffn_down,
            ]
        }) {
            assert!(
                t.scale_data.is_some(),
                "I2_S tensor {:?} missing scale_data",
                t.name
            );
            assert_eq!(t.scale_data.unwrap().len(), 32);
            count += 1;
        }
        assert_eq!(count, 210);
    });
}

// ── matvec produces finite, non-zero, deterministic output ────────────

#[test]
fn matvec_attn_q_produces_finite_nonzero_output() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.embedding_length as usize;
        let mut out = vec![0.0_f32; out_dim];
        bitlinear_i2s_matvec_f32(g.layers[0].attn_q, &input, &mut out).unwrap();
        assert_eq!(out.len(), out_dim);
        let nz = out.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nz > out_dim / 2,
            "more than half of attn_q output should be non-zero, got {}",
            nz
        );
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "attn_q output[{}] = {} is not finite", i, v);
        }
    });
}

#[test]
fn matvec_attn_k_v_produce_kv_dim_output() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let kv_dim = g.config.kv_dim as usize;
        let mut out_k = vec![0.0_f32; kv_dim];
        let mut out_v = vec![0.0_f32; kv_dim];
        bitlinear_i2s_matvec_f32(g.layers[0].attn_k, &input, &mut out_k).unwrap();
        bitlinear_i2s_matvec_f32(g.layers[0].attn_v, &input, &mut out_v).unwrap();
        assert_eq!(out_k.len(), 640);
        assert_eq!(out_v.len(), 640);
        for v in out_k.iter().chain(out_v.iter()) {
            assert!(v.is_finite());
        }
    });
}

#[test]
fn matvec_ffn_gate_up_produce_n_ff_output() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let n_ff = g.config.feed_forward_length as usize;
        let mut gate = vec![0.0_f32; n_ff];
        let mut up = vec![0.0_f32; n_ff];
        bitlinear_i2s_matvec_f32(g.layers[0].ffn_gate, &input, &mut gate).unwrap();
        bitlinear_i2s_matvec_f32(g.layers[0].ffn_up, &input, &mut up).unwrap();
        assert_eq!(gate.len(), 6912);
        assert_eq!(up.len(), 6912);
        for v in gate.iter().chain(up.iter()) {
            assert!(v.is_finite());
        }
        // Gate and up are different projections, so their outputs should differ.
        let mut differ = 0usize;
        for i in 0..n_ff {
            if (gate[i] - up[i]).abs() > 1e-6 {
                differ += 1;
            }
        }
        assert!(
            differ > n_ff / 2,
            "gate and up outputs should mostly differ; got {} differ",
            differ
        );
    });
}

#[test]
fn matvec_ffn_down_consumes_n_ff_input() {
    with_real_graph(|g| {
        // ffn_down has in_dim = n_ff, not n_embd. Feed a synthetic n_ff
        // input — Stage 4-C does not yet wire the full FFN composition.
        let n_ff = g.config.feed_forward_length as usize;
        let n_embd = g.config.embedding_length as usize;
        let input: Vec<f32> = (0..n_ff)
            .map(|i| ((i as f32) / (n_ff as f32)) * 0.5)
            .collect();
        let mut out = vec![0.0_f32; n_embd];
        bitlinear_i2s_matvec_f32(g.layers[0].ffn_down, &input, &mut out).unwrap();
        for &v in out.iter() {
            assert!(v.is_finite());
        }
    });
}

#[test]
fn matvec_is_deterministic_across_repeated_runs() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.embedding_length as usize;
        let mut a = vec![0.0_f32; out_dim];
        let mut b = vec![0.0_f32; out_dim];
        bitlinear_i2s_matvec_f32(g.layers[0].attn_q, &input, &mut a).unwrap();
        bitlinear_i2s_matvec_f32(g.layers[0].attn_q, &input, &mut b).unwrap();
        assert_eq!(a, b, "matvec must be deterministic across runs");
    });
}

#[test]
fn matvec_different_layers_produce_different_outputs() {
    with_real_graph(|g| {
        let input = realistic_input(g);
        let out_dim = g.config.embedding_length as usize;
        let mut a = vec![0.0_f32; out_dim];
        let mut b = vec![0.0_f32; out_dim];
        bitlinear_i2s_matvec_f32(g.layers[0].attn_q, &input, &mut a).unwrap();
        bitlinear_i2s_matvec_f32(g.layers[15].attn_q, &input, &mut b).unwrap();
        // Different layer weights → different outputs (with overwhelming probability)
        let mut differ = 0usize;
        for i in 0..out_dim {
            if (a[i] - b[i]).abs() > 1e-6 {
                differ += 1;
            }
        }
        assert!(
            differ > out_dim / 2,
            "different layers should produce mostly-different attn_q outputs"
        );
    });
}

#[test]
fn matvec_with_zero_input_is_zero_output() {
    with_real_graph(|g| {
        // For ANY weight, W · 0 = 0, and scale × 0 = 0. Confirms the
        // reduction adds nothing spurious.
        let n_embd = g.config.embedding_length as usize;
        let input = vec![0.0_f32; n_embd];
        let mut out = vec![1.0_f32; n_embd]; // sentinel
        bitlinear_i2s_matvec_f32(g.layers[0].attn_q, &input, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "zero input must yield zero output (dim {})", i);
        }
    });
}
