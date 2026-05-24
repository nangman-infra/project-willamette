//! Stage 4-A — ModelGraph + BitNetConfig integration tests.
//!
//! Verifies, against the real `ggml-model-i2_s.gguf`, that the model
//! topology matches the source-pinned plan in `docs/BITNET_FORWARD_PLAN.md`.
//! Each test skips with a SKIP message if the model file is missing.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::gguf::types::GgmlType;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::{BitNetConfig, ModelGraph};

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 4-A tests require it",
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
fn config_matches_inspect_log_values() {
    with_real_graph(|g| {
        let c = &g.config;
        assert_eq!(c.architecture, BitNetConfig::ARCHITECTURE);
        assert_eq!(c.architecture, "bitnet-b1.58");
        assert_eq!(c.block_count, 30);
        assert_eq!(c.embedding_length, 2560);
        assert_eq!(c.feed_forward_length, 6912);
        assert_eq!(c.context_length, 4096);
        assert_eq!(c.head_count, 20);
        assert_eq!(c.head_count_kv, 5);
        assert_eq!(c.head_dim, 128); // = 2560 / 20
        assert_eq!(c.kv_dim, 640); // = 128 * 5
        assert!(
            (c.layer_norm_rms_epsilon - 1e-5).abs() < 1e-9,
            "rms_eps = {}",
            c.layer_norm_rms_epsilon
        );
        assert_eq!(c.rope_dimension_count, 128);
        assert!(
            (c.rope_freq_base - 500_000.0).abs() < 1.0,
            "rope_freq_base = {}",
            c.rope_freq_base
        );
        assert_eq!(c.vocab_size, 128256);
    });
}

#[test]
fn graph_has_exactly_30_layers() {
    with_real_graph(|g| {
        assert_eq!(g.layers.len(), 30);
        // Layer indices are 0..30, in order.
        for (i, l) in g.layers.iter().enumerate() {
            assert_eq!(
                l.index as usize, i,
                "layer index mismatch at position {}",
                i
            );
        }
    });
}

#[test]
fn token_embd_is_f16_with_expected_shape() {
    with_real_graph(|g| {
        assert_eq!(g.token_embd.ggml_type, GgmlType::F16);
        assert_eq!(
            g.token_embd.shape,
            vec![g.config.embedding_length as u64, g.config.vocab_size as u64]
        );
        assert_eq!(g.token_embd.name, "token_embd.weight");
    });
}

#[test]
fn output_norm_is_f32_with_expected_shape() {
    with_real_graph(|g| {
        assert_eq!(g.output_norm.ggml_type, GgmlType::F32);
        assert_eq!(g.output_norm.shape, vec![g.config.embedding_length as u64]);
        assert_eq!(g.output_norm.name, "output_norm.weight");
    });
}

#[test]
fn output_weight_is_absent_for_this_file() {
    // Our file (ggml-model-i2_s.gguf) does NOT ship a separate output.weight
    // tensor. The flag should reflect that.
    with_real_graph(|g| {
        assert!(
            !g.has_output_weight_tensor,
            "expected no separate output.weight tensor in this GGUF; \
             if a future revision ships one, update this test together with \
             REFERENCE_COMMIT.md"
        );
    });
}

#[test]
fn lm_head_is_tied_to_token_embd() {
    with_real_graph(|g| {
        assert!(
            g.lm_head_is_tied(),
            "lm_head must be the same tensor as token_embd for BitNet b1.58 \
             (source: build_bitnet_158 in src/llama.cpp:15527 of pinned commit)"
        );
        // Pointer-identity check via name as a defensive cross-check.
        assert_eq!(g.lm_head.name, g.token_embd.name);
    });
}

#[test]
fn every_layer_has_4_f32_and_7_i2s_tensors() {
    with_real_graph(|g| {
        for layer in &g.layers {
            // F32 norm tensors (4)
            for (label, t) in [
                ("attn_norm", layer.attn_norm),
                ("attn_sub_norm", layer.attn_sub_norm),
                ("ffn_norm", layer.ffn_norm),
                ("ffn_sub_norm", layer.ffn_sub_norm),
            ] {
                assert_eq!(
                    t.ggml_type,
                    GgmlType::F32,
                    "layer {}: {} expected F32, got {:?}",
                    layer.index,
                    label,
                    t.ggml_type
                );
            }

            // I2_S BitLinear tensors (7)
            for (label, t) in [
                ("attn_q", layer.attn_q),
                ("attn_k", layer.attn_k),
                ("attn_v", layer.attn_v),
                ("attn_output", layer.attn_output),
                ("ffn_gate", layer.ffn_gate),
                ("ffn_up", layer.ffn_up),
                ("ffn_down", layer.ffn_down),
            ] {
                assert_eq!(
                    t.ggml_type,
                    GgmlType::BitNetI2S,
                    "layer {}: {} expected I2_S, got {:?}",
                    layer.index,
                    label,
                    t.ggml_type
                );
            }
        }
    });
}

#[test]
fn every_layer_has_expected_shapes() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as u64;
        let n_ff = g.config.feed_forward_length as u64;
        let kv_dim = g.config.kv_dim as u64;

        for layer in &g.layers {
            assert_eq!(layer.attn_norm.shape, vec![n_embd]);
            assert_eq!(layer.attn_sub_norm.shape, vec![n_embd]);
            assert_eq!(layer.ffn_norm.shape, vec![n_embd]);
            assert_eq!(layer.ffn_sub_norm.shape, vec![n_ff]);

            assert_eq!(layer.attn_q.shape, vec![n_embd, n_embd]);
            assert_eq!(layer.attn_k.shape, vec![n_embd, kv_dim]);
            assert_eq!(layer.attn_v.shape, vec![n_embd, kv_dim]);
            assert_eq!(layer.attn_output.shape, vec![n_embd, n_embd]);

            assert_eq!(layer.ffn_gate.shape, vec![n_embd, n_ff]);
            assert_eq!(layer.ffn_up.shape, vec![n_embd, n_ff]);
            assert_eq!(layer.ffn_down.shape, vec![n_ff, n_embd]);
        }
    });
}

#[test]
fn graph_references_a_distinct_tensor_per_role_per_layer() {
    // Sanity: no two roles within the same layer point at the same TensorView,
    // and no two layers share a role.
    with_real_graph(|g| {
        let mut all_addrs: Vec<(usize, &str)> = Vec::new();
        for layer in &g.layers {
            for (role, t) in [
                ("attn_norm", layer.attn_norm),
                ("attn_q", layer.attn_q),
                ("attn_k", layer.attn_k),
                ("attn_v", layer.attn_v),
                ("attn_output", layer.attn_output),
                ("attn_sub_norm", layer.attn_sub_norm),
                ("ffn_norm", layer.ffn_norm),
                ("ffn_gate", layer.ffn_gate),
                ("ffn_up", layer.ffn_up),
                ("ffn_down", layer.ffn_down),
                ("ffn_sub_norm", layer.ffn_sub_norm),
            ] {
                let addr = t as *const _ as usize;
                all_addrs.push((addr, role));
            }
        }
        // Add top-level tensors.
        all_addrs.push((g.token_embd as *const _ as usize, "token_embd"));
        all_addrs.push((g.output_norm as *const _ as usize, "output_norm"));
        // lm_head is intentionally aliased to token_embd, so don't include it.

        let total = all_addrs.len();
        let mut sorted = all_addrs.clone();
        sorted.sort_by_key(|(a, _)| *a);
        sorted.dedup_by_key(|(a, _)| *a);
        assert_eq!(
            sorted.len(),
            total,
            "expected {} distinct tensor refs across the graph, got {} after dedup",
            total,
            sorted.len()
        );
    });
}

#[test]
fn graph_covers_exactly_332_tensor_references() {
    // 30 layers × 11 per-layer roles + token_embd + output_norm = 332
    // (lm_head is an alias, not a new tensor)
    with_real_graph(|g| {
        let per_layer = 11usize;
        let total = g.layers.len() * per_layer + 2; // + token_embd + output_norm
        assert_eq!(total, 332);
    });
}
