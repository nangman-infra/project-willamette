//! Stage 4-D3 integration tests — full transformer block forward on real GGUF.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::attention::attention_block_forward_position_zero;
use project_willamette::model::block::transformer_block_forward_position_zero;
use project_willamette::model::ffn::ffn_block_forward;
use project_willamette::model::primitives::{embedding_gather_f16, f32_tensor_to_vec};
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 4-D3 tests require it",
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
fn block_forward_preserves_n_embd_and_is_finite() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();
        let mut out = vec![0.0_f32; n_embd];
        transformer_block_forward_position_zero(&x, &g.layers[0], &g.config, &mut out).unwrap();
        assert_eq!(out.len(), n_embd);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "non-finite block output at dim {}: {}", i, v);
        }
    });
}

#[test]
fn block_forward_is_deterministic() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();
        let mut a = vec![0.0_f32; n_embd];
        let mut b = vec![0.0_f32; n_embd];
        transformer_block_forward_position_zero(&x, &g.layers[0], &g.config, &mut a).unwrap();
        transformer_block_forward_position_zero(&x, &g.layers[0], &g.config, &mut b).unwrap();
        assert_eq!(a, b);
    });
}

#[test]
fn full_block_differs_from_attention_only() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();

        let attn_w = f32_tensor_to_vec(g.layers[0].attn_norm).unwrap();
        let asn_w = f32_tensor_to_vec(g.layers[0].attn_sub_norm).unwrap();
        let mut attn_only = vec![0.0_f32; n_embd];
        attention_block_forward_position_zero(
            &x,
            &attn_w,
            g.layers[0].attn_q,
            g.layers[0].attn_k,
            g.layers[0].attn_v,
            g.layers[0].attn_output,
            &asn_w,
            &g.config,
            &mut attn_only,
        )
        .unwrap();

        let mut full = vec![0.0_f32; n_embd];
        transformer_block_forward_position_zero(&x, &g.layers[0], &g.config, &mut full).unwrap();

        // Full block adds (residual #1 + FFN + residual #2) to attention output, so should differ.
        let mut differ = 0usize;
        for i in 0..n_embd {
            if (full[i] - attn_only[i]).abs() > 1e-6 {
                differ += 1;
            }
        }
        assert!(
            differ > n_embd / 2,
            "full block must differ from attention-only output"
        );
    });
}

#[test]
fn full_block_differs_from_ffn_only() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();

        let fn_w = f32_tensor_to_vec(g.layers[0].ffn_norm).unwrap();
        let fsn_w = f32_tensor_to_vec(g.layers[0].ffn_sub_norm).unwrap();
        let mut ffn_only = vec![0.0_f32; n_embd];
        ffn_block_forward(
            &x,
            &fn_w,
            g.layers[0].ffn_gate,
            g.layers[0].ffn_up,
            g.layers[0].ffn_down,
            &fsn_w,
            &g.config,
            &mut ffn_only,
        )
        .unwrap();

        let mut full = vec![0.0_f32; n_embd];
        transformer_block_forward_position_zero(&x, &g.layers[0], &g.config, &mut full).unwrap();

        let mut differ = 0usize;
        for i in 0..n_embd {
            if (full[i] - ffn_only[i]).abs() > 1e-6 {
                differ += 1;
            }
        }
        assert!(
            differ > n_embd / 2,
            "full block must differ from ffn-only output"
        );
    });
}

#[test]
fn same_function_handles_layer_0_and_layer_1() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();

        let mut out0 = vec![0.0_f32; n_embd];
        let mut out1 = vec![0.0_f32; n_embd];
        transformer_block_forward_position_zero(&x, &g.layers[0], &g.config, &mut out0).unwrap();
        transformer_block_forward_position_zero(&x, &g.layers[1], &g.config, &mut out1).unwrap();

        // Both must produce finite outputs and must differ from each other.
        for v in out0.iter().chain(out1.iter()) {
            assert!(v.is_finite());
        }
        let mut differ = 0usize;
        for i in 0..n_embd {
            if (out0[i] - out1[i]).abs() > 1e-6 {
                differ += 1;
            }
        }
        assert!(
            differ > n_embd / 2,
            "blk.0 and blk.1 should produce different outputs"
        );
    });
}
