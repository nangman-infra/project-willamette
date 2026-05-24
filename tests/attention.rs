//! Stage 4-D1 integration tests — single-token attention path on real GGUF.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::attention::attention_block_forward_position_zero;
use project_willamette::model::bitlinear::bitlinear_i2s_matvec_f32;
use project_willamette::model::primitives::{
    embedding_gather_f16, f32_tensor_to_vec, rms_norm_f32,
};
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 4-D1 tests require it",
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
fn attention_block_forward_produces_2560_finite_output() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();

        let an = f32_tensor_to_vec(g.layers[0].attn_norm).unwrap();
        let asn = f32_tensor_to_vec(g.layers[0].attn_sub_norm).unwrap();

        let mut out = vec![0.0_f32; n_embd];
        attention_block_forward_position_zero(
            &x,
            &an,
            g.layers[0].attn_q,
            g.layers[0].attn_k,
            g.layers[0].attn_v,
            g.layers[0].attn_output,
            &asn,
            &g.config,
            &mut out,
        )
        .expect("attention forward");

        assert_eq!(out.len(), n_embd);
        assert_eq!(out.len(), 2560);
        let nz = out.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nz > n_embd / 2,
            "more than half of attention output should be non-zero, got {}",
            nz
        );
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "non-finite attention output at dim {}: {}",
                i,
                v
            );
        }
    });
}

#[test]
fn attention_block_forward_is_deterministic() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();
        let an = f32_tensor_to_vec(g.layers[0].attn_norm).unwrap();
        let asn = f32_tensor_to_vec(g.layers[0].attn_sub_norm).unwrap();

        let mut a = vec![0.0_f32; n_embd];
        let mut b = vec![0.0_f32; n_embd];
        for out in [&mut a, &mut b] {
            attention_block_forward_position_zero(
                &x,
                &an,
                g.layers[0].attn_q,
                g.layers[0].attn_k,
                g.layers[0].attn_v,
                g.layers[0].attn_output,
                &asn,
                &g.config,
                out,
            )
            .unwrap();
        }
        assert_eq!(a, b, "attention forward must be deterministic");
    });
}

#[test]
fn attention_block_forward_q_k_v_path_shapes() {
    // Confirm the intermediate Q/K/V shapes are what we expect.
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let kv_dim = g.config.kv_dim as usize;

        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();
        let an = f32_tensor_to_vec(g.layers[0].attn_norm).unwrap();
        let mut x_norm = vec![0.0_f32; n_embd];
        rms_norm_f32(&x, &an, g.config.layer_norm_rms_epsilon, &mut x_norm).unwrap();

        let mut q = vec![0.0_f32; n_embd];
        let mut k = vec![0.0_f32; kv_dim];
        let mut v = vec![0.0_f32; kv_dim];
        bitlinear_i2s_matvec_f32(g.layers[0].attn_q, &x_norm, &mut q).unwrap();
        bitlinear_i2s_matvec_f32(g.layers[0].attn_k, &x_norm, &mut k).unwrap();
        bitlinear_i2s_matvec_f32(g.layers[0].attn_v, &x_norm, &mut v).unwrap();

        assert_eq!(q.len(), 2560);
        assert_eq!(k.len(), 640);
        assert_eq!(v.len(), 640);
        for v in q.iter().chain(k.iter()).chain(v.iter()) {
            assert!(v.is_finite());
        }
    });
}

#[test]
fn different_tokens_produce_different_attention_outputs() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let an = f32_tensor_to_vec(g.layers[0].attn_norm).unwrap();
        let asn = f32_tensor_to_vec(g.layers[0].attn_sub_norm).unwrap();

        let mut x_a = vec![0.0_f32; n_embd];
        let mut x_b = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x_a).unwrap();
        embedding_gather_f16(g.token_embd, 101193, &mut x_b).unwrap();

        let mut out_a = vec![0.0_f32; n_embd];
        let mut out_b = vec![0.0_f32; n_embd];
        for (x, out) in [(&x_a, &mut out_a), (&x_b, &mut out_b)] {
            attention_block_forward_position_zero(
                x,
                &an,
                g.layers[0].attn_q,
                g.layers[0].attn_k,
                g.layers[0].attn_v,
                g.layers[0].attn_output,
                &asn,
                &g.config,
                out,
            )
            .unwrap();
        }
        let mut differ = 0usize;
        for i in 0..n_embd {
            if (out_a[i] - out_b[i]).abs() > 1e-6 {
                differ += 1;
            }
        }
        assert!(
            differ > n_embd / 2,
            "different tokens should produce mostly-different attention outputs"
        );
    });
}
