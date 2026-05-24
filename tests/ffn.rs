//! Stage 4-D2 integration tests — FFN path against the real GGUF.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::memory::mmap::ModelMmap;
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
            "SKIP: real GGUF not found at {} — Stage 4-D2 tests require it",
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
fn ffn_block_forward_produces_n_embd_finite_output() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();

        let fn_w = f32_tensor_to_vec(g.layers[0].ffn_norm).unwrap();
        let fsn_w = f32_tensor_to_vec(g.layers[0].ffn_sub_norm).unwrap();

        let mut out = vec![0.0_f32; n_embd];
        ffn_block_forward(
            &x,
            &fn_w,
            g.layers[0].ffn_gate,
            g.layers[0].ffn_up,
            g.layers[0].ffn_down,
            &fsn_w,
            &g.config,
            &mut out,
        )
        .expect("ffn forward");

        assert_eq!(out.len(), n_embd);
        assert_eq!(out.len(), 2560);
        let nz = out.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nz > n_embd / 2,
            "more than half of ffn output should be non-zero, got {}",
            nz
        );
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "non-finite ffn output at dim {}: {}", i, v);
        }
    });
}

#[test]
fn ffn_block_forward_is_deterministic() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let mut x = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut x).unwrap();
        let fn_w = f32_tensor_to_vec(g.layers[0].ffn_norm).unwrap();
        let fsn_w = f32_tensor_to_vec(g.layers[0].ffn_sub_norm).unwrap();
        let mut a = vec![0.0_f32; n_embd];
        let mut b = vec![0.0_f32; n_embd];
        for out in [&mut a, &mut b] {
            ffn_block_forward(
                &x,
                &fn_w,
                g.layers[0].ffn_gate,
                g.layers[0].ffn_up,
                g.layers[0].ffn_down,
                &fsn_w,
                &g.config,
                out,
            )
            .unwrap();
        }
        assert_eq!(a, b);
    });
}

#[test]
fn ffn_zero_input_yields_zero_output() {
    // RMSNorm of zeros → zeros → matvecs → zeros → relu²(0)=0 → fused=0 →
    // RMSNorm(zeros)=zeros → ffn_down(zeros)=zeros.
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let fn_w = f32_tensor_to_vec(g.layers[0].ffn_norm).unwrap();
        let fsn_w = f32_tensor_to_vec(g.layers[0].ffn_sub_norm).unwrap();
        let x = vec![0.0_f32; n_embd];
        let mut out = vec![1.0_f32; n_embd]; // sentinel
        ffn_block_forward(
            &x,
            &fn_w,
            g.layers[0].ffn_gate,
            g.layers[0].ffn_up,
            g.layers[0].ffn_down,
            &fsn_w,
            &g.config,
            &mut out,
        )
        .unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "zero input must produce zero output (dim {})", i);
        }
    });
}

#[test]
fn ffn_different_tokens_produce_different_outputs() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let fn_w = f32_tensor_to_vec(g.layers[0].ffn_norm).unwrap();
        let fsn_w = f32_tensor_to_vec(g.layers[0].ffn_sub_norm).unwrap();
        let mut xa = vec![0.0_f32; n_embd];
        let mut xb = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut xa).unwrap();
        embedding_gather_f16(g.token_embd, 101193, &mut xb).unwrap();
        let mut a = vec![0.0_f32; n_embd];
        let mut b = vec![0.0_f32; n_embd];
        for (x, out) in [(&xa, &mut a), (&xb, &mut b)] {
            ffn_block_forward(
                x,
                &fn_w,
                g.layers[0].ffn_gate,
                g.layers[0].ffn_up,
                g.layers[0].ffn_down,
                &fsn_w,
                &g.config,
                out,
            )
            .unwrap();
        }
        let mut differ = 0;
        for i in 0..n_embd {
            if (a[i] - b[i]).abs() > 1e-6 {
                differ += 1;
            }
        }
        assert!(differ > n_embd / 2);
    });
}
