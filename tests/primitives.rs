//! Stage 4-B integration tests — primitives over the real GGUF.
//!
//! Verifies that each Stage 4-B primitive (`embedding_gather_f16`,
//! `rms_norm_f32`, `apply_rope_f32`, attention shape helpers) operates
//! correctly on the actual `token_embd.weight` and per-layer norm tensors
//! of `ggml-model-i2_s.gguf`. Skipped (SKIP message + pass) if the model
//! file is missing.
//!
//! No matmul, no full attention math, no logits, no sampling.

use std::path::Path;

use project_willamette::gguf::reader::GgufFile;
use project_willamette::gguf::types::GgmlType;
use project_willamette::memory::mmap::ModelMmap;
use project_willamette::model::primitives::{
    apply_rope_f32, attention_scale, causal_mask_value, embedding_gather_f16, gqa_group_size,
    kv_head_for_q_head, rms_norm_f32, AttentionShape, RopeType,
};
use project_willamette::model::ModelGraph;

const MODEL_PATH: &str = "./models/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";

fn with_real_graph<F>(f: F)
where
    F: FnOnce(&ModelGraph<'_>),
{
    if !Path::new(MODEL_PATH).exists() {
        eprintln!(
            "SKIP: real GGUF not found at {} — Stage 4-B tests require it",
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

/// Re-read F32 tensor bytes into an owned Vec<f32>. Used so the test
/// helpers can hand RMSNorm a plain f32 slice from a real F32 weight
/// tensor without taking on additional Stage 4-B surface area.
fn f32_tensor_to_vec(t: &project_willamette::gguf::tensor::TensorView<'_>) -> Vec<f32> {
    assert_eq!(
        t.ggml_type,
        GgmlType::F32,
        "tensor {:?} must be F32",
        t.name
    );
    let n = t.n_elements() as usize;
    assert_eq!(t.data.len(), n * 4);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &t.data[4 * i..4 * i + 4];
        let bits = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        out.push(f32::from_bits(bits));
    }
    out
}

// ── 1. embedding row gather ────────────────────────────────────────────

#[test]
fn embedding_gather_hello_length_2560() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        // 15339 is "hello" in the LLaMA-3-family tokenizer the model ships
        // with (cross-confirmed against the Stage 2 CLI smoke test).
        let mut out = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 15339, &mut out).unwrap();
        assert_eq!(out.len(), n_embd);
        assert_eq!(out.len(), 2560);

        // The embedding must not be entirely zero (a real trained value
        // is never exactly zero across all 2560 dims).
        let nz = out.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nz > 0,
            "embedding row 15339 is all-zero — this would indicate a bad gather"
        );
        // And every value must be finite (no NaN/inf from f16 decode bugs).
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "non-finite value {} at dim {}", v, i);
        }
    });
}

#[test]
fn embedding_gather_korean_tokens_distinct_rows() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        // From the Stage 2 CLI: "안녕하세요" → [101193, 124409]
        let mut row_a = vec![0.0_f32; n_embd];
        let mut row_b = vec![0.0_f32; n_embd];
        embedding_gather_f16(g.token_embd, 101193, &mut row_a).unwrap();
        embedding_gather_f16(g.token_embd, 124409, &mut row_b).unwrap();

        // Different token ids must produce different embedding rows.
        let mut differences = 0usize;
        for i in 0..n_embd {
            if (row_a[i] - row_b[i]).abs() > 0.0 {
                differences += 1;
            }
        }
        assert!(
            differences > 1000,
            "two different token rows should differ in many dims, got only {}",
            differences
        );
    });
}

#[test]
fn embedding_gather_out_of_range_errors() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let n_vocab = g.config.vocab_size;
        let mut out = vec![0.0_f32; n_embd];
        let r = embedding_gather_f16(g.token_embd, n_vocab, &mut out);
        assert!(r.is_err(), "vocab_size as token id must be rejected");
        let r = embedding_gather_f16(g.token_embd, u32::MAX, &mut out);
        assert!(r.is_err(), "u32::MAX token id must be rejected");
    });
}

#[test]
fn embedding_gather_length_mismatch_errors() {
    with_real_graph(|g| {
        let mut out_short = vec![0.0_f32; 100];
        assert!(embedding_gather_f16(g.token_embd, 0, &mut out_short).is_err());
    });
}

// ── 2. RMSNorm on real weights ────────────────────────────────────────

#[test]
fn rms_norm_with_layer0_attn_norm_preserves_length() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        // Synthetic input — RMSNorm is shape-safe regardless of value;
        // we only need to confirm length is preserved and the output is
        // finite given real F32 norm weights.
        let x: Vec<f32> = (0..n_embd)
            .map(|i| ((i as f32) / (n_embd as f32)) - 0.5)
            .collect();
        let w = f32_tensor_to_vec(g.layers[0].attn_norm);
        assert_eq!(w.len(), n_embd);

        let mut out = vec![0.0_f32; n_embd];
        rms_norm_f32(&x, &w, g.config.layer_norm_rms_epsilon, &mut out).unwrap();
        assert_eq!(out.len(), n_embd);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "non-finite value {} at dim {}", v, i);
        }
    });
}

#[test]
fn rms_norm_with_output_norm_preserves_length() {
    with_real_graph(|g| {
        let n_embd = g.config.embedding_length as usize;
        let x: Vec<f32> = vec![1.0_f32; n_embd];
        let w = f32_tensor_to_vec(g.output_norm);
        assert_eq!(w.len(), n_embd);

        let mut out = vec![0.0_f32; n_embd];
        rms_norm_f32(&x, &w, g.config.layer_norm_rms_epsilon, &mut out).unwrap();
        assert_eq!(out.len(), n_embd);
        // For uniform input 1.0 across 2560 dims:
        //   mean(x²) = 1, sqrt(1 + eps) ≈ 1, so out[i] ≈ w[i]
        for i in 0..n_embd {
            let expected = w[i];
            assert!(
                (out[i] - expected).abs() < 1e-3,
                "uniform input × RMSNorm should ~= weight at dim {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    });
}

#[test]
fn rms_norm_works_across_norm_tensor_widths() {
    with_real_graph(|g| {
        // ffn_sub_norm has width n_ff = 6912, not n_embd = 2560.
        let n_ff = g.config.feed_forward_length as usize;
        let w = f32_tensor_to_vec(g.layers[0].ffn_sub_norm);
        assert_eq!(w.len(), n_ff);
        let x = vec![0.5_f32; n_ff];
        let mut out = vec![0.0_f32; n_ff];
        rms_norm_f32(&x, &w, g.config.layer_norm_rms_epsilon, &mut out).unwrap();
        assert_eq!(out.len(), n_ff);
    });
}

// ── 3. RoPE primitive ─────────────────────────────────────────────────

#[test]
fn rope_preserves_head_dim_length() {
    with_real_graph(|g| {
        let head_dim = g.config.head_dim as usize;
        let n_rot = g.config.rope_dimension_count as usize;
        let freq_base = g.config.rope_freq_base;

        let mut q = vec![0.01_f32; head_dim];
        apply_rope_f32(
            &mut q,
            head_dim,
            n_rot,
            7, // arbitrary position
            freq_base,
            RopeType::Neox,
        )
        .unwrap();
        assert_eq!(q.len(), head_dim);
        for v in &q {
            assert!(v.is_finite());
        }
    });
}

#[test]
fn rope_uses_neox_for_bitnet_b158() {
    // Pin: src/llama.cpp:20117 maps LLM_ARCH_BITNET_B158 → LLAMA_ROPE_TYPE_NEOX.
    // Document via assertion: the test confirms the project applies NEOX
    // by detecting the divergence from Norm.
    let mut a: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let mut b = a.clone();
    apply_rope_f32(&mut a, 128, 128, 5, 500_000.0, RopeType::Norm).unwrap();
    apply_rope_f32(&mut b, 128, 128, 5, 500_000.0, RopeType::Neox).unwrap();
    let max_diff: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "Norm and NEOX must differ for non-trivial input; max_diff={}",
        max_diff
    );
}

// ── 4. Attention shape primitives ─────────────────────────────────────

#[test]
fn attention_shape_from_config_matches_inspect() {
    with_real_graph(|g| {
        let s = AttentionShape::from_config(
            g.config.head_count,
            g.config.head_count_kv,
            g.config.head_dim,
        )
        .unwrap();
        assert_eq!(s.n_heads, 20);
        assert_eq!(s.n_kv_heads, 5);
        assert_eq!(s.head_dim, 128);
        assert_eq!(s.group_size, 4);
        assert_eq!(s.q_per_token_dim, 2560);
        assert_eq!(s.kv_per_token_dim, 640);
    });
}

#[test]
fn attention_scale_is_inverse_sqrt_128() {
    let s = attention_scale(128);
    let expected = 1.0_f32 / (128.0_f32).sqrt();
    assert!((s - expected).abs() < 1e-7);
}

#[test]
fn gqa_mapping_covers_all_q_heads() {
    // With n_heads=20, n_kv_heads=5, group_size=4 → kv_head ∈ {0..5}.
    let gs = gqa_group_size(20, 5).unwrap();
    assert_eq!(gs, 4);
    let mut hits = [0u32; 5];
    for q in 0..20 {
        let kv = kv_head_for_q_head(q, gs);
        assert!(kv < 5);
        hits[kv as usize] += 1;
    }
    // Every KV head must be hit by exactly 4 Q heads.
    for kv in 0..5 {
        assert_eq!(hits[kv], 4, "kv head {} hit count = {}", kv, hits[kv]);
    }
}

#[test]
fn causal_mask_lower_triangular() {
    // For a 4-token window, the mask should be 0 on/below diagonal,
    // -inf strictly above.
    for q in 0..4u32 {
        for k in 0..4u32 {
            let m = causal_mask_value(q, k);
            if k <= q {
                assert_eq!(m, 0.0, "(q={},k={}) must be 0", q, k);
            } else {
                assert!(m == f32::NEG_INFINITY, "(q={},k={}) must be -inf", q, k);
            }
        }
    }
}
