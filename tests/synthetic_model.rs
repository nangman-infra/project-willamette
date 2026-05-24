//! In-CI synthetic-GGUF tests for the model load + forward path.
//!
//! The real `microsoft/bitnet-b1.58-2B-4T-gguf` file is intentionally
//! not in CI (1.1 GiB, plus the project rule against committing model
//! files). Without it, every `tests/*.rs` test that does
//! `Path::new(MODEL_PATH).exists()` SKIPs at runtime — leaving the
//! model load and forward-path code at 0 % CI coverage.
//!
//! This file builds a tiny in-memory GGUF (≈73 KB) that exercises the
//! same code paths:
//!
//!   * Header + metadata + tensor directory + tensor data section
//!   * Real BitNet b1.58 architecture metadata (block_count,
//!     embedding_length, head_count, rope dimension count, …)
//!   * All 11 tensor types `ModelGraph::from_gguf` requires per layer
//!     (norm × 4, BitLinear × 7) plus `token_embd` + `output_norm`
//!   * Deterministic tensor data: F32 norms all = 1.0,
//!     F16 token_embd all = 1.0, BitLinear I2_S all ternary 0
//!     (byte 0x55, code `01`) with scale = 1.0
//!
//! Because every BitLinear weight is ternary 0, every matvec returns
//! all-zero — so the forward path runs deterministically without
//! producing NaNs, while still exercising the full pipeline:
//! embedding gather → RMSNorm with cached F32 weights → BitLinear
//! matvec → RoPE → KV cache append/read → softmax → sub-norm → second
//! BitLinear → residual → FFN → norm → output norm.
//!
//! What this file does NOT verify:
//!   * Numerical correctness of the inference (that's
//!     `scripts/compare_reference.sh` against pinned bitnet.cpp).
//!   * Tokenizer roundtrip (covered by `tokenizer_synthetic.rs`
//!     and `tokenizer_roundtrip.rs`).

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};

use project_willamette::gguf::reader::{GgufFile, GGUF_MAGIC};
use project_willamette::gguf::types::GgmlType;
use project_willamette::model::cached_forward::forward_with_cache;
use project_willamette::model::forward::forward_single_token_position_zero;
use project_willamette::model::kv_cache::KVCache;
use project_willamette::model::multi_forward::multi_token_forward;
use project_willamette::model::ModelGraph;

// ── tiny model config ────────────────────────────────────────────────

/// Synthetic BitNet b1.58 config sized down for in-CI testing. Every
/// dimension is the smallest value the production code will accept:
/// `embedding_length` must be a multiple of `QK_I2_S = 128` so that
/// BitLinear matvec can pack the in_dim into 32-byte blocks.
const N_LAYERS: u32 = 2;
const N_EMBD: u32 = 128;
const N_FF: u32 = 128;
const HEAD_COUNT: u32 = 1;
const HEAD_COUNT_KV: u32 = 1;
const HEAD_DIM: u32 = N_EMBD / HEAD_COUNT; // = 128
const VOCAB_SIZE: u32 = 4;
const CONTEXT_LENGTH: u32 = 16;
const ALIGNMENT: u64 = 32;

// ── GGUF wire-format helpers ─────────────────────────────────────────

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.write_all(s.as_bytes()).unwrap();
}

fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(8).unwrap();
    write_gguf_string(buf, value);
}

fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(4).unwrap();
    buf.write_u32::<LittleEndian>(value).unwrap();
}

fn write_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(6).unwrap();
    buf.write_f32::<LittleEndian>(value).unwrap();
}

// ── tensor data generators ───────────────────────────────────────────

/// F16 byte representation of 1.0: sign=0 exponent=01111 mantissa=0
const F16_ONE_LE: [u8; 2] = [0x00, 0x3C];

/// I2_S code byte for "all zeros" — four 2-bit codes of `01` packed
/// into one byte. Code `01` decodes to ternary 0 (see
/// `src/model/bitlinear.rs::ternary_from_code`).
const I2S_ALL_ZERO_CODE_BYTE: u8 = 0x55;

fn f32_tensor_bytes_all_ones(n_elements: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_elements * 4);
    for _ in 0..n_elements {
        out.write_f32::<LittleEndian>(1.0).unwrap();
    }
    out
}

fn f16_tensor_bytes_all_ones(n_elements: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_elements * 2);
    for _ in 0..n_elements {
        out.extend_from_slice(&F16_ONE_LE);
    }
    out
}

/// Build the packed I2_S bytes plus the 32-byte trailing scale block.
/// `in_dim` must be a positive multiple of 128 (QK_I2_S).
fn i2s_tensor_with_scale(in_dim: usize, out_dim: usize) -> Vec<u8> {
    assert!(in_dim > 0 && in_dim.is_multiple_of(128));
    let packed_size = (in_dim / 4) * out_dim;
    let mut out = vec![I2S_ALL_ZERO_CODE_BYTE; packed_size];
    // 32-byte trailing scale block: 4 bytes f32 scale + 28 bytes zero.
    out.write_f32::<LittleEndian>(1.0).unwrap();
    out.extend_from_slice(&[0u8; 28]);
    out
}

// ── tensor descriptor ────────────────────────────────────────────────

struct TensorDesc {
    name: String,
    shape: Vec<u64>,
    ggml_type: GgmlType,
    data: Vec<u8>,
}

impl TensorDesc {
    fn f32_norm(name: impl Into<String>, dim: u32) -> Self {
        Self {
            name: name.into(),
            shape: vec![dim as u64],
            ggml_type: GgmlType::F32,
            data: f32_tensor_bytes_all_ones(dim as usize),
        }
    }

    fn f16_embd(name: impl Into<String>, n_embd: u32, vocab_size: u32) -> Self {
        Self {
            name: name.into(),
            shape: vec![n_embd as u64, vocab_size as u64],
            ggml_type: GgmlType::F16,
            data: f16_tensor_bytes_all_ones((n_embd * vocab_size) as usize),
        }
    }

    fn bitlinear(name: impl Into<String>, in_dim: u32, out_dim: u32) -> Self {
        Self {
            name: name.into(),
            shape: vec![in_dim as u64, out_dim as u64],
            ggml_type: GgmlType::BitNetI2S,
            data: i2s_tensor_with_scale(in_dim as usize, out_dim as usize),
        }
    }
}

// ── full GGUF builder ────────────────────────────────────────────────

fn build_synthetic_bitnet_gguf() -> Vec<u8> {
    // 1) Tensor list. Order matters only insofar as offsets are computed
    //    sequentially; the GGUF spec doesn't impose a particular order.
    let mut tensors: Vec<TensorDesc> = Vec::new();
    tensors.push(TensorDesc::f16_embd(
        "token_embd.weight",
        N_EMBD,
        VOCAB_SIZE,
    ));
    tensors.push(TensorDesc::f32_norm("output_norm.weight", N_EMBD));
    for il in 0..N_LAYERS {
        tensors.push(TensorDesc::f32_norm(
            format!("blk.{}.attn_norm.weight", il),
            N_EMBD,
        ));
        tensors.push(TensorDesc::f32_norm(
            format!("blk.{}.attn_sub_norm.weight", il),
            N_EMBD,
        ));
        tensors.push(TensorDesc::bitlinear(
            format!("blk.{}.attn_q.weight", il),
            N_EMBD,
            HEAD_DIM * HEAD_COUNT,
        ));
        let kv_dim = HEAD_DIM * HEAD_COUNT_KV;
        tensors.push(TensorDesc::bitlinear(
            format!("blk.{}.attn_k.weight", il),
            N_EMBD,
            kv_dim,
        ));
        tensors.push(TensorDesc::bitlinear(
            format!("blk.{}.attn_v.weight", il),
            N_EMBD,
            kv_dim,
        ));
        tensors.push(TensorDesc::bitlinear(
            format!("blk.{}.attn_output.weight", il),
            HEAD_DIM * HEAD_COUNT,
            N_EMBD,
        ));
        tensors.push(TensorDesc::f32_norm(
            format!("blk.{}.ffn_norm.weight", il),
            N_EMBD,
        ));
        tensors.push(TensorDesc::f32_norm(
            format!("blk.{}.ffn_sub_norm.weight", il),
            N_FF,
        ));
        tensors.push(TensorDesc::bitlinear(
            format!("blk.{}.ffn_gate.weight", il),
            N_EMBD,
            N_FF,
        ));
        tensors.push(TensorDesc::bitlinear(
            format!("blk.{}.ffn_up.weight", il),
            N_EMBD,
            N_FF,
        ));
        tensors.push(TensorDesc::bitlinear(
            format!("blk.{}.ffn_down.weight", il),
            N_FF,
            N_EMBD,
        ));
    }

    // 2) Compute each tensor's relative offset (within the data section)
    //    so we can emit a complete directory before the data bytes.
    let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut running: u64 = 0;
    for t in &tensors {
        // Each tensor starts at the next alignment boundary within the
        // data section, matching `align_offset` in reader.rs.
        let aligned = align(running, ALIGNMENT);
        offsets.push(aligned);
        running = aligned + t.data.len() as u64;
    }

    // 3) Header + metadata + tensor directory.
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();

    // Metadata KV count is fixed by the keys we emit below — 11.
    buf.write_u64::<LittleEndian>(11).unwrap();
    write_kv_string(&mut buf, "general.architecture", "bitnet-b1.58");
    write_kv_u32(&mut buf, "bitnet-b1.58.block_count", N_LAYERS);
    write_kv_u32(&mut buf, "bitnet-b1.58.embedding_length", N_EMBD);
    write_kv_u32(&mut buf, "bitnet-b1.58.feed_forward_length", N_FF);
    write_kv_u32(&mut buf, "bitnet-b1.58.context_length", CONTEXT_LENGTH);
    write_kv_u32(&mut buf, "bitnet-b1.58.attention.head_count", HEAD_COUNT);
    write_kv_u32(
        &mut buf,
        "bitnet-b1.58.attention.head_count_kv",
        HEAD_COUNT_KV,
    );
    write_kv_f32(
        &mut buf,
        "bitnet-b1.58.attention.layer_norm_rms_epsilon",
        1.0e-5,
    );
    write_kv_u32(&mut buf, "bitnet-b1.58.rope.dimension_count", HEAD_DIM);
    write_kv_f32(&mut buf, "bitnet-b1.58.rope.freq_base", 500_000.0);
    write_kv_u32(&mut buf, "bitnet-b1.58.vocab_size", VOCAB_SIZE);

    // Tensor directory.
    for (t, &off) in tensors.iter().zip(offsets.iter()) {
        write_gguf_string(&mut buf, &t.name);
        buf.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
        for &d in &t.shape {
            buf.write_u64::<LittleEndian>(d).unwrap();
        }
        buf.write_u32::<LittleEndian>(t.ggml_type.to_raw()).unwrap();
        buf.write_u64::<LittleEndian>(off).unwrap();
    }

    // 4) Pad to alignment boundary, then write tensor data.
    let header_end = buf.len() as u64;
    let data_section_start = align(header_end, ALIGNMENT);
    buf.resize(data_section_start as usize, 0);

    for (t, &off) in tensors.iter().zip(offsets.iter()) {
        let abs = data_section_start + off;
        if buf.len() < abs as usize {
            buf.resize(abs as usize, 0);
        }
        // Tensor data starts exactly at abs; the alignment above guarantees
        // the same offset we encoded in the directory.
        assert_eq!(buf.len() as u64, abs, "{}: write offset drift", t.name);
        buf.extend_from_slice(&t.data);
    }

    buf
}

fn align(offset: u64, alignment: u64) -> u64 {
    let r = offset % alignment;
    if r == 0 {
        offset
    } else {
        offset + (alignment - r)
    }
}

// ── tests ────────────────────────────────────────────────────────────

#[test]
fn synthetic_gguf_parses_to_complete_model_graph() {
    let buf = build_synthetic_bitnet_gguf();
    let gguf = GgufFile::parse(&buf).expect("synthetic GGUF must parse");

    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.tensor_count, 2 + N_LAYERS as u64 * 11);

    let graph = ModelGraph::from_gguf(&gguf).expect("ModelGraph must build");
    assert_eq!(graph.config.architecture, "bitnet-b1.58");
    assert_eq!(graph.config.block_count, N_LAYERS);
    assert_eq!(graph.config.embedding_length, N_EMBD);
    assert_eq!(graph.config.head_dim, HEAD_DIM);
    assert_eq!(graph.layers.len(), N_LAYERS as usize);
    assert!(graph.lm_head_is_tied());
}

#[test]
fn synthetic_model_pre_decodes_norm_caches() {
    let buf = build_synthetic_bitnet_gguf();
    let gguf = GgufFile::parse(&buf).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    // Stage 10-A invariant: every F32 norm tensor is decoded once at
    // load time and stored alongside its TensorView reference.
    assert_eq!(graph.output_norm_f32.len(), N_EMBD as usize);
    assert!(
        graph.output_norm_f32.iter().all(|&v| v == 1.0),
        "synthetic norms are all-ones — pre-decoded cache must reflect that"
    );

    for layer in &graph.layers {
        assert_eq!(layer.attn_norm_f32.len(), N_EMBD as usize);
        assert_eq!(layer.attn_sub_norm_f32.len(), N_EMBD as usize);
        assert_eq!(layer.ffn_norm_f32.len(), N_EMBD as usize);
        assert_eq!(layer.ffn_sub_norm_f32.len(), N_FF as usize);
        assert!(layer.attn_norm_f32.iter().all(|&v| v == 1.0));
        assert!(layer.attn_sub_norm_f32.iter().all(|&v| v == 1.0));
        assert!(layer.ffn_norm_f32.iter().all(|&v| v == 1.0));
        assert!(layer.ffn_sub_norm_f32.iter().all(|&v| v == 1.0));
    }
}

#[test]
fn synthetic_forward_runs_with_cached_norms_and_zero_weights() {
    // With all-zero BitLinear weights and all-1.0 norm/embed weights,
    // forward output should be finite (no NaN/inf) and have the right
    // length. We don't check numeric values — that's bitnet.cpp parity
    // territory, not unit-test territory.
    let buf = build_synthetic_bitnet_gguf();
    let gguf = GgufFile::parse(&buf).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let hidden = forward_single_token_position_zero(&graph, 0).expect("forward");
    assert_eq!(hidden.len(), N_EMBD as usize);
    assert!(hidden.iter().all(|v| v.is_finite()));
}

#[test]
fn synthetic_cached_forward_runs_for_two_positions() {
    // Exercises cached_forward (the chat hot path) for positions 0 + 1.
    // Cache continuity invariant: position must equal cache.position()
    // on entry. After two calls, cache should hold 2 tokens.
    let buf = build_synthetic_bitnet_gguf();
    let gguf = GgufFile::parse(&buf).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let mut cache = KVCache::new(N_LAYERS as usize, HEAD_DIM as usize, 8);
    let h0 = forward_with_cache(&graph, &mut cache, 0, 0).expect("forward pos 0");
    assert_eq!(h0.len(), N_EMBD as usize);
    assert!(h0.iter().all(|v| v.is_finite()));
    assert_eq!(cache.position(), 1);

    let h1 = forward_with_cache(&graph, &mut cache, 1, 1).expect("forward pos 1");
    assert_eq!(h1.len(), N_EMBD as usize);
    assert!(h1.iter().all(|v| v.is_finite()));
    assert_eq!(cache.position(), 2);
}

#[test]
fn synthetic_multi_token_forward_runs() {
    // Exercises multi_forward.rs in one call across two tokens.
    let buf = build_synthetic_bitnet_gguf();
    let gguf = GgufFile::parse(&buf).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let hidden = multi_token_forward(&graph, &[0, 1]).expect("multi-token forward");
    // multi_token_forward returns the last token's hidden state.
    assert_eq!(hidden.len(), N_EMBD as usize);
    assert!(hidden.iter().all(|v| v.is_finite()));
}

#[test]
fn synthetic_cached_vs_no_cache_agree_on_position_zero() {
    // Cache reuses pre-decoded norms; the no-cache path
    // (forward_single_token_position_zero / block.rs) uses them too.
    // For position 0 with the same token, both must agree (modulo
    // f32 rounding noise, which is zero here because every BitLinear
    // matvec returns exact 0.0).
    let buf = build_synthetic_bitnet_gguf();
    let gguf = GgufFile::parse(&buf).expect("parse");
    let graph = ModelGraph::from_gguf(&gguf).expect("graph");

    let h_no_cache = forward_single_token_position_zero(&graph, 0).expect("no-cache");
    let mut cache = KVCache::new(N_LAYERS as usize, HEAD_DIM as usize, 4);
    let h_cache = forward_with_cache(&graph, &mut cache, 0, 0).expect("cache");
    assert_eq!(h_no_cache.len(), h_cache.len());
    for (a, b) in h_no_cache.iter().zip(h_cache.iter()) {
        // Numerically these are computed by independent code paths
        // (block.rs vs cached_forward.rs) but produce the same result
        // because the synthetic weights make both deterministic.
        assert!(
            (a - b).abs() < 1.0e-4,
            "cache vs no-cache mismatch: {} vs {}",
            a,
            b
        );
    }
}
