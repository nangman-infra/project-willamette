//! Synthetic BitNet b1.58 GGUF builder.
//!
//! Used for two purposes:
//!
//!   1. **CI coverage** — `tests/synthetic_model.rs` exercises the
//!      full load / forward path against the smallest valid model
//!      (`Preset::Tiny`, ≈ 73 KB) so production code stays covered
//!      without committing the real 1.1 GiB GGUF.
//!   2. **Throughput benchmarking on humble hardware** — `Preset::Medium`
//!      builds a ≈ 110 M-parameter BitNet b1.58 model in the same
//!      scale class as the 110 M TinyStories Llama 2 model
//!      Karpathy / EXO Labs used on Pentium II hardware. Weights
//!      are random ternary `{-1, 0, +1}` so quality is meaningless
//!      — only the **matvec / forward throughput** is measured.
//!
//! ## What this module does NOT make
//!
//! * No tokenizer metadata (no `tokenizer.ggml.tokens` / `merges`).
//!   `inspect` and `bench` work without a tokenizer; `run` / `chat`
//!   / `tui` need one and won't work against the synthetic GGUF.
//!   This is intentional: the synthetic file is for kernel-level
//!   throughput measurement, not for end-to-end inference.
//! * No claim of inference *quality*. Random ternary weights produce
//!   garbage tokens by construction — see [[feedback-no-fake]]. The
//!   only honest use is timing.

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::gguf::reader::GGUF_MAGIC;
use crate::gguf::types::GgmlType;

/// Pre-defined model sizes. Each preset locks every dimension so the
/// resulting parameter count is reproducible across hosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// ≈ 73 KB. The minimum dimensions our loader will accept
    /// (`embedding_length` must be a multiple of `QK_I2_S = 128`).
    /// Used by `tests/synthetic_model.rs`.
    Tiny,
    /// ≈ 10 M parameters (n_embd 256, 6 layers, ff 512). Fits in L2 on
    /// almost any host; useful for "how much faster than scalar?" runs
    /// without the patience of medium.
    Small,
    /// ≈ 110 M parameters (n_embd 768, 12 layers, ff 2048, vocab 32000).
    /// Same scale class as Karpathy's `tinyllamas/stories110M.bin`,
    /// the model EXO Labs ran on a Pentium II 350 MHz at 35.9 tok/s.
    /// This is the preset for direct cross-architecture comparisons.
    Medium,
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub n_layers: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub rope_freq_base: f32,
    pub layer_norm_rms_epsilon: f32,
    /// Deterministic PRNG seed for the I2_S ternary code byte stream.
    /// Same seed → bit-identical GGUF file across hosts.
    pub seed: u64,
}

impl Preset {
    pub fn config(self) -> Config {
        match self {
            // QK_I2_S = 128 is the smallest multiple `embedding_length`
            // is allowed to take. Vocab 4 = absolute minimum, no
            // tokenizer needed.
            Preset::Tiny => Config {
                n_layers: 2,
                n_embd: 128,
                n_ff: 128,
                head_count: 1,
                head_count_kv: 1,
                head_dim: 128,
                vocab_size: 4,
                context_length: 16,
                rope_freq_base: 500_000.0,
                layer_norm_rms_epsilon: 1.0e-5,
                seed: 0,
            },
            // Roughly 10 M params: 256-d embeddings × 12000 vocab is
            // 3 M for token_embd; 6 layers × ~1 M per layer = 6 M.
            Preset::Small => Config {
                n_layers: 6,
                n_embd: 256,
                n_ff: 512,
                head_count: 4,
                head_count_kv: 4,
                head_dim: 64,
                vocab_size: 12_000,
                context_length: 1024,
                rope_freq_base: 500_000.0,
                layer_norm_rms_epsilon: 1.0e-5,
                seed: 1,
            },
            // TinyLlama-class sizing (Karpathy `stories110M`).
            // token_embd: 32000 × 768 = 24.6 M
            // per layer (12 total): ≈ 7.1 M = ~85 M
            // total ≈ 110 M.
            Preset::Medium => Config {
                n_layers: 12,
                n_embd: 768,
                n_ff: 2048,
                head_count: 12,
                head_count_kv: 12,
                head_dim: 64,
                vocab_size: 32_000,
                context_length: 2048,
                rope_freq_base: 500_000.0,
                layer_norm_rms_epsilon: 1.0e-5,
                seed: 2,
            },
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Tiny => "tiny",
            Preset::Small => "small",
            Preset::Medium => "medium",
        }
    }
}

/// Estimated parameter count (sum of every tensor's element count).
/// Approximate — counts the F16 token_embd, F32 norms, and ternary
/// BitLinear weights together as if they were one f32 array.
pub fn estimated_params(cfg: &Config) -> u64 {
    let n_embd = cfg.n_embd as u64;
    let n_ff = cfg.n_ff as u64;
    let n_layers = cfg.n_layers as u64;
    let vocab = cfg.vocab_size as u64;
    let kv_dim = cfg.head_dim as u64 * cfg.head_count_kv as u64;
    let q_dim = cfg.head_dim as u64 * cfg.head_count as u64;

    let token_embd = vocab * n_embd;
    let output_norm = n_embd;
    let per_layer = 2 * n_embd                  // attn_norm + attn_sub_norm (small ones)
        + (n_embd * q_dim)                      // attn_q
        + (n_embd * kv_dim)                     // attn_k
        + (n_embd * kv_dim)                     // attn_v
        + (q_dim * n_embd)                      // attn_output
        + n_embd                                // ffn_norm
        + n_ff                                  // ffn_sub_norm
        + (n_embd * n_ff)                       // ffn_gate
        + (n_embd * n_ff)                       // ffn_up
        + (n_ff * n_embd); // ffn_down
    token_embd + output_norm + n_layers * per_layer
}

// ── GGUF wire-format helpers (identical to tests/synthetic_model.rs) ─

fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.write_all(s.as_bytes()).unwrap();
}

fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(8).unwrap(); // STRING type tag
    write_gguf_string(buf, value);
}

fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(4).unwrap(); // U32 type tag
    buf.write_u32::<LittleEndian>(value).unwrap();
}

fn write_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
    write_gguf_string(buf, key);
    buf.write_u32::<LittleEndian>(6).unwrap(); // F32 type tag
    buf.write_f32::<LittleEndian>(value).unwrap();
}

// ── tensor data generators ───────────────────────────────────────────

/// F16 representation of 1.0 (sign 0, exp 01111, mantissa 0).
const F16_ONE_LE: [u8; 2] = [0x00, 0x3C];

/// Deterministic 64-bit xorshift PRNG. Used only here; no external
/// dep needed.
#[derive(Debug)]
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Avoid the absorbing 0 state.
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s
    }
    /// Returns a value in `{0, 1, 2}` — the three legal I2_S code
    /// values (0b00 = -1, 0b01 = 0, 0b10 = +1). `0b11` is the
    /// degenerate code the quantizer at `ggml-bitnet-mad.cpp:65` never
    /// produces; we don't either.
    fn next_legal_code(&mut self) -> u8 {
        (self.next_u64() % 3) as u8
    }
}

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
///
/// `seed_stream`: each call advances a shared PRNG so different
/// tensors get different (but reproducible) random ternary patterns.
/// All-zero (`Preset::Tiny`) was the original behaviour and is still
/// available via `all_zero=true` for the synth_test path that needs
/// deterministic zero outputs.
fn i2s_tensor_with_scale(
    in_dim: usize,
    out_dim: usize,
    rng: &mut Xorshift64,
    all_zero: bool,
) -> Vec<u8> {
    assert!(in_dim > 0 && in_dim.is_multiple_of(128));
    let packed_size = (in_dim / 4) * out_dim;
    let mut out: Vec<u8> = if all_zero {
        // Code 0b01 → ternary 0; four 0b01s packed → 0x55.
        vec![0x55_u8; packed_size]
    } else {
        let mut bytes = Vec::with_capacity(packed_size);
        for _ in 0..packed_size {
            let c0 = rng.next_legal_code();
            let c1 = rng.next_legal_code();
            let c2 = rng.next_legal_code();
            let c3 = rng.next_legal_code();
            bytes.push((c0 << 6) | (c1 << 4) | (c2 << 2) | c3);
        }
        bytes
    };
    // 32-byte trailing scale block: 4-byte f32 scale + 28 bytes zero.
    out.write_f32::<LittleEndian>(1.0).unwrap();
    out.extend_from_slice(&[0u8; 28]);
    out
}

// ── tensor descriptor + builder ──────────────────────────────────────

struct TensorDesc {
    name: String,
    shape: Vec<u64>,
    ggml_type: GgmlType,
    data: Vec<u8>,
}

const ALIGNMENT: u64 = 32;

fn align(offset: u64, alignment: u64) -> u64 {
    let r = offset % alignment;
    if r == 0 {
        offset
    } else {
        offset + (alignment - r)
    }
}

/// Build a complete GGUF byte buffer for the given preset.
///
/// `random_weights`: when true (the default for `Small` / `Medium`),
/// I2_S code bytes are drawn from a deterministic PRNG seeded by
/// `cfg.seed`; when false (used by `Tiny` so the test suite's
/// numerical assertions still hold), every code is `0` so every
/// BitLinear matvec returns 0.
pub fn build_gguf(preset: Preset, random_weights: bool) -> Vec<u8> {
    let cfg = preset.config();
    let mut rng = Xorshift64::new(cfg.seed);

    // 1) Tensor list.
    let mut tensors: Vec<TensorDesc> = Vec::new();
    tensors.push(TensorDesc {
        name: "token_embd.weight".to_string(),
        shape: vec![cfg.n_embd as u64, cfg.vocab_size as u64],
        ggml_type: GgmlType::F16,
        data: f16_tensor_bytes_all_ones((cfg.n_embd * cfg.vocab_size) as usize),
    });
    tensors.push(TensorDesc {
        name: "output_norm.weight".to_string(),
        shape: vec![cfg.n_embd as u64],
        ggml_type: GgmlType::F32,
        data: f32_tensor_bytes_all_ones(cfg.n_embd as usize),
    });

    let q_dim = cfg.head_dim * cfg.head_count;
    let kv_dim = cfg.head_dim * cfg.head_count_kv;

    for il in 0..cfg.n_layers {
        tensors.push(norm_desc(
            format!("blk.{}.attn_norm.weight", il),
            cfg.n_embd,
        ));
        tensors.push(norm_desc(
            format!("blk.{}.attn_sub_norm.weight", il),
            cfg.n_embd,
        ));
        tensors.push(bitlinear_desc(
            format!("blk.{}.attn_q.weight", il),
            cfg.n_embd,
            q_dim,
            &mut rng,
            !random_weights,
        ));
        tensors.push(bitlinear_desc(
            format!("blk.{}.attn_k.weight", il),
            cfg.n_embd,
            kv_dim,
            &mut rng,
            !random_weights,
        ));
        tensors.push(bitlinear_desc(
            format!("blk.{}.attn_v.weight", il),
            cfg.n_embd,
            kv_dim,
            &mut rng,
            !random_weights,
        ));
        tensors.push(bitlinear_desc(
            format!("blk.{}.attn_output.weight", il),
            q_dim,
            cfg.n_embd,
            &mut rng,
            !random_weights,
        ));
        tensors.push(norm_desc(format!("blk.{}.ffn_norm.weight", il), cfg.n_embd));
        tensors.push(norm_desc(
            format!("blk.{}.ffn_sub_norm.weight", il),
            cfg.n_ff,
        ));
        tensors.push(bitlinear_desc(
            format!("blk.{}.ffn_gate.weight", il),
            cfg.n_embd,
            cfg.n_ff,
            &mut rng,
            !random_weights,
        ));
        tensors.push(bitlinear_desc(
            format!("blk.{}.ffn_up.weight", il),
            cfg.n_embd,
            cfg.n_ff,
            &mut rng,
            !random_weights,
        ));
        tensors.push(bitlinear_desc(
            format!("blk.{}.ffn_down.weight", il),
            cfg.n_ff,
            cfg.n_embd,
            &mut rng,
            !random_weights,
        ));
    }

    // 2) Compute offsets.
    let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut running: u64 = 0;
    for t in &tensors {
        let aligned = align(running, ALIGNMENT);
        offsets.push(aligned);
        running = aligned + t.data.len() as u64;
    }

    // 3) Header + metadata + tensor directory.
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
    buf.write_u64::<LittleEndian>(11).unwrap(); // metadata KV count
    write_kv_string(&mut buf, "general.architecture", "bitnet-b1.58");
    write_kv_u32(&mut buf, "bitnet-b1.58.block_count", cfg.n_layers);
    write_kv_u32(&mut buf, "bitnet-b1.58.embedding_length", cfg.n_embd);
    write_kv_u32(&mut buf, "bitnet-b1.58.feed_forward_length", cfg.n_ff);
    write_kv_u32(&mut buf, "bitnet-b1.58.context_length", cfg.context_length);
    write_kv_u32(
        &mut buf,
        "bitnet-b1.58.attention.head_count",
        cfg.head_count,
    );
    write_kv_u32(
        &mut buf,
        "bitnet-b1.58.attention.head_count_kv",
        cfg.head_count_kv,
    );
    write_kv_f32(
        &mut buf,
        "bitnet-b1.58.attention.layer_norm_rms_epsilon",
        cfg.layer_norm_rms_epsilon,
    );
    write_kv_u32(&mut buf, "bitnet-b1.58.rope.dimension_count", cfg.head_dim);
    write_kv_f32(&mut buf, "bitnet-b1.58.rope.freq_base", cfg.rope_freq_base);
    write_kv_u32(&mut buf, "bitnet-b1.58.vocab_size", cfg.vocab_size);

    for (t, &off) in tensors.iter().zip(offsets.iter()) {
        write_gguf_string(&mut buf, &t.name);
        buf.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
        for &d in &t.shape {
            buf.write_u64::<LittleEndian>(d).unwrap();
        }
        buf.write_u32::<LittleEndian>(t.ggml_type.to_raw()).unwrap();
        buf.write_u64::<LittleEndian>(off).unwrap();
    }

    // 4) Pad + tensor data section.
    let header_end = buf.len() as u64;
    let data_section_start = align(header_end, ALIGNMENT);
    buf.resize(data_section_start as usize, 0);

    for (t, &off) in tensors.iter().zip(offsets.iter()) {
        let abs = data_section_start + off;
        if buf.len() < abs as usize {
            buf.resize(abs as usize, 0);
        }
        assert_eq!(buf.len() as u64, abs, "{}: write offset drift", t.name);
        buf.extend_from_slice(&t.data);
    }

    buf
}

fn norm_desc(name: String, dim: u32) -> TensorDesc {
    TensorDesc {
        name,
        shape: vec![dim as u64],
        ggml_type: GgmlType::F32,
        data: f32_tensor_bytes_all_ones(dim as usize),
    }
}

fn bitlinear_desc(
    name: String,
    in_dim: u32,
    out_dim: u32,
    rng: &mut Xorshift64,
    all_zero: bool,
) -> TensorDesc {
    TensorDesc {
        name,
        shape: vec![in_dim as u64, out_dim as u64],
        ggml_type: GgmlType::BitNetI2S,
        data: i2s_tensor_with_scale(in_dim as usize, out_dim as usize, rng, all_zero),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_preset_dims_match_qk_i2s_multiple() {
        let cfg = Preset::Tiny.config();
        assert!((cfg.n_embd as usize).is_multiple_of(128));
        assert!((cfg.n_ff as usize).is_multiple_of(128));
    }

    #[test]
    fn medium_preset_is_about_110m_params() {
        let cfg = Preset::Medium.config();
        let est = estimated_params(&cfg);
        assert!(
            (100_000_000..=130_000_000).contains(&est),
            "Medium preset estimated_params = {}, expected ~110M",
            est
        );
    }

    #[test]
    fn build_tiny_is_byte_stable() {
        // Tiny + all-zero weights → identical bytes across builds.
        let a = build_gguf(Preset::Tiny, false);
        let b = build_gguf(Preset::Tiny, false);
        assert_eq!(a, b, "Tiny build must be deterministic");
    }

    #[test]
    fn build_small_random_is_seed_stable() {
        // Same seed → same bytes.
        let a = build_gguf(Preset::Small, true);
        let b = build_gguf(Preset::Small, true);
        assert_eq!(a, b, "Small build with fixed seed must be deterministic");
    }

    #[test]
    fn xorshift_legal_code_is_in_range() {
        let mut rng = Xorshift64::new(42);
        for _ in 0..10_000 {
            let c = rng.next_legal_code();
            assert!(c <= 2, "code {} out of legal range 0..=2", c);
        }
    }
}
