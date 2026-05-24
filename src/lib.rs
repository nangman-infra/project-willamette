//! Project Willamette — Rust-native inference runtime for real BitNet / 1.58-bit
//! GGUF models.
//!
//! Stage 1 scope: real GGUF inspection only.
//!   * mmap-backed zero-copy file open
//!   * GGUF magic/version/header parsing
//!   * metadata key-value parsing (full GGUF v2/v3 value-type matrix)
//!   * tensor directory parsing with raw `u32` ggml_type preserved
//!
//! Out of scope for Stage 1 (intentionally not exposed):
//!   * tokenizer encode/decode
//!   * forward pass / generation
//!   * CPU SIMD kernels (AVX2/SSE2/NEON)
//!
//! Re-enabling any of those requires verifying the real
//! `microsoft/bitnet-b1.58-2B-4T-gguf` file first.

pub mod error;
pub mod gguf;
pub mod memory;
pub mod model;
pub mod tokenizer;
