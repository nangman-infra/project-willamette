//! Project Willamette - Rust-native, mmap-backed inference runtime for the
//! official BitNet b1.58 2B I2_S GGUF model.
//!
//! v0.10.0 includes GGUF parsing, byte-level BPE tokenization, model graph
//! loading, causal forward and generation with an i8 KV cache, sampling,
//! interactive chat/TUI support, and runtime-selected CPU kernels.

pub mod chat;
pub mod error;
pub mod gguf;
pub mod memory;
pub mod model;
pub mod synth;
pub mod tokenizer;
