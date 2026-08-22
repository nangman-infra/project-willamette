//! Project Willamette - Rust-native, mmap-backed CPU inference runtime for
//! BitNet I2_S and classic Llama F16/Q4_0/Q8_0 GGUF models.
//!
//! v0.13.0 includes GGUF parsing, GPT-2 and SentencePiece BPE tokenization, model graph
//! loading, causal forward and generation with an i8 KV cache, sampling,
//! interactive chat/TUI support, and runtime-selected CPU kernels.

pub mod chat;
pub mod error;
pub mod gguf;
pub mod memory;
pub mod model;
pub mod synth;
pub mod tokenizer;
