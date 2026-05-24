use std::path::PathBuf;
use thiserror::Error;

/// Centralised error type for Project Willamette.
///
/// Every error variant carries a human-readable message. No variant silently
/// falls back to fake/random/synthetic behaviour.
#[derive(Error, Debug)]
pub enum WillametteError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("GGUF parse error: {0}")]
    GgufParse(String),

    #[error("Invalid GGUF magic (expected 0x46554747, got 0x{0:08X})")]
    InvalidMagic(u32),

    #[error("Unsupported GGUF version: {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),

    #[error("Unsupported architecture: \"{0}\"")]
    UnsupportedArchitecture(String),

    #[error("Missing required metadata key(s): {}", .0.join(", "))]
    MissingMetadata(Vec<String>),

    #[error("Unsupported GGML tensor type: {0}")]
    UnsupportedTensorType(u32),

    #[error("Unsupported tokenizer: {0}")]
    UnsupportedTokenizer(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Invalid model file: {0}")]
    InvalidPath(PathBuf),

    #[error("Tensor data out of bounds: tensor \"{name}\" requires bytes [{offset}..{end}), file size = {file_len}")]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        end: u64,
        file_len: u64,
    },

    #[error("String length overflow: {0}")]
    StringOverflow(u64),

    #[error("Metadata value type error for key \"{key}\": expected {expected}, got {actual}")]
    MetadataTypeMismatch {
        key: String,
        expected: String,
        actual: String,
    },
}
