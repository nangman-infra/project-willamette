//! BitNetConfig — model hyper-parameters loaded purely from GGUF metadata.
//!
//! Source of truth: keys listed in [`docs/BITNET_FORWARD_PLAN.md`](../../docs/BITNET_FORWARD_PLAN.md)
//! §2, cross-checked against `src/llama.cpp:6117..6126` (which reads
//! `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS` and asserts `n_layer == 30 → MODEL_2B`)
//! at the pinned commit.
//!
//! This module does not infer values and does not invent defaults beyond a
//! single derived `head_dim`. The set of accepted `general.architecture`
//! strings is owned by [`crate::model::architecture::registry`] — today
//! the BitNet family (`bitnet-b1.58`, `bitnet-25`, `bitnet`); unknown
//! strings still return `UnsupportedArchitecture`.

use std::collections::HashMap;

use crate::error::WillametteError;
use crate::gguf::reader::GgufValue;

#[derive(Debug, Clone)]
pub struct BitNetConfig {
    pub architecture: String,

    pub block_count: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub context_length: u32,

    pub head_count: u32,
    pub head_count_kv: u32,
    /// Derived: `embedding_length / head_count`. Verified against
    /// `rope.dimension_count` at load time (they must be equal for the
    /// `build_bitnet_158` full-head RoPE path).
    pub head_dim: u32,
    /// Derived: `head_dim * head_count_kv`. Total K (and V) projection
    /// width per token under GQA.
    pub kv_dim: u32,

    pub layer_norm_rms_epsilon: f32,
    pub rope_dimension_count: u32,
    pub rope_freq_base: f32,

    pub vocab_size: u32,
}

impl BitNetConfig {
    /// Canonical Microsoft 2B architecture string. Kept for callers
    /// that want to write a synthetic GGUF (`src/synth.rs`). Real
    /// loaders go through the registry (see `from_gguf_metadata`).
    pub const ARCHITECTURE: &'static str = "bitnet-b1.58";

    /// Read a `BitNetConfig` from parsed GGUF metadata. Resolves
    /// `general.architecture` through the
    /// [`crate::model::architecture::registry`] — accepts any string
    /// claimed by a registered impl (today: `bitnet-b1.58`,
    /// `bitnet-25`, `bitnet`). Returns `UnsupportedArchitecture` for
    /// anything else.
    pub fn from_gguf_metadata(meta: &HashMap<String, GgufValue>) -> Result<Self, WillametteError> {
        let arch_string = required_str(meta, "general.architecture")?.to_string();
        let arch = crate::model::architecture::resolve(&arch_string).ok_or(
            WillametteError::UnsupportedArchitecture(arch_string.clone()),
        )?;
        arch.config_from_meta(&arch_string, meta)
    }

    /// Read a `BitNetConfig` using an explicit metadata-key prefix.
    /// Used by the architecture trait — every `ModelArchitecture`
    /// impl in the BitNet family delegates here after deciding which
    /// prefix to apply (`bitnet-b1.58.*`, `bitnet-25.*`, `bitnet.*`).
    ///
    /// `arch_string` is the value of `general.architecture` and is
    /// stored on the returned `BitNetConfig` so downstream code can
    /// see which alias was loaded.
    pub fn from_gguf_metadata_with_prefix(
        arch_string: &str,
        prefix: &str,
        meta: &HashMap<String, GgufValue>,
    ) -> Result<Self, WillametteError> {
        let arch = arch_string.to_string();
        let key = |suffix: &str| format!("{prefix}.{suffix}");

        let block_count = required_u32(meta, &key("block_count"))?;
        let embedding_length = required_u32(meta, &key("embedding_length"))?;
        let feed_forward_length = required_u32(meta, &key("feed_forward_length"))?;
        let context_length = required_u32(meta, &key("context_length"))?;

        let head_count = required_u32(meta, &key("attention.head_count"))?;
        let head_count_kv = required_u32(meta, &key("attention.head_count_kv"))?;
        let layer_norm_rms_epsilon = required_f32(meta, &key("attention.layer_norm_rms_epsilon"))?;

        let rope_dimension_count = required_u32(meta, &key("rope.dimension_count"))?;
        let rope_freq_base = required_f32(meta, &key("rope.freq_base"))?;

        let vocab_size = required_u32(meta, &key("vocab_size"))?;

        // Cross-checks (cite REFERENCE_COMMIT.md if any of these ever fail).
        if head_count == 0 {
            return Err(WillametteError::GgufParse(
                "head_count must be > 0".to_string(),
            ));
        }
        if embedding_length % head_count != 0 {
            return Err(WillametteError::GgufParse(format!(
                "embedding_length {} not divisible by head_count {}",
                embedding_length, head_count
            )));
        }
        let head_dim = embedding_length / head_count;

        if head_count_kv == 0 {
            return Err(WillametteError::GgufParse(
                "head_count_kv must be > 0".to_string(),
            ));
        }
        if head_count % head_count_kv != 0 {
            return Err(WillametteError::GgufParse(format!(
                "head_count {} not divisible by head_count_kv {} (GQA ratio must be integer)",
                head_count, head_count_kv
            )));
        }
        let kv_dim = head_dim * head_count_kv;

        // build_bitnet_158 asserts n_embd_head == hparams.n_rot
        if rope_dimension_count != head_dim {
            return Err(WillametteError::GgufParse(format!(
                "rope.dimension_count ({}) must equal head_dim ({}); \
                 BitNet b1.58 build_bitnet_158 asserts n_embd_head == n_rot",
                rope_dimension_count, head_dim
            )));
        }

        if block_count == 0 {
            return Err(WillametteError::GgufParse(
                "block_count must be > 0".to_string(),
            ));
        }
        if vocab_size == 0 {
            return Err(WillametteError::GgufParse(
                "vocab_size must be > 0".to_string(),
            ));
        }

        Ok(Self {
            architecture: arch,
            block_count,
            embedding_length,
            feed_forward_length,
            context_length,
            head_count,
            head_count_kv,
            head_dim,
            kv_dim,
            layer_norm_rms_epsilon,
            rope_dimension_count,
            rope_freq_base,
            vocab_size,
        })
    }
}

fn required_str<'a>(
    meta: &'a HashMap<String, GgufValue>,
    key: &str,
) -> Result<&'a str, WillametteError> {
    meta.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| WillametteError::MissingMetadata(vec![format!("{} (string)", key)]))
}

fn required_u32(meta: &HashMap<String, GgufValue>, key: &str) -> Result<u32, WillametteError> {
    let v = meta
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| WillametteError::MissingMetadata(vec![format!("{} (u32/u64)", key)]))?;
    if v > u32::MAX as u64 {
        return Err(WillametteError::GgufParse(format!(
            "metadata key {} value {} does not fit in u32",
            key, v
        )));
    }
    Ok(v as u32)
}

fn required_f32(meta: &HashMap<String, GgufValue>, key: &str) -> Result<f32, WillametteError> {
    meta.get(key)
        .and_then(|v| v.as_f32())
        .ok_or_else(|| WillametteError::MissingMetadata(vec![format!("{} (f32)", key)]))
}
