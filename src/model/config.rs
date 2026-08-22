//! Architecture-neutral model hyper-parameters loaded from GGUF metadata.
//!
//! Source of truth: keys listed in [`docs/BITNET_FORWARD_PLAN.md`](../../docs/BITNET_FORWARD_PLAN.md)
//! §2, cross-checked against `src/llama.cpp:6117..6126` (which reads
//! `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS` and asserts `n_layer == 30 → MODEL_2B`)
//! at the pinned commit.
//!
//! This module does not infer values and does not invent defaults beyond a
//! single derived `head_dim`. The set of accepted `general.architecture`
//! strings is owned by [`crate::model::architecture::registry`] — today the
//! BitNet family (`bitnet-b1.58`, `bitnet-25`, `bitnet`), classic `llama`, and
//! pinned `qwen2`;
//! unknown strings still return `UnsupportedArchitecture`.

use std::collections::HashMap;

use crate::error::WillametteError;
use crate::gguf::reader::GgufValue;

#[derive(Debug, Clone)]
pub struct ModelConfig {
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

/// Compatibility alias retained for existing BitNet callers.
pub type BitNetConfig = ModelConfig;

impl ModelConfig {
    /// Canonical Microsoft 2B architecture string. Kept for callers
    /// that want to write a synthetic GGUF (`src/synth.rs`). Real
    /// loaders go through the registry (see `from_gguf_metadata`).
    pub const ARCHITECTURE: &'static str = "bitnet-b1.58";

    /// Read a `BitNetConfig` from parsed GGUF metadata. Resolves
    /// `general.architecture` through the
    /// [`crate::model::architecture::registry`] — accepts any string
    /// claimed by a registered impl (today: `bitnet-b1.58`,
    /// `bitnet-25`, `bitnet`, `llama`, `qwen2`). Returns `UnsupportedArchitecture`
    /// for anything else.
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
        let kv_dim = head_dim.checked_mul(head_count_kv).ok_or_else(|| {
            WillametteError::GgufParse("head_dim * head_count_kv overflow".to_string())
        })?;

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

    /// Read the classic, unscaled Llama GGUF subset supported by Phase III-B.
    pub fn from_llama_metadata_with_prefix(
        arch_string: &str,
        prefix: &str,
        meta: &HashMap<String, GgufValue>,
    ) -> Result<Self, WillametteError> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        let block_count = required_u32(meta, &key("block_count"))?;
        let embedding_length = required_u32(meta, &key("embedding_length"))?;
        let feed_forward_length = required_u32(meta, &key("feed_forward_length"))?;
        let context_length = required_u32(meta, &key("context_length"))?;
        let head_count = required_u32(meta, &key("attention.head_count"))?;
        let head_count_kv =
            optional_u32(meta, &key("attention.head_count_kv"))?.unwrap_or(head_count);
        let layer_norm_rms_epsilon = required_f32(meta, &key("attention.layer_norm_rms_epsilon"))?;

        if head_count == 0 || embedding_length % head_count != 0 {
            return Err(WillametteError::GgufParse(format!(
                "embedding_length {embedding_length} must be divisible by non-zero head_count {head_count}"
            )));
        }
        let head_dim = embedding_length / head_count;
        let rope_dimension_count =
            optional_u32(meta, &key("rope.dimension_count"))?.unwrap_or(head_dim);
        let rope_freq_base = optional_f32(meta, &key("rope.freq_base"))?.unwrap_or(10_000.0);
        let vocab_size = tokenizer_vocab_size(meta)?;

        if head_count_kv == 0 || head_count % head_count_kv != 0 {
            return Err(WillametteError::GgufParse(format!(
                "head_count {head_count} must be divisible by non-zero head_count_kv {head_count_kv}"
            )));
        }
        let kv_dim = head_dim.checked_mul(head_count_kv).ok_or_else(|| {
            WillametteError::GgufParse("head_dim * head_count_kv overflow".to_string())
        })?;
        if rope_dimension_count != head_dim {
            return Err(WillametteError::NotImplemented(format!(
                "Llama partial RoPE is not supported: rope.dimension_count={rope_dimension_count}, head_dim={head_dim}"
            )));
        }
        if rope_dimension_count % 2 != 0 {
            return Err(WillametteError::GgufParse(format!(
                "Llama rope.dimension_count must be even, got {rope_dimension_count}"
            )));
        }
        let scaling_prefix = key("rope.scaling");
        let scaling_keys = meta
            .keys()
            .filter(|name| name.starts_with(&scaling_prefix))
            .collect::<Vec<_>>();
        let explicit_unscaled = scaling_keys.len() == 1
            && scaling_keys[0].as_str() == key("rope.scaling.type")
            && meta
                .get(scaling_keys[0].as_str())
                .and_then(GgufValue::as_str)
                == Some("none");
        if !scaling_keys.is_empty() && !explicit_unscaled {
            return Err(WillametteError::NotImplemented(
                "Llama RoPE scaling is not supported".to_string(),
            ));
        }
        if block_count == 0
            || embedding_length == 0
            || feed_forward_length == 0
            || context_length == 0
            || vocab_size == 0
        {
            return Err(WillametteError::GgufParse(
                "Llama dimensions and vocabulary must be non-zero".to_string(),
            ));
        }
        if !layer_norm_rms_epsilon.is_finite() || layer_norm_rms_epsilon <= 0.0 {
            return Err(WillametteError::GgufParse(
                "Llama RMS epsilon must be finite and positive".to_string(),
            ));
        }
        if !rope_freq_base.is_finite() || rope_freq_base <= 0.0 {
            return Err(WillametteError::GgufParse(
                "Llama RoPE frequency base must be finite and positive".to_string(),
            ));
        }

        Ok(Self {
            architecture: arch_string.to_string(),
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

fn optional_u32(
    meta: &HashMap<String, GgufValue>,
    key: &str,
) -> Result<Option<u32>, WillametteError> {
    let Some(value) = meta.get(key) else {
        return Ok(None);
    };
    let value = value
        .as_u64()
        .ok_or_else(|| WillametteError::MissingMetadata(vec![format!("{} (u32/u64)", key)]))?;
    u32::try_from(value).map(Some).map_err(|_| {
        WillametteError::GgufParse(format!(
            "metadata key {key} value {value} does not fit in u32"
        ))
    })
}

fn required_f32(meta: &HashMap<String, GgufValue>, key: &str) -> Result<f32, WillametteError> {
    meta.get(key)
        .and_then(|v| v.as_f32())
        .ok_or_else(|| WillametteError::MissingMetadata(vec![format!("{} (f32)", key)]))
}

fn optional_f32(
    meta: &HashMap<String, GgufValue>,
    key: &str,
) -> Result<Option<f32>, WillametteError> {
    let Some(value) = meta.get(key) else {
        return Ok(None);
    };
    value
        .as_f32()
        .map(Some)
        .ok_or_else(|| WillametteError::MissingMetadata(vec![format!("{} (f32)", key)]))
}

fn tokenizer_vocab_size(meta: &HashMap<String, GgufValue>) -> Result<u32, WillametteError> {
    let tokens = meta
        .get("tokenizer.ggml.tokens")
        .and_then(GgufValue::as_string_array)
        .ok_or_else(|| {
            WillametteError::MissingMetadata(vec![
                "tokenizer.ggml.tokens (string array)".to_string()
            ])
        })?;
    u32::try_from(tokens.len()).map_err(|_| {
        WillametteError::GgufParse("tokenizer vocabulary does not fit in u32".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_metadata(embedding_length: u32) -> HashMap<String, GgufValue> {
        HashMap::from([
            ("llama.block_count".to_string(), GgufValue::Uint32(1)),
            (
                "llama.embedding_length".to_string(),
                GgufValue::Uint32(embedding_length),
            ),
            (
                "llama.feed_forward_length".to_string(),
                GgufValue::Uint32(16),
            ),
            ("llama.context_length".to_string(), GgufValue::Uint32(32)),
            (
                "llama.attention.head_count".to_string(),
                GgufValue::Uint32(2),
            ),
            (
                "llama.attention.head_count_kv".to_string(),
                GgufValue::Uint32(1),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon".to_string(),
                GgufValue::Float32(1e-5),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(vec![GgufValue::Str("token".to_string())]),
            ),
        ])
    }

    #[test]
    fn llama_accepts_explicitly_unscaled_rope() {
        let mut metadata = llama_metadata(8);
        metadata.insert(
            "llama.rope.scaling.type".to_string(),
            GgufValue::Str("none".to_string()),
        );
        assert!(ModelConfig::from_llama_metadata_with_prefix("llama", "llama", &metadata).is_ok());
    }

    #[test]
    fn llama_rejects_odd_rope_dimension_at_load_time() {
        let metadata = llama_metadata(6);
        assert!(matches!(
            ModelConfig::from_llama_metadata_with_prefix("llama", "llama", &metadata),
            Err(WillametteError::GgufParse(message)) if message.contains("must be even")
        ));
    }

    #[test]
    fn qwen2_metadata_uses_prefix_and_implied_full_rope() {
        let mut metadata = HashMap::from([
            (
                "general.architecture".to_string(),
                GgufValue::Str("qwen2".to_string()),
            ),
            ("qwen2.block_count".to_string(), GgufValue::Uint32(2)),
            ("qwen2.embedding_length".to_string(), GgufValue::Uint32(8)),
            (
                "qwen2.feed_forward_length".to_string(),
                GgufValue::Uint32(16),
            ),
            ("qwen2.context_length".to_string(), GgufValue::Uint32(32)),
            (
                "qwen2.attention.head_count".to_string(),
                GgufValue::Uint32(2),
            ),
            (
                "qwen2.attention.head_count_kv".to_string(),
                GgufValue::Uint32(1),
            ),
            (
                "qwen2.attention.layer_norm_rms_epsilon".to_string(),
                GgufValue::Float32(1e-6),
            ),
            (
                "qwen2.rope.freq_base".to_string(),
                GgufValue::Float32(1_000_000.0),
            ),
        ]);
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(vec![
                GgufValue::Str("a".to_string()),
                GgufValue::Str("b".to_string()),
            ]),
        );

        let config = ModelConfig::from_gguf_metadata(&metadata).expect("qwen2 config");
        assert_eq!(config.architecture, "qwen2");
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.rope_dimension_count, 4);
        assert_eq!(config.vocab_size, 2);
    }
}
