//! Qwen2/Qwen2.5 GGUF architecture surface.

use std::collections::HashMap;

use super::{ForwardVariant, LayerTensorRole, ModelArchitecture};
use crate::error::WillametteError;
use crate::gguf::reader::GgufValue;
use crate::model::config::ModelConfig;

pub struct Qwen2Architecture;

impl ModelArchitecture for Qwen2Architecture {
    fn architecture_strings(&self) -> &'static [&'static str] {
        &["qwen2"]
    }

    fn metadata_prefix<'a>(&self, _arch_string: &'a str) -> &'a str {
        "qwen2"
    }

    fn config_from_meta(
        &self,
        arch_string: &str,
        meta: &HashMap<String, GgufValue>,
    ) -> Result<ModelConfig, WillametteError> {
        ModelConfig::from_llama_metadata_with_prefix(
            arch_string,
            self.metadata_prefix(arch_string),
            meta,
        )
    }

    fn layer_tensor_roles(&self) -> &'static [LayerTensorRole] {
        use LayerTensorRole::{
            AttnK, AttnKBias, AttnNorm, AttnOutput, AttnQ, AttnQBias, AttnV, AttnVBias, FfnDown,
            FfnGate, FfnNorm, FfnUp,
        };
        &[
            AttnNorm, AttnQ, AttnQBias, AttnK, AttnKBias, AttnV, AttnVBias, AttnOutput, FfnNorm,
            FfnGate, FfnUp, FfnDown,
        ]
    }

    fn forward_variant(&self) -> ForwardVariant {
        ForwardVariant::Qwen2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_qwen2_metadata_and_bias_contract() {
        let architecture = Qwen2Architecture;
        assert_eq!(architecture.metadata_prefix("qwen2"), "qwen2");
        assert_eq!(architecture.forward_variant(), ForwardVariant::Qwen2);
        for role in [
            LayerTensorRole::AttnQBias,
            LayerTensorRole::AttnKBias,
            LayerTensorRole::AttnVBias,
        ] {
            assert!(architecture.layer_tensor_roles().contains(&role));
        }
    }
}
