//! Classic Llama/Llama-2 GGUF architecture surface.

use std::collections::HashMap;

use super::{ForwardVariant, LayerTensorRole, ModelArchitecture};
use crate::error::WillametteError;
use crate::gguf::reader::GgufValue;
use crate::model::config::ModelConfig;

pub struct LlamaArchitecture;

impl ModelArchitecture for LlamaArchitecture {
    fn architecture_strings(&self) -> &'static [&'static str] {
        &["llama"]
    }

    fn metadata_prefix<'a>(&self, _arch_string: &'a str) -> &'a str {
        "llama"
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
            AttnK, AttnNorm, AttnOutput, AttnQ, AttnV, FfnDown, FfnGate, FfnNorm, FfnUp,
        };
        &[
            AttnNorm, AttnQ, AttnK, AttnV, AttnOutput, FfnNorm, FfnGate, FfnUp, FfnDown,
        ]
    }

    fn forward_variant(&self) -> ForwardVariant {
        ForwardVariant::VanillaLlama
    }
}
