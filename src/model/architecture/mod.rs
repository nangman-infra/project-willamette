//! Phase III — Model architecture registry.
//!
//! Lets us read GGUFs whose `general.architecture` string is a known
//! alias of an architecture we *do* support. The trait + registry is
//! deliberately minimal: it's the smallest abstraction that names the
//! seam, not a kitchen sink.
//!
//! Today this carries the BitNet family (`bitnet-b1.58` + `bitnet-25` +
//! `bitnet`) and the narrow classic Llama family. Adding Phi / Gemma later
//! means adding sibling implementations; see
//! [`docs/PHASE_III_ARCHITECTURE_RFC.md`](../../../docs/PHASE_III_ARCHITECTURE_RFC.md).
//!
//! Notable scope decisions:
//!
//! * [`ModelConfig`] stores the shared BitNet and classic-Llama hyperparameters.
//! * [`LayerTensorRole`] and [`ForwardVariant`] name the graph seams from
//!   RFC steps 3 and 4. Both current variants execute; future variants must
//!   fail with `NotImplemented` before running unsupported kernels.

pub mod bitnet;
pub mod llama;
pub mod registry;

use std::collections::HashMap;

use crate::error::WillametteError;
use crate::gguf::reader::GgufValue;
use crate::model::config::ModelConfig;

/// Per-layer GGUF tensors used by the currently planned transformer graphs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum LayerTensorRole {
    AttnNorm,
    AttnSubNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnNorm,
    FfnSubNorm,
    FfnGate,
    FfnUp,
    FfnDown,
}

impl LayerTensorRole {
    /// GGUF tensor-name component in `blk.{layer}.{suffix}.weight`.
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::AttnNorm => "attn_norm",
            Self::AttnSubNorm => "attn_sub_norm",
            Self::AttnQ => "attn_q",
            Self::AttnK => "attn_k",
            Self::AttnV => "attn_v",
            Self::AttnOutput => "attn_output",
            Self::FfnNorm => "ffn_norm",
            Self::FfnSubNorm => "ffn_sub_norm",
            Self::FfnGate => "ffn_gate",
            Self::FfnUp => "ffn_up",
            Self::FfnDown => "ffn_down",
        }
    }
}

/// Transformer-block topology selected by an architecture family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ForwardVariant {
    BitNetSubNorm,
    /// Classic pre-norm Llama topology without BitNet sub-norms.
    VanillaLlama,
}

/// One impl per architecture *family*. A family is "models whose
/// forward graph is identical, even if their `general.architecture`
/// string differs."
///
/// Object-safe: stored as `Box<dyn ModelArchitecture>` in the registry.
/// `Send + Sync + 'static` because the registry is global.
pub trait ModelArchitecture: Send + Sync + 'static {
    /// Every `general.architecture` string this impl claims. BitNet
    /// impl claims `["bitnet-b1.58", "bitnet-25", "bitnet"]`.
    fn architecture_strings(&self) -> &'static [&'static str];

    /// The GGUF metadata key prefix for this arch_string. For the
    /// BitNet family the prefix is literally the architecture string
    /// (`bitnet-b1.58.block_count`, `bitnet-25.block_count`, ...).
    /// For future Llama support the same metadata field is
    /// `llama.block_count` regardless of which alias was used. So the
    /// trait passes the chosen alias in and lets the impl decide.
    fn metadata_prefix<'a>(&self, arch_string: &'a str) -> &'a str;

    /// Read this architecture's shared model config from a parsed GGUF metadata
    /// map, given the chosen
    /// architecture string. The impl is responsible for using the
    /// right key prefix.
    fn config_from_meta(
        &self,
        arch_string: &str,
        meta: &HashMap<String, GgufValue>,
    ) -> Result<ModelConfig, WillametteError>;

    /// Tensor roles present in every layer of this architecture family.
    fn layer_tensor_roles(&self) -> &'static [LayerTensorRole];

    /// Transformer-block topology used by this architecture family.
    fn forward_variant(&self) -> ForwardVariant;
}

pub use bitnet::BitNetArchitecture;
pub use llama::LlamaArchitecture;
pub use registry::resolve;

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry always resolves the canonical Microsoft 2B
    /// architecture string. If this breaks, no model loads.
    #[test]
    fn canonical_bitnet_b1_58_is_resolved() {
        let arch = resolve("bitnet-b1.58").expect("must resolve");
        assert!(arch.architecture_strings().contains(&"bitnet-b1.58"));
        assert_eq!(arch.metadata_prefix("bitnet-b1.58"), "bitnet-b1.58");
    }

    /// The community-fine-tune alias resolves to the same impl.
    /// Without this, Aramis / Bifrost stay rejected.
    #[test]
    fn bitnet_25_alias_is_resolved() {
        let arch = resolve("bitnet-25").expect("must resolve");
        assert!(arch.architecture_strings().contains(&"bitnet-25"));
        assert_eq!(arch.metadata_prefix("bitnet-25"), "bitnet-25");
    }

    /// Unknown arches stay rejected (otherwise we silently accept
    /// anything and crash later inside the forward graph).
    #[test]
    fn unknown_architecture_returns_none() {
        assert!(resolve("phi3").is_none());
        assert!(resolve("").is_none());
    }
}
