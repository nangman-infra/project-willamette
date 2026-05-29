//! Phase III — Model architecture registry.
//!
//! Lets us read GGUFs whose `general.architecture` string is a known
//! alias of an architecture we *do* support. The trait + registry is
//! deliberately minimal: it's the smallest abstraction that names the
//! seam, not a kitchen sink.
//!
//! Today this carries the BitNet family
//! (`bitnet-b1.58` + `bitnet-25` + `bitnet`) under a single
//! [`ModelArchitecture`] impl. Adding Llama 2 / Phi / Gemma later means
//! adding one impl in [`bitnet::BitNetArchitecture`]'s sibling files —
//! see [`docs/PHASE_III_ARCHITECTURE_RFC.md`](../../../docs/PHASE_III_ARCHITECTURE_RFC.md).
//!
//! Notable scope decisions for Phase III step 2 (this commit):
//!
//! * `BitNetConfig` is the only config type. We don't introduce a
//!   `ModelConfig` enum yet — that's needed when a *non-BitNet* arch
//!   lands (RFC step 5 / Phase III-B). Trait method returns
//!   `BitNetConfig` directly. When the second config type appears
//!   it becomes an associated type or an enum then.
//! * `LayerTensorRole` and `ForwardVariant` are NOT defined here. RFC
//!   steps 3 and 4 introduce them. Building them now without a second
//!   forward graph to validate against would be the empty-cathedral
//!   shape that [[feedback-principled-design]] warns about ("structural
//!   form has one real impl + one reserved entry point, not zero").

pub mod bitnet;
pub mod registry;

use std::collections::HashMap;

use crate::error::WillametteError;
use crate::gguf::reader::GgufValue;
use crate::model::config::BitNetConfig;

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

    /// Read this arch's `BitNetConfig` (today's only config type)
    /// from a parsed GGUF metadata map, given the chosen
    /// architecture string. The impl is responsible for using the
    /// right key prefix.
    fn config_from_meta(
        &self,
        arch_string: &str,
        meta: &HashMap<String, GgufValue>,
    ) -> Result<BitNetConfig, WillametteError>;
}

pub use bitnet::BitNetArchitecture;
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
        assert!(resolve("llama").is_none());
        assert!(resolve("phi3").is_none());
        assert!(resolve("").is_none());
    }
}
