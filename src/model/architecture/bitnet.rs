//! `BitNetArchitecture` — the BitNet-family impl of [`ModelArchitecture`].
//!
//! Covers three `general.architecture` strings, all of which share the
//! same forward graph + tensor name layout + hyperparameter set
//! (only the metadata-key prefix differs):
//!
//! | string | seen in | metadata prefix |
//! | --- | --- | --- |
//! | `bitnet-b1.58` | `microsoft/bitnet-b1.58-2B-4T-gguf` (our reference) | `bitnet-b1.58` |
//! | `bitnet-25`   | `jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF`, `Bifrost-AI/Bitnet-b1.58-Bifrost-SOL-2B-4T-gguf` | `bitnet-25` |
//! | `bitnet`      | Microsoft's 24 / 26-layer BitNet variants from the original paper | `bitnet` |
//!
//! Equivalence was verified by inspecting Aramis and Bifrost: 332
//! tensors, identical `blk.N.{role}.weight` names (including
//! `attn_sub_norm` + `ffn_sub_norm`), identical hyperparameter values,
//! identical packed byte size. See
//! [`docs/PHASE_III_ARCHITECTURE_RFC.md`](../../../docs/PHASE_III_ARCHITECTURE_RFC.md) § 1.

use std::collections::HashMap;

use super::ModelArchitecture;
use crate::error::WillametteError;
use crate::gguf::reader::GgufValue;
use crate::model::config::BitNetConfig;

/// Stateless marker type. The registry holds one boxed instance.
pub struct BitNetArchitecture;

impl ModelArchitecture for BitNetArchitecture {
    fn architecture_strings(&self) -> &'static [&'static str] {
        // Order is informational only — resolve() does a linear find.
        // First entry is the canonical Microsoft 2B reference.
        &["bitnet-b1.58", "bitnet-25", "bitnet"]
    }

    fn metadata_prefix<'a>(&self, arch_string: &'a str) -> &'a str {
        // For every BitNet variant the metadata prefix equals the
        // architecture string itself. Verified for `bitnet-b1.58`
        // (microsoft/2B) and `bitnet-25` (Aramis, Bifrost). The
        // `bitnet` variant is the Microsoft paper-era naming and is
        // assumed to follow the same convention; it will be verified
        // the first time a `bitnet`-string GGUF shows up.
        arch_string
    }

    fn config_from_meta(
        &self,
        arch_string: &str,
        meta: &HashMap<String, GgufValue>,
    ) -> Result<BitNetConfig, WillametteError> {
        BitNetConfig::from_gguf_metadata_with_prefix(
            arch_string,
            self.metadata_prefix(arch_string),
            meta,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_strings_contains_three_aliases() {
        let a = BitNetArchitecture;
        let s = a.architecture_strings();
        assert!(s.contains(&"bitnet-b1.58"));
        assert!(s.contains(&"bitnet-25"));
        assert!(s.contains(&"bitnet"));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn prefix_equals_architecture_string_for_each_alias() {
        let a = BitNetArchitecture;
        for s in a.architecture_strings() {
            assert_eq!(a.metadata_prefix(s), *s);
        }
    }
}
