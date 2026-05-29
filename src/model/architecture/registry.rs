//! Global registry of [`ModelArchitecture`] impls.
//!
//! Initialised lazily on first call to [`resolve`]. Today it contains
//! exactly one entry (`BitNetArchitecture`). When Llama 2 / Phi-3 /
//! Gemma land — see [`docs/PHASE_III_ARCHITECTURE_RFC.md`](../../../docs/PHASE_III_ARCHITECTURE_RFC.md) §
//! 5.4 — adding them is one line in [`registry()`].

use std::sync::OnceLock;

use super::{BitNetArchitecture, ModelArchitecture};

/// All architectures the runtime can read GGUFs for. One entry per
/// *family* (same forward graph). The slice is `'static` once
/// initialised.
fn registry() -> &'static [Box<dyn ModelArchitecture>] {
    static REGISTRY: OnceLock<Vec<Box<dyn ModelArchitecture>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let v: Vec<Box<dyn ModelArchitecture>> = vec![Box::new(BitNetArchitecture)];
        v
    })
}

/// Look up the architecture impl that claims this `general.architecture`
/// string. Returns `None` for unknown architectures — the caller is
/// expected to raise `UnsupportedArchitecture` at that point so the
/// failure surface stays GGUF-loader-shaped, not "panicked in registry."
pub fn resolve(arch_string: &str) -> Option<&'static dyn ModelArchitecture> {
    registry()
        .iter()
        .map(|a| a.as_ref())
        .find(|a| a.architecture_strings().contains(&arch_string))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_at_least_one_arch() {
        assert!(!registry().is_empty());
    }

    /// OnceLock stability — same pointer across calls.
    #[test]
    fn registry_is_stable_across_calls() {
        let first = registry().as_ptr();
        for _ in 0..16 {
            assert_eq!(registry().as_ptr(), first);
        }
    }

    #[test]
    fn resolve_returns_same_impl_for_all_aliases_in_a_family() {
        // Sanity check that all BitNet aliases hit the same Box.
        let a = resolve("bitnet-b1.58").unwrap();
        let b = resolve("bitnet-25").unwrap();
        let c = resolve("bitnet").unwrap();
        // We can't directly compare `&'static dyn` pointers safely
        // without `Any`, but `architecture_strings` slice identity is
        // a reliable proxy — all three must yield the same slice
        // address since they come from the same impl method.
        let sa = a.architecture_strings().as_ptr();
        let sb = b.architecture_strings().as_ptr();
        let sc = c.architecture_strings().as_ptr();
        assert_eq!(sa, sb);
        assert_eq!(sa, sc);
    }
}
