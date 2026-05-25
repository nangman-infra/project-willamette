//! Runtime CPU dispatch — single source of truth for which BitLinear
//! kernel runs on this host, and how that choice is described to the
//! user (TUI dashboard, `--cpu-info` output, logs).
//!
//! ## Why a separate module
//!
//! Before this module existed, two pieces of code asked the same
//! question independently:
//!
//! 1. `src/model/bitlinear.rs::bitlinear_i2s_matvec_f32` — picked the
//!    runtime kernel via `cfg(target_arch)` + `is_aarch64_feature_detected!`.
//! 2. `src/chat/tui.rs::initial_dashboard_state` — built the
//!    "aarch64 NEON" / "x86_64 scalar" / etc. label by repeating the
//!    same arch checks.
//!
//! That meant the dashboard could (and did) display a kernel name
//! that wasn't strictly what `bitlinear.rs` was about to call — for
//! example, dashboard said "x86_64 scalar" while no x86_64 kernel
//! existed at all. With both call sites going through
//! [`active_kernel()`] there's exactly one decision to keep correct.
//!
//! ## Detection cost
//!
//! Runtime feature detection (`std::arch::is_*_feature_detected!`)
//! reads CPU-ID once per process and caches the result in stdlib —
//! it's cheap. We further memoise the *kernel choice* itself via
//! [`std::sync::OnceLock`] so [`select_kernel`] is a single atomic
//! pointer load on the hot path.
//!
//! ## What's intentionally NOT here
//!
//! The actual SIMD kernel implementations live next to the scalar
//! reference in `bitlinear.rs` / `bitlinear_neon.rs`. This module is
//! only concerned with *picking* and *naming*. Adding a Stage 6-B
//! SSE2 kernel later means adding one `Kernel` variant + one arm in
//! [`select_kernel`] — every caller picks it up for free.

use std::sync::OnceLock;

/// Which BitLinear matvec kernel will run on this host.
///
/// Variants reflect both the architecture and the SIMD path within
/// that architecture. `Scalar` is the always-available fallback used
/// when no specialised kernel is detected (or compiled in) for the
/// current target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Reference scalar implementation. Always works. Used on:
    ///   * any architecture without a SIMD kernel compiled in
    ///   * arches where SIMD was compiled but the CPU reports the
    ///     required feature is absent.
    Scalar,
    /// `bitlinear_neon::bitlinear_i2s_matvec_f32_neon` — requires
    /// aarch64 + NEON (universally true on Apple Silicon, Cortex-A57+).
    AArch64Neon,
    /// Stage 6-B target: 32-bit or 64-bit x86 with SSE2. **No code
    /// path yet** — Phase 3 will fill it. Selected if the host
    /// reports SSE2 but, until the kernel exists, dispatch falls back
    /// to Scalar with this variant returned only for the dashboard
    /// label.
    X86Sse2,
}

impl Kernel {
    /// Short human-readable label for the dashboard / logs.
    /// Format: `"<arch> <variant>"`, e.g. `"aarch64 NEON"`.
    pub fn label(self) -> &'static str {
        match self {
            Kernel::Scalar => match std::env::consts::ARCH {
                "aarch64" => "aarch64 scalar",
                "x86_64" => "x86_64 scalar",
                "x86" => "i686 scalar",
                _ => "scalar (generic)",
            },
            Kernel::AArch64Neon => "aarch64 NEON",
            Kernel::X86Sse2 => match std::env::consts::ARCH {
                "x86_64" => "x86_64 SSE2",
                "x86" => "i686 SSE2",
                _ => "x86 SSE2",
            },
        }
    }
}

/// Pick the best kernel for this host. Result is cached after the
/// first call; subsequent calls are an atomic load. Safe to call from
/// any thread.
pub fn active_kernel() -> Kernel {
    static CHOICE: OnceLock<Kernel> = OnceLock::new();
    *CHOICE.get_or_init(select_kernel)
}

/// Per-feature SIMD detection results — used by the dashboard for the
/// kernel-features (●/○) display. Order is meaningful (top-to-bottom
/// in the UI).
pub fn detected_features() -> Vec<(&'static str, bool)> {
    // `mut` is conditionally used — on aarch64 / x86 / x86_64 we
    // `push` into `out`; on armv7 / generic the cfg blocks below
    // expand to nothing and `out` is returned empty. The `#[allow]`
    // keeps `RUSTFLAGS=-D warnings` builds happy on targets where
    // no SIMD slots are advertised.
    #[allow(unused_mut)]
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    #[cfg(target_arch = "aarch64")]
    {
        out.push(("neon", std::arch::is_aarch64_feature_detected!("neon")));
        // dotprod (SDOT) is stable in std::arch detection on
        // recent toolchains but the *intrinsic* (`vdotq_s32`) is
        // gated behind unstable `stdarch_neon_dotprod`. The detection
        // here is purely informational.
        out.push((
            "dotprod",
            std::arch::is_aarch64_feature_detected!("dotprod"),
        ));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        out.push(("sse2", std::arch::is_x86_feature_detected!("sse2")));
        out.push(("sse4.1", std::arch::is_x86_feature_detected!("sse4.1")));
        out.push(("avx2", std::arch::is_x86_feature_detected!("avx2")));
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        // No SIMD slots advertised on this arch yet.
    }

    out
}

/// Internal: actually decide which kernel to run. Called once per
/// process via [`active_kernel`]'s OnceLock.
fn select_kernel() -> Kernel {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Kernel::AArch64Neon;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Stage 6-B: route to the SSE2 BitLinear kernel when the host
        // reports the feature. Pentium-M / antiX i686 falls here.
        // bitlinear::bitlinear_i2s_matvec_f32 holds the matching
        // `Kernel::X86Sse2` arm.
        if std::arch::is_x86_feature_detected!("sse2") {
            return Kernel::X86Sse2;
        }
    }

    Kernel::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_kernel_is_stable_across_calls() {
        // Important for OnceLock correctness: same answer every time.
        let first = active_kernel();
        for _ in 0..16 {
            assert_eq!(active_kernel(), first);
        }
    }

    #[test]
    fn label_is_non_empty() {
        // Every Kernel variant must produce a non-empty label —
        // otherwise the dashboard would render a blank line.
        for k in [Kernel::Scalar, Kernel::AArch64Neon, Kernel::X86Sse2] {
            assert!(!k.label().is_empty(), "empty label for {:?}", k);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_selects_neon_when_available() {
        // Apple Silicon and any Cortex-A57+ always reports NEON.
        assert!(std::arch::is_aarch64_feature_detected!("neon"));
        assert_eq!(active_kernel(), Kernel::AArch64Neon);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_features_include_sse2_slot() {
        let names: Vec<&str> = detected_features()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert!(names.contains(&"sse2"));
    }
}
