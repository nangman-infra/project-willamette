//! Optional per-stage timing instrumentation for the decode hot path.
//!
//! Activated by building with `RUSTFLAGS="--cfg willamette_stage_timing"`.
//! When the flag is **off**, every public item in this module reduces to a
//! zero-cost no-op (the `time_stage!` macro just evaluates the body, and
//! [`reset`] / [`snapshot`] / [`report`] do nothing or return an empty
//! summary). The default release build is therefore byte-for-byte
//! unaffected — this is verified by the cfg-gated `#[cfg(...)]` blocks
//! below: nothing outside the `#[cfg(willamette_stage_timing)]` arm
//! references `Instant`, `RefCell`, or any TLS state.
//!
//! Usage from inside `cached_forward.rs`:
//!
//! ```ignore
//! use crate::model::stage_timing::time_stage;
//! let mut q = vec![0.0_f32; ctx.n_embd];
//! time_stage!("matvec_q", {
//!     bitlinear_i2s_matvec_f32(layer.attn_q, &x_norm, &mut q)?;
//! });
//! ```
//!
//! The macro accepts an arbitrary block; the *value* of the block is
//! returned, so `?` and early returns inside the block work as expected.
//!
//! The stage names are stable string slices (`&'static str`) so the
//! accumulator can key on pointer-equal `&str` cheaply. Names used by
//! the production hot path are listed at the top of `report()`.
//!
//! ## Concurrency model
//!
//! The accumulator is thread-local (`thread_local!` + `RefCell`).
//! The decode hot path is single-threaded per request (rayon is used
//! *inside* the BitLinear matvec, but its row workers do not call
//! `time_stage!`). If you ever wire `time_stage!` into a multi-threaded
//! scope, totals from the worker threads will not roll up automatically
//! — that is intentional, the per-thread totals stay isolated.
//!
//! ## Why a macro instead of a helper function
//!
//! A helper `fn time_stage<R>(name: &str, body: impl FnOnce() -> R) -> R`
//! would force every closure to be a `move ||` that re-borrows captured
//! `&mut` state, which fights the existing call sites that use `?` on
//! `Result<(), WillametteError>` directly. A macro inlines the block and
//! lets `?` propagate naturally.

#[cfg(willamette_stage_timing)]
use std::cell::RefCell;
#[cfg(willamette_stage_timing)]
use std::time::Duration;

/// One entry in the per-stage breakdown — total time spent in a stage
/// and how many times the stage was entered. Always defined (even when
/// the cfg is off) so calling code can hold a `Vec<StageSample>` type
/// without cfg-gating its own signatures.
#[derive(Debug, Clone)]
pub struct StageSample {
    pub name: &'static str,
    pub total: std::time::Duration,
    pub calls: u64,
}

#[cfg(willamette_stage_timing)]
thread_local! {
    /// `(name_ptr, total_ns, calls)` triples. A `Vec` (not a `HashMap`)
    /// because we have ≤ ~12 distinct stages, all using static string
    /// pointers, so linear scan is faster than hashing.
    static STAGES: RefCell<Vec<(&'static str, u128, u64)>> =
        const { RefCell::new(Vec::new()) };
}

/// Record one observation of `stage_name` taking `dt`. Internal helper
/// used by the `time_stage!` macro; not exposed when the cfg is off.
#[cfg(willamette_stage_timing)]
#[doc(hidden)]
pub fn __record(stage_name: &'static str, dt: Duration) {
    STAGES.with(|cell| {
        let mut v = cell.borrow_mut();
        for entry in v.iter_mut() {
            if std::ptr::eq(entry.0, stage_name) {
                entry.1 += dt.as_nanos();
                entry.2 += 1;
                return;
            }
        }
        v.push((stage_name, dt.as_nanos(), 1));
    });
}

/// Time the given block, accumulating into stage `$name`. When the
/// `willamette_stage_timing` cfg is off, expands to just the block —
/// no `Instant::now()` call, no atomic, no branch. The block's value
/// is returned, so `?` and early returns work inside.
#[macro_export]
macro_rules! __time_stage_impl {
    ($name:literal, $body:block) => {{
        #[cfg(willamette_stage_timing)]
        let __t0 = std::time::Instant::now();
        let __r = $body;
        #[cfg(willamette_stage_timing)]
        $crate::model::stage_timing::__record($name, __t0.elapsed());
        __r
    }};
}

pub use crate::__time_stage_impl as time_stage;

/// Reset all accumulated samples on the current thread. Call once
/// before the decode loop so warm-up + prefill don't pollute the
/// measurement.
pub fn reset() {
    #[cfg(willamette_stage_timing)]
    STAGES.with(|cell| cell.borrow_mut().clear());
}

/// Return a snapshot of all stages observed on the current thread
/// since the last [`reset`]. Empty when the cfg is off.
pub fn snapshot() -> Vec<StageSample> {
    #[cfg(willamette_stage_timing)]
    {
        STAGES.with(|cell| {
            cell.borrow()
                .iter()
                .map(|(n, ns, c)| StageSample {
                    name: n,
                    total: Duration::from_nanos(*ns as u64),
                    calls: *c,
                })
                .collect()
        })
    }
    #[cfg(not(willamette_stage_timing))]
    {
        Vec::new()
    }
}

/// Format a human-readable breakdown of the current per-stage totals,
/// sorted by share of the measured total descending. Each row carries
/// the stage label, total ms across all calls, call count, mean µs per
/// call, and percentage of the summed total.
///
/// When the cfg is off, returns a single line stating that
/// instrumentation was not enabled.
pub fn report(decode_steps: usize) -> String {
    let samples = snapshot();
    if samples.is_empty() {
        return String::from(
            "stage timing: instrumentation not compiled in \
             (rebuild with `RUSTFLAGS=\"--cfg willamette_stage_timing\"`).\n",
        );
    }

    let mut samples = samples;
    samples.sort_by(|a, b| b.total.cmp(&a.total));
    let grand_ns: u128 = samples.iter().map(|s| s.total.as_nanos()).sum();
    let mut out = String::new();
    out.push_str(
        "Stage breakdown (sum over all decode steps; mean per call in µs):\n",
    );
    out.push_str(
        "  stage                              total_ms     calls   mean_us    share\n",
    );
    out.push_str(
        "  --------------------------------  ---------  --------  --------  -------\n",
    );
    for s in &samples {
        let total_ms = s.total.as_secs_f64() * 1000.0;
        let mean_us = if s.calls > 0 {
            s.total.as_nanos() as f64 / s.calls as f64 / 1000.0
        } else {
            0.0
        };
        let share = if grand_ns > 0 {
            100.0 * s.total.as_nanos() as f64 / grand_ns as f64
        } else {
            0.0
        };
        out.push_str(&format!(
            "  {:<32}  {:>9.3}  {:>8}  {:>8.2}  {:>6.2}%\n",
            s.name, total_ms, s.calls, mean_us, share,
        ));
    }
    let grand_ms = grand_ns as f64 / 1.0e6;
    out.push_str(&format!(
        "  {:<32}  {:>9.3}\n",
        "TOTAL (sum of stages)", grand_ms,
    ));
    if decode_steps > 0 {
        out.push_str(&format!(
            "  ({:.3} ms per decode step across {} steps)\n",
            grand_ms / decode_steps as f64,
            decode_steps,
        ));
    }
    out
}
