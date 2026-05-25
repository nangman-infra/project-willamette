//! System monitor — periodic CPU + memory sampler for the TUI's
//! right-pane dashboard.
//!
//! Wraps the `sysinfo` crate. A background thread polls at 1 Hz
//! (slower polling under-samples per-core %; faster polling burns
//! CPU on retro hardware, which is what we're explicitly trying to
//! avoid). Each snapshot goes through an `mpsc::Sender<SysSnapshot>`
//! to the UI thread, which keeps the most recent one in its
//! `UiState`.
//!
//! Why a polling thread vs polling inside the UI loop:
//! * `sysinfo::System::refresh_cpu_usage()` needs ≥ 200 ms between
//!   refreshes for the per-core % to be meaningful. Doing that in
//!   the UI thread would block redraws.
//! * On retro hardware the UI thread already has plenty to do
//!   forwarding key events; system polling shouldn't compete.
//!
//! What this module deliberately does NOT do:
//! * No GPU monitoring. (We're CPU-only by thesis.)
//! * No process tree / per-thread breakdown. (Not actionable for
//!   the thesis owner.)
//! * No disk / network. (Off-thesis.)
//! * No temperature. `sysinfo` can read it on some platforms but
//!   it's unreliable on macOS and adds Linux dependencies; skip.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use sysinfo::{CpuRefreshKind, MemoryRefreshKind, Pid, ProcessRefreshKind, RefreshKind, System};

/// One sample of system + process state. Cheap to clone; UI keeps
/// the most recent one.
#[derive(Debug, Clone)]
pub struct SysSnapshot {
    /// CPU brand string, e.g. "Apple M4" or "Intel(R) Core(TM) i7-…".
    /// Captured once on the first poll — doesn't change across samples.
    pub cpu_brand: String,
    /// Architecture as reported by Rust. "aarch64", "x86_64", "x86", …
    pub arch: &'static str,
    /// Logical core count.
    pub logical_cores: usize,
    /// Physical core count (sysinfo best-effort; may equal logical).
    pub physical_cores: usize,
    /// Per-core usage percentages, length = logical_cores.
    pub per_core_pct: Vec<f32>,
    /// Aggregate CPU usage across all cores (0..100).
    pub overall_pct: f32,
    /// Usage of our own process (0..100*logical_cores; we normalize
    /// to 0..100 by dividing by logical_cores for display).
    pub process_pct_normalized: f32,
    /// Resident set size of our process, in bytes.
    pub process_rss_bytes: u64,
    /// Total system memory in bytes.
    pub total_mem_bytes: u64,
    /// Used system memory in bytes.
    pub used_mem_bytes: u64,
}

impl SysSnapshot {
    /// Build a snapshot that's "all zeros / unknown" — used by the
    /// UI before the first real sample arrives.
    pub fn placeholder() -> Self {
        Self {
            cpu_brand: "(unknown)".to_string(),
            arch: std::env::consts::ARCH,
            logical_cores: 0,
            physical_cores: 0,
            per_core_pct: Vec::new(),
            overall_pct: 0.0,
            process_pct_normalized: 0.0,
            process_rss_bytes: 0,
            total_mem_bytes: 0,
            used_mem_bytes: 0,
        }
    }
}

/// Spawn a daemon thread that polls system stats at `period` and
/// sends each `SysSnapshot` to `tx`. Returns an `AtomicBool` shutdown
/// flag — set it to `true` to stop the polling loop on the next tick.
pub fn spawn_sysmon(period: Duration, tx: Sender<SysSnapshot>) -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop);
    thread::Builder::new()
        .name("willamette-sysmon".to_string())
        .spawn(move || sysmon_loop(period, tx, stop_clone))
        .expect("sysmon thread spawn");
    stop
}

fn sysmon_loop(period: Duration, tx: Sender<SysSnapshot>, stop: Arc<AtomicBool>) {
    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything())
            .with_processes(ProcessRefreshKind::new().with_cpu().with_memory()),
    );
    // First refresh — establishes the baseline.
    sys.refresh_cpu_usage();
    // sysinfo needs at least one prior refresh to compute %.
    // Give it a short warm-up so the first emitted snapshot is real.
    thread::sleep(Duration::from_millis(250));
    sys.refresh_cpu_usage();
    sys.refresh_memory();

    let pid = Pid::from_u32(std::process::id());

    // Static / first-sample fields.
    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .unwrap_or_else(|| "(unknown)".into());
    let logical_cores = sys.cpus().len();
    let physical_cores = sys.physical_core_count().unwrap_or(logical_cores);

    while !stop.load(Ordering::Relaxed) {
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), false);

        let per_core_pct: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
        let overall_pct = sys.global_cpu_usage();

        let (proc_pct_raw, proc_rss) = sys
            .process(pid)
            .map(|p| (p.cpu_usage(), p.memory()))
            .unwrap_or((0.0, 0));
        // sysinfo's process cpu_usage returns 0..100 * logical_cores
        // (so 100% per fully-loaded core). Normalise so an idle
        // process is 0 and saturating all cores is 100.
        let process_pct_normalized = if logical_cores > 0 {
            proc_pct_raw / logical_cores as f32
        } else {
            proc_pct_raw
        };

        let snap = SysSnapshot {
            cpu_brand: cpu_brand.clone(),
            arch: std::env::consts::ARCH,
            logical_cores,
            physical_cores,
            per_core_pct,
            overall_pct,
            process_pct_normalized,
            process_rss_bytes: proc_rss,
            total_mem_bytes: sys.total_memory(),
            used_mem_bytes: sys.used_memory(),
        };
        if tx.send(snap).is_err() {
            // UI thread dropped the receiver — exit cleanly.
            break;
        }

        thread::sleep(period);
    }
}

/// One-shot synchronous snapshot — used by `dashboard` tests and by
/// the UI to populate the dashboard before the first async tick.
/// Costs a single `~250ms` sleep because sysinfo needs that long
/// between two CPU refreshes to compute a meaningful %.
pub fn snapshot_now() -> SysSnapshot {
    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything())
            .with_processes(ProcessRefreshKind::new().with_cpu().with_memory()),
    );
    sys.refresh_cpu_usage();
    thread::sleep(Duration::from_millis(250));
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), false);
    let logical_cores = sys.cpus().len();
    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .unwrap_or_else(|| "(unknown)".into());
    let physical_cores = sys.physical_core_count().unwrap_or(logical_cores);
    let per_core_pct: Vec<f32> = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
    let overall_pct = sys.global_cpu_usage();
    let (proc_pct_raw, proc_rss) = sys
        .process(pid)
        .map(|p| (p.cpu_usage(), p.memory()))
        .unwrap_or((0.0, 0));
    let process_pct_normalized = if logical_cores > 0 {
        proc_pct_raw / logical_cores as f32
    } else {
        proc_pct_raw
    };
    SysSnapshot {
        cpu_brand,
        arch: std::env::consts::ARCH,
        logical_cores,
        physical_cores,
        per_core_pct,
        overall_pct,
        process_pct_normalized,
        process_rss_bytes: proc_rss,
        total_mem_bytes: sys.total_memory(),
        used_mem_bytes: sys.used_memory(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_has_safe_defaults() {
        let p = SysSnapshot::placeholder();
        assert_eq!(p.logical_cores, 0);
        assert!(p.per_core_pct.is_empty());
        // arch must always be a real Rust target_arch string.
        assert!(!p.arch.is_empty());
    }

    #[test]
    fn snapshot_now_returns_real_data() {
        let snap = snapshot_now();
        // Logical cores should be >= 1 on any sane test host.
        assert!(snap.logical_cores >= 1);
        assert_eq!(snap.per_core_pct.len(), snap.logical_cores);
        // overall pct should be in [0, 100 * logical_cores] before
        // sysinfo normalises (it normalises internally, so 0..100).
        assert!(snap.overall_pct >= 0.0);
        // CPU brand should not be empty on real hosts.
        assert!(!snap.cpu_brand.is_empty());
        // RSS should be non-zero (we're a running process).
        assert!(snap.process_rss_bytes > 0);
        // System memory total should be > 0.
        assert!(snap.total_mem_bytes > 0);
    }

    #[test]
    fn process_pct_is_normalized_to_zero_to_hundred_range() {
        let snap = snapshot_now();
        // sysinfo's process cpu can briefly exceed 100% per core on
        // multi-core scheduling glitches, but our normalisation
        // divides by logical_cores. Allow some slack.
        assert!(snap.process_pct_normalized >= 0.0);
        assert!(
            snap.process_pct_normalized <= 110.0,
            "normalised proc pct should not blow past 100% by much; got {}",
            snap.process_pct_normalized
        );
    }
}
