//! Profiling support.
//!
//! Recording is delegated to `xctrace` (Apple's CLI driver for Instruments)
//! because no public programmatic equivalent exists. Synchronization
//! is handled by `xctrace record --launch`, which blocks until the
//! recorder is ready before spawning the child — no race window.
//!
//! Phase boundaries are marked with [`interval`] guards. Today they are
//! stubs that just keep a wall-clock duration for the inner `profile-inner`
//! run; once the workload stabilizes we'll add an `os_signpost`-emitting
//! C shim so the labels land on the Instruments timeline alongside Metal
//! dispatches and CPU samples.

use std::time::Instant;

/// RAII guard for a profiled phase. Drops with no side-effects today;
/// later wired to `os_signpost_interval_end`.
pub struct Interval {
    #[allow(dead_code)]
    category: &'static str,
    #[allow(dead_code)]
    name: &'static str,
    #[allow(dead_code)]
    started: Instant,
}

/// Begin a profiled interval. The returned guard ends it on drop.
pub fn interval(category: &'static str, name: &'static str) -> Interval {
    Interval {
        category,
        name,
        started: Instant::now(),
    }
}
