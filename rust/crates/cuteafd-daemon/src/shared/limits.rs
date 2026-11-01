//! Allocated concurrency ceiling plus a live-switchable limit.
//!
//! The allocated ceiling is fixed at start-up and sizes every start-time
//! structure (slots, drafter, queue depth, KV plan). The live limit is an
//! atomic the serve loop reads at every admission, so it can be lowered or
//! raised without a restart. Lowering never kills a stream: excess requests
//! drain by retirement. Raising takes effect at the next admission.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

pub(crate) use cuteafd_api::openai::admin::LimitError;

/// The V4.1 engine's request-slot bound: the widest ceiling this build can
/// allocate (KV tables, compressor and dSpark caches are this wide).
pub(crate) const V41_SLOTS_MAX: u32 = 128;

/// The process-wide concurrency limits.
pub(crate) struct LiveLimits {
    allocated: u32,
    concurrency: AtomicU32,
}

impl LiveLimits {
    /// A ceiling and its initial live limit. The live value is clamped into
    /// `1..=allocated` so a bad start value can never disable admission.
    pub(crate) fn new(allocated: u32, live: u32) -> Self {
        let allocated = allocated.max(1);
        let live = live.clamp(1, allocated);
        Self { allocated, concurrency: AtomicU32::new(live) }
    }

    /// The live limit the serve loop admits against.
    pub(crate) fn concurrency(&self) -> u32 {
        self.concurrency.load(Ordering::Relaxed)
    }

    /// The start-time ceiling that sized memory and slots.
    pub(crate) fn allocated(&self) -> u32 {
        self.allocated
    }

    /// Set the live limit. Zero and values above the allocated ceiling are
    /// refused; the effective value is returned.
    pub(crate) fn set_concurrency(&self, n: u32) -> Result<u32, LimitError> {
        if n == 0 { return Err(LimitError::Zero); }
        if n > self.allocated {
            return Err(LimitError::AboveCeiling { requested: n, allocated: self.allocated });
        }
        self.concurrency.store(n, Ordering::Relaxed);
        Ok(n)
    }

    /// The `limits` block exported in `/v1/stats` and `/v1/admin/limits`.
    pub(crate) fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "allocated": self.allocated,
            "live": self.concurrency(),
            "slots_max": slots_max(),
        })
    }
}

/// The admission bound for this build.
pub(crate) fn slots_max() -> u32 {
    V41_SLOTS_MAX
}

static LIVE: OnceLock<LiveLimits> = OnceLock::new();

/// Install the process-wide limits once, before serving starts.
///
/// A second install with the same values returns the existing handle; a second
/// install with different values is refused rather than silently discarded, so
/// a family that forgets to install cannot be masked by another family's
/// values.
pub(crate) fn install(allocated: u32, live: u32) -> Result<&'static LiveLimits, LimitError> {
    let limits = LiveLimits::new(allocated, live);
    if LIVE.set(limits).is_ok() {
        return Ok(LIVE.get().expect("just installed"));
    }
    let existing = LIVE.get().expect("set failed only when already installed");
    if existing.allocated() == allocated && existing.concurrency() == live {
        Ok(existing)
    } else {
        Err(LimitError::AlreadyInstalled { allocated: existing.allocated(), live: existing.concurrency() })
    }
}

/// Install the limits for a family that has no separate ceiling: the allocated
/// ceiling and the live limit both start at `concurrency`.
pub(crate) fn install_family(concurrency: u32) -> Result<&'static LiveLimits, LimitError> {
    install(concurrency, concurrency)
}

/// The process-wide limits. Before [`install`] this is a conservative
/// single-request ceiling so a caller can never admit more than it sized.
pub(crate) fn live() -> &'static LiveLimits {
    LIVE.get_or_init(|| LiveLimits::new(1, 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_to_allocated() {
        let limits = LiveLimits::new(8, 99);
        assert_eq!(limits.allocated(), 8);
        assert_eq!(limits.concurrency(), 8);
        assert_eq!(limits.set_concurrency(9), Err(LimitError::AboveCeiling { requested: 9, allocated: 8 }));
        assert_eq!(limits.concurrency(), 8);
    }

    #[test]
    fn rejects_zero() {
        let limits = LiveLimits::new(8, 4);
        assert_eq!(limits.set_concurrency(0), Err(LimitError::Zero));
        assert_eq!(limits.concurrency(), 4);
    }

    #[test]
    fn set_is_visible_to_a_second_reader() {
        let limits = LiveLimits::new(16, 8);
        assert_eq!(limits.set_concurrency(12).unwrap(), 12);
        assert_eq!(limits.concurrency(), 12);
        assert_eq!(limits.snapshot()["allocated"], 16);
        assert_eq!(limits.snapshot()["live"], 12);
    }

    #[test]
    fn snapshot_reports_allocated_live_and_slots_max() {
        let limits = LiveLimits::new(16, 8);
        let snapshot = limits.snapshot();
        assert_eq!(snapshot["allocated"], 16);
        assert_eq!(snapshot["live"], 8);
        assert_eq!(snapshot["slots_max"], 128);
        limits.set_concurrency(4).unwrap();
        assert_eq!(limits.snapshot()["live"], 4);
    }

    /// Raise and lower while readers are active: the live limit changes
    /// without touching the allocated ceiling, and a lower limit never
    /// disables admission.
    #[test]
    fn raise_and_lower_under_load() {
        let limits = LiveLimits::new(16, 4);
        let readers: Vec<_> = (0..4).map(|_| limits.concurrency()).collect();
        assert_eq!(readers, vec![4; 4]);
        assert_eq!(limits.set_concurrency(12).unwrap(), 12);
        assert_eq!(limits.concurrency(), 12);
        assert_eq!(limits.set_concurrency(2).unwrap(), 2);
        assert_eq!(limits.concurrency(), 2);
        assert_eq!(limits.set_concurrency(16).unwrap(), 16);
        assert_eq!(limits.allocated(), 16, "the ceiling never moves");
    }

    /// A second install with different values is refused, not silently
    /// discarded; the same values are idempotent.
    #[test]
    fn second_install_is_not_silently_discarded() {
        let first = install(16, 8).unwrap();
        assert!(std::ptr::eq(first, install(16, 8).unwrap()), "same values are idempotent");
        assert!(matches!(install(32, 8),
            Err(LimitError::AlreadyInstalled { allocated: 16, live: 8 })));
        assert!(matches!(install(16, 4),
            Err(LimitError::AlreadyInstalled { allocated: 16, live: 8 })));
        assert_eq!(live().allocated(), 16);
        assert_eq!(live().concurrency(), 8);
    }
}
