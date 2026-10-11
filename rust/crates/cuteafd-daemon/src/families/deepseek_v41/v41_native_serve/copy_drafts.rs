//! V4.1 copy-draft opt-in and lifetime serving counters.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub(super) use crate::shared::speculation::copy::{LatestWindow as CopyDrafter, cap, merge, WINDOW, MIN_DRAFTS};

/// Opt-in: copy-heavy edits gain, but normal C1 misses the no-regression bar.
pub(super) fn enabled() -> bool {
    std::env::var("CUTEAFD_COPY_DRAFTS").is_ok_and(|value| value == "1")
}

static REQUEST_ROUNDS: AtomicU64 = AtomicU64::new(0);
static VERIFIED: AtomicU64 = AtomicU64::new(0);
static ACCEPTED: AtomicU64 = AtomicU64::new(0);

/// Count one request's verified copy round.
pub(super) fn record_round(verified: usize, accepted: usize) {
    REQUEST_ROUNDS.fetch_add(1, Relaxed);
    VERIFIED.fetch_add(verified as u64, Relaxed);
    ACCEPTED.fetch_add(accepted as u64, Relaxed);
}

/// Lifetime copy counters for the per-second `stats` payload. The length
/// policy observes every verified row, so `dspark_policy`'s verified and
/// accepted drafts include these.
pub(super) fn stats() -> serde_json::Value {
    serde_json::json!({
        "request_rounds": REQUEST_ROUNDS.load(Relaxed),
        "verified_drafts": VERIFIED.load(Relaxed),
        "accepted_drafts": ACCEPTED.load(Relaxed),
    })
}
