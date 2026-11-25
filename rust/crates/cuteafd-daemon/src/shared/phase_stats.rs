//! Always-on serve-loop phase counters, published in `/v1/stats` `phases` for every family.
//!
//! Monotonic counters read only on `/v1/stats` requests: a few relaxed atomic adds per
//! decode call, per idle wait and per queued prompt. No allocation, no lock, no log line,
//! and on the hot decode and prefill paths no clock read that the serve loop did not
//! already take (single-lane loops reuse `cycle` around decode and `started`/`timer` at a
//! prompt's first chunk; `idle_wait` runs only when blocked idle).
//!
//! The prefill-round counters live with the round itself
//! ([`crate::shared::prefill_share::round_stats`]) and are merged into the same snapshot.
//! The three time counters `prefill_round`, `decode_call` and `idle_wait` never overlap;
//! `decode_call_with_prefill_waiting` is a subset of `decode_call`.
//!
//! Graph instantiation time lives beside the shared graph bank
//! ([`crate::shared::decode_graph::stats`]) and is published top-level as
//! `graph_instantiate_ns` by [`crate::shared::probe::graph_capture_stats`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// One serve loop's counters. Serving uses [`GLOBAL`]; a test host owns its own.
pub(crate) struct PhaseStats {
    /// Wall time of every decode call (one `lanes.run_one(|lease| decode_round(..))` call
    /// on single-lane families, or one `P::decode_round` call on V4.1, which may run many
    /// lane rounds on the two-lane and distributed paths).
    decode_call_ns: AtomicU64,
    /// The wall time of the decode calls that ran while prompts waited in the prefill
    /// queue (calls the decode share pays for). A part of `decode_call_ns`.
    decode_call_with_prefill_waiting_ns: AtomicU64,
    /// Time blocked waiting for a request with nothing admitted and nothing queued.
    idle_wait_ns: AtomicU64,
    decode_calls: AtomicU64,
    /// Active member sequences of every decode call (one per request per call it ran in).
    decode_call_members: AtomicU64,
    /// From a prompt's entry into the prefill queue to its first chunk, summed over the
    /// prompts that started.
    prefill_queue_wait_ns: AtomicU64,
}

pub(crate) static GLOBAL: PhaseStats = PhaseStats::new();

impl PhaseStats {
    pub(crate) const fn new() -> Self {
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            decode_call_ns: ZERO,
            decode_call_with_prefill_waiting_ns: ZERO,
            idle_wait_ns: ZERO,
            decode_calls: ZERO,
            decode_call_members: ZERO,
            prefill_queue_wait_ns: ZERO,
        }
    }

    /// One decode call: its wall time, whether prompts were waiting to prefill behind
    /// it, and its active member count.
    pub(crate) fn record_decode_call(&self, ns: u64, prefill_waiting: bool, members: u64) {
        self.decode_call_ns.fetch_add(ns, Ordering::Relaxed);
        if prefill_waiting {
            self.decode_call_with_prefill_waiting_ns
                .fetch_add(ns, Ordering::Relaxed);
        }
        self.decode_calls.fetch_add(1, Ordering::Relaxed);
        self.decode_call_members
            .fetch_add(members, Ordering::Relaxed);
    }

    /// Time spent blocked waiting for a request with nothing admitted and nothing queued.
    pub(crate) fn record_idle_wait(&self, ns: u64) {
        self.idle_wait_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Time an idle receive (only called when nothing is admitted, queued or encoding)
    /// and record its wall duration.
    pub(crate) fn time_idle_wait<T>(&self, wait: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let value = wait();
        self.record_idle_wait(started.elapsed().as_nanos() as u64);
        value
    }

    /// One prompt's wait from entering the prefill queue to starting its first chunk.
    pub(crate) fn record_prefill_queue_wait(&self, ns: u64) {
        self.prefill_queue_wait_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Record a queued prompt's wait once, at its first chunk (`chunks == 0`), reusing
    /// the `queued_at` and chunk-start `now` clocks the serve loop already took.
    pub(crate) fn record_first_chunk_wait(&self, chunks: usize, queued_at: Instant, now: Instant) {
        if chunks == 0 {
            self.record_prefill_queue_wait(now.saturating_duration_since(queued_at).as_nanos() as u64);
        }
    }

    #[cfg(test)]
    pub(crate) fn prefill_queue_wait_ns(&self) -> u64 {
        self.prefill_queue_wait_ns.load(Ordering::Relaxed)
    }

    /// `decoding` is the number of requests decoding at the last publish: read from the
    /// published `active` count, never a stored gauge that could go stale across idle wait.
    pub(crate) fn snapshot(
        &self,
        rounds: &crate::shared::prefill_share::round_stats::RoundStats,
        decoding: usize,
    ) -> serde_json::Value {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        serde_json::json!({
            "phase_ns_total": {
                "prefill_round": rounds.round_ns(),
                "decode_call": get(&self.decode_call_ns),
                "decode_call_with_prefill_waiting": get(&self.decode_call_with_prefill_waiting_ns),
                "idle_wait": get(&self.idle_wait_ns),
            },
            "rounds_total": {"prefill": rounds.rounds(), "decode_calls": get(&self.decode_calls)},
            "decode_call_members_total": get(&self.decode_call_members),
            "prefill_prompt_rounds_total": rounds.prompt_rounds(),
            "prefill_queue_wait_ns_total": get(&self.prefill_queue_wait_ns),
            "active_decode": decoding,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::PhaseStats;
    use crate::shared::prefill_share::round_stats::RoundStats;
    use std::time::{Duration, Instant};

    #[test]
    fn phase_counters_publish_recorded_totals_without_graph_duplicate() {
        let (stats, rounds) = (PhaseStats::new(), RoundStats::new());
        stats.record_decode_call(1_500_000, true, 7);
        stats.record_decode_call(1_000_000, false, 2);
        stats.record_idle_wait(250_000);
        stats.record_prefill_queue_wait(400_000);
        let published = stats.snapshot(&rounds, 4);
        assert_eq!(
            published,
            serde_json::json!({
                "phase_ns_total": {
                    "prefill_round": 0,
                    "decode_call": 2_500_000,
                    "decode_call_with_prefill_waiting": 1_500_000,
                    "idle_wait": 250_000,
                },
                "rounds_total": {"prefill": 0, "decode_calls": 2},
                "decode_call_members_total": 9,
                "prefill_prompt_rounds_total": 0,
                "prefill_queue_wait_ns_total": 400_000,
                "active_decode": 4,
            })
        );
        assert!(
            published["phase_ns_total"].get("graph_instantiate").is_none(),
            "graph_instantiate_ns is published top-level once, not duplicated under phases"
        );
    }

    #[test]
    fn first_chunk_wait_charges_once_from_existing_clocks() {
        let stats = PhaseStats::new();
        let now = Instant::now();
        let queued = now - Duration::from_millis(5);
        stats.record_first_chunk_wait(0, queued, now);
        let first = stats.prefill_queue_wait_ns();
        assert_eq!(first, 5_000_000);
        // Later chunks of the same prompt do not charge again.
        stats.record_first_chunk_wait(1, queued, now + Duration::from_millis(20));
        stats.record_first_chunk_wait(2, queued, now + Duration::from_millis(40));
        assert_eq!(stats.prefill_queue_wait_ns(), first);
        // Reversed clocks saturate to zero without underflow.
        stats.record_first_chunk_wait(0, now + Duration::from_millis(1), now);
        assert_eq!(stats.prefill_queue_wait_ns(), first);
    }

    #[test]
    fn time_idle_wait_returns_value_and_records_elapsed_time() {
        let (stats, rounds) = (PhaseStats::new(), RoundStats::new());
        let received = stats.time_idle_wait(|| {
            std::thread::sleep(Duration::from_millis(2));
            Some(42u32)
        });
        assert_eq!(received, Some(42));
        let idle = stats.snapshot(&rounds, 0)["phase_ns_total"]["idle_wait"]
            .as_u64()
            .unwrap();
        assert!(idle >= 1_000_000, "idle_wait should record wait time, got {idle} ns");
    }

    /// Every serve loop (covering all 15 families) records idle wait, decode calls and
    /// first-chunk queue wait into the shared `phase_stats::GLOBAL` bank.
    #[test]
    fn every_serve_loop_wires_shared_phase_recording_points() {
        let single_lane_loops = [
            ("shared/qwen/scheduler.rs", include_str!("qwen/scheduler.rs")),
            ("families/glm5/serve.rs", include_str!("../families/glm5/serve.rs")),
            ("families/glm5_flash/serve.rs", include_str!("../families/glm5_flash/serve.rs")),
            ("families/mimo_v2/serve.rs", include_str!("../families/mimo_v2/serve.rs")),
            ("families/deepseek_v4/serve.rs", include_str!("../families/deepseek_v4/serve.rs")),
        ];
        for (name, source) in single_lane_loops {
            assert!(
                source.contains("phase_stats::GLOBAL.time_idle_wait("),
                "{name} missing idle_wait recording"
            );
            assert!(
                source.contains("phase_stats::GLOBAL.record_decode_call("),
                "{name} missing decode_call recording"
            );
            assert_eq!(
                source.matches("phase_stats::GLOBAL.record_first_chunk_wait(").count(),
                2,
                "{name} must record prefill queue wait on both singleton and packed chunk paths"
            );
        }
        let v41 = include_str!("../families/deepseek_v41/v41_native_serve/scheduler.rs");
        let v41_queue = include_str!("../families/deepseek_v41/v41_native_serve/scheduler/queue.rs");
        assert!(v41.contains("phase_stats::GLOBAL.time_idle_wait("));
        assert!(v41.contains("phase_stats::GLOBAL.record_decode_call("));
        assert!(v41_queue.contains("host.phases().record_prefill_queue_wait("));
        // V4.1's serving_stats delegates phases, exchange and graph_instantiate_ns to shared code.
        let v41_stats = v41
            .split_once("fn serving_stats<")
            .unwrap()
            .1
            .split_once("\n}\n")
            .unwrap()
            .0;
        assert!(v41_stats.contains("\"active\": decoding"));
        assert!(!v41_stats.contains("\"phases\":"));
        assert!(!v41_stats.contains("\"exchange\":"));
        assert!(!v41_stats.contains("\"graph_instantiate"));
        assert!(!v41_stats.contains("decode_graph::stats"));
    }
}
