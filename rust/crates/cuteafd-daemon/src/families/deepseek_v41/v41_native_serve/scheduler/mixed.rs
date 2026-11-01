//! V4.1 prefill admission on the shared [`PrefillQueue`] (design §4).
//!
//! An admitted prompt parks only while other streams decode, and only when it
//! is an uncached plain prompt: a probe cold replay and a scoring request keep
//! their ordinary whole-prompt prefill. Share 0 never parks, so the queue is
//! never touched and the admission path is the previous rule.

use std::time::Instant;

use cuteafd_api::openai::NativeRequest;

use crate::families::deepseek_v41::v41_backbone_cache::CacheLease;
use crate::families::deepseek_v41::v41_native_serve::prefix::ImageKeys;
use crate::families::deepseek_v41::v41_native_serve::scores::RetainedScores;

use super::admission;
use super::super::prefill_target::{V41Prefill, V41PrefillTimes};

/// An admitted prompt waiting in the shared [`PrefillQueue`]: it holds its slot
/// and lease, and its prefill cursor survives between rounds. `scores` is the
/// finished prompt's first-token row, set by the round that finalizes it.
pub(super) struct Parked<S> {
    pub(super) prepared: admission::Prepared,
    pub(super) id: u64,
    pub(super) lease: CacheLease,
    pub(super) slot: usize,
    pub(super) image_keys: ImageKeys,
    pub(super) hit: Option<(usize, Option<RetainedScores>)>,
    pub(super) restore: (Instant, Instant),
    pub(super) cursor: V41Prefill<S>,
    pub(super) times: V41PrefillTimes,
    pub(super) scores: Option<RetainedScores>,
    pub(super) parked_at: Instant,
}

/// Whether an admitted prompt parks instead of prefilling whole: only with a
/// positive share, while other streams decode, for an uncached plain prompt
/// (no probe cold steps, not scoring). Images are already encoded at this point.
pub(super) fn parks(share: f64, decoding: bool, job: &NativeRequest, cached: usize,
    prompt_len: usize) -> bool {
    share > 0.0
        && decoding
        && !job.probe.as_ref().is_some_and(|probe| !probe.spec.cold_steps.is_empty())
        && crate::shared::probe::scoring(&job.probe).is_none()
        && cached != prompt_len
}

/// A member the packed prefill already prefilled and KV-committed is never also
/// parked: packing already paid the cost and short prompts are not what park is
/// for.
pub(super) fn parks_member(packed: bool, share: f64, decoding: bool, job: &NativeRequest,
    cached: usize, prompt_len: usize) -> bool {
    !packed && parks(share, decoding, job, cached, prompt_len)
}

/// Shared prefill counters, exported under `share` in `/v1/stats`.
pub(super) mod stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// Prefill rounds issued while a prompt was parked.
    pub(crate) static CALLS: AtomicU64 = AtomicU64::new(0);
    /// Longest time a prompt waited in the queue, milliseconds.
    pub(crate) static PARK_WAIT_MS_MAX: AtomicU64 = AtomicU64::new(0);
    /// Duration of the last prefill round, milliseconds.
    pub(crate) static CALL_MS_LAST: AtomicU64 = AtomicU64::new(0);
    /// Prompts parked right now.
    pub(crate) static PARKED_NOW: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn snapshot() -> serde_json::Value {
        serde_json::json!({
            "parked_now": PARKED_NOW.load(Relaxed),
            "calls": CALLS.load(Relaxed),
            "park_wait_ms_max": PARK_WAIT_MS_MAX.load(Relaxed),
            "call_ms_last": CALL_MS_LAST.load(Relaxed),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_api::openai::probe::{Probe, ProbeColdStep, ProbeSpec};

    fn job(probe: Option<std::sync::Arc<Probe>>) -> NativeRequest {
        let (events, _receive) = tokio::sync::mpsc::unbounded_channel();
        NativeRequest { prefix_cache_source: Default::default(), literal_spans: Vec::new(),
            prompt: String::new(), constraint: None, images: Vec::new(), media: Vec::new(),
            audio: Vec::new(), max_tokens: 16, sampling: Default::default(),
            stop_token_ids: Vec::new(), events, usage: None, probe }
    }

    #[test]
    fn share_zero_never_parks() {
        let plain = job(None);
        for decoding in [false, true] {
            for (cached, prompt_len) in [(0, 8), (4, 8), (8, 8)] {
                assert!(!parks(0.0, decoding, &plain, cached, prompt_len));
            }
        }
    }

    #[test]
    fn parks_only_uncached_plain_prompts_while_decoding() {
        let plain = job(None);
        assert!(parks(0.2, true, &plain, 0, 8));
        assert!(parks(0.2, true, &plain, 4, 8));
        assert!(!parks(0.2, false, &plain, 0, 8), "nothing decodes: prefill whole");
        assert!(!parks(0.2, true, &plain, 8, 8), "whole-prompt hit: nothing to prefill");
        let replay = job(Some(Probe::new(ProbeSpec { cold: true,
            cold_steps: vec![ProbeColdStep { end: 4, decode: false }], ..Default::default() })));
        assert!(!parks(0.2, true, &replay, 0, 8), "cold replay keeps its whole-prompt path");
        let scoring = job(Some(Probe::new(ProbeSpec { score_from: Some(0), ..Default::default() })));
        assert!(!parks(0.2, true, &scoring, 0, 8), "scoring keeps its whole-prompt path");
    }

    #[test]
    fn parks_member_never_parks_a_packed_member() {
        let plain = job(None);
        assert!(parks_member(false, 0.2, true, &plain, 0, 8));
        assert!(!parks_member(true, 0.2, true, &plain, 0, 8));
    }

    /// A prompt of `left` chunks, recording the order chunks ran in.
    struct Prompt {
        id: usize,
        left: usize,
    }

    /// Share 0 is the previous rule: the queue is never pushed to, so the
    /// admission path makes exactly the plan/execute calls `run_to_end` makes.
    #[test]
    fn share_zero_round_makes_the_same_plan_and_execute_calls_as_run_to_end() {
        use crate::shared::prefill_share::{run_to_end, run_unit, ChunkBudget, PrefillDriver,
            PrefillQueue, PrefillUnit};
        use std::cell::Cell;
        use std::time::Duration;

        struct CountingDriver { plans: Cell<usize>, executes: Cell<usize> }
        impl PrefillDriver for CountingDriver {
            type Cursor = usize;
            type Unit = ();
            type Output = usize;
            fn plan(&self, cursor: &usize, _budget: ChunkBudget) -> anyhow::Result<PrefillUnit<()>> {
                self.plans.set(self.plans.get() + 1);
                anyhow::ensure!(*cursor > 0, "no units left");
                Ok(PrefillUnit { work: (), rows: 1, estimate: Duration::from_micros(1),
                    finalizes: *cursor == 1 })
            }
            fn execute(&mut self, cursor: &mut usize, unit: &PrefillUnit<()>)
                -> anyhow::Result<Option<usize>> {
                self.executes.set(self.executes.get() + 1);
                *cursor -= 1;
                Ok(unit.finalizes.then_some(42))
            }
            fn observe(&mut self, _unit: &PrefillUnit<()>, _took: Duration) {}
            fn abort_and_drain(&mut self, _cursor: &mut usize) -> anyhow::Result<()> { Ok(()) }
        }

        let mut driver = CountingDriver { plans: Cell::new(0), executes: Cell::new(0) };
        let mut cursor = 4;
        assert_eq!(run_to_end(&mut driver, &mut cursor).unwrap(), 42);
        let (plans, executes) = (driver.plans.get(), driver.executes.get());
        assert_eq!((plans, executes), (4, 4));

        let mut driver = CountingDriver { plans: Cell::new(0), executes: Cell::new(0) };
        let mut queue = PrefillQueue::new(0.0);
        queue.push(4usize);
        let finished = queue.round(|left| match run_unit(&mut driver, left, ChunkBudget::WHOLE)? {
            Some(_) => Ok(crate::shared::prefill_share::Chunk::Done),
            None => Ok(crate::shared::prefill_share::Chunk::More),
        });
        assert_eq!(finished.len(), 1);
        assert_eq!((driver.plans.get(), driver.executes.get()), (plans, executes),
            "share 0 makes the same plan/execute calls as run_to_end");
    }

    /// One round runs one call per prompt and rotates the prompts it did not
    /// serve behind the ones it did (design §5.2 test 3).
    #[test]
    fn parked_round_runs_one_call_per_prompt_and_rotates() {
        use crate::shared::prefill_share::{Chunk, PrefillQueue};
        let mut queue = PrefillQueue::new(0.2).one_wave_rounds();
        queue.push(Prompt { id: 0, left: 2 });
        queue.push(Prompt { id: 1, left: 2 });
        let mut ran = Vec::new();
        queue.round(|p| {
            ran.push(p.id);
            p.left -= 1;
            Ok(if p.left == 0 { Chunk::Done } else { Chunk::More })
        });
        assert_eq!(ran, vec![0], "one call per round");
        queue.settle(true);
        queue.stepped(queue.decode_seconds());
        ran.clear();
        queue.round(|p| {
            ran.push(p.id);
            p.left -= 1;
            Ok(if p.left == 0 { Chunk::Done } else { Chunk::More })
        });
        assert_eq!(ran, vec![1], "the unserved prompt rotates in");
    }

    /// A running stream is owed its share of a prefill round's time before the
    /// next round: the queue does not start another chunk until the debt is paid.
    #[test]
    fn a_running_stream_is_owed_decode_steps_while_a_prompt_prefills() {
        use crate::shared::prefill_share::{Chunk, PrefillQueue};
        let mut queue = PrefillQueue::new(0.2).one_wave_rounds();
        queue.push(Prompt { id: 0, left: 3 });
        let mut ran = Vec::new();
        queue.round(|p| {
            ran.push(p.id);
            p.left -= 1;
            Ok(if p.left == 0 { Chunk::Done } else { Chunk::More })
        });
        assert_eq!(ran, vec![0], "one chunk per round");
        queue.settle(true);
        let owed = queue.decode_seconds();
        assert!(owed > 0.0, "the stream is owed decode time");
        assert!(!queue.due(true), "the stream runs its owed steps first");
        queue.stepped(owed);
        assert!(queue.due(true), "the next prefill round is due after the owed steps");
    }
}
