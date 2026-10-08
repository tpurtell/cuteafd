//! Prefill rounds that leave running streams a share of the time, shared by
//! every family's serve loop (GLM 5.3, GLM 5.3 Flash, MiMo, Qwen 3.8, V4).
//!
//! A serve loop used to prefill a whole prompt when it admitted it, so a long
//! prompt stalled every stream for its whole prefill (a 64K prompt: tens of
//! seconds without a token). Here admitted prompts wait in a [`PrefillQueue`]
//! and prefill in rounds: each waiting prompt runs one chunk (one engine
//! prefill call, a lane wave where the family pipelines lanes) in arrival
//! order, until the round has spent [`ROUND_S`]. After a round of `t` seconds
//! the running requests are owed `t * share / (1 - share)` seconds of decode
//! steps (one step at least) before the next round, so a stream keeps about
//! `share` of its rate while a long prompt prefills and the prompt takes about
//! `1 / (1 - share)` times as long. Nothing is owed while nothing decodes or
//! nothing waits.
//!
//! Share 0 is the previous rule: a round prefills every waiting prompt whole
//! before the next step. What a chunk computes never changes: a prompt's
//! chunks are cut at the same rows; only when other sequences step moves.
//! After Hugh Madden's glm53f-afd decode share (MIT, v1.1.0 49392af).
use anyhow::Result;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::time::Instant;

/// A round starts no further chunk once it has spent this long (seconds):
/// a burst of short prompts prefills in one round, a long prompt gets one
/// chunk per round.
pub(crate) const ROUND_S: f64 = 1.0;

#[derive(Debug, Clone, Copy, clap::Args)]
pub(crate) struct DecodeShareArgs {
    /// Share of the time running requests keep while prompts prefill: after
    /// a prefill round (one chunk of each waiting prompt) of t seconds they
    /// step for t * share / (1 - share) seconds. 0 prefills whole prompts
    /// before the next step (the previous behaviour).
    #[arg(long, env = "CUTEAFD_DECODE_SHARE", default_value_t = 0.2)]
    pub decode_share: f64,
}

impl DecodeShareArgs {
    pub fn queue<P>(&self) -> Result<PrefillQueue<P>> {
        anyhow::ensure!((0.0..1.0).contains(&self.decode_share),
            "--decode-share must be in [0, 1), got {}", self.decode_share);
        tracing::info!(decode_share = self.decode_share, "{}", if self.decode_share > 0.0 {
            "prefill in rounds of one chunk per prompt; running requests keep their share between rounds"
        } else {
            "prompts prefill whole before the next decode step"
        });
        Ok(PrefillQueue::new(self.decode_share))
    }
}

/// A chunk's outcome for [`PrefillQueue::round`].
pub(crate) enum Chunk {
    /// More of the prompt remains.
    More,
    /// The prompt is prefilled.
    Done,
}

/// Admitted prompts waiting to prefill, and the decode time owed.
pub(crate) struct PrefillQueue<P> {
    waiting: VecDeque<P>,
    share: f64,
    owed: f64,
    last_round: f64,
    round_s: f64,
}

impl<P> PrefillQueue<P> {
    pub fn new(share: f64) -> Self {
        Self { waiting: VecDeque::new(), share, owed: 0.0, last_round: 0.0, round_s: ROUND_S }
    }

    pub fn len(&self) -> usize {
        self.waiting.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &P> {
        self.waiting.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut P> {
        self.waiting.iter_mut()
    }

    pub fn push(&mut self, prompt: P) {
        self.waiting.push_back(prompt);
    }

    /// Whether the loop runs a prefill round before its next step: something
    /// waits and the running requests (if any) are owed nothing.
    pub fn due(&self, decoding: bool) -> bool {
        !self.waiting.is_empty() && (!decoding || self.share == 0.0 || self.owed <= 0.0)
    }

    /// One prefill round. `chunk` prefills the prompt's next chunk. With
    /// share 0 every waiting prompt runs to its end; otherwise each gets one
    /// chunk at most, in arrival order, until the round has spent
    /// [`ROUND_S`], and prompts that ran go behind those that did not.
    /// Returns the prompts that finished (`Ok`) or failed (`Err`); call
    /// [`Self::settle`] once the finished ones joined the running requests.
    pub fn round(&mut self, mut chunk: impl FnMut(&mut P) -> Result<Chunk>) -> Vec<(P, Result<()>)> {
        let started = Instant::now();
        let whole = self.share == 0.0;
        let mut out = Vec::new();
        let mut ran = VecDeque::new();
        while let Some(mut prompt) = self.waiting.pop_front() {
            if !whole && (!ran.is_empty() || !out.is_empty()) && started.elapsed().as_secs_f64() >= self.round_s {
                self.waiting.push_front(prompt);
                break;
            }
            loop {
                match chunk(&mut prompt) {
                    Ok(Chunk::Done) => out.push((prompt, Ok(()))),
                    Ok(Chunk::More) if whole => continue,
                    Ok(Chunk::More) => ran.push_back(prompt),
                    Err(error) => out.push((prompt, Err(error))),
                }
                break;
            }
        }
        self.waiting.extend(ran);
        self.last_round = started.elapsed().as_secs_f64();
        out
    }

    /// Opt-in adjacent pairing. The oldest prompt always runs; its immediate
    /// neighbour joins only when `eligible` accepts both. `chunk` receives one
    /// or two prompts and returns their independent outcomes (the second is
    /// ignored for a singleton). No prompt is pulled past an ineligible one.
    /// A pair consumes one elapsed-time budget, with one chunk per member.
    pub fn round_pairs(&mut self, eligible: impl Fn(&P, &P) -> bool,
        mut chunk: impl FnMut(&mut [P]) -> [Result<Chunk>; 2]) -> Vec<(P, Result<()>)> {
        let started = Instant::now();
        let whole = self.share == 0.0;
        let mut out = Vec::new();
        let mut ran = VecDeque::new();
        let mut steps = 0;
        while !self.waiting.is_empty() {
            if !whole && steps > 0 && started.elapsed().as_secs_f64() >= self.round_s { break; }
            let mut batch = vec![self.waiting.pop_front().expect("nonempty prefill queue")];
            if self.waiting.front().is_some_and(|next| eligible(&batch[0], next)) {
                batch.push(self.waiting.pop_front().expect("eligible adjacent prompt"));
            }
            let outcomes = chunk(&mut batch);
            let mut unfinished = Vec::new();
            for (prompt, outcome) in batch.into_iter().zip(outcomes) {
                match outcome {
                    Ok(Chunk::Done) => out.push((prompt, Ok(()))),
                    Ok(Chunk::More) if whole => unfinished.push(prompt),
                    Ok(Chunk::More) => ran.push_back(prompt),
                    Err(error) => out.push((prompt, Err(error))),
                }
            }
            // Share zero finishes the oldest requests before admitting another
            // chunk group. Positive share rotates them behind unserved prompts.
            for prompt in unfinished.into_iter().rev() { self.waiting.push_front(prompt); }
            steps += 1;
        }
        self.waiting.extend(ran);
        self.last_round = started.elapsed().as_secs_f64();
        out
    }

    /// After a round: the running requests are owed their share of its time
    /// while prompts still wait.
    pub fn settle(&mut self, decoding: bool) {
        self.owed = if decoding && !self.waiting.is_empty() {
            self.last_round * self.share / (1.0 - self.share)
        } else {
            0.0
        };
    }

    /// A step (drafting, verify, streaming) took `seconds`.
    pub fn stepped(&mut self, seconds: f64) {
        self.owed -= seconds;
    }

    /// Prompts waiting, for cleanup when the loop fails.
    pub fn drain(&mut self) -> impl Iterator<Item = P> + '_ {
        self.waiting.drain(..)
    }
}

/// Runs `f` with the engine's phase profile isolated: returns the phases `f`
/// added and leaves the profile as it was (decode steps between a prompt's
/// chunks keep their own phases).
pub(crate) fn isolated_phases<const N: usize, R>(profile: &RefCell<[f64; N]>, f: impl FnOnce() -> R)
    -> (R, [f64; N]) {
    let before = std::mem::replace(&mut *profile.borrow_mut(), [0.0; N]);
    let result = f();
    let mine = std::mem::replace(&mut *profile.borrow_mut(), before);
    (result, mine)
}

pub(crate) fn add_phases<const N: usize>(total: &mut [f64; N], phases: [f64; N]) {
    for (t, p) in total.iter_mut().zip(phases) {
        *t += p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A prompt of `left` chunks, recording the order chunks ran in.
    struct Prompt {
        id: usize,
        left: usize,
    }

    fn run(queue: &mut PrefillQueue<Prompt>, log: &mut Vec<usize>, sleep_ms: u64) -> Vec<usize> {
        queue.round(|p| {
            log.push(p.id);
            std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
            p.left -= 1;
            Ok(if p.left == 0 { Chunk::Done } else { Chunk::More })
        }).into_iter().map(|(p, r)| {
            r.unwrap();
            p.id
        }).collect()
    }

    #[test]
    fn share_zero_prefills_whole_prompts_in_one_round() {
        let mut queue = PrefillQueue::new(0.0);
        queue.push(Prompt { id: 0, left: 3 });
        queue.push(Prompt { id: 1, left: 2 });
        let mut log = Vec::new();
        assert_eq!(run(&mut queue, &mut log, 0), vec![0, 1]);
        assert_eq!(log, vec![0, 0, 0, 1, 1]);
        queue.settle(true);
        assert!(!queue.due(true));
    }

    #[test]
    fn a_long_prompt_runs_one_chunk_per_round_and_owes_the_share() {
        let mut queue = PrefillQueue::new(0.2);
        queue.push(Prompt { id: 0, left: 3 });
        let mut log = Vec::new();
        assert!(queue.due(true));
        assert!(run(&mut queue, &mut log, 20).is_empty());
        queue.settle(true);
        // 20 ms of prefill owes 5 ms of steps.
        assert!(!queue.due(true));
        assert!(queue.due(false), "nothing decoding: prefill continues");
        queue.stepped(0.004);
        assert!(!queue.due(true));
        queue.stepped(0.002);
        assert!(queue.due(true));
        assert_eq!(log, vec![0]);
    }

    #[test]
    fn short_prompts_share_a_round_and_rotate_behind_unserved_ones() {
        let mut queue = PrefillQueue::new(0.2);
        queue.push(Prompt { id: 0, left: 2 });
        queue.push(Prompt { id: 1, left: 1 });
        queue.push(Prompt { id: 2, left: 1 });
        let mut log = Vec::new();
        assert_eq!(run(&mut queue, &mut log, 0), vec![1, 2]);
        assert_eq!(log, vec![0, 1, 2]);
        assert_eq!(run(&mut queue, &mut log, 0), vec![0]);
        queue.settle(true);
        assert!(!queue.due(true) && queue.is_empty());
    }

    #[test]
    fn a_round_stops_starting_chunks_after_its_budget() {
        let mut queue = PrefillQueue::new(0.5);
        queue.round_s = 0.01;
        queue.push(Prompt { id: 0, left: 2 });
        queue.push(Prompt { id: 1, left: 2 });
        let mut log = Vec::new();
        run(&mut queue, &mut log, 15);
        assert_eq!(log, vec![0]);
        run(&mut queue, &mut log, 0);
        // Prompt 1 did not run last round, so it goes first.
        assert_eq!(log, vec![0, 1, 0]);
    }

    fn pair_step(batch: &mut [Prompt], log: &mut Vec<Vec<usize>>) -> [Result<Chunk>; 2] {
        log.push(batch.iter().map(|p| p.id).collect());
        let mut outcomes = [Ok(Chunk::Done), Ok(Chunk::Done)];
        for (p, out) in batch.iter_mut().zip(&mut outcomes) {
            p.left -= 1;
            *out = Ok(if p.left == 0 { Chunk::Done } else { Chunk::More });
        }
        outcomes
    }

    #[test]
    fn pairs_rotate_after_unserved_requests_and_owe_one_round() {
        let mut queue = PrefillQueue::new(0.2);
        queue.round_s = 0.0; // One batch per round, without timing sleeps.
        for id in 0..4 { queue.push(Prompt { id, left: 2 }); }
        let mut log = Vec::new();
        assert!(queue.round_pairs(|_, _| true, |batch| pair_step(batch, &mut log)).is_empty());
        assert_eq!(log, [vec![0, 1]]);
        assert_eq!(queue.waiting.iter().map(|p| p.id).collect::<Vec<_>>(), [2, 3, 0, 1]);
        let elapsed = queue.last_round;
        queue.settle(true);
        assert_eq!(queue.owed, elapsed * 0.2 / 0.8);
        assert!(!queue.due(true));
        queue.stepped(queue.owed);
        assert!(queue.due(true));
        queue.round_pairs(|_, _| true, |batch| pair_step(batch, &mut log));
        assert_eq!(log, [vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn pairing_never_skips_an_ineligible_adjacent_request() {
        let mut queue = PrefillQueue::new(0.2);
        for id in 0..4 { queue.push(Prompt { id, left: 1 }); }
        let mut log = Vec::new();
        let finished = queue.round_pairs(|a, b| a.id != 1 && b.id != 1,
            |batch| pair_step(batch, &mut log));
        assert_eq!(log, [vec![0], vec![1], vec![2, 3]]);
        assert_eq!(finished.into_iter().map(|(p, result)| { result.unwrap(); p.id }).collect::<Vec<_>>(), [0, 1, 2, 3]);
    }

    #[test]
    fn paired_failure_does_not_discard_healthy_neighbour_or_repeat_chunk() {
        let mut queue = PrefillQueue::new(0.2);
        queue.push(Prompt { id: 0, left: 2 });
        queue.push(Prompt { id: 1, left: 2 });
        let mut log = Vec::new();
        let finished = queue.round_pairs(|_, _| true, |batch| {
            let mut out = pair_step(batch, &mut log);
            out[0] = Err(anyhow::anyhow!("client cancelled"));
            out
        });
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].0.id, 0);
        assert!(finished[0].1.is_err());
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.waiting[0].id, 1);
        assert_eq!(queue.waiting[0].left, 1);
        let finished = queue.round_pairs(|_, _| true, |batch| pair_step(batch, &mut log));
        assert_eq!(finished[0].0.id, 1);
        assert!(finished[0].1.is_ok());
        assert_eq!(log, [vec![0, 1], vec![1]]);
    }

    #[test]
    fn zero_share_finishes_oldest_pair_before_later_requests() {
        let mut queue = PrefillQueue::new(0.0);
        for id in 0..3 { queue.push(Prompt { id, left: if id == 2 { 1 } else { 2 } }); }
        let mut log = Vec::new();
        let finished = queue.round_pairs(|_, _| true, |batch| pair_step(batch, &mut log));
        assert_eq!(log, [vec![0, 1], vec![0, 1], vec![2]]);
        assert_eq!(finished.len(), 3);
        assert!(queue.is_empty());
    }

    #[test]
    fn ineligible_pairs_match_original_singleton_rounds() {
        for share in [0.0, 0.2] {
            let mut serial = PrefillQueue::new(share);
            let mut paired = PrefillQueue::new(share);
            serial.round_s = 0.0;
            paired.round_s = 0.0;
            for id in 0..4 {
                serial.push(Prompt { id, left: id + 1 });
                paired.push(Prompt { id, left: id + 1 });
            }
            while !serial.is_empty() {
                let mut serial_log = Vec::new();
                let serial_done = run(&mut serial, &mut serial_log, 0);
                let mut pair_log = Vec::new();
                let pair_done = paired.round_pairs(|_, _| false, |batch| pair_step(batch, &mut pair_log))
                    .into_iter().map(|(p, result)| { result.unwrap(); p.id }).collect::<Vec<_>>();
                assert_eq!(pair_log.into_iter().flatten().collect::<Vec<_>>(), serial_log);
                assert_eq!(pair_done, serial_done);
                assert_eq!(paired.waiting.iter().map(|p| (p.id, p.left)).collect::<Vec<_>>(),
                    serial.waiting.iter().map(|p| (p.id, p.left)).collect::<Vec<_>>());
            }
            assert!(paired.is_empty());
        }
    }
}
