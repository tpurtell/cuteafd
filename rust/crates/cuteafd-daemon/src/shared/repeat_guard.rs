//! Exact-repeat loop detection over generated token ids.
//!
//! Design: the loop-detection design §3 (rule R), §9 (rule F) and §10
//! (patience: 8 copies, span 384). Pure and family-agnostic: no engine types,
//! no tokenizer, no allocation beyond the shingle map.
//!
//! Rule R: slide a 32-token shingle over the generated tokens one token at a
//! time; a position is "seen" when that exact shingle already occurred at least
//! `copies - 1` times earlier in the same reply; abort when `run` consecutive
//! positions are seen (the eighth-or-later verbatim copy of a >= 543-token block
//! at the shipped defaults of 8 copies and a run of 512).
//!
//! Rule F: the tail of the reply is `copies` identical copies of a short unit of
//! `period` tokens (`period` in `1..=256`); abort when the copies after the first
//! span at least `max((copies - 1) * period, fast_span)` tokens. Equivalent
//! streaming form: `run[P]` counts consecutive positions with `t[i] == t[i - P]`
//! and fires when `run[P] >= max((copies - 1) * P, fast_span)`.
//!
//! Both rules are off when their knob is zero (`run == 0`, `fast_span == 0`).

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// The shingle width for rule R (design §3: 32 tokens is ~110 chars).
pub(crate) const WINDOW: usize = 32;
/// The ring width for rule F (design §9: `period` in `1..=256`).
pub(crate) const FAST_WINDOW: usize = 256;
/// The most distinct shingles rule R remembers; beyond this it stops inserting
/// and never fires on a new shingle (design §3, memory bound).
pub(crate) const MAX_ENTRIES: usize = 65_536;

/// Which rule fired, so the abort log can name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoopHit {
    /// Rule R: the eighth-or-later verbatim copy of a >= 543-token block.
    /// `period` is the distance between the current shingle and its previous
    /// occurrence (the block length), distinct from the span from the first copy.
    Repeat { period: usize },
    /// Rule F: `copies` identical copies of a short unit of `period` tokens.
    Fast { period: u32 },
}

/// One shingle's history: occurrences so far (saturating at `copies`), the
/// pushed index at which the shingle was first seen, and the pushed index of
/// its most recent occurrence (for the rule-R period).
#[derive(Clone, Copy)]
struct Shingle {
    count: u16,
    first: usize,
    last: usize,
}

/// The per-request loop detector. Construct one at admission from the live
/// knobs; a knob change applies to new requests only (same semantics as
/// `concurrency`).
pub(crate) struct RepeatGuard {
    run: u32,
    copies: u16,
    fast_span: u32,
    fast_copies: u32,
    ring: [u32; WINDOW],
    cursor: usize,
    counts: HashMap<u64, Shingle>,
    consecutive: u32,
    first_seen_at: Option<usize>,
    pushed: usize,
    fast_ring: [u32; FAST_WINDOW],
    fast_cursor: usize,
    fast_len: usize,
    fast_run: [u32; FAST_WINDOW + 1],
}

impl RepeatGuard {
    /// A guard with the four live knobs. `run == 0` disables rule R and
    /// `fast_span == 0` disables rule F. `copies` and `fast_copies` are clamped
    /// into `2..=64` (design §10: the shipped default is 8).
    pub(crate) fn new(run: u32, copies: u32, fast_span: u32, fast_copies: u32) -> Self {
        Self {
            run,
            copies: copies.clamp(2, 64) as u16,
            fast_span,
            fast_copies: fast_copies.clamp(2, 64),
            ring: [0; WINDOW],
            cursor: 0,
            counts: HashMap::new(),
            consecutive: 0,
            first_seen_at: None,
            pushed: 0,
            fast_ring: [0; FAST_WINDOW],
            fast_cursor: 0,
            fast_len: 0,
            fast_run: [0; FAST_WINDOW + 1],
        }
    }

    /// Feed one generated token. Returns the rule that fired, if any.
    pub(crate) fn push(&mut self, token: u32) -> Option<LoopHit> {
        self.pushed += 1;
        let pushed = self.pushed;
        let mut hit = None;

        // The ring is always maintained so `last_ids` is meaningful even when
        // rule R is off (a rule-F abort still logs the tail).
        self.ring[self.cursor] = token;
        self.cursor = (self.cursor + 1) % WINDOW;

        if self.run != 0 && pushed >= WINDOW {
            let hash = self.hash_ring();
            let entry = self.counts.get(&hash).copied();
            let seen = entry.is_some_and(|s| s.count >= self.copies - 1);
            let period = entry.map(|s| pushed - s.last);
            match self.counts.get_mut(&hash) {
                Some(shingle) => {
                    if shingle.count < self.copies {
                        shingle.count += 1;
                    }
                    shingle.last = pushed;
                }
                None => {
                    if self.counts.len() < MAX_ENTRIES {
                        self.counts.insert(hash, Shingle { count: 1, first: pushed - WINDOW, last: pushed - WINDOW });
                    }
                }
            }
            if seen {
                if self.consecutive == 0 {
                    self.first_seen_at = self.counts.get(&hash).map(|s| s.first);
                }
                self.consecutive += 1;
                if self.consecutive >= self.run {
                    hit = Some(LoopHit::Repeat { period: period.unwrap_or(0) });
                }
            } else {
                self.consecutive = 0;
                self.first_seen_at = None;
            }
        }

        if self.fast_span != 0 {
            let mut fired = None;
            for period in 1..=FAST_WINDOW {
                let matches = self.fast_len >= period
                    && token == self.fast_ring[(self.fast_cursor + FAST_WINDOW - period) % FAST_WINDOW];
                if matches {
                    self.fast_run[period] += 1;
                } else {
                    self.fast_run[period] = 0;
                }
                if fired.is_none() {
                    let threshold = (self.fast_copies - 1)
                        .saturating_mul(period as u32)
                        .max(self.fast_span);
                    if self.fast_run[period] >= threshold {
                        fired = Some(period as u32);
                    }
                }
            }
            self.fast_ring[self.fast_cursor] = token;
            self.fast_cursor = (self.fast_cursor + 1) % FAST_WINDOW;
            if self.fast_len < FAST_WINDOW {
                self.fast_len += 1;
            }
            if let Some(period) = fired {
                if hit.is_none() {
                    hit = Some(LoopHit::Fast { period });
                }
            }
        }

        hit
    }

    /// The current run of consecutive seen positions (rule R).
    pub(crate) fn consecutive(&self) -> u32 {
        self.consecutive
    }

    /// The distance from the first copy of the current run to now (rule R).
    pub(crate) fn period_hint(&self) -> Option<usize> {
        self.first_seen_at.map(|first| self.pushed - first)
    }

    /// The last 32 generated token ids, oldest first (for the abort log line).
    pub(crate) fn last_ids(&self) -> [u32; WINDOW] {
        let mut ids = [0u32; WINDOW];
        for (i, id) in ids.iter_mut().enumerate() {
            *id = self.ring[(self.cursor + i) % WINDOW];
        }
        ids
    }

    fn hash_ring(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        for i in 0..WINDOW {
            self.ring[(self.cursor + i) % WINDOW].hash(&mut hasher);
        }
        hasher.finish()
    }
}

impl Default for RepeatGuard {
    /// Disabled: neither rule fires.
    fn default() -> Self {
        Self::new(0, 8, 0, 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic token-id streams (no real session text or ids), built from the
    /// deterministic `block()` generator below.
    struct Fixture {
        looped: Vec<u32>,
        draft_then_write: Vec<u32>,
        fast_loop: Vec<u32>,
    }

    fn fixture() -> Fixture {
        // looped: 200 distinct prefix ids, then a 640-id block x 19.
        let mut looped = block(200, 101);
        looped.extend(repeat(&block(640, 102), 19));
        // draft_then_write: A(1500) + B(1500) + A once more + 500 distinct ids
        // (two copies of A, so the guard never fires).
        let a = block(1500, 103);
        let mut draft_then_write = a.clone();
        draft_then_write.extend(block(1500, 104));
        draft_then_write.extend_from_slice(&a);
        draft_then_write.extend(block(500, 105));
        // fast_loop: 300 prefix ids, then a 24-id unit x 80.
        let mut fast_loop = block(300, 106);
        fast_loop.extend(repeat(&block(24, 107), 80));
        Fixture { looped, draft_then_write, fast_loop }
    }

    #[test]
    fn fixture_is_synthetic() {
        // Pin the generator output so nobody re-pastes real ids.
        let f = fixture();
        assert_eq!(f.looped.len(), 200 + 640 * 19);
        assert_eq!(f.draft_then_write.len(), 1500 + 1500 + 1500 + 500);
        assert_eq!(f.fast_loop.len(), 300 + 24 * 80);
        assert_eq!(&f.looped[..200], &block(200, 101)[..]);
        assert_eq!(&f.looped[200..840], &block(640, 102)[..]);
        assert_eq!(&f.draft_then_write[..1500], &block(1500, 103)[..]);
        assert_eq!(&f.draft_then_write[3000..4500], &block(1500, 103)[..]);
        assert_eq!(&f.fast_loop[..300], &block(300, 106)[..]);
        assert_eq!(&f.fast_loop[300..324], &block(24, 107)[..]);
    }

    fn shipped() -> RepeatGuard {
        RepeatGuard::new(512, 8, 384, 8)
    }

    fn feed(guard: &mut RepeatGuard, ids: &[u32]) -> Option<(usize, LoopHit)> {
        for (i, &token) in ids.iter().enumerate() {
            if let Some(hit) = guard.push(token) {
                return Some((i + 1, hit));
            }
        }
        None
    }

    /// A deterministic pseudo-random block of distinct ids.
    fn block(len: usize, seed: u64) -> Vec<u32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (x >> 33) as u32
            })
            .collect()
    }

    fn repeat(unit: &[u32], copies: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity(unit.len() * copies);
        for _ in 0..copies {
            ids.extend_from_slice(unit);
        }
        ids
    }

    #[test]
    fn disabled_guard_never_fires() {
        let mut guard = RepeatGuard::new(0, 8, 0, 8);
        assert!(feed(&mut guard, &fixture().looped).is_none());
    }

    #[test]
    fn second_copy_does_not_fire() {
        let unit = block(600, 1);
        let mut guard = shipped();
        assert!(feed(&mut guard, &repeat(&unit, 2)).is_none());
    }

    #[test]
    fn seven_copies_never_fire() {
        let unit = block(600, 2);
        let mut guard = shipped();
        assert!(feed(&mut guard, &repeat(&unit, 7)).is_none());
    }

    #[test]
    fn eighth_copy_fires_at_run() {
        let unit = block(600, 3);
        let mut guard = shipped();
        let (at, hit) = feed(&mut guard, &repeat(&unit, 9)).expect("the eighth copy fires");
        assert!(matches!(hit, LoopHit::Repeat { .. }), "rule R fires, got {hit:?}");
        // The first seen position is 7 * 600 + 32 (the shingle needs 32 tokens);
        // the run of 512 ends 511 positions later.
        assert_eq!(at, 7 * 600 + 31 + 512);
    }

    #[test]
    fn repeat_hit_reports_the_block_period() {
        let unit = block(600, 9);
        let mut guard = shipped();
        let (_, hit) = feed(&mut guard, &repeat(&unit, 9)).expect("rule R fires");
        // The period is the distance between copies (600), not the span from the
        // first copy (which `period_hint` reports).
        assert_eq!(hit, LoopHit::Repeat { period: 600 });
        assert!(guard.period_hint().is_some_and(|span| span > 600));
    }

    #[test]
    fn looped_fixture_fires_before_token_32000() {
        let mut guard = RepeatGuard::new(512, 8, 0, 8);
        let (at, hit) = feed(&mut guard, &fixture().looped).expect("the looped fixture fires");
        assert!(matches!(hit, LoopHit::Repeat { .. }), "rule R fires, got {hit:?}");
        // Later than the LD-05b (seen = 2) point of 3,127, and well inside the cap.
        assert!(at > 3_127, "fires at {at}, expected later than the LD-05b point");
        assert!(at < 32_000, "fires at {at}, expected before the 32k cap");
    }

    #[test]
    fn draft_then_write_fixture_never_fires() {
        let mut guard = shipped();
        assert!(feed(&mut guard, &fixture().draft_then_write).is_none());
    }

    #[test]
    fn window_shorter_than_32_never_fires() {
        let mut guard = RepeatGuard::new(512, 8, 0, 8);
        assert!(feed(&mut guard, &block(31, 4)).is_none());
    }

    #[test]
    fn map_is_bounded() {
        let mut guard = RepeatGuard::new(512, 8, 0, 8);
        let total = MAX_ENTRIES + 100 + WINDOW;
        for i in 0..total {
            assert!(guard.push(i as u32).is_none(), "distinct shingles never fire");
        }
        assert_eq!(guard.counts.len(), MAX_ENTRIES);
        // Beyond the cap nothing more is inserted.
        for i in 0..1_000 {
            assert!(guard.push((total + i) as u32).is_none());
        }
        assert_eq!(guard.counts.len(), MAX_ENTRIES);
    }

    #[test]
    fn fast_eight_copies_fire_at_span() {
        let unit = block(30, 5);
        let mut guard = shipped();
        let (at, hit) = feed(&mut guard, &repeat(&unit, 20)).expect("rule F fires");
        assert_eq!(hit, LoopHit::Fast { period: 30 });
        assert_eq!(at, 30 + 384);
    }

    #[test]
    fn fast_long_unit_needs_eight_copies() {
        let unit = block(100, 6);
        let mut guard = shipped();
        assert!(feed(&mut guard, &repeat(&unit, 7)).is_none());
        let mut guard = shipped();
        let (at, hit) = feed(&mut guard, &repeat(&unit, 9)).expect("rule F fires");
        assert_eq!(hit, LoopHit::Fast { period: 100 });
        assert_eq!(at, 100 + 700);
    }

    #[test]
    fn fast_same_token_385_fires_384_not() {
        let mut guard = shipped();
        assert!(feed(&mut guard, &[7u32; 384]).is_none());
        let mut guard = shipped();
        let (at, hit) = feed(&mut guard, &[7u32; 385]).expect("rule F fires");
        assert_eq!(hit, LoopHit::Fast { period: 1 });
        assert_eq!(at, 385);
    }

    #[test]
    fn fast_off_when_span_zero() {
        let unit = block(30, 7);
        let mut guard = RepeatGuard::new(0, 8, 0, 8);
        assert!(feed(&mut guard, &repeat(&unit, 20)).is_none());
    }

    #[test]
    fn fast_three_copies_never_fire() {
        let unit = block(30, 8);
        let mut guard = shipped();
        assert!(feed(&mut guard, &repeat(&unit, 3)).is_none());
    }

    #[test]
    fn fast_loop_fixture_fires_after_the_ld05b_point() {
        let mut guard = RepeatGuard::new(0, 8, 384, 8);
        let (at, hit) = feed(&mut guard, &fixture().fast_loop).expect("the fast fixture fires");
        assert!(matches!(hit, LoopHit::Fast { .. }), "rule F fires, got {hit:?}");
        // LD-05b (K = 4, span 192) fired at 294; patience moves it later.
        assert!(at > 294, "fires at {at}, expected later than the LD-05b point");
        assert!(at < 32_000, "fires at {at}, expected before the 32k cap");
    }

    /// The committed fixture is the same synthetic streams the tests generate,
    /// so the evidence is reproducible without a GPU or a live engine.
    #[test]
    fn committed_fixture_matches_the_generated_streams() {
        let data: serde_json::Value = serde_json::from_str(include_str!("repeat_guard_fixture.json"))
            .expect("the committed fixture parses");
        let f = fixture();
        let ids = |name: &str| -> Vec<u32> {
            data[name].as_array().expect("an id array").iter()
                .map(|v| v.as_u64().expect("an integer id") as u32).collect()
        };
        assert_eq!(ids("looped"), f.looped);
        assert_eq!(ids("draft_then_write"), f.draft_then_write);
        assert_eq!(ids("fast_loop"), f.fast_loop);
    }

    #[test]
    fn last_ids_are_kept_when_rule_r_is_off() {
        // A rule-F-only guard still logs a meaningful tail (the review's
        // `last_ids=[0;32]` bug).
        let mut guard = RepeatGuard::new(0, 8, 384, 8);
        let ids: Vec<u32> = (1..=40).collect();
        for &id in &ids {
            guard.push(id);
        }
        let expected: [u32; WINDOW] = ids[8..].try_into().unwrap();
        assert_eq!(guard.last_ids(), expected);
    }
}
