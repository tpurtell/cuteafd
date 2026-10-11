//! Two copy searches over a request's own history.
//!
//! Whether a copy span is used is the shared draft policy's acceptance-gated
//! decision (PLAN decision 9). Copies are opt-in per family.
//!
//! The rule comes from ashhart/TensorFold (MIT); this is a reimplementation from
//! its description.
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Tokens that must match before a copy is proposed. Coincidental shorter
/// matches (indentation, `self.`) cost more verification than they return.
pub const WINDOW: usize = 8;
/// A copy proposal carries at least this many tokens.
pub const MIN_DRAFTS: usize = 2;
/// Rounds without copy proposals after a copy round that accepted nothing.
pub const BACKOFF_ROUNDS: u8 = 2;

/// Most tokens one copy round may propose, the bound dSpark's proposals obey:
/// `max_drafts` is the draft limit (at most the loaded draft width), and a round
/// always emits one token beyond its accepted drafts.
pub fn cap(remaining: usize, max_drafts: usize) -> usize {
    remaining.saturating_sub(1).min(max_drafts)
}

/// V4.1's latest-occurrence eight-token-window policy.
///
/// One request's copy windows: the absolute end position of the latest
/// occurrence of each eight-token window of its history, keyed by a 64-bit
/// hash (16 bytes an entry, no per-token allocation), built on first use and
/// extended as the history grows; plus the request's backoff.
#[derive(Default)]
pub struct LatestWindow {
    latest: HashMap<u64, usize, BuildHasherDefault<KeyHasher>>,
    /// Windows that end before this position are indexed.
    indexed: usize,
    /// Rounds left before copies are proposed again.
    backoff: u8,
}

impl LatestWindow {
    /// This round's copied drafts, or `None`. `history` is the request's whole
    /// token history, prompt then emitted tokens, ending with the anchor; it
    /// only grows between calls. Call once per decode round: each call counts
    /// one backoff round.
    ///
    /// The drafts are up to `cap` tokens that followed the most recent earlier
    /// occurrence of the history's final eight tokens. When that occurrence ends
    /// fewer than `cap` tokens before the anchor, its continuation runs into
    /// the tokens being proposed, so copying continues from the same distance:
    /// what re-matching after each accepted token would propose.
    pub fn propose(&mut self, history: &[u32], cap: usize) -> Option<Vec<u32>> {
        if self.backoff > 0 {
            self.backoff -= 1;
            return None;
        }
        if cap < MIN_DRAFTS || history.len() <= WINDOW {
            return None;
        }
        let end = history.len() - 1;
        // Index every window that ends before the anchor. The current suffix
        // stays out until a later round, so a lookup finds only an earlier
        // occurrence.
        for last in self.indexed.max(WINDOW - 1)..end {
            self.latest.insert(key(&history[last + 1 - WINDOW..=last]), last);
        }
        self.indexed = self.indexed.max(end);
        let suffix = &history[end + 1 - WINDOW..];
        let earlier = *self.latest.get(&key(suffix))?;
        let distance = end.checked_sub(earlier).filter(|&distance| distance > 0)?;
        // Distinct windows can share a key: copy only after the tokens match.
        if history.get(earlier + 1 - WINDOW..=earlier)? != suffix {
            return None;
        }
        Some((0..cap).map(|index| history[earlier + 1 + index % distance]).collect())
    }

    /// Close a verified copy round in which `accepted` copied tokens matched.
    pub fn observe(&mut self, accepted: usize) {
        if accepted == 0 {
            self.backoff = BACKOFF_ROUNDS;
        }
    }
}

/// A lane round's verify inputs in member order: each member's copy where it
/// has one, otherwise the drafter's next output. The drafter drafted exactly
/// the members without a copy, in order.
pub fn merge(copies: Vec<Option<Vec<u32>>>, drafted: Vec<Vec<u32>>) -> Result<Vec<Vec<u32>>> {
    let mut drafted = drafted.into_iter();
    let merged = copies.into_iter().map(|copy| copy.or_else(|| drafted.next()))
        .collect::<Option<Vec<_>>>().context("fewer draft outputs than drafting requests")?;
    ensure!(drafted.next().is_none(), "more draft outputs than drafting requests");
    Ok(merged)
}

/// A window's key: 64 well-mixed bits. Vocabulary ids fit in 32 bits.
fn key(window: &[u32]) -> u64 {
    window.chunks_exact(2).fold(0x243F_6A88_85A3_08D3, |hash, pair| {
        mix(hash ^ (u64::from(pair[0]) | (u64::from(pair[1]) << 32)))
    })
}

/// SplitMix64's finalizer: a bijection that spreads every input bit.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Window keys are already mixed, so the index uses them as they are.
#[derive(Default)]
struct KeyHasher(u64);
impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = mix(self.0 ^ u64::from(byte));
        }
    }
    fn write_u64(&mut self, key: u64) {
        self.0 = key;
    }
}

#[cfg(test)]
mod latest_window_tests {
    use super::*;

    /// Brute force: search backwards for the latest earlier occurrence of the
    /// final window, then copy one token at a time, each read from the
    /// history as extended by the tokens already copied.
    fn reference(history: &[u32], cap: usize) -> Option<Vec<u32>> {
        if cap < MIN_DRAFTS || history.len() <= WINDOW {
            return None;
        }
        let end = history.len() - 1;
        let suffix = &history[end + 1 - WINDOW..];
        let earlier = (WINDOW - 1..end).rev().find(|&last| &history[last + 1 - WINDOW..=last] == suffix)?;
        let mut extended = history.to_vec();
        for index in 0..cap {
            extended.push(extended[earlier + 1 + index]);
        }
        Some(extended[end + 1..].to_vec())
    }

    /// Distinct filler tokens that cannot match anything else in a test.
    fn filler(from: u32, count: u32) -> Vec<u32> {
        (from..from + count).collect()
    }

    #[test]
    fn copies_what_followed_the_latest_earlier_window() {
        let window: Vec<u32> = (1..=8).collect();
        let mut history = filler(1000, 5);
        history.extend(&window);
        history.extend([50, 51, 52, 53, 54, 55, 56, 57, 58]);
        history.extend(filler(2000, 4));
        history.extend(&window);
        history.extend([60, 61, 62, 63, 64, 65, 66, 67, 68]);
        history.extend(filler(3000, 4));
        history.extend(&window);
        // Two earlier occurrences: the later one's continuation is copied.
        assert_eq!(LatestWindow::default().propose(&history, 5), Some(vec![60, 61, 62, 63, 64]));
        assert_eq!(LatestWindow::default().propose(&history, 7), Some(vec![60, 61, 62, 63, 64, 65, 66]));
        assert_eq!(reference(&history, 7), Some(vec![60, 61, 62, 63, 64, 65, 66]));
    }

    #[test]
    fn enters_only_on_eight_matching_tokens() {
        let mut history = filler(1000, 5);
        history.extend(1..=8);
        history.extend([50, 51, 52]);
        history.extend(filler(2000, 3));
        // The final seven tokens occurred earlier; the eighth from the end did not.
        let mut seven = history.clone();
        seven.push(99);
        seven.extend(2..=8);
        assert_eq!(LatestWindow::default().propose(&seven, 7), None);
        let mut eight = history;
        eight.extend(1..=8);
        assert_eq!(LatestWindow::default().propose(&eight, 7), Some(vec![50, 51, 52, 2000, 2001, 2002, 1]));
        // Too short a history to hold an earlier window.
        assert_eq!(LatestWindow::default().propose(&[7; 8], 7), None);
        assert_eq!(LatestWindow::default().propose(&[7; 9], 7), Some(vec![7; 7]));
    }

    #[test]
    fn proposes_two_to_cap_tokens() {
        let mut history = filler(1000, 3);
        history.extend(1..=8);
        history.extend(filler(50, 10));
        history.extend(1..=8);
        for cap in 0..MIN_DRAFTS {
            assert_eq!(LatestWindow::default().propose(&history, cap), None, "cap {cap}");
        }
        for cap in MIN_DRAFTS..=7 {
            let drafts = LatestWindow::default().propose(&history, cap).unwrap();
            assert_eq!(drafts, filler(50, cap as u32), "cap {cap}");
        }
    }

    /// A source that ends just before the anchor overlaps the suffix; its
    /// continuation keeps the source's period instead of stopping at the anchor.
    #[test]
    fn overlapping_sources_repeat_their_period() {
        // A run: the latest earlier window ends one token before the anchor.
        let mut run = filler(1000, 4);
        run.extend([5; 12]);
        assert_eq!(LatestWindow::default().propose(&run, 7), Some(vec![5; 7]));
        // Period three: "a b c a b c ..." continues "a b c ..." from the anchor.
        let mut period = filler(1000, 4);
        period.extend([1, 2, 3].repeat(4));
        assert_eq!(period.last(), Some(&3));
        assert_eq!(LatestWindow::default().propose(&period, 7), Some(vec![1, 2, 3, 1, 2, 3, 1]));
        // Period five against a cap of three: nothing wraps.
        let mut five = filler(1000, 4);
        five.extend([1, 2, 3, 4, 5].repeat(3));
        assert_eq!(LatestWindow::default().propose(&five, 3), Some(vec![1, 2, 3]));
        for history in [&run, &period, &five] {
            for cap in 0..=7 {
                assert_eq!(LatestWindow::default().propose(history, cap), reference(history, cap));
            }
        }
    }

    /// Positions are absolute and the current suffix never matches itself.
    #[test]
    fn a_window_seen_only_at_the_end_has_no_copy() {
        let mut drafter = LatestWindow::default();
        let history = filler(1000, 40);
        for end in 1..=history.len() {
            assert_eq!(drafter.propose(&history[..end], 7), None, "prefix {end}");
        }
        // The same window again, 40 tokens later, copies from the first.
        let mut repeated = history.clone();
        repeated.extend(filler(1000, 8));
        assert_eq!(drafter.propose(&repeated, 7), Some(filler(1008, 7)));
    }

    /// A key shared by two different windows must not propose the other
    /// window's continuation. A 64-bit collision cannot be found on demand, so
    /// the test plants one.
    #[test]
    fn a_hash_collision_is_rejected_by_the_token_check() {
        let mut history = filler(1000, 30);
        history.extend(1..=8);
        let mut drafter = LatestWindow::default();
        assert_eq!(drafter.propose(&history, 7), None);
        let suffix = &history[history.len() - WINDOW..];
        // Point the suffix's key at a different window that ends earlier.
        drafter.latest.insert(key(suffix), 20);
        assert_eq!(drafter.propose(&history, 7), None);
        // The same entry at a genuine occurrence is used.
        let mut genuine = filler(1000, 12);
        genuine.extend(1..=8);
        genuine.extend(filler(2000, 10));
        genuine.extend(1..=8);
        let mut drafter = LatestWindow::default();
        assert_eq!(drafter.propose(&genuine, 3), Some(filler(2000, 3)));
        assert_eq!(drafter.latest.get(&key(&genuine[genuine.len() - WINDOW..])), Some(&19));
    }

    #[test]
    fn backs_off_two_rounds_after_a_copy_accepts_nothing() {
        let mut history = filler(1000, 3);
        history.extend(1..=8);
        history.extend(filler(50, 10));
        history.extend(1..=8);
        let mut drafter = LatestWindow::default();
        assert!(drafter.propose(&history, 7).is_some());
        drafter.observe(1);
        assert!(drafter.propose(&history, 7).is_some(), "an accepted copy keeps copying");
        drafter.observe(0);
        assert_eq!(drafter.propose(&history, 7), None, "first backoff round");
        assert_eq!(drafter.propose(&history, 7), None, "second backoff round");
        assert!(drafter.propose(&history, 7).is_some(), "copying resumes on the third round");
        // Rounds that could not copy anyway still count toward the backoff.
        drafter.observe(0);
        assert_eq!(drafter.propose(&history, 0), None);
        assert_eq!(drafter.propose(&history, 1), None);
        assert!(drafter.propose(&history, 7).is_some());
    }

    #[test]
    fn the_cap_follows_the_remaining_budget_and_the_draft_limit() {
        // A round emits one token beyond its accepted drafts.
        assert_eq!(cap(0, 7), 0);
        assert_eq!(cap(1, 7), 0);
        assert_eq!(cap(2, 7), 1);
        assert_eq!(cap(3, 7), 2);
        assert_eq!(cap(8, 7), 7);
        assert_eq!(cap(1000, 7), 7);
        assert_eq!(cap(1000, 5), 5);
        // A budget of two leaves one draft: below the minimum, so no copy.
        let history = [5u32; 20];
        assert_eq!(LatestWindow::default().propose(&history, cap(2, 7)), None);
        assert_eq!(LatestWindow::default().propose(&history, cap(3, 7)), Some(vec![5, 5]));
        assert_eq!(LatestWindow::default().propose(&history, cap(1000, 5)), Some(vec![5; 5]));
    }

    /// The index built on first use and extended round by round agrees with a
    /// brute-force search at every length, and holds at most one entry per
    /// indexed window.
    #[test]
    fn the_incremental_index_matches_a_brute_force_search() {
        // A small alphabet repeats windows often; copied spans add long matches.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let mut history: Vec<u32> = Vec::new();
        while history.len() < 3000 {
            if history.len() > 64 && next(8) == 0 {
                let start = next(history.len() as u64 - 32) as usize;
                let span = 8 + next(24) as usize;
                let copied = history[start..start + span].to_vec();
                history.extend(copied);
            } else {
                history.push(next(3) as u32);
            }
        }
        let mut drafter = LatestWindow::default();
        let mut proposed = 0;
        for end in 1..=history.len() {
            let cap = end % 8;
            let drafts = drafter.propose(&history[..end], cap);
            assert_eq!(drafts, reference(&history[..end], cap), "prefix {end} cap {cap}");
            proposed += usize::from(drafts.is_some());
            assert!(drafter.latest.len() <= end.saturating_sub(WINDOW));
        }
        assert!(proposed > 1000, "the fixture exercises copies ({proposed})");
        // A drafter that first sees the whole history agrees too.
        assert_eq!(LatestWindow::default().propose(&history, 7), reference(&history, 7));
    }

    #[test]
    fn merge_keeps_member_order() {
        let copies = vec![Some(vec![1, 2, 3]), None, Some(vec![4, 5, 6]), None];
        let drafted = vec![vec![7, 8], vec![9]];
        assert_eq!(merge(copies.clone(), drafted).unwrap(),
            vec![vec![1, 2, 3], vec![7, 8], vec![4, 5, 6], vec![9]]);
        // Without copies the drafter's outputs pass through unchanged.
        assert_eq!(merge(vec![None, None], vec![vec![1], vec![2, 3]]).unwrap(), vec![vec![1], vec![2, 3]]);
        assert!(merge(copies.clone(), vec![vec![7]]).is_err());
        assert!(merge(copies, vec![vec![7], vec![8], vec![9]]).is_err());
    }
}

// Ported from Hugh Madden's mimo26f-afd v1.3.0, crates/mimo26-coordinator/src/copy.rs (MIT).
// Copyright (c) 2026 Turquoise Bay AI Pty Ltd
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/// Context tokens that must match before a copy is proposed.
pub const MATCH: usize = 8;
/// Backward match length counted up to this when choosing among occurrences.
const EXTEND: usize = 64;
/// Earlier occurrences tried per proposal, newest first.
const CANDIDATES: usize = 64;
const NONE: u32 = u32::MAX;

/// A request's index of the [`MATCH`]-grams in its context, extended as the context grows.
#[derive(Default)]
pub struct LongestBackward {
    /// The last position (the gram's final token) of each gram seen.
    head: HashMap<[u32; MATCH], u32>,
    /// `prev[p]`: the previous position with the gram ending at `p`, or [`NONE`].
    prev: Vec<u32>,
}

impl LongestBackward {
    /// Up to `k` tokens to copy after `ctx`, or none when `k < 2` or the last [`MATCH`] tokens of
    /// `ctx` did not occur earlier. Positions before `ctx.len() - 1` are indexed first, so the
    /// context must only grow between calls (a request's history does).
    pub fn propose(&mut self, ctx: &[u32], k: usize) -> Vec<u32> {
        let n = ctx.len();
        if k < 2 || n <= MATCH {
            return Vec::new();
        }
        // Index every gram with at least one token after it.
        for p in self.prev.len()..n - 1 {
            if p + 1 < MATCH {
                self.prev.push(NONE);
                continue;
            }
            let g = gram(&ctx[p + 1 - MATCH..=p]);
            self.prev.push(self.head.insert(g, p as u32).unwrap_or(NONE));
        }
        let Some(&newest) = self.head.get(&gram(&ctx[n - MATCH..])) else {
            return Vec::new();
        };
        // The occurrence matching furthest back (ties: the most recent).
        let (mut best, mut best_len, mut p, mut tried) = (NONE, 0usize, newest, 0usize);
        while p != NONE && tried < CANDIDATES {
            let q = p as usize;
            let mut len = MATCH;
            while len < EXTEND && len <= q && ctx[q - len] == ctx[n - 1 - len] {
                len += 1;
            }
            if len > best_len {
                (best, best_len) = (p, len);
                if len == EXTEND {
                    break;
                }
            }
            p = self.prev[q];
            tried += 1;
        }
        let from = best as usize + 1;
        let mut out = Vec::with_capacity(k);
        for j in 0..k {
            let t = if from + j < n { ctx[from + j] } else { out[from + j - n] };
            out.push(t);
        }
        out
    }
}

fn gram(t: &[u32]) -> [u32; MATCH] {
    std::array::from_fn(|i| t[i])
}

#[cfg(test)]
mod longest_backward_tests {
    use super::*;

    fn ids(s: &str) -> Vec<u32> {
        s.bytes().map(u32::from).collect()
    }

    #[test]
    fn copies_what_followed_an_earlier_match() {
        let mut c = LongestBackward::default();
        let ctx = ids("fn alpha(x) { return x + 1; }\n// again: fn alpha(x) { ret");
        assert_eq!(c.propose(&ctx, 7), ids("urn x +"));
        // Short matches do not count.
        let mut c = LongestBackward::default();
        assert!(c.propose(&ids("abcdefg xyz abcdefg"), 7).is_empty());
        // Nothing when k < 2 or the context is short.
        let mut c = LongestBackward::default();
        assert!(c.propose(&ctx, 1).is_empty());
        assert!(c.propose(&ctx[..MATCH], 7).is_empty());
    }

    #[test]
    fn longest_backward_match_wins_then_most_recent() {
        // "12345678" occurs twice; only the first occurrence is preceded by "XY", as the tail is.
        let mut c = LongestBackward::default();
        let ctx = ids("XY12345678AB..zz12345678CD..XY12345678");
        assert_eq!(c.propose(&ctx, 2), ids("AB"));
        // Equal evidence: the most recent occurrence.
        let mut c = LongestBackward::default();
        let ctx = ids("..12345678AB..12345678CD..12345678");
        assert_eq!(c.propose(&ctx, 2), ids("CD"));
    }

    #[test]
    fn a_growing_context_is_indexed_incrementally() {
        let mut c = LongestBackward::default();
        let mut ctx = ids("the quick brown fox jumps over the lazy dog; ");
        for t in ids("the quick") {
            ctx.push(t);
            let got = c.propose(&ctx, 7);
            let full = LongestBackward::default().propose(&ctx, 7);
            assert_eq!(got, full);
        }
        assert_eq!(c.propose(&ctx, 7), ids(" brown "));
    }

    #[test]
    fn a_periodic_run_repeats() {
        let mut c = LongestBackward::default();
        let ctx = ids("start: ab-ab-ab-ab-ab");
        assert_eq!(c.propose(&ctx, 7), ids("-ab-ab-"));
    }
}
