//! Token-prefix radix retention shared by the in-memory snapshot cache
//! (`cuteafd-daemon::v41_native_serve::prefix`) and the disk persistence index (`cuteafd-persist`).
//!
//! Moved here verbatim from the daemon so both tiers apply one reuse rule: the most computation
//! saved wins, where a partial match replays from an even-aligned position minus a 128-token
//! window and a populated exact ancestor can beat a slightly longer partial match
//! (`Reusable::skipped`). Prompt repeats and completed turns keep separate bounded banks
//! (`Retention`); a tie prefers the completed turn.
//!
//! The reuse rule is a parameter ([`ReuseRule`]) so the generic engine's families share this
//! radix: V4.1 keeps `ReuseRule::V41` (`Radix::new`, `Retention::new`), paged families resume at
//! a page boundary, and families whose snapshot carries positional state reuse exact frontiers
//! only.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bounded retained-prefix host quota, independent of currently free RAM.
pub const fn default_host_cache_bytes(total: u64) -> u64 {
    let quarter = total / 4;
    if quarter < 32 << 30 { quarter } else { 32 << 30 }
}

/// Full-context inactive KV quota, with four sequences of overage. Explicit quotas bypass it.
pub fn host_cache_required_bytes(context: u64, concurrency: u64,
    unit_rows: u64, page_bytes: u64, mark_bytes: u64, draft_bytes: u64, chunk: u64) -> Option<u64> {
    if context == 0 || unit_rows == 0 || page_bytes == 0 || chunk == 0 { return None; }
    let snapshots = concurrency.checked_add(4)?;
    let pages = context.div_ceil(unit_rows).checked_mul(snapshots)?;
    let mut chunks = 0u64;
    for (count, bytes) in [(pages, page_bytes), (snapshots.checked_mul(2)?, mark_bytes), (snapshots, draft_bytes)] {
        if bytes == 0 { continue; }
        let per = chunk / bytes;
        if per == 0 { return None; }
        chunks = chunks.checked_add(count.div_ceil(per))?;
    }
    chunks.checked_mul(chunk)
}

/// Snapshot size classes shared by offline admission and the live host tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCacheGeometry {
    pub unit_rows: u64,
    pub page_bytes: u64,
    pub mark_bytes: u64,
    pub draft_bytes: u64,
}

/// Extra KV kept free after a pressure eviction, independently of the quota.
pub const DEFAULT_HOST_CACHE_EVICT_TOKENS: u64 = 16 * 1024;

impl HostCacheGeometry {
    /// Page, positional-mark and draft size-class bytes for one headroom extent.
    pub fn eviction_headroom_bytes(self, tokens: u64) -> Option<[u64; 3]> {
        if tokens == 0 { return Some([0; 3]); }
        if self.unit_rows == 0 { return None; }
        Some([tokens.div_ceil(self.unit_rows).checked_mul(self.page_bytes)?,
            self.mark_bytes.max(1), self.draft_bytes])
    }

    pub fn chunk_bytes(self) -> u64 {
        (256u64 << 20).max(self.page_bytes).max(self.mark_bytes).max(self.draft_bytes)
    }

    /// Whole conversations (shared prompt/turn KV extent, two independent positional marks).
    pub fn conversations(self, bytes: u64, context: u64, chunk: u64) -> u64 {
        if context == 0 || chunk == 0 { return 0; }
        // The same allocator size classes/rounding as admission; binary search avoids overflow.
        let fits = |n: u64| {
            let Some(pages) = context.checked_add(self.unit_rows.saturating_sub(1))
                .and_then(|rows| rows.checked_div(self.unit_rows)).and_then(|pages| pages.checked_mul(n)) else { return false };
            let Some(marks) = n.checked_mul(2) else { return false };
            let mut chunks = 0u64;
            for (count, size) in [(pages, self.page_bytes), (marks, self.mark_bytes.max(1)), (n, self.draft_bytes)] {
                if size == 0 { continue; }
                let per = chunk / size;
                if per == 0 { return false; }
                let Some(total) = chunks.checked_add(count.div_ceil(per)) else { return false };
                chunks = total;
            }
            chunks.checked_mul(chunk).is_some_and(|needed| needed <= bytes)
        };
        let (mut low, mut high) = (0u64, bytes);
        while low < high {
            let mid = low + (high - low) / 2 + 1;
            if fits(mid) { low = mid; } else { high = mid - 1; }
        }
        low
    }

    /// Explicit quotas bypass geometry and caps; automatic quotas hold whole chunks.
    pub fn resolve(self, explicit: Option<u64>, context: u64, concurrency: u64,
        host_total: u64, chunk: u64) -> Option<u64> {
        if let Some(bytes) = explicit { return Some(bytes); }
        let required = host_cache_required_bytes(context, concurrency, self.unit_rows,
            self.page_bytes, self.mark_bytes.max(1), self.draft_bytes, chunk)?;
        Some(required.min(default_host_cache_bytes(host_total)) / chunk * chunk)
    }
}

#[cfg(test)]
mod host_budget_tests {
    use super::*;
    #[test]
    fn eviction_headroom_uses_the_resolved_family_classes() {
        let geometry = HostCacheGeometry { unit_rows: 512, page_bytes: 455_680,
            mark_bytes: 2_922_816, draft_bytes: 12_288 };
        assert_eq!(geometry.eviction_headroom_bytes(DEFAULT_HOST_CACHE_EVICT_TOKENS),
            Some([32 * 455_680, 2_922_816, 12_288]));
        assert_eq!(geometry.eviction_headroom_bytes(0), Some([0; 3]));
        assert_eq!(geometry.eviction_headroom_bytes(513), Some([2 * 455_680, 2_922_816, 12_288]));
    }

    #[test]
    fn shared_resolver_preserves_explicit_overrides_and_whole_chunk_caps() {
        let geometry = HostCacheGeometry { unit_rows: 64, page_bytes: 512,
            mark_bytes: 140, draft_bytes: 0 };
        assert_eq!(geometry.resolve(None, 65, 2, 32768, 1024), Some(8 * 1024));
        assert_eq!(geometry.resolve(None, 4096, 16, 32768, 1024), Some(8 * 1024));
        assert_eq!(geometry.resolve(Some(40001), 0, u64::MAX, 1, 0), Some(40001));
        assert_eq!(geometry.resolve(Some(0), 0, 0, 0, 0), Some(0));
        assert_eq!(geometry.resolve(None, 0, 1, 32768, 1024), None);
    }

    #[test]
    fn concurrency_default_and_small_host_cap() {
        let required = |concurrency| host_cache_required_bytes(1 << 20, concurrency, 512, 4096, 1024, 0, 1 << 20).unwrap();
        assert_eq!(required(1), 41 << 20);
        assert_eq!(required(16), 161 << 20);
        // Tail and draft are distinct size classes, each requiring its own chunk.
        assert_eq!(host_cache_required_bytes(1, 1, 1, 1, 1, 1, 1024), Some(3072));
        let small_host = 128 << 20;
        assert_eq!(required(16).min(default_host_cache_bytes(small_host)), 32 << 20);
        assert_eq!(default_host_cache_bytes(256 << 30), 32 << 30);
        assert!(host_cache_required_bytes(u64::MAX, u64::MAX, 1, 4096, 1024, 0, 1 << 20).is_none());
    }
}

/// Position-addressed window with enough append-only space for one uncommitted step.
/// Snapshots retain only `window` visible rows, never speculative slack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendRing {
    pub window: u64,
    pub step: u64,
    pub rows: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingRun {
    pub slot: u64,
    pub offset: u64,
    pub rows: u64,
}

impl AppendRing {
    pub fn new(window: u64, step: u64, alignment: u64) -> Option<Self> {
        if window == 0 || step == 0 || alignment == 0 { return None; }
        let rows = window.checked_add(step)?.checked_add(alignment - 1)?
            .checked_div(alignment)?.checked_mul(alignment)?;
        Some(Self { window, step, rows })
    }

    /// At most two contiguous copies, with offsets in logical position order.
    pub fn runs(self, start: u64, rows: u64) -> Option<Vec<RingRun>> {
        start.checked_add(rows)?;
        if rows > self.rows { return None; }
        if rows == 0 { return Some(Vec::new()); }
        let slot = start % self.rows;
        let first = rows.min(self.rows - slot);
        let mut runs = vec![RingRun { slot, offset: 0, rows: first }];
        if first < rows { runs.push(RingRun { slot: 0, offset: first, rows: rows - first }); }
        Some(runs)
    }

    pub fn visible(self, frontier: u64) -> Vec<RingRun> {
        let rows = frontier.min(self.window);
        self.runs(frontier - rows, rows).expect("visible window fits the ring")
    }

    pub fn proposal(self, base: u64, rows: u64) -> Option<Vec<RingRun>> {
        if rows > self.step { return None; }
        self.runs(base, rows)
    }
}

#[cfg(test)]
mod append_ring_tests {
    use super::*;

    #[test]
    fn proposal_never_overwrites_visible_rows_including_wraps() {
        for step in [1, 7, 64, 513, 4096] {
            let ring = AppendRing::new(512, step, 64).unwrap();
            assert!(ring.rows >= 512 + step && ring.rows % 64 == 0);
            for base in [0, 1, 511, 512, 513, ring.rows - 1, ring.rows, 2 * ring.rows + 7] {
                let mut bytes = vec![0u8; ring.rows as usize];
                for run in ring.visible(base) {
                    bytes[run.slot as usize..(run.slot + run.rows) as usize].fill(19);
                }
                let saved = bytes.clone();
                for run in ring.proposal(base, step).unwrap() {
                    bytes[run.slot as usize..(run.slot + run.rows) as usize].fill(29);
                }
                for run in ring.visible(base) {
                    let range = run.slot as usize..(run.slot + run.rows) as usize;
                    assert_eq!(bytes[range.clone()], saved[range]);
                }
                assert!(ring.proposal(base, step + 1).is_none());
            }
        }
        assert!(AppendRing::new(0, 1, 64).is_none());
        assert!(AppendRing::new(512, 0, 64).is_none());
        assert!(AppendRing::new(512, 1, 0).is_none());
        assert!(AppendRing::new(u64::MAX, 1, 64).is_none());
        assert!(AppendRing::new(512, 64, 64).unwrap().proposal(u64::MAX, 1).is_none());
    }
}

/// Device mark slots: two per active lane plus two, retaining both snapshot banks
/// within the byte budget. Disabled retention takes no arena.
pub fn mark_slots_for(lanes: u64, entries: u64, slot_bytes: u64, budget_bytes: u64) -> u64 {
    if entries == 0 { return 0; }
    let floor = lanes.max(1).saturating_mul(2).saturating_add(2);
    let wanted = entries.saturating_mul(2).saturating_add(2);
    wanted.max(floor).min(budget_bytes / slot_bytes.max(1))
}

/// A fixed staging ring: two asynchronous captures, or a capture and a serialized restore.
/// Full rings wait for the oldest write; host-off retains the legacy lane/retention rule.
pub const HOST_MARK_STAGING_SLOTS: u64 = 2;

pub fn mark_slots_with_host(lanes: u64, entries: u64, slot_bytes: u64, budget_bytes: u64, host: bool) -> u64 {
    if entries == 0 { 0 } else if host { HOST_MARK_STAGING_SLOTS }
    else { mark_slots_for(lanes, entries, slot_bytes, budget_bytes) }
}

/// How much computation a retained frontier saves for a query that shares `common` tokens with
/// it. An exact ancestor (`common == frontier`) always saves `common`. A partial match resumes at
/// `common` rounded down to `align`, minus `replay` tokens replayed before it; `align == 0`
/// disables partial reuse (a family whose snapshot holds positional state, such as sliding-window
/// rings, cannot resume byte-identically anywhere but at the frontier it captured).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReuseRule {
    pub align: usize,
    pub replay: Option<usize>,
}
impl ReuseRule {
    /// V4.1: even-aligned, with a 128-token approximate encoder replay window.
    pub const V41: Self = Self { align: 2, replay: Some(128) };
    /// Exact frontiers only.
    pub const EXACT: Self = Self { align: 0, replay: None };
    /// A family whose whole state lives in pages of `page_rows` rows: resume at the page boundary.
    pub const fn paged(page_rows: usize) -> Self {
        Self { align: page_rows, replay: None }
    }
    /// Whether a partial match can be reused at all.
    pub const fn partial(self) -> bool {
        self.align > 0
    }
    /// Tokens a match of `common` against a frontier at `frontier` saves.
    pub fn skipped(self, common: usize, frontier: usize) -> usize {
        if common == frontier {
            common
        } else if self.align == 0 {
            0
        } else {
            (common / self.align * self.align).saturating_sub(self.replay.unwrap_or(0))
        }
    }
}

struct Node<T> {
    edge: Vec<u32>,
    value: Option<(u64, T)>,
    children: BTreeMap<u32, Node<T>>,
}
#[derive(Clone, Copy)]
struct Reusable {
    common: usize,
    frontier: usize,
    clock: u64,
}
impl Reusable {
    fn skipped(self, rule: ReuseRule) -> usize {
        rule.skipped(self.common, self.frontier)
    }
}
impl<T> Node<T> {
    fn empty(edge: Vec<u32>) -> Self {
        Self {
            edge,
            value: None,
            children: BTreeMap::new(),
        }
    }
    fn insert(&mut self, tokens: &[u32], value: T, clock: u64) -> bool {
        if tokens.is_empty() {
            return self.value.replace((clock, value)).is_none();
        }
        let child = self
            .children
            .entry(tokens[0])
            .or_insert_with(|| Self::empty(tokens.to_vec()));
        let common = child
            .edge
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if common < child.edge.len() {
            let mut old = std::mem::replace(child, Self::empty(tokens[..common].to_vec()));
            old.edge.drain(..common);
            child.children.insert(old.edge[0], old);
        }
        child.insert(&tokens[common..], value, clock)
    }
    fn lookup<'a>(
        &'a mut self,
        tokens: &[u32],
        position: usize,
        clock: u64,
    ) -> Option<(usize, &'a T)> {
        if let Some(first) = tokens.first() {
            if let Some(child) = self.children.get_mut(first) {
                if tokens.starts_with(&child.edge) {
                    let length = child.edge.len();
                    if let Some(found) = child.lookup(&tokens[length..], position + length, clock) {
                        return Some(found);
                    }
                }
            }
        }
        self.value.as_mut().map(|(used, value)| {
            *used = clock;
            (position, &*value)
        })
    }
    fn oldest(&self) -> Option<u64> {
        self.value
            .as_ref()
            .map(|v| v.0)
            .into_iter()
            .chain(self.children.values().filter_map(Self::oldest))
            .min()
    }
    fn any_frontier(&self, position: usize) -> Option<Reusable> {
        self.value
            .as_ref()
            .map(|&(clock, _)| Reusable {
                common: position,
                frontier: position,
                clock,
            })
            .or_else(|| {
                self.children
                    .values()
                    .find_map(|child| child.any_frontier(position + child.edge.len()))
            })
    }
    fn find_reusable(&self, tokens: &[u32], position: usize, rule: ReuseRule) -> Option<Reusable> {
        let mut best = self.value.as_ref().map(|&(clock, _)| Reusable {
            common: position,
            frontier: position,
            clock,
        });
        if let Some(child) = tokens.first().and_then(|first| self.children.get(first)) {
            let common = child
                .edge
                .iter()
                .zip(tokens)
                .take_while(|(a, b)| a == b)
                .count();
            let candidate = if common == child.edge.len() {
                child.find_reusable(&tokens[common..], position + common, rule)
            } else {
                child
                    .any_frontier(position + child.edge.len())
                    .map(|found| Reusable {
                        common: position + common,
                        ..found
                    })
            };
            if let Some(candidate) = candidate {
                if best.is_none_or(|old| candidate.skipped(rule) > old.skipped(rule)) {
                    best = Some(candidate);
                }
            }
        }
        if best.is_none() && position > 0 {
            best = self.any_frontier(position).map(|found| Reusable {
                common: position,
                ..found
            });
        }
        best.filter(|found| found.skipped(rule) > 0)
    }
    fn value_at_clock(&self, clock: u64) -> Option<&T> {
        self.value.as_ref().filter(|(used, _)| *used == clock).map(|(_, value)| value)
            .or_else(|| self.children.values().find_map(|child| child.value_at_clock(clock)))
    }
    fn refresh(&mut self, old: u64, new: u64) -> Option<&T> {
        if let Some((clock, value)) = self.value.as_mut() {
            if *clock == old {
                *clock = new;
                return Some(value);
            }
        }
        self.children
            .values_mut()
            .find_map(|child| child.refresh(old, new))
    }
    fn exact_clock(&self, tokens: &[u32]) -> Option<u64> {
        if tokens.is_empty() {
            return self.value.as_ref().map(|v| v.0);
        }
        let child = self.children.get(&tokens[0])?;
        tokens
            .strip_prefix(child.edge.as_slice())
            .and_then(|rest| child.exact_clock(rest))
    }
    fn collect<'n>(&'n self, values: &mut Vec<&'n T>) {
        values.extend(self.value.as_ref().map(|(_, value)| value));
        for child in self.children.values() {
            child.collect(values);
        }
    }
    fn oldest_where(&self, keep: &dyn Fn(&T) -> bool) -> Option<u64> {
        self.value
            .as_ref()
            .filter(|(_, value)| !keep(value))
            .map(|v| v.0)
            .into_iter()
            .chain(self.children.values().filter_map(|child| child.oldest_where(keep)))
            .min()
    }
    fn evict(&mut self, clock: u64) -> Option<T> {
        let mut evicted = match self.value.as_ref() {
            Some((c, _)) if *c == clock => self.value.take().map(|(_, value)| value),
            _ => None,
        };
        for child in self.children.values_mut() {
            if let Some(value) = child.evict(clock) {
                evicted = Some(value);
            }
        }
        self.children
            .retain(|_, child| child.value.is_some() || !child.children.is_empty());
        for child in self.children.values_mut() {
            while child.value.is_none() && child.children.len() == 1 {
                let (_, next) = child.children.pop_first().unwrap();
                child.edge.extend(next.edge);
                child.value = next.value;
                child.children = next.children;
            }
        }
        evicted
    }
}
pub struct Radix<T> {
    root: Node<T>,
    clock: u64,
    entries: usize,
    limit: usize,
    rule: ReuseRule,
}
impl<T> Radix<T> {
    /// A bank under V4.1's reuse rule.
    pub fn new(limit: usize) -> Self {
        Self::with_rule(limit, ReuseRule::V41)
    }
    pub fn with_rule(limit: usize, rule: ReuseRule) -> Self {
        Self {
            root: Node::empty(Vec::new()),
            clock: 0,
            entries: 0,
            limit,
            rule,
        }
    }
    pub fn rule(&self) -> ReuseRule {
        self.rule
    }
    /// Insert, evicting the least recently used entry if the bank is over its limit; returns
    /// that evicted entry so its owner can act before it is dropped.
    pub fn insert(&mut self, tokens: &[u32], value: T) -> Option<T> {
        if self.limit == 0 || tokens.is_empty() {
            return None;
        }
        self.clock = self
            .clock
            .checked_add(1)
            .expect("prefix access clock exhausted");
        self.entries += usize::from(self.root.insert(tokens, value, self.clock));
        let mut evicted = None;
        while self.entries > self.limit {
            evicted = self.evict_oldest();
        }
        evicted
    }
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<(usize, &T)> {
        self.clock = self
            .clock
            .checked_add(1)
            .expect("prefix access clock exhausted");
        self.root.lookup(tokens, 0, self.clock)
    }
    /// Prefer the most computation saved: a populated exact ancestor can beat
    /// a slightly longer partial match which needs a complete replay window.
    pub fn lookup_reusable(&mut self, tokens: &[u32]) -> Option<(usize, usize, &T)> {
        let found = self.root.find_reusable(tokens, 0, self.rule)?;
        self.clock = self
            .clock
            .checked_add(1)
            .expect("prefix access clock exhausted");
        let value = self
            .root
            .refresh(found.clock, self.clock)
            .expect("selected retained frontier");
        Some((found.common, found.frontier, value))
    }
    /// Read-only lookup: no clocks or LRU order change, even on a hit.
    pub fn peek_reusable(&self, tokens: &[u32]) -> Option<(usize, usize, &T)> {
        let found = self.root.find_reusable(tokens, 0, self.rule)?;
        Some((found.common, found.frontier, self.root.value_at_clock(found.clock)?))
    }
    /// Retained entries in this bank.
    pub fn entries(&self) -> usize {
        self.entries
    }
    /// Every retained entry, in no particular order. Read-only: no clock or LRU order changes.
    pub fn values(&self) -> Vec<&T> {
        let mut values = Vec::with_capacity(self.entries);
        self.root.collect(&mut values);
        values
    }
    /// Values in this bank's eviction order, without changing the access clocks.
    pub fn eviction_order(&self) -> Vec<&T> {
        fn collect<'a, T>(node: &'a Node<T>, out: &mut Vec<(u64, &'a T)>) {
            if let Some((clock, value)) = &node.value { out.push((*clock, value)); }
            for child in node.children.values() { collect(child, out); }
        }
        let mut values = Vec::with_capacity(self.entries);
        collect(&self.root, &mut values);
        values.sort_unstable_by_key(|(clock, _)| *clock);
        values.into_iter().map(|(_, value)| value).collect()
    }

    /// The bound this bank was created with; zero disables retention entirely.
    pub fn limit(&self) -> usize {
        self.limit
    }
    /// True when no entry and no edge remain: eviction has fully collapsed the tree.
    pub fn is_empty(&self) -> bool {
        self.root.value.is_none() && self.root.children.is_empty()
    }
    pub fn evict_one(&mut self) -> bool {
        self.evict_oldest().is_some()
    }
    /// Evict the least recently used entry and return it; `None` when empty.
    pub fn evict_oldest(&mut self) -> Option<T> {
        let oldest = self.root.oldest()?;
        let evicted = self.root.evict(oldest);
        self.entries -= 1;
        evicted
    }
    /// Evict the least recently used entry for which `keep` is false and return it.
    pub fn evict_oldest_where(&mut self, keep: &dyn Fn(&T) -> bool) -> Option<T> {
        let oldest = self.root.oldest_where(keep)?;
        let evicted = self.root.evict(oldest);
        self.entries -= 1;
        evicted
    }
    /// Remove and return the entry whose token sequence is exactly `tokens`.
    pub fn remove_exact(&mut self, tokens: &[u32]) -> Option<T> {
        let clock = self.root.exact_clock(tokens)?;
        let removed = self.root.evict(clock);
        self.entries -= 1;
        removed
    }
}
/// Prompt repeats and completed agentic turns have separate bounded banks.
/// A turn slot cannot be consumed by its own prompt snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotKind {
    Prompt,
    Turn,
}
pub struct Retention<T> {
    prompts: Radix<T>,
    turns: Radix<T>,
}
impl<T> Retention<T> {
    /// Both banks under V4.1's reuse rule.
    pub fn new(limit: usize) -> Self {
        Self::with_rule(limit, ReuseRule::V41)
    }
    pub fn with_rule(limit: usize, rule: ReuseRule) -> Self {
        Self { prompts: Radix::with_rule(limit, rule), turns: Radix::with_rule(limit, rule) }
    }
    pub fn rule(&self) -> ReuseRule {
        self.prompts.rule
    }
    pub fn bank(&self, kind: SnapshotKind) -> &Radix<T> {
        match kind {
            SnapshotKind::Prompt => &self.prompts,
            SnapshotKind::Turn => &self.turns,
        }
    }
    pub fn bank_mut(&mut self, kind: SnapshotKind) -> &mut Radix<T> {
        match kind { SnapshotKind::Prompt => &mut self.prompts, SnapshotKind::Turn => &mut self.turns }
    }
    pub fn lookup_reusable(&mut self, tokens: &[u32]) -> Option<(usize, usize, &T)> {
        let rule = self.prompts.rule;
        let prompt = self.prompts.root.find_reusable(tokens, 0, rule);
        let turn = self.turns.root.find_reusable(tokens, 0, self.turns.rule);
        // Prefer completed turns on a tie. Refresh only the chosen bank's LRU.
        let kind = match (prompt, turn) {
            (Some(p), Some(t)) if p.skipped(rule) > t.skipped(rule) => SnapshotKind::Prompt,
            (_, Some(_)) => SnapshotKind::Turn,
            (Some(_), None) => SnapshotKind::Prompt,
            (None, None) => return None,
        };
        self.bank_mut(kind).lookup_reusable(tokens)
    }
    /// The same selection as lookup, without refreshing either bank.
    pub fn peek_reusable(&self, tokens: &[u32]) -> Option<(usize, usize, &T)> {
        let prompt = self.prompts.peek_reusable(tokens);
        let turn = self.turns.peek_reusable(tokens);
        match (prompt, turn) {
            (Some(p), Some(t)) if self.rule().skipped(p.0, p.1) > self.rule().skipped(t.0, t.1) => Some(p),
            (_, Some(t)) => Some(t),
            (p, None) => p,
        }
    }
    /// `evict_one`, returning the evicted entry and its bank.
    pub fn evict_oldest(&mut self) -> Option<(SnapshotKind, T)> {
        self.prompts
            .evict_oldest()
            .map(|value| (SnapshotKind::Prompt, value))
            .or_else(|| self.turns.evict_oldest().map(|value| (SnapshotKind::Turn, value)))
    }
    /// Every retained entry of both banks, in no particular order. Read-only.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.prompts.values().into_iter().chain(self.turns.values())
    }
    /// `evict_oldest` restricted to entries for which `keep` is false, same bank order.
    pub fn evict_one_where(&mut self, keep: &dyn Fn(&T) -> bool) -> Option<(SnapshotKind, T)> {
        self.prompts
            .evict_oldest_where(keep)
            .map(|value| (SnapshotKind::Prompt, value))
            .or_else(|| self.turns.evict_oldest_where(keep).map(|value| (SnapshotKind::Turn, value)))
    }
    pub fn evict_one(&mut self) -> bool {
        // Under global-page pressure, reclaim prompt snapshots before completed
        // turns; their pages may remain shared until the turn also expires.
        self.evict_oldest().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    /// V4.1's rule is pinned: `Radix::new`/`Retention::new` keep {2, Some(128)}, which is the
    /// formula the engine shipped with (`(common / 2 * 2) - 128` for partial matches).
    #[test]
    fn v41_reuse_rule_is_pinned() {
        assert_eq!(ReuseRule::V41, ReuseRule { align: 2, replay: Some(128) });
        assert_eq!(Radix::<u8>::new(4).rule(), ReuseRule::V41);
        assert_eq!(Retention::<u8>::new(4).rule(), ReuseRule::V41);
        for frontier in [1usize, 2, 127, 128, 129, 130, 255, 256, 4096, 4243] {
            for common in 0..=frontier {
                let old = if common == frontier { common } else { (common / 2 * 2).saturating_sub(128) };
                assert_eq!(ReuseRule::V41.skipped(common, frontier), old, "{common}/{frontier}");
            }
        }
    }

    #[test]
    fn paged_and_exact_rules_resume_where_their_state_is_valid() {
        let tokens: Vec<u32> = (1..=300).collect();
        let mut partial = tokens[..200].to_vec();
        partial.push(9999);
        let mut paged = Radix::with_rule(4, ReuseRule::paged(64));
        paged.insert(&tokens, 300);
        // 200 common tokens resume at the last whole page, 192, with no replay.
        let (common, frontier, _) = paged.lookup_reusable(&partial).unwrap();
        assert_eq!(ReuseRule::paged(64).skipped(common, frontier), 192);
        assert!(paged.lookup_reusable(&tokens[..63]).is_none());
        let mut exact = Retention::with_rule(4, ReuseRule::EXACT);
        exact.bank_mut(SnapshotKind::Prompt).insert(&tokens, 300);
        assert!(exact.lookup_reusable(&partial).is_none());
        let mut longer = tokens.clone();
        longer.extend([7, 8, 9]);
        assert_eq!(exact.lookup_reusable(&longer), Some((300, 300, &300)));
        assert_eq!(exact.lookup_reusable(&tokens), Some((300, 300, &300)));
        // An exact ancestor still wins over a longer entry that only matches partially.
        exact.bank_mut(SnapshotKind::Turn).insert(&tokens[..100], 100);
        assert_eq!(exact.lookup_reusable(&partial), Some((100, 100, &100)));
        assert!(!ReuseRule::EXACT.partial() && ReuseRule::V41.partial());
    }
    #[test]
    fn twenty_four_completed_turns_do_not_compete_with_prompt_snapshots() {
        let mut retained = Retention::new(24);
        for i in 100..124 {
            retained.bank_mut(SnapshotKind::Prompt).insert(&[i, 1], (false, i));
            retained.bank_mut(SnapshotKind::Turn).insert(&[i, 1, 2], (true, i));
        }
        assert_eq!((retained.prompts.entries, retained.turns.entries), (24, 24));
        for i in 100..124 {
            assert_eq!(retained.lookup_reusable(&[i, 1]), Some((2, 2, &(false, i))));
            assert_eq!(retained.lookup_reusable(&[i, 1, 2, 3]), Some((3, 3, &(true, i))));
        }
        for i in 200..224 {
            retained.bank_mut(SnapshotKind::Prompt).insert(&[i, 1], (false, i));
        }
        for i in 100..124 {
            assert_eq!(retained.lookup_reusable(&[i, 1, 2, 3]), Some((3, 3, &(true, i))));
        }
        retained.bank_mut(SnapshotKind::Turn).insert(&[999, 1, 2], (true, 999));
        assert!(retained.lookup_reusable(&[100, 1, 2, 3]).is_none());
        assert_eq!(retained.turns.entries, 24);
        for _ in 0..24 { assert!(retained.evict_one()); }
        assert_eq!((retained.prompts.entries, retained.turns.entries), (0, 24));
        for _ in 0..24 { assert!(retained.evict_one()); }
        assert!(!retained.evict_one());
        let mut disabled = Retention::new(0);
        disabled.bank_mut(SnapshotKind::Turn).insert(&[1], (true, 1));
        assert!(disabled.lookup_reusable(&[1]).is_none());
    }

    #[test]
    fn values_visit_every_entry_of_both_banks_without_touching_lru_order() {
        let mut retained = Retention::new(8);
        retained.bank_mut(SnapshotKind::Prompt).insert(&[1, 2], 12);
        retained.bank_mut(SnapshotKind::Prompt).insert(&[1, 2, 3], 123);
        retained.bank_mut(SnapshotKind::Prompt).insert(&[4], 4);
        retained.bank_mut(SnapshotKind::Turn).insert(&[1, 2, 3, 5], 1235);
        let mut values: Vec<u32> = retained.values().copied().collect();
        values.sort_unstable();
        assert_eq!(values, [4, 12, 123, 1235]);
        // [1, 2] is still the oldest prompt entry: visiting refreshed nothing.
        assert_eq!(retained.evict_oldest(), Some((SnapshotKind::Prompt, 12)));
        assert_eq!(retained.values().count(), 3);
        assert_eq!(Retention::<u32>::new(4).values().count(), 0);
    }

    /// Contract pin: generated-turn retention is opportunistic, prompt-prefix
    /// reuse is the guarantee. A child that re-renders a completed turn from text
    /// may re-tokenize it differently; the completed turn then matches only
    /// partially and, because a partial match replays a 128-token encoder window,
    /// it saves less computation than the exact prompt ancestor. The reuse rule
    /// therefore selects the prompt snapshot and the reported frontier is the
    /// prompt length, not the completed turn. This is the v10 65536/262144 shape
    /// (`hit == context`) and the reason it is not a runtime publication fault.
    #[test]
    fn short_partial_completed_turn_defers_to_the_exact_prompt_snapshot() {
        let context = 4096usize;
        let prompt: Vec<u32> = (1..=context as u32).collect();
        let mut turn = prompt.clone();
        turn.extend(10_000..10_079); // 79-token completed turn (v10 65536 shape)
        let mut retained = Retention::new(8);
        retained.bank_mut(SnapshotKind::Prompt).insert(&prompt, "prompt");
        retained.bank_mut(SnapshotKind::Turn).insert(&turn, "turn");
        // The child re-sends the prompt, a turn that re-encodes with one token
        // different before the end, and then its own instruction.
        let mut divergent = turn.clone();
        divergent[context + 78] = 77_777;
        divergent.extend(20_000..20_044);
        assert_eq!(
            retained.lookup_reusable(&divergent),
            Some((context, context, &"prompt"))
        );
        // The same child whose turn round-trips exactly reuses the whole turn.
        let mut exact = turn.clone();
        exact.extend(20_000..20_044);
        assert_eq!(
            retained.lookup_reusable(&exact),
            Some((context + 79, context + 79, &"turn"))
        );
    }

    /// A partial completed turn longer than the 128-token replay window can still
    /// save more than the prompt ancestor; the reported frontier is then the
    /// aligned LCP minus the replay window. That is the v9 262144 shape
    /// (`hit = 262158 = 262144 + 14` for a 147-token turn matching 142 tokens).
    #[test]
    fn partial_completed_turn_that_saves_more_than_the_prompt_is_reused_from_its_window() {
        let context = 4096usize;
        let prompt: Vec<u32> = (1..=context as u32).collect();
        let mut turn = prompt.clone();
        turn.extend(10_000..10_147); // 147-token completed turn (v9 262144 shape)
        let mut retained = Retention::new(8);
        retained.bank_mut(SnapshotKind::Prompt).insert(&prompt, "prompt");
        retained.bank_mut(SnapshotKind::Turn).insert(&turn, "turn");
        let mut divergent = turn.clone();
        divergent[context + 142] = 77_777; // matches 142 of 147 generated tokens
        divergent.extend(20_000..20_044);
        let (common, frontier, value) = retained.lookup_reusable(&divergent).unwrap();
        assert_eq!((common, frontier, value), (context + 142, context + 147, &"turn"));
        // `v41_native_serve::prefix::restore` reports the replay start for a
        // partial match: `(common / 2 * 2) - 128`.
        assert_eq!((common / 2 * 2) - 128, context + 14);
    }

    #[test]
    fn partial_radix_match_accounts_for_alignment_replay_and_exact_ancestors() {
        let tokens: Vec<u32> = (1..=512).collect();
        let mut radix = Radix::new(16);
        radix.insert(&tokens, 512);
        let mut partial = tokens[..451].to_vec();
        partial.push(9999);
        assert_eq!(radix.lookup_reusable(&partial), Some((451, 512, &512)));
        radix.insert(&tokens[..384], 384);
        // Replaying at 450 only skips 322 tokens; the populated 384-token
        // ancestor skips more work and preserves its complete window state.
        assert_eq!(radix.lookup_reusable(&partial), Some((384, 384, &384)));
        assert_eq!(radix.lookup_reusable(&tokens), Some((512, 512, &512)));
        assert!(radix.lookup_reusable(&tokens[..128]).is_none());
        assert!(radix.lookup_reusable(&[9999]).is_none());
        assert_eq!(
            radix.lookup_reusable(&tokens[..258]),
            Some((258, 384, &384))
        );
        assert!(radix.evict_one());
        assert_eq!(radix.entries, 1);
        assert_eq!(
            radix.lookup_reusable(&tokens[..258]),
            Some((258, 384, &384))
        );
        assert!(radix.evict_one());
        assert_eq!(radix.entries, 0);
        assert!(radix.root.children.is_empty());
    }

    #[test]
    fn token_radix_splits_edges_and_returns_longest_retained_frontier() {
        let mut radix = Radix::new(16);
        radix.insert(&[10, 20, 30, 40], 4);
        radix.insert(&[10, 20], 2);
        radix.insert(&[10, 20, 50], 3);
        assert_eq!(radix.lookup(&[10, 20, 30, 40, 60]), Some((4, &4)));
        assert_eq!(radix.lookup(&[10, 20, 30]), Some((2, &2)));
        assert_eq!(radix.lookup(&[10, 20, 50]), Some((3, &3)));
        assert!(radix.lookup(&[10]).is_none());
        assert!(radix.lookup(&[99, 20]).is_none());
        radix.insert(&[10, 20], 99);
        assert_eq!(radix.entries, 3);
        assert_eq!(radix.lookup(&[10, 20]), Some((2, &99)));
    }
    #[test]
    fn evict_oldest_where_skips_kept_entries_in_bank_order() {
        let mut retained = Retention::new(16);
        retained.bank_mut(SnapshotKind::Prompt).insert(&[1, 2], "p-old");
        retained.bank_mut(SnapshotKind::Prompt).insert(&[1, 3], "p-new");
        retained.bank_mut(SnapshotKind::Turn).insert(&[9, 9], "t-old");
        let keep = |v: &&str| *v == "p-old";
        assert_eq!(retained.evict_one_where(&keep), Some((SnapshotKind::Prompt, "p-new")));
        assert_eq!(retained.lookup_reusable(&[1, 3]), None);
        assert_eq!(retained.lookup_reusable(&[1, 2]), Some((2, 2, &"p-old")));
        assert_eq!(retained.evict_one_where(&keep), Some((SnapshotKind::Turn, "t-old")));
        assert_eq!(retained.lookup_reusable(&[9, 9]), None);
        assert_eq!(retained.evict_one_where(&keep), None);
        assert_eq!(retained.bank(SnapshotKind::Prompt).entries(), 1);
        assert_eq!(retained.evict_oldest(), Some((SnapshotKind::Prompt, "p-old")));
        assert!(retained.bank(SnapshotKind::Prompt).is_empty());
    }
    #[test]
    fn radix_eviction_drops_saved_owners_and_preserves_recent_branch() {
        struct Owner(Rc<Cell<usize>>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let dropped = Rc::new(Cell::new(0));
        let mut radix = Radix::new(2);
        radix.insert(&[1, 2], Owner(dropped.clone()));
        radix.insert(&[1, 2, 3], Owner(dropped.clone()));
        radix.lookup(&[1, 2]);
        radix.insert(&[1, 4], Owner(dropped.clone()));
        assert_eq!(dropped.get(), 1);
        assert_eq!(radix.lookup(&[1, 2, 3]).unwrap().0, 2);
        assert!(radix.evict_one());
        assert!(radix.evict_one());
        assert!(!radix.evict_one());
        assert_eq!(dropped.get(), 3);
        assert!(radix.root.children.is_empty());
    }
}

#[cfg(test)]
mod staging_tests {
    use super::*;
    #[test]
    fn host_working_set_is_constant_and_host_off_is_legacy() {
        for lanes in [1, 2, 8, 16, 128] {
            for entries in [0, 1, 20, 128] {
                assert_eq!(mark_slots_with_host(lanes, entries, 428 << 20, 2 << 30, true),
                    if entries == 0 { 0 } else { 2 });
                assert_eq!(mark_slots_with_host(lanes, entries, 428 << 20, 2 << 30, false),
                    mark_slots_for(lanes, entries, 428 << 20, 2 << 30));
            }
        }
    }
    #[test]
    fn host_budget_charges_two_marks_per_conversation_and_capacity_uses_size_classes() {
        let g = HostCacheGeometry { unit_rows: 64, page_bytes: 512, mark_bytes: 140, draft_bytes: 0 };
        assert_eq!(host_cache_required_bytes(65, 2, 64, 512, 140, 0, 1024), Some(8 * 1024));
        assert_eq!(g.conversations(8 * 1024, 65, 1024), 6);
        assert_eq!(g.conversations(7 * 1024, 65, 1024), 5);
        assert_eq!(g.conversations(0, 65, 1024), 0);
        assert_eq!(g.conversations(1024, 0, 1024), 0);
    }
}
