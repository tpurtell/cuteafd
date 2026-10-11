//! What a family tells the prefix cache, and the device work it does for it.
use super::entry::Mark;
use super::marks::MarkSlot;
use super::pages::TailCopy;
use cuteafd_core::prefix::ReuseRule;
use cuteafd_hostcache::copy::DeviceRange;
use cuteafd_hostcache::pool::Layout;
use serde::Serialize;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Where a family keeps the positional marks of its snapshots.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum MarkStore {
    /// Fixed slots of a device arena the family preallocates ([`super::MarkArena`]).
    #[default]
    Arena,
    /// `pages` pages of the shared page pool per mark: taken at capture (evicting like any
    /// snapshot's rows), released with the snapshot, written and read by
    /// [`PrefixFamily::capture_pages`] and [`PrefixFamily::restore_pages`].
    ///
    /// A mark writes arbitrary bytes into its pages, whereas rows only ever hold records. The
    /// pool's first `reserved` pages are therefore never handed out, to marks or rows: a family
    /// whose kernels read page 0 as the stand-in for masked entries (GLM 5.3 Flash's decode
    /// sparse MLA reads record slot 0 for every masked candidate and weights it by zero, and
    /// 0 x NaN is NaN) keeps it as it was zeroed at start-up.
    Pool { pages: usize, reserved: usize },
}

impl MarkStore {
    /// Pool pages one mark takes (none in an arena).
    pub fn pages(self) -> usize {
        match self {
            Self::Arena => 0,
            Self::Pool { pages, .. } => pages,
        }
    }

    /// Leading pool pages no allocation hands out (none with arena marks).
    pub fn reserved(self) -> usize {
        match self {
            Self::Arena => 0,
            Self::Pool { reserved, .. } => reserved,
        }
    }
}

/// Which retained snapshot page pressure evicts first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum Eviction {
    /// Least recently used, prompts before turns at equal use ([`super::victim`]).
    #[default]
    LeastRecent,
    /// The least recently used snapshot whose eviction frees a page: one with a page (or mark
    /// page) no live placement and no other snapshot holds. A snapshot every page of which a
    /// running request or a longer snapshot also holds frees nothing and stays, so its prefix
    /// stays reusable (V4.1's `Gain::Pages` rule, generalized).
    FreesPages,
}

/// Whether a restore reproduces the state a prefill of the same tokens would have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RestoreFidelity {
    /// Every byte the model reads is the snapshot's (AGENTS "Exact prefix-cache restores").
    Exact,
    /// Shared pages hold exact rows through `source_end`, but positional state restarts empty
    /// at `replay_start` and the replayed rows rebuild it approximately (a partial match).
    ApproximateReplay,
}

/// Two frontiers of one restore: shared pages are exact through `source_end`; prefill resumes at
/// `replay_start` (<= `source_end`), rebuilding positional state from there. Equal for an exact
/// frontier. V4.1's partial match keeps compressed sources through the even-aligned common prefix
/// but rebuilds its windows from 128 rows earlier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LaggedState {
    pub source_end: usize,
    pub replay_start: usize,
}

/// A snapshot the cache selected for a prompt, before the family plans its restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreCandidate<'m> {
    /// Tokens the prompt and the snapshot share, raw (not yet rounded to a media boundary).
    pub common: usize,
    /// The snapshot's length.
    pub snapshot_end: usize,
    /// The prompt's length.
    pub target_end: usize,
    /// The resume the cache ranked this snapshot by: the reuse rule's, rounded down to a point
    /// outside every image of either prompt.
    pub resume: usize,
    /// Media spans of the prompt and of the snapshot.
    pub media: &'m [cuteafd_core::MediaSpan],
    pub saved_media: &'m [cuteafd_core::MediaSpan],
}

impl RestoreCandidate<'_> {
    /// `at`, rounded down to a point inside no image of either prompt.
    pub fn media_safe(&self, at: usize) -> usize {
        let mut at = at;
        loop {
            let rounded = crate::media::round_frontier(crate::media::round_frontier(at, self.media), self.saved_media);
            if rounded == at { return at; }
            at = rounded;
        }
    }
}

/// How a family restores a selected snapshot ([`PrefixFamily::plan_restore`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RestorePlan {
    pub snapshot_end: usize,
    pub target_end: usize,
    pub lag: LaggedState,
    pub fidelity: RestoreFidelity,
}

impl RestorePlan {
    /// The default plan under `rule`: an exact frontier restores its mark; a partial match
    /// resumes at the rule's replay start with no mark, its pages forked through the same point.
    pub fn under(rule: ReuseRule, hit: RestoreCandidate<'_>) -> Self {
        let _ = rule;
        let exact = hit.common == hit.snapshot_end;
        let resume = if exact { hit.snapshot_end } else { hit.resume };
        Self {
            snapshot_end: hit.snapshot_end,
            target_end: hit.target_end,
            lag: LaggedState { source_end: resume, replay_start: resume },
            fidelity: if exact { RestoreFidelity::Exact } else { RestoreFidelity::ApproximateReplay },
        }
    }
    /// Where prefill resumes.
    pub fn resume(&self) -> usize {
        self.lag.replay_start
    }
    pub fn exact(&self) -> bool {
        self.fidelity == RestoreFidelity::Exact
    }
    /// The invariants every plan keeps: replay never starts past the shared source frontier,
    /// nor the source frontier past the snapshot or the prompt, and an exact plan resumes at the
    /// snapshot's own frontier.
    pub fn check(&self, common: usize) -> Result<(), &'static str> {
        let LaggedState { source_end, replay_start } = self.lag;
        if replay_start > source_end || source_end > common.min(self.snapshot_end) || source_end > self.target_end {
            return Err("restore plan frontiers out of order");
        }
        if self.exact() && (source_end != self.snapshot_end || replay_start != source_end) {
            return Err("an exact restore resumes at its snapshot's frontier");
        }
        Ok(())
    }
}

/// What a restore may read besides the snapshot: the prompt's native token ids (never the radix'
/// media-keyed copy) and its media spans, so a partial restore can rebuild token-derived state
/// (V4.1's Engram lookback) at `replay_start`.
#[derive(Debug, Clone, Copy)]
pub struct RestoreContext<'t> {
    pub native_tokens: &'t [u32],
    pub media: &'t [cuteafd_core::MediaSpan],
}

/// A family's handle on a queued asynchronous capture ([`PrefixFamily::queue_capture`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaptureTicket(pub u32);

/// A family's snapshot geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FamilyLayout {
    /// Token rows per page (the sharing unit).
    pub page_rows: usize,
    /// Pages in the device pool.
    pub pages: usize,
    /// Device bytes of one page index, over every buffer that holds its rows (MiMo: one
    /// record block per full-attention layer). A family with several page geometries lists
    /// them as segments of one page index; there is one page class today.
    pub page_bytes: usize,
    /// Bytes of a positional mark (0: the pages are the whole state).
    pub mark_bytes: usize,
    /// Bytes of drafter state kept with a snapshot (0: drafters restore cold, masked by
    /// `context_valid_from`).
    pub draft_bytes: usize,
    /// How partial matches are reused. Exact frontiers restore the mark byte-exactly; a partial
    /// rule (`align > 0`) resumes at `common` rounded down to `align` minus `replay` rows, with
    /// no mark: the family starts its positional state empty there and the prefill of the
    /// replayed rows rebuilds it approximately (V4.1-style).
    pub rule: ReuseRule,
    /// Where marks live: an arena of `PrefixConfig::mark_slots` (the default), or pool pages.
    pub mark_store: MarkStore,
}

impl FamilyLayout {
    /// The host tier's slab sizes for this family.
    pub fn host_layout(&self) -> Layout {
        Layout::family(self.page_bytes, self.mark_bytes.max(1), self.draft_bytes)
    }
}

/// The device side of snapshots for one family. Every call only enqueues work on the family's
/// stream (in stream order with its forward passes); [`PrefixFamily::drain`] waits for it. The
/// cache drains before it publishes a snapshot to the host tier and before it releases pages or
/// mark slots, so no queued copy ever reads storage someone else was handed.
///
/// `M` is the family's snapshot metadata: small host data that describes the device bytes (ring
/// frontiers, token-derived history), captured with the mark and handed back on restore. It is
/// plain data, never a device handle, so a host-tier entry stays self-describing. `()` for
/// families whose pages and mark are the whole state.
pub trait PrefixFamily<M: Clone + Default = ()> {
    type Placement;

    fn layout(&self) -> FamilyLayout;
    /// The metadata of a snapshot of `placement` at `len`, taken with its mark.
    fn capture_meta(&self, placement: &Self::Placement, len: usize) -> Result<M, BoxError> {
        let _ = (placement, len);
        Ok(M::default())
    }
    /// How to restore a selected snapshot. The default follows the layout's [`ReuseRule`]:
    /// exact frontiers restore the mark, partial matches replay. A family may resume an exact
    /// ancestor differently (V4.1 continues its encoder past a long suffix) but never plans an
    /// approximate restore for an exact frontier.
    fn plan_restore(&self, hit: RestoreCandidate<'_>) -> RestorePlan {
        RestorePlan::under(self.layout().rule, hit)
    }
    /// Which snapshot page pressure evicts first.
    fn eviction(&self) -> Eviction {
        Eviction::LeastRecent
    }
    /// Install a freshly built placement's pages wherever the family's kernels read them (V4.1:
    /// every compressed source's page table). Runs once per placement, before any restore or
    /// prefill; `discard` undoes it when the cache abandons the placement.
    fn bind(&self, placement: &mut Self::Placement) -> Result<(), BoxError> {
        let _ = placement;
        Ok(())
    }
    /// The cache abandons a bound placement whose restore failed (its pages go back to the pool
    /// once the family drained): reset whatever `bind` and a partial restore installed.
    fn discard(&self, placement: &mut Self::Placement) -> Result<(), BoxError> {
        let _ = placement;
        Ok(())
    }
    /// Restore a selected snapshot into a bound placement whose pages were forked through
    /// `plan.lag.source_end`. `mark` and `saved` are the snapshot's for an exact plan and `None`
    /// for a partial one, which starts positional state empty at `plan.lag.replay_start` and may
    /// rebuild token-derived state from `context`. The default restores the mark at the resume
    /// point ([`PrefixFamily::restore`] / `restore_pages`).
    fn restore_with(&self, mark: Option<&Mark>, saved: Option<&M>, placement: &mut Self::Placement,
        plan: &RestorePlan, context: &RestoreContext<'_>) -> Result<(), BoxError> {
        let _ = (saved, context);
        match mark {
            Some(Mark::Pages(pages)) => self.restore_pages(pages, placement, plan.resume()),
            Some(Mark::Slot(slot)) => self.restore(Some(*slot), placement, plan.resume()),
            None => self.restore(None, placement, plan.resume()),
        }
    }
    /// Asynchronous capture: enqueue the snapshot's tail copy (`tail`, in place of
    /// [`PrefixFamily::copy_rows`]) and the mark copy of `placement` at `len` into `slot` on a
    /// stream the scheduler does not wait on, returning a ticket to poll. `Ok(None)` (the default)
    /// before enqueueing anything: the family captures synchronously instead.
    fn queue_capture(&self, slot: MarkSlot, placement: &Self::Placement, len: usize, tail: Option<TailCopy>)
        -> Result<Option<CaptureTicket>, BoxError> {
        let _ = (slot, placement, len, tail);
        Ok(None)
    }
    /// Whether every copy of a queued capture has landed (target and drafter alike).
    fn capture_ready(&self, ticket: CaptureTicket) -> Result<bool, BoxError> {
        let _ = ticket;
        Ok(true)
    }
    /// Drain a queued capture's copies and forget it; its storage returns to the cache only if
    /// this succeeds.
    fn abort_capture(&self, ticket: CaptureTicket) -> Result<(), BoxError> {
        let _ = ticket;
        Ok(())
    }
    /// The host tier wrote `pages` (and the snapshot's mark) of a restored snapshot: publish them
    /// wherever the family keeps copies of pages (V4.1's peer FP4 replicas). Runs before any
    /// placement can see the pages.
    fn host_restored(&self, pages: &[u32]) -> Result<(), BoxError> {
        let _ = pages;
        Ok(())
    }
    /// The placement's pages, in row order.
    fn pages<'p>(&self, placement: &'p Self::Placement) -> &'p [u32];
    /// Committed rows: the length a snapshot of this placement may capture (a speculative
    /// family reports the rows its state is consistent at, not rows still being verified).
    fn commit_point(&self, placement: &Self::Placement) -> usize;
    /// How far behind its commit point a snapshot of `placement` can still be captured exactly
    /// (MiMo: its 256-slot rings still hold the window of any position up to 124 rows back;
    /// recurrent state: 0).
    fn capture_reach(&self) -> usize {
        0
    }
    /// Copy the positional state of `placement` at `len` (at most `capture_reach` rows before
    /// its commit point) into `slot`.
    fn capture(&self, slot: MarkSlot, placement: &Self::Placement, len: usize) -> Result<(), BoxError>;
    /// Make `placement` continue at `len`: copy `mark` back into its own positional state and
    /// set its length. `None` is a partial hit (or a mark-less family): start the positional
    /// state empty at `len`; the caller prefills from `len`, replaying the rule's window.
    /// Touches tables and buffers only, never graph shapes or workspaces.
    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Self::Placement, len: usize) -> Result<(), BoxError>;
    /// Copy rows `[0, copy.rows)` of every paged buffer from page `copy.from` to page `copy.to`.
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError>;
    /// Wait for every copy enqueued so far.
    fn drain(&self) -> Result<(), BoxError>;
    /// Device ranges of one page (host tier), concatenated in this order on the host.
    fn page_segments(&self, page: u32) -> Vec<DeviceRange>;
    /// Device ranges of one mark slot (host tier).
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange>;
    /// A mark-less family's stand-in for the mark of its host snapshots (the host tier keeps a
    /// tail per snapshot): at most `max(mark_bytes, 1)` bytes of device scratch that host
    /// restores overwrite and nothing reads. Empty (the default): no host snapshots without a
    /// mark.
    fn host_tail(&self) -> Vec<DeviceRange> {
        Vec::new()
    }
    /// [`MarkStore::Pool`]: as [`PrefixFamily::capture`], into the mark's pool `pages` (whose
    /// rows the cache took for it and nothing else reads), laid out in the order
    /// [`PrefixFamily::mark_page_segments`] lists.
    fn capture_pages(&self, pages: &[u32], placement: &Self::Placement, len: usize) -> Result<(), BoxError> {
        let _ = (pages, placement, len);
        Err(POOL_MARKS_UNSUPPORTED.into())
    }
    /// [`MarkStore::Pool`]: as [`PrefixFamily::restore`] with a mark, from the mark in `pages`.
    fn restore_pages(&self, pages: &[u32], placement: &mut Self::Placement, len: usize) -> Result<(), BoxError> {
        let _ = (pages, placement, len);
        Err(POOL_MARKS_UNSUPPORTED.into())
    }
    /// [`MarkStore::Pool`]: device ranges of the mark held in `pages` (host tier), `mark_bytes`
    /// in all, concatenated in this order on the host.
    fn mark_page_segments(&self, pages: &[u32]) -> Result<Vec<DeviceRange>, BoxError> {
        let _ = pages;
        Err(POOL_MARKS_UNSUPPORTED.into())
    }
}

const POOL_MARKS_UNSUPPORTED: &str = "this family keeps its marks in a device arena, not in pool pages";
