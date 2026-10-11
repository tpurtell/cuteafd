//! What a family tells the prefix cache, and the device work it does for it.
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

/// Which GPU holds a page's rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum PageOwners {
    /// One GPU, or every GPU holds its own copy of every page (today's head split).
    #[default]
    Uniform,
    /// Token-split `context` attention: page id `i` lives on GPU `i & 1` at local index `i >> 1`
    /// (a [`super::RefPagePool::parity`] pool), and a sequence's page `j` has parity `j % 2`.
    /// Replicated per-page or per-mark bytes (V4's C128 records, its window marks) are stored
    /// once from GPU0 and copied to GPU1 after a host restore ([`PrefixFamily::page_replicas`]).
    Parity,
}

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
    /// Which GPU holds each page ([`PageOwners::Parity`]: two half pools; marks in an arena).
    pub page_owners: PageOwners,
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
pub trait PrefixFamily {
    type Placement;

    fn layout(&self) -> FamilyLayout;
    /// Record schema carried by host snapshots; formats cannot share restored bytes.
    fn record_format(&self) -> &'static str {
        "opaque_v1"
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
    /// Under [`PageOwners::Parity`] both pages sit at the same logical position: rows partitioned
    /// by page live on (and copy on the stream of) `copy.owner` only; replicated rows copy on
    /// every GPU that holds them.
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError>;
    /// Wait for every copy enqueued so far.
    fn drain(&self) -> Result<(), BoxError>;
    /// Device ranges of one page (host tier), concatenated in this order on the host. Each page's
    /// bytes are stored once: under [`PageOwners::Parity`], its partitioned rows on its owner
    /// GPU and GPU0's copy of any replicated rows. Two pages of the same parity list segments of
    /// the same lengths in the same order (a restore lands in a fresh page of that parity).
    fn page_segments(&self, page: u32) -> Vec<DeviceRange>;
    /// Device ranges of one mark slot (host tier): GPU0's copy of a mark every GPU holds.
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange>;
    /// The GPU that holds `page`'s partitioned rows: `None` under [`PageOwners::Uniform`], `Some(page
    /// & 1)` under [`PageOwners::Parity`] (the cache checks it before any host copy).
    fn page_device(&self, page: u32) -> Option<u8> {
        let _ = page;
        None
    }
    /// Replicated bytes of `page` that [`PrefixFamily::page_segments`] stores from GPU0 only:
    /// `(GPU0 range, GPU1 range)` pairs of equal length, copied GPU0 -> GPU1 by
    /// [`PrefixFamily::copy_replicas`] once a host restore landed. Empty: nothing replicated.
    fn page_replicas(&self, page: u32) -> Vec<(DeviceRange, DeviceRange)> {
        let _ = page;
        Vec::new()
    }
    /// As [`PrefixFamily::page_replicas`] for a mark slot.
    fn mark_replicas(&self, slot: MarkSlot) -> Vec<(DeviceRange, DeviceRange)> {
        let _ = slot;
        Vec::new()
    }
    /// Enqueue `copies` (source on GPU0, destination on GPU1) on the destination GPU's stream,
    /// ordered before its next forward pass; [`PrefixFamily::drain`] waits for them.
    fn copy_replicas(&self, copies: &[(DeviceRange, DeviceRange)]) -> Result<(), BoxError> {
        if copies.is_empty() { Ok(()) } else { Err("this family has no replicated snapshot bytes".into()) }
    }
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
