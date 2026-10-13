//! The token-split (`context`) attention exchange between the two coordinator
//! GPUs (PLAN.md "v3 attention placement without latent replication", section
//! 1; K2).
//!
//! **Page owners.** Page `j` of every sequence lives on GPU `j mod 2` by its
//! logical index ([`page_owner`]). The prefix cache's parity pool hands it an
//! id of the same parity, and the id's local index on that GPU is `id >> 1`
//! ([`local_pages`]).
//!
//! **Exchanges.** Three per context layer, each with its own flag per
//! (layer parity, lane) slot ([`slot`]). Every one is an SM push into the
//! peer's receive buffer that publishes a device-side sequence, matched by a
//! spinning acquire wait on the receiving stream, as [`super::PeerExchange`]
//! does. So there is no host round trip and no event, and the exchanges
//! replay inside per-device decode graphs.
//! - **Query.** Each GPU produces its own heads' absorbed query `[R, H/2, D]`.
//!   [`ContextExchange::push_query`] writes it into this GPU's half of the
//!   assembled `[R, H, D]` buffer (GPU0's heads first) and into the same half
//!   of the peer's buffer, in one launch. [`ContextExchange::wait_query`]
//!   returns the assembled buffer the partial kernel reads.
//! - **Candidates** (indexer layers). Each GPU's scored top-k over its own
//!   pages (FP32 scores and global logical indices, `[R, K]` each) goes into
//!   its half of the assembled `[R, 2K]` score and index lists, GPU0's run
//!   first, which is what `dsa_candidate_merge` reads.
//! - **Partial.** Each GPU computes all heads over its local selection, then
//!   pushes the normalized BF16 partial and FP32 base-2 LSE of the peer's heads
//!   (`[R, H/2, 512]` + `[R, H/2]`). [`ContextExchange::wait_partial`] returns
//!   the peer's partial of this GPU's heads, and [`combine_order`] puts the
//!   operands of `lse_combine2` in the fixed order: GPU0's partial, then
//!   GPU1's.
//!
//! **Order.** A wait is queued only after this GPU queued every push the peer
//! needs before it can produce what the wait expects (`order::fixtures`
//! checks the context decode schedule). Buffers alternate by layer parity, so
//! a push never lands on rows the receiver may still read. The receiver
//! consumes slot `s` before its own next push on slot `s`, and the sender's
//! next push on `s` waits behind its own wait on that slot two layers later.
//!
//! **Storage.** One allocation per GPU, laid out by
//! [`ContextExchangeLayout`]: exactly the solver's "context exchange"
//! demand.
//!
//! **Lifetime.** The owner drains both streams before dropping this. Terminal
//! abort reuses the head split's words ([`super::PeerExchange::abort_word`]),
//! so the owning `PeerExchange` must outlive this exchange: drop this first.
use super::{on_device, RankDevice};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::peer_exchange::PeerPlane;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::placement::{ContextBuffers, ContextExchangeLayout, CONTEXT_EXCHANGES};
use std::ffi::c_void;

/// The GPU that owns logical page `page` of any sequence.
pub(crate) fn page_owner(page: usize) -> usize {
    page % 2
}

/// The (context ordinal parity, lane) slot on decode lane `lane`. Count only
/// context layers when they interleave with other attention kinds.
pub(crate) fn slot(context_index: usize, lane: usize) -> usize {
    2 * lane + context_index % 2
}

/// `rank`'s compact half table of a sequence: the local page index of each
/// logical page it owns (`j = rank, rank + 2, ...`), as the scored top-k
/// (`page_stride` 2, `page_offset` rank), the candidate merge and the staging
/// gather read it.
pub(crate) fn local_pages(pages: &[u32], rank: usize) -> Result<Vec<u32>> {
    pages.iter().enumerate().skip(rank).step_by(2).map(|(j, &id)| {
        ensure!(id as usize & 1 == page_owner(j), "logical page {j} has page id {id} of the wrong parity");
        Ok(id >> 1)
    }).collect()
}

/// Rows of a sequence's first `len` rows that live on `rank`'s pages.
pub(crate) fn shard_rows(len: usize, page_rows: usize, rank: usize) -> usize {
    let (full, tail) = (len / page_rows, len % page_rows);
    let full_mine = if rank == 0 { full.div_ceil(2) } else { full / 2 };
    full_mine * page_rows + if tail > 0 && page_owner(full) == rank { tail } else { 0 }
}

/// The operands of `lse_combine2` on `rank` for its heads: GPU0's partial
/// first, then GPU1's, whichever GPU runs the combine.
pub(crate) fn combine_order<T>(rank: usize, own: T, received: T) -> [T; 2] {
    if rank == 0 { [own, received] } else { [received, own] }
}

/// The `{family}_sparse_mla_partial_h{begin}_s{N}_m{rows}` variants a GPU with `sms` SMs runs,
/// as (row cap, program) ascending: the exporter's split plan per (SM count, row cap)
/// (`row_caps`; a variant without it serves the full `rows` of its `device_sm_counts`), else
/// the one variant. A step of `t` rows runs the smallest cap >= `t` ([`pick_partial`]): a few
/// verify rows plan more splits.
pub(crate) fn partial_programs(manifest: &serde_json::Value, family: &str, begin: usize, rows: usize, sms: usize)
    -> Result<Vec<(usize, String)>> {
    let programs = manifest["programs"].as_array().context("program manifest")?;
    let prefix = format!("{family}_sparse_mla_partial_h{begin}_s");
    let suffix = format!("_m{rows}");
    let variants: Vec<&serde_json::Value> = programs.iter()
        .filter(|p| p["name"].as_str().is_some_and(|n| n.starts_with(&prefix) && n.ends_with(&suffix))).collect();
    let mut out: Vec<(usize, String)> = Vec::new();
    for p in &variants {
        let name = p["name"].as_str().context("program name")?.to_string();
        let caps: Vec<usize> = match p["params"]["row_caps"].get(sms.to_string()) {
            Some(caps) => caps.as_array().context("row caps")?.iter().filter_map(|c| c.as_u64().map(|c| c as usize)).collect(),
            None if p["params"]["device_sm_counts"].as_array()
                .is_some_and(|sm| sm.iter().any(|s| s.as_u64() == Some(sms as u64))) => vec![rows],
            None => Vec::new(),
        };
        out.extend(caps.into_iter().map(|cap| (cap, name.clone())));
    }
    if out.is_empty() && variants.len() == 1 {
        out.push((rows, variants[0]["name"].as_str().context("program name")?.to_string()));
    }
    out.sort();
    out.dedup_by_key(|(cap, _)| *cap);
    ensure!(out.last().is_some_and(|(cap, _)| *cap >= rows), "no {prefix}*{suffix} variant planned for {sms} SMs at {rows} rows");
    Ok(out)
}

/// Select once at load, using numerical plan rows rather than live execution rows.
/// Every exported alternative has the same full execution capacity in its name.
pub(crate) fn partial_program(manifest: &serde_json::Value, family: &str, begin: usize,
    execution_capacity_rows: usize, sms: usize,
    plan: &cuteafd_core::coordinator_programs::AttentionReductionPlan) -> Result<String> {
    ensure!(plan.capacity_rows() <= execution_capacity_rows,
        "attention numerical domain exceeds execution capacity");
    let partials = partial_programs(manifest, family, begin, execution_capacity_rows, sms)?;
    partials.into_iter().find(|(cap, _)| *cap >= plan.rows()).map(|(_, name)| name)
        .context("attention plan has no partial reduction recipe")
}

/// The variant of `partials` (ascending caps, [`partial_programs`]) a step of `rows` rows runs.
pub(crate) fn pick_partial(partials: &[(usize, String)], rows: usize) -> &str {
    partials.iter().find(|(cap, _)| *cap >= rows).or(partials.last()).map_or("", |(_, name)| name.as_str())
}

/// One cache plane within an ownership unit. Its row coordinates can differ
/// from token coordinates (pooled index keys use one row per four tokens).
#[derive(Debug, Clone, Copy)]
pub(crate) struct CommitPlane {
    pub first: usize,
    pub rows: usize,
    pub rows_per_unit: usize,
    pub row_bytes: usize,
    pub base: usize,
    pub unit_bytes: usize,
}

/// Staging-to-pool copies, one `[source | destination | bytes]` list per plane.
/// All lists are zero-padded to `limit`, and counts may differ between planes.
pub(crate) struct CommitBlocks {
    pub table: Vec<u64>,
    pub counts: Vec<usize>,
}

/// Ownership and offset arithmetic shared by every gathered context prefill.
/// `pages` are parity pool IDs at logical unit positions, never physical IDs
/// guessed from a sequence. Callers upload once after draining table users.
pub(crate) fn commit_blocks(rank: usize, pages: &[u32], planes: &[CommitPlane], limit: usize)
    -> Result<CommitBlocks> {
    ensure!(rank < 2 && limit > 0, "commit needs rank 0/1 and positive capacity");
    let mut table = Vec::new();
    let mut counts = Vec::new();
    for plane in planes {
        ensure!(plane.rows_per_unit > 0 && plane.row_bytes > 0, "zero commit plane geometry");
        let end = plane.first.checked_add(plane.rows).context("commit rows overflow")?;
        let payload = plane.rows_per_unit.checked_mul(plane.row_bytes)
            .and_then(|n| plane.base.checked_add(n)).context("commit plane overflow")?;
        ensure!(payload <= plane.unit_bytes, "commit plane extends past its unit");
        ensure!(end.div_ceil(plane.rows_per_unit) <= pages.len(), "commit past the sequence's units");
        let mut lists: [Vec<u64>; 3] = Default::default();
        if plane.rows > 0 {
            for unit in (plane.first / plane.rows_per_unit..end.div_ceil(plane.rows_per_unit))
                .filter(|&unit| page_owner(unit) == rank) {
                let id = pages[unit];
                ensure!(id as usize % 2 == rank, "logical unit {unit} has id {id} of the wrong parity");
                let start = unit * plane.rows_per_unit;
                let first = plane.first.max(start) - start;
                let count = end.min(start.saturating_add(plane.rows_per_unit)) - start - first;
                let offset = plane.base.checked_add(first.checked_mul(plane.row_bytes).context("commit offset overflow")?)
                    .context("commit offset overflow")?;
                let address = |unit: usize| unit.checked_mul(plane.unit_bytes).and_then(|n| n.checked_add(offset))
                    .context("commit address overflow");
                lists[0].push(address(unit)? as u64);
                lists[1].push(address((id >> 1) as usize)? as u64);
                lists[2].push(count.checked_mul(plane.row_bytes).context("commit size overflow")? as u64);
            }
        }
        let count = lists[0].len();
        ensure!(count <= limit, "{count} commit units past {limit}");
        counts.push(count);
        table.extend(lists.into_iter().flat_map(|list| {
            let padding = limit - list.len();
            list.into_iter().chain(std::iter::repeat_n(0, padding))
        }));
    }
    Ok(CommitBlocks { table, counts })
}

/// Partial value width (the latent's 512 BF16 values per head).
pub(crate) const PARTIAL_DIM: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Exchange {
    Query = 0,
    Candidates = 1,
    Partial = 2,
}

/// A slot's receive buffers on one GPU.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotBuffers {
    /// Assembled query `[rows, 2, query_half]` (GPU0's heads first).
    pub query: *mut c_void,
    /// Assembled candidate scores `[rows, 2K]` f32, then indices `[rows, 2K]` i32.
    pub scores: *mut c_void,
    pub indices: *mut c_void,
    /// The peer's partial of this GPU's heads: BF16 `[rows, H/2, 512]`, then
    /// FP32 LSE `[rows, H/2]`.
    pub partial: *mut c_void,
    pub lse: *mut c_void,
}

pub(crate) struct ContextExchange<'a> {
    library: &'a NativeLibrary,
    ranks: [RankDevice; 2],
    layout: ContextExchangeLayout,
    /// One allocation per GPU: the slots' buffers, then the control words.
    storage: [DeviceAllocation<'a>; 2],
    /// Terminal abort words of the owning head split, per rank.
    abort: Option<[*const u32; 2]>,
}

impl<'a> ContextExchange<'a> {
    /// Allocates the exchange `buffers` describes on both GPUs (zeroed),
    /// enabling peer access both ways; `ranks[0]`'s device must be current,
    /// and is again on return. `abort`: the head split's terminal abort words
    /// ([`super::PeerExchange::abort_word`]), which waits then honor.
    pub fn new(library: &'a NativeLibrary, ranks: [RankDevice; 2], buffers: &ContextBuffers,
        abort: Option<[*const u32; 2]>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("context-exchange");
        ensure!(ranks[0].device != ranks[1].device, "a context split exchanges between two GPUs");
        let layout = ContextExchangeLayout::new(buffers);
        ensure!(layout.slots > 0 && layout.rows > 0, "context exchange needs decode rows and a lane");
        ensure!(layout.query_bytes % 16 == 0 && layout.candidate_bytes % 16 == 0 && layout.partial_bytes % 16 == 0,
            "context exchange rows must be 16-byte multiples");
        ensure!(layout.candidate_bytes % 8 == 0, "a candidate is an FP32 score and an i32 index");
        let heads = layout.partial_bytes / (PARTIAL_DIM as u64 * 2 + 4);
        ensure!(heads > 0 && heads * (PARTIAL_DIM as u64 * 2 + 4) == layout.partial_bytes && heads % 4 == 0,
            "partial rows of {} B are not [H, 512] BF16 + [H] FP32", layout.partial_bytes);
        let bytes = usize::try_from(layout.bytes().context("context exchange bytes overflow")?)?;
        let side = |rank: usize| -> Result<DeviceAllocation<'a>> {
            on_device(library, ranks[rank].device, ranks[0].device, || {
                library.cuda_enable_peer(ranks[1 - rank].device)?;
                library.peer_exchange_initialize()?;
                let allocation = DeviceAllocation::new(library, bytes)?;
                library.cuda_zero_bytes(allocation.buffer, bytes)?;
                Ok(allocation)
            })
        };
        Ok(Self { library, ranks, layout, storage: [side(0)?, side(1)?], abort })
    }

    /// Bytes allocated on each GPU (the solver charged exactly this).
    pub fn bytes(&self) -> usize {
        self.storage[0].buffer.bytes
    }

    pub fn layout(&self) -> ContextExchangeLayout {
        self.layout
    }

    /// Heads per GPU (`H/2`).
    pub fn heads(&self) -> usize {
        (self.layout.partial_bytes / (PARTIAL_DIM as u64 * 2 + 4)) as usize
    }

    /// Candidates per GPU (`K`).
    pub fn topk(&self) -> usize {
        (self.layout.candidate_bytes / 8) as usize
    }

    fn slot_stride(&self) -> usize {
        let [q, c, p] = self.layout.slot_bytes().expect("checked in new");
        (q + c + p) as usize
    }

    fn base(&self, rank: usize, slot: usize) -> Result<*mut u8> {
        ensure!(rank < 2 && (slot as u64) < self.layout.slots, "context slot {slot} of {} on rank {rank}", self.layout.slots);
        // SAFETY: the offset stays inside the allocation sized by the layout.
        Ok(unsafe { self.storage[rank].buffer.ptr.cast::<u8>().add(slot * self.slot_stride()) })
    }

    /// Rank `rank`'s receive buffers of `slot`.
    pub fn buffers(&self, rank: usize, slot: usize) -> Result<SlotBuffers> {
        let base = self.base(rank, slot)?;
        let rows = self.layout.rows as usize;
        let [q, c, _] = self.layout.slot_bytes().expect("checked in new").map(|b| b as usize);
        let half_lists = rows * 2 * self.topk() * 4;
        let partial_values = rows * self.heads() * PARTIAL_DIM * 2;
        debug_assert_eq!(c, 2 * half_lists);
        // SAFETY: every offset lies inside this slot's region (layout).
        unsafe {
            Ok(SlotBuffers {
                query: base.cast(),
                scores: base.add(q).cast(),
                indices: base.add(q + half_lists).cast(),
                partial: base.add(q + c).cast(),
                lse: base.add(q + c + partial_values).cast(),
            })
        }
    }

    /// Control words of (`slot`, `exchange`) on `rank`: `[0]` the sequence the
    /// peer publishes, `[1..3]` this GPU's send state, `[3]` its receive state.
    fn control(&self, rank: usize, slot: usize, exchange: Exchange) -> *mut u32 {
        let words = self.slot_stride() * self.layout.slots as usize;
        let index = slot * CONTEXT_EXCHANGES as usize + exchange as usize;
        // SAFETY: the control block follows the slots inside the allocation (layout).
        unsafe { self.storage[rank].buffer.ptr.cast::<u8>().add(words + index * 16).cast() }
    }

    fn check_rows(&self, rows: usize) -> Result<()> {
        ensure!(rows > 0 && rows as u64 <= self.layout.rows, "context exchange of {rows} rows (capacity {})", self.layout.rows);
        Ok(())
    }

    /// Queues `planes` on `from`'s stream, then the next sequence of (`slot`,
    /// `exchange`) into the peer's flag.
    fn push(&self, from: usize, slot: usize, exchange: Exchange, planes: &[PeerPlane]) -> Result<()> {
        let to = 1 - from;
        let link = (self.control(to, slot, exchange), self.control(from, slot, exchange).wrapping_add(1));
        // SAFETY: peer access is enabled both ways (new); every plane's source is
        // final on this stream and unchanged until later work on it; local and
        // peer destinations are this exchange's buffers, read by their GPU only
        // after the matching wait (local ones are stream-ordered after the push).
        on_device(self.library, self.ranks[from].device, self.ranks[0].device, || unsafe {
            self.library.peer_push_planes(planes, Some(link), 0, self.ranks[from].stream)
        })
    }

    /// Queues on `at`'s stream: wait for the peer's next push of (`slot`, `exchange`).
    fn wait(&self, at: usize, slot: usize, exchange: Exchange) -> Result<()> {
        let flag = self.control(at, slot, exchange);
        let recv_state = flag.wrapping_add(3);
        // SAFETY: this GPU's control words; the matching push is queued by the
        // peer before or after, in the order `order::fixtures` checks.
        on_device(self.library, self.ranks[at].device, self.ranks[0].device, || unsafe {
            match self.abort {
                Some(words) => self.library.peer_wait_abortable(flag, recv_state, words[at], self.ranks[at].stream),
                None => self.library.peer_wait(flag, recv_state, self.ranks[at].stream),
            }
        })
    }

    /// Plane of `rows` rows of `width` bytes from `source` (`pitch` apart) into
    /// column `column` of rows `row_pitch` apart at `destination`.
    fn plane(destination: *mut c_void, column: usize, row_pitch: usize, source: *const c_void, pitch: usize,
        rows: usize, width: usize) -> PeerPlane {
        PeerPlane {
            // SAFETY (pointer arithmetic only): inside the destination row.
            destination: unsafe { destination.cast::<u8>().add(column) }.cast(),
            source,
            rows: rows as u64,
            row_bytes: width as u64,
            destination_pitch: row_pitch as u64,
            source_pitch: pitch as u64,
        }
    }

    /// Queues on `rank`'s stream: this GPU's query half `[rows, H/2, D]`
    /// (`source`, rows `pitch` apart) into its half of both GPUs' assembled
    /// query, then the query flag. When `source` already is this GPU's half
    /// of its own assembled buffer (the producer wrote it there), only the
    /// peer copy runs.
    pub fn push_query(&self, rank: usize, slot: usize, source: *const c_void, pitch: usize, rows: usize) -> Result<()> {
        self.check_rows(rows)?;
        let half = self.layout.query_bytes as usize;
        let row = 2 * half;
        let column = rank * half;
        let own = self.buffers(rank, slot)?.query;
        let peer = self.buffers(1 - rank, slot)?.query;
        let in_place = std::ptr::eq(source, own.cast::<u8>().wrapping_add(column).cast()) && pitch == row;
        let mut planes = vec![Self::plane(peer, column, row, source, pitch, rows, half)];
        if !in_place {
            planes.push(Self::plane(own, column, row, source, pitch, rows, half));
        }
        self.push(rank, slot, Exchange::Query, &planes)
    }

    /// Queues the wait for the peer's query half on `rank`'s stream; returns
    /// the assembled `[rows, H, D]` query, valid on that stream after the wait.
    pub fn wait_query(&self, rank: usize, slot: usize) -> Result<*mut c_void> {
        self.wait(rank, slot, Exchange::Query)?;
        Ok(self.buffers(rank, slot)?.query)
    }

    /// Queues on `rank`'s stream: this GPU's scored candidates (`scores` f32
    /// and `indices` i32, `[rows, K]` each, rows contiguous) into its half of
    /// both GPUs' assembled `[rows, 2K]` lists (GPU0's run first), then the
    /// candidate flag.
    pub fn push_candidates(&self, rank: usize, slot: usize, scores: *const c_void, indices: *const c_void,
        rows: usize) -> Result<()> {
        self.check_rows(rows)?;
        let list = self.topk() * 4;
        let (own, peer) = (self.buffers(rank, slot)?, self.buffers(1 - rank, slot)?);
        let column = rank * list;
        self.push(rank, slot, Exchange::Candidates, &[
            Self::plane(peer.scores, column, 2 * list, scores, list, rows, list),
            Self::plane(peer.indices, column, 2 * list, indices, list, rows, list),
            Self::plane(own.scores, column, 2 * list, scores, list, rows, list),
            Self::plane(own.indices, column, 2 * list, indices, list, rows, list),
        ])
    }

    /// Queues the wait for the peer's candidates on `rank`'s stream; returns
    /// the assembled (scores, indices) lists the candidate merge reads.
    pub fn wait_candidates(&self, rank: usize, slot: usize) -> Result<(*mut c_void, *mut c_void)> {
        self.wait(rank, slot, Exchange::Candidates)?;
        let buffers = self.buffers(rank, slot)?;
        Ok((buffers.scores, buffers.indices))
    }

    /// Queues on `rank`'s stream: its partial of the peer's heads (`partial`
    /// BF16 `[rows, H/2, 512]`, `lse` FP32 `[rows, H/2]`, both contiguous) into
    /// the peer's receive buffer, then the partial flag.
    pub fn push_partial(&self, rank: usize, slot: usize, partial: *const c_void, lse: *const c_void, rows: usize)
        -> Result<()> {
        self.check_rows(rows)?;
        let peer = self.buffers(1 - rank, slot)?;
        let values = rows * self.heads() * PARTIAL_DIM * 2;
        let lse_bytes = rows * self.heads() * 4;
        self.push(rank, slot, Exchange::Partial, &[
            Self::plane(peer.partial, 0, values, partial, values, 1, values),
            Self::plane(peer.lse, 0, lse_bytes, lse, lse_bytes, 1, lse_bytes),
        ])
    }

    /// Queues the wait for the peer's partial of `rank`'s heads; returns
    /// (partial, lse), valid on that stream after the wait. Combine them with
    /// this GPU's own in [`combine_order`].
    pub fn wait_partial(&self, rank: usize, slot: usize) -> Result<(*mut c_void, *mut c_void)> {
        self.wait(rank, slot, Exchange::Partial)?;
        let buffers = self.buffers(rank, slot)?;
        Ok((buffers.partial, buffers.lse))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_pool_commit_planes_cover_each_owned_row_once() {
        // GLM Flash: 256 tokens per unit, 64 pooled index rows. Open tails
        // live in per-sequence state and are not committed as completed pools.
        let pages = [4u32, 9, 6, 11];
        let first = 253;
        let end = 773;
        let planes = [CommitPlane { first, rows: end - first, rows_per_unit: 256,
            row_bytes: 528, base: 0, unit_bytes: 256 * 528 },
            CommitPlane { first: first / 4, rows: end / 4 - first / 4, rows_per_unit: 64,
                row_bytes: 128, base: 0, unit_bytes: 64 * 132 },
            CommitPlane { first: first / 4, rows: end / 4 - first / 4, rows_per_unit: 64,
                row_bytes: 4, base: 64 * 128, unit_bytes: 64 * 132 }];
        let mut coverage = [vec![0; end], vec![0; end / 4], vec![0; end / 4]];
        for rank in 0..2 {
            let blocks = commit_blocks(rank, &pages, &planes, 3).unwrap();
            assert_eq!(blocks.table.len(), 27);
            for (kind, plane) in planes.iter().enumerate() {
                let lists = &blocks.table[kind * 9..][..9];
                for i in 0..blocks.counts[kind] {
                    let source = lists[i] as usize;
                    let unit = source / plane.unit_bytes;
                    let local_first = (source % plane.unit_bytes - plane.base) / plane.row_bytes;
                    let count = lists[6 + i] as usize / plane.row_bytes;
                    assert_eq!(page_owner(unit), rank);
                    assert_eq!(lists[3 + i] as usize, (pages[unit] >> 1) as usize * plane.unit_bytes
                        + plane.base + local_first * plane.row_bytes);
                    for row in unit * plane.rows_per_unit + local_first..unit * plane.rows_per_unit + local_first + count {
                        coverage[kind][row] += 1;
                    }
                }
            }
        }
        for (plane, coverage) in planes.iter().zip(coverage) {
            assert!(coverage[..plane.first].iter().all(|&n| n == 0));
            assert!(coverage[plane.first..].iter().all(|&n| n == 1));
        }
        assert!(commit_blocks(0, &[5], &planes[..1], 1).is_err());
        assert!(commit_blocks(0, &pages, &planes, 1).is_err());
        assert!(commit_blocks(2, &pages, &planes, 3).is_err());
    }

    #[test]
    fn partial_variants_follow_the_family_prefix_and_decode_rows() {
        // V4 Pro's second head range (h64) at 64 decode rows, beside a GLM and an m128 variant.
        let manifest = serde_json::json!({"programs": [
            {"name": "dsv4p_sparse_mla_partial_h64_s2_m64", "params": {"device_sm_counts": [170, 188],
                "row_caps": {"170": [64], "188": [64]}}},
            {"name": "dsv4p_sparse_mla_partial_h64_s16_m64", "params": {"device_sm_counts": [170, 188],
                "row_caps": {"170": [1, 8], "188": [1, 8]}}},
            {"name": "dsv4p_sparse_mla_partial_h64_s4_m128", "params": {"device_sm_counts": [188]}},
            {"name": "glm_sparse_mla_partial_h64_s4_m64", "params": {"device_sm_counts": [188]}},
            {"name": "dsv4p2_sparse_mla_partial_h64_s4_m64", "params": {"device_sm_counts": [188]}}]});
        let plans = partial_programs(&manifest, "dsv4p", 64, 64, 188).unwrap();
        assert_eq!(plans, [(1, "dsv4p_sparse_mla_partial_h64_s16_m64".to_string()),
            (8, "dsv4p_sparse_mla_partial_h64_s16_m64".to_string()), (64, "dsv4p_sparse_mla_partial_h64_s2_m64".to_string())]);
        assert_eq!([1, 8, 9, 64, 65].map(|rows| pick_partial(&plans, rows)), ["dsv4p_sparse_mla_partial_h64_s16_m64",
            "dsv4p_sparse_mla_partial_h64_s16_m64", "dsv4p_sparse_mla_partial_h64_s2_m64",
            "dsv4p_sparse_mla_partial_h64_s2_m64", "dsv4p_sparse_mla_partial_h64_s2_m64"]);
        // Another row count selects only its own variants.
        assert_eq!(partial_programs(&manifest, "dsv4p", 64, 128, 188).unwrap(),
            [(128, "dsv4p_sparse_mla_partial_h64_s4_m128".to_string())]);
        assert!(partial_programs(&manifest, "dsv4p", 64, 128, 170).is_ok(), "the one variant serves any SM count");
        assert!(partial_programs(&manifest, "dsv4p", 0, 64, 188).is_err(), "no head-0 variant");
        assert!(partial_programs(&manifest, "dsv4f", 64, 64, 188).is_err(), "family prefixes do not cross");
        assert_eq!(pick_partial(&[], 4), "");
        for (drafts, chosen) in [(None, "dsv4p_sparse_mla_partial_h64_s16_m64"),
            (Some(7), "dsv4p_sparse_mla_partial_h64_s16_m64"),
            (Some(8), "dsv4p_sparse_mla_partial_h64_s2_m64")] {
            let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(64, drafts, None).unwrap();
            assert_eq!(partial_program(&manifest, "dsv4p", 64, 64, 188, &plan).unwrap(), chosen);
        }
        let wide = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(128, None, None).unwrap();
        assert!(partial_program(&manifest, "dsv4p", 64, 64, 188, &wide).is_err());
    }

    #[test]
    fn commit_planes_allow_empty_frontiers_and_reject_bad_geometry() {
        let plane = CommitPlane { first: 4, rows: 0, rows_per_unit: 4, row_bytes: 2,
            base: 0, unit_bytes: 8 };
        let blocks = commit_blocks(1, &[4], &[plane], 2).unwrap();
        assert_eq!(blocks.counts, [0]);
        assert_eq!(blocks.table, [0; 6]);
        assert!(commit_blocks(0, &[5], &[CommitPlane { first: 0, rows: 1, ..plane }], 1).is_err());
        assert!(commit_blocks(0, &[4], &[CommitPlane { first: usize::MAX, rows: 1, ..plane }], 1).is_err());
        assert!(commit_blocks(0, &[4], &[CommitPlane { base: 1, ..plane }], 1).is_err());
        assert!(commit_blocks(0, &[4], &[CommitPlane { rows_per_unit: 0, ..plane }], 1).is_err());
        assert!(commit_blocks(0, &[4], &[plane], 0).is_err());
    }

    #[test]
    fn owners_follow_the_logical_page_and_shards_split_rows() {
        assert_eq!((0..6).map(page_owner).collect::<Vec<_>>(), [0, 1, 0, 1, 0, 1]);
        assert_eq!((slot(0, 0), slot(1, 0), slot(2, 0), slot(7, 1)), (0, 1, 0, 3));
        // Ids from a parity pool: logical page j has parity j.
        let pages = [4u32, 9, 0, 3, 6];
        assert_eq!(local_pages(&pages, 0).unwrap(), [2, 0, 3]);
        assert_eq!(local_pages(&pages, 1).unwrap(), [4, 1]);
        assert!(local_pages(&[4, 6], 1).is_err(), "page 1 of parity 0 is refused");
        // 64-row pages: 200 rows = pages 0, 1, 2 full and 8 rows on page 3 (GPU1).
        assert_eq!((shard_rows(200, 64, 0), shard_rows(200, 64, 1)), (128, 72));
        assert_eq!((shard_rows(130, 64, 0), shard_rows(130, 64, 1)), (66, 64));
        assert_eq!((shard_rows(0, 64, 0), shard_rows(0, 64, 1)), (0, 0));
        assert_eq!((shard_rows(5, 64, 0), shard_rows(5, 64, 1)), (5, 0));
        for len in 0..1000 {
            assert_eq!(shard_rows(len, 64, 0) + shard_rows(len, 64, 1), len);
        }
        assert_eq!(combine_order(0, "own", "peer"), ["own", "peer"]);
        assert_eq!(combine_order(1, "own", "peer"), ["peer", "own"]);
    }

    #[test]
    fn the_layout_is_the_solver_charge() {
        let buffers = ContextBuffers { query_row_bytes: 36_864, partial_row_bytes: 32_896, candidate_row_bytes: 16_384,
            decode_rows: 64, lanes: 1, ..Default::default() };
        let layout = ContextExchangeLayout::new(&buffers);
        assert_eq!(layout.bytes(), Some(2 * 64 * (2 * 36_864 + 32_896 + 2 * 16_384) + 2 * 3 * 16));
        let demands = buffers.demands().unwrap();
        assert_eq!(demands[1].bytes, layout.bytes().unwrap());
        assert_eq!(32_896 / (PARTIAL_DIM as u64 * 2 + 4), 32);
    }
}

#[cfg(test)]
#[path = "context_cuda_tests.rs"]
mod cuda_tests;
