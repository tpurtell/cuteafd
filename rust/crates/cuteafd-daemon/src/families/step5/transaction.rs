//! Append-only target steps: mutable proposal rows never overlap the old frontier.
use super::{
    scratch::BufferOwner,
    cache::CacheRank,
    prefix::{Frontier, Placement, Step5Prefix},
};
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{PrefixCache, RefPagePool};
use cuteafd_ffi::{
    programs::{Program, Programs, Scalar},
    CuteafdDeviceBuffer,
};
use cuteafd_hostcache::copy::CopyEngine;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    rc::Rc,
};

fn view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(
        offset.checked_add(bytes).is_some_and(|n| n <= buffer.bytes),
        "Step-5 staged view extent"
    );
    Ok(CuteafdDeviceBuffer {
        ptr: buffer.ptr.wrapping_byte_add(offset),
        bytes,
        ..buffer
    })
}

/// One lane per rank, allocated before execution. CSA K/Z must survive until the
/// acceptance count is known; projection and attention scratch can be reused.
struct RegionScratch<'a> {
    starts: Allocation<'a>,
    counts: Allocation<'a>,
    unrounded: Allocation<'a>,
    compress: Program<'a>,
}

pub(super) struct Staging<'a> {
    pub stream: Rc<Stream<'a>>,
    raw: Vec<(Allocation<'a>, Allocation<'a>)>,
    sparse: Option<RegionScratch<'a>>,
    rows: usize,
    regions: usize,
    bytes: usize,
    busy: Cell<bool>,
    cache_owner: usize,
    positions: Allocation<'a>,
    requests: Allocation<'a>,
    ring_table: Allocation<'a>,
    page_table: Allocation<'a>,
    table_width: usize,
    sequences: usize,
}

pub(super) struct StepInputs<'s, 'a> {
    pub stream: &'s crate::shared::memory::device::Stream<'a>,
    pub positions: CuteafdDeviceBuffer,
    pub requests: CuteafdDeviceBuffer,
    pub rows: usize,
    pub base: usize,
    pub extent: usize,
    pub heads: usize,
}

/// Borrowed only inside the transaction callback, after this layer was staged.
/// The transaction retains page leases, staging metadata and all queued inputs.
pub(super) struct AttentionRead<'s, 'a> {
    pub inputs: StepInputs<'s, 'a>,
    pub start: usize,
    pub key: CuteafdDeviceBuffer,
    pub value: CuteafdDeviceBuffer,
    pub summaries: Option<CuteafdDeviceBuffer>,
    pub table: CuteafdDeviceBuffer,
    pub table_width: usize,
    pub pages: usize,
}

// Every sequence can prepend seven committed CSA rows. Keep each prefix
// disjoint until acceptance, while compressor scratch is reused in stream order.
fn retained_wave_rows(rows: usize, sequences: usize) -> Result<usize> {
    ensure!(rows > 0 && sequences > 0, "Step-5 staging requires rows and sequences");
    rows.checked_add(sequences.min(rows).checked_mul(7)
        .context("Step-5 retained pending rows overflow")?)
        .context("Step-5 retained wave rows overflow")
}

impl<'a> Staging<'a> {
    pub fn bytes(
        layout: &cuteafd_loader::serving_capacity::step5::Step5CacheLayout,
        rank: usize,
    ) -> Result<usize> {
        Self::bytes_for_sequences(layout, rank, 1)
    }

    pub(super) fn bytes_for_sequences(
        layout: &cuteafd_loader::serving_capacity::step5::Step5CacheLayout,
        rank: usize,
        sequences: usize,
    ) -> Result<usize> {
        cuteafd_loader::plan::families::step5::admission::staging_bytes(layout, rank, sequences)
    }

    pub fn new(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        cache: &CacheRank<'a>,
        available: usize,
    ) -> Result<Self> {
        Self::new_for_sequences(device, programs, cache, available, 1)
    }

    pub(super) fn new_for_sequences(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        cache: &CacheRank<'a>,
        available: usize,
        sequences: usize,
    ) -> Result<Self> {
        ensure!(sequences <= usize::try_from(cache.sequences())?,
            "Step-5 staging sequence capacity exceeds cache admission");
        let bytes = Self::bytes_for_sequences(cache.layout(), cache.rank(), sequences)?;
        ensure!(
            device.id == cache.stream.device.id
                && std::ptr::eq(device.library, cache.stream.device.library),
            "Step-5 staging device mismatch"
        );
        ensure!(
            bytes <= available,
            "Step-5 staging needs {bytes} bytes, admitted only {available}"
        );
        let rows = usize::try_from(cache.layout().ring.step)?;
        let retained = retained_wave_rows(rows, sequences)?;
        let regions = (rows + 7) / 8;
        let sparse = if cache.layout().ranks[cache.rank()].regions.is_empty() { None } else {
            let spec = programs.spec("step5_region_compress")?;
            ensure!(spec.scalar_kinds == b"il" && spec.scratch.is_empty()
                && spec.geometry["region_tokens"] == 8 && spec.geometry["proxy_dim"] == 256,
                "Step-5 region compression ABI");
            let compress = device.run(|| programs.program("step5_region_compress",
                &["keys", "logits", "starts", "counts", "output", "unrounded"]))?;
            Some(RegionScratch {
                starts: Allocation::new(device, (regions * 4).max(256))?,
                counts: Allocation::new(device, (regions * 4).max(256))?,
                unrounded: Allocation::new(device, (regions * 256 * 4).max(256))?,
                compress,
            })
        };
        let raw = cache.layout().ranks[cache.rank()]
            .regions
            .iter()
            .map(|_| {
                Ok((
                    Allocation::new(device, retained * 256 * 2)?,
                    Allocation::new(device, retained * 256 * 2)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let table_width = usize::try_from(cache.layout().context.div_ceil(64))?;
        let stream = Rc::new(Stream::new(device)?);
        cache.register_lane(&stream)?;
        let owner = Self {
            stream,
            raw,
            sparse,
            positions: Allocation::new(device, (rows * 8).max(256))?,
            requests: Allocation::new(device, (rows * 4).max(256))?,
            ring_table: Allocation::new(device, (table_width * 4).max(256))?,
            page_table: Allocation::new(device, (table_width * 4).max(256) * sequences)?,
            table_width,
            sequences,
            rows,
            regions,
            bytes,
            busy: Cell::new(false),
            cache_owner: cache.identity(),
        };
        let starts: Vec<u8> = (0..regions as i32)
            .flat_map(|r| (r * 8).to_le_bytes())
            .collect();
        let counts: Vec<u8> = (0..regions).flat_map(|_| 8i32.to_le_bytes()).collect();
        if let Some(sparse) = &owner.sparse {
            device.run(|| device.library.copy_h2d(sparse.starts.buffer, &starts))?;
            device.run(|| device.library.copy_h2d(sparse.counts.buffer, &counts))?;
        }
        device.run(|| {
            device
                .library
                .copy_h2d(owner.requests.buffer, &vec![0; rows * 4])
        })?;
        let physical_pages = cache.layout().ring.rows / 64;
        let table: Vec<u8> = (0..table_width)
            .flat_map(|page| ((page as u64 % physical_pages) as i32).to_le_bytes())
            .collect();
        device.run(|| device.library.copy_h2d(owner.ring_table.buffer, &table))?;
        Ok(owner)
    }
    pub(super) fn positions(&self) -> CuteafdDeviceBuffer { self.positions.buffer }

    pub fn allocation_bytes(&self) -> usize {
        self.bytes
    }

    pub(super) fn validate_cache(&self, cache: &CacheRank<'_>) -> Result<()> {
        ensure!(self.cache_owner == cache.identity()
            && self.stream.device.id == cache.stream.device.id
            && std::ptr::eq(self.stream.device.library, cache.stream.device.library),
            "Step-5 staging cache binding differs");
        Ok(())
    }

    pub fn owns_compressor(&self, cache: &CacheRank<'_>, layer: usize) -> bool {
        self.cache_owner == cache.identity() && self.sparse.is_some()
            && cache.layout().ranks[cache.rank()].regions.iter().position(|r| r.layer == layer)
                .and_then(|index| self.raw.get(index))
                .is_some_and(|(key, logits)| !key.buffer.ptr.is_null() && !logits.buffer.ptr.is_null())
    }
}

/// One contiguous query slice with its own native state slot and admitted pages.
#[derive(Debug, Clone, Copy)]
pub(super) struct SequenceRows<'p> {
    pub sequence: u64,
    pub start: usize,
    pub rows: usize,
    pub base: usize,
    pub pages: &'p [u32],
}

pub(super) fn validate_wave(sequences: &[SequenceRows<'_>], pool: &RefPagePool,
    slots: usize, capacity: usize, context: usize, pages: usize,
) -> Result<usize> {
    ensure!(!sequences.is_empty() && !pool.is_parity() && pool.page_rows() == 64
        && pool.capacity().checked_add(pool.reserved()) == Some(pages),
        "Step-5 wave pool geometry differs from native admission");
    let mut slots_seen = BTreeSet::new();
    let mut referenced = BTreeSet::new();
    let mut writable = BTreeSet::new();
    let mut total = 0usize;
    for sequence in sequences {
        ensure!(sequence.sequence < slots as u64 && slots_seen.insert(sequence.sequence),
            "Step-5 wave duplicate or unadmitted sequence slot");
        let end = sequence.base.checked_add(sequence.rows).context("Step-5 wave frontier overflow")?;
        ensure!(sequence.rows > 0 && sequence.start == total && end <= context,
            "Step-5 wave row slice or context extent");
        total = total.checked_add(sequence.rows).context("Step-5 wave rows overflow")?;
        ensure!(total <= capacity && end.div_ceil(64) <= sequence.pages.len()
            && sequence.pages.len() <= context.div_ceil(64),
            "Step-5 wave exceeds row or page-table admission");
        let mut unique = BTreeSet::new();
        for (logical, &page) in sequence.pages.iter().enumerate() {
            // RefPagePool::refs indexes directly, so bounds precede every read.
            ensure!((page as usize) >= pool.reserved() && (page as usize) < pages
                && unique.insert(page), "Step-5 wave invalid or duplicate physical page");
            let refs = pool.refs(page);
            ensure!(refs > 0, "Step-5 wave references an unallocated page");
            let write = logical >= sequence.base / 64;
            ensure!(!writable.contains(&page) && (!write || (!referenced.contains(&page) && refs == 1)),
                "Step-5 wave writable page is shared or aliases another sequence");
            referenced.insert(page);
            if write { writable.insert(page); }
        }
    }
    Ok(total)
}

#[derive(Debug, Clone)]
struct Sequence {
    sequence: u64,
    start: usize,
    rows: usize,
    base: usize,
    raw: usize,
    pages: Vec<u32>,
}

fn retain_wave(sequences: &[SequenceRows<'_>]) -> Result<Vec<Sequence>> {
    let mut raw = 0usize;
    sequences.iter().map(|sequence| {
        let kept = Sequence { sequence: sequence.sequence, start: sequence.start,
            rows: sequence.rows, base: sequence.base, raw, pages: sequence.pages.to_vec() };
        raw = raw.checked_add(sequence.base % 8).and_then(|n| n.checked_add(sequence.rows))
            .context("Step-5 retained sequence rows overflow")?;
        Ok(kept)
    }).collect()
}

fn expert_rows(sequences: &[Sequence], kind: cuteafd_transport::ExpertV2SourceKind)
    -> Result<Vec<cuteafd_transport::ExpertProtocolV2RowDescriptor>> {
    let mut rows = Vec::new();
    for sequence in sequences {
        ensure!(sequence.start == rows.len(), "Step-5 expert row slice differs from native Step");
        let end = sequence.base.checked_add(sequence.rows).context("Step-5 expert position overflow")?;
        for position in sequence.base..end {
            let row = u32::try_from(rows.len())?;
            rows.push(cuteafd_transport::ExpertProtocolV2RowDescriptor {
                row_id: u64::from(row), source_kind: kind,
                source_request_id: sequence.sequence, token_position: u64::try_from(position)?,
                route_offset: row.checked_mul(8).context("Step-5 expert route offset overflow")?, route_count: 8,
            });
        }
    }
    Ok(rows)
}

// Only these scoped operations borrow the shared allocator. A live Step owns
// page references, never a RefMut that could cross a callback or a device wait.
fn fork_at_begin<E: CopyEngine>(pool: &RefCell<PrefixCache<E, Frontier>>,
    family: &Step5Prefix<'_, '_>, pages: &[u32], base: usize, end: usize,
) -> Result<Vec<u32>> {
    Ok(pool.try_borrow_mut().context("Step-5 prefix pool already borrowed")?
        .fork_step(family, pages, base, end)?)
}

fn release_at_publish<E: CopyEngine>(pool: &RefCell<PrefixCache<E, Frontier>>,
    family: &Step5Prefix<'_, '_>, pages: &[u32],
) -> Result<()> {
    pool.try_borrow_mut().context("Step-5 prefix pool already borrowed")?.release(family, pages)?;
    Ok(())
}

fn abort<E: CopyEngine>(pool: &RefCell<PrefixCache<E, Frontier>>,
    family: &Step5Prefix<'_, '_>, pages: &[u32],
) -> Result<()> {
    release_at_publish(pool, family, pages)
}

/// Shared by the serving backend and prefix adapter. Taking before dropping
/// avoids holding the RefCell borrow while an abandoned Step drains its owners.
pub(super) struct Pending<T>(Rc<RefCell<Option<T>>>);

impl<T> Clone for Pending<T> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T> Default for Pending<T> {
    fn default() -> Self { Self(Rc::new(RefCell::new(None))) }
}

impl<T> Pending<T> {
    pub(super) fn retire(&self) { drop(self.take()); }

    pub(super) fn take(&self) -> Option<T> { self.0.borrow_mut().take() }

    /// Verification cannot touch a reused slot until stale evidence is drained.
    pub(super) fn verify<O>(&self, work: impl FnOnce() -> Result<(T, O)>) -> Result<O> {
        self.retire();
        let (evidence, output) = work()?;
        let mut pending = self.0.borrow_mut();
        ensure!(pending.is_none(), "Step-5 pending evidence already occupied");
        *pending = Some(evidence);
        Ok(output)
    }
}

/// Exclusively borrows native lane owners through preparation, publication or
/// Drop. Diagnostic steps also borrow a placement and shared page lease.
pub(super) struct Step<'e, 'a, E: CopyEngine> {
    ranks: Vec<&'e RefCell<CacheRank<'a>>>,
    staging: Vec<&'e Staging<'a>>,
    lease: Option<(&'e RefCell<PrefixCache<E, Frontier>>, &'e mut Placement)>,
    sequences: Vec<Sequence>,
    live: bool,
    active: bool,
    complete: Vec<BTreeSet<usize>>,
    projected: Vec<BTreeSet<usize>>,
    attended: Vec<BTreeSet<usize>>,
    rows: usize,
    base: usize,
    extent: usize,
    failed: bool,
    inputs: Vec<&'e dyn BufferOwner>,
    submission: Rc<Cell<bool>>,
}

impl<'e, 'a, E: CopyEngine> Step<'e, 'a, E> {
    pub(super) fn rows(&self) -> usize { self.rows }

    pub(super) fn expert_rows(&self, kind: cuteafd_transport::ExpertV2SourceKind)
        -> Result<Vec<cuteafd_transport::ExpertProtocolV2RowDescriptor>> {
        expert_rows(&self.sequences, kind)
    }

    pub fn validate_binding(&self, ranks: &[&RefCell<CacheRank<'a>>],
        stages: &[&Staging<'a>], rows: usize) -> Result<()> {
        validate_active_rows(self.rows, rows)?;
        ensure!(!self.failed && ranks.len() == self.ranks.len()
            && stages.len() == self.staging.len()
            && self.ranks.iter().zip(ranks).all(|(actual, expected)| std::ptr::eq(*actual, *expected))
            && self.staging.iter().zip(stages).all(|(actual, expected)| std::ptr::eq(*actual, *expected)),
            "Step-5 pipeline transaction owner/stream/row binding differs");
        Ok(())
    }

    /// A whole-pipeline guard must publish peer aborts before any local drain.
    pub fn in_submission<T>(&mut self,
        _scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>,
        work: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let _mode = SubmissionMode::enter(self.submission.clone())?;
        let result = work(self);
        if result.is_err() { self.failed = true; }
        result
    }

    pub(super) fn in_submissions<T>(
        _scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>,
        steps: &mut [&mut Self],
        work: impl FnOnce(&mut [&mut Self]) -> Result<T>,
    ) -> Result<T> {
        let _modes = steps.iter().map(|step| SubmissionMode::enter(step.submission.clone()))
            .collect::<Result<Vec<_>>>()?;
        let result = work(steps);
        if result.is_err() {
            for step in steps { step.failed = true; }
        }
        result
    }

    pub fn begin(
        ranks: Vec<&'e RefCell<CacheRank<'a>>>,
        staging: Vec<&'e Staging<'a>>,
        pool: &'e RefCell<PrefixCache<E, Frontier>>,
        placement: &'e mut Placement,
        rows: usize,
    ) -> Result<Self> {
        let family = Step5Prefix::new(ranks.clone())?;
        ensure!(
            ranks.len() == staging.len() && rows > 0,
            "Step-5 step rank/row count"
        );
        let base = placement.committed;
        let end = base.checked_add(rows).context("Step-5 step end overflow")?;
        for (cache, stage) in ranks.iter().zip(&staging) {
            let cache = cache.borrow();
            ensure!(
                !cache.poisoned.get()
                    && rows <= stage.rows
                    && rows as u64 <= cache.layout().ring.step
                    && end as u64 <= cache.layout().context
                    && placement.sequence < cache.sequences(),
                "Step-5 step outside admitted capacity"
            );
            ensure!(
                !cache.busy[placement.sequence as usize].get() && !stage.busy.get(),
                "Step-5 sequence or staging lane already in flight"
            );
            ensure!(
                stage.cache_owner == cache.identity(),
                "Step-5 staging belongs to another cache owner"
            );
        }
        let pages = fork_at_begin(pool, &family, &placement.pages, base, end)?;
        let admitted = (|| {
            let cache = ranks[0].borrow();
            let shared = pool.try_borrow().context("Step-5 prefix pool already borrowed")?;
            validate_wave(&[SequenceRows { sequence: placement.sequence, start: 0, rows, base, pages: &pages }],
                shared.pool(), usize::try_from(cache.sequences())?, staging[0].rows,
                usize::try_from(cache.layout().context)?, usize::try_from(cache.page_count())?)
        })();
        if let Err(error) = admitted {
            abort(pool, &family, &pages)?;
            return Err(error);
        }
        let sequences = retain_wave(&[SequenceRows { sequence: placement.sequence,
            start: 0, rows, base, pages: &pages }])?;
        Self::prepare(ranks, staging, sequences, Some((pool, placement)))
    }

    /// The scheduler owns these page references. Retain only descriptors, never
    /// its placements or allocator borrow, across verification and acceptance.
    pub(super) fn begin_wave(
        ranks: Vec<&'e RefCell<CacheRank<'a>>>, staging: Vec<&'e Staging<'a>>,
        sequences: &[SequenceRows<'_>], pool: &RefPagePool,
    ) -> Result<Self> {
        Step5Prefix::new(ranks.clone())?;
        ensure!(ranks.len() == staging.len(), "Step-5 wave rank inventory");
        let cache = ranks[0].borrow();
        validate_wave(sequences, pool, usize::try_from(cache.sequences())?, staging[0].rows,
            usize::try_from(cache.layout().context)?, usize::try_from(cache.page_count())?)?;
        drop(cache);
        Self::prepare(ranks, staging, retain_wave(sequences)?, None)
    }

    fn prepare(ranks: Vec<&'e RefCell<CacheRank<'a>>>, staging: Vec<&'e Staging<'a>>,
        sequences: Vec<Sequence>, lease: Option<(&'e RefCell<PrefixCache<E, Frontier>>, &'e mut Placement)>,
    ) -> Result<Self> {
        let rows = sequences.iter().map(|sequence| sequence.rows).sum();
        let extent = sequences.iter().map(|sequence| sequence.base + sequence.rows).max()
            .context("Step-5 empty wave")?;
        let complete = vec![BTreeSet::new(); ranks.len()];
        let mut step = Self { ranks, staging, lease, base: sequences[0].base, extent,
            sequences, live: true, active: false, projected: complete.clone(), attended: complete.clone(),
            complete, rows, failed: false, inputs: Vec::new(),
            submission: Rc::new(Cell::new(false)) };
        for (owner, stage) in step.ranks.iter().zip(&step.staging) {
            let cache = owner.borrow();
            stage.validate_cache(&cache)?;
            ensure!(!cache.poisoned.get() && !stage.busy.get() && rows <= stage.rows
                && step.sequences.len() <= stage.sequences && extent as u64 <= cache.layout().context,
                "Step-5 wave outside admitted lane capacity");
            for sequence in &step.sequences {
                ensure!(sequence.sequence < cache.sequences()
                    && !cache.busy[sequence.sequence as usize].get(),
                    "Step-5 wave sequence unavailable");
            }
        }
        for (owner, stage) in step.ranks.iter().zip(&step.staging) {
            for sequence in &step.sequences { owner.borrow().busy[sequence.sequence as usize].set(true); }
            stage.busy.set(true);
        }
        step.active = true;
        let positions: Vec<u8> = step.sequences.iter().flat_map(|sequence|
            (sequence.base..sequence.base + sequence.rows).flat_map(|p| (p as i64).to_le_bytes())).collect();
        for stage in &step.staging {
            stage.stream.device.run(|| {
                // SAFETY: owned metadata and lane allocations survive the Drop drain.
                unsafe { stage.stream.device.library.copy_h2d_ordered(
                    stage.positions.buffer, &positions, stage.stream.raw) }
            })?;
            for (index, sequence) in step.sequences.iter().enumerate() {
                let table: Vec<u8> = sequence.pages.iter().flat_map(|p| (*p as i32).to_le_bytes()).collect();
                let bytes = (stage.table_width * 4).max(256);
                let target = view(stage.page_table.buffer, index * bytes, bytes)?;
                stage.stream.device.run(|| {
                    // SAFETY: sequence-local tables are disjoint; ordered uploads retain sources.
                    unsafe { stage.stream.device.library.copy_h2d_ordered(target, &table, stage.stream.raw) }
                })?;
            }
        }
        Ok(step)
    }

    pub fn pages(&self) -> &[u32] {
        &self.sequences[0].pages
    }

    /// Every producer/consumer uses its private lane stream. Failure poisons this
    /// proposal, and Drop drains before any borrowed layer owner can retire.
    pub fn project<O, T>(
        &mut self,
        rank: usize,
        layer: usize,
        input_owner: &'e O,
        work: impl FnOnce(&'e O, StepInputs<'_, 'a>) -> Result<T>,
    ) -> Result<T> {
        ensure!(!self.failed, "Step-5 failed step cannot project");
        let owner = self.ranks.get(rank).context("Step-5 projection rank")?;
        let cache = owner.borrow();
        let stage = self.staging[rank];
        ensure!(
            cache.layout().ranks[rank]
                .sliding
                .iter()
                .any(|r| r.layer == layer && !r.mtp)
                || cache.layout().ranks[rank]
                    .regions
                    .iter()
                    .any(|r| r.layer == layer),
            "Step-5 projection layer"
        );
        ensure!(
            self.projected[rank].insert(layer),
            "Step-5 duplicate projection"
        );
        self.failed = true;
        let mut guard = stage
            .stream
            .cancellation_guard("Step-5 projection callback");
        guard.complete = self.submission.get();
        let result = work(
            input_owner,
            StepInputs {
                stream: &stage.stream,
                positions: stage.positions.buffer,
                requests: stage.requests.buffer,
                rows: self.rows,
                base: self.base,
                extent: self.extent,
                heads: cache.layout().query_heads(rank, layer)?,
            },
        );
        if result.is_ok() {
            self.failed = false;
            guard.complete = true;
        }
        result
    }

    /// Pin non-attention owners and order chain copies/FFNs on the lane stream.
    pub fn compose<O, T>(
        &mut self,
        rank: usize,
        owner: &'e O,
        work: impl FnOnce(&'e O, StepInputs<'_, 'a>) -> Result<T>,
    ) -> Result<T> {
        ensure!(!self.failed, "Step-5 failed step cannot compose");
        let cache = self.ranks.get(rank).context("Step-5 composition rank")?.borrow();
        self.failed = true;
        let stage = self.staging[rank];
        let mut guard = stage.stream.cancellation_guard("Step-5 composition callback");
        guard.complete = self.submission.get();
        let result = work(owner, StepInputs {
            stream: &stage.stream,
            positions: stage.positions.buffer,
            requests: stage.requests.buffer,
            rows: self.rows,
            base: self.base,
            extent: self.extent,
            heads: 64 / self.ranks.len(),
        });
        if result.is_ok() {
            self.failed = false;
            guard.complete = true;
        }
        result
    }

    pub fn attend<O>(
        &mut self,
        rank: usize,
        layer: usize,
        input_owner: &'e O,
        mut work: impl FnMut(&'e O, AttentionRead<'_, 'a>) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.failed
                && self
                    .complete
                    .get(rank)
                    .is_some_and(|layers| layers.contains(&layer)),
            "Step-5 attention requires a successfully staged layer"
        );
        let cache = self.ranks[rank].borrow();
        let state = &cache.layout().ranks[rank];
        let stage = self.staging[rank];
        self.failed = true;
        let mut guard = stage.stream.cancellation_guard("Step-5 attention callback");
        guard.complete = self.submission.get();
        for (index, sequence) in self.sequences.iter().enumerate() {
            let (key, value, summaries, table, pages) =
                if let Some(ring) = state.sliding.iter().find(|r| r.layer == layer && !r.mtp) {
                    (cache.request_component(ring.key, sequence.sequence)?,
                        cache.request_component(ring.value, sequence.sequence)?, None,
                        stage.ring_table.buffer, usize::try_from(cache.layout().ring.rows / 64)?)
                } else {
                    let region = state.regions.iter().find(|r| r.layer == layer)
                        .context("Step-5 attention layer")?;
                    let (key, value) = cache.csa_planes(layer)?;
                    let summary = region.summaries.map(|s| cache.request_component(s, sequence.sequence)).transpose()?;
                    let bytes = (stage.table_width * 4).max(256);
                    (key, value, summary, view(stage.page_table.buffer, index * bytes, bytes)?,
                        usize::try_from(cache.page_count())?)
                };
            work(input_owner, AttentionRead { inputs: StepInputs {
                stream: &stage.stream, positions: stage.positions.buffer, requests: stage.requests.buffer,
                rows: sequence.rows, base: sequence.base, extent: sequence.base + sequence.rows,
                heads: cache.layout().query_heads(rank, layer)?,
            }, start: sequence.start, key, value, summaries, table, table_width: stage.table_width, pages })?;
        }
        self.attended[rank].insert(layer);
        self.failed = false;
        guard.complete = true;
        Ok(())
    }

    /// BF16 projections are ordered on the lane stream. Borrowed allocations
    /// remain live through Drop's drain, including partial-enqueue cancellation.
    pub fn stage(
        &mut self,
        rank: usize,
        layer: usize,
        key: &'e dyn BufferOwner,
        value: &'e dyn BufferOwner,
        index: Option<(&'e dyn BufferOwner, &'e dyn BufferOwner)>,
    ) -> Result<()> {
        ensure!(
            rank < self.ranks.len() && !self.complete[rank].contains(&layer),
            "Step-5 duplicate or invalid staged layer"
        );
        ensure!(!self.failed, "Step-5 failed step cannot be reused");
        self.failed = true;
        self.inputs.extend([key, value]);
        if let Some((key, z)) = index {
            self.inputs.extend([key, z]);
        }
        let (key, value) = (key.buffer(), value.buffer());
        let index = index.map(|(key, z)| (key.buffer(), z.buffer()));
        let cache = self.ranks[rank].borrow();
        let state = &cache.layout().ranks[rank];
        let stage = self.staging[rank];
        let mut guard = stage.stream.cancellation_guard("Step-5 staged layer");
        guard.complete = self.submission.get();
        for sequence in &self.sequences {
            let kv_row = cache.layout().ranks[rank].sliding.iter().find(|r| r.layer == layer && !r.mtp)
                .map(|ring| ring.key.bytes as usize / cache.layout().ring.rows as usize)
                .or_else(|| state.regions.iter().find(|r| r.layer == layer).map(|region| region.page_bytes as usize / 64))
                .context("Step-5 staged layer geometry")?;
            let key = view(key, sequence.start * kv_row, sequence.rows * kv_row)?;
            let value = view(value, sequence.start * kv_row, sequence.rows * kv_row)?;
            let index = index.map(|(key, z)| -> Result<_> { Ok((view(key, sequence.start * 512, sequence.rows * 512)?,
                view(z, sequence.start * 512, sequence.rows * 512)?)) }).transpose()?;
            if let Some(ring) = state.sliding.iter().find(|r| r.layer == layer && !r.mtp) {
                ensure!(index.is_none(), "SWA must not carry indexer state");
                let row = ring.key.bytes as usize / cache.layout().ring.rows as usize;
                let targets = [
                    cache.request_component(ring.key, sequence.sequence)?,
                    cache.request_component(ring.value, sequence.sequence)?,
                ];
                let sources = [key, value];
                for source in sources {
                    self.validate_source(&cache, source, sequence.rows * row)?;
                }
                for run in cache
                    .layout()
                    .ring
                    .proposal(sequence.base as u64, sequence.rows as u64)
                    .context("Step-5 ring proposal")?
                {
                    for (target, source) in targets.into_iter().zip(sources) {
                        copy(
                            &stage.stream,
                            view(target, run.slot as usize * row, run.rows as usize * row)?,
                            view(source, run.offset as usize * row, run.rows as usize * row)?,
                        )?;
                    }
                }
            } else {
                let at = state
                    .regions
                    .iter()
                    .position(|r| r.layer == layer)
                    .context("Step-5 target layer")?;
                let region = &state.regions[at];
                let stage = self.staging[rank];
                let row = usize::try_from(region.page_bytes / 64)?;
                for source in [key, value] {
                    self.validate_source(&cache, source, sequence.rows * row)?;
                }
                let (ik, iz) = index.context("CSA requires indexer K/Z")?;
                for source in [ik, iz] {
                    self.validate_source(&cache, source, sequence.rows * 512)?;
                }
                let pending = sequence.base % 8;
                let old = cache.pending(layer, sequence.sequence, false)?;
                let raw = [view(stage.raw[at].0.buffer, sequence.raw * 512, (pending + sequence.rows) * 512)?,
                    view(stage.raw[at].1.buffer, sequence.raw * 512, (pending + sequence.rows) * 512)?];
                for ((target, previous), source) in raw.into_iter().zip([old.0, old.1]).zip([ik, iz]) {
                    if pending > 0 {
                        copy(
                            &stage.stream,
                            view(target, 0, pending * 512)?,
                            view(previous, 0, pending * 512)?,
                        )?;
                    }
                    copy(
                        &stage.stream,
                        view(target, pending * 512, sequence.rows * 512)?,
                        view(source, 0, sequence.rows * 512)?,
                    )?;
                }
                let targets = cache.csa_planes(layer)?;
                let mut position = sequence.base;
                while position < sequence.base + sequence.rows {
                    let page = sequence.pages[position / 64] as usize;
                    let count = (64 - position % 64).min(sequence.base + sequence.rows - position);
                    for (target, source) in [targets.0, targets.1].into_iter().zip([key, value]) {
                        copy(
                            &stage.stream,
                            view(target, (page * 64 + position % 64) * row, count * row)?,
                            view(source, (position - sequence.base) * row, count * row)?,
                        )?;
                    }
                    position += count;
                }
                let completed = (pending + sequence.rows) / 8;
                if let Some(summary) = region.summaries.filter(|_| completed > 0) {
                    ensure!(completed <= stage.regions, "Step-5 staged summary capacity");
                    let target = cache.request_component(summary, sequence.sequence)?;
                    let output = view(target, sequence.base / 8 * 256, completed * 256)?;
                    let sparse = stage.sparse.as_ref().context("Step-5 CSA compressor owner")?;
                    stage.stream.device.run(|| {
                        // SAFETY: admitted raw staging, immutable metadata, summary suffix and
                        // scratch belong to this lane; no old visible summary is overwritten.
                        unsafe {
                            sparse.compress.launch(
                                &[
                                    raw[0].ptr,
                                    raw[1].ptr,
                                    sparse.starts.buffer.ptr,
                                    sparse.counts.buffer.ptr,
                                    output.ptr,
                                    sparse.unrounded.buffer.ptr,
                                ],
                                &[
                                    Scalar::I32(completed as i32),
                                    Scalar::I64((pending + sequence.rows) as i64),
                                ],
                                stage.stream.raw,
                            )
                        }
                    })?;
                }
            }
        }
        guard.complete = true;
        self.complete[rank].insert(layer);
        self.failed = false;
        Ok(())
    }

    fn validate_source(
        &self,
        cache: &CacheRank<'_>,
        source: CuteafdDeviceBuffer,
        bytes: usize,
    ) -> Result<()> {
        ensure!(
            !source.ptr.is_null()
                && source.bytes >= bytes
                && source.device_id == cache.stream.device.id,
            "Step-5 projection source extent/device"
        );
        Ok(())
    }

    fn validate_complete(&self) -> Result<()> {
        ensure!(!self.failed, "Step-5 failed step cannot publish");
        for (rank, owner) in self.ranks.iter().enumerate() {
            let cache = owner.borrow();
            let state = &cache.layout().ranks[rank];
            ensure!(self.projected[rank].is_subset(&self.attended[rank]),
                "Step-5 cannot publish projected layers without attention");
            ensure!(self.complete[rank].len()
                == state.sliding.iter().filter(|s| !s.mtp).count() + state.regions.len(),
                "Step-5 cannot publish an incomplete target step");
        }
        Ok(())
    }

    fn prepare_publication(&self, accepted: &[usize]) -> Result<()> {
        self.validate_complete()?;
        ensure!(accepted.len() == self.sequences.len()
            && accepted.iter().zip(&self.sequences).all(|(&rows, sequence)| rows <= sequence.rows),
            "Step-5 accepted wave extent");
        for (rank, owner) in self.ranks.iter().enumerate() {
            let cache = owner.borrow();
            let stage = self.staging[rank];
            let mut guard = stage.stream.cancellation_guard("Step-5 prepare publication");
            guard.complete = self.submission.get();
            for (sequence, &accepted) in self.sequences.iter().zip(accepted) {
                if accepted == 0 { continue; }
                let end = sequence.base + accepted;
                let raw_offset = sequence.raw + (sequence.base % 8 + accepted) / 8 * 8;
                let tail = end % 8;
                for (at, region) in cache.layout().ranks[rank].regions.iter().enumerate() {
                    let targets = cache.pending(region.layer, sequence.sequence, true)?;
                    for (target, source) in [targets.0, targets.1].into_iter().zip([
                        stage.raw[at].0.buffer, stage.raw[at].1.buffer,
                    ]) {
                        zero(&stage.stream, target)?;
                        if tail > 0 {
                            copy(&stage.stream, view(target, 0, tail * 512)?,
                                view(source, raw_offset * 512, tail * 512)?)?;
                        }
                    }
                }
                let target = cache.length(sequence.sequence, true)?;
                stage.stream.device.run(|| {
                    // SAFETY: private length bank and retained source survive lane drainage.
                    unsafe { stage.stream.device.library.copy_h2d_ordered(
                        target, &(end as u64).to_le_bytes(), stage.stream.raw) }
                })?;
            }
            guard.complete = true;
        }
        super::execution::collect_lane_results(self.staging.iter().map(|stage| stage.stream.drain()))
    }

    pub fn publish(mut self, accepted: usize) -> Result<()> {
        ensure!(self.lease.is_some() && accepted <= self.rows,
            "Step-5 diagnostic publication requires an owned page lease");
        if accepted == 0 { return Ok(()); }
        self.prepare_publication(&[accepted])?;
        let family = Step5Prefix::new(self.ranks.clone())?;
        let (pool, placement) = self.lease.as_mut().context("Step-5 page lease absent")?;
        // No active bank changes until all writes drain and the old lease releases.
        release_at_publish(pool, &family, &placement.pages)?;
        for owner in &self.ranks { owner.borrow().publish_bank(placement.sequence); }
        placement.pages = std::mem::take(&mut self.sequences[0].pages);
        placement.committed = self.base + accepted;
        self.live = false;
        Ok(())
    }

    /// Scheduler-owned pages are never released or replaced here. All native
    /// banks publish together after every rank has prepared and drained.
    pub(super) fn publish_wave(mut self, placements: &mut [(&mut Placement, usize)]) -> Result<()> {
        ensure!(self.lease.is_none() && placements.len() == self.sequences.len(),
            "Step-5 scheduler publication inventory");
        for ((placement, accepted), sequence) in placements.iter().zip(&self.sequences) {
            ensure!(placement.sequence == sequence.sequence && placement.pages == sequence.pages
                && *accepted <= sequence.rows
                && (placement.committed == sequence.base || placement.committed == sequence.base + *accepted),
                "Step-5 scheduler publication frontier differs from retained evidence");
        }
        let accepted: Vec<_> = placements.iter().map(|(_, rows)| *rows).collect();
        self.prepare_publication(&accepted)?;
        for ((placement, accepted), sequence) in placements.iter_mut().zip(&self.sequences) {
            if *accepted > 0 {
                for owner in &self.ranks { owner.borrow().publish_bank(sequence.sequence); }
            }
            placement.committed = sequence.base + *accepted;
        }
        self.live = false;
        Ok(())
    }
}

pub(super) fn validate_active_rows(rows: usize, capacity: usize) -> Result<()> {
    ensure!(rows > 0 && rows <= capacity, "Step-5 active rows outside admitted capacity");
    Ok(())
}

struct SubmissionMode(Rc<Cell<bool>>);
impl SubmissionMode {
    fn enter(active: Rc<Cell<bool>>) -> Result<Self> {
        ensure!(!active.get(), "Step-5 nested submission scope");
        active.set(true);
        Ok(Self(active))
    }
}
impl Drop for SubmissionMode {
    fn drop(&mut self) { self.0.set(false); }
}

impl<E: CopyEngine> Drop for Step<'_, '_, E> {
    fn drop(&mut self) {
        if self.live && self.active {
            for stage in &self.staging { stage.stream.drain_or_abort("Step-5 abandoned proposal"); }
        }
        if self.active {
            for (owner, stage) in self.ranks.iter().zip(&self.staging) {
                for sequence in &self.sequences { owner.borrow().busy[sequence.sequence as usize].set(false); }
                stage.busy.set(false);
            }
        }
        if !self.live { return; }
        if let Some((pool, _)) = &self.lease {
            if let Ok(family) = Step5Prefix::new(self.ranks.clone()) {
                if let Err(error) = abort(pool, &family, &self.sequences[0].pages) {
                    for owner in &self.ranks { owner.borrow().poisoned.set(true); }
                    tracing::error!(%error, "Step-5 aborted page lease quarantined");
                }
            }
        }
    }
}

fn copy(
    stream: &Stream<'_>,
    target: CuteafdDeviceBuffer,
    source: CuteafdDeviceBuffer,
) -> Result<()> {
    stream.device.run(|| {
        // SAFETY: checked disjoint views and caller-owned inputs remain live through the step drain.
        unsafe {
            stream.device.library.copy_d2d_async(
                target,
                source,
                source.bytes,
                stream.raw,
            )
        }
    })
}
fn zero(stream: &Stream<'_>, target: CuteafdDeviceBuffer) -> Result<()> {
    stream.device.run(|| {
        // SAFETY: private pending bank is not read by the committed frontier.
        unsafe {
            stream.device.library.cuda_zero_bytes_async(
                target,
                target.bytes,
                stream.raw,
            )
        }
    })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use cuteafd_engine::prefix::{BoxError, PrefixConfig, PrefixFamily};
    use cuteafd_hostcache::copy::StubCopyEngine;

    #[test]
    fn failed_selection_retires_pending_before_same_slot_verification() -> Result<()> {
        #[derive(Default)]
        struct DeviceWork {
            queued: Vec<u32>,
            writes: Vec<u32>,
            events: Vec<&'static str>,
            busy: bool,
            released: bool,
        }
        struct Evidence(Rc<RefCell<DeviceWork>>);
        impl Drop for Evidence {
            fn drop(&mut self) {
                let mut work = self.0.borrow_mut();
                // A Step may synchronize stale evidence, but never enqueue a
                // write or release scheduler-owned page references on Drop.
                assert!(work.queued.is_empty() || !work.released);
                let queued = std::mem::take(&mut work.queued);
                work.writes.extend(queued);
                work.busy = false;
                work.events.push("retire");
            }
        }
        let pending = Pending::default();
        let device = Rc::new(RefCell::new(DeviceWork::default()));
        let mut pool = RefPagePool::new(1, 64);
        let first = pool.alloc(1)?;
        pending.verify(|| {
            let mut work = device.borrow_mut();
            work.busy = true;
            work.queued.push(first[0]);
            work.events.push("verify");
            Ok((Evidence(device.clone()), ()))
        })?;
        // Selection fails before synchronization. The scheduler's failure path
        // drains the family, then releases pages and the native sequence slot.
        {
            let mut work = device.borrow_mut();
            work.events.push("selection failed");
            let queued = std::mem::take(&mut work.queued);
            work.writes.extend(queued);
            work.events.push("family drain");
        }
        pool.release(&first);
        device.borrow_mut().released = true;
        device.borrow_mut().events.push("scheduler release");
        let reused = pool.alloc(1)?;
        assert_eq!(first, reused);
        pending.verify(|| {
            let mut work = device.borrow_mut();
            assert!(!work.busy);
            assert_eq!(work.writes, first);
            assert_eq!(work.events.last(), Some(&"retire"));
            work.released = false;
            work.busy = true;
            work.queued.push(reused[0]);
            work.events.push("verify reused");
            Ok((Evidence(device.clone()), ()))
        })?;
        assert_eq!(device.borrow().events, ["verify", "selection failed", "family drain",
            "scheduler release", "retire", "verify reused"]);
        assert_eq!(device.borrow().writes, first);
        pending.retire();
        assert_eq!(device.borrow().writes, [first[0], reused[0]]);
        pool.release(&reused);
        assert_eq!(pool.free(), 1);
        Ok(())
    }

    #[test]
    fn expert_descriptors_follow_owned_sequence_positions_and_explicit_kind() -> Result<()> {
        use cuteafd_transport::ExpertV2SourceKind;
        let kept = retain_wave(&[
            SequenceRows { sequence: 7, start: 0, rows: 2, base: 65, pages: &[4, 5] },
            SequenceRows { sequence: 2, start: 2, rows: 1, base: 91, pages: &[4, 6] },
        ])?;
        for kind in [ExpertV2SourceKind::Prefill, ExpertV2SourceKind::Decode, ExpertV2SourceKind::MtpVerify] {
            let rows = expert_rows(&kept, kind)?;
            assert_eq!(rows.iter().map(|row| (row.source_request_id, row.token_position)).collect::<Vec<_>>(),
                [(7, 65), (7, 66), (2, 91)]);
            for (index, row) in rows.iter().enumerate() {
                assert_eq!(row.row_id, index as u64);
                assert_eq!(row.source_kind, kind);
                assert_eq!((row.route_offset, row.route_count), (index as u32 * 8, 8));
            }
        }
        let mut invalid = kept.clone();
        invalid[1].start = 1;
        assert!(expert_rows(&invalid, ExpertV2SourceKind::Decode).is_err());
        invalid[1].start = 2;
        invalid[1].base = usize::MAX;
        assert!(expert_rows(&invalid, ExpertV2SourceKind::Decode).is_err());
        Ok(())
    }

    #[test]
    fn retained_wave_owns_descriptors_and_disjoint_pending_prefixes() {
        let mut first = vec![4, 5];
        let second = vec![4, 6];
        let kept = retain_wave(&[
            SequenceRows { sequence: 7, start: 0, rows: 17, base: 65, pages: &first },
            SequenceRows { sequence: 2, start: 17, rows: 3, base: 71, pages: &second },
        ]).unwrap();
        first.fill(99);
        assert_eq!(kept[0].pages, [4, 5]);
        assert_eq!(kept[1].pages, [4, 6]);
        assert_eq!((kept[0].sequence, kept[0].start, kept[0].raw), (7, 0, 0));
        assert_eq!((kept[1].sequence, kept[1].start, kept[1].raw), (2, 17, 18));
        assert_eq!(kept[1].raw + kept[1].base % 8 + kept[1].rows, 28);
        for sequences in 1..=16 {
            let pages = [0];
            let rows: Vec<_> = (0..sequences).map(|sequence| SequenceRows {
                sequence: sequence as u64, start: sequence, rows: 1, base: 7, pages: &pages,
            }).collect();
            let kept = retain_wave(&rows).unwrap();
            for (index, sequence) in kept.iter().enumerate() { assert_eq!(sequence.raw, index * 8); }
            assert_eq!(kept.last().unwrap().raw + 8, retained_wave_rows(sequences, sequences).unwrap());
        }
        assert!(retain_wave(&[SequenceRows { sequence: 0, start: 0, rows: usize::MAX,
            base: 7, pages: &[] }]).is_err());
    }

    #[test]
    fn wave_admission_shares_only_immutable_pages_and_preserves_row_slices() {
        let mut pool = RefPagePool::new(16, 64);
        let source = pool.alloc(2).unwrap();
        let first = pool.fork(&source, 65, 3).unwrap();
        let second = pool.fork(&source, 65, 3).unwrap();
        let sequences = [
            SequenceRows { sequence: 3, start: 0, rows: 1, base: 65, pages: &first.pages },
            SequenceRows { sequence: 1, start: 1, rows: 17, base: 65, pages: &second.pages },
        ];
        assert_eq!(validate_wave(&sequences, &pool, 4, 18, 192, 16).unwrap(), 18);
        assert_eq!(first.pages[0], second.pages[0]);
        assert_ne!(first.pages[1], second.pages[1]);
        assert!(first.copy.is_some() && second.copy.is_some());
        let mut invalid = sequences;
        invalid[1].pages = &first.pages;
        assert!(validate_wave(&invalid, &pool, 4, 18, 192, 16).is_err());
        let readonly = SequenceRows { sequence: 0, start: 0, rows: 1, base: 128, pages: &first.pages };
        let writer = SequenceRows { sequence: 1, start: 1, rows: 1, base: 65, pages: &first.pages };
        assert!(validate_wave(&[readonly, writer], &pool, 4, 18, 192, 16).is_err());
        assert!(validate_wave(&[SequenceRows { start: 0, ..writer }, SequenceRows { start: 1, ..readonly }],
            &pool, 4, 18, 192, 16).is_err());
        let reversed = [SequenceRows { start: 0, ..sequences[1] }, SequenceRows { start: 17, ..sequences[0] }];
        assert_eq!(validate_wave(&reversed, &pool, 4, 18, 192, 16).unwrap(), 18);
        pool.release(&source);
        pool.release(&first.pages);
        pool.release(&second.pages);
        assert_eq!(pool.free(), 16);
    }

    #[test]
    fn wave_admission_rejects_invalid_inventory_before_pool_indexing() {
        let mut pool = RefPagePool::new(8, 64);
        let pages = pool.alloc(3).unwrap();
        let valid = SequenceRows { sequence: 0, start: 0, rows: 2, base: 63, pages: &pages };
        assert_eq!(validate_wave(&[valid], &pool, 2, 16, 192, 8).unwrap(), 2);
        for invalid in [
            SequenceRows { sequence: 2, ..valid },
            SequenceRows { start: 1, ..valid },
            SequenceRows { rows: 0, ..valid },
            SequenceRows { rows: 17, ..valid },
            SequenceRows { base: usize::MAX, ..valid },
            SequenceRows { base: 191, ..valid },
            SequenceRows { pages: &pages[..1], ..valid },
            SequenceRows { pages: &[0, 0], ..valid },
            SequenceRows { pages: &[0, 7], ..valid },
            SequenceRows { pages: &[0, 8], ..valid },
            SequenceRows { pages: &[0, u32::MAX], ..valid },
        ] {
            assert!(validate_wave(&[invalid], &pool, 2, 16, 192, 8).is_err(), "{invalid:?}");
        }
        assert!(validate_wave(&[], &pool, 2, 16, 192, 8).is_err());
        assert!(validate_wave(&[valid], &pool, 2, 16, 192, 9).is_err());
        assert!(validate_wave(&[valid, SequenceRows { start: 2, ..valid }], &pool, 2, 16, 192, 8).is_err());
        pool.share(&pages[1..2]);
        assert!(validate_wave(&[valid], &pool, 2, 16, 192, 8).is_err());
        pool.release(&pages[1..2]);
        let mut reserved = RefPagePool::with_reserved(8, 64, 1);
        let allocation = reserved.alloc(2).unwrap();
        let reserved_valid = SequenceRows { pages: &allocation, ..valid };
        validate_wave(&[reserved_valid], &reserved, 2, 16, 192, 8).unwrap();
        assert!(validate_wave(&[SequenceRows { pages: &[0, 1], ..valid }], &reserved, 2, 16, 192, 8).is_err());
        let wrong_rows = RefPagePool::new(8, 32);
        assert!(validate_wave(&[valid], &wrong_rows, 2, 16, 192, 8).is_err());
        let parity = RefPagePool::parity(8, 64, 0);
        assert!(validate_wave(&[valid], &parity, 2, 16, 192, 8).is_err());
    }

    #[test]
    fn staging_wave_admission_retains_each_pending_prefix_and_page_table() {
        for rows in [1usize, 16, 64, 513, 4096] {
            let layout = super::super::cache::tests::layout_for(8192, 2, 0, rows as u64);
            let single = Staging::bytes(&layout, 0).unwrap();
            let table = (8192usize.div_ceil(64) * 4).max(256);
            for sequences in [1usize, 2, 4, 16, 64] {
                let retained = rows + 7 * sequences.min(rows);
                assert_eq!(retained_wave_rows(rows, sequences).unwrap(), retained);
                let extra_raw = (retained - (rows + 7)) * 512 * 2 * 23;
                assert_eq!(Staging::bytes_for_sequences(&layout, 0, sequences).unwrap(),
                    single + extra_raw + table * (sequences - 1));
            }
        }
        for (rows, sequences) in [(0, 1), (1, 0), (usize::MAX, 1), (usize::MAX, usize::MAX)] {
            assert!(retained_wave_rows(rows, sequences).is_err());
        }
    }

    #[test]
    fn active_rows_do_not_require_a_full_admitted_bucket() {
        for capacity in [1, 16, 64, 640, 4096] {
            for rows in 1..=capacity { validate_active_rows(rows, capacity).unwrap(); }
            for rows in [0, capacity + 1, usize::MAX] {
                assert!(validate_active_rows(rows, capacity).is_err());
            }
        }
        assert!(validate_active_rows(1, 0).is_err());
    }

    #[test]
    fn submission_mode_resets_on_success_error_and_unwind_without_nested_reset() {
        let active = Rc::new(Cell::new(false));
        for panic in [false, true] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _scope = SubmissionMode::enter(active.clone()).unwrap();
                assert!(active.get());
                assert!(SubmissionMode::enter(active.clone()).is_err());
                assert!(active.get());
                if panic { panic!("partial pipeline enqueue"); }
            }));
            assert_eq!(result.is_err(), panic);
            assert!(!active.get());
        }
    }

    #[test]
    fn peer_without_csa_has_no_compressor_workspace_charge() -> Result<()> {
        let mut layout = super::super::cache::tests::layout_for(1024, 2, 0, 64);
        layout.ranks[1].regions.clear();
        layout.geometry.ranks[1].persistent_unit_bytes = 0;
        // Positions, request ids and the two attention page tables only.
        assert_eq!(Staging::bytes(&layout, 1)?, 64 * 8 + 256 + 2 * 256);
        assert!(Staging::bytes(&layout, 0)? > Staging::bytes(&layout, 1)?);
        Ok(())
    }

    fn read(device: Device<'_>, buffer: CuteafdDeviceBuffer) -> Result<Vec<u8>> {
        let mut bytes = vec![0; buffer.bytes];
        device.run(|| device.library.copy_d2h(&mut bytes, buffer))?;
        Ok(bytes)
    }

    fn mark(
        cache: &RefCell<CacheRank<'_>>,
        sequence: u64,
        slot: u64,
        frontier: usize,
    ) -> Result<Vec<u8>> {
        cache
            .borrow_mut()
            .capture(sequence, slot, frontier as u64)?;
        let cache = cache.borrow();
        read(cache.stream.device, cache.mark_view(slot as u32)?)
    }

    /// Other layers are ownership fillers, not a numerical full-target claim.
    pub(in crate::families::step5) fn stage_other_layers<'e, 'a>(
        step: &mut Step<'e, 'a, StubCopyEngine>,
        selected: usize,
        key: &'e dyn BufferOwner,
        value: &'e dyn BufferOwner,
        ik: &'e Allocation<'a>,
        iz: &'e Allocation<'a>,
    ) -> Result<()> {
        for layer in 0..92 {
            if layer != selected {
                step.stage(0, layer, key, value, (layer % 4 == 3).then_some((ik, iz)))?;
            }
        }
        Ok(())
    }

    #[test]
    fn staging_admission_charges_every_csa_layer_and_step_extent() {
        for step in [1, 64, 513, 4096] {
            let layout = super::super::cache::tests::layout_for(8192, 2, 0, step);
            let regions = (step as usize + 7) / 8;
            let expected = 23 * (step as usize + 7) * 512 * 2
                + (regions * 4).max(256) * 2
                + (regions * 1024).max(256)
                + (step as usize * 8).max(256)
                + (step as usize * 4).max(256)
                + (8192usize.div_ceil(64) * 4).max(256) * 2;
            assert_eq!(Staging::bytes(&layout, 0).unwrap(), expected);
            assert_eq!(Staging::bytes(&layout, 1).unwrap(), expected);
            assert!(Staging::bytes(&layout, 2).is_err());
        }
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_PROGRAM_MANIFEST and CUDA"]
    fn native_scheduler_wave_matches_owned_single_sequence_publication() -> Result<()> {
        // SAFETY: the explicitly selected artifact implements the CUDA ABI.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let programs = library.programs()?.with_manifest(std::path::Path::new(
            &std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
        let layout = super::super::cache::tests::layout_for(256, 1, 0, 64);
        let batched = RefCell::new(CacheRank::new(device, layout.clone(), 0, 2, 2, 16, usize::MAX)?);
        let serial = RefCell::new(CacheRank::new(device, layout, 0, 2, 2, 16, usize::MAX)?);
        let wave_stage = Staging::new_for_sequences(device, &programs, &batched.borrow(), usize::MAX, 2)?;
        let serial_stage = Staging::new(device, &programs, &serial.borrow(), usize::MAX)?;
        let wave_family = Step5Prefix::new(vec![&batched])?;
        let serial_family = Step5Prefix::new(vec![&serial])?;
        let config = PrefixConfig { entries: 0, mark_slots: 2, keep_logits: false, min_tokens: 1 };
        let mut wave_pool = PrefixCache::<StubCopyEngine, Frontier>::new(wave_family.layout(), config.clone(), None)?;
        let serial_pool = RefCell::new(PrefixCache::<StubCopyEngine, Frontier>::new(serial_family.layout(), config, None)?);
        let mut wave_placements = Vec::new();
        let mut serial_placements = Vec::new();
        for sequence in 0..2 {
            wave_placements.push(wave_pool.admit_cold(&wave_family, 1, 64,
                |pages| Placement { pages, sequence, committed: 0 })?.placement);
            serial_placements.push(serial_pool.borrow_mut().admit_cold(&serial_family, 1, 64,
                |pages| Placement { pages, sequence, committed: 0 })?.placement);
        }
        for (pass, (counts, accepted)) in [([7usize, 3], [7usize, 3]), ([17, 5], [9, 4])].into_iter().enumerate() {
            let total: usize = counts.iter().sum();
            let make = |stride: usize, start: usize, rows: usize, salt: usize| -> Result<Allocation<'_>> {
                let allocation = Allocation::new(device, (rows * stride).max(256))?;
                let bytes: Vec<u8> = (start..start + rows).flat_map(|row|
                    std::iter::repeat_n(((row + pass * 13 + salt) % 31 + 1) as u8, stride)).collect();
                device.run(|| library.copy_h2d(allocation.buffer, &bytes))?;
                Ok(allocation)
            };
            let key = make(4 * 192 * 2, 0, total, 0)?;
            let value = make(4 * 192 * 2, 0, total, 3)?;
            let ik = make(512, 0, total, 7)?;
            let iz = make(512, 0, total, 11)?;
            let descriptors = [
                SequenceRows { sequence: 0, start: 0, rows: counts[0], base: wave_placements[0].committed,
                    pages: &wave_placements[0].pages },
                SequenceRows { sequence: 1, start: counts[0], rows: counts[1], base: wave_placements[1].committed,
                    pages: &wave_placements[1].pages },
            ];
            let mut wave = Step::<StubCopyEngine>::begin_wave(vec![&batched], vec![&wave_stage], &descriptors, wave_pool.pool())?;
            // No scheduler placement or allocator borrow remains in the proposal.
            let owned_pages: Vec<_> = wave_placements.iter().map(|placement| placement.pages.clone()).collect();
            for layer in 0..92 {
                wave.stage(0, layer, &key, &value, (layer % 4 == 3).then_some((&ik, &iz)))?;
            }
            for placement in &mut wave_placements { placement.committed += accepted[placement.sequence as usize]; }
            let mut publications: Vec<_> = wave_placements.iter_mut().zip(accepted).collect();
            wave.publish_wave(&mut publications)?;
            for (placement, pages) in wave_placements.iter().zip(owned_pages) { assert_eq!(placement.pages, pages); }
            let mut start = 0;
            for sequence in 0..2 {
                let key = make(4 * 192 * 2, start, counts[sequence], 0)?;
                let value = make(4 * 192 * 2, start, counts[sequence], 3)?;
                let ik = make(512, start, counts[sequence], 7)?;
                let iz = make(512, start, counts[sequence], 11)?;
                let mut step = Step::begin(vec![&serial], vec![&serial_stage], &serial_pool,
                    &mut serial_placements[sequence], counts[sequence])?;
                for layer in 0..92 {
                    step.stage(0, layer, &key, &value, (layer % 4 == 3).then_some((&ik, &iz)))?;
                }
                step.publish(accepted[sequence])?;
                let frontier = wave_placements[sequence].committed;
                assert_eq!(frontier, serial_placements[sequence].committed);
                assert_eq!(mark(&batched, sequence as u64, sequence as u64, frontier)?,
                    mark(&serial, sequence as u64, sequence as u64, frontier)?);
                for layer in (3..92).step_by(4) {
                    let wave_planes = batched.borrow().csa_planes(layer)?;
                    let serial_planes = serial.borrow().csa_planes(layer)?;
                    for (wave_plane, serial_plane) in [wave_planes.0, wave_planes.1].into_iter()
                        .zip([serial_planes.0, serial_planes.1]) {
                        let bytes = frontier * 4 * 192 * 2;
                        assert_eq!(read(device, view(wave_plane,
                            wave_placements[sequence].pages[0] as usize * 64 * 4 * 192 * 2, bytes)?)?,
                            read(device, view(serial_plane,
                            serial_placements[sequence].pages[0] as usize * 64 * 4 * 192 * 2, bytes)?)?);
                    }
                }
                start += counts[sequence];
            }
        }
        for placement in wave_placements { wave_pool.release(&wave_family, &placement.pages)?; }
        for placement in serial_placements { serial_pool.borrow_mut().release(&serial_family, &placement.pages)?; }
        assert_eq!(wave_pool.pool().free(), 16);
        assert_eq!(serial_pool.borrow().pool().free(), 16);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_PROGRAM_MANIFEST and CUDA"]
    fn native_transaction_abort_and_whole_target_publication_are_exact(
    ) -> std::result::Result<(), BoxError> {
        // SAFETY: the explicitly selected artifact implements the CUDA ABI.
        let library =
            unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let programs = library
            .programs()?
            .with_manifest(std::path::Path::new(&std::env::var(
                "CUTEAFD_PROGRAM_MANIFEST",
            )?))?;
        // Two logical head ranks on one device qualify ownership without claiming
        // transport/peer execution. Both rank-local streams and banks participate.
        let layout = super::super::cache::tests::layout_for(1024, 2, 0, 64);
        let owners = (0..2)
            .map(|rank| {
                Ok(RefCell::new(CacheRank::new(
                    device,
                    layout.clone(),
                    rank,
                    2,
                    3,
                    40,
                    usize::MAX,
                )?))
            })
            .collect::<Result<Vec<_>>>()?;
        let stages = owners
            .iter()
            .map(|owner| Staging::new(device, &programs, &owner.borrow(), usize::MAX))
            .collect::<Result<Vec<_>>>()?;
        assert!(Staging::new(
            device,
            &programs,
            &owners[0].borrow(),
            stages[0].allocation_bytes() - 1
        )
        .is_err());
        let refs: Vec<_> = owners.iter().collect();
        let lanes: Vec<_> = stages.iter().collect();
        let family = Step5Prefix::new(refs.clone())?;
        let pool = RefCell::new(PrefixCache::<StubCopyEngine, Frontier>::new(
            family.layout(),
            PrefixConfig {
                entries: 0,
                mark_slots: 3,
                keep_logits: false,
                min_tokens: 1,
            },
            None,
        )?);
        let input = |bytes| -> Result<Allocation<'_>> {
            let allocation = Allocation::new(device, bytes)?;
            // BF16 1.0, valid for both raw projections and compressor logits.
            let data: Vec<u8> = [0x80, 0x3f].into_iter().cycle().take(bytes).collect();
            device.run(|| library.copy_h2d(allocation.buffer, &data))?;
            Ok(allocation)
        };
        let key = input(64 * 2 * 192 * 2)?;
        let value = input(key.buffer.bytes)?;
        let ik = input(64 * 512)?;
        let iz = input(64 * 512)?;
        let second_stages = owners.iter()
            .map(|owner| Staging::new(device, &programs, &owner.borrow(), usize::MAX))
            .collect::<Result<Vec<_>>>()?;
        for (first, second) in stages.iter().zip(&second_stages) {
            assert_ne!(first.stream.raw, second.stream.raw);
        }
        let mut first = pool.borrow_mut().admit_cold(&family, 1, 1, |pages|
            Placement { pages, sequence: 0, committed: 0 })?.placement;
        let mut second = pool.borrow_mut().admit_cold(&family, 1, 1, |pages|
            Placement { pages, sequence: 1, committed: 0 })?.placement;
        let free = pool.borrow().pool().free();
        {
            let mut first_step = Step::begin(refs.clone(), lanes.clone(), &pool, &mut first, 64)?;
            first_step.stage(0, 0, &key, &value, None)?;
            // The first lane has queued work and retains its owned lease, not a pool borrow.
            assert!(pool.try_borrow_mut().is_ok());
            let mut second_step = Step::begin(refs.clone(), second_stages.iter().collect(),
                &pool, &mut second, 64)?;
            second_step.stage(0, 0, &key, &value, None)?;
            assert!(first_step.pages().iter().all(|page| !second_step.pages().contains(page)));
            // Cache capture on the busy sequence remains forbidden even though the allocator is free.
            assert!(owners[0].borrow_mut().capture(0, 0, 0).is_err());
            drop(second_step);
            first_step.stage(1, 0, &key, &value, None)?;
        }
        assert_eq!(pool.borrow().pool().free(), free);
        assert_eq!((first.committed, second.committed), (0, 0));
        pool.borrow_mut().release(&family, &first.pages)?;
        pool.borrow_mut().release(&family, &second.pages)?;
        assert_eq!(pool.borrow().pool().free(), 40);

        for base in [0, 7, 63, 511, 575, 576] {
            let mut placement = pool.borrow_mut()
                .admit_cold(&family, base.max(1), base.max(1), |pages| Placement {
                    pages,
                    sequence: 0,
                    committed: base,
                })?
                .placement;
            let old_pages = placement.pages.clone();
            let old_free = pool.borrow().pool().free();
            // Seed arbitrary old visible bytes, including pending and summary state.
            for owner in &owners {
                let cache = owner.borrow();
                for span in layout.snapshot_spans(cache.rank(), base as u64, &[])? {
                    let state = &layout.ranks[cache.rank()];
                    let component = std::iter::once(state.length)
                        .chain(state.sliding.iter().flat_map(|r| [r.key, r.value]))
                        .chain(state.regions.iter().flat_map(|r| {
                            [Some(r.pending_key), Some(r.pending_z), r.summaries]
                                .into_iter()
                                .flatten()
                        }))
                        .find(|s| {
                            span.offset >= s.offset
                                && span.offset + span.bytes <= s.offset + s.bytes
                        })
                        .unwrap();
                    let target = cache.request_component(component, 0)?;
                    let target = view(
                        target,
                        (span.offset - component.offset) as usize,
                        span.bytes as usize,
                    )?;
                    device.run(|| library.copy_h2d(target, &vec![0x39; target.bytes]))?;
                }
                device.run(|| {
                    library.copy_h2d(cache.length(0, false)?, &(base as u64).to_le_bytes())
                })?;
                for &page in &old_pages {
                    for plane in cache.page_views(page)? {
                        device.run(|| library.copy_h2d(plane, &vec![0x29; plane.bytes]))?;
                    }
                }
            }
            let saved = owners
                .iter()
                .map(|owner| mark(owner, 0, 0, base))
                .collect::<Result<Vec<_>>>()?;
            let other = owners
                .iter()
                .map(|owner| mark(owner, 1, 1, base))
                .collect::<Result<Vec<_>>>()?;
            let page_bytes = owners
                .iter()
                .map(|owner| {
                    old_pages
                        .iter()
                        .flat_map(|p| owner.borrow().page_views(*p).unwrap())
                        .map(|p| read(device, p))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            {
                let mut step = Step::begin(refs.clone(), lanes.clone(), &pool, &mut placement, 64)?;
                assert!(step.compose(2, &key, |_, _| Ok(())).is_err());
                step.compose(0, &key, |_, meta| {
                    assert_eq!(meta.rows, 64);
                    assert_eq!(meta.base, base);
                    assert_eq!(meta.heads, 32);
                    Ok(())
                })?;
                let failed: Result<()> = step.compose(0, &key, |owner, meta| {
                    meta.stream.device.run(|| {
                        // SAFETY: the transaction retains both disjoint owners through failure drainage.
                        unsafe { library.copy_d2d_async(value.buffer, owner.buffer,
                            owner.buffer.bytes, meta.stream.raw) }
                    })?;
                    anyhow::bail!("injected composition error")
                });
                assert!(failed.is_err());
                assert!(step.compose(0, &key, |_, _| Ok(())).is_err());
                assert!(step.publish(1).is_err());
            }
            assert_eq!(placement.pages, old_pages);
            assert_eq!(placement.committed, base);
            assert_eq!(pool.borrow().pool().free(), old_free);
            let compose_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                let mut step = Step::begin(refs.clone(), lanes.clone(), &pool, &mut placement, 64)?;
                step.compose(0, &key, |_, _| -> Result<()> {
                    panic!("injected composition panic");
                })
            }));
            assert!(compose_panic.is_err());
            assert_eq!(placement.pages, old_pages);
            assert_eq!(placement.committed, base);
            assert_eq!(pool.borrow().pool().free(), old_free);
            let killed =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                    let mut step =
                        Step::begin(refs.clone(), lanes.clone(), &pool, &mut placement, 64)?;
                    step.stage(0, 0, &key, &value, None)?;
                    step.stage(0, 3, &key, &value, Some((&ik, &iz)))?;
                    assert!(owners[0].borrow_mut().restore(0, 0, base as u64).is_err());
                    panic!("injected mid-target cancellation");
                }));
            assert!(killed.is_err());
            assert_eq!(placement.committed, base);
            assert_eq!(placement.pages, old_pages);
            assert_eq!(pool.borrow().pool().free(), old_free);
            for (rank, owner) in owners.iter().enumerate() {
                assert_eq!(mark(owner, 0, 2, base)?, saved[rank]);
                let after = old_pages
                    .iter()
                    .flat_map(|p| owner.borrow().page_views(*p).unwrap())
                    .map(|p| read(device, p))
                    .collect::<Result<Vec<_>>>()?;
                assert_eq!(after, page_bytes[rank]);
                owner.borrow_mut().restore(0, 0, base as u64)?;
                assert_eq!(mark(owner, 0, 2, base)?, saved[rank]);
                assert_eq!(mark(owner, 1, 2, base)?, other[rank]);
            }
            // An incomplete target cannot publish even after valid layer writes.
            let mut step = Step::begin(refs.clone(), lanes.clone(), &pool, &mut placement, 64)?;
            step.stage(0, 0, &key, &value, None)?;
            assert!(step.publish(1).is_err());
            assert_eq!(placement.committed, base);
            for accepted in [0, 1, 7, 8, 63, 64] {
                let before = placement.committed;
                let mut step =
                    Step::begin(refs.clone(), lanes.clone(), &pool, &mut placement, 64)?;
                for rank in 0..2 {
                    for layer in 0..92 {
                        step.stage(
                            rank,
                            layer,
                            &key,
                            &value,
                            (layer % 4 == 3).then_some((&ik, &iz)),
                        )?;
                    }
                }
                step.publish(accepted)?;
                assert_eq!(placement.committed, before + accepted);
                for (rank, owner) in owners.iter().enumerate() {
                    let cache = owner.borrow();
                    assert_eq!(
                        read(device, cache.length(0, false)?)?,
                        ((before + accepted) as u64).to_le_bytes()
                    );
                    drop(cache);
                    assert_eq!(mark(owner, 1, 2, base)?, other[rank]);
                    let committed = mark(owner, 0, 0, placement.committed)?;
                    owner
                        .borrow_mut()
                        .restore(0, 0, placement.committed as u64)?;
                    assert_eq!(mark(owner, 0, 2, placement.committed)?, committed);
                }
            }
            pool.borrow_mut().release(&family, &placement.pages)?;
            assert_eq!(pool.borrow().pool().free(), 40);
        }
        Ok(())
    }
}
