//! Append-only target steps: mutable proposal rows never overlap the old frontier.
use super::{
    cache::CacheRank,
    prefix::{Frontier, Placement, Step5Prefix},
};
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::PrefixCache;
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
}

pub(super) struct StepInputs<'s, 'a> {
    pub stream: &'s crate::shared::memory::device::Stream<'a>,
    pub positions: CuteafdDeviceBuffer,
    pub requests: CuteafdDeviceBuffer,
    pub rows: usize,
    pub base: usize,
    pub heads: usize,
}

/// Borrowed only inside the transaction callback, after this layer was staged.
/// The transaction retains page leases, staging metadata and all queued inputs.
pub(super) struct AttentionRead<'s, 'a> {
    pub inputs: StepInputs<'s, 'a>,
    pub key: CuteafdDeviceBuffer,
    pub value: CuteafdDeviceBuffer,
    pub summaries: Option<CuteafdDeviceBuffer>,
    pub table: CuteafdDeviceBuffer,
    pub table_width: usize,
    pub pages: usize,
}

impl<'a> Staging<'a> {
    pub fn bytes(
        layout: &cuteafd_loader::serving_capacity::step5::Step5CacheLayout,
        rank: usize,
    ) -> Result<usize> {
        let layers = layout
            .ranks
            .get(rank)
            .context("Step-5 staging rank")?
            .regions
            .len();
        let rows = usize::try_from(layout.ring.step)?
            .checked_add(7)
            .context("Step-5 staging rows")?;
        let regions = rows / 8;
        let metadata = regions
            .checked_mul(4)
            .context("Step-5 region metadata overflow")?
            .max(256);
        let scratch = regions
            .checked_mul(256 * 4)
            .context("Step-5 region scratch overflow")?
            .max(256);
        let step = usize::try_from(layout.ring.step)?;
        let table = usize::try_from(layout.context.div_ceil(64))?
            .checked_mul(4)
            .context("Step-5 attention table overflow")?
            .max(256);
        rows.checked_mul(256 * 2 * 2)
            .and_then(|n| n.checked_mul(layers))
            .and_then(|n| n.checked_add(if layers == 0 { 0 } else { metadata.checked_mul(2)? }))
            .and_then(|n| n.checked_add(if layers == 0 { 0 } else { scratch }))
            .and_then(|n| n.checked_add(step.checked_mul(8)?.max(256)))
            .and_then(|n| n.checked_add(step.checked_mul(4)?.max(256)))
            .and_then(|n| n.checked_add(table.checked_mul(2)?))
            .context("Step-5 staging bytes overflow")
    }

    pub fn new(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        cache: &CacheRank<'a>,
        available: usize,
    ) -> Result<Self> {
        let bytes = Self::bytes(cache.layout(), cache.rank())?;
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
                    Allocation::new(device, (rows + 7) * 256 * 2)?,
                    Allocation::new(device, (rows + 7) * 256 * 2)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let table_width = usize::try_from(cache.layout().context.div_ceil(64))?;
        let owner = Self {
            raw,
            sparse,
            positions: Allocation::new(device, (rows * 8).max(256))?,
            requests: Allocation::new(device, (rows * 4).max(256))?,
            ring_table: Allocation::new(device, (table_width * 4).max(256))?,
            page_table: Allocation::new(device, (table_width * 4).max(256))?,
            table_width,
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
    pub fn allocation_bytes(&self) -> usize {
        self.bytes
    }

    pub fn owns_compressor(&self, cache: &CacheRank<'_>, layer: usize) -> bool {
        self.cache_owner == cache.identity() && self.sparse.is_some()
            && cache.layout().ranks[cache.rank()].regions.iter().position(|r| r.layer == layer)
                .and_then(|index| self.raw.get(index))
                .is_some_and(|(key, logits)| !key.buffer.ptr.is_null() && !logits.buffer.ptr.is_null())
    }
}

/// Exclusively borrows the placement and lane through preparation, publication,
/// or Drop. Shared prefix allocation owns every proposal page reference.
pub(super) struct Step<'e, 'a, E: CopyEngine> {
    ranks: Vec<&'e RefCell<CacheRank<'a>>>,
    staging: Vec<&'e Staging<'a>>,
    pool: &'e mut PrefixCache<E, Frontier>,
    placement: &'e mut Placement,
    pages: Option<Vec<u32>>,
    complete: Vec<BTreeSet<usize>>,
    projected: Vec<BTreeSet<usize>>,
    attended: Vec<BTreeSet<usize>>,
    rows: usize,
    base: usize,
    failed: bool,
    inputs: Vec<&'e Allocation<'a>>,
    submission: Rc<Cell<bool>>,
}

impl<'e, 'a, E: CopyEngine> Step<'e, 'a, E> {
    pub fn validate_binding(&self, ranks: &[&RefCell<CacheRank<'a>>],
        stages: &[&Staging<'a>], rows: usize) -> Result<()> {
        ensure!(!self.failed && self.rows == rows && ranks.len() == self.ranks.len()
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

    pub fn begin(
        ranks: Vec<&'e RefCell<CacheRank<'a>>>,
        staging: Vec<&'e Staging<'a>>,
        pool: &'e mut PrefixCache<E, Frontier>,
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
        let pages = pool.fork_step(&family, &placement.pages, base, end)?;
        for (cache, stage) in ranks.iter().zip(&staging) {
            cache.borrow().busy[placement.sequence as usize].set(true);
            stage.busy.set(true);
        }
        let complete = vec![BTreeSet::new(); ranks.len()];
        let step = Self {
            ranks,
            staging,
            pool,
            placement,
            pages: Some(pages),
            projected: complete.clone(),
            attended: complete.clone(),
            complete,
            rows,
            base,
            failed: false,
            inputs: Vec::new(),
            submission: Rc::new(Cell::new(false)),
        };
        let positions: Vec<u8> = (base..end).flat_map(|p| (p as i64).to_le_bytes()).collect();
        let table: Vec<u8> = step
            .pages()
            .iter()
            .flat_map(|p| (*p as i32).to_le_bytes())
            .collect();
        for (owner, stage) in step.ranks.iter().zip(&step.staging) {
            let cache = owner.borrow();
            ensure!(
                step.pages().len() <= stage.table_width,
                "Step-5 proposal table capacity"
            );
            cache.stream.device.run(|| {
                // SAFETY: the step owns metadata until its Drop drain; ordered uploads retain sources.
                unsafe {
                    cache.stream.device.library.copy_h2d_ordered(
                        stage.positions.buffer,
                        &positions,
                        cache.stream.raw,
                    )?;
                    cache.stream.device.library.copy_h2d_ordered(
                        stage.page_table.buffer,
                        &table,
                        cache.stream.raw,
                    )
                }
            })?;
        }
        Ok(step)
    }

    pub fn pages(&self) -> &[u32] {
        self.pages.as_deref().expect("unpublished Step-5 step")
    }

    /// Every producer/consumer joins the cache stream. Failure poisons this
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
        let mut guard = cache
            .stream
            .cancellation_guard("Step-5 projection callback");
        guard.complete = self.submission.get();
        let result = work(
            input_owner,
            StepInputs {
                stream: &cache.stream,
                positions: stage.positions.buffer,
                requests: stage.requests.buffer,
                rows: self.rows,
                base: self.base,
                heads: cache.layout().query_heads(rank, layer)?,
            },
        );
        if result.is_ok() {
            self.failed = false;
            guard.complete = true;
        }
        result
    }

    /// Pin non-attention owners and order chain copies/FFNs on the cache stream.
    pub fn compose<O, T>(
        &mut self,
        rank: usize,
        owner: &'e O,
        work: impl FnOnce(&'e O, StepInputs<'_, 'a>) -> Result<T>,
    ) -> Result<T> {
        ensure!(!self.failed, "Step-5 failed step cannot compose");
        let cache = self.ranks.get(rank).context("Step-5 composition rank")?.borrow();
        self.failed = true;
        let mut guard = cache.stream.cancellation_guard("Step-5 composition callback");
        guard.complete = self.submission.get();
        let stage = self.staging[rank];
        let result = work(owner, StepInputs {
            stream: &cache.stream,
            positions: stage.positions.buffer,
            requests: stage.requests.buffer,
            rows: self.rows,
            base: self.base,
            heads: 64 / self.ranks.len(),
        });
        if result.is_ok() {
            self.failed = false;
            guard.complete = true;
        }
        result
    }

    pub fn attend<O, T>(
        &mut self,
        rank: usize,
        layer: usize,
        input_owner: &'e O,
        work: impl FnOnce(&'e O, AttentionRead<'_, 'a>) -> Result<T>,
    ) -> Result<T> {
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
        let (key, value, summaries, table, pages) =
            if let Some(ring) = state.sliding.iter().find(|r| r.layer == layer && !r.mtp) {
                (
                    cache.request_component(ring.key, self.placement.sequence)?,
                    cache.request_component(ring.value, self.placement.sequence)?,
                    None,
                    stage.ring_table.buffer,
                    usize::try_from(cache.layout().ring.rows / 64)?,
                )
            } else {
                let region = state
                    .regions
                    .iter()
                    .find(|r| r.layer == layer)
                    .context("Step-5 attention layer")?;
                let (key, value) = cache.csa_planes(layer)?;
                let summary = region
                    .summaries
                    .map(|s| cache.request_component(s, self.placement.sequence))
                    .transpose()?;
                (
                    key,
                    value,
                    summary,
                    stage.page_table.buffer,
                    usize::try_from(cache.page_count())?,
                )
            };
        self.failed = true;
        let mut guard = cache.stream.cancellation_guard("Step-5 attention callback");
        guard.complete = self.submission.get();
        let result = work(
            input_owner,
            AttentionRead {
                inputs: StepInputs {
                    stream: &cache.stream,
                    positions: stage.positions.buffer,
                    requests: stage.requests.buffer,
                    rows: self.rows,
                    base: self.base,
                    heads: cache.layout().query_heads(rank, layer)?,
                },
                key,
                value,
                summaries,
                table,
                table_width: stage.table_width,
                pages,
            },
        );
        if result.is_ok() {
            self.attended[rank].insert(layer);
            self.failed = false;
            guard.complete = true;
        }
        result
    }

    /// BF16 projections are ordered on the cache stream. Borrowed allocations
    /// remain live through Drop's drain, including partial-enqueue cancellation.
    pub fn stage(
        &mut self,
        rank: usize,
        layer: usize,
        key: &'e Allocation<'a>,
        value: &'e Allocation<'a>,
        index: Option<(&'e Allocation<'a>, &'e Allocation<'a>)>,
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
        let (key, value) = (key.buffer, value.buffer);
        let index = index.map(|(key, z)| (key.buffer, z.buffer));
        let cache = self.ranks[rank].borrow();
        let state = &cache.layout().ranks[rank];
        let sequence = self.placement.sequence;
        let mut guard = cache.stream.cancellation_guard("Step-5 staged layer");
        guard.complete = self.submission.get();
        if let Some(ring) = state.sliding.iter().find(|r| r.layer == layer && !r.mtp) {
            ensure!(index.is_none(), "SWA must not carry indexer state");
            let row = ring.key.bytes as usize / cache.layout().ring.rows as usize;
            let targets = [
                cache.request_component(ring.key, sequence)?,
                cache.request_component(ring.value, sequence)?,
            ];
            let sources = [key, value];
            for source in sources {
                self.validate_source(&cache, source, self.rows * row)?;
            }
            for run in cache
                .layout()
                .ring
                .proposal(self.base as u64, self.rows as u64)
                .context("Step-5 ring proposal")?
            {
                for (target, source) in targets.into_iter().zip(sources) {
                    copy(
                        &cache,
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
                self.validate_source(&cache, source, self.rows * row)?;
            }
            let (ik, iz) = index.context("CSA requires indexer K/Z")?;
            for source in [ik, iz] {
                self.validate_source(&cache, source, self.rows * 512)?;
            }
            let pending = self.base % 8;
            let old = cache.pending(layer, sequence, false)?;
            let raw = [stage.raw[at].0.buffer, stage.raw[at].1.buffer];
            for ((target, previous), source) in raw.into_iter().zip([old.0, old.1]).zip([ik, iz]) {
                if pending > 0 {
                    copy(
                        &cache,
                        view(target, 0, pending * 512)?,
                        view(previous, 0, pending * 512)?,
                    )?;
                }
                copy(
                    &cache,
                    view(target, pending * 512, self.rows * 512)?,
                    view(source, 0, self.rows * 512)?,
                )?;
            }
            let targets = cache.csa_planes(layer)?;
            let mut position = self.base;
            while position < self.base + self.rows {
                let page = self.pages()[position / 64] as usize;
                let count = (64 - position % 64).min(self.base + self.rows - position);
                for (target, source) in [targets.0, targets.1].into_iter().zip([key, value]) {
                    copy(
                        &cache,
                        view(target, (page * 64 + position % 64) * row, count * row)?,
                        view(source, (position - self.base) * row, count * row)?,
                    )?;
                }
                position += count;
            }
            let completed = (pending + self.rows) / 8;
            if let Some(summary) = region.summaries.filter(|_| completed > 0) {
                ensure!(completed <= stage.regions, "Step-5 staged summary capacity");
                let target = cache.request_component(summary, sequence)?;
                let output = view(target, self.base / 8 * 256, completed * 256)?;
                let sparse = stage.sparse.as_ref().context("Step-5 CSA compressor owner")?;
                cache.stream.device.run(|| {
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
                                Scalar::I64((pending + self.rows) as i64),
                            ],
                            cache.stream.raw,
                        )
                    }
                })?;
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

    pub fn publish(mut self, accepted: usize) -> Result<()> {
        ensure!(
            !self.failed && accepted <= self.rows,
            "Step-5 failed step or invalid accepted count"
        );
        if accepted == 0 {
            return Ok(());
        }
        let end = self.base + accepted;
        for (rank, owner) in self.ranks.iter().enumerate() {
            let cache = owner.borrow();
            let state = &cache.layout().ranks[rank];
            ensure!(
                self.projected[rank].is_subset(&self.attended[rank]),
                "Step-5 cannot publish projected layers without attention"
            );
            ensure!(
                self.complete[rank].len()
                    == state.sliding.iter().filter(|s| !s.mtp).count() + state.regions.len(),
                "Step-5 cannot publish an incomplete target step"
            );
        }
        for (rank, owner) in self.ranks.iter().enumerate() {
            let cache = owner.borrow();
            let mut guard = cache
                .stream
                .cancellation_guard("Step-5 prepare publication");
            guard.complete = self.submission.get();
            let sequence = self.placement.sequence;
            let raw_offset = (self.base % 8 + accepted) / 8 * 8;
            let tail = end % 8;
            for (at, region) in cache.layout().ranks[rank].regions.iter().enumerate() {
                let targets = cache.pending(region.layer, sequence, true)?;
                for (target, source) in [targets.0, targets.1].into_iter().zip([
                    self.staging[rank].raw[at].0.buffer,
                    self.staging[rank].raw[at].1.buffer,
                ]) {
                    zero(&cache, target)?;
                    if tail > 0 {
                        copy(
                            &cache,
                            view(target, 0, tail * 512)?,
                            view(source, raw_offset * 512, tail * 512)?,
                        )?;
                    }
                }
            }
            let target = cache.length(sequence, true)?;
            cache.stream.device.run(|| {
                // SAFETY: private length bank on this stream; source is retained by the library.
                unsafe {
                    cache.stream.device.library.copy_h2d_ordered(
                        target,
                        &(end as u64).to_le_bytes(),
                        cache.stream.raw,
                    )
                }
            })?;
            guard.complete = true;
        }
        for owner in &self.ranks {
            owner.borrow().stream.drain()?;
        }
        let family = Step5Prefix::new(self.ranks.clone())?;
        // Release can fail only before the selector changes. Both leases are
        // retained until all device work is known complete.
        self.pool.release(&family, &self.placement.pages)?;
        for owner in &self.ranks {
            owner.borrow().publish_bank(self.placement.sequence);
        }
        self.placement.pages = self.pages.take().expect("unpublished Step-5 pages");
        self.placement.committed = end;
        Ok(())
    }
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
        if self.pages.is_some() {
            for owner in &self.ranks {
                owner
                    .borrow()
                    .stream
                    .drain_or_abort("Step-5 abandoned proposal");
            }
        }
        for (owner, stage) in self.ranks.iter().zip(&self.staging) {
            owner.borrow().busy[self.placement.sequence as usize].set(false);
            stage.busy.set(false);
        }
        let Some(pages) = self.pages.take() else {
            return;
        };
        if let Ok(family) = Step5Prefix::new(self.ranks.clone()) {
            if let Err(error) = self.pool.release(&family, &pages) {
                for owner in &self.ranks {
                    owner.borrow().poisoned.set(true);
                }
                tracing::error!(%error, "Step-5 aborted page lease quarantined");
            }
        }
    }
}

fn copy(
    cache: &CacheRank<'_>,
    target: CuteafdDeviceBuffer,
    source: CuteafdDeviceBuffer,
) -> Result<()> {
    cache.stream.device.run(|| {
        // SAFETY: checked disjoint views and caller-owned inputs remain live through the step drain.
        unsafe {
            cache.stream.device.library.copy_d2d_async(
                target,
                source,
                source.bytes,
                cache.stream.raw,
            )
        }
    })
}
fn zero(cache: &CacheRank<'_>, target: CuteafdDeviceBuffer) -> Result<()> {
    cache.stream.device.run(|| {
        // SAFETY: private pending bank is not read by the committed frontier.
        unsafe {
            cache.stream.device.library.cuda_zero_bytes_async(
                target,
                target.bytes,
                cache.stream.raw,
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
        key: &'e Allocation<'a>,
        value: &'e Allocation<'a>,
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
        let mut pool = PrefixCache::<StubCopyEngine, Frontier>::new(
            family.layout(),
            PrefixConfig {
                entries: 0,
                mark_slots: 3,
                keep_logits: false,
                min_tokens: 1,
            },
            None,
        )?;
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
        for base in [0, 7, 63, 511, 575, 576] {
            let mut placement = pool
                .admit_cold(&family, base.max(1), base.max(1), |pages| Placement {
                    pages,
                    sequence: 0,
                    committed: base,
                })?
                .placement;
            let old_pages = placement.pages.clone();
            let old_free = pool.pool().free();
            // Seed arbitrary old visible bytes, including pending and summary state.
            for owner in &owners {
                let cache = owner.borrow();
                for span in layout.snapshot_spans(cache.rank(), base as u64)? {
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
                let mut step = Step::begin(refs.clone(), lanes.clone(), &mut pool, &mut placement, 64)?;
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
            assert_eq!(pool.pool().free(), old_free);
            let compose_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                let mut step = Step::begin(refs.clone(), lanes.clone(), &mut pool, &mut placement, 64)?;
                step.compose(0, &key, |_, _| -> Result<()> {
                    panic!("injected composition panic");
                })
            }));
            assert!(compose_panic.is_err());
            assert_eq!(placement.pages, old_pages);
            assert_eq!(placement.committed, base);
            assert_eq!(pool.pool().free(), old_free);
            let killed =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                    let mut step =
                        Step::begin(refs.clone(), lanes.clone(), &mut pool, &mut placement, 64)?;
                    step.stage(0, 0, &key, &value, None)?;
                    step.stage(0, 3, &key, &value, Some((&ik, &iz)))?;
                    assert!(owners[0].borrow_mut().restore(0, 0, base as u64).is_err());
                    panic!("injected mid-target cancellation");
                }));
            assert!(killed.is_err());
            assert_eq!(placement.committed, base);
            assert_eq!(placement.pages, old_pages);
            assert_eq!(pool.pool().free(), old_free);
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
            let mut step = Step::begin(refs.clone(), lanes.clone(), &mut pool, &mut placement, 64)?;
            step.stage(0, 0, &key, &value, None)?;
            assert!(step.publish(1).is_err());
            assert_eq!(placement.committed, base);
            for accepted in [0, 1, 7, 8, 63, 64] {
                let before = placement.committed;
                let mut step =
                    Step::begin(refs.clone(), lanes.clone(), &mut pool, &mut placement, 64)?;
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
            pool.release(&family, &placement.pages)?;
            assert_eq!(pool.pool().free(), 40);
        }
        Ok(())
    }
}
