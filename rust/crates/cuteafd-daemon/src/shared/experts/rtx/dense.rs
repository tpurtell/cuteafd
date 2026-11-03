//! Coordinator dense operands as push-group ranks, for any family.
//!
//! A loader in push mode allocates every operand exactly as the mount does and
//! records how its bytes arrive instead of reading them:
//! - an [`Operand`] (class 2) is checkpoint rectangles landing unchanged in its
//!   allocation;
//! - a [`Step`] (class 3) is rectangles staged in pinned host memory, then GPU
//!   [`Job`]s (FP8 quantization, BF16 widening, MLA absorption, checked copies)
//!   that write their operands on the step's GPU. Steps hold at most
//!   [`STEP_BYTES`] (whole scale units), so staging stays bounded.
//!
//! The mount runs the same records: rectangles read into the operand, or into a
//! pinned buffer the same jobs read. A rank's logical space is its operands in
//! `[0, base)` (device) and its steps in `[base, base + host_bytes)`, delivered
//! through the consumer's host path; a step's jobs run once its bytes are
//! complete, and its staging returns to a small pool.
#[path = "dense/records.rs"]
mod records;
pub(crate) use records::*;
use crate::shared::memory::{device::Device, push_consumer::PushDestination,
    weight_arena::{Coverage, ExpectedCoverage}, HostAllocation};
use crate::shared::fp8_load_quant::LoadKernels;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::push_load::builder::{Builder, CheckedPlan, Destination, Source};
use std::ffi::c_void;

/// `rect`'s rows as plan entries landing at `offset` of rank `rank`.
pub(crate) fn append(rect: &Rect, builder: &mut Builder, rank: u32, offset: u64) -> Result<()> {
    let source = Source { path: rect.path.clone(), offset: rect.base,
        bytes: (rect.rows.end * rect.row_bytes) as u64 };
    builder.rows(&source, (rect.rows.start * rect.row_bytes + rect.cols.start) as u64,
        u32::try_from(rect.rows.len())?, u32::try_from(rect.width())?, rect.row_bytes as u64, rect.width() as u64,
        Destination { rank, offset, bytes: rect.bytes() as u64 })?;
    Ok(())
}


/// Independent pinned owners retained by one DenseArena through Done/drop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostStaging {
    pub staging_bytes: usize,
    pub largest_step: usize,
    pub table_bytes: usize,
}

impl HostStaging {
    /// Bound retained transforms from the receiver ring, not the source's send
    /// concurrency. A missing slot retains its payload plus at most two boundary
    /// steps. Each file may additionally hold one step at its stream frontier.
    /// Mixed-file or overlapping source intervals cannot use this argument;
    /// charge their entire footprint separately.
    pub fn for_steps<'s>(steps: impl IntoIterator<Item = &'s [Rect]>, table_bytes: usize) -> Result<Self> {
        use std::collections::{BTreeMap, BTreeSet};
        let mut files = BTreeSet::new();
        let mut extents = BTreeMap::<std::path::PathBuf, Vec<(u64, u64, usize)>>::new();
        let (mut total, mut fallback, mut largest) = (0usize, 0usize, 0usize);
        for rects in steps {
            let bytes = rects.iter().try_fold(0usize, |sum, rect| sum.checked_add(rect.bytes())
                .context("dense step size overflow"))?;
            total = total.checked_add(bytes).context("dense staging footprint overflow")?;
            largest = largest.max(bytes);
            let mut contiguous = !rects.is_empty();
            let mut end = 0;
            for (i, rect) in rects.iter().enumerate() {
                ensure!(!rect.rows.is_empty() && !rect.cols.is_empty() && rect.cols.end <= rect.row_bytes,
                    "invalid staged rectangle");
                let first = rect.base.checked_add((rect.rows.start * rect.row_bytes + rect.cols.start) as u64)
                    .context("dense source start overflow")?;
                if i > 0 && (rect.path != rects[0].path || first != end) { contiguous = false; }
                end = rect.base.checked_add(((rect.rows.end - 1) * rect.row_bytes + rect.cols.end) as u64)
                    .context("dense source end overflow")?;
            }
            if contiguous {
                let rect = &rects[0];
                files.insert(rect.path.clone());
                let first = rect.base + (rect.rows.start * rect.row_bytes + rect.cols.start) as u64;
                extents.entry(rect.path.clone()).or_default().push((first, end, bytes));
            } else {
                fallback = fallback.checked_add(bytes).context("dense fallback staging overflow")?;
            }
        }
        // Any overlapping intervals are conservatively charged in full. Such
        // steps may remain open while all later source rows are delivered.
        for entries in extents.values_mut() {
            entries.sort_unstable_by_key(|entry| entry.0);
            let (mut end, mut group_bytes, mut count) = (0, 0usize, 0usize);
            for &(first, last, bytes) in entries.iter() {
                if first >= end {
                    if count > 1 { fallback = fallback.checked_add(group_bytes).context("dense overlap staging overflow")?; }
                    group_bytes = 0;
                    count = 0;
                }
                group_bytes = group_bytes.checked_add(bytes).context("dense overlap footprint overflow")?;
                count += 1;
                end = end.max(last);
            }
            if count > 1 { fallback = fallback.checked_add(group_bytes).context("dense overlap staging overflow")?; }
        }
        let window = cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES.checked_add(largest.checked_mul(2)
            .context("dense boundary staging overflow")?).and_then(|bytes|
                bytes.checked_mul(cuteafd_ffi::push_ring::DEFAULT_SLOTS)).context("dense ring staging overflow")?;
        let frontiers = files.len().checked_mul(largest).context("dense frontier staging overflow")?;
        let staging_bytes = fallback.checked_add(window).and_then(|bytes| bytes.checked_add(frontiers))
            .context("dense pinned staging overflow")?.min(total);
        let bound = Self { staging_bytes, largest_step: largest, table_bytes };
        bound.bytes()?;
        Ok(bound)
    }
    pub fn bytes(self) -> Result<usize> {
        self.staging_bytes.checked_add(self.table_bytes).context("dense pinned staging overflow")
    }
}

/// One GPU's operands and steps in its rank's logical space.
pub(crate) struct RankLayout {
    pub device: i32,
    /// (logical start, operand), ascending, device space `[0, base)`.
    pub operands: Vec<(u64, Operand)>,
    /// (start relative to `base`, step), ascending.
    pub steps: Vec<(u64, Step)>,
    pub base: u64,
    pub host_bytes: u64,
    /// Device-to-device transforms of this GPU's landed operands, run at Done.
    pub finishes: Vec<Finish>,
    pub host_staging: HostStaging,
}

impl RankLayout {
    /// `device`'s records in loader order: 256-byte aligned with a gap byte
    /// between neighbours, so no slot fragment spans two of them.
    pub fn new(operands: Vec<Operand>, steps: Vec<Step>, device: i32) -> Result<Self> {
        let mut operands = operands;
        let mut transformed = Vec::new();
        for step in steps.into_iter().filter(|s| s.device == device) {
            if step.jobs.iter().all(|job| matches!(job, Job::Upload { .. })) && !step.jobs.is_empty() {
                for job in &step.jobs {
                    let Job::Upload { at, bytes, dst } = *job else { unreachable!() };
                    ensure!(bytes <= dst.bytes && at.checked_add(bytes).is_some_and(|end| end <= step.bytes()),
                        "direct upload outside its step");
                    let (mut source_at, mut lands) = (0usize, Vec::new());
                    for rect in &step.rects {
                        let (lo, hi) = (at.max(source_at), (at + bytes).min(source_at + rect.bytes()));
                        if lo < hi {
                            let (lo, hi) = (lo - source_at, hi - source_at);
                            if lo % rect.width() == 0 && hi % rect.width() == 0 {
                                lands.push((rect.sub(lo / rect.width()..hi / rect.width()), source_at + lo - at));
                                source_at += rect.bytes();
                                continue;
                            }
                            let mut cursor = lo;
                            while cursor < hi {
                                let row = cursor / rect.width();
                                let column = cursor % rect.width();
                                let end = hi.min((row + 1) * rect.width());
                                let piece = Rect { cols: rect.cols.start + column..rect.cols.start + column + end - cursor,
                                    ..rect.sub(row..row + 1) };
                                lands.push((piece, source_at + cursor - at));
                                cursor = end;
                            }
                        }
                        source_at += rect.bytes();
                    }
                    operands.push(Operand { device, buffer: dst, lands, bytes });
                }
            } else { transformed.push(step); }
        }
        let mut placed = Vec::new();
        let mut cursor = 0u64;
        for operand in operands.into_iter().filter(|o| o.device == device) {
            ensure!(operand.bytes > 0, "empty pushed operand");
            let start = cursor.next_multiple_of(256);
            cursor = start + operand.bytes as u64 + 1;
            placed.push((start, operand));
        }
        let base = cursor.next_multiple_of(256);
        let mut staged = Vec::new();
        let mut cursor = 0u64;
        for step in transformed {
            let start = cursor.next_multiple_of(256);
            cursor = start + step.bytes() as u64 + 1;
            staged.push((start, step));
        }
        let mut table_bytes = 0;
        for (_, step) in &staged {
            ensure!(step.bytes() > 0 && step.bytes() <= STEP_BYTES, "dense step exceeds bounded staging");
            for job in &step.jobs {
                let bytes = match job {
                    Job::Quant { rows, layout, .. } =>
                        crate::shared::fp8_load_quant::quant_table_bytes(rows.len(), *layout)?,
                    Job::ScaleCodes { .. } => crate::shared::fp8_load_quant::scale_code_table_bytes(1)?,
                    _ => 0,
                };
                table_bytes = table_bytes.max(bytes);
            }
        }
        let host_staging = HostStaging::for_steps(staged.iter().map(|(_, step)| step.rects.as_slice()), table_bytes)?;
        host_staging.bytes()?;
        Ok(Self { device, operands: placed, steps: staged, base, host_bytes: if cursor == 0 { 0 } else { cursor },
            finishes: Vec::new(), host_staging })
    }

    /// [`Self::new`] with `device`'s finishes from `finishes`.
    pub fn with_finishes(mut self, finishes: &[Finish]) -> Self {
        self.finishes = finishes.iter().filter(|f| f.device() == self.device).cloned().collect();
        self
    }

    /// The rank's entries as a device-free plan: operands from 0, steps from
    /// the `host_base` the group driver passes (`base` when alone).
    pub fn plan(&self) -> super::push::DensePlan {
        let operands: Vec<(u64, Vec<(Rect, usize)>)> = self.operands.iter()
            .map(|(start, o)| (*start, o.lands.clone())).collect();
        let steps: Vec<(u64, Vec<Rect>)> = self.steps.iter().map(|(start, s)| (*start, s.rects.clone())).collect();
        super::push::DensePlan { base: self.base, host_bytes: self.host_bytes, host_staging: self.host_staging, peer_copies: Vec::new(),
            append: Box::new(move |builder, rank, host_base| {
                for (start, lands) in &operands {
                    for (rect, at) in lands { append(rect, builder, rank, start + *at as u64)?; }
                }
                for (start, rects) in &steps {
                    let mut at = host_base + start;
                    for rect in rects { append(rect, builder, rank, at)?; at += rect.bytes() as u64; }
                }
                Ok(())
            }) }
    }

    /// This rank's checked entries alone, in a group of `world` ranks.
    pub fn checked(&self, root: &std::path::Path, world: usize, rank: usize) -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        *arenas.get_mut(rank).context("coordinator rank outside group")? = self.base + self.host_bytes;
        let mut builder = Builder::new(root, arenas);
        (self.plan().append)(&mut builder, u32::try_from(rank)?, self.base)?;
        Ok(builder.finish()?)
    }
}

/// A rank's operands as a push destination: device fragments land in the
/// operand allocations, host fragments fill their step's staging (taken from
/// a pool on its first fragment); a complete step runs its jobs at once and
/// returns its staging. Done requires exactly the planned footprint and every
/// step run; nothing reads these operands before commit.
pub(crate) struct DenseArena<'a> {
    device: Device<'a>,
    layout: RankLayout,
    expected: ExpectedCoverage,
    coverage: Coverage,
    kernels: LoadKernels<'a>,
    stream: *mut c_void,
    /// Open steps: index -> (staging, delivered bytes).
    open: std::collections::HashMap<usize, (HostAllocation<'a>, u64)>,
    pool: Vec<HostAllocation<'a>>,
    scratch: Option<crate::shared::memory::DeviceAllocation<'a>>,
    run: usize,
    /// Bytes of staging held by open steps now, and their peak.
    staged: u64,
    peak: u64,
    /// Bytes of staging allocated (open steps and the pool), and their peak.
    held: u64,
    peak_held: u64,
}

impl<'a> DenseArena<'a> {
    pub fn new(device: Device<'a>, layout: RankLayout, expected: ExpectedCoverage) -> Result<Self> {
        ensure!(layout.device == device.id, "operand layout belongs to another GPU");
        let stream = device.run(|| device.library.cuda_stream_create())?;
        let kernels = match LoadKernels::new(device.library) {
            Ok(kernels) => kernels,
            Err(error) => {
                // SAFETY: the stream was just created and has no work.
                device.run(|| unsafe { device.library.cuda_stream_destroy(stream) })?;
                return Err(error);
            }
        };
        Ok(Self { device, layout, expected, coverage: Coverage::default(), kernels, stream,
            open: Default::default(), pool: Vec::new(), scratch: None, run: 0, staged: 0, peak: 0, held: 0, peak_held: 0 })
    }

    fn operand(&self, offset: u64, bytes: u64) -> Result<(CuteafdDeviceBuffer, usize)> {
        let index = self.layout.operands.partition_point(|(start, _)| *start <= offset).checked_sub(1)
            .context("fragment before the first operand")?;
        let (start, operand) = &self.layout.operands[index];
        let at = usize::try_from(offset - start)?;
        ensure!(at.checked_add(usize::try_from(bytes)?).is_some_and(|end| end <= operand.bytes),
            "fragment crosses an operand boundary");
        Ok((operand.buffer, at))
    }

    fn step(&self, offset: u64, bytes: u64) -> Result<(usize, usize)> {
        let relative = offset.checked_sub(self.layout.base).context("host fragment below the steps")?;
        let index = self.layout.steps.partition_point(|(start, _)| *start <= relative).checked_sub(1)
            .context("fragment before the first step")?;
        let (start, step) = &self.layout.steps[index];
        let at = usize::try_from(relative - start)?;
        ensure!(at.checked_add(usize::try_from(bytes)?).is_some_and(|end| end <= step.bytes()),
            "fragment crosses a step boundary");
        Ok((index, at))
    }
}

impl Drop for DenseArena<'_> {
    fn drop(&mut self) {
        let library = self.device.library;
        let stream = self.stream;
        // SAFETY: every job drains its stream before returning; this owner's
        // stream has no other users.
        let _ = self.device.run(|| unsafe {
            library.cuda_stream_synchronize(stream)?;
            library.cuda_stream_destroy(stream)
        }).map_err(|error| tracing::error!(%error, "coordinator dense stream teardown"));
    }
}

impl PushDestination for DenseArena<'_> {
    type Delivered = ();
    fn belongs_to(&self, device: Device<'_>) -> bool {
        self.device.id == device.id && std::ptr::eq(self.device.library, device.library)
    }
    fn logical_bytes(&self) -> usize { (self.layout.base + self.layout.host_bytes) as usize }
    fn device_bytes(&self) -> usize { self.layout.base as usize }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(bytes > 0 && self.expected.contains(offset, bytes), "fragment outside checked footprint");
        if offset < self.layout.base { self.operand(offset, bytes).map(|_| ()) } else { self.step(offset, bytes).map(|_| ()) }
    }
    fn destination(&self, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
        let (buffer, at) = self.operand(offset as u64, bytes as u64)?;
        // SAFETY: `at + bytes` lies inside this live operand allocation (checked above).
        Ok(CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(at).cast() }, bytes, ..buffer })
    }
    fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let length = bytes.len() as u64;
        ensure!(self.expected.contains(offset, length), "staged fragment outside checked footprint");
        let (index, at) = self.step(offset, length)?;
        let need = self.layout.steps[index].1.bytes();
        if !self.open.contains_key(&index) {
            let staging = match self.pool.iter().position(|s| s.buffer.bytes == need) {
                Some(found) => self.pool.swap_remove(found),
                None => {
                    // Exact-size buffers keep tiny coded-scale steps cheap.
                    // Discard unused size classes before allocating; retained
                    // pools must never exceed the same admitted byte bound.
                    while self.held + need as u64 > self.layout.host_staging.staging_bytes as u64 {
                        let Some(unused) = self.pool.pop() else { break; };
                        self.held -= unused.buffer.bytes as u64;
                        drop(unused);
                    }
                    ensure!(self.held + need as u64 <= self.layout.host_staging.staging_bytes as u64,
                        "dense rank GPU {} needs {} pinned staging bytes, beyond admitted ring/step bound {}",
                        self.device.id, self.held + need as u64, self.layout.host_staging.staging_bytes);
                    let allocation = HostAllocation::named(self.device.library, need, "staging/coordinator-dense")?;
                    self.held += need as u64;
                    self.peak_held = self.peak_held.max(self.held);
                    allocation
                }
            };
            self.staged += staging.buffer.bytes as u64;
            self.peak = self.peak.max(self.staged);
            self.open.insert(index, (staging, 0));
        }
        let (staging, delivered) = self.open.get_mut(&index).context("open step")?;
        staging.bytes_mut()[at..at + bytes.len()].copy_from_slice(bytes);
        *delivered += length;
        self.expected.record(&mut self.coverage, offset, length)?;
        if *delivered == need as u64 {
            let (staging, _) = self.open.remove(&index).context("complete step")?;
            let (library, stream, step) = (self.device.library, self.stream, &self.layout.steps[index].1);
            let (kernels, scratch) = (&mut self.kernels, &mut self.scratch);
            self.device.run(|| step.run(kernels, library, &staging.bytes()[..need], stream, scratch))?;
            self.run += 1;
            self.staged -= staging.buffer.bytes as u64;
            self.pool.push(staging);
        }
        Ok(())
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(offset + bytes <= self.layout.base && self.expected.contains(offset, bytes),
            "completed fragment is outside checked plan footprint");
        self.expected.record(&mut self.coverage, offset, bytes)
    }
    fn done(self) -> Result<()> {
        ensure!(self.expected.complete(&self.coverage), "rank Done before all planned operand bytes completed");
        ensure!(self.open.is_empty() && self.run == self.layout.steps.len(), "rank Done before every staged step ran");
        let (library, stream) = (self.device.library, self.stream);
        let finished = std::time::Instant::now();
        self.device.run(|| {
            for finish in &self.layout.finishes { finish.run(library, stream)?; }
            // SAFETY: the finishes' buffers are this rank's live operand allocations.
            unsafe { library.cuda_stream_synchronize(stream) }
        })?;
        let finish_ms = finished.elapsed().as_secs_f64() * 1e3;
        tracing::info!(device = self.device.id, operands = self.layout.operands.len(), steps = self.run,
            finishes = self.layout.finishes.len(), finish_ms,
            peak_staged_mib = self.peak as f64 / (1u64 << 20) as f64,
            peak_pinned_mib = self.peak_held as f64 / (1u64 << 20) as f64, "coordinator dense rank delivered");
        #[cfg(test)]
        testing::PEAKS.with(|p| p.borrow_mut().push((self.peak, self.peak_held)));
        Ok(())
    }
}

/// Both of `devices`' (one or two GPUs) plans and destinations, as ranks
/// `sparks..`.
#[allow(clippy::type_complexity)]
pub(crate) fn plan_ranks<'a>(snapshot: &std::path::Path, operands: Vec<Operand>, steps: Vec<Step>,
    devices: &[Device<'a>], sparks: usize)
    -> Result<(Vec<super::push::DensePlan>, Vec<DenseArena<'a>>)> {
    plan_records(snapshot, Records { operands, steps, ..Default::default() }, devices, sparks)
}

/// [`plan_ranks`] for a whole record set (each GPU's finishes run at its Done).
#[allow(clippy::type_complexity)]
pub(crate) fn plan_records<'a>(snapshot: &std::path::Path, records: Records,
    devices: &[Device<'a>], sparks: usize)
    -> Result<(Vec<super::push::DensePlan>, Vec<DenseArena<'a>>)> {
    let Records { operands, steps, finishes, peer_copies } = records;
    ensure!(peer_copies.iter().all(|copy| devices.iter().any(|d| d.id == copy.source_device)
        && devices.iter().any(|d| d.id == copy.destination_device)), "peer load copy outside the push group");
    ensure!(finishes.iter().all(|f| devices.iter().any(|d| d.id == f.device())),
        "dense finish on a GPU outside the push group");
    let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
    let world = sparks + devices.len();
    let mut operands = operands;
    let mut steps = steps;
    let mut out = (Vec::new(), Vec::new());
    for (gpu, device) in devices.iter().enumerate() {
        let (mine, rest): (Vec<_>, Vec<_>) = operands.into_iter().partition(|o| o.device == device.id);
        operands = rest;
        let (my_steps, rest): (Vec<_>, Vec<_>) = steps.into_iter().partition(|s| s.device == device.id);
        steps = rest;
        let layout = RankLayout::new(mine, my_steps, device.id)?.with_finishes(&finishes);
        let checked = layout.checked(&root, world, sparks + gpu)?;
        ensure!(checked.plan.files.iter().all(|f| cuteafd_loader::push_load::is_push_shard(&root.join(f))),
            "coordinator dense source is not a SparkNest shard");
        let expected = ExpectedCoverage::from_plan(&checked, u32::try_from(sparks + gpu)?)?;
        let mut plan = layout.plan();
        plan.peer_copies = peer_copies.iter().filter(|copy| copy.destination_device == device.id).cloned().collect();
        out.0.push(plan);
        out.1.push(DenseArena::new(*device, layout, expected)?);
    }
    ensure!(operands.is_empty() && steps.is_empty(), "dense records on a GPU outside the push group");
    Ok(out)
}

#[cfg(test)]
pub(crate) mod testing {
    use anyhow::Result;

    thread_local! {
        /// (peak staged, peak pinned) bytes of each delivered dense rank on this thread.
        pub(crate) static PEAKS: std::cell::RefCell<Vec<(u64, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    use std::os::unix::fs::FileExt;

    /// A one-source fake SparkNest for one rank: packs the rank's planned runs
    /// into ring slots in plan order, Filled per slot (each written only after its
    /// previous use was released), then Done.
    pub(crate) struct FakeRank {
        ring: std::fs::File,
        slot_bytes: usize,
        slots: Vec<Vec<u8>>,
        next: std::cell::Cell<usize>,
        released: std::cell::RefCell<std::collections::BTreeSet<u32>>,
        capacity: usize,
    }

    impl FakeRank {
        pub(crate) fn new(plan: &cuteafd_loader::push_load::builder::CheckedPlan, mount: &std::path::Path, rank: u32,
            ring: &cuteafd_ffi::push_ring::RegisteredRing<'_>) -> Result<Self> {
            use cuteafd_loader::push_load::slot;
            let slot_bytes = ring.slot_bytes();
            let sources = plan.plan.files.iter().map(|f| std::fs::File::open(mount.join(f))).collect::<std::io::Result<Vec<_>>>()?;
            let (mut slots, mut packer) = (Vec::new(), slot::Packer::new(vec![0; slot_bytes]));
            for entry in plan.plan.entries.iter().filter(|e| e.rank == rank) {
                let pattern = &plan.plan.patterns[entry.pattern as usize];
                for (src, dst) in pattern.runs() {
                    let mut run = vec![0u8; pattern.run as usize];
                    sources[entry.file as usize].read_exact_at(&mut run, entry.src_base + src)?;
                    let mut bytes = &run[..];
                    let mut at = entry.dst_base + dst;
                    while !bytes.is_empty() {
                        let n = packer.push(at, bytes);
                        if n == 0 { slots.push(packer.finish().0); packer = slot::Packer::new(vec![0; slot_bytes]); continue; }
                        at += n as u64;
                        bytes = &bytes[n..];
                    }
                }
            }
            if !packer.is_empty() { slots.push(packer.finish().0); }
            use std::os::fd::AsFd;
            Ok(Self { ring: std::fs::File::from(ring.fd().as_fd().try_clone_to_owned()?), slot_bytes, slots,
                next: std::cell::Cell::new(0), released: Default::default(), capacity: ring.slots() })
        }
    }

    impl crate::shared::memory::push_consumer::RankEvents for FakeRank {
        fn next(&self) -> std::io::Result<cuteafd_loader::push_load::proto::Event> {
            use cuteafd_loader::push_load::proto::Event;
            let i = self.next.get();
            let Some(bytes) = self.slots.get(i) else { return Ok(Event::Done) };
            let slot = (i % self.capacity) as u32;
            if i >= self.capacity && !self.released.borrow_mut().remove(&slot) {
                return Err(std::io::Error::other("fake source reused an unreleased slot"));
            }
            self.ring.write_all_at(bytes, slot as u64 * self.slot_bytes as u64)?;
            self.next.set(i + 1);
            Ok(Event::Filled { slot, len: bytes.len() as u32 })
        }
        fn try_next(&self) -> std::io::Result<cuteafd_loader::push_load::Poll> {
            // Slots fill only after releases: report empty while every slot is in flight.
            let i = self.next.get();
            if i < self.slots.len() && i >= self.capacity && !self.released.borrow().contains(&((i % self.capacity) as u32)) {
                return Ok(cuteafd_loader::push_load::Poll::Empty);
            }
            self.next().map(cuteafd_loader::push_load::Poll::Ready)
        }
        fn release(&self, slot: u32) -> std::io::Result<()> {
            self.released.borrow_mut().insert(slot);
            Ok(())
        }
    }

    /// Replays `layout`'s checked plan for group rank `rank` (of `world`)
    /// through a ring into a [`super::DenseArena`] on `device`, reading the
    /// sources under `root`.
    pub(crate) fn replay(device: crate::shared::memory::device::Device<'_>, layout: super::RankLayout,
        root: &std::path::Path, world: usize, rank: usize) -> Result<()> {
        let checked = layout.checked(root, world, rank)?;
        let expected = crate::shared::memory::weight_arena::ExpectedCoverage::from_plan(&checked, rank as u32)?;
        device.run(|| -> Result<()> {
            let ring = cuteafd_ffi::push_ring::RegisteredRing::new(device.library, 8,
                cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES)?;
            let fake = FakeRank::new(&checked, root, rank as u32, &ring)?;
            let arena = super::DenseArena::new(device, layout, expected)?;
            crate::shared::memory::push_consumer::consume(device, arena, ring, fake, 8)
        })
    }
}
