//! One admitted rank allocation. Only completed, initialized weights expose
//! execution views; a loading arena exposes destinations solely to the copier.
use super::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::{collections::BTreeMap, ops::Range};

#[derive(Clone, Debug)]
pub(crate) struct ArenaLayout {
    regions: Vec<Range<usize>>,
    bytes: usize,
    validation_bytes: usize,
}

impl ArenaLayout {
    /// Keep each existing layer layout intact, including its own padding.
    pub fn new(sizes: impl IntoIterator<Item = usize>, admitted: usize) -> Result<Self> {
        let mut regions = Vec::new();
        let mut bytes = 0usize;
        for size in sizes {
            ensure!(size > 0, "empty rank weight region");
            bytes = bytes
                .checked_add(255)
                .context("rank arena alignment overflow")?
                & !255;
            let end = bytes
                .checked_add(size)
                .context("rank arena extent overflow")?;
            regions.push(bytes..end);
            bytes = end;
        }
        ensure!(
            !regions.is_empty() && bytes <= admitted,
            "rank weight arena needs {bytes} bytes, admitted {admitted}"
        );
        Ok(Self {
            regions,
            bytes,
            validation_bytes: 0,
        })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn with_validation(mut self, bytes: usize, admitted_host: usize) -> Result<Self> {
        ensure!(
            bytes <= admitted_host && self.bytes.checked_add(bytes).is_some(),
            "host validation tail exceeds admission"
        );
        self.validation_bytes = bytes;
        Ok(self)
    }
    pub fn logical_bytes(&self) -> usize {
        self.bytes + self.validation_bytes
    }
    pub fn region(&self, index: usize) -> Result<Range<usize>> {
        self.regions
            .get(index)
            .cloned()
            .context("rank arena region absent")
    }
}

#[cfg(test)]
#[path = "weight_arena_tests.rs"]
mod real_tests;

const MAX_COVERAGE_INTERVALS: usize = 1_000_000;

#[derive(Default)]
pub(crate) struct Coverage {
    intervals: BTreeMap<u64, u64>,
    bytes: u64,
    peak: usize,
}
impl Coverage {
    pub(crate) fn insert(&mut self, start: u64, bytes: u64) -> Result<()> {
        self.insert_bounded(start, bytes, MAX_COVERAGE_INTERVALS)
    }
    fn insert_bounded(&mut self, start: u64, bytes: u64, cap: usize) -> Result<()> {
        let mut end = start.checked_add(bytes).context("coverage end overflow")?;
        ensure!(bytes > 0, "empty delivery fragment");
        let mut start = start;
        if let Some((&left, &right)) = self.intervals.range(..=start).next_back() {
            if right >= start {
                start = left;
                end = end.max(right);
            }
        }
        while let Some((&left, &right)) = self.intervals.range(start..).next() {
            if left > end {
                break;
            }
            self.intervals.remove(&left);
            self.bytes -= right - left;
            end = end.max(right);
        }
        ensure!(
            self.intervals.len() < cap,
            "push coverage interval cap exceeded"
        );
        self.bytes = self
            .bytes
            .checked_add(end - start)
            .context("coverage total overflow")?;
        self.intervals.insert(start, end);
        self.peak = self.peak.max(self.intervals.len());
        Ok(())
    }
}

/// Exact destination footprint of a checked plan, excluding generated padding.
/// Dense dimensions collapse before enumeration, so packed tensors cost one node.
#[derive(Default)]
pub(crate) struct ExpectedCoverage(Coverage);
impl ExpectedCoverage {
    pub fn from_plan(
        plan: &cuteafd_loader::push_load::builder::CheckedPlan,
        rank: u32,
    ) -> Result<Self> {
        use cuteafd_loader::push_load::plan::Steps;
        let mut expected = Self::default();
        for entry in plan.plan.entries.iter().filter(|e| e.rank == rank) {
            let pattern = &plan.plan.patterns[entry.pattern as usize];
            let mut span = u64::from(pattern.run);
            let mut sparse = Vec::new();
            for dim in pattern.dims.iter().rev() {
                let count = match &dim.steps {
                    Steps::Count(n) => u64::from(*n),
                    Steps::Indices(v) => v.len() as u64,
                };
                if count == 1 {
                    continue;
                }
                if sparse.is_empty() && dim.dst_stride == span {
                    span = span
                        .checked_mul(count)
                        .context("expected footprint overflow")?;
                } else {
                    sparse.push((count, dim.dst_stride));
                }
            }
            fn insert(
                coverage: &mut Coverage,
                sparse: &[(u64, u64)],
                base: u64,
                span: u64,
            ) -> Result<()> {
                if let Some((&(count, stride), rest)) = sparse.split_first() {
                    for index in 0..count {
                        insert(
                            coverage,
                            rest,
                            base.checked_add(
                                index
                                    .checked_mul(stride)
                                    .context("expected stride overflow")?,
                            )
                            .context("expected base overflow")?,
                            span,
                        )?;
                    }
                } else {
                    coverage.insert(base, span)?;
                }
                Ok(())
            }
            insert(&mut expected.0, &sparse, entry.dst_base, span)?;
        }
        ensure!(
            Some(&expected.0.bytes) == plan.totals.rank_bytes.get(rank as usize),
            "checked plan destination total differs from footprint"
        );
        Ok(expected)
    }
    /// Delivered coverage equals this footprint exactly.
    pub(crate) fn complete(&self, delivered: &Coverage) -> bool {
        delivered.bytes == self.0.bytes && delivered.intervals == self.0.intervals
    }
    pub(crate) fn bytes(&self) -> u64 { self.0.bytes }
    pub(crate) fn contains(&self, start: u64, bytes: u64) -> bool {
        start.checked_add(bytes).is_some_and(|end| {
            self.0
                .intervals
                .range(..=start)
                .next_back()
                .is_some_and(|(_, &limit)| end <= limit)
        })
    }
}

pub(crate) struct LoadingArena<'a> {
    allocation: Allocation<'a>,
    layout: ArenaLayout,
    expected: ExpectedCoverage,
    coverage: Coverage,
    validation: Vec<u8>,
}

/// Post-Done owner: semantic checks and descriptor uploads may now run, but
/// execution still cannot borrow a published view until initialization succeeds.
pub(crate) struct DeliveredArena<'a> {
    allocation: Allocation<'a>,
    layout: ArenaLayout,
    validation: Vec<u8>,
    expected: ExpectedCoverage,
}

pub(crate) struct ReadyArena<'a> {
    allocation: Allocation<'a>,
    layout: ArenaLayout,
}

/// Non-owning, lifetime-bound layer region; it cannot outlive the rank owner.
pub(crate) struct WeightView<'s> {
    buffer: CuteafdDeviceBuffer,
    _owner: std::marker::PhantomData<&'s ReadyArena<'s>>,
}

fn slice(buffer: CuteafdDeviceBuffer, range: Range<usize>) -> Result<CuteafdDeviceBuffer> {
    ensure!(
        range.start <= range.end && range.end <= buffer.bytes,
        "weight view exceeds rank arena"
    );
    // SAFETY: the byte range was checked inside the owner's live allocation.
    let ptr = unsafe { buffer.ptr.cast::<u8>().add(range.start).cast() };
    Ok(CuteafdDeviceBuffer {
        ptr,
        bytes: range.len(),
        ..buffer
    })
}

impl<'a> LoadingArena<'a> {
    pub fn new(
        device: Device<'a>,
        layout: ArenaLayout,
        expected: ExpectedCoverage,
    ) -> Result<Self> {
        ensure!(
            expected
                .0
                .intervals
                .last_key_value()
                .is_none_or(|(_, &end)| end <= layout.logical_bytes() as u64),
            "expected footprint exceeds admitted arena"
        );
        let validation = vec![0; layout.validation_bytes];
        let allocation = Allocation::new(device, layout.bytes())?;
        Ok(Self {
            allocation,
            layout,
            expected,
            coverage: Coverage::default(),
            validation,
        })
    }

    pub fn belongs_to(&self, device: Device<'_>) -> bool {
        self.allocation.device.id == device.id
            && std::ptr::eq(self.allocation.device.library, device.library)
    }
    pub fn device_bytes(&self) -> usize { self.layout.bytes() }
    pub fn logical_bytes(&self) -> usize { self.layout.logical_bytes() }
    pub fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(bytes > 0 && self.expected.contains(offset, bytes), "fragment outside checked footprint");
        Ok(())
    }

    /// Copy-only destination. The ring consumer must retain this owner and
    /// drain its streams before dropping it on every failure path.
    pub fn destination(&self, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
        slice(
            self.allocation.buffer,
            offset..offset.checked_add(bytes).context("fragment end overflow")?,
        )
    }

    /// Host-tail fragments route directly from the registered ring. Device and
    /// host spans must be split at the logical prefix boundary by the consumer.
    pub fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        ensure!(
            offset >= self.layout.bytes as u64
                && self.expected.contains(offset, bytes.len() as u64),
            "validation fragment outside checked host tail"
        );
        let start = usize::try_from(offset - self.layout.bytes as u64)?;
        let end = start
            .checked_add(bytes.len())
            .context("validation fragment overflow")?;
        self.validation
            .get_mut(start..end)
            .context("validation fragment exceeds host tail")?
            .copy_from_slice(bytes);
        self.coverage.insert(offset, bytes.len() as u64)
    }

    /// Called only after a slot's CUDA copy event completes, before Release.
    pub fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= self.layout.bytes as u64),
            "completed fragment exceeds rank arena"
        );
        ensure!(
            self.expected.contains(offset, bytes),
            "completed fragment is outside checked plan footprint"
        );
        self.coverage.insert(offset, bytes)?;
        ensure!(
            self.coverage.bytes <= self.expected.0.bytes,
            "rank unique delivery exceeds checked plan"
        );
        Ok(())
    }

    /// Caller observed rank Done and drained every copy. Missing bytes cannot
    /// become execution-visible even when the control socket says Done.
    pub fn done(self) -> Result<DeliveredArena<'a>> {
        ensure!(
            self.coverage.bytes == self.expected.0.bytes
                && self.coverage.intervals == self.expected.0.intervals,
            "rank Done before all planned bytes completed"
        );
        Ok(DeliveredArena {
            allocation: self.allocation,
            layout: self.layout,
            validation: self.validation,
            expected: self.expected,
        })
    }
}

impl<'a> DeliveredArena<'a> {
    pub fn device(&self) -> Device<'a> { self.allocation.device }
    pub fn buffer(&self, range: Range<usize>) -> Result<CuteafdDeviceBuffer> {
        slice(self.allocation.buffer, range)
    }
    /// Only omitted bytes are generated. Delivered source bytes are never
    /// replaced by zero padding, including replayed or sparse tensor images.
    pub fn zero_padding(&self) -> Result<()> {
        let result = self.device().run(|| {
            let queued = (|| {
                // Fold adjacent equal-width/equal-pitch gaps in constant host
                // memory. A padded plane then needs one enqueue, not one per row.
                let mut pending: Option<(usize, usize, usize, usize)> = None;
                let flush = |(start, width, pitch, rows): (usize, usize, usize, usize)| -> Result<()> {
                    let end = start + (rows - 1) * pitch + width;
                    // SAFETY: only checked complement gaps are selected; this
                    // owner is retained until the unconditional drain below.
                    unsafe { self.device().library.zero_pitched_async(
                        self.buffer(start..end)?, pitch, width, rows, std::ptr::null_mut()) }
                };
                let mut gap = |start: usize, end: usize| -> Result<()> {
                    if start == end { return Ok(()); }
                    let width = end - start;
                    if let Some((base, previous_width, pitch, rows)) = pending {
                        if previous_width == width && (rows == 1 || Some(start - base) == rows.checked_mul(pitch)) {
                            pending = Some((base, width, if rows == 1 { start - base } else { pitch }, rows + 1));
                            return Ok(());
                        }
                        flush((base, previous_width, pitch, rows))?;
                    }
                    pending = Some((start, width, width, 1));
                    Ok(())
                };
                let mut cursor = 0usize;
                for (&start, &end) in &self.expected.0.intervals {
                    let start = usize::try_from(start)?.min(self.layout.bytes);
                    gap(cursor, start)?;
                    cursor = usize::try_from(end)?.min(self.layout.bytes);
                }
                gap(cursor, self.layout.bytes)?;
                if let Some(group) = pending { flush(group)?; }
                Ok(())
            })();
            // A failed enqueue can still leave earlier zero fills in flight.
            // SAFETY: the current device owns the default stream and arena.
            let drained = unsafe { self.device().library.cuda_stream_synchronize(std::ptr::null_mut()) };
            drained?;
            Ok(queued)
        });
        // Device binding/restoration can fail outside the drain closure too.
        if result.is_err() { self.device().library.quarantine_module_after_failed_drain(); }
        result?
    }
    pub fn validation(&self) -> &[u8] {
        &self.validation
    }
    pub fn region(&self, index: usize) -> Result<CuteafdDeviceBuffer> {
        slice(self.allocation.buffer, self.layout.region(index)?)
    }

    /// Preserve semantic validation and generated descriptors, after Done and
    /// before execution. An error consumes and discards the unpublished arena.
    pub fn initialize(
        self,
        initialize: impl FnOnce(&Self) -> Result<()>,
    ) -> Result<ReadyArena<'a>> {
        initialize(&self)?;
        Ok(ReadyArena {
            allocation: self.allocation,
            layout: self.layout,
        })
    }
}

impl ReadyArena<'_> {
    pub fn region(&self, index: usize) -> Result<WeightView<'_>> {
        Ok(WeightView {
            buffer: slice(self.allocation.buffer, self.layout.region(index)?)?,
            _owner: std::marker::PhantomData,
        })
    }
}

impl WeightView<'_> {
    pub fn buffer(&self) -> CuteafdDeviceBuffer {
        self.buffer
    }
    pub fn subregion(&self, offset: usize, bytes: usize) -> Result<WeightView<'_>> {
        Ok(WeightView {
            buffer: slice(
                self.buffer,
                offset
                    ..offset
                        .checked_add(bytes)
                        .context("weight subregion overflow")?,
            )?,
            _owner: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;

    fn expected(bytes: u64) -> Result<ExpectedCoverage> {
        let mut value = ExpectedCoverage::default();
        if bytes != 0 {
            value.0.insert(0, bytes)?;
        }
        Ok(value)
    }

    #[test]
    fn layout_admits_before_allocation_and_keeps_layer_offsets() -> Result<()> {
        let layout = ArenaLayout::new([257, 32], 544)?;
        assert_eq!(layout.region(0)?, 0..257);
        assert_eq!(layout.region(1)?, 512..544);
        assert!(ArenaLayout::new([257, 32], 543).is_err());
        assert!(ArenaLayout::new([usize::MAX, 1], usize::MAX).is_err());
        assert!(ArenaLayout::new([], 0).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn only_done_initialized_arena_publishes_stable_nonowning_views() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device {
            library: &library,
            id: 1,
        };
        fixture.configure_memory(&library, 1, 4096)?;
        let mut loading =
            LoadingArena::new(device, ArenaLayout::new([256, 256], 512)?, expected(16)?)?;
        let base = loading.destination(0, 16)?.ptr;
        assert!(loading.destination(510, 4).is_err());
        loading.copied(0, 8)?;
        loading.copied(8, 8)?;
        loading.copied(0, 16)?; // Replayed source file is idempotent.
        assert!(loading.copied(512, 1).is_err());
        let delivered = loading.done()?;
        let ready = delivered.initialize(|arena| {
            assert_eq!(arena.region(0)?.ptr, base);
            Ok(())
        })?;
        let view = ready.region(0)?;
        assert_eq!(view.buffer().ptr, base);
        assert_eq!(view.subregion(8, 8)?.buffer().bytes, 8);
        assert!(view.subregion(250, 8).is_err());
        assert_eq!(fixture.events()?.matches('D').count(), 1);
        assert_eq!(fixture.events()?.matches('d').count(), 0);
        drop(ready);
        assert_eq!(fixture.events()?.matches('d').count(), 1);
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn host_validation_tail_is_admitted_routed_and_checked_only_after_done() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let mut footprint = expected(4)?;
        footprint.0.insert(256, 4)?;
        let layout = ArenaLayout::new([256], 256)?.with_validation(4, 4)?;
        assert!(ArenaLayout::new([256], 256)?.with_validation(4, 3).is_err());
        let mut loading = LoadingArena::new(device, layout, footprint)?;
        assert!(loading.copied(16, 4).is_err()); // Equal-size padding is not planned payload.
        assert!(loading.copied(256, 4).is_err()); // No device copy to host tail.
        loading.copied(0, 4)?;
        loading.copy_validation(256, &0xcbac1fedu32.to_le_bytes())?;
        let delivered = loading.done()?;
        assert_eq!(delivered.validation(), &0xcbac1fedu32.to_le_bytes());
        let _ready = delivered.initialize(|value| {
            ensure!(
                u32::from_le_bytes(value.validation().try_into()?) == 0xcbac1fed,
                "bad multiplier"
            );
            Ok(())
        })?;
        assert_eq!(fixture.events()?.matches('D').count(), 1);
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn incomplete_delivery_or_failed_initialization_never_publishes() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let loading = LoadingArena::new(device, ArenaLayout::new([256], 256)?, expected(4)?)?;
        assert!(loading.done().is_err());
        let delivered =
            LoadingArena::new(device, ArenaLayout::new([256], 256)?, expected(0)?)?.done()?;
        assert!(delivered
            .initialize(|_| anyhow::bail!("semantic check failed"))
            .is_err());
        assert_eq!(fixture.events()?.matches('D').count(), 2);
        assert_eq!(fixture.events()?.matches('d').count(), 2);
        Ok(())
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    #[test]
    fn expected_footprint_excludes_padding_and_merges_dense_axes() -> Result<()> {
        use cuteafd_loader::push_load::{
            builder::CheckedPlan,
            plan::{Dim, Entry, Pattern, Plan, Steps},
        };
        let plan = Plan {
            files: vec!["weights".into()],
            patterns: vec![Pattern {
                run: 4,
                dims: vec![
                    Dim {
                        steps: Steps::Count(2),
                        src_stride: 64,
                        dst_stride: 16,
                    },
                    Dim {
                        steps: Steps::Indices(vec![0, 3]),
                        src_stride: 4,
                        dst_stride: 4,
                    },
                ],
            }],
            entries: vec![Entry {
                file: 0,
                pattern: 0,
                rank: 0,
                src_base: 0,
                dst_base: 0,
            }],
        };
        let totals = plan.check(&[128], &[32])?;
        let checked = CheckedPlan {
            plan,
            file_sizes: vec![128],
            arenas: vec![32],
            totals,
        };
        let expected = ExpectedCoverage::from_plan(&checked, 0)?;
        assert_eq!(expected.0.intervals, [(0, 8), (16, 24)].into());
        assert!(expected.contains(4, 4));
        assert!(!expected.contains(4, 8));
        assert!(!expected.contains(8, 4));
        assert!(ExpectedCoverage::from_plan(&checked, 2).is_err());
        Ok(())
    }
    #[test]
    fn replay_merges_partial_overlaps_and_cap_fails_closed() -> Result<()> {
        let mut coverage = Coverage::default();
        coverage.insert_bounded(4, 4, 2)?;
        coverage.insert_bounded(16, 4, 2)?;
        assert!(coverage.insert_bounded(32, 4, 2).is_err());
        coverage.insert_bounded(0, 18, 2)?;
        coverage.insert_bounded(0, 20, 2)?;
        assert_eq!(coverage.intervals, [(0, 20)].into());
        assert_eq!(coverage.bytes, 20);
        assert_eq!(coverage.peak, 2);
        Ok(())
    }
}
