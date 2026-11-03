//! EXLR8 unit arena ownership, shared by Spark and coordinator K1 execution.
use super::layer::{EXPERT_READ_LANES, ExpertLoadBudget};
use super::rtx::dense::{self, DenseLoad, Operand, Rect};
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_loader::plan::exlr8::residency::Residency;

mod execution;
pub(crate) mod dense_execution;
pub(crate) mod dense_weights;
mod shared;
pub(crate) use execution::{grouped_decode_from_row, Exlr8Execution};
pub(crate) use shared::{NativeExperts, NativePlan};

pub(crate) use cuteafd_loader::plan::exlr8::PrefillActivations as Exlr8PrefillActivations;

const BANKS: usize = 2;

pub(crate) struct Exlr8Weights<'a> {
    buffer: CuteafdDeviceBuffer,
    _arena: ArenaOwner<'a>,
    pub(crate) layout: Residency,
    pub(crate) budget: ExpertLoadBudget,
}

enum ArenaOwner<'a> {
    Mount { _allocation: DeviceAllocation<'a> },
    Push { _rank: std::rc::Rc<crate::shared::memory::weight_arena::ReadyArena<'a>>, region: usize },
}

pub(crate) fn view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(
        !buffer.ptr.is_null() && offset.checked_add(bytes).is_some_and(|n| n <= buffer.bytes),
        "EXLR8 arena view outside allocation"
    );
    // SAFETY: checked byte extent is inside the live arena owned by the caller.
    Ok(CuteafdDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
        bytes,
        ..buffer
    })
}

/// Records discontiguous payload ranges into the existing native allocation.
/// Descriptor/sign bytes are initialized by the owner, never overwritten by delivery.
fn record_payload(layout: &Residency, buffer: CuteafdDeviceBuffer,
    mut publish: impl FnMut(Operand) -> Result<()>) -> Result<()> {
    let mut blocks = std::collections::BTreeSet::new();
    for unit in &layout.loads {
        for (component, (source, &destination)) in unit.source.iter().zip(&unit.destination).enumerate() {
            if component == 2 && !blocks.insert(destination) { continue; }
            ensure!(source.bytes > 0, "empty EXLR8 payload component");
            source.offset.checked_add(source.bytes as u64).context("EXLR8 source extent overflow")?;
            let buffer = view(buffer, destination, source.bytes)?;
            let rect = Rect { name: format!("exlr8/{}/{}/{}/{component}", layout.layer, unit.expert, unit.unit),
                path: source.path.clone(), base: source.offset, row_bytes: source.bytes,
                rows: 0..1, cols: 0..source.bytes };
            publish(Operand { device: buffer.device_id, buffer, lands: vec![(rect, 0)], bytes: source.bytes })?;
        }
    }
    Ok(())
}

impl<'a> Exlr8Weights<'a> {
    #[cfg(test)]
    pub(crate) fn image(&self) -> CuteafdDeviceBuffer { self.buffer }

    pub(crate) fn from_push(
        rank: std::rc::Rc<crate::shared::memory::weight_arena::ReadyArena<'a>>,
        region: usize, layout: Residency, mut budget: ExpertLoadBudget,
    ) -> Result<Self> {
        let buffer = rank.region(region)?.buffer();
        ensure!(buffer.bytes == layout.resident_bytes && buffer.bytes == budget.resident_bytes,
            "pushed EXLR8 region differs from admission");
        budget.device_staging_bytes = 0;
        budget.pinned_host_bytes = 0;
        budget.read_scratch_bytes = layout.host_bytes();
        Ok(Self { buffer, _arena: ArenaOwner::Push { _rank: rank, region }, layout, budget })
    }

    /// Enumerate retained backing and direct views, never loading budgets.
    pub(crate) fn weight_receipts(
        states: &[Self], prefix: &str,
        location: cuteafd_loader::placement::planner::Location, physical_device: i32,
        lifetime: cuteafd_loader::placement::planner::Lifetime,
    ) -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::{Allocation, Domain, Receipt, View};
        use std::collections::BTreeSet;
        ensure!(!prefix.is_empty() && !states.is_empty() && location.domain == Domain::Device,
            "empty or invalid EXLR8 weight census");
        let receipt = |id: String, buffer: CuteafdDeviceBuffer| -> Result<Receipt> {
            ensure!(buffer.device_id == physical_device && !buffer.ptr.is_null() && buffer.bytes > 0,
                "EXLR8 retained buffer has wrong physical device or empty address/extent");
            let mut allocation = Allocation::buffer(id, location, buffer.bytes as u64, lifetime,
                Category::Experts, "EXLR8 expert weights");
            allocation.alignment = 256;
            Ok(Receipt { allocation, address: buffer.ptr as u64 })
        };
        let direct = |id: String, buffer: CuteafdDeviceBuffer, backing: &str,
                      owner: CuteafdDeviceBuffer| -> Result<Receipt> {
            let start = buffer.ptr as usize;
            ensure!(start >= owner.ptr as usize && start.checked_add(buffer.bytes).is_some_and(|end|
                (owner.ptr as usize).checked_add(owner.bytes).is_some_and(|limit| end <= limit)),
                "EXLR8 retained view exceeds physical owner");
            let mut record = receipt(id, buffer)?;
            record.allocation.view = Some(View { backing: backing.into(),
                offset: (start - owner.ptr as usize) as u64 });
            Ok(record)
        };
        let pushed = match &states[0]._arena { ArenaOwner::Push { _rank, .. } => Some(_rank), _ => None };
        let mut out = Vec::new();
        let mut identities = BTreeSet::new();
        let mut regions = BTreeSet::new();
        if let Some(rank) = pushed {
            out.push(receipt(format!("{prefix}.arena.rank"), rank.physical_buffer())?);
        }
        for state in states {
            let name = format!("target.{}.{}", state.layout.layer,
                if state.layout.set == cuteafd_loader::plan::exlr8::residency::ExpertSet::Routed {
                    "routed"
                } else { "shared" });
            ensure!(identities.insert(name.clone()), "duplicate retained EXLR8 layer/set");
            let (backing, owner, region) = match (&state._arena, pushed) {
                (ArenaOwner::Mount { _allocation }, None) => {
                    let backing = format!("{prefix}.arena.{name}");
                    out.push(receipt(backing.clone(), _allocation.buffer)?);
                    (backing, _allocation.buffer, _allocation.buffer)
                }
                (ArenaOwner::Push { _rank, region }, Some(rank)) => {
                    ensure!(std::rc::Rc::ptr_eq(_rank, rank) && regions.insert(*region),
                        "EXLR8 census mixes physical ranks or repeats a rank region");
                    (format!("{prefix}.arena.rank"), rank.physical_buffer(), rank.region(*region)?.buffer())
                }
                _ => anyhow::bail!("EXLR8 census mixes physical arena topologies"),
            };
            ensure!(state.buffer.ptr == region.ptr && state.buffer.bytes == region.bytes
                && state.buffer.device_id == region.device_id, "EXLR8 retained region differs from backing");
            out.push(direct(format!("{prefix}.layer.{name}.region"), state.buffer, &backing, owner)?);
            let mut add = |suffix: String, buffer| -> Result<()> {
                out.push(direct(format!("{prefix}.layer.{name}.{suffix}"), buffer, &backing, owner)?);
                Ok(())
            };
            add("signs.in".into(), state.signs(false)?)?;
            add("signs.out".into(), state.signs(true)?)?;
            add("pointers".into(), state.pointers()?)?;
            add("bits".into(), state.bits()?)?;
            if state.layout.set == cuteafd_loader::plan::exlr8::residency::ExpertSet::Routed {
                for (suffix, buffer) in ["compact.counts", "compact.units"].into_iter().zip(state.compact()?) {
                    add(suffix.into(), buffer)?;
                }
            }
            for unit in &state.layout.loads {
                for (index, label) in ["payload", "channel", "block"].into_iter().enumerate() {
                    add(format!("unit.{}.{}.{}.{label}", unit.expert, unit.unit, unit.projection),
                        view(state.buffer, unit.destination[index], unit.source[index].bytes)?)?;
                }
            }
        }
        if let Some(rank) = pushed {
            ensure!(regions.len() == rank.region_count(), "EXLR8 census omits a retained rank region");
        }
        Ok(out)
    }

    pub(crate) fn plan(layout: &Residency) -> Result<ExpertLoadBudget> {
        let descriptor_bytes = layout.descriptor_host_bytes()?;
        let descriptor_upload = layout.experts.checked_mul(layout.units)
            .and_then(|n| n.checked_mul(3 * 3 * 8))
            .context("EXLR8 descriptor upload overflow")?;
        let bank_bytes = layout
            .staging_bytes
            .max(16)
            .checked_mul(EXPERT_READ_LANES * BANKS)
            .and_then(|bytes| bytes.checked_add(descriptor_upload.max(layout.hidden)))
            .context("EXLR8 pinned staging overflow")?;
        Ok(ExpertLoadBudget {
            resident_bytes: layout.resident_bytes,
            device_staging_bytes: 0,
            pinned_host_bytes: bank_bytes,
            read_scratch_bytes: descriptor_bytes.checked_add(layout.host_bytes())
                .context("EXLR8 host residency overflow")?,
        })
    }

    pub(crate) fn pointers(&self) -> Result<CuteafdDeviceBuffer> {
        view(
            self.buffer,
            self.layout.pointers_offset,
            self.layout.experts * self.layout.units * 3 * 3 * 8,
        )
    }
    pub(crate) fn bits(&self) -> Result<CuteafdDeviceBuffer> {
        view(
            self.buffer,
            self.layout.bits_offset,
            self.layout.experts * self.layout.units * 3 * 4,
        )
    }
    pub(crate) fn compact(&self) -> Result<[CuteafdDeviceBuffer; 2]> {
        ensure!(self.layout.set == cuteafd_loader::plan::exlr8::residency::ExpertSet::Routed,
            "compact descriptors require routed weights");
        Ok([view(self.buffer, self.layout.compact_offsets[0], self.layout.experts * 4)?,
            view(self.buffer, self.layout.compact_offsets[1], self.layout.experts * self.layout.units * 4)?])
    }
    pub(crate) fn signs(&self, output: bool) -> Result<CuteafdDeviceBuffer> {
        view(
            self.buffer,
            self.layout.signs_offsets[usize::from(output)],
            self.layout.hidden,
        )
    }

    /// `available` excludes all caller-owned workspace/graph reservations. Admit
    /// pinned bytes against host/UMA headroom before calling this device loader.
    pub(crate) fn load(
        library: &'a NativeLibrary,
        layout: Residency,
        available: usize,
    ) -> Result<Self> {
        let budget = Self::plan(&layout)?;
        layout.check_budget(available, &[])?;
        let arena =
            DeviceAllocation::named(library, layout.resident_bytes, "experts/weights/exlr8")?;
        let (pointers, bits) = layout.descriptors(arena.buffer.ptr as u64)?;
        // Synchronous uploads finish before the bounded pinned banks are reused.
        for (offset, bytes) in layout.signs_offsets.iter().zip(&layout.signs) {
            library.copy_h2d(view(arena.buffer, *offset, bytes.len())?, bytes)?;
        }
        let pointers: Vec<u8> = pointers.iter().flat_map(|v| v.to_le_bytes()).collect();
        library.copy_h2d(
            view(arena.buffer, layout.pointers_offset, pointers.len())?,
            &pointers,
        )?;
        let bits: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        library.copy_h2d(view(arena.buffer, layout.bits_offset, bits.len())?, &bits)?;
        if layout.set == cuteafd_loader::plan::exlr8::residency::ExpertSet::Routed {
            let (counts, mapping, _) = layout.compact_descriptors()?;
            for (offset, values) in layout.compact_offsets.into_iter().zip([counts, mapping]) {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                library.copy_h2d(view(arena.buffer, offset, bytes.len())?, &bytes)?;
            }
        }
        if dense::is_recording() {
            let load = DenseLoad::mount(library);
            record_payload(&layout, arena.buffer, |operand| load.operand(operand))?;
            return Ok(Self { buffer: arena.buffer, _arena: ArenaOwner::Mount { _allocation: arena },
                layout, budget });
        }
        let per_bank = layout.staging_bytes.max(16);
        let mut hosts = (0..BANKS).map(|_| (0..EXPERT_READ_LANES)
            .map(|_| HostAllocation::named(library, per_bank, "experts/load/exlr8"))
            .collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
        // Declared after the banks so every exit drains before releasing pinned memory.
        let streams = (0..BANKS).map(|_| Ok(LoadStream {
            library, raw: library.cuda_stream_create()?,
        })).collect::<Result<Vec<_>>>()?;
        let started = std::time::Instant::now();
        let mut uploaded_blocks = std::collections::BTreeSet::new();
        for (batch, first) in (0..layout.loads.len())
            .step_by(EXPERT_READ_LANES)
            .enumerate()
        {
            let bank = batch % BANKS;
            let count = (layout.loads.len() - first).min(EXPERT_READ_LANES);
            // SAFETY: the bank's previous uploads use this owned stream.
            unsafe {
                library.cuda_stream_synchronize(streams[bank].raw)?;
            }
            std::thread::scope(|scope| -> Result<()> {
                let readers = hosts[bank]
                    .iter_mut()
                    .take(count)
                    .zip(&layout.loads[first..first + count])
                    .map(|(host, unit)| {
                        let bytes = &mut host.bytes_mut()[..unit.staging_bytes()];
                        scope.spawn(move || unit.read(bytes))
                    })
                    .collect::<Vec<_>>();
                for reader in readers {
                    reader
                        .join()
                        .map_err(|_| anyhow::anyhow!("EXLR8 reader panicked"))??;
                }
                Ok(())
            })?;
            for (host, unit) in hosts[bank].iter().zip(&layout.loads[first..first + count]) {
                let mut offset = 0;
                for (component, (source, &destination)) in
                    unit.source.iter().zip(&unit.destination).enumerate()
                {
                    // Deduplicated blocks must have one writer across streams.
                    if component == 2 && !uploaded_blocks.insert(destination) {
                        offset += source.bytes;
                        continue;
                    }
                    // SAFETY: the pinned bank has unit.staging_bytes bytes; it
                    // is untouched until its stream is synchronized above.
                    let input = CuteafdHostBuffer {
                        ptr: unsafe { host.buffer.ptr.cast::<u8>().add(offset).cast() },
                        bytes: source.bytes,
                        ..host.buffer
                    };
                    // SAFETY: checked destination extent and pinned lifetime.
                    unsafe {
                        library.copy_host_buffer_h2d_async(
                            view(arena.buffer, destination, source.bytes)?,
                            input,
                            source.bytes,
                            streams[bank].raw,
                        )?;
                    }
                    offset += source.bytes;
                }
            }
        }
        for stream in &streams {
            // SAFETY: these owned streams retain the arena and pinned banks.
            unsafe { library.cuda_stream_synchronize(stream.raw)? };
        }
        tracing::info!(layer = layout.layer, rank = layout.rank, world = layout.world,
            manifest = %layout.manifest, payload_bytes = layout.payload_bytes,
            resident_bytes = layout.resident_bytes, seconds = started.elapsed().as_secs_f64(),
            "EXLR8 native units loaded");
        Ok(Self {
            buffer: arena.buffer,
            _arena: ArenaOwner::Mount { _allocation: arena },
            layout,
            budget,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::exlr8::residency::ExpertSet;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn retained_mount_and_push_exlr8_backing_reconciles_direct_unit_views() -> Result<()> {
        use crate::shared::memory::{device::Device, push_initialize, weight_arena::*};
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::placement::planner::{self, *};
        use cuteafd_loader::plan::exlr8::residency::WeightLayout;
        use std::rc::Rc;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let snapshot = fixture.directory().join("snapshot");
        std::fs::create_dir(&snapshot)?;
        cuteafd_loader::plan::testing::exlr8::fixture(&snapshot, false);
        let verify = |allocations: Vec<Allocation>, states: &[Exlr8Weights<'_>], location, physical| -> Result<()> {
            let facts = Facts { allocations, deferred: vec![], sequential: vec![], allowances: vec![] };
            let budgets = vec![Budget { id: "CPU fixture".into(), locations: vec![location],
                physical_bytes: 1 << 30, logical_bytes: 1 << 30, headroom: 0,
                baseline: Baseline::Forecast { occupied: 0, source: "CPU fixture".into(), assumed: false } }];
            let phases = planner::evaluate(&facts, &budgets)?;
            let plan = ResolvedPlan { request: Request { context: 64, pool: Pin::Exact(64),
                    onboard: Pin::Auto, prefill: Pin::Auto, operation: Operation::Serving },
                decisions: Decisions { attention: vec![], onboard_layers: vec![], expert_homes: Default::default(),
                    embedding: None, prefill: PrefillShape { rows: 64, waves: 1 }, lanes: 1 },
                pool_tokens: 64, facts, budgets, phases };
            let receipts = Exlr8Weights::weight_receipts(states, "test.weights", location, physical, Lifetime::RESIDENT)?;
            planner::reconcile(&plan, Phase::Load, &receipts)?;
            assert!(Exlr8Weights::weight_receipts(states, "test.weights", location, 1-physical, Lifetime::RESIDENT).is_err());
            let owner = receipts.iter().position(|r| r.allocation.view.is_none()).unwrap();
            let views = receipts.iter().enumerate().filter(|(_, r)| r.allocation.view.is_some())
                .map(|(i, _)| i).collect::<Vec<_>>();
            for negative in 0..11 {
                let mut bad = receipts.clone();
                match negative {
                    0 => { bad.pop(); },
                    1 => { let mut extra = bad[owner].clone(); extra.allocation.id.push_str(".extra"); bad.push(extra); },
                    2 => bad[owner].allocation.bytes += 1,
                    3 => { bad[views[0]].allocation.bytes += 1; bad[views[1]].allocation.bytes -= 1; },
                    4 => bad[owner].allocation.location = Location::coordinator(7),
                    5 => bad[owner].address = 0,
                    6 => bad[views[0]].address += 256,
                    7 => bad[views[0]].allocation.view.as_mut().unwrap().offset += 256,
                    8 => bad[views[0]].allocation.view.as_mut().unwrap().backing = bad[views[1]].allocation.id.clone(),
                    9 => bad[views[0]].allocation.dtype = "bf16".into(),
                    10 => bad[owner].allocation.lifetime.first = Phase::PostTargetKv,
                    _ => unreachable!(),
                }
                assert!(planner::reconcile(&plan, Phase::Load, &bad).is_err(), "negative {negative}");
            }
            Ok(())
        };
        for (world, rank, physical, location) in [(1, 0, 1, Location::coordinator(0)),
            (2, 1, 1, Location::coordinator(0)),
            (4, 2, 0, Location { device: planner::Device::Spark(3), domain: Domain::Device })] {
            library.cuda_set_device(physical)?;
            fixture.configure_memory(&library, physical, 1 << 30)?;
            let plans = [ExpertSet::Routed, ExpertSet::Shared].into_iter().map(|set|
                Residency::plan(&snapshot, "mixed", 3, world, rank, set)).collect::<std::result::Result<Vec<_>, _>>()?;
            let refs = plans.iter().collect::<Vec<_>>();
            let mounted = WeightLayout::from_residencies("test.weights", &refs, WeightArenaTopology::PerLayer,
                location, Lifetime::RESIDENT)?;
            let pushed = WeightLayout::from_residencies("test.weights", &refs, WeightArenaTopology::SharedRank,
                location, Lifetime::RESIDENT)?;
            assert_eq!(mounted.bytes()?, pushed.bytes()?);
            let states = plans.iter().cloned().map(|layout| Exlr8Weights::load(&library, layout, 1 << 30))
                .collect::<Result<Vec<_>>>()?;
            verify(mounted.allocations, &states, location, physical)?;
            let bytes = pushed.bytes()? as usize;
            let arena = ArenaLayout::new(plans.iter().map(|r| r.resident_bytes), bytes)?;
            let mut loading = LoadingArena::new(Device { library: &library, id: physical }, arena.clone(),
                ExpectedCoverage::spans(&[(0, bytes as u64)]))?;
            // Reuse actual loaded images; protocol source coverage is not this
            // constructor test's authority. Post-Done relocation still runs.
            for (region, state) in states.iter().enumerate() {
                let mut image = vec![0; state.buffer.bytes];
                library.copy_d2h(&mut image, state.buffer)?;
                library.copy_h2d(loading.destination(arena.region(region)?.start, image.len())?, &image)?;
            }
            loading.copied(0, bytes as u64)?;
            let layers = plans.iter().enumerate().map(|(region, residency)|
                push_initialize::Layer::Exlr8 { region, residency }).collect::<Vec<_>>();
            let ready = Rc::new(push_initialize::initialize(loading.done()?, &layers, None, 0)?);
            let mut pushed_states = plans.into_iter().enumerate().map(|(region, layout)| {
                let budget = Exlr8Weights::plan(&layout)?;
                Exlr8Weights::from_push(ready.clone(), region, layout, budget)
            }).collect::<Result<Vec<_>>>()?;
            verify(pushed.allocations, &pushed_states, location, physical)?;
            let receipts = Exlr8Weights::weight_receipts(&pushed_states, "test.weights", location,
                physical, Lifetime::RESIDENT)?;
            assert_eq!(receipts.iter().filter(|r| r.allocation.view.is_none()).count(), 1);
            let second = receipts.iter().find(|r| r.allocation.id == "test.weights.layer.target.3.shared.region").unwrap();
            assert_eq!(second.allocation.view.as_ref().unwrap().offset, arena.region(1)?.start as u64);
            let mut pointers = vec![0; pushed_states[0].pointers()?.bytes];
            library.copy_d2h(&mut pointers, pushed_states[0].pointers()?)?;
            let actual = pointers.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
            assert_eq!(actual, pushed_states[0].layout.descriptors(pushed_states[0].buffer.ptr as u64)?.0);
            assert!(Exlr8Weights::weight_receipts(&pushed_states[..1], "test.weights", location,
                physical, Lifetime::RESIDENT).is_err());
            let buffer = pushed_states[1].buffer;
            pushed_states[1].buffer.ptr = std::ptr::null_mut();
            assert!(Exlr8Weights::weight_receipts(&pushed_states, "test.weights", location,
                physical, Lifetime::RESIDENT).is_err());
            pushed_states[1].buffer = buffer;
            drop(pushed_states);
            drop(ready);
            drop(states);
            assert_eq!(library.cuda_memory_info()?, (1 << 30, 1 << 30));
        }
        Ok(())
    }

    #[test]
    fn exlr8_recorded_payload_matches_mount_units_without_descriptor_writes() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::exlr8::fixture(root.path(), false);
        for set in [ExpertSet::Shared, ExpertSet::Routed] {
            for world in [1, 2, 3] {
                let layout = Residency::plan(root.path(), "mixed", 3, world, 0, set)?;
                let mut expected = vec![0xa5; layout.resident_bytes];
                let mut actual = expected.clone();
                for unit in &layout.loads {
                    let mut bytes = vec![0; unit.staging_bytes()];
                    unit.read(&mut bytes)?;
                    let mut at = 0;
                    for (source, &destination) in unit.source.iter().zip(&unit.destination) {
                        expected[destination..destination + source.bytes].copy_from_slice(&bytes[at..at + source.bytes]);
                        at += source.bytes;
                    }
                }
                let buffer = CuteafdDeviceBuffer { ptr: actual.as_mut_ptr().cast(),
                    bytes: actual.len(), device_id: 3, ..Default::default() };
                let sink = dense::DenseSink::default();
                record_payload(&layout, buffer, |operand| sink.operand(operand))?;
                assert!(actual.iter().all(|&byte| byte == 0xa5));
                let records = sink.take_all();
                let mut destinations = std::collections::BTreeSet::new();
                for operand in records.operands {
                    assert_eq!(operand.device, 3);
                    assert!(destinations.insert(operand.buffer.ptr as usize));
                    for (rect, at) in &operand.lands {
                        let offset = operand.buffer.ptr as usize - buffer.ptr as usize + at;
                        rect.read(&mut actual[offset..offset + rect.bytes()])?;
                    }
                }
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn exlr8_loader_admits_both_sixteen_lane_banks_and_descriptor_upload() {
        let layout = Residency { layer: 3, world: 4, rank: 0, hidden: 6144,
            experts: 256, units: 16, set: ExpertSet::Routed, set_id: String::new(),
            manifest: String::new(), loads: vec![], signs: [vec![], vec![]],
            signs_offsets: [0, 0], pointers_offset: 0, bits_offset: 0, compact_offsets: [0; 2],
            resident_bytes: 4096, payload_bytes: 0, staging_bytes: 400_000 };
        assert_eq!((BANKS, EXPERT_READ_LANES), (2, 16));
        let budget = Exlr8Weights::plan(&layout).unwrap();
        let upload = 256 * 16 * 3 * 3 * 8;
        assert_eq!(budget.pinned_host_bytes, 2 * 16 * 400_000 + upload);
        assert_eq!(budget.read_scratch_bytes - layout.host_bytes(),
            256 * 16 * 3 * (3 * 8 + 4) * 2 + 256 * 17 * 8);
        // Each bank must drain its previous DMA before readers overwrite it;
        // the final drain precedes returning the arena and dropping both banks.
        let source = include_str!("exlr8.rs").split("\n#[cfg(test)]\nmod tests {").next().unwrap();
        let reuse = source.find("library.cuda_stream_synchronize(streams[bank].raw)").unwrap();
        let readers = source.find("unit.read(bytes)").unwrap();
        let upload = source.find("library.copy_host_buffer_h2d_async(").unwrap();
        let final_drain = source.find("library.cuda_stream_synchronize(stream.raw)").unwrap();
        assert!(reuse < readers && readers < upload && upload < final_drain);
    }
}
