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

const BANKS: usize = Residency::LOAD_BANKS;
const _: () = assert!(EXPERT_READ_LANES == Residency::LOAD_LANES);

pub(crate) struct Exlr8Weights<'a> {
    buffer: CuteafdDeviceBuffer,
    _arena: ArenaOwner<'a>,
    pub(crate) layout: Residency,
    pub(crate) budget: ExpertLoadBudget,
}

enum ArenaOwner<'a> {
    Mount { _allocation: DeviceAllocation<'a> },
    Push { _rank: std::rc::Rc<crate::shared::memory::weight_arena::ReadyArena<'a>> },
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
    pub(crate) fn allocation_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        match &self._arena {
            ArenaOwner::Mount { _allocation } => {
                if _allocation.backing.as_ref().and_then(|owner| owner.weight_backing()).is_some() {
                    _allocation.inventory(inventory, name)
                } else {
                    inventory.backed_device(name, self.buffer, _allocation.buffer)
                }
            }
            ArenaOwner::Push { _rank } => inventory.backed_device(name, self.buffer, _rank.allocation_buffer()),
        }
    }

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
        Ok(Self { buffer, _arena: ArenaOwner::Push { _rank: rank }, layout, budget })
    }

    pub(crate) fn plan(layout: &Residency) -> Result<ExpertLoadBudget> {
        let descriptor_bytes = layout.descriptor_host_bytes()?;
        let bank_bytes = layout.mount_pinned_bytes()?.checked_add(layout.sync_upload_bytes()?)
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

    pub(crate) fn shared_request(layout: &Residency) -> Result<cuteafd_loader::placement::planner::Allocation> {
        use cuteafd_loader::placement::planner::{Allocation, Lifetime, Location};
        ensure!(layout.set == cuteafd_loader::plan::exlr8::residency::ExpertSet::Shared
            && matches!(layout.world, 1 | 2) && layout.rank < layout.world && layout.resident_bytes > 0,
            "invalid native shared allocation request");
        Ok(Allocation::buffer(format!("glm.weights.gpu.{}.shared.layer.{}.backing", layout.rank, layout.layer),
            Location::coordinator(layout.rank as u8), layout.resident_bytes as u64, Lifetime::RESIDENT,
            cuteafd_core::memory_layout::Category::Weights, "GLM target weights"))
    }

    /// `available` excludes all caller-owned workspace/graph reservations. Admit
    /// pinned bytes against host/UMA headroom before calling this device loader.
    pub(crate) fn load(
        library: &'a NativeLibrary,
        layout: Residency,
        available: usize,
    ) -> Result<Self> {
        layout.check_budget(available, &[])?;
        let allocation = DeviceAllocation::new(library, layout.resident_bytes, crate::shared::memory::tag(crate::shared::memory::MemoryKind::Experts, "experts/weights/exlr8"))?;
        Self::load_into(library, layout, available, allocation)
    }

    pub(crate) fn load_into(library: &'a NativeLibrary, layout: Residency, available: usize,
        arena: DeviceAllocation<'a>) -> Result<Self> {
        let budget = Self::plan(&layout)?;
        layout.check_budget(available, &[])?;
        let shared = if layout.set == cuteafd_loader::plan::exlr8::residency::ExpertSet::Shared {
            Some(Self::shared_request(&layout)?)
        } else { None };
        let device = library.cuda_get_device()?;
        ensure!(arena.buffer.bytes == layout.resident_bytes, "EXLR8 arena extent differs from plan");
        if let Some(request) = shared {
            ensure!(arena.buffer.bytes as u64 == request.bytes && arena.buffer.device_id == device
                && !arena.buffer.ptr.is_null(), "native shared allocation differs from request");
        }
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
            .map(|_| HostAllocation::new(library, per_bank, crate::shared::memory::tag(crate::shared::memory::MemoryKind::Staging, "experts/load/exlr8")))
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
    #[test]
    fn arena_mount_views_preserve_units_descriptors_and_signs() -> Result<()> {
        use crate::shared::memory::device::Device;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        let device = Device { library: &library, id: 1 };
        for padded in [false, true] {
            let root = tempfile::tempdir()?;
            cuteafd_loader::plan::testing::exlr8::fixture(root.path(), padded);
            for world in [1, 2, 6] {
                for rank in 0..world {
                    let plan = Residency::plan(root.path(), "mixed", 3, world, rank, ExpertSet::Routed)?;
                    // A second independent layer image exercises a nonzero base.
                    let regions = vec![("first".into(), plan.resident_bytes),
                        ("second".into(), plan.resident_bytes)];
                    let before = fixture.events()?.matches('D').count();
                    let arena = super::super::weight_layout::arena(device, "exlr8", regions)?;
                    let mut weights = Vec::new();
                    for name in ["first", "second"] {
                        let allocation = DeviceAllocation::from_allocation(arena.view(name, plan.resident_bytes)?);
                        weights.push(device.run(|| Exlr8Weights::load_into(&library, plan.clone(),
                            usize::MAX, allocation))?);
                    }
                    assert_eq!(fixture.events()?.matches('D').count() - before, 1);
                    let layout = arena.layout().clone();
                    drop(arena);
                    let mut inventory = crate::shared::memory::receipts::Inventory::new(
                        cuteafd_loader::placement::planner::CoordinatorDevices::new(&[1])?, "fixture",
                        cuteafd_loader::placement::planner::Lifetime::TARGET,
                        cuteafd_core::memory_layout::Category::Experts, "expert fixture");
                    for (index, weight) in weights.iter().enumerate() {
                        weight.allocation_receipts(&mut inventory, &format!("layer.{index}"))?;
                    }
                    super::super::weight_layout::assert_receipts(&inventory.receipts, &[layout])?;
                    for weight in &weights {
                        for unit in &weight.layout.loads {
                            let mut staged = vec![0; unit.staging_bytes()];
                            unit.read(&mut staged)?;
                            let mut cursor = 0;
                            for (source, &offset) in unit.source.iter().zip(&unit.destination) {
                                let mut actual = vec![0; source.bytes];
                                device.run(|| library.copy_d2h_quiescent(&mut actual,
                                    view(weight.buffer, offset, source.bytes)?))?;
                                assert_eq!(actual, staged[cursor..cursor + source.bytes]);
                                cursor += source.bytes;
                            }
                        }
                        for (index, signs) in weight.layout.signs.iter().enumerate() {
                            let mut actual = vec![0; signs.len()];
                            device.run(|| library.copy_d2h_quiescent(&mut actual, weight.signs(index != 0)?))?;
                            assert_eq!(&actual, signs);
                        }
                        let (pointers, bits) = weight.layout.descriptors(weight.buffer.ptr as u64)?;
                        for (buffer, expected) in [(weight.pointers()?, pointers.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>()),
                            (weight.bits()?, bits.iter().flat_map(|n| n.to_le_bytes()).collect())] {
                            let mut actual = vec![0; buffer.bytes];
                            device.run(|| library.copy_d2h_quiescent(&mut actual, buffer))?;
                            assert_eq!(actual, expected);
                        }
                    }
                    drop(weights);
                }
            }
        }
        assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
        Ok(())
    }

    use super::*;
    use cuteafd_loader::plan::exlr8::residency::ExpertSet;

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
