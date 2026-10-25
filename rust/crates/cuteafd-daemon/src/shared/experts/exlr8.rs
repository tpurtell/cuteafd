//! EXLR8 unit arena ownership, shared by Spark and coordinator K1 execution.
use super::layer::{EXPERT_READ_LANES, ExpertLoadBudget};
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_loader::plan::exlr8::residency::Residency;

mod execution;
pub(crate) use execution::{grouped_decode_from_row, Exlr8Execution};

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
        Ok(Self { buffer, _arena: ArenaOwner::Push { _rank: rank }, layout, budget })
    }

    pub(crate) fn plan(layout: &Residency) -> Result<ExpertLoadBudget> {
        let descriptor_bytes = layout
            .experts
            .checked_mul(layout.units)
            .and_then(|n| n.checked_mul(3 * (3 * 8 + 4) * 2))
            .context("EXLR8 descriptor staging overflow")?;
        let compact_bytes = if layout.set == cuteafd_loader::plan::exlr8::residency::ExpertSet::Routed {
            layout.experts.checked_mul(layout.units.checked_add(1).context("compact unit overflow")?)
                .and_then(|n| n.checked_mul(8)).context("compact descriptor overflow")?
        } else { 0 };
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
            read_scratch_bytes: descriptor_bytes.checked_add(compact_bytes)
                .and_then(|n| n.checked_add(layout.host_bytes()))
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
        let per_bank = layout.staging_bytes.max(16);
        let mut hosts = (0..BANKS)
            .map(|_| {
                (0..EXPERT_READ_LANES)
                    .map(|_| HostAllocation::named(library, per_bank, "experts/load/exlr8"))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        // Drain queued uploads before pinned banks or the arena on all exits.
        let streams = (0..BANKS)
            .map(|_| {
                Ok(LoadStream {
                    library,
                    raw: library.cuda_stream_create()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
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
