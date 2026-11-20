//! Owned compressed EXL3 device residency. The native execution adapter binds
//! these buffers only after loading completes and must retain them for graphs.
use super::layer::{ExpertLayer, ExpertLoadBudget, EXPERT_READ_LANES};
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream, Tag};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{OfficialV41Catalog, V41Exl3Layer, V41Exl3Residency};

const JOBS_PER_EXPERT: usize = 9;
const BANKS: usize = 2;
pub(crate) mod execution;
pub(crate) mod partial;
#[cfg(test)]
mod partial_screen;
pub(crate) mod worker;

/// Family-qualified package name for one decoder-tier pair and codebook.
pub(crate) fn package_name_for(
    family: &str,
    tiers: &[usize],
    codebook: cuteafd_loader::Exl3Codebook,
) -> String {
    codebook.package_name(family, tiers)
}

/// Resolve the checkpoint family's exact decoder-tier + codebook package and layout.
pub(crate) fn aot_layout_directory_for(
    native_lib: &std::path::Path,
    tiers: &[usize],
    codebook: cuteafd_loader::Exl3Codebook,
    layout: &str,
) -> std::path::PathBuf {
    let root = native_lib
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("exl3");
    let family = cuteafd_core::expert_geometry().family().unwrap_or("unknown");
    root.join(package_name_for(family, tiers, codebook)).join(layout)
}

/// Read an EXL3 export manifest, failing closed with an actionable error when a
/// MUL1 routed checkpoint's `-mul1` package has not been exported (including
/// when only the sibling MCG package is installed).
pub(crate) fn read_export_manifest_bytes(
    directory: &std::path::Path,
    required_codebook: Option<cuteafd_loader::Exl3Codebook>,
) -> Result<Vec<u8>> {
    let path = directory.join("v41_exl3.json");
    match std::fs::read(&path) {
        Ok(bytes) => Ok(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mul1_package = directory.ancestors().find_map(|ancestor| {
                let name = ancestor.file_name()?.to_str()?;
                name.strip_suffix("-mul1")
                    .filter(|stem| stem.starts_with("exl3-"))
                    .map(|stem| (ancestor, name, stem))
            });
            if let Some((pkg_dir, pkg_name, mcg_name)) = mul1_package {
                let mcg_installed = pkg_dir
                    .parent()
                    .is_some_and(|root| root.join(mcg_name).exists());
                if mcg_installed {
                    anyhow::bail!(
                        "EXL3 MUL1 routed-expert programs are not exported at {} \
                         (only MCG package {mcg_name} is installed; refusing fallback to MCG — \
                         export and install {pkg_name} with --codebook mul1)",
                        path.display()
                    );
                }
                anyhow::bail!(
                    "EXL3 MUL1 routed-expert programs are not exported at {} \
                     (export and install {pkg_name} with --codebook mul1)",
                    path.display()
                );
            }
            if required_codebook == Some(cuteafd_loader::Exl3Codebook::Mul1) {
                anyhow::bail!(
                    "EXL3 MUL1 routed-expert programs are not exported at {} \
                     (export with --codebook mul1; refusing fallback to MCG)",
                    path.display()
                );
            }
            Err(error).with_context(|| format!("missing EXL3 export manifest {}", path.display()))
        }
        Err(error) => Err(error).with_context(|| format!("reading EXL3 export manifest {}", path.display())),
    }
}

pub(crate) struct Exl3Weights<'a> {
    buffers: Vec<WeightBuffer>,
    _arena: ArenaOwner<'a>,
    pub(crate) layout: V41Exl3Residency,
    pub(crate) budget: ExpertLoadBudget,
}

enum ArenaOwner<'a> {
    Mount(DeviceAllocation<'a>),
    Push { _rank: std::rc::Rc<crate::shared::memory::weight_arena::ReadyArena<'a>> },
}

struct WeightBuffer { buffer: CuteafdDeviceBuffer }

/// One explicitly sized allocation avoids independent CUDA allocation padding
/// for every projection/rotation table. All kernel pointers remain 256B aligned.
fn arena_layout(plan: &V41Exl3Residency) -> Result<(Vec<usize>, usize)> {
    Ok(plan.device_arena_layout()?)
}

fn residency_selection(layer: ExpertLayer) -> Result<(V41Exl3Layer, usize, usize)> {
    Ok(match layer {
        ExpertLayer::Backbone { layer, rank } => (V41Exl3Layer::Backbone(layer), 4, rank),
        ExpertLayer::BackboneFull { layer } => (V41Exl3Layer::Backbone(layer), 1, 0),
        ExpertLayer::BackboneTp2 { layer, rank } => (V41Exl3Layer::Backbone(layer), 2, rank),
        // Declared TP shard on a compressed checkpoint: the shard count
        // rides on the layer, so the whole-block H128 partition drives residency
        // directly. The two- and four-rank native layers stay untouched above.
        ExpertLayer::BackboneExl3Tp { layer, rank, world } => {
            ensure!(
                (1..=8).contains(&world) && rank < world,
                "EXL3 Spark shards require TP1..8 and rank below world, got TP{world} rank {rank}"
            );
            (V41Exl3Layer::Backbone(layer), world, rank)
        }
        ExpertLayer::Dspark { stage } => (V41Exl3Layer::Dspark(stage), 1, 0),
        ExpertLayer::DsparkTp2 { .. } => anyhow::bail!("TP2 dSpark EXL3 is not implemented"),
        // Native Spark TP shards are native-checkpoint only; EXL3 selects
        // `BackboneExl3Tp` before reaching here.
        ExpertLayer::BackboneSparkTp { .. } => anyhow::bail!(
            "native Spark TP shards require a native expert checkpoint"
        ),
    })
}

pub(crate) fn planned_residency(catalog: &OfficialV41Catalog, layer: ExpertLayer) -> Result<V41Exl3Residency> {
    let spark = matches!(layer, ExpertLayer::BackboneExl3Tp { .. });
    let (layer, world, rank) = residency_selection(layer)?;
    let manifest = catalog.exl3().context("EXL3 residency requires a routed EXL3 checkpoint")?;
    if spark { catalog.exl3_spark_residency(layer, world, rank) }
        else { manifest.residency(layer, world, rank) }
}

fn budget(catalog: &OfficialV41Catalog, plan: &V41Exl3Residency) -> Result<ExpertLoadBudget> {
    ensure!(
        plan.loads.len() == plan.experts * JOBS_PER_EXPERT,
        "EXL3 staging inventory mismatch"
    );
    let staging = plan
        .loads
        .chunks(JOBS_PER_EXPERT)
        .map(|jobs| jobs.iter().map(|j| j.bytes).sum::<usize>())
        .max()
        .unwrap_or(0);
    let init_bytes = plan
        .buffers
        .iter()
        .map(|b| b.initial_words.len() * 4)
        .max()
        .unwrap_or(0);
    // Admit a fabric-bandwidth-delay window in pageable host RAM.
    let scratch = plan
        .scratch_bytes(catalog)?
        .checked_mul(64)
        .context("EXL3 scratch overflow")?
        .max(cuteafd_loader::EXPERT_READ_WINDOW_BYTES);
    Ok(ExpertLoadBudget {
        resident_bytes: arena_layout(plan)?.1,
        device_staging_bytes: 0,
        pinned_host_bytes: staging.max(init_bytes) * EXPERT_READ_LANES * BANKS,
        read_scratch_bytes: scratch * EXPERT_READ_LANES * BANKS,
    })
}

impl<'a> Exl3Weights<'a> {
    pub(crate) fn allocation_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        if let ArenaOwner::Mount(owner) = &self._arena {
            owner.inventory(inventory, name)?;
            if let Some(backing) = owner.backing.as_ref().and_then(|a| a.weight_backing()) {
                for (index, buffer) in self.buffers.iter().enumerate() {
                    inventory.weight_view(&format!("{name}.{index}"), buffer.buffer,
                        crate::shared::memory::device::WeightArenaBacking { id: backing.id, buffer: backing.buffer, kind: backing.kind })?;
                }
                return Ok(());
            }
        }
        let owner = match &self._arena {
            ArenaOwner::Mount(owner) => owner.backing.as_ref().and_then(|a| a.weight_backing()).map_or(owner.buffer, |b| b.buffer),
            ArenaOwner::Push { _rank } => _rank.allocation_buffer(),
        };
        for (index, buffer) in self.buffers.iter().enumerate() {
            inventory.backed_device(&format!("{name}.{index}"), buffer.buffer, owner)?;
        }
        Ok(())
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        match &self._arena {
            ArenaOwner::Mount(arena) => arena.buffer.bytes,
            ArenaOwner::Push { .. } => self.budget.resident_bytes,
        }
    }

    pub(crate) fn from_push(
        rank: std::rc::Rc<crate::shared::memory::weight_arena::ReadyArena<'a>>,
        region: usize, layout: V41Exl3Residency, mut budget: ExpertLoadBudget,
    ) -> Result<Self> {
        let owner = rank.region(region)?;
        let (offsets, bytes) = layout.device_arena_layout()?;
        ensure!(bytes == budget.resident_bytes && owner.buffer().bytes == bytes,
            "pushed EXL3 region differs from admission");
        let buffers = layout.buffers.iter().zip(offsets).map(|(spec, offset)| {
            Ok(WeightBuffer { buffer: owner.subregion(offset, spec.bytes.max(16))?.buffer() })
        }).collect::<Result<Vec<_>>>()?;
        // The push ring and validation image are gone before execution inventory.
        budget.device_staging_bytes = 0;
        budget.pinned_host_bytes = 0;
        budget.read_scratch_bytes = 0;
        Ok(Self { buffers, _arena: ArenaOwner::Push { _rank: rank }, layout, budget })
    }

    pub(crate) fn plan(
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<ExpertLoadBudget> {
        budget(catalog, &planned_residency(catalog, layer)?)
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        let index = self
            .layout
            .buffers
            .iter()
            .position(|b| b.name == name)
            .with_context(|| format!("missing EXL3 device buffer {name}"))?;
        Ok(self.buffers[index].buffer)
    }

    /// A full compressed projection bank feeds both GPU slices, including
    /// replicated rotations, with each MCG word validated once per projection.
    pub(crate) fn load_tp2_pair(
        devices: [crate::shared::memory::device::Device<'a>; 2],
        catalog: &OfficialV41Catalog,
        layers: &[usize],
        available: [usize; 2],
    ) -> Result<[crate::shared::memory::device::DeviceOwner<'a, Vec<Self>>; 2]> {
        use super::paired_load::{self, Fences, Projection};
        let started = std::time::Instant::now();
        let mut weights = [devices[0].own(|| Ok(Vec::new()))?, devices[1].own(|| Ok(Vec::new()))?];
        let mut remaining = available;
        let layouts = (0..2).map(|rank| {
            let selections = layers.iter().map(|&layer| ExpertLayer::BackboneTp2 { layer, rank }).collect::<Vec<_>>();
            let layout = super::weight_layout::layout(devices[rank].id, "exl3",
                super::exl3::regions(catalog, selections.iter().copied())?)?;
            let staging = selections.into_iter().map(|layer| Self::plan(catalog, layer))
                .collect::<Result<Vec<_>>>()?.iter().map(|p| p.device_staging_bytes).max().unwrap_or(0);
            super::weight_layout::admit(&layout, staging, 0, available[rank])?;
            Ok(layout)
        }).collect::<Result<Vec<_>>>()?;
        let arenas = layouts.into_iter().enumerate().map(|(rank, layout)|
            crate::shared::memory::weight_arena::DeviceWeightArena::new(devices[rank], layout, cuteafd_core::memory_layout::ArenaTag::experts("experts/exl3-tp2")))
            .collect::<Result<Vec<_>>>()?;
        let mut full_bytes = 0;
        let mut pinned_admitted = 0;
        for &layer in layers {
            let mut pinned = 0;
            for rank in 0..2 {
                let plan = planned_residency(catalog, ExpertLayer::BackboneTp2 { layer, rank })?;
                let budget = budget(catalog, &plan)?;
                ensure!(budget.peak_device_bytes()? <= remaining[rank], "paired EXL3 admission exceeded");
                remaining[rank] -= budget.resident_bytes;
                pinned += budget.pinned_host_bytes;
                for jobs in plan.loads.chunks(JOBS_PER_EXPERT) {
                    let bytes = jobs.iter().try_fold(0usize, |total, job|
                        Ok::<_, anyhow::Error>(total + usize::try_from(catalog.tensor(&job.tensor)?.metadata.byte_length)?))?;
                    full_bytes = full_bytes.max(bytes);
                }
                // Initial descriptor tables use this bank too.
                full_bytes = full_bytes.max(plan.buffers.iter().map(|b| b.initial_words.len() * 4).max().unwrap_or(0));
                devices[rank].run(|| {
                    let (offsets, bytes) = arena_layout(&plan)?;
                    let arena = DeviceAllocation::from_allocation(arenas[rank].view(&super::weight_layout::name(ExpertLayer::BackboneTp2 { layer, rank }, 0), bytes)?);
                    let buffers = plan.buffers.iter().zip(offsets).map(|(spec, offset)| {
                        let mut buffer = arena.buffer;
                        // SAFETY: checked arena layout reserves every aligned buffer.
                        unsafe { buffer.ptr = buffer.ptr.cast::<u8>().add(offset).cast(); }
                        buffer.bytes = spec.bytes.max(16);
                        WeightBuffer { buffer }
                    }).collect();
                    weights[rank].push(Self { buffers, _arena: ArenaOwner::Mount(arena), layout: plan, budget });
                    Ok(())
                })?;
            }
            pinned_admitted = pinned_admitted.max(pinned);
        }
        if super::rtx::dense::is_recording() {
            use super::rtx::dense::{DenseLoad, Operand, Rect};
            for rank in 0..2 {
                let load = DenseLoad::mount(devices[rank].library);
                for weight in weights[rank].iter() {
                    for job in &weight.layout.loads {
                        let tensor = catalog.tensor(&job.tensor)?;
                        let slice = catalog.exl3_tensor_slice(&job.tensor, 2, rank)?;
                        let rect = Rect { name: job.tensor.clone(), path: catalog.snapshot().join(&tensor.shard),
                            base: tensor.metadata.byte_offset, row_bytes: slice.source_row_bytes,
                            rows: 0..slice.rows, cols: slice.column_start_bytes..slice.column_start_bytes + slice.selected_row_bytes };
                        ensure!(rect.bytes() == job.bytes, "recorded EXL3 slice differs from upload");
                        for &(buffer, offset) in &job.destinations {
                            let mut destination = weight.buffers[buffer].buffer;
                            ensure!(offset.checked_add(job.bytes).is_some_and(|end| end <= destination.bytes),
                                "recorded EXL3 upload exceeds destination");
                            // SAFETY: checked subregion belongs to this retained weight owner.
                            destination.ptr = unsafe { destination.ptr.cast::<u8>().add(offset).cast() };
                            destination.bytes = job.bytes;
                            load.operand(Operand { device: devices[rank].id, buffer: destination,
                                lands: vec![(rect.clone(), 0)], bytes: job.bytes })?;
                        }
                    }
                }
            }
            return Ok(weights);
        }
        let mut hosts = paired_load::banks(devices[0], EXPERT_READ_LANES, full_bytes, pinned_admitted)?;
        let mut fences = Fences::new(devices)?;
        // Drain before pinned banks and GPU arenas on every error/unwind path.
        let streams = paired_load::streams(devices)?;
        let allocation_seconds = started.elapsed().as_secs_f64();
        for index in 0..weights[0].len() {
            // Initialize each GPU's descriptors using pinned storage only after
            // the previous layer's consumers of this bank have finished.
            for rank in 0..2 {
                for (spec, buffer) in weights[rank][index].layout.buffers.iter().zip(&weights[rank][index].buffers) {
                    if spec.initial_words.is_empty() { continue; }
                    paired_load::drain(&streams)?;
                    for (dst, word) in hosts[0][0].bytes_mut().chunks_exact_mut(4).zip(&spec.initial_words) {
                        dst.copy_from_slice(&word.to_le_bytes());
                    }
                    devices[rank].run(|| {
                        // SAFETY: descriptor source stays alive and unchanged through drain.
                        unsafe { devices[rank].library.copy_host_buffer_h2d_async(buffer.buffer,
                            hosts[0][0].buffer, spec.initial_words.len() * 4, streams[rank].raw)?; }
                        Ok(())
                    })?;
                }
            }
            paired_load::drain(&streams)?;
        }
        let mut group = 0;
        for (index, &layer) in layers.iter().enumerate() {
            let layer_started = std::time::Instant::now();
            let experts = weights[0][index].layout.experts;
            let mut storage_bytes_read = 0;
            let mut read_seconds = 0.;
            let mut bank_wait_seconds = 0.;
            let mut upload_submit_seconds = 0.;
            for first in (0..experts).step_by(EXPERT_READ_LANES) {
                let bank = group % BANKS;
                let wait_started = std::time::Instant::now();
                fences.reuse(bank)?;
                bank_wait_seconds += wait_started.elapsed().as_secs_f64();
                let count = EXPERT_READ_LANES.min(experts - first);
                let plans = (first..first + count).map(|expert| {
                    let mut offset = 0;
                    (expert * JOBS_PER_EXPERT..(expert + 1) * JOBS_PER_EXPERT).map(|job| {
                        let name = &weights[0][index].layout.loads[job].tensor;
                        ensure!(*name == weights[1][index].layout.loads[job].tensor
                            && weights[0][index].layout.codebook == weights[1][index].layout.codebook,
                            "paired EXL3 projection mismatch");
                        let projection = Projection::new(catalog, name.clone(),
                            weights[0][index].layout.codebook, offset,
                            [catalog.exl3_tensor_slice(name, 2, 0)?, catalog.exl3_tensor_slice(name, 2, 1)?])?;
                        offset += projection.bytes;
                        Ok(projection)
                    }).collect::<Result<Vec<_>>>()
                }).collect::<Result<Vec<_>>>()?;
                paired_load::trace_overlap(&streams, "read_start", layer, first)?;
                let read_started = std::time::Instant::now();
                storage_bytes_read += paired_load::read_group(catalog, &plans, &mut hosts[bank])?;
                read_seconds += read_started.elapsed().as_secs_f64();
                let upload_started = std::time::Instant::now();
                for rank in 0..2 {
                    for (lane, jobs) in plans.iter().enumerate() {
                        for (slot, projection) in jobs.iter().enumerate() {
                            let weight = &weights[rank][index];
                            let job = &weight.layout.loads[(first + lane) * JOBS_PER_EXPERT + slot];
                            ensure!(job.bytes == projection.slices[rank].bytes(), "paired EXL3 slice size mismatch");
                            for &(buffer, offset) in &job.destinations {
                                let mut destination = weight.buffers[buffer].buffer;
                                ensure!(offset.checked_add(job.bytes).is_some_and(|end| end <= destination.bytes),
                                    "paired EXL3 upload exceeds destination");
                                // SAFETY: checked resident destination extent belongs to this GPU.
                                unsafe { destination.ptr = destination.ptr.cast::<u8>().add(offset).cast(); }
                                destination.bytes = job.bytes;
                                projection.upload(devices[rank], rank, hosts[bank][lane].buffer,
                                    destination, &streams[rank])?;
                            }
                        }
                    }
                }
                upload_submit_seconds += upload_started.elapsed().as_secs_f64();
                fences.record(bank, &streams)?;
                paired_load::trace_overlap(&streams, "both_submitted", layer, first)?;
                group += 1;
            }
            tracing::info!(layer, storage_bytes_read, read_seconds, bank_wait_seconds, upload_submit_seconds,
                elapsed_seconds = layer_started.elapsed().as_secs_f64(), "EXL3 TP2 shared-read layer timeline");
        }
        paired_load::drain(&streams)?;
        tracing::info!(allocation_seconds, elapsed_seconds = started.elapsed().as_secs_f64(),
            pinned_host_bytes = full_bytes * EXPERT_READ_LANES * BANKS, pinned_admitted,
            read_scratch_bytes = 0, "EXL3 TP2 shared-read load complete");
        Ok(weights)
    }

    pub(crate) fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
    ) -> Result<Self> {
        ensure!(Self::plan(catalog, layer)?.peak_device_bytes()? <= available_device_bytes,
            "EXL3 weights and staging exceed admission");
        let device = crate::shared::memory::device::Device { library, id: library.cuda_get_device()? };
        let arena = super::exl3::arena(device, catalog, [layer])?;
        Self::load_into(library, catalog, layer, available_device_bytes, &arena)
    }

    pub(crate) fn load_into(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
        layer: ExpertLayer, available_device_bytes: usize,
        arena: &crate::shared::memory::weight_arena::DeviceWeightArena<'a>) -> Result<Self> {
        let layout = planned_residency(catalog, layer)?;
        let budget = budget(catalog, &layout)?;
        ensure!(
            budget.peak_device_bytes()? <= available_device_bytes,
            "EXL3 layer needs {} device bytes, budget is {available_device_bytes}",
            budget.resident_bytes
        );
        let (offsets, bytes) = arena_layout(&layout)?;
        let arena = DeviceAllocation::from_allocation(arena.view(&super::weight_layout::name(layer, 0), bytes)?);
        let buffers: Vec<_> = layout.buffers.iter().zip(offsets).map(|(spec, offset)| {
            let mut buffer = arena.buffer;
            buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };
            buffer.bytes = spec.bytes.max(16);
            WeightBuffer { buffer }
        }).collect();
        let per_host = budget.pinned_host_bytes / (EXPERT_READ_LANES * BANKS);
        let per_scratch = budget.read_scratch_bytes / (EXPERT_READ_LANES * BANKS);
        let mut hosts = (0..BANKS)
            .map(|_| {
                (0..EXPERT_READ_LANES)
                    .map(|_| HostAllocation::new(library, per_host, Tag::staging("experts/exl3-load")))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scratch = (0..BANKS)
            .map(|_| {
                (0..EXPERT_READ_LANES)
                    .map(|_| vec![0; per_scratch])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Declared after every buffer owner: drain queued copies first on errors.
        let streams = (0..BANKS)
            .map(|_| {
                Ok(LoadStream {
                    library,
                    raw: library.cuda_stream_create()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for (spec, allocation) in layout.buffers.iter().zip(&buffers) {
            if spec.initial_words.is_empty() {
                continue;
            }
            let host = &mut hosts[0][0];
            for (dst, word) in host
                .bytes_mut()
                .chunks_exact_mut(4)
                .zip(&spec.initial_words)
            {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            unsafe {
                library.copy_host_buffer_h2d_async(
                    allocation.buffer,
                    host.buffer,
                    spec.initial_words.len() * 4,
                    streams[0].raw,
                )?;
                library.cuda_stream_synchronize(streams[0].raw)?;
            }
        }
        let load_started = std::time::Instant::now();
        let mut storage_bytes_read = 0;
        for (group, first) in (0..layout.experts).step_by(EXPERT_READ_LANES).enumerate() {
            let bank = group % BANKS;
            let count = (layout.experts - first).min(EXPERT_READ_LANES);
            // Other bank uploads can continue while these CPU readers run.
            unsafe {
                library.cuda_stream_synchronize(streams[bank].raw)?;
            }
            std::thread::scope(|scope| -> Result<()> {
                let mut readers = Vec::with_capacity(count);
                for (lane, (host, scratch)) in hosts[bank]
                    .iter_mut()
                    .zip(&mut scratch[bank])
                    .take(count)
                    .enumerate()
                {
                    let plan = &layout;
                    let bytes = host.bytes_mut();
                    readers.push(scope.spawn(move || OfficialV41Catalog::count_storage_reads(|| {
                        let mut offset = 0;
                        for job in
                            (first + lane) * JOBS_PER_EXPERT..(first + lane + 1) * JOBS_PER_EXPERT
                        {
                            let size = plan.loads[job].bytes;
                            plan.read_into(
                                catalog,
                                job,
                                &mut bytes[offset..offset + size],
                                scratch,
                            )?;
                            offset += size;
                        }
                        Ok(())
                    })));
                }
                for reader in readers {
                    storage_bytes_read += reader
                        .join()
                        .map_err(|_| anyhow::anyhow!("EXL3 reader panicked"))??.1;
                }
                Ok(())
            })?;
            for (lane, host) in hosts[bank].iter().take(count).enumerate() {
                let mut source_offset = 0;
                for job in &layout.loads
                    [(first + lane) * JOBS_PER_EXPERT..(first + lane + 1) * JOBS_PER_EXPERT]
                {
                    for &(buffer, offset) in &job.destinations {
                        ensure!(
                            offset
                                .checked_add(job.bytes)
                                .is_some_and(|end| end <= buffers[buffer].buffer.bytes),
                            "EXL3 upload exceeds destination"
                        );
                        unsafe {
                            let mut source = host.buffer;
                            source.ptr = source.ptr.cast::<u8>().add(source_offset).cast();
                            source.bytes = job.bytes;
                            let mut destination = buffers[buffer].buffer;
                            destination.ptr = destination.ptr.cast::<u8>().add(offset).cast();
                            destination.bytes = job.bytes;
                            library.copy_host_buffer_h2d_async(
                                destination,
                                source,
                                job.bytes,
                                streams[bank].raw,
                            )?;
                        }
                    }
                    source_offset += job.bytes;
                }
            }
        }
        for stream in &streams {
            unsafe {
                library.cuda_stream_synchronize(stream.raw)?;
            }
        }
        tracing::info!(?layer, storage_bytes_read, elapsed_seconds = load_started.elapsed().as_secs_f64(),
            "EXL3 expert load timeline");
        Ok(Self {
            buffers,
            _arena: ArenaOwner::Mount(arena),
            layout,
            budget,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn arena_mount_exl3_views_preserve_all_tensor_destinations() -> Result<()> {
        use crate::shared::memory::device::Device;
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(2, 4);
        let tensors = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
            || tensor.0.contains(".experts.0.")).collect::<Vec<_>>();
        let mut config = qwen4_config(2);
        config["text_config"]["num_experts"] = 1.into();
        config["text_config"]["num_experts_per_tok"] = 1.into();
        write_snapshot(root.path(), &config, &tensors, None);
        let path = root.path().join("model-00001-of-00001.safetensors");
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        use std::os::unix::fs::FileExt;
        for tensor in cuteafd_loader::read_safetensors_metadata(&path)? {
            if tensor.name.ends_with(".mcg") {
                file.write_all_at(&0xcbac1fedu32.to_le_bytes(), tensor.byte_offset)?;
            }
        }
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let device = Device { library: &library, id: 1 };
        for world in [1, 2] {
            for rank in 0..world {
                let selections = (0..2).map(|layer| if world == 1 { ExpertLayer::BackboneFull { layer } }
                    else { ExpertLayer::BackboneTp2 { layer, rank } }).collect::<Vec<_>>();
                let before = fixture.events()?.matches('D').count();
                let arena = super::arena(device, &catalog, selections.iter().copied())?;
                let weights = device.run(|| selections.into_iter().map(|selection|
                    Exl3Weights::load_into(&library, &catalog, selection, usize::MAX, &arena))
                    .collect::<Result<Vec<_>>>())?;
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
                    let mut staging = vec![0; weight.layout.staging_bytes()];
                    let mut scratch = vec![0; weight.layout.scratch_bytes(&catalog)? * 64];
                    for index in 0..weight.layout.loads.len() {
                        let bytes = weight.layout.read_into(&catalog, index, &mut staging, &mut scratch)?;
                        for &(buffer, offset) in &weight.layout.loads[index].destinations {
                            let source = weight.buffers[buffer].buffer;
                            ensure!(offset + bytes <= source.bytes, "fixture destination bounds");
                            // SAFETY: checked span belongs to the retained weight backing.
                            let source = CuteafdDeviceBuffer { ptr: unsafe {
                                source.ptr.cast::<u8>().add(offset).cast() }, bytes, ..source };
                            let mut actual = vec![0; bytes];
                            device.run(|| library.copy_d2h(&mut actual, source))?;
                            assert_eq!(actual, staging[..bytes]);
                        }
                    }
                }
                drop(weights);
            }
        }
        assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
        Ok(())
    }

    use super::*;

    #[test]
    fn larger_read_windows_preserve_device_and_pinned_admission() -> Result<()> {
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(1, 4);
        let tensors: Vec<_> = tensors.into_iter().filter(|tensor| !tensor.0.contains(".experts.")
            || tensor.0.contains(".experts.0.")).collect();
        let mut config = qwen4_config(1);
        config["text_config"]["num_experts"] = serde_json::json!(1);
        config["text_config"]["num_experts_per_tok"] = serde_json::json!(1);
        write_snapshot(root.path(), &config, &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        for rank in 0..2 {
            let plan = planned_residency(&catalog,
                ExpertLayer::BackboneExl3Tp { layer: 0, rank, world: 2 })?;
            let actual = budget(&catalog, &plan)?;
            let staging = plan.loads.chunks(JOBS_PER_EXPERT)
                .map(|jobs| jobs.iter().map(|job| job.bytes).sum::<usize>()).max().unwrap_or(0);
            let init = plan.buffers.iter().map(|buffer| buffer.initial_words.len() * 4).max().unwrap_or(0);
            assert_eq!(actual.resident_bytes, arena_layout(&plan)?.1);
            assert_eq!(actual.device_staging_bytes, 0);
            assert_eq!(actual.pinned_host_bytes, staging.max(init) * EXPERT_READ_LANES * BANKS);
            let previous = plan.scratch_bytes(&catalog)? * 64 * EXPERT_READ_LANES * BANKS;
            assert_eq!(actual.read_scratch_bytes,
                previous.max(cuteafd_loader::EXPERT_READ_WINDOW_BYTES * EXPERT_READ_LANES * BANKS));
        }
        Ok(())
    }

    #[test]
    fn declared_shard_residency_selection_admits_whole_experts() -> Result<()> {
        for world in 1..=8 {
            for rank in 0..world {
                let (layer, selected_world, selected_rank) = residency_selection(
                    ExpertLayer::BackboneExl3Tp { layer: 7, rank, world })?;
                assert!(matches!(layer, V41Exl3Layer::Backbone(7)));
                assert_eq!((selected_world, selected_rank), (world, rank));
            }
            assert!(residency_selection(ExpertLayer::BackboneExl3Tp { layer: 7, rank: world, world }).is_err());
        }
        for world in [0, 9] {
            assert!(residency_selection(ExpertLayer::BackboneExl3Tp { layer: 7, rank: 0, world }).is_err());
        }
        assert!(matches!(residency_selection(ExpertLayer::BackboneTp2 { layer: 7, rank: 1 })?,
                         (V41Exl3Layer::Backbone(7), 2, 1)));
        assert!(matches!(residency_selection(ExpertLayer::Backbone { layer: 7, rank: 3 })?,
                         (V41Exl3Layer::Backbone(7), 4, 3)));
        Ok(())
    }

    #[test]
    fn qwen_coordinator_tp2_keeps_legacy_residency_while_sparks_use_compact() -> Result<()> {
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(1, 4);
        write_snapshot(root.path(), &qwen4_config(1), &tensors, None);
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        for rank in 0..2 {
            let coordinator = planned_residency(&catalog, ExpertLayer::BackboneTp2 { layer: 0, rank })?;
            let legacy = catalog.exl3().unwrap().residency(V41Exl3Layer::Backbone(0), 2, rank)?;
            assert_eq!(serde_json::to_value(&coordinator)?, serde_json::to_value(&legacy)?);
            assert!(coordinator.unit_map.is_none());
            let spark = planned_residency(&catalog,
                ExpertLayer::BackboneExl3Tp { layer: 0, rank, world: 2 })?;
            assert!(spark.unit_map.is_some());
            assert_eq!(spark.aot_layout(), format!("tp2-compact-rank{rank}"));
        }
        Ok(())
    }

    #[test]
    fn shared_exl3_exposes_only_explicit_codebook_package_resolvers() {
        use cuteafd_loader::Exl3Codebook;
        assert_eq!(package_name_for("glm", &[4, 5], Exl3Codebook::Mcg), "exl3-glm-k45");
        assert_eq!(package_name_for("glm", &[4, 5], Exl3Codebook::Mul1), "exl3-glm-k45-mul1");
        let source = include_str!("exl3.rs").split("\n#[cfg(test)]").next().unwrap();
        assert!(!source.contains("fn package_name("));
        assert!(!source.contains("fn aot_layout_directory("));
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_EXL3_SNAPSHOT and CUDA memory for one TP4 layer"]
    fn compressed_layer_uploads_match_staged_bytes() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let catalog = cuteafd_loader::read_expert_catalog(std::path::Path::new(&std::env::var("CUTEAFD_EXL3_SNAPSHOT")?))?;
        let rank = std::env::var("CUTEAFD_EXL3_TEST_RANK").unwrap_or_else(|_| "2".into()).parse()?;
        let layer = ExpertLayer::Backbone { layer: 0, rank };
        let planned = Exl3Weights::plan(&catalog, layer)?;
        let (free, _) = library.cuda_memory_info()?;
        ensure!(
            free > planned.resident_bytes + 256 * 1024 * 1024,
            "insufficient free GPU memory for bounded residency test: free={free}, payload={}", planned.resident_bytes
        );
        let owned = Exl3Weights::load(&library, &catalog, layer, planned.resident_bytes)?;
        assert_eq!(owned.budget.device_staging_bytes, 0);
        let mut staging = vec![0; owned.layout.staging_bytes()];
        let mut scratch = vec![0; owned.layout.scratch_bytes(&catalog)? * 64];
        let mut checked = 0;
        // Sample first/middle/last experts, all projections, all duplicated destinations.
        for expert in [0, owned.layout.experts / 2, owned.layout.experts - 1] {
            for index in expert * JOBS_PER_EXPERT..(expert + 1) * JOBS_PER_EXPERT {
                let size = owned
                    .layout
                    .read_into(&catalog, index, &mut staging, &mut scratch)?;
                for &(buffer, offset) in &owned.layout.loads[index].destinations {
                    let mut view = owned.buffers[buffer].buffer;
                    view.ptr = unsafe { view.ptr.cast::<u8>().add(offset).cast() };
                    view.bytes = size;
                    let mut actual = vec![0; size];
                    library.copy_d2h(&mut actual, view)?;
                    ensure!(actual == staging[..size], "uploaded EXL3 bytes differ for {}", owned.layout.loads[index].tensor);
                    checked += 1;
                }
            }
        }
        for spec in &owned.layout.buffers {
            if spec.initial_words.is_empty() {
                continue;
            }
            let mut actual = vec![0; spec.bytes];
            library.copy_d2h(&mut actual, owned.buffer(&spec.name)?)?;
            let expected: Vec<_> = spec
                .initial_words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            assert_eq!(actual, expected);
        }
        println!(
            "EXL3 GPU residency: rank={rank}, width={}, bytes={}, sampled_destinations={checked}, metadata=exact",
            owned.layout.intermediate, owned.budget.resident_bytes
        );
        Ok(())
    }
}

pub(crate) fn arena<'a>(device: crate::shared::memory::device::Device<'a>, catalog: &cuteafd_loader::OfficialV41Catalog,
    layers: impl IntoIterator<Item = super::layer::ExpertLayer>) -> Result<crate::shared::memory::weight_arena::DeviceWeightArena<'a>> {
    super::weight_layout::arena(device, "exl3", regions(catalog, layers)?)
}

pub(crate) fn regions(catalog: &cuteafd_loader::OfficialV41Catalog,
    layers: impl IntoIterator<Item = super::layer::ExpertLayer>) -> Result<Vec<(String, usize)>> {
    layers.into_iter().map(|layer| {
        let layout = super::exl3::planned_residency(catalog, layer)?;
        Ok((super::weight_layout::name(layer, 0), layout.device_arena_layout()?.1))
    }).collect()
}
