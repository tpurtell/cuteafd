//! Owned compressed EXL3 device residency. The native execution adapter binds
//! these buffers only after loading completes and must retain them for graphs.
use super::layer::{ExpertLayer, ExpertLoadBudget, EXPERT_READ_LANES};
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{OfficialV41Catalog, V41Exl3Layer, V41Exl3Partition, V41Exl3Residency};

const JOBS_PER_EXPERT: usize = 9;
const BANKS: usize = 2;
pub(crate) mod execution;
pub(crate) mod worker;

/// EXL3 package directory name for one decoder-tier family: V4.1 keeps
/// `exl3-k<tiers>`; other expert geometries ship as `exl3-<family>-k<tiers>`
/// (for example `exl3-dsv4p-k23`), so one image can carry several models.
pub(crate) fn package_name(family: &str, tiers: &[usize]) -> String {
    let tag: String = tiers.iter().map(usize::to_string).collect();
    if family == "v41" {
        format!("exl3-k{tag}")
    } else {
        format!("exl3-{family}-k{tag}")
    }
}

/// Resolve one EXL3 AOT layout directory for the running checkpoint.
///
/// v7 images ship decoder-tier families side by side under
/// `<libdir>/exl3/exl3-k<tiers>/<layout>`; older images keep a single
/// family at the legacy `<libdir>/exl3/<layout>`. The checkpoint's decoder
/// tiers select the matching family first; otherwise the legacy location
/// is returned and the downstream module-info bits check reports any
/// mismatch with the checkpoint. Other expert geometries resolve only their
/// own `exl3-<family>-k<tiers>` package, never the legacy V4.1 location.
pub(crate) fn aot_layout_directory(
    native_lib: &std::path::Path,
    tiers: &[usize],
    layout: &str,
) -> std::path::PathBuf {
    let root = native_lib
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("exl3");
    let family = cuteafd_core::expert_geometry().family().unwrap_or("unknown");
    if !tiers.is_empty() {
        let package = root.join(package_name(family, tiers)).join(layout);
        if package.is_dir() || family != "v41" {
            return package;
        }
    }
    root.join(layout)
}

pub(crate) struct Exl3Weights<'a> {
    buffers: Vec<WeightBuffer>,
    _arena: DeviceAllocation<'a>,
    pub(crate) layout: V41Exl3Residency,
    pub(crate) budget: ExpertLoadBudget,
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
        // Implicit compact TP shard on a compressed checkpoint: the shard count
        // rides on the layer, so the whole-block H128 partition drives residency
        // directly. One-, three- and six-rank groups reach this layer; the two-rank
        // compact profile keeps `BackboneTp2` untouched above.
        ExpertLayer::BackboneExl3Tp { layer, rank, world } => {
            ensure!(
                matches!(world, 1 | 3 | 6) && rank < world,
                "EXL3 Spark shards support implicit one-, three- and six-rank groups, got TP{world} rank {rank}"
            );
            (V41Exl3Layer::Backbone(layer), world, rank)
        }
        ExpertLayer::Dspark { stage } => (V41Exl3Layer::Dspark(stage), 1, 0),
        ExpertLayer::DsparkTp2 { .. } => anyhow::bail!("TP2 dSpark EXL3 is not implemented"),
        // Replicated native TP×EP groups are native-checkpoint only; EXL3
        // publications are rejected at topology admission before reaching here.
        ExpertLayer::BackboneReplicatedTp { .. } => anyhow::bail!(
            "replicated TP×EP expert groups require the official native checkpoint"
        ),
    })
}

fn layout(catalog: &OfficialV41Catalog, layer: ExpertLayer, partition: V41Exl3Partition) -> Result<V41Exl3Residency> {
    let (layer, world, rank) = residency_selection(layer)?;
    catalog
        .exl3()
        .context("EXL3 residency requires a routed EXL3 checkpoint")?
        .residency_with_layout(layer, world, rank, partition)
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
    // Batch 64 source rows per pread, matching the native FP4 loader policy.
    let scratch = plan
        .scratch_bytes(catalog)?
        .checked_mul(64)
        .context("EXL3 scratch overflow")?;
    Ok(ExpertLoadBudget {
        resident_bytes: arena_layout(plan)?.1,
        device_staging_bytes: 0,
        pinned_host_bytes: staging.max(init_bytes) * EXPERT_READ_LANES * BANKS,
        read_scratch_bytes: scratch * EXPERT_READ_LANES * BANKS,
    })
}

impl<'a> Exl3Weights<'a> {
    pub(crate) fn plan(
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<ExpertLoadBudget> {
        Self::plan_with_layout(catalog, layer, V41Exl3Partition::Disjoint)
    }

    pub(crate) fn plan_with_layout(
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        partition: V41Exl3Partition,
    ) -> Result<ExpertLoadBudget> {
        budget(catalog, &layout(catalog, layer, partition)?)
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
        layers: std::ops::Range<usize>,
        available: [usize; 2],
    ) -> Result<[crate::shared::memory::device::DeviceOwner<'a, Vec<Self>>; 2]> {
        use super::paired_load::{self, Fences, Projection};
        let started = std::time::Instant::now();
        let mut weights = [devices[0].own(|| Ok(Vec::new()))?, devices[1].own(|| Ok(Vec::new()))?];
        let mut remaining = available;
        let mut full_bytes = 0;
        let mut pinned_admitted = 0;
        for layer in layers.clone() {
            let mut pinned = 0;
            for rank in 0..2 {
                let plan = layout(catalog, ExpertLayer::BackboneTp2 { layer, rank }, V41Exl3Partition::Disjoint)?;
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
                    let arena = DeviceAllocation::new(devices[rank].library, bytes)?;
                    let buffers = plan.buffers.iter().zip(offsets).map(|(spec, offset)| {
                        let mut buffer = arena.buffer;
                        // SAFETY: checked arena layout reserves every aligned buffer.
                        unsafe { buffer.ptr = buffer.ptr.cast::<u8>().add(offset).cast(); }
                        buffer.bytes = spec.bytes.max(16);
                        WeightBuffer { buffer }
                    }).collect();
                    weights[rank].push(Self { buffers, _arena: arena, layout: plan, budget });
                    Ok(())
                })?;
            }
            pinned_admitted = pinned_admitted.max(pinned);
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
        for (index, layer) in layers.enumerate() {
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
                        ensure!(*name == weights[1][index].layout.loads[job].tensor,
                            "paired EXL3 projection mismatch");
                        let projection = Projection::new(catalog, name.clone(), offset,
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
        Self::load_with_layout(library, catalog, layer, available_device_bytes, V41Exl3Partition::Disjoint)
    }

    pub(crate) fn load_with_layout(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
        partition: V41Exl3Partition,
    ) -> Result<Self> {
        let layout = layout(catalog, layer, partition)?;
        let budget = budget(catalog, &layout)?;
        ensure!(
            budget.peak_device_bytes()? <= available_device_bytes,
            "EXL3 layer needs {} device bytes, budget is {available_device_bytes}",
            budget.resident_bytes
        );
        let (offsets, bytes) = arena_layout(&layout)?;
        let arena = DeviceAllocation::new(library, bytes)?;
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
                    .map(|_| HostAllocation::new(library, per_host))
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
            _arena: arena,
            layout,
            budget,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_shard_residency_selection_admits_whole_experts() -> Result<()> {
        for world in [1, 3, 6] {
            for rank in 0..world {
                let (layer, selected_world, selected_rank) = residency_selection(
                    ExpertLayer::BackboneExl3Tp { layer: 7, rank, world })?;
                assert!(matches!(layer, V41Exl3Layer::Backbone(7)));
                assert_eq!((selected_world, selected_rank), (world, rank));
            }
            assert!(residency_selection(ExpertLayer::BackboneExl3Tp { layer: 7, rank: world, world }).is_err());
        }
        for world in [0, 2, 4, 5, 7] {
            assert!(residency_selection(ExpertLayer::BackboneExl3Tp { layer: 7, rank: 0, world }).is_err());
        }
        assert!(matches!(residency_selection(ExpertLayer::BackboneTp2 { layer: 7, rank: 1 })?,
                         (V41Exl3Layer::Backbone(7), 2, 1)));
        assert!(matches!(residency_selection(ExpertLayer::Backbone { layer: 7, rank: 3 })?,
                         (V41Exl3Layer::Backbone(7), 4, 3)));
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_EXL3_SNAPSHOT and CUDA memory for one TP4 layer"]
    fn compressed_layer_uploads_match_staged_bytes() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let catalog = cuteafd_loader::read_official_v41_catalog(
            cuteafd_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("CUTEAFD_EXL3_SNAPSHOT")?),
        )?;
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
