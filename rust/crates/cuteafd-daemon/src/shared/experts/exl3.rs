//! Owned compressed EXL3 device residency. The native execution adapter binds
//! these buffers only after loading completes and must retain them for graphs.
use super::layer::{ExpertLayer, ExpertLoadBudget, EXPERT_READ_LANES};
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
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

/// Family-qualified package name for one decoder-tier pair.
pub(crate) fn package_name(family: &str, tiers: &[usize]) -> String {
    let tag: String = tiers.iter().map(usize::to_string).collect();
    format!("exl3-{family}-k{tag}")
}

/// Resolve the checkpoint family's exact decoder-tier package and layout.
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
    root.join(package_name(family, tiers)).join(layout)
}

pub(crate) struct Exl3Weights<'a> {
    buffers: Vec<WeightBuffer>,
    _arena: ArenaOwner<'a>,
    pub(crate) layout: V41Exl3Residency,
    pub(crate) budget: ExpertLoadBudget,
}

enum ArenaOwner<'a> {
    Mount(DeviceAllocation<'a>),
    Push { _rank: std::rc::Rc<crate::shared::memory::weight_arena::ReadyArena<'a>>, region: usize },
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
        // Replicated native TP×EP groups are native-checkpoint only; EXL3
        // publications are rejected at topology admission before reaching here.
        ExpertLayer::BackboneReplicatedTp { .. } => anyhow::bail!(
            "replicated TP×EP expert groups require the official native checkpoint"
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
    pub(crate) fn resident_bytes(&self) -> usize {
        match &self._arena {
            ArenaOwner::Mount(arena) => arena.buffer.bytes,
            ArenaOwner::Push { .. } => self.budget.resident_bytes,
        }
    }

    /// Independent retained-owner census. Shared rank storage is charged once;
    /// region and kernel views point directly to its actual physical address.
    pub(crate) fn weight_receipts(
        states: &[Self],
        prefix: &str,
        location: cuteafd_loader::placement::planner::Location,
        physical_device: i32,
        lifetime: cuteafd_loader::placement::planner::Lifetime,
    ) -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::{Allocation, Domain, Receipt, View};
        use std::collections::BTreeSet;
        ensure!(!prefix.is_empty() && !states.is_empty() && location.domain == Domain::Device,
            "empty or invalid EXL3 weight census");
        let receipt = |id: String, buffer: CuteafdDeviceBuffer| -> Result<Receipt> {
            ensure!(buffer.device_id == physical_device && !buffer.ptr.is_null() && buffer.bytes > 0,
                "EXL3 retained buffer has wrong physical device or empty address/extent");
            let mut allocation = Allocation::buffer(
                id, location, buffer.bytes as u64, lifetime, Category::Experts, "EXL3 expert weights");
            allocation.alignment = 256;
            Ok(Receipt { allocation, address: buffer.ptr as u64 })
        };
        let direct_view = |id: String, buffer: CuteafdDeviceBuffer, backing: &str,
                           owner: CuteafdDeviceBuffer| -> Result<Receipt> {
            let start = buffer.ptr as usize;
            ensure!(start >= owner.ptr as usize && start.checked_add(buffer.bytes).is_some_and(|end|
                (owner.ptr as usize).checked_add(owner.bytes).is_some_and(|limit| end <= limit)),
                "EXL3 retained view exceeds physical owner");
            let mut view = receipt(id, buffer)?;
            view.allocation.view = Some(View {
                backing: backing.into(), offset: (start - owner.ptr as usize) as u64,
            });
            Ok(view)
        };
        let pushed = match &states[0]._arena { ArenaOwner::Push { _rank, .. } => Some(_rank), _ => None };
        let mut out = Vec::new();
        let mut layers = BTreeSet::new();
        let mut regions = BTreeSet::new();
        if let Some(rank) = pushed {
            out.push(receipt(format!("{prefix}.arena.rank"), rank.physical_buffer())?);
        }
        for state in states {
            let name = match state.layout.layer {
                V41Exl3Layer::Backbone(layer) => format!("target.{layer}"),
                V41Exl3Layer::Dspark(stage) => format!("draft.{stage}"),
            };
            ensure!(layers.insert(name.clone()), "duplicate retained EXL3 layer");
            let (backing, owner, region) = match (&state._arena, pushed) {
                (ArenaOwner::Mount(arena), None) => {
                    let backing = format!("{prefix}.arena.{name}");
                    out.push(receipt(backing.clone(), arena.buffer)?);
                    (backing, arena.buffer, arena.buffer)
                }
                (ArenaOwner::Push { _rank, region }, Some(rank)) => {
                    ensure!(std::rc::Rc::ptr_eq(_rank, rank) && regions.insert(*region),
                        "EXL3 census mixes physical ranks or repeats a rank region");
                    (format!("{prefix}.arena.rank"), rank.physical_buffer(), rank.region(*region)?.buffer())
                }
                _ => anyhow::bail!("EXL3 census mixes physical arena topologies"),
            };
            out.push(direct_view(format!("{prefix}.layer.{name}.region"), region, &backing, owner)?);
            ensure!(state.buffers.len() == state.layout.buffers.len() && !state.buffers.is_empty(),
                "EXL3 retained kernel buffer inventory differs");
            let mut buffers = BTreeSet::new();
            for (spec, buffer) in state.layout.buffers.iter().zip(&state.buffers) {
                ensure!(!spec.name.is_empty() && buffers.insert(&spec.name), "invalid retained EXL3 kernel identity");
                // Check both layer-region containment and direct physical backing.
                direct_view(String::new(), buffer.buffer, &backing, region)?;
                out.push(direct_view(format!("{prefix}.layer.{name}.buffer.{}", spec.name),
                    buffer.buffer, &backing, owner)?);
            }
        }
        if let Some(rank) = pushed {
            ensure!(regions.len() == rank.region_count(), "EXL3 census omits a retained rank region");
        }
        Ok(out)
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
        Ok(Self { buffers, _arena: ArenaOwner::Push { _rank: rank, region }, layout, budget })
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
                    let arena = DeviceAllocation::new(devices[rank].library, bytes)?;
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
        let layout = planned_residency(catalog, layer)?;
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
            _arena: ArenaOwner::Mount(arena),
            layout,
            budget,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconcile_weight_owners(
        allocations: Vec<cuteafd_loader::placement::planner::Allocation>,
        states: &[Exl3Weights<'_>],
        location: cuteafd_loader::placement::planner::Location,
        physical: i32,
    ) -> Result<()> {
        use cuteafd_loader::placement::planner::{self, *};
        let facts = Facts { allocations, deferred: vec![], sequential: vec![], allowances: vec![] };
        let budgets = vec![Budget {
            id: "CPU fixture".into(), locations: vec![location],
            physical_bytes: 1 << 30, logical_bytes: 1 << 30, headroom: 0,
            baseline: Baseline::Forecast { occupied: 0, source: "CPU fixture".into(), assumed: false },
        }];
        let phases = planner::evaluate(&facts, &budgets)?;
        let plan = ResolvedPlan {
            request: Request { context: 64, pool: Pin::Exact(64), onboard: Pin::Auto,
                prefill: Pin::Auto, operation: Operation::Serving },
            decisions: Decisions { attention: vec![], onboard_layers: vec![],
                expert_homes: Default::default(), embedding: None,
                prefill: PrefillShape { rows: 64, waves: 1 }, lanes: 1 },
            pool_tokens: 64, facts, budgets, phases,
        };
        let receipts = Exl3Weights::weight_receipts(states, "test.weights", location, physical, Lifetime::RESIDENT)?;
        planner::reconcile(&plan, Phase::Load, &receipts)?;
        assert!(Exl3Weights::weight_receipts(states, "test.weights", location, 1 - physical, Lifetime::RESIDENT).is_err());
        let physical: Vec<_> = receipts.iter().enumerate().filter(|(_, r)| r.allocation.view.is_none())
            .map(|(i, _)| i).collect();
        let views: Vec<_> = receipts.iter().enumerate().filter(|(_, r)| r.allocation.view.is_some())
            .map(|(i, _)| i).collect();
        for negative in 0..11 {
            let mut bad = receipts.clone();
            match negative {
                0 => { bad.pop(); },
                1 => { let mut extra = bad[physical[0]].clone(); extra.allocation.id.push_str(".extra"); bad.push(extra); },
                2 => bad[physical[0]].allocation.bytes += 1,
                3 => { bad[views[0]].allocation.bytes += 1; bad[views[1]].allocation.bytes -= 1; },
                4 => bad[physical[0]].allocation.location = Location::coordinator(7),
                5 => bad[physical[0]].address = 0,
                6 => bad[views[0]].address += 256,
                7 => bad[views[0]].allocation.view.as_mut().unwrap().offset += 256,
                8 => bad[views[0]].allocation.view.as_mut().unwrap().backing = bad[views[1]].allocation.id.clone(),
                9 => bad[views[0]].allocation.dtype = "bf16".into(),
                10 => bad[physical[0]].allocation.lifetime.first = Phase::PostTargetKv,
                _ => unreachable!(),
            }
            assert!(planner::reconcile(&plan, Phase::Load, &bad).is_err(), "negative {negative}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn retained_mount_and_shared_push_weight_arenas_reconcile_direct_views() -> Result<()> {
        use crate::shared::memory::{device::Device, push_initialize, weight_arena::*};
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::{Exl3WeightLayout, plan::testing::*};
        use cuteafd_loader::placement::planner::WeightArenaTopology;
        use cuteafd_loader::placement::planner::{Device as PlannedDevice, Domain, Lifetime, Location};
        use std::io::{Read, Seek, SeekFrom, Write};
        use std::rc::Rc;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let snapshot = fixture.directory().join("snapshot");
        let mut config = glm5_config();
        config["hidden_size"] = serde_json::json!(128);
        config["moe_intermediate_size"] = serde_json::json!(256);
        config["n_routed_experts"] = serde_json::json!(2);
        config["num_experts_per_tok"] = serde_json::json!(1);
        config["first_k_dense_replace"] = serde_json::json!(0);
        config["mlp_layer_types"] = serde_json::json!(["sparse", "sparse"]);
        config["quantization_config"] = exl3_compact(4);
        let mut tensors = Vec::new();
        for layer in 0..2 {
            for expert in 0..2 {
                for (projection, (name, n, k)) in [("gate_proj", 256, 128), ("up_proj", 256, 128),
                    ("down_proj", 128, 256)].into_iter().enumerate() {
                    tensors.extend(exl3(&format!("model.layers.{layer}.mlp.experts.{expert}.{name}"),
                        n, k, 4 + (layer + expert + projection) % 2));
                }
            }
        }
        write_snapshot(&snapshot, &config, &tensors, None);
        // Sparse header fixtures use zero payloads; only this required format
        // scalar needs a real value for the actual mount loader's validation.
        let mut file = std::fs::OpenOptions::new().read(true).write(true)
            .open(snapshot.join("model-00001-of-00001.safetensors"))?;
        let mut size = [0; 8];
        file.read_exact(&mut size)?;
        let size = u64::from_le_bytes(size);
        let mut header = vec![0; size as usize];
        file.read_exact(&mut header)?;
        let header: serde_json::Value = serde_json::from_slice(&header)?;
        for (name, tensor) in header.as_object().unwrap() {
            if name.ends_with(".mcg") {
                file.seek(SeekFrom::Start(8 + size + tensor["data_offsets"][0].as_u64().unwrap()))?;
                file.write_all(&0xcbac1fedu32.to_le_bytes())?;
            }
        }
        drop(file);
        let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
        for mode in 0..3 {
            let physical = if mode == 2 { 0 } else { 1 };
            let location = if mode == 2 { Location { device: PlannedDevice::Spark(3), domain: Domain::Device } }
                else { Location::coordinator(0) };
            let selection = |layer| match mode {
                0 => ExpertLayer::BackboneFull { layer },
                1 => ExpertLayer::BackboneTp2 { layer, rank: 1 },
                _ => ExpertLayer::BackboneExl3Tp { layer, rank: 0, world: 2 },
            };
            library.cuda_set_device(physical)?;
            fixture.configure_memory(&library, physical, 1 << 30)?;
            let planned = (0..2).map(|layer| planned_residency(&catalog, selection(layer))).collect::<Result<Vec<_>>>()?;
            let residencies = planned.iter().collect::<Vec<_>>();
            let mounted = Exl3WeightLayout::from_residencies("test.weights", &residencies,
                WeightArenaTopology::PerLayer, location, Lifetime::RESIDENT)?;
            let pushed = Exl3WeightLayout::from_residencies("test.weights", &residencies,
                WeightArenaTopology::SharedRank, location, Lifetime::RESIDENT)?;
            assert_eq!(mounted.bytes()?, pushed.bytes()?);
            let states = (0..2).map(|layer| Exl3Weights::load(&library, &catalog,
                selection(layer), 1 << 30)).collect::<Result<Vec<_>>>()?;
            reconcile_weight_owners(mounted.allocations, &states, location, physical)?;
            let sizes = planned.iter().map(|r| r.device_arena_layout().map(|(_, bytes)| bytes))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let bytes = pushed.bytes()? as usize;
            let arena = ArenaLayout::new(sizes, bytes)?;
            let expected = ExpectedCoverage::spans(&[(0, bytes as u64)]);
            let device = Device { library: &library, id: physical };
            let mut loading = LoadingArena::new(device, arena.clone(), expected)?;
            // Deliver each actual mounted image into the corresponding rank
            // region. Protocol/source coverage is tested elsewhere, not here.
            for (region, state) in states.iter().enumerate() {
                let ArenaOwner::Mount(owner) = &state._arena else { unreachable!() };
                let mut image = vec![0; owner.buffer.bytes];
                library.copy_d2h(&mut image, owner.buffer)?;
                let start = arena.region(region)?.start;
                library.copy_h2d(loading.destination(start, image.len())?, &image)?;
            }
            loading.copied(0, bytes as u64)?;
            let layers = planned.iter().enumerate().map(|(region, residency)|
                push_initialize::Layer::Exl3 { region, residency }).collect::<Vec<_>>();
            let ready = Rc::new(push_initialize::initialize(loading.done()?, &layers, None, 0)?);
            let mut pushed_states = planned.into_iter().enumerate().map(|(region, layout)| {
                let budget = budget(&catalog, &layout)?;
                Exl3Weights::from_push(ready.clone(), region, layout, budget)
            }).collect::<Result<Vec<_>>>()?;
            reconcile_weight_owners(pushed.allocations, &pushed_states, location, physical)?;
            let receipts = Exl3Weights::weight_receipts(&pushed_states, "test.weights", location, physical, Lifetime::RESIDENT)?;
            assert_eq!(receipts.iter().filter(|r| r.allocation.view.is_none()).count(), 1);
            let second = receipts.iter().find(|r| r.allocation.id == "test.weights.layer.target.1.region").unwrap();
            assert_eq!(second.allocation.view.as_ref().unwrap().offset, arena.region(1)?.start as u64);
            assert!(second.allocation.view.as_ref().unwrap().offset > 0);
            assert!(Exl3Weights::weight_receipts(&pushed_states[..1], "test.weights", location, physical, Lifetime::RESIDENT).is_err());
            let buffer = pushed_states[1].buffers[0].buffer;
            pushed_states[1].buffers[0].buffer.ptr = std::ptr::null_mut();
            assert!(Exl3Weights::weight_receipts(&pushed_states, "test.weights", location, physical, Lifetime::RESIDENT).is_err());
            pushed_states[1].buffers[0].buffer = buffer;
            drop(pushed_states);
            drop(ready);
            drop(states);
            assert_eq!(library.cuda_memory_info()?, (1 << 30, 1 << 30));
        }
        Ok(())
    }

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
