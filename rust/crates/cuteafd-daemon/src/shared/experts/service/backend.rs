//! Select expert storage and execution from the checkpoint format at startup.
use super::*;
use crate::shared::experts::execution::ExpertExecution;
use crate::shared::experts::exl3::{worker::Exl3Worker, Exl3Weights};
use crate::shared::experts::fp8::{worker::Fp8Worker, Fp8Experts};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::OfficialV41Catalog;
use cuteafd_transport::{ExpertProtocolV2DeviceResponseRef, ExpertProtocolV2ResponseRef};
use std::rc::Rc;

pub(super) enum Weights<'a> {
    Full(Vec<ExpertWeights<'a>>),
    Exl3(Rc<Vec<Exl3Weights<'a>>>),
    Fp8(Vec<Rc<Fp8Experts<'a>>>),
}

impl<'a> Weights<'a> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Full(weights) => weights.len(),
            Self::Exl3(weights) => weights.len(),
            Self::Fp8(experts) => experts.iter().map(|group| group.layers.len()).sum(),
        }
    }
    /// Logical intermediate values this worker loaded per expert, reported as
    /// part of the structured startup evidence. `None` when nothing is resident
    /// (an all-remote worker), where the value would be a claim, not a fact.
    pub(super) fn intermediate(&self) -> Option<usize> {
        match self {
            Self::Full(weights) => weights.first().map(|weight| weight.intermediate()),
            // EXL3 packs its own tiered layout; the logical intermediate is the
            // checkpoint's own value, reported by the EXL3 weights themselves.
            Self::Exl3(_) => None,
            Self::Fp8(experts) => experts.first().map(|group| group.module.info().slice),
        }
    }
    pub(super) fn execution(
        &self,
        library: &'a NativeLibrary,
        config: &NativeExpertServiceConfig,
        remaining: usize,
    ) -> Result<Execution<'_, 'a>> {
        Ok(match self {
            Self::Full(weights) => {
                let mut execution = weights[0].execution(config.capacity, remaining)?;
                execution.install_native_group(config.native_group()?)?;
                execution.install_worker_timing(config.rank)?;
                Execution::Full(execution)
            }
            Self::Exl3(weights) => {
                let mut execution = Exl3Worker::new(
                    library,
                    weights.clone(),
                    &config.exl3_directory_for(
                        weights
                            .first()
                            .map(|weight| weight.layout.tiers.as_slice())
                            .unwrap_or(&[]),
                    ),
                    config.capacity,
                    remaining,
                    config.exl3_schedule,
                )?;
                execution.install_topology(config.topology)?;
                Execution::Exl3(execution)
            }
            Self::Fp8(experts) => {
                let workspace = fp8_worker_workspace(config)?.checked_mul(experts.len())
                    .context("expert group workspace overflow")?;
                ensure!(workspace <= remaining, "expert worker workspace exceeds the device budget");
                let mut workers = Vec::with_capacity(experts.len());
                let mut bindings = Vec::new();
                for (group, experts) in experts.iter().enumerate() {
                    bindings.extend((0..experts.layers.len()).map(|index| (group, index)));
                    workers.push(Fp8Worker::new(library, experts.clone(), config.capacity)?);
                }
                Execution::Fp8(Fp8Groups { workers, bindings, selected: 0,
                    read_bytes: experts.iter().map(|group| group.weight_read_bytes()).max().unwrap_or(0) })
            }
        })
    }
}

/// One shared worker per homogeneous native package, indexed by absolute layer.
/// Each worker owns its stream and input/output buffers; admission sums all of them.
pub(super) struct Fp8Groups<'a> {
    workers: Vec<Fp8Worker<'a>>,
    bindings: Vec<(usize, usize)>,
    selected: usize,
    read_bytes: usize,
}
impl<'a> Fp8Groups<'a> {
    fn bind_layer(&mut self, index: usize) -> Result<()> {
        let &(group, layer) = self.bindings.get(index).context("requested expert layer is not resident")?;
        self.workers[group].bind_layer(layer)?;
        self.selected = group;
        Ok(())
    }
    fn installed_grouping(&self, loaded: cuteafd_transport::worker_selection::WorkerSelection,
        rank: usize) -> Result<cuteafd_transport::worker_selection::InstalledExpertGrouping> {
        if self.workers.len() == 1 { return self.workers[0].installed_grouping(loaded, rank); }
        conservative_grouping(loaded, rank, self.workers.first().context("no expert workers")?.capacity(), self.read_bytes)
    }
    unsafe fn execute_mapped_request(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, slot: CuteafdDeviceBuffer, hidden: Option<CuteafdDeviceBuffer>)
        -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        // SAFETY: the selected worker retains the same input/output and stream contract.
        unsafe { self.workers[self.selected].execute_mapped_request(request, executor_id, exchange, slot, hidden) }
    }
    fn execute_host_chunks<F>(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, row_indices: &mut [u32], max_frame_bytes: usize, sink: F) -> Result<()>
        where F: FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()> {
        self.workers[self.selected].execute_host_chunks(request, executor_id, exchange, row_indices, max_frame_bytes, sink)
    }
}

fn conservative_grouping(loaded: cuteafd_transport::worker_selection::WorkerSelection,
    rank: usize, capacity: usize, read_bytes: usize)
    -> Result<cuteafd_transport::worker_selection::InstalledExpertGrouping> {
    use cuteafd_transport::worker_selection::{ExpertGroupingSelector, ExpertTrafficContract,
        InstalledExpertGrouping, InstalledExpertVariant};
    // One report spans heterogeneous packages. Charge every routed occurrence
    // at the largest expert extent, explicitly not an installed kernel tile.
    let variant = InstalledExpertVariant::from_piecewise_queries(capacity,
        ExpertTrafficContract::ConservativeRouteUpperBound, |_| Ok(1))?;
    InstalledExpertGrouping::new(loaded, rank, capacity, ExpertGroupingSelector::Fp8Nearest,
        vec![variant])?.with_weight_read_bytes(read_bytes)
}

#[derive(Default)]
struct ExpertGroupBytes { resident: usize, scratch: usize, transient: usize }
impl ExpertGroupBytes {
    fn add(&mut self, layer_bytes: usize, layers: usize, scratch: usize, transient: usize) -> Result<()> {
        self.resident = self.resident.checked_add(layer_bytes.checked_mul(layers)
            .context("expert resident overflow")?).context("expert resident sum overflow")?;
        self.scratch = self.scratch.checked_add(scratch.max(256)).context("expert scratch sum overflow")?;
        self.transient = self.transient.max(transient);
        Ok(())
    }
    fn permanent(&self) -> Result<usize> {
        self.resident.checked_add(self.scratch).context("expert permanent overflow")
    }
}

pub(super) enum Execution<'w, 'a> {
    Full(ExpertExecution<'w, 'a>),
    Exl3(Exl3Worker<'a>),
    Fp8(Fp8Groups<'a>),
}
impl<'w, 'a> Execution<'w, 'a> {
    pub(super) fn installed_grouping(&self, loaded: cuteafd_transport::worker_selection::WorkerSelection,
        rank: usize) -> Result<Option<cuteafd_transport::worker_selection::InstalledExpertGrouping>> {
        match self {
            Self::Full(execution) => execution.installed_grouping(loaded, rank).map(Some),
            Self::Exl3(execution) => execution.installed_grouping(loaded, rank).map(Some),
            Self::Fp8(execution) => execution.installed_grouping(loaded, rank).map(Some),
        }
    }
    pub(super) fn bind_layer(&mut self, weights: &'w Weights<'a>, index: usize) -> Result<()> {
        match (self, weights) {
            (Self::Full(execution), Weights::Full(weights)) => execution.bind_layer(
                weights
                    .get(index)
                    .context("requested expert layer is not resident on this Spark")?,
            ),
            (Self::Exl3(execution), Weights::Exl3(_)) => execution.bind_layer(index),
            (Self::Fp8(execution), Weights::Fp8(_)) => execution.bind_layer(index),
            _ => anyhow::bail!("expert backend/weight mismatch"),
        }
    }
    pub(super) unsafe fn execute_mapped_request(
        &mut self,
        request: &BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        slot: CuteafdDeviceBuffer,
        hidden: Option<CuteafdDeviceBuffer>,
    ) -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        match self {
            Self::Full(execution) => {
                execution.execute_mapped_request(request, executor_id, exchange, slot, hidden)
            }
            Self::Exl3(execution) => {
                execution.execute_mapped_request(request, executor_id, exchange, slot, hidden)
            }
            Self::Fp8(execution) => execution.execute_mapped_request(request, executor_id, exchange, slot, hidden),
        }
    }
    pub(super) fn execute_host_chunks<F>(
        &mut self,
        request: &BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        row_indices: &mut [u32],
        max_frame_bytes: usize,
        sink: F,
    ) -> Result<()>
    where
        F: FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    {
        match self {
            Self::Full(execution) => execution.execute_host_chunks(
                request,
                executor_id,
                exchange,
                row_indices,
                max_frame_bytes,
                sink,
            ),
            Self::Exl3(execution) => execution.execute_host_chunks(
                request,
                executor_id,
                exchange,
                row_indices,
                max_frame_bytes,
                sink,
            ),
            Self::Fp8(execution) => {
                execution.execute_host_chunks(request, executor_id, exchange, row_indices, max_frame_bytes, sink)
            }
        }
    }
}

#[cfg(test)]
mod admission_tests {
    #[test]
    fn mixed_groups_sum_persistent_weights_and_scratch_but_max_load_transients() {
        let mut bytes = super::ExpertGroupBytes::default();
        bytes.add(100, 85, 300, 700).unwrap();
        bytes.add(200, 3, 400, 1100).unwrap();
        assert_eq!((bytes.resident,bytes.scratch,bytes.transient),(9100,700,1100));
        assert_eq!(bytes.permanent().unwrap(),9800);
        assert!(bytes.add(usize::MAX,2,0,0).is_err());
        let mut bytes = super::ExpertGroupBytes::default();
        bytes.add(usize::MAX,1,256,0).unwrap();
        assert!(bytes.permanent().is_err());
    }

    #[test]
    fn mixed_worker_report_names_conservative_occurrences_not_native_grouping() {
        use cuteafd_transport::worker_selection::{WorkerIdentity,WorkerSelection,ExpertTrafficContract};
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"),"{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"),"{}").unwrap();
        let geometry = cuteafd_core::ExpertGeometry::STEP5;
        let identity = WorkerIdentity::from_snapshot(dir.path(),geometry,6,None).unwrap();
        let loaded = WorkerSelection::new(identity.clone(),&(3..91).collect::<Vec<_>>()).unwrap();
        let report = super::conservative_grouping(loaded,0,16,6291456).unwrap();
        assert_eq!(report.traffic_contract(),ExpertTrafficContract::ConservativeRouteUpperBound);
        assert_eq!(report.weight_read_bytes().unwrap(),6291456);
        for rows in [0,1,8,16] {
            assert_eq!(report.group_rows_for_input(rows,false).unwrap(),1);
            assert_eq!(report.group_rows_for_input(rows,true).unwrap(),1);
        }
        report.validate_for(&identity,0,16,3..91).unwrap();
        assert!(report.validate_for(&identity,0,16,91..92).is_err());
    }

    #[test]
    fn fp8_load_transients_do_not_reduce_serve_residency() {
        // The load and serve peaks fit independently, not simultaneously.
        let (available, resident, staging, scratch, serve_overhead) = (100, 70, 25, 10, 20);
        assert!(resident + staging <= available);
        assert!(resident + scratch + serve_overhead <= available);
        assert!(resident + staging + scratch + serve_overhead > available);
    }
}

fn fp8_worker_workspace(config: &NativeExpertServiceConfig) -> Result<usize> {
    if config.bf16_ingress {
        let io = super::selected_spark_io(config, cuteafd_core::expert_geometry(),
            super::worker_ring_geometry(config)?)?;
        Ok(usize::try_from(io.intake).context("BF16 worker intake exceeds address space")?)
    } else {
        Ok(Fp8Worker::workspace_bytes(config.capacity as usize))
    }
}

/// The checkpoint's own FP8 experts (E4M3 + FP32 128x128 block scales) for
/// this rank's intermediate slice, run by the `fp8-<family>` package.
pub(super) fn load_fp8<'a>(
    library: &'a NativeLibrary,
    catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig,
) -> Result<(Weights<'a>, usize)> {
    let tensors = catalog.fp8().context("native expert residency requires checkpoint tensors")?;
    ensure!(config.topology.is_none() && (1..=8).contains(&config.world),
        "native expert packages require implicit Spark TP1..8 groups");
    let layers = config.resident_layers(catalog.routed_experts().layers)?;
    let mut groups = Vec::<(cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors, Vec<usize>)>::new();
    for layer in layers {
        let format = tensors.layer_format(layer)?;
        if let Some((previous, ids)) = groups.last_mut().filter(|(previous, ids)|
            previous.format() == format && previous.activation(ids[0]) == tensors.activation(layer)) {
            previous.validate_layer(layer)?;
            ids.push(layer);
        } else { groups.push((tensors.for_layer(layer)?, vec![layer])); }
    }
    ensure!(!groups.is_empty(), "native expert worker has no resident layers");
    let workspace = fp8_worker_workspace(config)?.checked_mul(groups.len()).context("expert workspace overflow")?;
    let mut plans = Vec::new();
    let mut totals = ExpertGroupBytes::default();
    for (view, ids) in &groups {
        let directory = config.fp8_package.clone().unwrap_or_else(||
            crate::shared::experts::fp8::package_directory(&config.library, config.world, view.format()));
        let (directory, slicing) = crate::shared::experts::fp8::exact_layout(&directory, view, config.world, config.rank);
        let family = crate::shared::experts::fp8::package_family(cuteafd_core::expert_geometry().family(), view.format());
        let format = match view.format() {
            cuteafd_loader::formats::fp8_experts::ExpertFormat::Nvfp4 => "nvfp4",
            cuteafd_loader::formats::fp8_experts::ExpertFormat::Bf16 => "bf16",
            _ => "fp8",
        };
        if let Ok(recipe) = cuteafd_loader::plan::experts::spark_recipe(
            &format!("{family}:{format}"), view.shape().intermediate, config.world) {
            recipe.validate_installed(directory.parent().context("expert package has no root")?, config.rank,
                slicing, view.rank_width(config.world, config.rank, slicing)?)?;
        }
        if matches!(config.world, 5 | 7 | 8) {
            ensure!(slicing == cuteafd_loader::formats::fp8_experts::Slicing::Blocks(128)
                || view.shape().intermediate % (128 * config.world) == 0,
                "Spark TP{} requires whole-H128 rank-width packages", config.world);
        }
        let metadata = crate::shared::experts::fp8::package_metadata(&directory, view, ids,
            config.world, config.rank, slicing)?;
        let bf16 = metadata.info.wire_input.then(|| crate::shared::experts::fp8::bf16_sibling(&directory))
            .flatten().filter(|path| path.is_dir());
        let mut group_scratch = metadata.scratch_for(config.capacity as usize)?.max(256);
        if let Some(path) = &bf16 {
            let sibling = crate::shared::experts::fp8::package_metadata(path, view, ids, config.world, config.rank, slicing)?;
            crate::shared::experts::fp8::validate_bf16_package(&metadata.info, &sibling.info, config.capacity as usize)?;
            group_scratch = group_scratch.max(sibling.scratch_for(config.capacity as usize)?);
        }
        let bytes = crate::shared::experts::fp8::Fp8Layer::bytes_for(view, config.world, config.rank, slicing)?;
        let reader = bytes.checked_mul(config.world).and_then(|n| n.checked_mul(16))
            .context("expert reader scratch overflow")?.div_ceil(view.shape().experts);
        let transient = bytes.checked_mul(2).and_then(|n| n.checked_add(reader))
            .context("expert staging overflow")?;
        totals.add(bytes, ids.len(), group_scratch, transient)?;
        plans.push((directory, bf16, bytes, group_scratch));
    }
    let permanent = totals.permanent()?;
    let peak = spark_admission_budget(config, permanent, 0, totals.transient, 0, workspace)?;
    ensure!(peak.load_peak <= config.device_budget && peak.serve_peak <= config.device_budget,
        "native expert groups exceed budget: load {} serve {} budget {}", peak.load_peak, peak.serve_peak, config.device_budget);
    admit_worker_peak(library, peak.load_peak, peak.serve_peak)?;
    tracing::info!(groups = groups.len(), resident = totals.resident, scratch = totals.scratch,
        transient = totals.transient, workspace,
        "native expert groups admitted before allocation");
    let mut owners = Vec::new();
    for ((view, ids), (directory, bf16, bytes, group_scratch)) in groups.iter().zip(plans) {
        let budget = bytes.checked_mul(ids.len()).and_then(|n| n.checked_add(group_scratch)).context("expert group budget overflow")?;
        owners.push(Rc::new(Fp8Experts::load_with_bf16(library, view, &directory, bf16.as_deref(), ids,
            config.world, config.rank, config.capacity as usize, budget)?));
    }
    Ok((Weights::Fp8(owners), config.device_budget - permanent))
}

pub(super) fn load_exl3<'a>(
    library: &'a NativeLibrary,
    catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig,
) -> Result<(Weights<'a>, usize)> {
    let exl3_tiers: &[usize] = catalog
        .exl3()
        .map(|manifest| manifest.decoder_tiers())
        .unwrap_or(&[]);
    let exl3_directory = config.exl3_directory_for(exl3_tiers);
    Exl3Worker::validate_exports(&exl3_directory, config.capacity, config.rank,
        config.exl3_schedule)?;

    let workspace = Exl3Worker::plan(&exl3_directory, config.capacity, config.exl3_schedule)
        .context("EXL3 checkpoint requires matching native AOT artifacts; set --exl3-aot-dir for a custom export")?;
    let plans = config.resident_layers(catalog.routed_experts().layers)?
        .map(|layer| {
            Exl3Weights::plan(
                catalog,
                config.exl3_selection(layer)?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let resident = plans.iter().try_fold(0usize, |total, p| {
        total
            .checked_add(p.resident_bytes)
            .context("EXL3 resident budget overflow")
    })?;
    ensure!(
        resident
            .checked_add(workspace)
            .context("EXL3 worker budget overflow")?
            <= config.device_budget,
        "compressed EXL3 weights and workspace exceed device budget"
    );
    tracing::info!(rank=config.rank, world=config.world, first_layer=config.first_layer,
        layer_count=plans.len(), resident_bytes=resident, workspace_bytes=workspace,
        device_budget_bytes=config.device_budget, schedule=config.exl3_schedule.name(),
        "EXL3 Spark residency plan");
    let staging = plans.iter().map(|p| p.device_staging_bytes).max().unwrap_or(0);
    let pinned = plans.iter().map(|p| p.pinned_host_bytes).max().unwrap_or(0);
    let scratch = plans.iter().map(|p| p.read_scratch_bytes).max().unwrap_or(0);
    let peak = spark_admission_budget(config, resident, staging, pinned, scratch, workspace)?;
    admit_worker_peak(library, peak.load_peak, peak.serve_peak)?;
    let mut weights = Vec::with_capacity(plans.len());
    let mut remaining = config.device_budget;
    for (index, plan) in plans.iter().enumerate() {
        let layer = config.first_layer + index;
        let started = std::time::Instant::now();
        let weight = Exl3Weights::load(
            library,
            catalog,
            config.exl3_selection(layer)?,
            remaining,
        )?;
        remaining = remaining
            .checked_sub(plan.resident_bytes)
            .context("EXL3 resident budget exhausted")?;
        tracing::info!(
            rank = config.rank,
            layer,
            resident_bytes = plan.resident_bytes,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "compressed EXL3 expert layer loaded"
        );
        weights.push(weight);
    }
    Ok((Weights::Exl3(Rc::new(weights)), remaining))
}
