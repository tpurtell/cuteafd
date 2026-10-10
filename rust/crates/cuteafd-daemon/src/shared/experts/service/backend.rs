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
    Fp8(Rc<Fp8Experts<'a>>),
}

impl<'a> Weights<'a> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Full(weights) => weights.len(),
            Self::Exl3(weights) => weights.len(),
            Self::Fp8(experts) => experts.layers.len(),
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
            Self::Fp8(experts) => Some(experts.module.info().slice),
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
                Execution::Full(execution)
            }
            Self::Exl3(weights) => Execution::Exl3(Exl3Worker::new(
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
            )?),
            Self::Fp8(experts) => {
                ensure!(Fp8Worker::workspace_bytes(config.capacity as usize) <= remaining,
                    "FP8 worker workspace exceeds the device budget");
                Execution::Fp8(Fp8Worker::new(library, experts.clone(), config.capacity)?)
            }
        })
    }
}

pub(super) enum Execution<'w, 'a> {
    Full(ExpertExecution<'w, 'a>),
    Exl3(Exl3Worker<'a>),
    Fp8(Fp8Worker<'a>),
}
impl<'w, 'a> Execution<'w, 'a> {
    pub(super) fn is_paired(&self) -> bool { matches!(self, Self::Exl3(worker) if worker.is_paired()) }
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

fn fp8_resident_budget(available: usize, resident: usize, serve_peak: usize) -> usize {
    available.saturating_sub(serve_peak.saturating_sub(resident))
}

#[cfg(test)]
mod admission_tests {
    #[test]
    fn fp8_load_transients_do_not_reduce_serve_residency() {
        // The load and serve peaks fit independently, not simultaneously.
        let (available, resident, staging, scratch, serve_overhead) = (100, 70, 25, 10, 20);
        assert!(resident + staging <= available);
        assert!(resident + scratch + serve_overhead <= available);
        assert_eq!(super::fp8_resident_budget(available, resident, resident + serve_overhead), 80);
        assert_eq!(super::fp8_resident_budget(10, 20, 40), 0);
    }
}

/// The checkpoint's own FP8 experts (E4M3 + FP32 128x128 block scales) for
/// this rank's intermediate slice, run by the `fp8-<family>` package.
pub(super) fn load_fp8<'a>(
    library: &'a NativeLibrary,
    catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig,
) -> Result<(Weights<'a>, usize)> {
    let tensors = catalog.fp8().context("FP8 residency requires the checkpoint's FP8 experts")?;
    // FP8 slices are whole 128-row blocks (TP2/TP4 of 2048); MXFP4 slices are
    // whole 32-blocks padded to 128 (MiMo V2.6 Pro: TP6, TP2), NVFP4 ones whole
    // 16-blocks padded to 128 (TP3 as well). `slice` checks it; the package
    // directory has a layout per built world.
    ensure!(config.topology.is_none() && matches!(config.world, 2 | 3 | 4 | 6),
        "FP8/MXFP4/NVFP4 experts serve implicit Spark TP2, TP3, TP4 or TP6 groups");
    tensors.slice(config.world)?;
    let directory = config.fp8_package.clone()
        .unwrap_or_else(|| crate::shared::experts::fp8::package_directory(&config.library, config.world, tensors.format()));
    let (directory, slicing) = crate::shared::experts::fp8::exact_layout(&directory, tensors, config.world, config.rank);
    let layers = config.resident_layers(catalog.routed_experts().layers)?;
    let workspace = Fp8Worker::workspace_bytes(config.capacity as usize);
    let budget = config.device_budget.checked_sub(workspace).context("FP8 worker workspace exceeds the budget")?;
    tracing::info!(rank = config.rank, world = config.world, first_layer = layers.start, layer_count = layers.len(),
        package = %directory.display(), "FP8 Spark residency plan");
    // The BF16-input sibling package, when built, lets the coordinator send unquantized rows.
    let bf16 = crate::shared::experts::fp8::bf16_sibling(&directory).filter(|d| d.is_dir());
    let layer_bytes = crate::shared::experts::fp8::Fp8Layer::bytes_for(tensors, config.world, config.rank, slicing)?;
    let resident = layer_bytes.checked_mul(layers.len()).context("FP8 resident overflow")?;
    // Per-projection pageable buffers and parallel reader scratch are temporary;
    // reserve two whole layers conservatively for the host upload transients.
    let reader_scratch = layer_bytes.checked_mul(config.world).and_then(|bytes| bytes.checked_mul(16))
        .context("FP8 reader scratch overflow")?.div_ceil(tensors.shape().experts);
    let host_staging = layer_bytes.checked_mul(2).and_then(|bytes| bytes.checked_add(reader_scratch))
        .context("FP8 host staging overflow")?;
    let peak = spark_admission_budget(config, resident, 0, host_staging, 0, workspace)?;
    admit_worker_peak(library, peak.load_peak, peak.serve_peak)?;
    // Loading buffers are gone before package scratch and worker storage exist;
    // admit their peak above, rather than subtracting both lifetimes here.
    let actual_budget = fp8_resident_budget(worker_available(library)?, resident, peak.serve_peak);
    let budget = budget.min(actual_budget);
    let experts = Fp8Experts::load_with_bf16(library, tensors, &directory, bf16.as_deref(), layers,
        config.world, config.rank, config.capacity as usize, budget)?;
    if let Some(bf16) = bf16 {
        tracing::info!(package = %bf16.display(), "FP8 experts also take BF16 rows");
    }
    tracing::info!(rank = config.rank, ?slicing, width = tensors.rank_width(config.world, config.rank, slicing)?,
        "FP8 expert slice layout");
    let resident: usize = experts.layers.len()
        * crate::shared::experts::fp8::Fp8Layer::bytes_for(tensors, config.world, config.rank, slicing)?;
    let remaining = config.device_budget.saturating_sub(resident);
    Ok((Weights::Fp8(Rc::new(experts)), remaining))
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
    let partition = Exl3Worker::partition(&exl3_directory, config.capacity, config.rank,
        config.exl3_schedule)?;
    ensure!(config.world == 4 || partition == cuteafd_loader::V41Exl3Partition::Disjoint,
        "an implicit Spark TP2/TP3 group cannot use paired TP4 artifacts");
    let workspace = Exl3Worker::plan(&exl3_directory, config.capacity, config.exl3_schedule)
        .context("EXL3 checkpoint requires matching native AOT artifacts; set --exl3-aot-dir for a custom export")?;
    let plans = config.resident_layers(catalog.routed_experts().layers)?
        .map(|layer| {
            Exl3Weights::plan_with_layout(
                catalog,
                config.selection(layer)?,
                partition,
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
        let weight = Exl3Weights::load_with_layout(
            library,
            catalog,
            config.selection(layer)?,
            remaining,
            partition,
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
