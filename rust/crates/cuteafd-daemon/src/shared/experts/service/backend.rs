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
    Exlr8(Vec<crate::shared::experts::exlr8::Exlr8Weights<'a>>),
}

impl<'a> Weights<'a> {
    pub(super) fn load_inventory(&self) -> Vec<cuteafd_core::expert_geometry::SparkAllocationTerm> {
        let budgets: Vec<_> = match self {
            Self::Full(weights) => weights.iter().map(|w| w.budget).collect(),
            Self::Exl3(weights) => weights.iter().map(|w| w.budget).collect(),
            Self::Fp8(_) => return vec![],
            Self::Exlr8(weights) => weights.iter().map(|weight| {
                let mut budget = weight.budget;
                budget.read_scratch_bytes -= weight.layout.host_bytes();
                budget
            }).collect(),
        };
        let terms = [
            ("device", "expert_load_staging", budgets.iter().map(|b| b.device_staging_bytes).max().unwrap_or(0)),
            ("pinned", "expert_load_staging", budgets.iter().map(|b| b.pinned_host_bytes).max().unwrap_or(0)),
            ("host", "expert_load_read_scratch", budgets.iter().map(|b| b.read_scratch_bytes).max().unwrap_or(0)),
        ];
        terms.into_iter().map(|(space, scope, bytes)| cuteafd_core::expert_geometry::SparkAllocationTerm {
            space: space.into(), scope: scope.into(), tensor: None, format: None, bytes: bytes as u64,
            allocations: 0,
            method: "checkpoint loader plan high-water payload; sequential layer maxima; transient, not retained; allocation count not sampled".into(),
        }).collect()
    }

    pub(super) fn retained_host_inventory(&self) -> Result<Vec<cuteafd_core::expert_geometry::SparkAllocationTerm>> {
        let Self::Exlr8(weights) = self else { return Ok(vec![]); };
        retained_host_terms(weights.iter().map(|weight| weight.layout.host_bytes()))
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Full(weights) => weights.len(),
            Self::Exlr8(weights) => weights.len(),
            Self::Exl3(weights) => weights.len(),
            Self::Fp8(experts) => experts.iter().map(|group| group.layers.len()).sum(),
        }
    }
    pub(super) fn layer_ids(&self) -> Result<Vec<usize>> {
        match self {
            Self::Full(weights) => Ok(weights.iter().map(|weight| weight.layer.layer()).collect()),
            Self::Exlr8(weights) => Ok(weights.iter().map(|weight| weight.layout.layer).collect()),
            Self::Exl3(weights) => weights.iter().map(|weight| match weight.layout.layer {
                cuteafd_loader::V41Exl3Layer::Backbone(layer) => Ok(layer),
                _ => anyhow::bail!("draft layer in resident Spark backbone"),
            }).collect(),
            Self::Fp8(experts) => {
                let mut ids = experts.iter().flat_map(|group| &group.layers).map(|layer| layer.layer).collect::<Vec<_>>();
                ids.sort_unstable();
                Ok(ids)
            },
        }
    }

    pub(super) fn loaded_residency(&self, library: &NativeLibrary, capacity: u32)
        -> Result<(u64, Vec<cuteafd_core::SparkSlice>)> {
        use cuteafd_core::SparkSlice;
        let mut resident = 0u64;
        let mut slices = Vec::with_capacity(self.len());
        match self {
            Self::Full(weights) => for weight in weights {
                let info = weight.layer.select_info(library, capacity, weight.is_nvfp4())?;
                ensure!(weight.intermediate() == info.kernel_intermediate as usize,
                    "native resident carrier width differs from installed kernel");
                let rank = match weight.layer {
                    ExpertLayer::Backbone { rank, .. } | ExpertLayer::BackboneTp2 { rank, .. }
                    | ExpertLayer::BackboneReplicatedTp { rank, .. } => rank,
                    ExpertLayer::BackboneFull { .. } => 0,
                    other => anyhow::bail!("non-native backbone worker layer {other:?}"),
                };
                resident = resident.checked_add(u64::try_from(weight.resident_bytes()?)?)
                    .context("native resident sum overflow")?;
                slices.push(SparkSlice { unit_map: None, layer: u32::try_from(weight.layer.layer())?,
                    first: u32::try_from(rank.checked_mul(info.logical_intermediate as usize)
                        .context("native slice start overflow")?)?,
                    width: info.logical_intermediate, stored_width: info.kernel_intermediate,
                    tiers: vec![], projection_counts: vec![] });
            },
            Self::Exl3(weights) => for weight in weights.iter() {
                let plan = &weight.layout;
                let layer = match plan.layer {
                    cuteafd_loader::V41Exl3Layer::Backbone(layer) => layer,
                    _ => anyhow::bail!("draft layer in resident Spark backbone"),
                };
                resident = resident.checked_add(u64::try_from(weight.resident_bytes())?)
                    .context("EXL3 resident sum overflow")?;
                slices.push(SparkSlice { unit_map: plan.unit_map.as_ref().map(|map| map.fingerprint()), layer: u32::try_from(layer)?,
                    first: u32::try_from(plan.intermediate_start)?, width: u32::try_from(plan.intermediate)?,
                    stored_width: u32::try_from(plan.intermediate)?,
                    tiers: plan.tiers.iter().copied().map(u32::try_from).collect::<Result<_, _>>()?,
                    projection_counts: plan.projection_counts.iter().map(|counts| Ok([
                        u32::try_from(counts[0])?, u32::try_from(counts[1])?, u32::try_from(counts[2])?,
                    ])).collect::<Result<_, std::num::TryFromIntError>>()? });
            },
            Self::Exlr8(weights) => for weight in weights {
                let layout = &weight.layout;
                resident = resident.checked_add(layout.resident_bytes as u64).context("EXLR8 resident sum overflow")?;
                slices.push(layout.spark_slice()?);
            },
            Self::Fp8(experts) => {
                for group in experts {
                    resident = resident.checked_add(u64::try_from(group.resident_bytes())?)
                        .context("native expert group resident overflow")?;
                    for layer in &group.layers {
                        ensure!(layer.slice.stored_width as usize == group.module.info().slice,
                            "loaded native slice differs from installed package");
                        slices.push(layer.slice.clone());
                    }
                }
            }
        }
        slices.sort_unstable_by_key(|slice| slice.layer);
        Ok((resident, slices))
    }

    pub(super) fn validate_ingress(&self, library: &NativeLibrary,
        model: &cuteafd_transport::worker_selection::ModelAdmission) -> Result<()> {
        if matches!(self, Self::Exlr8(_)) {
            ensure!(model.endpoints.iter().all(|endpoint| endpoint.ingress == cuteafd_core::SparkIngress::Fp8K32),
                "EXLR8 backend requires FP8 K32 ingress");
            return Ok(());
        }
        let (fp8, bf16) = match self {
            Self::Full(weights) => {
                let first = weights.first().context("sealed native worker has no weights")?;
                let input = first.layer.select_info(library, model.compiled_capacity, first.is_nvfp4())?.input_dtype;
                (input == cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32,
                    input == cuteafd_transport::ExpertV2Dtype::Bf16 as u32)
            }
            Self::Exl3(_) => (true, false),
            Self::Exlr8(_) => unreachable!("validated above"),
            Self::Fp8(experts) => (
                experts.iter().all(|group| group.wire_input()),
                experts.iter().all(|group| !group.wire_input() || group.bf16_module.is_some())),
        };
        for endpoint in &model.endpoints {
            let supported = supports_ingress(endpoint.ingress, fp8, bf16);
            ensure!(supported, "loaded Spark backend does not support sealed ingress {:?}", endpoint.ingress);
        }
        Ok(())
    }

    /// Logical intermediate values this worker loaded per expert, reported as
    /// part of the structured startup evidence. `None` when nothing is resident
    /// (an all-remote worker), where the value would be a claim, not a fact.
    pub(super) fn intermediate(&self) -> Option<usize> {
        match self {
            Self::Full(weights) => weights.first().map(|weight| weight.intermediate()),
            // EXL3 packs its own tiered layout; the logical intermediate is the
            // checkpoint's own value, reported by the EXL3 weights themselves.
            Self::Exl3(_) | Self::Exlr8(_) => None,
            Self::Fp8(experts) => experts.first().map(|group| group.module.info().slice)
                .filter(|width| experts.iter().all(|group| group.module.info().slice == *width)),
        }
    }
    pub(super) fn execution(
        &self,
        library: &'a NativeLibrary,
        config: &NativeExpertServiceConfig,
        remaining: usize,
        modules: &mut super::modules::WarmModules<'a>,
    ) -> Result<Execution<'_, 'a>> {
        Ok(match self {
            Self::Exlr8(weights) => Execution::Exlr8(super::exlr8::Worker::new(library,
                modules.exlr8()?, weights, config, remaining)?),
            Self::Full(weights) => {
                let mut execution = weights[0].execution(config.compiled_capacity()?, config.capacity, remaining)?;
                execution.install_native_group(config.native_group()?)?;
                execution.install_worker_timing(config.rank)?;
                Execution::Full(execution)
            }
            Self::Exl3(weights) => {
                let mut execution = Exl3Worker::with_modules(
                    library,
                    weights.clone(),
                    &{
                        let plan = &weights.first().context("EXL3 worker has no resident weights")?.layout;
                        config.exl3_directory_for(&plan.tiers, &plan.aot_layout())
                    },
                    config.capacity,
                    remaining,
                    config.exl3_schedule,
                    Some(modules.take_exl3()?),
                )?;
                execution.install_topology(config.spark_world()?.ownership_topology())?;
                Execution::Exl3(execution)
            }
            Self::Fp8(experts) => {
                let workspace = fp8_worker_workspace(config)?.checked_mul(experts.len())
                    .context("expert group workspace overflow")?;
                ensure!(workspace <= remaining, "expert worker workspace exceeds the device budget");
                let mut workers = Vec::with_capacity(experts.len());
                let mut bindings = Vec::new();
                for (group, experts) in experts.iter().enumerate() {
                    bindings.extend(experts.layers.iter().enumerate().map(|(index, layer)| (layer.layer, group, index)));
                    workers.push(Fp8Worker::new(library, experts.clone(), config.capacity)?);
                }
                let bindings = fp8_bindings(bindings)?;
                Execution::Fp8(Fp8Groups { workers, bindings, selected: 0,
                    read_bytes: experts.iter().map(|group| group.weight_read_bytes()).max().unwrap_or(0) })
            }
        })
    }
}

fn fp8_bindings(mut layers: Vec<(usize, usize, usize)>) -> Result<Vec<(usize, usize)>> {
    layers.sort_unstable_by_key(|&(layer, _, _)| layer);
    ensure!(layers.windows(2).all(|pair| pair[0].0 != pair[1].0),
        "native expert groups contain duplicate absolute layers");
    Ok(layers.into_iter().map(|(_, group, index)| (group, index)).collect())
}

fn retained_host_terms(mut bytes: impl Iterator<Item = usize>)
    -> Result<Vec<cuteafd_core::expert_geometry::SparkAllocationTerm>> {
    let bytes = bytes.try_fold(0usize, |total, bytes|
        total.checked_add(bytes).context("EXLR8 retained host inventory overflow"))?;
    Ok(vec![cuteafd_core::expert_geometry::SparkAllocationTerm {
        space: "host".into(), scope: "experts/weights/exlr8_plan".into(),
        tensor: None, format: Some("exlr8".into()), bytes: u64::try_from(bytes)?,
        allocations: 0,
        method: "checkpoint residency host payload; sum across retained layers; admitted separately from native device/pinned workspace; allocation count not sampled".into(),
    }])
}

#[cfg(test)]
#[test]
fn exlr8_retained_host_inventory_sums_layers_instead_of_transient_max() {
    let terms = retained_host_terms([7, 11, 13].into_iter()).unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].space, "host");
    assert_eq!(terms[0].bytes, 31);
    assert!(terms[0].method.contains("retained layers"));
    assert!(retained_host_terms([usize::MAX, 1].into_iter()).is_err());
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
    fn stream_idle(&self) -> Result<bool> {
        for worker in &self.workers { if !worker.stream_idle()? { return Ok(false); } }
        Ok(true)
    }
}

pub(super) fn conservative_grouping(loaded: cuteafd_transport::worker_selection::WorkerSelection,
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
struct ExpertGroupBytes { resident: usize, scratch: usize, host: usize, pinned: usize }
impl ExpertGroupBytes {
    fn add(&mut self, layer_bytes: usize, layers: usize, scratch: usize) -> Result<()> {
        self.resident = self.resident.checked_add(layer_bytes.checked_mul(layers)
            .context("expert resident overflow")?).context("expert resident sum overflow")?;
        self.scratch = self.scratch.checked_add(scratch.max(256)).context("expert scratch sum overflow")?;
        Ok(())
    }
    fn load(&mut self, load: cuteafd_loader::formats::fp8_experts::Fp8LayerLoad) {
        self.host = self.host.max(load.host);
        self.pinned = self.pinned.max(load.pinned);
    }
    fn permanent(&self) -> Result<usize> {
        self.resident.checked_add(self.scratch).context("expert permanent overflow")
    }
}

pub(super) enum Execution<'w, 'a> {
    Full(ExpertExecution<'w, 'a>),
    Exl3(Exl3Worker<'a>),
    Fp8(Fp8Groups<'a>),
    Exlr8(super::exlr8::Worker<'w, 'a>),
}
impl<'w, 'a> Execution<'w, 'a> {
    pub(super) fn installed_grouping(&self, loaded: cuteafd_transport::worker_selection::WorkerSelection,
        rank: usize) -> Result<Option<cuteafd_transport::worker_selection::InstalledExpertGrouping>> {
        match self {
            Self::Full(execution) => execution.installed_grouping(loaded, rank).map(Some),
            Self::Exl3(execution) => execution.installed_grouping(loaded, rank).map(Some),
            Self::Fp8(execution) => execution.installed_grouping(loaded, rank).map(Some),
            Self::Exlr8(execution) => execution.grouping(loaded, rank).map(Some),
        }
    }
    /// Lane-owned mode on a multi-lane rank: mapped requests return with their
    /// work queued; the owner emits once [`Self::stream_idle`] holds. EXLR8
    /// has no deferred path and is served on lane zero only.
    pub(super) fn defer(&mut self) -> Result<()> {
        match self {
            Self::Full(execution) => execution.defer(),
            Self::Exl3(execution) => execution.defer(),
            Self::Fp8(groups) => groups.workers.iter_mut().for_each(Fp8Worker::defer),
            Self::Exlr8(_) => anyhow::bail!("EXLR8 Spark workers have one execution lane"),
        }
        Ok(())
    }
    /// Whether every request this execution queued has completed.
    pub(super) fn stream_idle(&self) -> Result<bool> {
        match self {
            Self::Full(execution) => execution.stream_idle(),
            Self::Exl3(execution) => execution.stream_idle(),
            Self::Fp8(groups) => groups.stream_idle(),
            Self::Exlr8(_) => Ok(true),
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
            (Self::Exlr8(execution), Weights::Exlr8(_)) => execution.bind_layer(index),
            _ => anyhow::bail!("expert backend/weight mismatch"),
        }
    }
    /// # Safety
    /// The transport retains all mapped views until execution drains and emission completes.
    pub(super) unsafe fn execute_mapped(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, mapped: cuteafd_transport::ProtocolV2RequestDevicePayload)
        -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        match self {
            // SAFETY: the caller retains every mapped owner through the worker drain.
            Self::Exlr8(execution) => unsafe { execution.execute_mapped(request, executor_id, exchange, mapped) },
            _ => match mapped.response_slot {
                // SAFETY: existing backends receive the same retained mapped views.
                Some(slot) => unsafe { self.execute_mapped_request(request, executor_id, exchange, slot, Some(mapped.hidden_payload)) },
                None => Ok(None),
            }
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
            Self::Exlr8(_) => Ok(None),
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
            Self::Exlr8(execution) => execution.execute_host_chunks(request, executor_id, exchange,
                row_indices, max_frame_bytes, sink),
        }
    }
}

fn supports_ingress(ingress: cuteafd_core::SparkIngress, fp8: bool, bf16: bool) -> bool {
    match ingress {
        cuteafd_core::SparkIngress::Fp8K32 => fp8,
        cuteafd_core::SparkIngress::Bf16 => bf16,
        // FP8 remains legal at every row count, including the BF16 decode extent.
        cuteafd_core::SparkIngress::Bf16Decode { .. } => fp8 && bf16,
    }
}

#[cfg(test)]
mod admission_tests {
    #[test]
    fn mixed_ingress_requires_both_backends_even_at_full_decode_extent() {
        use cuteafd_core::{SparkEndpoint, SparkIngress};
        for rows in [16, 80] {
            let endpoint = SparkEndpoint { ring_depth: None, capacity: 80, ingress: SparkIngress::Bf16Decode { rows } };
            for (fp8, bf16) in [(false, false), (false, true), (true, false), (true, true)] {
                assert_eq!(super::supports_ingress(endpoint.ingress, fp8, bf16), fp8 && bf16);
            }
        }
        assert!(super::supports_ingress(SparkIngress::Fp8K32, true, false));
        assert!(!super::supports_ingress(SparkIngress::Fp8K32, false, true));
        assert!(super::supports_ingress(SparkIngress::Bf16, false, true));
        assert!(!super::supports_ingress(SparkIngress::Bf16, true, false));
    }

    #[test]
    fn group_admission_sums_retained_storage_but_maxes_load_transients() {
        let mut bytes = super::ExpertGroupBytes::default();
        use cuteafd_loader::formats::fp8_experts::Fp8LayerLoad;
        bytes.add(1000, 3, 100).unwrap();
        bytes.load(Fp8LayerLoad { resident: 1000, host: 2000, pinned: 1000 });
        bytes.add(2000, 2, 500).unwrap();
        bytes.load(Fp8LayerLoad { resident: 2000, host: 1500, pinned: 2000 });
        assert_eq!(bytes.resident, 7000);
        assert_eq!(bytes.scratch, 756);
        assert_eq!((bytes.host, bytes.pinned), (2000, 2000));
        assert_eq!(bytes.permanent().unwrap(), 7756);
        assert!(bytes.add(usize::MAX, 2, 0).is_err());
    }

    #[test]
    fn step5_mixed_load_projection_owners_fit_unchanged_100_gib_budget() {
        use cuteafd_loader::formats::fp8_experts::Fp8LayerLoad;
        let mut bytes = super::ExpertGroupBytes::default();
        // The rejected max-card receipt: remove only its old fictitious load
        // transient and retain the identical native resident/scratch owners.
        let permanent = 100_826_826_752usize;
        bytes.load(Fp8LayerLoad { resident: 1_107_566_592,
            host: 469_860_352, pinned: 369_098_752 });
        bytes.load(Fp8LayerLoad { resident: 2_214_592_512,
            host: 738_197_504, pinned: 738_197_504 });
        let headroom = cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES as usize;
        let peak = permanent + bytes.host + bytes.pinned + headroom;
        assert_eq!(peak, 104_987_576_320);
        assert!(peak < 100 << 30);
        // The pinned owner from the larger group must survive even if a later
        // group has smaller host buffers.
        bytes.load(Fp8LayerLoad { resident: 10, host: 1, pinned: 2 });
        assert_eq!((bytes.host, bytes.pinned), (738_197_504, 738_197_504));
    }
}

fn fp8_worker_workspace(config: &NativeExpertServiceConfig) -> Result<usize> {
    if config.bf16_ingress {
        let io = super::selected_spark_io(config, cuteafd_core::expert_geometry(),
            super::worker_ring_geometry(config)?)?;
        // A BF16 sibling still retains full-capacity intake, even for a mixed
        // decode-only seal. Shrinking that owner is a separate change (#292).
        Ok(usize::try_from(io.intake).context("BF16 worker intake exceeds address space")?
            .max(Fp8Worker::workspace_bytes(config.capacity as usize)))
    } else {
        Ok(Fp8Worker::workspace_bytes(config.capacity as usize))
    }
}

// Installation can narrow the checkpoint's qualified recipe, never widen it.
pub(super) fn validate_fp8_recipe(package: &str, intermediate: usize, world: usize,
    directory: &std::path::Path, rank: usize,
    slicing: cuteafd_loader::formats::fp8_experts::Slicing, width: usize) -> Result<()> {
    let recipe = cuteafd_loader::plan::experts::spark_recipe(package, intermediate, world)?;
    let root = directory.parent().context("Spark package layout has no package root")?;
    recipe.validate_installed(root, rank, slicing, width)
        .context("installed Spark packages do not match the planner recipe; padded fallback cannot widen rank storage")
}

/// The checkpoint's own FP8 experts (E4M3 + FP32 128x128 block scales) for
/// this rank's intermediate slice, run by the `fp8-<family>` package.
pub(super) fn load_fp8<'a>(
    library: &'a NativeLibrary,
    catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig,
    mut groups: Vec<super::modules::Fp8Group>,
    push: Option<&mut std::net::TcpStream>,
) -> Result<(Weights<'a>, usize)> {
    ensure!(!groups.is_empty(), "native expert worker has no resident groups");
    let workspace = fp8_worker_workspace(config)?.checked_mul(groups.len())
        .context("expert group workspace overflow")?;
    let mut totals = ExpertGroupBytes::default();
    let mut budgets = Vec::with_capacity(groups.len());
    for group in &groups {
        let mut resident = 0usize;
        for &layer in &group.layers {
            let tensors = group.tensors.for_layer_metadata(layer)?;
            let load = tensors.layer_load(layer, config.world, config.rank, group.slicing)?;
            resident = resident.checked_add(load.resident).context("expert group resident overflow")?;
            totals.load(load);
        }
        // Upload staging survives projection/group boundaries until the worker
        // releases it before execution arenas are allocated.
        totals.add(resident, 1, group.scratch)?;
        budgets.push(resident.checked_add(group.scratch).context("expert group budget overflow")?);
    }
    let permanent = totals.permanent()?;
    let peak = spark_admission_budget(config, permanent, 0, totals.pinned, totals.host, workspace)?;
    ensure!(peak.load_peak <= config.device_budget && peak.serve_peak <= config.device_budget,
        "native expert groups exceed budget: load {} serve {} budget {}",
        peak.load_peak, peak.serve_peak, config.device_budget);
    admit_worker_peak(library, peak.load_peak, peak.serve_peak)?;
    tracing::info!(groups = groups.len(), resident = totals.resident, scratch = totals.scratch,
        host = totals.host, pinned = totals.pinned, workspace, "native expert groups admitted before allocation");
    let mut owners = Vec::with_capacity(groups.len());
    for (group, budget) in groups.iter_mut().zip(budgets) {
        owners.push(Fp8Experts::prepare_with_modules(library, &group.tensors, &group.directory,
            group.bf16_directory.as_deref(), &group.layers, config.world, config.rank,
            config.capacity as usize, budget, Some(group.modules.take()
                .context("native group modules already consumed")?))?);
    }
    let pushed = push.map(|stream| super::push_fp8::load(library, catalog, config, &groups,
        totals.resident, stream)).transpose()?.flatten();
    if let Some(layers) = pushed {
        // Match the prepared package groups without ever allocating mount weights.
        for layer in layers {
            let group = groups.iter().position(|group| group.layers.contains(&layer.layer))
                .context("native push layer has no prepared package")?;
            owners[group].layers.push(layer);
        }
    } else {
        // Failed candidates have crossed the all-rank drain barrier. Reuse the
        // original modules/scratch; no second package preparation on fallback.
        for (owner, group) in owners.iter_mut().zip(&groups) {
            owner.load_mount(library, &group.tensors, &group.layers, group.slicing)?;
        }
    }
    Ok((Weights::Fp8(owners.into_iter().map(Rc::new).collect()), config.device_budget - permanent))
}

pub(super) fn load_exl3<'a>(
    library: &'a NativeLibrary,
    catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig,
    push: Option<&mut std::net::TcpStream>,
) -> Result<(Weights<'a>, usize)> {
    let exl3_directory = config.exl3_directory(catalog)?;
    Exl3Worker::validate_exports(&exl3_directory, config.capacity, config.rank,
        config.exl3_schedule)?;
    config.validate_exl3_residency(catalog, &exl3_directory)?;

    let workspace = Exl3Worker::plan(&exl3_directory, config.capacity, config.exl3_schedule)
        .context("EXL3 checkpoint requires matching native AOT artifacts; set --exl3-aot-dir for a custom export")?;
    let layers = config.resident_layers(catalog.routed_experts().layers)?;
    let plans = layers.iter().map(|&layer| {
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
    tracing::info!(rank=config.rank, world=config.world, layer_ids=?config.resident_layers(catalog.routed_experts().layers)?,
        layer_count=plans.len(), resident_bytes=resident, workspace_bytes=workspace,
        device_budget_bytes=config.device_budget, schedule=config.exl3_schedule.name(),
        "EXL3 Spark residency plan");
    let staging = plans.iter().map(|p| p.device_staging_bytes).max().unwrap_or(0);
    let pinned = plans.iter().map(|p| p.pinned_host_bytes).max().unwrap_or(0);
    let scratch = plans.iter().map(|p| p.read_scratch_bytes).max().unwrap_or(0);
    let peak = spark_admission_budget(config, resident, staging, pinned, scratch, workspace)?;
    tracing::info!(rank=config.rank, resident_bytes=resident, device_staging_bytes=staging,
        pinned_host_bytes=pinned, read_scratch_bytes=scratch, load_peak_bytes=peak.load_peak,
        serve_peak_bytes=peak.serve_peak, "EXL3 Spark admitted loader peak");
    admit_worker_peak(library, peak.load_peak, peak.serve_peak)?;
    if let Some(stream) = push {
        if let Some(weights) = super::push::load(library, catalog, config, &layers, &plans, resident, stream)? {
            return Ok((Weights::Exl3(Rc::new(weights)), config.device_budget - resident));
        }
    }
    let mut remaining = config.device_budget;
    let weights = crate::shared::experts::load_in_rank_order(&layers, config.rank, |index, layer| {
        let plan = &plans[index];
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
        Ok(weight)
    })?;
    Ok((Weights::Exl3(Rc::new(weights)), remaining))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::{formats::fp8_experts::Slicing,
        plan::experts::{InstalledSparkRecipeRefusal, SparkRecipeRefusal}};

    #[test]
    fn rotated_width_groups_bind_in_absolute_layer_order() -> Result<()> {
        let layers = [vec![3, 4, 5, 8, 9], vec![6, 7, 10]];
        let bindings = layers.iter().enumerate().flat_map(|(group, layers)|
            layers.iter().enumerate().map(move |(index, &layer)| (layer, group, index))).collect();
        assert_eq!(fp8_bindings(bindings)?,
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (0, 3), (0, 4), (1, 2)]);
        assert!(fp8_bindings(vec![(3, 0, 0), (3, 1, 0)]).is_err());
        Ok(())
    }

    #[test]
    fn fp8_load_recipe_refuses_unqualified_package_before_installation() -> Result<()> {
        let root = tempfile::tempdir()?;
        let directory = root.path().join("tp3");
        std::fs::create_dir(&directory)?;
        std::fs::write(directory.join("libcuteafd_fp8moe.so"), b"present")?;
        let error = validate_fp8_recipe("qwen4:fp8", 640, 3, &directory, 0,
            Slicing::Padded, 256).unwrap_err();
        assert!(error.downcast_ref::<SparkRecipeRefusal>().is_some(), "{error:#}");
        Ok(())
    }

    #[test]
    fn fp8_load_recipe_requires_installed_artifacts_and_exact_widths() -> Result<()> {
        let root = tempfile::tempdir()?;
        let directory = root.path().join("tp6-w384");
        for (slicing, width, expected) in [
            (Slicing::Padded, 384, "padded"),
            (Slicing::Blocks(128), 256, "width"),
            (Slicing::Blocks(128), 384, "missing"),
        ] {
            let error = validate_fp8_recipe("mimo:fp8", 2048, 6, &directory, 0, slicing, width)
                .unwrap_err();
            let refusal = error.downcast_ref::<InstalledSparkRecipeRefusal>().unwrap();
            assert!(matches!((expected, refusal),
                ("padded", InstalledSparkRecipeRefusal::PaddedExactMismatch)
                | ("width", InstalledSparkRecipeRefusal::WidthMismatch { .. })
                | ("missing", InstalledSparkRecipeRefusal::MissingArtifact(_))), "{error:#}");
        }
        Ok(())
    }
}
