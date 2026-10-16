//! Complete routed-expert layers resident on the coordinator GPU.
//!
//! An ordered inventory of absolute layers skips the Spark exchange: their experts run on the RTX
//! through the geometry's `rtx_backbone` kernels (`cuteafd_{family}_local_*`),
//! reading the same FP8 K32 wire rows and device routes the Sparks would get,
//! and the local reducer adds the shared expert. EXL3 checkpoints run the
//! coordinator's `exl3-<family>-k<tiers>/rtx-tp1` package instead.
#[cfg(test)]
#[path = "local/shared_tp2_tests.rs"]
mod shared_tp2_tests;
use crate::shared::experts::exl3::{
    aot_layout_directory,
    execution::{Exl3Execution, Exl3InputFormat, Exl3ReductionPlan, Exl3RowPolicy, Exl3Workspace},
    Exl3Weights,
};
use crate::shared::experts::layer::{ExpertLayer, ExpertWeights};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs, V41LocalExpertReducer,
    V41_EXPERT_POINTER_COUNT,
};
use cuteafd_loader::OfficialV41Catalog;
use std::{ffi::c_void, path::Path, rc::Rc};

/// The exported capacities (rows per launch) a layer runs for at most
/// `max_rows` live rows: each once, up to the first at or above `max_rows`
/// for native, non-EXL3 kernels. EXL3 also admits canonical decode scratch.
fn capacities(max_rows: usize) -> Vec<u32> {
    const EXPORTED: [u32; 6] = [1, 16, 80, 256, 1024, 4096];
    let top = EXPORTED.into_iter().find(|&c| c as usize >= max_rows).unwrap_or(4096);
    EXPORTED.into_iter().filter(|&c| c <= top).collect()
}

struct State<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; V41_EXPERT_POINTER_COUNT],
}

/// A resident expert layer: backbone layer `n`, or dSpark stage `n`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LocalLayer {
    Backbone(usize),
    Stage(usize),
}

enum Backend<'a> {
    Native {
        /// Strictly increasing absolute IDs matching the resident weight slots.
        ids: Vec<usize>,
        layers: Vec<ExpertWeights<'a>>,
        /// dSpark stage experts (same geometry), loaded before backbone layers.
        stages: Vec<ExpertWeights<'a>>,
        states: Vec<State<'a>>,
        reduction_plan: crate::shared::experts::ExpertReductionPlan,
        _scratch: DeviceAllocation<'a>,
    },
    /// One execution per package capacity over shared resident weights:
    /// stages `0..stages` first, then the exact ordered backbone inventory.
    Exl3 {
        executions: Vec<Exl3Execution<'a>>,
        row_policy: Exl3RowPolicy,
        reduction_plan: Exl3ReductionPlan,
        stages: usize,
        layers: Vec<usize>,
    },
}

pub(crate) struct LocalExperts<'a> {
    // Each EXL3 execution holds the resident weights through an `Rc`.
    backend: Backend<'a>,
    reducer: V41LocalExpertReducer<'a>,
    pub output: DeviceAllocation<'a>,
    topk: usize,
    device: i32,
}

/// Admission for complete local layers, using the same package and weight
/// plans as loading. Peak includes the transient device staging of the last
/// layer; the KV pool must leave those bytes available during onboarding.
pub(crate) struct LocalPlan {
    pub layers: Vec<usize>,
    pub peak_bytes: usize,
}

/// Policy setup may precede a lazy window's first execution. Install through
/// its admitted loader before querying the package; reuse any resident window
/// because execution grouping is package-wide, not layer-specific.
pub(crate) fn installed_window_group_rows<T>(
    resident: &std::cell::RefCell<Option<(Vec<usize>, T)>>,
    install: impl FnOnce() -> Result<()>,
    rows: usize,
    group: impl FnOnce(&T, usize) -> Result<usize>,
) -> Result<usize> {
    let missing = resident.borrow().is_none();
    if missing { install()?; }
    let window = resident.borrow();
    let (_, experts) = window.as_ref().context("installed expert window missing after installation")?;
    group(experts, rows)
}

fn validate_backbone(layers: &[usize], first: usize, end: usize) -> Result<()> {
    ensure!(layers.windows(2).all(|ids| ids[0] < ids[1])
        && layers.iter().all(|&layer| (first..end).contains(&layer)),
        "invalid local expert backbone inventory {layers:?}, routed bounds {first}..{end}");
    Ok(())
}

fn resident_index(layers: &[usize], layer: usize) -> Result<usize> {
    layers.binary_search(&layer).map_err(|_| anyhow::anyhow!("local expert layer {layer} is not resident in {layers:?}"))
}

fn exl3_index(layers: &[usize], stages: usize, layer: LocalLayer) -> Result<usize> {
    match layer {
        LocalLayer::Stage(n) if n < stages => Ok(n),
        LocalLayer::Backbone(n) => Ok(stages + resident_index(layers, n)?),
        _ => anyhow::bail!("local expert layer {layer:?} is not resident"),
    }
}

fn require_exl3_manifest(directory: &Path) -> Result<()> {
    let manifest = directory.join("v41_exl3.json");
    ensure!(manifest.is_file(), "EXL3 package manifest not found: {}", manifest.display());
    Ok(())
}

pub(crate) fn workspace_bytes(library: &NativeLibrary, native_lib: &Path, catalog: &OfficialV41Catalog,
    max_rows: usize) -> Result<Option<usize>> {
    let shape = *catalog.routed_experts();
    let capacities = capacities(max_rows);
    let workspace = if let Some(manifest) = catalog.exl3() {
        let directory = aot_layout_directory(native_lib, manifest.decoder_tiers(), "rtx-tp1");
        let directories: Vec<_> = Exl3RowPolicy::active().capacities(max_rows)?.iter()
            .map(|c| directory.join(format!("m{c}"))).collect();
        Exl3ReductionPlan::new(&directories)?;
        for directory in &directories { require_exl3_manifest(directory)?; }
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)? + max_rows * shape.hidden * 2
    } else {
        let mut scratch = 0;
        let mut infos = Vec::new();
        for capacity in capacities {
            let Ok(kernel) = library.v41_local_expert_kernel(capacity) else { return Ok(None); };
            scratch = scratch.max(usize::try_from(kernel.info().scratch_bytes)?);
            infos.push(*kernel.info());
        }
        crate::shared::experts::ExpertReductionPlan::native(library, infos)?;
        let measured = scratch + max_rows * (shape.hidden * 2 + shape.topk * 8);
        let planned = cuteafd_loader::serving_capacity::deepseek_v4_native_workspace(shape.hidden as u64,
            shape.intermediate as u64, shape.experts as u64, shape.topk as u64, max_rows as u64)?;
        ensure!(measured as u64 == planned,
            "V4 local expert scratch differs from the standard export: native {measured}, planned {planned}");
        measured
    };
    Ok(Some(workspace))
}

// Admission and loading traverse the same ordered absolute backbone IDs.
pub(crate) fn plan(library: &NativeLibrary, native_lib: &Path, catalog: &OfficialV41Catalog,
    draft_stages: usize, backbone: &[usize], max_rows: usize, budget: usize) -> Result<LocalPlan> {
    let shape = *catalog.routed_experts();
    validate_backbone(backbone, shape.first_layer, shape.layers)?;
    let empty = || LocalPlan { layers: Vec::new(), peak_bytes: 0 };
    if backbone.is_empty() && draft_stages == 0 { return Ok(empty()); }
    let capacities = capacities(max_rows);
    let workspace = if let Some(manifest) = catalog.exl3() {
        let directory = aot_layout_directory(native_lib, manifest.decoder_tiers(), "rtx-tp1");
        let directories: Vec<_> = Exl3RowPolicy::active().capacities(max_rows)?.iter()
            .map(|c| directory.join(format!("m{c}"))).collect();
        Exl3ReductionPlan::new(&directories)?;
        for directory in &directories { require_exl3_manifest(directory)?; }
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)? + max_rows * shape.hidden * 2
    } else {
        let mut scratch = 0;
        let mut infos = Vec::new();
        for capacity in capacities {
            let Ok(kernel) = library.v41_local_expert_kernel(capacity) else { return Ok(empty()); };
            scratch = scratch.max(usize::try_from(kernel.info().scratch_bytes)?);
            infos.push(*kernel.info());
        }
        crate::shared::experts::ExpertReductionPlan::native(library, infos)?;
        scratch + max_rows * (shape.hidden * 2 + shape.topk * 8)
    };
    let weight_plan = |layer| -> Result<_> {
        if catalog.exl3().is_some() { Exl3Weights::plan(catalog, layer) }
        else { ExpertWeights::plan(library, catalog, layer) }
    };
    let (mut resident, mut peak) = (workspace, workspace);
    for stage in 0..draft_stages {
        let selection = if catalog.exl3().is_some() { ExpertLayer::Dspark { stage } }
            else { ExpertLayer::BackboneFull { layer: shape.layers + stage } };
        let weights = weight_plan(selection)?;
        peak = peak.max(resident.checked_add(weights.peak_device_bytes()?).context("local expert peak overflow")?);
        ensure!(peak <= budget, "dSpark stage {stage} experts need {peak} bytes, budget is {budget}");
        resident = resident.checked_add(weights.resident_bytes).context("local expert resident overflow")?;
    }
    let mut layers = Vec::new();
    for &layer in backbone {
        let weights = weight_plan(ExpertLayer::BackboneFull { layer })?;
        let next_peak = peak.max(resident.checked_add(weights.peak_device_bytes()?).context("local expert peak overflow")?);
        if next_peak > budget { break; }
        peak = next_peak;
        resident = resident.checked_add(weights.resident_bytes).context("local expert resident overflow")?;
        layers.push(layer);
    }
    Ok(if layers.is_empty() && draft_stages == 0 { empty() } else { LocalPlan { layers, peak_bytes: peak } })
}

impl<'a> LocalExperts<'a> {
    /// Loads draft stages first, then an exact ordered backbone inventory while
    /// complete layers fit. Callers validate the returned inventory against admission.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        library: &'a NativeLibrary,
        native_lib: &Path,
        catalog: &OfficialV41Catalog,
        draft_stages: usize,
        backbone: &[usize],
        max_rows: usize,
        budget: usize,
        stream: *mut c_void,
    ) -> Result<Option<Self>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("local-experts");
        let shape = *catalog.routed_experts();
        validate_backbone(backbone, shape.first_layer, shape.layers)?;
        if backbone.is_empty() && draft_stages == 0 {
            return Ok(None);
        }
        let capacities = capacities(max_rows);
        if let Some(manifest) = catalog.exl3() {
            let directory = aot_layout_directory(native_lib, manifest.decoder_tiers(), "rtx-tp1");
            let row_policy = Exl3RowPolicy::active();
            let capacities = row_policy.capacities(max_rows)?;
            return Self::load_exl3(library, catalog, &directory, &capacities, row_policy, draft_stages, backbone, max_rows, budget);
        }
        let mut states = Vec::new();
        let mut scratch_bytes = 0usize;
        for &capacity in &capacities {
            let kernel = match library.v41_local_expert_kernel(capacity) {
                Ok(kernel) => kernel,
                Err(error) => {
                    tracing::warn!("no coordinator expert kernels for this geometry ({error:#}); \
                        build with CUTEAFD_*_EXPERT_FAMILIES=<family>:rtx_backbone to keep layers local");
                    return Ok(None);
                }
            };
            scratch_bytes = scratch_bytes.max(usize::try_from(kernel.info().scratch_bytes)?);
            states.push(State { kernel, slots: [std::ptr::null_mut(); V41_EXPERT_POINTER_COUNT] });
        }
        let reduction_plan = crate::shared::experts::ExpertReductionPlan::native(library,
            states.iter().map(|s| *s.kernel.info()))?;
        let workspace = scratch_bytes + max_rows * (shape.hidden * 2 + shape.topk * 8);
        ensure!(budget > workspace, "local experts need {workspace} workspace bytes, budget is {budget}");
        let mut remaining = budget - workspace;
        tracing::info!(budget, workspace, scratch_bytes, "loading coordinator expert layers");
        let mut stages = Vec::new();
        for stage in 0..draft_stages {
            let layer = ExpertLayer::BackboneFull { layer: shape.layers + stage };
            let weights = ExpertWeights::load(library, catalog, layer, remaining)
                .with_context(|| format!("dSpark stage {stage} experts with {remaining} bytes left"))?;
            remaining -= weights.budget().resident_bytes;
            stages.push(weights);
        }
        let mut layers = Vec::new();
        for &layer in backbone {
            let plan = ExpertWeights::plan(library, catalog, ExpertLayer::BackboneFull { layer })?;
            if plan.peak_device_bytes()? > remaining {
                break;
            }
            let weights = ExpertWeights::load(library, catalog, ExpertLayer::BackboneFull { layer }, remaining)
                .with_context(|| format!("coordinator expert layer {layer} with {remaining} bytes left"))?;
            remaining -= weights.budget().resident_bytes;
            tracing::debug!(layer, resident = weights.budget().resident_bytes, remaining, "coordinator expert layer resident");
            layers.push(weights);
        }
        if layers.is_empty() && stages.is_empty() {
            return Ok(None);
        }
        let scratch = DeviceAllocation::new(library, scratch_bytes.max(256))?;
        for state in &mut states {
            // SAFETY: the arena is exclusively owned and sized for every variant.
            unsafe {
                state.kernel.bind_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, &mut state.slots)?;
                state.kernel.initialize_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, stream)?;
                library.cuda_stream_synchronize(stream)?;
            }
        }
        Ok(Some(Self {
            backend: Backend::Native { ids: backbone[..layers.len()].to_vec(), layers, stages, states, reduction_plan, _scratch: scratch },
            reducer: library.v41_local_expert_reducer()?,
            output: DeviceAllocation::new(library, max_rows * shape.hidden * 2)?,
            topk: shape.topk,
            device: library.cuda_get_device()?,
        }))
    }

    /// EXL3 residency: whole-intermediate (world 1) layers bound to the
    /// coordinator package's capacities, which share one scratch arena.
    #[allow(clippy::too_many_arguments)]
    fn load_exl3(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        directory: &Path,
        capacities: &[u32],
        row_policy: Exl3RowPolicy,
        draft_stages: usize,
        backbone: &[usize],
        max_rows: usize,
        budget: usize,
    ) -> Result<Option<Self>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("local-experts");
        let shape = *catalog.routed_experts();
        let directories: Vec<_> = capacities.iter().map(|c| directory.join(format!("m{c}"))).collect();
        if let Some(missing) = directories.iter().find(|d| !d.join("v41_exl3.json").is_file()) {
            tracing::warn!(package = %missing.display(), "no coordinator EXL3 package for this geometry; \
                build with CUTEAFD_*_EXPERT_FAMILIES=<family>:exl3-k<tiers> to keep layers local");
            return Ok(None);
        }
        let output_bytes = max_rows * shape.hidden * 2;
        let reduction_plan = Exl3ReductionPlan::new(&directories)?;
        let workspace = Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)?
            .checked_add(output_bytes).context("local EXL3 workspace overflow")?;
        ensure!(budget > workspace, "local EXL3 experts need {workspace} workspace bytes, budget is {budget}");
        let mut remaining = budget - workspace;
        tracing::info!(budget, workspace, package = %directory.display(), "loading coordinator EXL3 expert layers");
        let mut weights = Vec::new();
        for stage in 0..draft_stages {
            let weight = Exl3Weights::load(library, catalog, ExpertLayer::Dspark { stage }, remaining)
                .with_context(|| format!("dSpark stage {stage} EXL3 experts with {remaining} bytes left"))?;
            remaining -= weight.budget.resident_bytes;
            weights.push(weight);
        }
        let mut layers = 0;
        for &layer in backbone {
            let selection = ExpertLayer::BackboneFull { layer };
            if Exl3Weights::plan(catalog, selection)?.peak_device_bytes()? > remaining {
                break;
            }
            let weight = Exl3Weights::load(library, catalog, selection, remaining)
                .with_context(|| format!("coordinator EXL3 expert layer {layer} with {remaining} bytes left"))?;
            remaining -= weight.budget.resident_bytes;
            tracing::debug!(layer, resident = weight.budget.resident_bytes, remaining, "coordinator EXL3 layer resident");
            weights.push(weight);
            layers += 1;
        }
        if weights.is_empty() {
            return Ok(None);
        }
        let weights = Rc::new(weights);
        let arena = Exl3Workspace::new(library, &directories)?;
        let mut executions = Vec::with_capacity(capacities.len());
        for (&capacity, directory) in capacities.iter().zip(&directories) {
            // SAFETY: the package is the image's own verified artifact; every
            // execution shares one arena and runs only on this struct's caller
            // stream, one launch at a time.
            let execution = unsafe {
                Exl3Execution::with_shared_workspace(library, weights.clone(), directory,
                    Exl3InputFormat::Fp8K32, Some(arena.clone()))?
            };
            ensure!(execution.capacity() == capacity as usize && execution.output_element_bytes() == 4,
                "coordinator EXL3 package must match capacity {capacity} with FP32 output");
            executions.push(execution);
        }
        Ok(Some(Self {
            backend: Backend::Exl3 { executions, row_policy, reduction_plan, stages: draft_stages, layers: backbone[..layers].to_vec() },
            reducer: library.v41_local_expert_reducer()?,
            output: DeviceAllocation::new(library, output_bytes)?,
            topk: shape.topk,
            device: library.cuda_get_device()?,
        }))
    }

    pub fn group_rows(&self, rows: usize) -> Result<usize> {
        match &self.backend {
            Backend::Native { states, reduction_plan, .. } => {
                if rows == 0 {
                    return Ok(states.last().context("no installed local expert capacity")?.kernel.group_rows());
                }
                let selected = reduction_plan.execution_index(rows, rows,
                    states.iter().map(|s| s.kernel.info().capacity_rows as usize))?;
                states[selected].kernel.group_rows_for(rows)
            }
            Backend::Exl3 { executions, row_policy, .. } => {
                let selected = if rows == 0 { executions.last() } else {
                    executions.iter().find(|e| e.capacity() >= row_policy.required_capacity(rows))
                };
                Ok(selected.context("no local EXL3 capacity for this many rows")?.group_rows())
            }
        }
    }

    pub fn layers(&self) -> &[usize] {
        match &self.backend {
            Backend::Native { ids, .. } => ids,
            Backend::Exl3 { layers, .. } => layers,
        }
    }

    pub fn stages(&self) -> usize {
        match &self.backend {
            Backend::Native { stages, .. } => stages.len(),
            Backend::Exl3 { stages, .. } => *stages,
        }
    }

    /// Runs layer `layer`'s experts for `rows` wire rows with device routes
    /// and writes routed + shared into [`Self::output`].
    ///
    /// # Safety
    /// `wire` holds `rows` FP8 K32 rows, `ids`/`weights` `rows * topk` U32
    /// expert ids and FP32 route weights, and `shared` `rows` BF16 rows on this
    /// device, all complete in stream order and unchanged until it drains.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(
        &mut self,
        layer: LocalLayer,
        rows: usize,
        wire: *mut c_void,
        ids: *mut c_void,
        weights: *mut c_void,
        shared: *mut c_void,
        stream: *mut c_void,
    ) -> Result<()> {
        let (backbone_ids, layers, stages, states, reduction_plan) = match &mut self.backend {
            Backend::Native { ids, layers, stages, states, reduction_plan, .. } =>
                (ids, layers, stages, states, reduction_plan),
            Backend::Exl3 { executions, row_policy, reduction_plan, stages, layers } => {
                let index = exl3_index(layers, *stages, layer)?;
                let selected = reduction_plan.execution_index(rows, row_policy.required_capacity(rows),
                    executions.iter().map(|e| e.capacity()))?;
                let execution = &mut executions[selected];
                let buffer = |ptr: *mut c_void, bytes: usize| CuteafdDeviceBuffer {
                    ptr, bytes, device_id: self.device, ..Default::default()
                };
                let h = cuteafd_core::expert_geometry().hidden as usize;
                let inputs = [
                    buffer(wire, rows * (h + h / 32)),
                    buffer(ids, rows * self.topk * 4),
                    buffer(weights, rows * self.topk * 4),
                ];
                // SAFETY: the caller's inputs are complete in stream order and
                // live until it drains; the execution's arena is exclusive to
                // this stream.
                unsafe {
                    let values = execution.launch_layer(index, inputs, rows, stream)?;
                    return self.reducer.finish(values.ptr.cast(), shared.cast(), self.output.buffer.ptr.cast(),
                        rows as u32, true, stream);
                }
            }
        };
        let resident = match layer {
            LocalLayer::Backbone(n) => layers.get(resident_index(backbone_ids, n)?),
            LocalLayer::Stage(n) => stages.get(n),
        }.with_context(|| format!("local expert layer {layer:?} is not resident"))?;
        let selected = reduction_plan.execution_index(rows, rows,
            states.iter().map(|s| s.kernel.info().capacity_rows as usize))?;
        let state = &mut states[selected];
        resident.bind(&state.kernel, &mut state.slots)?;
        state.slots[0] = wire;
        state.slots[1] = ids;
        state.slots[2] = weights;
        let info = state.kernel.info();
        let args = V41ExpertLaunchArgs {
            tensors: state.slots,
            num_tokens: rows as i32,
            max_rows: info.max_rows,
            scatter_rows: (rows * self.topk) as i32,
            rows_padded: info.rows_padded,
            max_tasks: info.max_tasks,
            max_phys_tiles: info.max_phys_tiles,
            max_active_clusters: info.max_active_clusters,
            stream,
        };
        // SAFETY: slots are bound to resident weights, the initialized arena
        // and this call's inputs; the stream orders everything.
        unsafe {
            state.kernel.launch(&args)?;
            self.reducer.finish(state.slots[41].cast(), shared.cast(), self.output.buffer.ptr.cast(),
                rows as u32, state.kernel.accumulates_tokens(), stream)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_backbone_inventory_preserves_absolute_ids_and_stage_namespace() -> Result<()> {
        let ids = [2, 5, 9];
        validate_backbone(&ids, 2, 10)?;
        validate_backbone(&[], 2, 10)?; // stage-only package
        for (slot, layer) in ids.into_iter().enumerate() {
            assert_eq!(resident_index(&ids, layer)?, slot);
            assert_eq!(exl3_index(&ids, 2, LocalLayer::Backbone(layer))?, 2 + slot);
        }
        assert_eq!(exl3_index(&ids, 2, LocalLayer::Stage(0))?, 0);
        assert_eq!(exl3_index(&ids, 2, LocalLayer::Stage(1))?, 1);
        assert!(exl3_index(&ids, 2, LocalLayer::Stage(2)).is_err());
        assert_eq!(exl3_index(&[], 1, LocalLayer::Stage(0))?, 0);
        assert!(exl3_index(&[], 1, LocalLayer::Backbone(0)).is_err());
        for absent in [0, 3, 4, 6, 8, 10] { assert!(resident_index(&ids, absent).is_err()); }
        for invalid in [&[2, 2][..], &[5, 2][..], &[1, 5][..], &[2, 10][..]] {
            assert!(validate_backbone(invalid, 2, 10).is_err());
        }
        Ok(())
    }

    #[test]
    fn installed_window_grouping_installs_once_before_geometry_queries() -> Result<()> {
        use crate::shared::draft::resource::{geometry_with_grouping, LayerHome};
        use std::cell::{Cell, RefCell};
        let resident = RefCell::new(None);
        let installs = Cell::new(0);
        let queries = Cell::new(0);
        let homes = [LayerHome::Dense, LayerHome::Local { slice_bytes: 10. },
            LayerHome::Local { slice_bytes: 20. }];
        let geometry = geometry_with_grouping(&homes, 288, 8, 64, 3, 0, |layer, rows| {
            installed_window_group_rows(&resident, || {
                assert_eq!((layer, rows), (1, 0));
                installs.set(installs.get() + 1);
                // A one-layer window still shares its actual package grouping.
                *resident.borrow_mut() = Some((vec![layer], vec![(1, 3), (16, 5), (80, 7)]));
                Ok(())
            }, rows, |variants, rows| {
                assert_eq!(installs.get(), 1);
                queries.set(queries.get() + 1);
                Ok(if rows == 0 { variants.last().unwrap().1 } else {
                    variants.iter().find(|(capacity, _)| *capacity >= rows).unwrap().1
                })
            })
        })?;
        assert_eq!((installs.get(), queries.get()), (1, 65 * 2));
        assert_eq!(resident.borrow().as_ref().unwrap().0, vec![1]);
        for rows in 0..=64 {
            let expected = match rows { 0 => 7, 1 => 3, 2..=16 => 5, _ => 7 };
            assert_eq!(geometry.group_rows(0, rows), 1);
            assert_eq!(geometry.group_rows(1, rows), expected);
            assert_eq!(geometry.group_rows(2, rows), expected);
        }
        Ok(())
    }

    #[test]
    fn installed_window_grouping_reuses_existing_window() -> Result<()> {
        let resident = std::cell::RefCell::new(Some((vec![12, 15], 11)));
        let group = installed_window_group_rows(&resident, || panic!("resident window must not reload"),
            23, |group, rows| { assert_eq!(rows, 23); Ok(*group) })?;
        assert_eq!(group, 11);
        assert_eq!(*resident.borrow(), Some((vec![12, 15], 11)));
        Ok(())
    }

    #[test]
    fn installed_window_grouping_propagates_install_and_query_errors() {
        let resident = std::cell::RefCell::<Option<(Vec<usize>, usize)>>::new(None);
        let error = installed_window_group_rows(&resident, || anyhow::bail!("admitted window does not fit"),
            0, |_, _| panic!("failed install must not query")).unwrap_err();
        assert_eq!(error.to_string(), "admitted window does not fit");
        assert!(resident.borrow().is_none());
        let error = installed_window_group_rows(&resident, || Ok(()), 0,
            |_, _| panic!("missing install must not query")).unwrap_err();
        assert_eq!(error.to_string(), "installed expert window missing after installation");
        *resident.borrow_mut() = Some((vec![0], 8));
        let error = installed_window_group_rows(&resident, || panic!("resident window must not reload"),
            129, |_, _| anyhow::bail!("no installed capacity")).unwrap_err();
        assert_eq!(error.to_string(), "no installed capacity");
    }

    #[test]
    fn missing_exl3_manifest_names_the_resolved_path() -> Result<()> {
        let root = tempfile::tempdir_in(std::env::current_dir()?)?;
        let directory = root.path().join("exl3/exl3-qwen4-k45/rtx-tp1/m256");
        let manifest = directory.join("v41_exl3.json");
        let error = require_exl3_manifest(&directory).unwrap_err();
        assert_eq!(error.to_string(), format!("EXL3 package manifest not found: {}", manifest.display()));
        std::fs::create_dir_all(&directory)?;
        std::fs::write(manifest, b"{}")?;
        require_exl3_manifest(&directory)
    }
}
