//! Complete routed-expert layers resident on the coordinator GPU.
//!
//! Layers `0..count` skip the Spark exchange: their experts run on the RTX
//! through the geometry's `rtx_backbone` kernels (`cuteafd_{family}_local_*`),
//! reading the same FP8 K32 wire rows and device routes the Sparks would get,
//! and the local reducer adds the shared expert. EXL3 checkpoints run the
//! coordinator's `exl3-<family>-k<tiers>/rtx-tp1` package instead.
use crate::shared::experts::exl3::{
    aot_layout_directory,
    execution::{Exl3Execution, Exl3InputFormat, Exl3RowPolicy, Exl3Workspace},
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
/// (`placement::inventory::exl3_capacities`, shared with the planner).
fn capacities(max_rows: usize) -> Vec<u32> {
    cuteafd_loader::placement::inventory::exl3_capacities(max_rows as u64).into_iter().map(|c| c as u32).collect()
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
        /// Backbone layers `first..first + layers.len()`.
        first: usize,
        layers: Vec<ExpertWeights<'a>>,
        /// dSpark stage experts (same geometry), loaded before backbone layers.
        stages: Vec<ExpertWeights<'a>>,
        states: Vec<State<'a>>,
        _scratch: DeviceAllocation<'a>,
    },
    /// One execution per package capacity over shared resident weights:
    /// stages `0..stages` first, then backbone layers `first..first + layers`.
    Exl3 {
        executions: Vec<Exl3Execution<'a>>,
        row_policy: Exl3RowPolicy,
        stages: usize,
        first: usize,
        layers: usize,
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
    pub layers: usize,
    pub peak_bytes: usize,
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
        let directories: Vec<_> = capacities.iter().map(|c| directory.join(format!("m{c}"))).collect();
        for directory in &directories { require_exl3_manifest(directory)?; }
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)? + max_rows * shape.hidden * 2
    } else {
        let mut scratch = 0;
        for capacity in capacities {
            let Ok(kernel) = library.v41_local_expert_kernel(capacity) else { return Ok(None); };
            scratch = scratch.max(usize::try_from(kernel.info().scratch_bytes)?);
        }
        let measured = scratch + max_rows * (shape.hidden * 2 + shape.topk * 8);
        let planned = cuteafd_loader::serving_capacity::deepseek_v4_native_workspace(shape.hidden as u64,
            shape.intermediate as u64, shape.experts as u64, shape.topk as u64, max_rows as u64)?;
        ensure!(measured as u64 == planned,
            "V4 local expert scratch differs from the standard export: native {measured}, planned {planned}");
        measured
    };
    Ok(Some(workspace))
}

// Shared legacy planner for GLM Flash and Qwen lazy expert windows.
pub(crate) fn plan(library: &NativeLibrary, native_lib: &Path, catalog: &OfficialV41Catalog,
    draft_stages: usize, max_layers: usize, max_rows: usize, budget: usize) -> Result<LocalPlan> {
    let shape = *catalog.routed_experts();
    let empty = || LocalPlan { layers: 0, peak_bytes: 0 };
    if max_layers <= shape.first_layer && draft_stages == 0 { return Ok(empty()); }
    let capacities = capacities(max_rows);
    let workspace = if let Some(manifest) = catalog.exl3() {
        let directory = aot_layout_directory(native_lib, manifest.decoder_tiers(), "rtx-tp1");
        let directories: Vec<_> = capacities.iter().map(|c| directory.join(format!("m{c}"))).collect();
        for directory in &directories { require_exl3_manifest(directory)?; }
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)? + max_rows * shape.hidden * 2
    } else {
        let mut scratch = 0;
        for capacity in capacities {
            let Ok(kernel) = library.v41_local_expert_kernel(capacity) else { return Ok(empty()); };
            scratch = scratch.max(usize::try_from(kernel.info().scratch_bytes)?);
        }
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
    let mut layers = 0;
    for layer in shape.first_layer..max_layers.min(shape.layers) {
        let weights = weight_plan(ExpertLayer::BackboneFull { layer })?;
        let next_peak = peak.max(resident.checked_add(weights.peak_device_bytes()?).context("local expert peak overflow")?);
        if next_peak > budget { break; }
        peak = next_peak;
        resident = resident.checked_add(weights.resident_bytes).context("local expert resident overflow")?;
        layers += 1;
    }
    Ok(if layers == 0 && draft_stages == 0 { empty() } else { LocalPlan { layers, peak_bytes: peak } })
}

impl<'a> LocalExperts<'a> {
    /// Loads the first `draft_stages` dSpark stages, then backbone layers from
    /// the model's first routed layer while they fit in `budget` bytes
    /// (leaving room for the kernels' workspace), below layer `max_layers`.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        library: &'a NativeLibrary,
        native_lib: &Path,
        catalog: &OfficialV41Catalog,
        draft_stages: usize,
        max_layers: usize,
        max_rows: usize,
        budget: usize,
        stream: *mut c_void,
    ) -> Result<Option<Self>> {
        let first = catalog.routed_experts().first_layer;
        Self::load_range(library, native_lib, catalog, draft_stages, first..max_layers.max(first), max_rows,
            budget, stream)
    }

    /// [`Self::load`] over the backbone layers `backbone` (clipped to the
    /// model's routed layers); layers load in order while they fit.
    #[allow(clippy::too_many_arguments)]
    pub fn load_range(
        library: &'a NativeLibrary,
        native_lib: &Path,
        catalog: &OfficialV41Catalog,
        draft_stages: usize,
        backbone: std::ops::Range<usize>,
        max_rows: usize,
        budget: usize,
        stream: *mut c_void,
    ) -> Result<Option<Self>> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("local-experts");
        let shape = *catalog.routed_experts();
        let backbone = backbone.start.max(shape.first_layer)..backbone.end.min(shape.layers);
        if backbone.is_empty() && draft_stages == 0 {
            return Ok(None);
        }
        let capacities = capacities(max_rows);
        if let Some(manifest) = catalog.exl3() {
            let directory = aot_layout_directory(native_lib, manifest.decoder_tiers(), "rtx-tp1");
            let row_policy = Exl3RowPolicy::active();
            let capacities = if row_policy == Exl3RowPolicy::GlmFlashK64 {
                row_policy.capacities(max_rows)?
            } else { capacities };
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
        for layer in backbone.clone() {
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
            backend: Backend::Native { first: backbone.start, layers, stages, states, _scratch: scratch },
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
        backbone: std::ops::Range<usize>,
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
        for layer in backbone.clone() {
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
            backend: Backend::Exl3 { executions, row_policy, stages: draft_stages, first: backbone.start, layers },
            reducer: library.v41_local_expert_reducer()?,
            output: DeviceAllocation::new(library, output_bytes)?,
            topk: shape.topk,
            device: library.cuda_get_device()?,
        }))
    }

    pub fn layers(&self) -> usize {
        match &self.backend {
            Backend::Native { layers, .. } => layers.len(),
            Backend::Exl3 { layers, .. } => *layers,
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
        let (first, layers, stages, states) = match &mut self.backend {
            Backend::Native { first, layers, stages, states, .. } => (*first, layers, stages, states),
            Backend::Exl3 { executions, row_policy, stages, first, layers } => {
                let index = match layer {
                    LocalLayer::Stage(n) if n < *stages => n,
                    LocalLayer::Backbone(n) if (*first..*first + *layers).contains(&n) => *stages + n - *first,
                    _ => anyhow::bail!("local expert layer {layer:?} is not resident"),
                };
                let execution = executions.iter_mut().find(|e| e.capacity() >= row_policy.required_capacity(rows))
                    .context("no local EXL3 capacity for this many rows")?;
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
            LocalLayer::Backbone(n) => n.checked_sub(first).and_then(|n| layers.get(n)),
            LocalLayer::Stage(n) => stages.get(n),
        }.with_context(|| format!("local expert layer {layer:?} is not resident"))?;
        let state = states.iter_mut().find(|s| s.kernel.info().capacity_rows as usize >= rows)
            .context("no local expert capacity for this many rows")?;
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
