//! One rank's native FP8-K32 TP2 experts, using V4.1's rank launch sequence.
use super::{
    ExpertInput, PartialDtype, PartialLayout, Routes, RtxExpertLayer, RtxSelectionPolicy, RtxShard,
};
use crate::shared::experts::layer::{ExpertFormat, ExpertLayer, ExpertLoadBudget, ExpertWeights};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event, Stream};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::{expert_geometry, ExpertGeometry};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs, V41ExpertOutputKind,
    V41Tp2ExpertReducer,
};
use std::{ffi::c_void, rc::Rc};

pub(crate) fn kernel_capacities(max_rows: usize) -> Result<Vec<u32>> {
    ensure!((1..=4096).contains(&max_rows), "invalid TP2 rank capacity");
    let capacities = [1, 16, 80, 256, 1024, 4096];
    let compiled = capacities
        .into_iter()
        .find(|&c| c as usize >= max_rows)
        .unwrap();
    Ok(capacities.into_iter().filter(|&c| c <= compiled).collect())
}

pub(super) fn output_bytes(geometry: ExpertGeometry, rows: usize) -> Result<usize> {
    rows.checked_mul(geometry.hidden as usize)
        .and_then(|n| n.checked_mul(4))
        .context("TP2 output extent overflow")
}

pub(super) fn validate_layers(layers: &[usize], rank: u8) -> Result<()> {
    ensure!(
        rank < 2 && !layers.is_empty()
            && layers.windows(2).all(|ids| ids[0] < ids[1])
            && layers.iter().all(|&layer| layer < expert_geometry().layers as usize),
        "invalid TP2 rank or ordered absolute layer IDs"
    );
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeOutputPolicy {
    TokenSums,
    KernelPlanes,
}

struct RankState<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; 44],
}

pub(super) trait CompletionEvent {
    fn record(&self, stream: *mut c_void) -> Result<()>;
    fn drain(&self, stream: *mut c_void) -> Result<()>;
    fn wait(&self) -> Result<()>;
}

impl CompletionEvent for Event<'_> {
    fn record(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: enqueue's caller retains its rank-local stream through recording.
        self.device
            .run(|| unsafe { self.device.library.cuda_event_record(self.raw, stream) })
    }
    fn drain(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: the caller already published every peer abort word and retains the stream.
        self.device
            .run(|| unsafe { self.device.library.cuda_stream_synchronize(stream) })
    }
    fn wait(&self) -> Result<()> {
        // SAFETY: the retained event outlives every queued operation using this arena.
        self.device
            .run(|| unsafe { self.device.library.cuda_event_synchronize(self.raw) })
    }
}

/// Must be the first owner field: drains before weights/scratch/output release.
pub(super) struct RankCompletion<E: CompletionEvent> {
    event: E,
    queued: bool,
    failed_record_stream: Option<usize>,
}

impl<E: CompletionEvent> RankCompletion<E> {
    pub(super) fn new(event: E) -> Self {
        Self {
            event,
            queued: false,
            failed_record_stream: None,
        }
    }
    pub(super) fn finish(
        &mut self,
        result: Result<()>,
        stream: *mut c_void,
        abort: &mut dyn FnMut(),
    ) -> Result<()> {
        let recorded = self.event.record(stream);
        self.queued |= recorded.is_ok();
        if recorded.is_err() {
            self.failed_record_stream = Some(stream as usize);
        }
        let result = result.and(recorded);
        if result.is_err() {
            // An unmatched peer wait may precede this launch. Never drain here:
            // the engine publishes abort first, then drains every consumer.
            abort();
        }
        result
    }
}

impl<E: CompletionEvent> Drop for RankCompletion<E> {
    fn drop(&mut self) {
        if let Some(stream) = self.failed_record_stream {
            crate::shared::decode_graph::fatal_drain(
                self.event.drain(stream as *mut c_void),
                "RTX failed completion record Drop",
            );
        } else if self.queued {
            crate::shared::decode_graph::fatal_drain(self.event.wait(), "RTX rank completion Drop");
        }
    }
}

/// Exclusive to one rank's serialized stream; destroy graphs before this owner.
pub(crate) struct NativeTp2<'a> {
    completion: RankCompletion<Event<'a>>,
    device: Device<'a>,
    rank: u8,
    layers: Vec<usize>,
    geometry: ExpertGeometry,
    max_rows: usize,
    workspace_bytes: usize,
    states: DeviceOwner<'a, Vec<RankState<'a>>>,
    selection: RtxSelectionPolicy,
    output_policy: NativeOutputPolicy,
    nvfp4: bool,
    reduction_plan: Option<crate::shared::experts::ExpertReductionPlan>,
    weights: Rc<DeviceOwner<'a, Vec<ExpertWeights<'a>>>>,
    _scratch: Allocation<'a>,
    output: Allocation<'a>,
    reducer: V41Tp2ExpertReducer<'a>,
}

#[derive(Clone)]
pub(crate) struct NativeTp2Weights<'a> {
    device: Device<'a>,
    rank: u8,
    layers: Vec<usize>,
    max_rows: usize,
    workspace_bytes: usize,
    weights: Rc<DeviceOwner<'a, Vec<ExpertWeights<'a>>>>,
    selection: RtxSelectionPolicy,
    output_policy: NativeOutputPolicy,
}

impl<'a> NativeTp2Weights<'a> {
    pub(crate) fn execution(&self) -> Result<NativeTp2<'a>> {
        NativeTp2::execution(
            self.device,
            self.rank,
            self.layers.clone(),
            self.max_rows,
            self.workspace_bytes,
            self.weights.clone(),
            self.selection,
            self.output_policy,
        )
    }
    pub(crate) fn device(&self) -> Device<'a> {
        self.device
    }
    pub(crate) fn layers(&self) -> &[usize] {
        &self.layers
    }
    pub(crate) fn max_rows(&self) -> usize {
        self.max_rows
    }
    pub(crate) fn uses_bf16_input(&self) -> bool {
        self.weights[0].is_nvfp4()
    }
}

impl<'a> NativeTp2<'a> {
    /// CPU-only load plan. Solver half cost is resident + staging from this
    /// plan; workspace is charged once per rank/lane, not once per layer.
    pub(crate) fn plan_layer(
        library: &NativeLibrary,
        catalog: &cuteafd_loader::OfficialV41Catalog,
        layer: usize,
        rank: u8,
    ) -> Result<ExpertLoadBudget> {
        validate_layers(
            &[layer],
            rank,
        )?;
        ensure!(
            !ExpertFormat::of(catalog).is_nvfp4() && !ExpertFormat::of(catalog).is_exl3(),
            "NativeTp2 requires native MXFP4; use the checkpoint's native-format backend"
        );
        ExpertWeights::plan(
            library,
            catalog,
            ExpertLayer::BackboneTp2 {
                layer,
                rank: rank as usize,
            },
        )
    }

    /// Max kernel scratch across unique compiled capacities plus one FP32
    /// `[max_rows, hidden]` output. Weight and module bytes are separate.
    pub(crate) fn workspace_bytes_for(library: &NativeLibrary, max_rows: usize) -> Result<usize> {
        Self::workspace_bytes_for_selection(library, max_rows, RtxSelectionPolicy::CanonicalDecode)
    }

    pub(crate) fn workspace_bytes_for_selection(
        library: &NativeLibrary,
        max_rows: usize,
        selection: RtxSelectionPolicy,
    ) -> Result<usize> {
        Self::workspace_bytes_for_format(
            library,
            max_rows,
            selection,
            NativeOutputPolicy::TokenSums,
            false,
        )
    }

    pub(crate) fn workspace_bytes_for_format(
        library: &NativeLibrary,
        max_rows: usize,
        selection: RtxSelectionPolicy,
        output_policy: NativeOutputPolicy,
        nvfp4: bool,
    ) -> Result<usize> {
        ensure!(
            !nvfp4
                || (output_policy == NativeOutputPolicy::KernelPlanes
                    && selection == RtxSelectionPolicy::FirstCapacity),
            "NVFP4 W4A4 requires BF16 route planes and explicit FirstCapacity selection"
        );
        let layer = ExpertLayer::BackboneTp2 { layer: 0, rank: 0 };
        let infos = kernel_capacities(max_rows)?
            .into_iter()
            .map(|capacity| layer.select_info(library, capacity, nvfp4))
            .collect::<Result<Vec<_>>>()?;
        if selection == RtxSelectionPolicy::CanonicalDecode && !nvfp4 {
            crate::shared::experts::ExpertReductionPlan::native(library, infos.iter().copied())?;
        }
        let scratch = infos
            .iter()
            .map(|info| usize::try_from(info.scratch_bytes))
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap();
        scratch
            .checked_add(Self::partial_bytes(
                expert_geometry(),
                max_rows,
                output_policy,
                nvfp4,
            )?)
            .context("TP2 workspace overflow")
    }

    fn partial_bytes(
        geometry: ExpertGeometry,
        rows: usize,
        output_policy: NativeOutputPolicy,
        nvfp4: bool,
    ) -> Result<usize> {
        let columns = if output_policy == NativeOutputPolicy::KernelPlanes {
            geometry.topk as usize
        } else {
            1
        };
        rows.checked_mul(geometry.hidden as usize)
            .and_then(|n| n.checked_mul(columns))
            .and_then(|n| n.checked_mul(if nvfp4 { 2 } else { 4 }))
            .context("TP2 partial output extent overflow")
    }

    fn admit(
        device: Device<'a>,
        catalog: &cuteafd_loader::OfficialV41Catalog,
        rank: u8,
        layers: &[usize],
        max_rows: usize,
        budget: usize,
        selection: RtxSelectionPolicy,
        output_policy: NativeOutputPolicy,
    ) -> Result<usize> {
        validate_layers(layers, rank)?;
        let format = ExpertFormat::of(catalog);
        ensure!(!format.is_exl3(), "native TP2 excludes EXL3 checkpoints");
        let workspace = Self::workspace_bytes_for_format(
            device.library,
            max_rows,
            selection,
            output_policy,
            format.is_nvfp4(),
        )?;
        let mut resident = workspace;
        for &layer in layers {
            let plan = ExpertWeights::plan(
                device.library,
                catalog,
                ExpertLayer::BackboneTp2 {
                    layer,
                    rank: rank as usize,
                },
            )?;
            resident = resident
                .checked_add(plan.resident_bytes)
                .context("TP2 resident overflow")?;
            ensure!(
                resident
                    .checked_add(plan.device_staging_bytes)
                    .context("TP2 peak overflow")?
                    <= budget,
                "TP2 weights + workspace + staging exceed budget {budget}"
            );
        }
        Ok(workspace)
    }

    /// Load each projection once into shared pinned banks, then pack each half
    /// on its owning GPU. Both rank budgets precede allocation and payload reads.
    pub(crate) fn load_pair(
        devices: [Device<'a>; 2],
        catalog: &cuteafd_loader::OfficialV41Catalog,
        layers: Vec<usize>,
        max_rows: usize,
        budgets: [usize; 2],
    ) -> Result<[Self; 2]> {
        Self::load_pair_with_selection(
            devices,
            catalog,
            layers,
            max_rows,
            budgets,
            RtxSelectionPolicy::CanonicalDecode,
        )
    }

    pub(crate) fn load_pair_with_selection(
        devices: [Device<'a>; 2],
        catalog: &cuteafd_loader::OfficialV41Catalog,
        layers: Vec<usize>,
        max_rows: usize,
        budgets: [usize; 2],
        selection: RtxSelectionPolicy,
    ) -> Result<[Self; 2]> {
        ensure!(
            ExpertFormat::of(catalog) == ExpertFormat::Native,
            "NativeTp2 requires MXFP4; choose the native-format executor"
        );
        Self::load_pair_with_output(
            devices,
            catalog,
            layers,
            max_rows,
            budgets,
            selection,
            NativeOutputPolicy::TokenSums,
        )
    }

    pub(crate) fn load_pair_with_output(
        devices: [Device<'a>; 2],
        catalog: &cuteafd_loader::OfficialV41Catalog,
        layers: Vec<usize>,
        max_rows: usize,
        budgets: [usize; 2],
        selection: RtxSelectionPolicy,
        output_policy: NativeOutputPolicy,
    ) -> Result<[Self; 2]> {
        let owners = Self::load_pair_weights(
            devices,
            catalog,
            layers,
            max_rows,
            budgets,
            selection,
            output_policy,
        )?;
        Ok([owners[0].execution()?, owners[1].execution()?])
    }

    pub(crate) fn load_pair_weights(
        devices: [Device<'a>; 2],
        catalog: &cuteafd_loader::OfficialV41Catalog,
        layers: Vec<usize>,
        max_rows: usize,
        budgets: [usize; 2],
        selection: RtxSelectionPolicy,
        output_policy: NativeOutputPolicy,
    ) -> Result<[NativeTp2Weights<'a>; 2]> {
        ensure!(
            devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid native TP2 device pair"
        );
        let label = format!(
            "experts/TP2 expert layer halves {layers:?}"
        );
        let _memory_scope = cuteafd_ffi::memory_ledger::scope_owned(&label);
        let workspaces = [
            Self::admit(
                devices[0],
                catalog,
                0,
                &layers,
                max_rows,
                budgets[0],
                selection,
                output_policy,
            )?,
            Self::admit(
                devices[1],
                catalog,
                1,
                &layers,
                max_rows,
                budgets[1],
                selection,
                output_policy,
            )?,
        ];
        let available = [budgets[0] - workspaces[0], budgets[1] - workspaces[1]];
        // Diagnostic A/B arm retains the previous loader with real I/O counters.
        let weights = if ExpertFormat::of(catalog).is_nvfp4()
            || std::env::var("CUTEAFD_TP2_SHARED_READ").as_deref() == Ok("0")
        {
            let mut remaining = available;
            let mut weights = [
                devices[0].own(|| Ok(Vec::new()))?,
                devices[1].own(|| Ok(Vec::new()))?,
            ];
            for &layer in &layers {
                // SAFETY: fresh, unaliased GPU owners only; both workers drain
                // before returning and DeviceOwner destroys on its owning GPU.
                let pair = unsafe {
                    cuteafd_ffi::synchronized_load::load_pair(|rank| {
                        devices[rank].own(|| {
                            let _memory_scope = cuteafd_ffi::memory_ledger::scope_owned(&label);
                            Ok(vec![ExpertWeights::load(
                                devices[rank].library,
                                catalog,
                                ExpertLayer::BackboneTp2 { layer, rank },
                                remaining[rank],
                            )?])
                        })
                    })
                }?;
                for (rank, mut loaded) in pair.into_iter().enumerate() {
                    remaining[rank] -= loaded[0].budget().resident_bytes;
                    devices[rank].run(|| {
                        weights[rank].append(loaded.get_mut());
                        Ok(())
                    })?;
                }
            }
            weights
        } else {
            ExpertWeights::load_tp2_pair(devices, catalog, &layers, available)?
        };
        let [left, right] = weights;
        let owners = [Rc::new(left), Rc::new(right)];
        Ok(std::array::from_fn(|rank| NativeTp2Weights {
            device: devices[rank],
            rank: rank as u8,
            layers: layers.clone(),
            max_rows,
            workspace_bytes: workspaces[rank],
            weights: owners[rank].clone(),
            selection,
            output_policy,
        }))
    }

    pub(crate) fn load(
        device: Device<'a>,
        catalog: &cuteafd_loader::OfficialV41Catalog,
        rank: u8,
        layers: Vec<usize>,
        max_rows: usize,
        budget: usize,
    ) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope_owned(&format!(
            "experts/TP2 expert layer halves {layers:?}"
        ));
        ensure!(
            ExpertFormat::of(catalog) == ExpertFormat::Native,
            "NativeTp2 requires MXFP4; choose the native-format executor"
        );
        let workspace_bytes = Self::admit(
            device,
            catalog,
            rank,
            &layers,
            max_rows,
            budget,
            RtxSelectionPolicy::CanonicalDecode,
            NativeOutputPolicy::TokenSums,
        )?;
        let weights = device.own(|| {
            let mut loaded = Vec::with_capacity(layers.len());
            let mut remaining = budget - workspace_bytes;
            for &layer in &layers {
                let weight = ExpertWeights::load(
                    device.library,
                    catalog,
                    ExpertLayer::BackboneTp2 {
                        layer,
                        rank: rank as usize,
                    },
                    remaining,
                )?;
                remaining -= weight.budget().resident_bytes;
                loaded.push(weight);
            }
            Ok(loaded)
        })?;
        Self::finish_load(
            device,
            rank,
            layers,
            max_rows,
            workspace_bytes,
            weights,
            RtxSelectionPolicy::CanonicalDecode,
            NativeOutputPolicy::TokenSums,
        )
    }

    fn finish_load(
        device: Device<'a>,
        rank: u8,
        layers: Vec<usize>,
        max_rows: usize,
        workspace_bytes: usize,
        weights: DeviceOwner<'a, Vec<ExpertWeights<'a>>>,
        selection: RtxSelectionPolicy,
        output_policy: NativeOutputPolicy,
    ) -> Result<Self> {
        Self::execution(
            device,
            rank,
            layers,
            max_rows,
            workspace_bytes,
            Rc::new(weights),
            selection,
            output_policy,
        )
    }

    /// Each lane shares only immutable weights; launch slots, scratch, output and
    /// completion events are allocated afresh on the weight owner's device.
    pub(crate) fn fork_lane(&self) -> Result<Self> {
        Self::execution(
            self.device,
            self.rank,
            self.layers.clone(),
            self.max_rows,
            self.workspace_bytes,
            self.weights.clone(),
            self.selection,
            self.output_policy,
        )
    }

    pub(crate) fn execution(
        device: Device<'a>,
        rank: u8,
        layers: Vec<usize>,
        max_rows: usize,
        workspace_bytes: usize,
        weights: Rc<DeviceOwner<'a, Vec<ExpertWeights<'a>>>>,
        selection: RtxSelectionPolicy,
        output_policy: NativeOutputPolicy,
    ) -> Result<Self> {
        validate_layers(&layers, rank)?;
        ensure!(
            weights.device.id == device.id
                && std::ptr::eq(weights.device.library, device.library)
                && weights.len() == layers.len(),
            "native TP2 weight owner mismatch"
        );
        let geometry = expert_geometry();
        let nvfp4 = weights[0].is_nvfp4();
        ensure!(
            weights.iter().all(|weight| weight.is_nvfp4() == nvfp4),
            "native TP2 mixed weight formats"
        );
        ensure!(
            !nvfp4
                || (output_policy == NativeOutputPolicy::KernelPlanes
                    && selection == RtxSelectionPolicy::FirstCapacity),
            "NVFP4 W4A4 requires BF16 route planes and explicit FirstCapacity selection"
        );
        let kernels = device.run(|| {
            kernel_capacities(max_rows)?
                .into_iter()
                .map(|c| {
                    ExpertLayer::BackboneTp2 {
                        layer: layers[0],
                        rank: rank as usize,
                    }
                    .select_kernel(device.library, c, nvfp4)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for kernel in &kernels {
            let info = kernel.info();
            ensure!(
                info.hidden_size == geometry.hidden
                    && info.topk == geometry.topk
                    && info.experts == geometry.experts
                    && info.logical_intermediate == geometry.intermediate / 2
                    && info.input_dtype == if nvfp4 { 1 } else { 7 }
                    && (kernel.output_kind() == V41ExpertOutputKind::Bf16Routes) == nvfp4,
                "TP2 native geometry/format mismatch"
            );
        }
        let reduction_plan = match selection {
            RtxSelectionPolicy::FirstCapacity => None,
            RtxSelectionPolicy::CanonicalDecode if !nvfp4 => {
                Some(crate::shared::experts::ExpertReductionPlan::native(
                    device.library,
                    kernels.iter().map(|k| *k.info()),
                )?)
            }
            RtxSelectionPolicy::CanonicalDecode => None,
        };
        let scratch_bytes = kernels
            .iter()
            .map(|k| k.info().scratch_bytes as usize)
            .max()
            .unwrap();
        let scratch = Allocation::new(device, scratch_bytes)?;
        let output = Allocation::new(
            device,
            Self::partial_bytes(geometry, max_rows, output_policy, nvfp4)?,
        )?;
        let stream = Stream::new(device)?;
        let states = device.own(|| {
            let mut states = Vec::with_capacity(kernels.len());
            for kernel in kernels {
                let mut slots = [std::ptr::null_mut(); 44];
                // SAFETY: one startup stream initializes the arena, drained before publication.
                let initialized = unsafe {
                    kernel.bind_scratch(scratch.buffer.ptr, scratch_bytes as u64, &mut slots)?;
                    kernel.initialize_scratch(scratch.buffer.ptr, scratch_bytes as u64, stream.raw)
                };
                let drained = stream.drain();
                initialized.and(drained)?;
                states.push(RankState { kernel, slots });
            }
            Ok(states)
        })?;
        let reducer = device.run(|| device.library.v41_tp2_expert_reducer())?;
        ensure!(
            output_policy == NativeOutputPolicy::KernelPlanes
                || if geometry.same_shape(&ExpertGeometry::DEEPSEEK_V41) {
                    reducer.supports_route_sums()
                } else {
                    reducer.supports_geometry_route_sums()
                },
            "TP2 ordered FP32 route sums unavailable"
        );
        Ok(Self {
            completion: RankCompletion::new(Event::new(device)?),
            device,
            rank,
            layers,
            geometry,
            max_rows,
            workspace_bytes,
            states,
            selection,
            output_policy,
            nvfp4,
            reduction_plan,
            weights,
            _scratch: scratch,
            output,
            reducer,
        })
    }

    pub(crate) fn immutable_weights(&self) -> NativeTp2Weights<'a> {
        NativeTp2Weights {
            device: self.device,
            rank: self.rank,
            layers: self.layers.clone(),
            max_rows: self.max_rows,
            workspace_bytes: self.workspace_bytes,
            weights: self.weights.clone(),
            selection: self.selection,
            output_policy: self.output_policy,
        }
    }

    pub(super) fn allocation(&self) -> &Allocation<'a> {
        &self.output
    }
    pub(super) fn execution_device(&self) -> Device<'a> {
        self.device
    }
    pub(super) fn bytes_per_expert(&self, layer: usize) -> Result<usize> {
        ensure!(
            self.layers.contains(&layer),
            "TP2 weight metadata layer unavailable"
        );
        Ok(self.weights[super::resident_index(&self.layers, layer)?]
            .budget()
            .resident_bytes
            / self.geometry.experts as usize)
    }

    fn execution_index(&self, rows: usize) -> Result<usize> {
        self.selection.execution_index(
            rows,
            rows,
            self.states
                .iter()
                .map(|state| state.kernel.info().capacity_rows as usize),
            || {
                self.reduction_plan
                    .as_ref()
                    .context("missing canonical TP2 plan")?
                    .execution_index(
                        rows,
                        rows,
                        self.states
                            .iter()
                            .map(|state| state.kernel.info().capacity_rows as usize),
                    )
            },
        )
    }
}

impl RtxExpertLayer for NativeTp2<'_> {
    fn shard(&self) -> RtxShard {
        RtxShard::Tp2 { rank: self.rank }
    }
    fn partial(&self) -> PartialDtype {
        if self.nvfp4 {
            PartialDtype::Bf16
        } else {
            PartialDtype::F32
        }
    }
    fn partial_layout(&self, rows: usize) -> Result<PartialLayout> {
        ensure!(
            (1..=self.max_rows).contains(&rows),
            "invalid TP2 partial rows"
        );
        let kind = self.states[self.execution_index(rows)?]
            .kernel
            .output_kind();
        Ok(
            if self.output_policy == NativeOutputPolicy::TokenSums
                || kind == V41ExpertOutputKind::Fp32Tokens
            {
                PartialLayout::Tokens
            } else {
                PartialLayout::Routes {
                    topk: self.geometry.topk as usize,
                }
            },
        )
    }
    fn layers(&self) -> &[usize] {
        &self.layers
    }
    fn device(&self) -> i32 {
        self.device.id
    }
    fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    fn group_rows(&self, rows: usize) -> Result<usize> {
        if rows == 0 {
            return Ok(self
                .states
                .last()
                .context("no installed TP2 expert capacity")?
                .kernel
                .group_rows());
        }
        ensure!(rows <= self.max_rows, "invalid TP2 grouping row count");
        let selected = self.execution_index(rows)?;
        self.states[selected].kernel.group_rows_for(rows)
    }
    fn output(&self) -> *mut c_void {
        self.output.buffer.ptr
    }
    unsafe fn enqueue(
        &mut self,
        layer: usize,
        rows: usize,
        input: ExpertInput,
        routes: Routes,
        stream: *mut c_void,
        abort: &mut dyn FnMut(),
    ) -> Result<()> {
        let result = (|| {
            ensure!(
                self.layers.contains(&layer) && (1..=self.max_rows).contains(&rows),
                "TP2 layer/rows not resident"
            );
            let wire = match (self.nvfp4, input) {
                (false, ExpertInput::Fp8K32(wire)) | (true, ExpertInput::Bf16(wire)) => wire,
                _ => bail!("native TP2 input does not match checkpoint format"),
            };
            ensure!(
                !wire.is_null() && !routes.ids.is_null() && !routes.weights.is_null(),
                "null TP2 input"
            );
            let selected = self.execution_index(rows)?;
            self.device.run(|| {
                let state = &mut self.states[selected];
                // SAFETY: caller retains complete rank-local inputs and owns exclusive stream/output use.
                unsafe {
                    self.weights[super::resident_index(&self.layers, layer)?]
                        .bind(&state.kernel, &mut state.slots)?;
                    state.slots[0] = wire;
                    state.slots[1] = routes.ids;
                    state.slots[2] = routes.weights;
                    let info = state.kernel.info();
                    state.kernel.launch(&V41ExpertLaunchArgs {
                        tensors: state.slots,
                        num_tokens: rows as i32,
                        max_rows: info.max_rows,
                        scatter_rows: rows as i32 * self.geometry.topk as i32,
                        rows_padded: info.rows_padded,
                        max_tasks: info.max_tasks,
                        max_phys_tiles: info.max_phys_tiles,
                        max_active_clusters: info.max_active_clusters,
                        stream,
                    })?;
                    let mut source = CuteafdDeviceBuffer {
                        ptr: state.slots[41],
                        bytes: output_bytes(self.geometry, rows)?,
                        device_id: self.device.id,
                        flags: 0,
                    };
                    match state.kernel.output_kind() {
                        V41ExpertOutputKind::Fp32Routes => {
                            source.bytes *= self.geometry.topk as usize;
                            if self.output_policy == NativeOutputPolicy::KernelPlanes {
                                self.device.library.copy_d2d_async(
                                    self.output.buffer,
                                    source,
                                    source.bytes,
                                    stream,
                                )
                            } else if self.geometry.same_shape(&ExpertGeometry::DEEPSEEK_V41) {
                                self.reducer.sum_routes(
                                    source,
                                    self.output.buffer,
                                    rows as u32,
                                    stream,
                                )
                            } else {
                                self.reducer.sum_routes_geometry(
                                    source,
                                    self.output.buffer,
                                    rows as u32,
                                    self.geometry,
                                    stream,
                                )
                            }
                        }
                        V41ExpertOutputKind::Fp32Tokens => self.device.library.copy_d2d_async(
                            self.output.buffer,
                            source,
                            source.bytes,
                            stream,
                        ),
                        V41ExpertOutputKind::Bf16Routes => {
                            source.bytes = source.bytes / 2 * self.geometry.topk as usize;
                            self.device.library.copy_d2d_async(
                                self.output.buffer,
                                source,
                                source.bytes,
                                stream,
                            )
                        }
                    }
                }
            })
        })();
        self.completion.finish(result, stream, abort)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_absolute_inventory_accepts_sparse_ids_and_rejects_aliases() {
        for rank in 0..2 {
            assert!(validate_layers(&[0, 2, 4], rank).is_ok());
            for invalid in [&[][..], &[2, 2], &[3, 1], &[expert_geometry().layers as usize]] {
                assert!(validate_layers(invalid, rank).is_err());
            }
        }
        assert!(validate_layers(&[0], 2).is_err());
    }

    #[test]
    fn queued_drop_waits_before_releasing_storage() -> Result<()> {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Arc, Condvar, Mutex,
        };
        struct PendingEvent {
            gate: Arc<(Mutex<bool>, Condvar)>,
            waiting: mpsc::Sender<()>,
        }
        impl CompletionEvent for PendingEvent {
            fn record(&self, _: *mut c_void) -> Result<()> {
                Ok(())
            }
            fn drain(&self, _: *mut c_void) -> Result<()> {
                self.wait()
            }
            fn wait(&self) -> Result<()> {
                self.waiting.send(())?;
                let (lock, wake) = &*self.gate;
                let _complete = wake
                    .wait_while(lock.lock().unwrap(), |done| !*done)
                    .unwrap();
                Ok(())
            }
        }
        struct Storage(Arc<AtomicUsize>);
        impl Drop for Storage {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct Executor {
            completion: RankCompletion<PendingEvent>,
            _storage: Storage,
        }
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let freed = Arc::new(AtomicUsize::new(0));
        let (started, waiting) = mpsc::channel();
        let mut executor = Executor {
            completion: RankCompletion::new(PendingEvent {
                gate: gate.clone(),
                waiting: started,
            }),
            _storage: Storage(freed.clone()),
        };
        executor
            .completion
            .finish(Ok(()), std::ptr::null_mut(), &mut || {
                panic!("success must not abort")
            })?;
        let (returned, done) = mpsc::channel();
        let dropper = std::thread::spawn(move || {
            drop(executor);
            returned.send(()).unwrap();
        });
        waiting.recv_timeout(std::time::Duration::from_secs(5))?;
        let released_early = freed.load(Ordering::SeqCst);
        let returned_early = done.try_recv().is_ok();
        // Release before asserting, so a regression never leaves a blocked thread.
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        dropper.join().unwrap();
        assert_eq!(released_early, 0);
        assert!(!returned_early);
        assert_eq!(freed.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn rank1_launch_error_aborts_waiting_rank0_before_return_without_inline_drain() -> Result<()> {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Arc, Condvar, Mutex,
        };
        use std::time::Duration;
        struct ProbeEvent {
            fail_record: bool,
            gate: Arc<(Mutex<bool>, Condvar)>,
            drains: Arc<AtomicUsize>,
        }
        impl CompletionEvent for ProbeEvent {
            fn record(&self, _: *mut c_void) -> Result<()> {
                ensure!(!self.fail_record, "injected record failure");
                Ok(())
            }
            fn drain(&self, _: *mut c_void) -> Result<()> {
                self.wait()
            }
            fn wait(&self) -> Result<()> {
                self.drains.fetch_add(1, Ordering::SeqCst);
                let (lock, wake) = &*self.gate;
                let _aborted = wake
                    .wait_while(lock.lock().unwrap(), |aborted| !*aborted)
                    .unwrap();
                Ok(())
            }
        }
        for (fail_launch, fail_record) in [(true, false), (false, true), (true, true)] {
            let gate = Arc::new((Mutex::new(false), Condvar::new()));
            let peer_gate = gate.clone();
            let (waiting_tx, waiting_rx) = mpsc::channel();
            let (peer_tx, peer_rx) = mpsc::channel();
            let peer = std::thread::spawn(move || {
                let (lock, wake) = &*peer_gate;
                let guard = lock.lock().unwrap();
                waiting_tx.send(()).unwrap();
                let _aborted = wake.wait_while(guard, |aborted| !*aborted).unwrap();
                peer_tx.send(()).unwrap();
            });
            waiting_rx.recv_timeout(Duration::from_secs(5))?;
            let rank_gate = gate.clone();
            let drains = Arc::new(AtomicUsize::new(0));
            let rank_drains = drains.clone();
            let (returned_tx, returned_rx) = mpsc::channel();
            let rank = std::thread::spawn(move || {
                let mut completion = RankCompletion::new(ProbeEvent {
                    fail_record,
                    gate: rank_gate.clone(),
                    drains: rank_drains.clone(),
                });
                let mut abort = || {
                    *rank_gate.0.lock().unwrap() = true;
                    rank_gate.1.notify_all();
                };
                let launch = if fail_launch {
                    Err(anyhow::anyhow!("injected rank1 launch failure"))
                } else {
                    Ok(())
                };
                let result = completion.finish(launch, std::ptr::null_mut(), &mut abort);
                returned_tx
                    .send((
                        result.is_err(),
                        *rank_gate.0.lock().unwrap(),
                        rank_drains.load(Ordering::SeqCst),
                    ))
                    .unwrap();
                drop(completion);
            });
            let returned = returned_rx.recv_timeout(Duration::from_secs(5));
            // Always release before assertions, so a regression cannot hang the suite.
            *gate.0.lock().unwrap() = true;
            gate.1.notify_all();
            peer.join().unwrap();
            rank.join().unwrap();
            assert_eq!(returned?, (true, true, 0));
            assert_eq!(
                drains.load(Ordering::SeqCst),
                1,
                "Drop drains only after abort"
            );
            peer_rx.recv_timeout(Duration::from_secs(5))?;
        }
        Ok(())
    }

    #[test]
    fn capacities_round_up_once() {
        for (rows, expected) in [
            (1, vec![1]),
            (3, vec![1, 16]),
            (80, vec![1, 16, 80]),
            (255, vec![1, 16, 80, 256]),
            (4096, vec![1, 16, 80, 256, 1024, 4096]),
        ] {
            assert_eq!(kernel_capacities(rows).unwrap(), expected);
        }
        assert!(kernel_capacities(0).is_err());
        assert!(kernel_capacities(4097).is_err());
    }
    #[test]
    fn output_extent_uses_each_geometry() {
        for g in [
            ExpertGeometry::DEEPSEEK_V41,
            ExpertGeometry::DEEPSEEK_V4_FLASH,
            ExpertGeometry::DEEPSEEK_V4_PRO,
        ] {
            for rows in [1, 3, 16, 80, 255, 4096] {
                assert_eq!(output_bytes(g, rows).unwrap(), rows * g.hidden as usize * 4);
            }
        }
        assert!(output_bytes(ExpertGeometry::DEEPSEEK_V4_PRO, usize::MAX).is_err());
    }
}
