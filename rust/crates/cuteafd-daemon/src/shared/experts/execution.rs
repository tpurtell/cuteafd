//! Per-wave stable expert I/O, scratch, stream and graph ownership.
use super::layer::{ExpertLayer, ExpertWeights};
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41CompactReducer, V41ExpertKernel, V41ExpertLaunchArgs,
    V41RouteReducer,
};
use std::ffi::c_void;
pub(crate) mod timing;
mod grouping;
pub(crate) use grouping::{ExpertGroupingError, InstalledExpertGrouping};
use timing::ExpertTiming;

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum NativeExpertCapacityError {
    #[error("native expert live capacity {live} must be positive and no greater than compiled capacity {compiled} (maximum 4096)")]
    InvalidExtent { compiled: u32, live: u32 },
    #[error("native expert rows {rows} exceed live capacity {live}")]
    RowsExceedLive { rows: usize, live: u32 },
    #[error("native expert execution requires positive rows")]
    EmptyRows,
}

fn validate_capacity(compiled: u32, live: u32) -> std::result::Result<(), NativeExpertCapacityError> {
    if live == 0 || compiled < live || compiled > 4096 {
        return Err(NativeExpertCapacityError::InvalidExtent { compiled, live });
    }
    Ok(())
}

fn validate_live_rows(rows: usize, live: u32) -> std::result::Result<(), NativeExpertCapacityError> {
    if rows == 0 { return Err(NativeExpertCapacityError::EmptyRows); }
    if rows > live as usize { return Err(NativeExpertCapacityError::RowsExceedLive { rows, live }); }
    Ok(())
}

/// Native roles that publish token-major FP32 routed partials and therefore
/// need the compact output buffer plus the compact reducer: legacy grouped
/// Spark TP4 (role 1) and the Spark shard families TP2/TP3/pure-TP6 (roles
/// 5/6/7). The full-width RTX backbone (role 2) owns its separate local reducer
/// and never enters this execution.
fn compact_output_role(role: u32) -> bool {
    matches!(role, 1 | 5 | 6 | 7)
}

/// Admit the exact states owned by this wave before allocating their scratch.
/// NVFP4 and draft roles retain their separate dispatch contracts.
fn native_reduction_plan(layer: ExpertLayer, library: &NativeLibrary, capacity: u32,
    nvfp4: bool) -> Result<Option<super::ExpertReductionPlan>> {
    if nvfp4 || !compact_output_role(layer.role()) { return Ok(None); }
    let mut capacities = vec![capacity]; // execution_state index 0: main
    if capacity > 1 { capacities.push(1); } // index 1: serial
    if capacity > 80 { capacities.push(80); } // index 2: grouped decode
    let infos = capacities.into_iter().map(|rows| layer.select_info(library, rows, false))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(super::ExpertReductionPlan::native(library, infos)?))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertExecutionBudget {
    pub scratch_bytes: usize,
    pub decode_scratch_bytes: usize,
    pub small_scratch_bytes: usize,
    pub hidden_bytes: usize,
    pub routing_bytes: usize,
    pub output_and_shared_bytes: usize,
}
impl ExpertExecutionBudget {
    fn from_info(info: cuteafd_ffi::V41ExpertInfo, live: u32, decode: usize, small: usize) -> Result<Self> {
        validate_capacity(info.capacity_rows, live)?;
        Ok(Self {
            scratch_bytes: usize::try_from(info.scratch_bytes)?,
            decode_scratch_bytes: decode,
            small_scratch_bytes: small,
            hidden_bytes: (live as usize).checked_mul(info.input_row_bytes()?).context("hidden buffer overflow")?,
            routing_bytes: (live as usize).checked_mul(info.topk as usize * 8).context("routing buffer overflow")?,
            output_and_shared_bytes: (live as usize)
                .checked_mul(cuteafd_core::expert_geometry().row_bytes() as usize * if info.role == 0 { 2 } else { 1 })
                .context("output buffer overflow")?,
        })
    }

    pub fn total(self) -> Result<usize> {
        [
            self.scratch_bytes,
            self.decode_scratch_bytes,
            self.small_scratch_bytes,
            self.hidden_bytes,
            self.routing_bytes,
            self.output_and_shared_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |sum, n| {
            sum.checked_add(n)
                .context("expert execution budget overflow")
        })
    }
}

/// Dedicated decode state shares the wave's immutable weights and input buffers.
/// Its arena and kernel are initialized before serving or graph capture.
struct DecodeExecution<'library> {
    kernel: V41ExpertKernel<'library>,
    scratch: DeviceAllocation<'library>,
    slots: [*mut c_void; 44],
}

/// One exclusive wave's buffers and graph borrow immutable resident weights.
/// Upstream producers and transport must obey this owner's stream/lifetime contract.
pub(crate) struct ExpertExecution<'weights, 'library> {
    // Destroy stream before allocations; Drop drains and destroys the graph first.
    pub(crate) stream: LoadStream<'library>,
    pub(crate) _weights: &'weights ExpertWeights<'library>,
    library: &'library NativeLibrary,
    pub(crate) kernel: V41ExpertKernel<'library>,
    decode: Option<DecodeExecution<'library>>,
    small: Option<DecodeExecution<'library>>,
    reduction_plan: Option<super::ExpertReductionPlan>,
    reducer: V41RouteReducer<'library>,
    timing: Option<ExpertTiming<'library>>,
    worker_timing: Option<timing::WorkerTiming<'library>>,
    scratch: DeviceAllocation<'library>,
    pub(crate) hidden: DeviceAllocation<'library>,
    ids: DeviceAllocation<'library>,
    routing: DeviceAllocation<'library>,
    /// Pinned route ids then weights, uploaded asynchronously per request. The
    /// stream drains before each response is emitted, so the next request may
    /// rewrite this staging.
    route_staging: crate::shared::memory::HostAllocation<'library>,
    compact_reducer: Option<V41CompactReducer<'library>>,
    compact_output: Option<DeviceAllocation<'library>>,
    output: Option<DeviceAllocation<'library>>,
    pub(crate) shared: Option<DeviceAllocation<'library>>,
    slots: [*mut c_void; 44],
    graph: Option<(*mut c_void, u32, bool)>,
    /// Replicated-group index this worker owns; set once at startup for an
    /// explicit `TP×EP` topology, `None` for every legacy path. The group is
    /// independent of the resident layer, so rebinding a layer never changes it.
    native_group: Option<u8>,
    /// Lane-owned execution on a rank serving several execution lanes: a
    /// mapped request returns with its work queued; the owner emits it once
    /// [`Self::stream_idle`] reports this stream drained.
    deferred: bool,
    live_capacity: u32,
    budget: ExpertExecutionBudget,
}
impl<'library> ExpertWeights<'library> {
    pub fn execution_budget(&self, compiled: u32, live: u32) -> Result<ExpertExecutionBudget> {
        let library = self.buffers[0].library;
        Self::plan_execution(self.layer, library, compiled, live, self.is_nvfp4())
    }
    pub(crate) fn plan_execution(
        layer: ExpertLayer,
        library: &NativeLibrary,
        compiled: u32,
        live: u32,
        nvfp4: bool,
    ) -> Result<ExpertExecutionBudget> {
        validate_capacity(compiled, live)?;
        let _ = native_reduction_plan(layer, library, compiled, nvfp4)?;
        let info = layer.select_info(library, compiled, nvfp4)?;
        let decode = if compact_output_role(info.role) && compiled > 1 {
            usize::try_from(layer.select_info(library, 1, nvfp4)?.scratch_bytes)?
        } else { 0 };
        let small = if compact_output_role(info.role) && compiled > 80 {
            usize::try_from(layer.select_info(library, 80, nvfp4)?.scratch_bytes)?
        } else { 0 };
        ExpertExecutionBudget::from_info(info, live, decode, small)
    }
    pub fn execution(
        &self,
        compiled: u32,
        live: u32,
        available_device_bytes: usize,
    ) -> Result<ExpertExecution<'_, 'library>> {
        validate_capacity(compiled, live)?;
        let library = self.buffers[0].library;
        let reduction_plan = native_reduction_plan(self.layer, library, compiled, self.is_nvfp4())?;
        let budget = self.execution_budget(compiled, live)?;
        ensure!(
            budget.total()? <= available_device_bytes,
            "expert execution buffers exceed device budget"
        );
        let kernel = self.layer.select_kernel(library, compiled, self.is_nvfp4())?;
        let reducer = library.v41_route_reducer()?;
        // Role-gated, DEBUG-only diagnostics. The same routed breakdown is
        // available for the replicated Spark TP2/TP3 shards (roles 5/6), whose
        // compact output and reducer buffers are already allocated.
        let timing = if compact_output_role(kernel.info().role)
            && tracing::enabled!(target: "cuteafd::expert_timing", tracing::Level::DEBUG)
        {
            Some(ExpertTiming::new(library)?)
        } else {
            None
        };
        let scratch = DeviceAllocation::named(library, budget.scratch_bytes, "experts/workspace/main_scratch")?;
        let hidden = DeviceAllocation::named(library, budget.hidden_bytes, "experts/workspace/hidden")?;
        let ids = DeviceAllocation::named(library, budget.routing_bytes / 2, "experts/workspace/routes")?;
        let routing = DeviceAllocation::named(library, budget.routing_bytes / 2, "experts/workspace/routes")?;
        let output = if kernel.info().role == 0 {
            Some(DeviceAllocation::named(library, budget.hidden_bytes, "experts/workspace/hidden")?)
        } else {
            None
        };
        let compact_output = if compact_output_role(kernel.info().role) {
            Some(DeviceAllocation::named(
                library,
                budget.output_and_shared_bytes,
                "experts/workspace/compact_output",
            )?)
        } else {
            None
        };
        let compact_reducer = if compact_output_role(kernel.info().role) {
            Some(library.v41_compact_reducer()?)
        } else {
            None
        };
        let shared = if kernel.info().role == 0 {
            Some(DeviceAllocation::named(library, budget.hidden_bytes, "experts/workspace/hidden")?)
        } else {
            None
        };
        let mut slots = [std::ptr::null_mut(); 44];
        unsafe {
            kernel.bind_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, &mut slots)?;
        }
        self.bind(&kernel, &mut slots)?;
        slots[0] = hidden.buffer.ptr;
        slots[1] = ids.buffer.ptr;
        slots[2] = routing.buffer.ptr;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        unsafe {
            kernel.initialize_scratch(
                scratch.buffer.ptr,
                scratch.buffer.bytes as u64,
                stream.raw,
            )?;
            library.cuda_stream_synchronize(stream.raw)?;
        }
        let prepare_small = |planned_capacity, scratch_bytes| -> Result<Option<DecodeExecution<'library>>> {
            if scratch_bytes == 0 { return Ok(None); }
            let decode_kernel =
                self.layer.select_kernel(library, planned_capacity, self.is_nvfp4())?;
            ensure!(
                decode_kernel.info().input_dtype == kernel.info().input_dtype,
                "decode and grouped expert input formats differ"
            );
            let decode_scratch = DeviceAllocation::named(library, scratch_bytes, if planned_capacity == 1 {
                "experts/workspace/serial_scratch"
            } else { "experts/workspace/grouped_decode_scratch" })?;
            let mut decode_slots = slots;
            unsafe {
                decode_kernel.bind_scratch(
                    decode_scratch.buffer.ptr,
                    decode_scratch.buffer.bytes as u64,
                    &mut decode_slots,
                )?;
            }
            self.bind(&decode_kernel, &mut decode_slots)?;
            unsafe {
                let initialized = decode_kernel.initialize_scratch(
                    decode_scratch.buffer.ptr,
                    decode_scratch.buffer.bytes as u64,
                    stream.raw,
                );
                // Drain even on initialization failure before releasing its arena.
                let drained = library.cuda_stream_synchronize(stream.raw);
                initialized.and(drained)?;
            }
            Ok(Some(DecodeExecution {
                kernel: decode_kernel,
                scratch: decode_scratch,
                slots: decode_slots,
            }))
        };
        let decode = prepare_small(1, budget.decode_scratch_bytes)?;
        let small = prepare_small(80, budget.small_scratch_bytes)?;
        let route_staging = crate::shared::memory::HostAllocation::named(library, budget.routing_bytes, "experts/workspace/route_staging")?;
        Ok(ExpertExecution {
            stream,
            _weights: self,
            library,
            route_staging,
            kernel,
            decode,
            small,
            reduction_plan,
            reducer,
            timing,
            worker_timing: None,
            compact_reducer,
            compact_output,
            scratch,
            hidden,
            ids,
            routing,
            output,
            shared,
            slots,
            deferred: false,
            graph: None,
            native_group: None,
            live_capacity: live,
            budget,
        })
    }
}
impl<'weights, 'library> ExpertExecution<'weights, 'library> {
    pub(crate) fn install_worker_timing(&mut self, rank: usize) -> Result<()> {
        self.worker_timing = timing::WorkerTiming::from_env(self.library, rank)?;
        Ok(())
    }

    /// Bind this worker to its replicated group before serving. Called once at
    /// startup; every remote request for this worker must then carry the native
    /// ownership contract and is unpacked for exactly this group.
    pub(crate) fn install_native_group(&mut self, group: Option<u8>) -> Result<()> {
        ensure!(
            self.graph.is_none(),
            "cannot install a replicated group on a captured expert graph"
        );
        ensure!(
            self.native_group.is_none(),
            "replicated expert group already installed"
        );
        ensure!(
            group.map_or(true, |group| group < 3),
            "replicated expert group index out of range"
        );
        self.native_group = group;
        Ok(())
    }
    /// Reuse a wave's workspace across resident layers after its prior work drains.
    /// Captured graphs retain weight addresses and cannot be rebound.
    pub fn bind_layer(&mut self, weights: &'weights ExpertWeights<'library>) -> Result<()> {
        ensure!(
            self.graph.is_none(),
            "cannot rebind a captured expert graph"
        );
        ensure!(
            std::ptr::eq(self.library, weights.buffers[0].library),
            "expert layer belongs to a different native library"
        );
        ensure!(self._weights.is_nvfp4() == weights.is_nvfp4(),
            "cannot rebind expert workspace across quantization families");
        self.synchronize()?;
        let mut slots = self.slots;
        weights.bind(&self.kernel, &mut slots)?;
        for decode in [&mut self.decode, &mut self.small].into_iter().flatten() {
            let mut decode_slots = decode.slots;
            weights.bind(&decode.kernel, &mut decode_slots)?;
            decode.slots = decode_slots;
        }
        self.slots = slots;
        self._weights = weights;
        Ok(())
    }

    pub fn budget(&self) -> ExpertExecutionBudget {
        self.budget
    }
    pub fn stream(&self) -> *mut c_void {
        self.stream.raw
    }
    /// Borrowed hidden in the native kernel's input format, I32 route IDs and
    /// FP32 route weights; never free these views. dSpark always uses BF16.
    /// Enqueue producers on stream() or provide event ordering before launch/replay.
    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 3] {
        [self.hidden.buffer, self.ids.buffer, self.routing.buffer]
    }
    pub fn shared(&self) -> Option<CuteafdDeviceBuffer> {
        self.shared.as_ref().map(|b| b.buffer)
    }
    pub fn output(&self) -> Option<CuteafdDeviceBuffer> {
        self.output.as_ref().map(|b| b.buffer)
    }
    /// Borrow the output state selected for a launch of exactly `rows` rows.
    /// A one-row view is not a prefix of the multi-row state's output.
    pub fn route_partials(&self, rows: u32) -> Result<CuteafdDeviceBuffer> {
        validate_live_rows(rows as usize, self.live_capacity)?;
        let (kernel, slots, scratch) = self.execution_state(rows)?;
        ensure!(kernel.output_kind() == cuteafd_ffi::V41ExpertOutputKind::Fp32Routes,
            "expert output is not FP32 route planes");
        Ok(CuteafdDeviceBuffer {
            ptr: slots[41],
            bytes: rows as usize * kernel.info().topk as usize * cuteafd_core::expert_geometry().hidden as usize * 4,
            device_id: scratch.buffer.device_id,
            flags: 0,
        })
    }
    fn execution_state(
        &self,
        rows: u32,
    ) -> Result<(
        &V41ExpertKernel<'library>,
        &[*mut c_void; 44],
        &DeviceAllocation<'library>,
    )> {
        validate_live_rows(rows as usize, self.live_capacity)?;
        if let Some(plan) = &self.reduction_plan {
            let capacities = [Some(self.kernel.info().capacity_rows),
                self.decode.as_ref().map(|s| s.kernel.info().capacity_rows),
                self.small.as_ref().map(|s| s.kernel.info().capacity_rows)];
            let index = plan.execution_index(rows as usize, rows as usize,
                capacities.into_iter().flatten().map(|c| c as usize))?;
            return match index {
                0 => Ok((&self.kernel, &self.slots, &self.scratch)),
                1 => self.decode.as_ref().map(|s| (&s.kernel, &s.slots, &s.scratch))
                    .context("admitted native expert decode state missing"),
                2 => self.small.as_ref().map(|s| (&s.kernel, &s.slots, &s.scratch))
                    .context("admitted native expert grouped state missing"),
                _ => anyhow::bail!("invalid fixed native expert state index"),
            };
        }
        if rows == 1 {
            if let Some(decode) = &self.decode {
                return Ok((&decode.kernel, &decode.slots, &decode.scratch));
            }
        }
        if let Some(small) = &self.small {
            if rows <= small.kernel.info().capacity_rows {
                return Ok((&small.kernel, &small.slots, &small.scratch));
            }
        }
        Ok((&self.kernel, &self.slots, &self.scratch))
    }

    /// Zero queries the main compiled tile; positive rows follow the launch selector.
    pub fn group_rows(&self, rows: usize) -> Result<usize> {
        if rows == 0 { return Ok(self.kernel.group_rows()); }
        validate_live_rows(rows, self.live_capacity)?;
        self.execution_state(u32::try_from(rows)?)?.0.group_rows_for(rows)
    }

    pub(crate) fn installed_grouping(&self, loaded: cuteafd_transport::worker_selection::WorkerSelection,
        rank: usize) -> Result<cuteafd_transport::worker_selection::InstalledExpertGrouping> {
        use cuteafd_transport::worker_selection::{ExpertGroupingSelector, InstalledExpertGrouping, InstalledExpertVariant};
        let mut kernels = vec![&self.kernel];
        kernels.extend([&self.decode, &self.small].into_iter().flatten().map(|state| &state.kernel));
        let variants = kernels.into_iter().map(|kernel| InstalledExpertVariant::from_queries(
            kernel.info().capacity_rows as usize, |rows| kernel.group_rows_for(rows))).collect::<Result<Vec<_>>>()?;
        let report = InstalledExpertGrouping::new(loaded, rank, self.live_capacity as usize,
            ExpertGroupingSelector::Native, variants)?;
        for rows in 0..=self.live_capacity as usize {
            ensure!(report.group_rows(rows)? == self.group_rows(rows)?,
                "native installed grouping differs from actual selector at rows {rows}");
        }
        Ok(report)
    }

    /// Leave mapped requests queued at return (see `deferred`).
    pub(crate) fn defer(&mut self) { self.deferred = true; }

    /// Whether every request queued on this execution's stream has completed.
    pub(crate) fn stream_idle(&self) -> Result<bool> {
        // SAFETY: the owned stream is live for the worker's lifetime.
        unsafe { self.library.cuda_stream_query(self.stream.raw) }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }

    /// # Safety
    /// Initialize hidden in the kernel-advertised input format, in-range I32
    /// expert IDs and finite nonnegative
    /// FP32 routing weights for `rows` before this operation, with stream ordering.
    /// Initialize shared BF16 output too if requested. External readers/writers must
    /// finish before storage is reused; all operations belong to this GPU worker.
    pub unsafe fn launch(&mut self, rows: u32, include_shared: bool) -> Result<()> {
        unsafe { self.launch_on(rows, include_shared, self.stream.raw) }
    }
    pub(crate) unsafe fn launch_on(
        &mut self,
        rows: u32,
        include_shared: bool,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            !include_shared || self.shared.is_some(),
            "shared output belongs on coordinator RTX"
        );
        let (kernel, slots, _) = self.execution_state(rows)?;
        let args = V41ExpertLaunchArgs::new(kernel.info(), *slots, rows, stream)?;
        unsafe {
            kernel.launch(&args)?;
        }
        if let Some(output) = &self.output {
            unsafe {
                self.reducer.launch(
                    [
                        slots[41].cast(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    ],
                    if include_shared {
                        self.shared.as_ref().unwrap().buffer.ptr.cast()
                    } else {
                        std::ptr::null()
                    },
                    output.buffer.ptr.cast(),
                    rows,
                    1,
                    3,
                    stream,
                )?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// Same initialized-buffer and ordering contract as launch; captures fixed rows
    /// and shared-output participation. Input values may change between replays.
    pub unsafe fn capture(&mut self, rows: u32, include_shared: bool) -> Result<()> {
        ensure!(
            self.graph.is_none(),
            "expert execution already has a captured graph"
        );
        unsafe {
            self.launch(rows, include_shared)?;
        }
        self.synchronize()?;
        unsafe {
            self.library.cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launch = unsafe { self.launch(rows, include_shared) };
        // End capture on both paths so an error cannot leave the stream capturing.
        let captured = unsafe { self.library.cuda_graph_end_capture(self.stream.raw) };
        match (launch, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows, include_shared));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                unsafe {
                    self.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(error)
            }
            (Err(error), Err(_)) => Err(error),
            (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// The captured rows' inputs (and optional shared output) must satisfy launch's
    /// contract, with writes ordered before replay and reads ordered after it.
    pub unsafe fn replay(&mut self) -> Result<()> {
        let (graph, _, _) = self.graph.context("expert graph has not been captured")?;
        unsafe { self.library.cuda_graph_launch(graph, self.stream.raw) }
    }
}
impl Drop for ExpertExecution<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining V4.1 expert execution");
        }
        if let Some((graph, _, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying V4.1 expert execution graph");
            }
        }
    }
}

/// Reusable host exchange for the TCP fallback; RDMA can consume device route views.
pub(crate) struct HostExpertExchange {
    pub(crate) ids: Vec<i32>,
    pub(crate) routing: Vec<f32>,
    pub(crate) partials: Vec<u8>,
}
impl HostExpertExchange {
    /// Exact bytes owned by the reusable exchange at its admitted capacity.
    pub fn bytes_for(capacity: u32) -> Result<usize> {
        ensure!((1..=4096).contains(&capacity), "unsupported native host exchange capacity");
        let geometry = cuteafd_core::expert_geometry();
        let routes = capacity as usize * geometry.topk as usize;
        let row_bytes = geometry.hidden as usize * 2;
        routes.checked_mul(8)
            .and_then(|routes| routes.checked_add(capacity as usize * row_bytes))
            .context("native host exchange size overflow")
    }
    pub fn new(capacity: u32) -> Result<Self> {
        let bytes = Self::bytes_for(capacity)?;
        let routes = capacity as usize * cuteafd_core::expert_geometry().topk as usize;
        Ok(Self {
            ids: vec![0; routes],
            routing: vec![0.0; routes],
            partials: vec![0; bytes - routes * 8],
        })
    }
}
impl ExpertExecution<'_, '_> {
    /// Execute once and deliver bounded frames using caller-owned index scratch.
    /// The sink must finish consuming each borrowed chunk before returning.
    /// A send failure aborts the wave; callers must not retry it on the same
    /// partially delivered response stream without resetting receiver state.
    pub fn execute_host_chunks<F>(
        &mut self,
        request: &cuteafd_transport::expert::BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        row_indices: &mut [u32],
        max_frame_bytes: usize,
        mut sink: F,
    ) -> Result<()>
    where
        F: FnMut(cuteafd_transport::ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    {
        let chunk_rows = request.response_chunk_rows(max_frame_bytes)?;
        ensure!(
            row_indices.len() >= chunk_rows as usize,
            "response row-index scratch is too short"
        );
        let response = self.execute_host_request(request, executor_id, exchange)?;
        let stride = cuteafd_core::expert_geometry().row_bytes() as usize;
        for start in (0..request.rows()).step_by(chunk_rows as usize) {
            let end = start.saturating_add(chunk_rows).min(request.rows());
            let payload =
                &response.partial_output_payload[start as usize * stride..end as usize * stride];
            sink(request.response_chunk(
                executor_id,
                start,
                payload,
                row_indices,
                max_frame_bytes,
            )?)?;
        }
        Ok(())
    }

    /// Write the complete rank plane into the transport's registered send slot.
    /// Return None before execution when bounded frames or checksums need fallback.
    /// # Safety
    /// The slot is GPU-accessible on this device and exclusively owned until the
    /// caller emits the returned response; the transport retains it through send completion.
    pub unsafe fn execute_mapped_request(&mut self,
        request: &cuteafd_transport::expert::BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, slot: CuteafdDeviceBuffer,
        hidden: Option<CuteafdDeviceBuffer>,
    ) -> Result<Option<cuteafd_transport::ExpertProtocolV2DeviceResponseRef<'static>>> {
        let prefix = cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        if !request.permits_device_response() || slot.bytes < prefix + bytes { return Ok(None); }
        let output = CuteafdDeviceBuffer {
            ptr: unsafe { slot.ptr.cast::<u8>().add(prefix).cast() }, bytes, ..slot
        };
        // Validate the destination descriptor before launching any GPU work.
        let response = request.response_device(executor_id, output)?;
        self.execute_request_output(request, executor_id, exchange, Some(output), hidden)?;
        Ok(Some(response))
    }

    /// Host fallback from a validated wire request to a compact BF16 rank response.
    /// The borrowed response prevents reuse of exchange storage until encoding/send ends.
    pub fn execute_host_request<'a>(
        &mut self,
        request: &cuteafd_transport::expert::BackboneRequest<'_>,
        executor_id: u64,
        exchange: &'a mut HostExpertExchange,
    ) -> Result<cuteafd_transport::ExpertProtocolV2ResponseRef<'a>> {
        self.execute_request_output(request, executor_id, exchange, None, None)?;
        request.response(executor_id, &exchange.partials[..request.plane_bytes()?])
    }

    fn execute_request_output(&mut self,
        request: &cuteafd_transport::expert::BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, destination: Option<CuteafdDeviceBuffer>,
        hidden_view: Option<CuteafdDeviceBuffer>,
    ) -> Result<()> {
        let layer = match self._weights.layer {
            ExpertLayer::Backbone { layer, .. }
            | ExpertLayer::BackboneReplicatedTp { layer, .. } => layer,
            _ => anyhow::bail!("backbone requests cannot execute on RTX dSpark weights"),
        };
        ensure!(
            request.layer() as usize == layer,
            "request does not match resident expert layer"
        );
        ensure!(executor_id != 0, "native response needs executor identity");
        validate_live_rows(request.rows() as usize, self.live_capacity)?;
        request.require_input_dtype(self.kernel.info().input_dtype)?;
        let routes = request.rows() as usize * cuteafd_core::expert_geometry().topk as usize;
        let bytes = request.plane_bytes()?;
        ensure!(
            exchange.partials.len() >= bytes,
            "host exchange is too small"
        );
        // The admission contract and the resident shard must agree: a
        // topology-bound worker unpacks exactly its own group's routes and masks
        // every other route to the kernel's unassigned sentinel, while a legacy
        // worker rejects an ownership-encoded request outright.
        match (self.native_group, request.is_native_group()) {
            (Some(group), true) => request.copy_native_group_routes_into(
                &mut exchange.ids,
                &mut exchange.routing,
                group,
            )?,
            (None, false) => request.copy_routes_into(&mut exchange.ids, &mut exchange.routing)?,
            _ => anyhow::bail!(
                "replicated expert group admission differs from this worker's bound topology"
            ),
        }
        let started = self.timing.as_ref().map(|_| std::time::Instant::now());
        // Supported hosts and CUDA use the checkpoint/wire little-endian byte order.
        ensure!(
            cfg!(target_endian = "little"),
            "native host exchange requires little-endian storage"
        );
        // Every earlier request drained this stream before emitting its response
        // (and bind_layer drains on rebind), so all three inputs are enqueued
        // back to back with no host wait: the hidden rows as a device copy from
        // the mapped request frame when the transport exposes one, and the route
        // ids/weights from pinned staging.
        let hidden_bytes = request.hidden().len();
        ensure!(routes * 8 <= self.route_staging.buffer.bytes && hidden_bytes <= self.hidden.buffer.bytes,
            "native request staging is too small");
        {
            let staging = self.route_staging.bytes_mut();
            unsafe {
                staging[..routes * 4].copy_from_slice(
                    std::slice::from_raw_parts(exchange.ids.as_ptr().cast::<u8>(), routes * 4));
                staging[routes * 4..routes * 8].copy_from_slice(
                    std::slice::from_raw_parts(exchange.routing.as_ptr().cast::<u8>(), routes * 4));
            }
        }
        let uploaded = (|| -> Result<()> { unsafe {
            match hidden_view.filter(|view| view.bytes >= hidden_bytes) {
                Some(view) => self.library.copy_d2d_async(self.hidden.buffer,
                    CuteafdDeviceBuffer { bytes: hidden_bytes, ..view }, hidden_bytes, self.stream.raw)?,
                None => self.library.copy_h2d_async(self.hidden.buffer, request.hidden(), self.stream.raw)?,
            }
            let mut ids = self.route_staging.buffer; ids.bytes = routes * 4;
            self.library.copy_host_buffer_h2d_async(self.ids.buffer, ids, routes * 4, self.stream.raw)?;
            let mut weights = self.route_staging.buffer;
            weights.ptr = weights.ptr.cast::<u8>().add(routes * 4).cast(); weights.bytes = routes * 4;
            self.library.copy_host_buffer_h2d_async(self.routing.buffer, weights, routes * 4, self.stream.raw)?;
            Ok(())
        } })();
        if let Err(error) = uploaded { self.synchronize()?; return Err(error); }
        let uploaded_us = started.map(|t| t.elapsed().as_micros() as u64);
        if let Some(timing) = &self.timing {
            unsafe {
                timing.record(0, self.stream.raw)?;
            }
        }
        // SAFETY: this worker retains events, inputs and stream through the barrier below.
        unsafe {
            if let Some(timing) = &self.worker_timing { timing.record(0, self.stream.raw)?; }
            self.launch(request.rows(), false)?;
        }
        if let Some(timing) = &self.timing {
            unsafe {
                timing.record(1, self.stream.raw)?;
            }
        }
        let output = destination.unwrap_or(self
            .compact_output
            .as_ref()
            .context("missing compact output")?
            .buffer);
        unsafe {
            let reducer = self.compact_reducer.as_ref().context("missing compact reducer")?;
            let (kernel, slots, _) = self.execution_state(request.rows())?;
            match kernel.output_kind() {
                cuteafd_ffi::V41ExpertOutputKind::Fp32Tokens =>
                    reducer.compact_tokens(slots[41].cast(), output.ptr.cast(), request.rows(), self.stream.raw)?,
                cuteafd_ffi::V41ExpertOutputKind::Bf16Routes =>
                    reducer.compact_bf16_routes(slots[41].cast(), output.ptr.cast(), request.rows(), self.stream.raw)?,
                cuteafd_ffi::V41ExpertOutputKind::Fp32Routes =>
                    reducer.compact(self.route_partials(request.rows())?.ptr.cast(),
                        output.ptr.cast(), request.rows(), self.stream.raw)?,
            }
        }
        if let Some(timing) = &self.timing {
            unsafe {
                timing.record(2, self.stream.raw)?;
            }
        }
        // SAFETY: the reducer and kernel precede this end event on the owned stream.
        unsafe {
            if let Some(timing) = &self.worker_timing { timing.record(1, self.stream.raw)?; }
        }
        // A deferred mapped request completes when the owner observes the
        // stream idle; the transport holds both slots until then.
        if self.deferred && destination.is_some() { return Ok(()); }
        self.synchronize()?;
        if let Some(timing) = &mut self.worker_timing {
            // SAFETY: synchronize completed both event records.
            unsafe { timing.publish(request.request_id(), request.layer(), request.rows(), executor_id, "native")?; }
        }
        let executed_us = started.map(|t| t.elapsed().as_micros() as u64);
        if destination.is_none() {
            self.library.copy_d2h(&mut exchange.partials[..bytes], CuteafdDeviceBuffer { bytes, ..output })?;
        }
        if let (Some(timing), Some(started)) = (&self.timing, started) {
            let total_us = started.elapsed().as_micros() as u64;
            let (kernel_us, compact_us) = unsafe { timing.elapsed_us()? };
            let mut histogram = vec![0u32; cuteafd_core::expert_geometry().experts as usize];
            for &expert in &exchange.ids[..routes] {
                // Ownership-masked routes carry the out-of-range sentinel and
                // are not real expert work.
                if let Some(slot) = histogram.get_mut(expert as usize) {
                    *slot += 1;
                }
            }
            let active_experts = histogram.iter().filter(|&&count| count != 0).count();
            // Exact expert-local M=1..16; final bin is M>=17. These counts
            // describe routed rows before the kernel pads them to its M tile.
            let mut expert_rows_histogram = [0u32; 17];
            let mut expert_rows_tail_routes = 0u32;
            for &count in &histogram {
                if count > 0 {
                    expert_rows_histogram[count.min(17) as usize - 1] += 1;
                    if count > 16 {
                        expert_rows_tail_routes += count;
                    }
                }
            }
            let unique_expert_weight_bytes =
                self._weights.budget().resident_bytes / histogram.len() * active_experts;
            tracing::debug!(target: "cuteafd::expert_timing",
                layer, executor_id, rows=request.rows(), active_experts, direct_registered_output=destination.is_some(),
                kernel_capacity=self.execution_state(request.rows())?.0.info().capacity_rows,
                max_expert_rows=histogram.iter().copied().max().unwrap_or(0),
                ?expert_rows_histogram, expert_rows_tail_routes,
                unique_expert_weight_bytes, output_bytes=bytes,
                upload_us=uploaded_us.unwrap(), kernel_us, compact_us,
                execution_host_us=executed_us.unwrap()-uploaded_us.unwrap(),
                download_us=total_us-executed_us.unwrap(), total_us,
                "native expert execution");
        }
        Ok(())
    }
}

#[cfg(test)]
mod mapped_tests;

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    fn rows_above_live_are_refused_even_when_the_compiled_export_covers_them() {
        validate_capacity(4096, 2048).unwrap();
        validate_live_rows(2048, 2048).unwrap();
        let error = validate_live_rows(2049, 2048).unwrap_err();
        assert_eq!(error, NativeExpertCapacityError::RowsExceedLive { rows: 2049, live: 2048 });
        let error = anyhow::Error::from(error);
        assert!(error.downcast_ref::<NativeExpertCapacityError>().is_some());
        assert_eq!(error.to_string(), "native expert rows 2049 exceed live capacity 2048");
        assert!(validate_capacity(1024, 2048).is_err());
        assert!(validate_capacity(4096, 0).is_err());
        assert!(validate_capacity(4097, 2048).is_err());
    }

    #[test]
    fn compiled_scratch_is_retained_but_row_storage_uses_live_extent() {
        let info = cuteafd_ffi::V41ExpertInfo { role: 1, capacity_rows: 4096,
            hidden_size: 5120, topk: 8, input_dtype: 7, scratch_bytes: 1 << 30,
            ..Default::default() };
        let compiled = ExpertExecutionBudget::from_info(info, 4096, 128, 256).unwrap();
        let live = ExpertExecutionBudget::from_info(info, 2048, 128, 256).unwrap();
        assert_eq!(live.scratch_bytes, compiled.scratch_bytes);
        assert_eq!(live.decode_scratch_bytes, compiled.decode_scratch_bytes);
        assert_eq!(live.small_scratch_bytes, compiled.small_scratch_bytes);
        assert_eq!(live.hidden_bytes * 2, compiled.hidden_bytes);
        assert_eq!(live.routing_bytes * 2, compiled.routing_bytes);
        assert_eq!(live.output_and_shared_bytes * 2, compiled.output_and_shared_bytes);
    }
}

#[cfg(test)]
mod timing_role_tests {
    /// The timing gate is the same predicate that selects the compact output
    /// and reducer, so the Spark shard families (roles 5/6/7) and the legacy
    /// grouped Spark TP4 (role 1) all publish the routed breakdown; the dSpark,
    /// full-width RTX and RTX TP2 roles do not.
    #[test]
    fn routed_timing_is_enabled_for_the_compact_output_roles() {
        for role in [1u32, 5, 6, 7] {
            assert!(super::compact_output_role(role), "role {role} must be timed");
        }
        for role in [0u32, 2, 3, 4, 8, u32::MAX] {
            assert!(!super::compact_output_role(role), "role {role} must not be timed");
        }
    }

    /// A pure TP6EP1 worker must allocate the compact output buffer and load
    /// the compact reducer exactly like the other Spark shard families; a
    /// missing role 7 would silently fall back to the uncompacted path.
    #[test]
    fn pure_tp6_role_selects_the_compact_output_path() {
        assert!(super::compact_output_role(
            crate::shared::spark_topology::SPARK_TP6_ROLE
        ));
        assert_eq!(crate::shared::spark_topology::SPARK_TP6_ROLE, 7);
    }
}

#[cfg(test)]
mod exchange_tests {
    use super::HostExpertExchange;
    #[test]
    fn exchange_charges_bf16_rows_and_route_extents() {
        let geometry = cuteafd_core::expert_geometry();
        for capacity in [1, 16, 64, 65, 4096] {
            let exchange = HostExpertExchange::new(capacity).unwrap();
            assert_eq!(exchange.ids.len(), capacity as usize * geometry.topk as usize);
            assert_eq!(exchange.routing.len(), exchange.ids.len());
            assert_eq!(exchange.partials.len(), capacity as usize * geometry.hidden as usize * 2);
            assert_eq!(HostExpertExchange::bytes_for(capacity).unwrap(),
                exchange.ids.len() * 4 + exchange.routing.len() * 4 + exchange.partials.len());
        }
        for capacity in [0, 4097, u32::MAX] {
            assert!(HostExpertExchange::new(capacity).is_err());
            assert!(HostExpertExchange::bytes_for(capacity).is_err());
        }
    }
}
