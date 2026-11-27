//! Spark request adapter for the FP8 family: FP8 K32 wire rows and routes in,
//! the compact BF16 rank partial out, written straight into the registered
//! send slot when the transport permits it.
use super::{Fp8Experts, Fp8Weights};
use crate::shared::experts::execution::HostExpertExchange;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{
    expert::BackboneRequest, ExpertProtocolV2DeviceResponseRef, ExpertProtocolV2ResponseRef, ExpertV2Dtype,
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};
use std::rc::Rc;
use crate::shared::memory::Tag;
use super::super::execution::waves::{Arrival, Wave, WaveStaging};

pub(crate) struct Fp8Worker<'a> {
    // Drain the stream before dropping the experts or the inputs.
    stream: Rc<LoadStream<'a>>,
    experts: Fp8Experts<'a, Rc<Fp8Weights<'a>>>,
    library: &'a NativeLibrary,
    inputs: [DeviceAllocation<'a>; 3],
    output: DeviceAllocation<'a>,
    reply: super::super::reply::Fp8Reply<'a>,
    capacity: usize,
    layer: usize,
    executor_id: u64,
    timing: Option<Rc<std::cell::RefCell<super::super::execution::timing::WorkerTiming<'a>>>>,
    /// Mapped requests return queued; each reply waits only for its unit fence.
    deferred: bool,
    pending: super::super::execution::timing::PendingUnits<'a>,
    /// The owner stream's pace clock, shared with every group on that stream.
    clock: super::super::execution::timing::PaceClock<'a>,
    route_staging: super::super::service::routes::RouteStaging<'a>,
    /// Merged-wave staging of a prefill owner with a row-addressed program.
    waves: Option<WaveStaging<'a>>,
    timing_units: std::collections::VecDeque<bool>,
}

impl Drop for Fp8Worker<'_> {
    fn drop(&mut self) {
        // Shared stream owners may outlive this group's allocations.
        crate::shared::decode_graph::fatal_drain(self.synchronize(), "FP8 worker inputs");
    }
}

fn device_part(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes), "segment device extent overflow");
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

/// The package's ingress row: FP8 K32 wire rows, exact NVFP4 wire rows, or BF16 rows.
fn input_row_bytes(info: &cuteafd_ffi::fp8_moe::Fp8MoeInfo) -> usize { info.input_row_bytes() }

/// The input dtype a package's row-addressed program reads.
fn segment_input_dtype(info: &cuteafd_ffi::fp8_moe::Fp8MoeInfo) -> ExpertV2Dtype {
    if info.nvfp4_input { ExpertV2Dtype::Nvfp4E2m1E4m3K16Amax }
    else if info.wire_input { ExpertV2Dtype::Fp8E4m3Ue8m0K32 } else { ExpertV2Dtype::Bf16 }
}

/// Queue one wave's expert math on `stream` (no copies, no fences).
#[allow(clippy::too_many_arguments)]
fn launch_wave(experts: &Fp8Experts<'_, Rc<Fp8Weights<'_>>>, reply: &super::super::reply::Fp8Reply<'_>,
    timing: &Option<Rc<std::cell::RefCell<super::super::execution::timing::WorkerTiming<'_>>>>,
    timing_units: &mut std::collections::VecDeque<bool>, stream: *mut std::ffi::c_void, wave: &Wave<'_>) -> Result<()> {
    // SAFETY: every member's slice and reply, the routes, tables, weights and
    // admitted scratch remain live until the member fences recorded after this.
    unsafe {
        if let Some(timing) = timing { timing.borrow().record(0, stream)?; }
        if wave.output_dtype != ExpertV2Dtype::Bf16 {
            experts.run_reply(reply, wave.layer, wave.rows, [wave.input, wave.ids, wave.weights, wave.output],
                false, wave.segmented, wave.output_dtype, stream)?;
        } else if wave.segmented {
            experts.run_segments(wave.layer, wave.rows, [wave.input.ptr, wave.ids.ptr, wave.weights.ptr, wave.output.ptr],
                stream)?;
        } else {
            experts.run(wave.layer, wave.rows, wave.input.ptr, wave.ids.ptr, wave.weights.ptr, wave.output.ptr, stream)?;
        }
        if let Some(timing) = timing { timing.borrow().record(1, stream)?; }
    }
    timing_units.extend((0..wave.members.len()).map(|index| index == 0));
    Ok(())
}

fn validate_worker_layers(layers: &[usize]) -> Result<()> {
    ensure!(!layers.is_empty(), "FP8 worker has no layers");
    super::validate_layer_ids(layers)
}

impl<'a> Fp8Worker<'a> {
    pub(crate) fn capacity(&self) -> usize { self.capacity }
    /// Merged-wave device I/O beyond the ordinary owner planes: address and
    /// route tables at the widest merged wave, and every staged unit's input
    /// slice at the package's ingress row (past a BF16 sibling's plane).
    pub(crate) fn segment_io_bytes(module: &cuteafd_ffi::fp8_moe::Fp8MoeModule,
        extent: cuteafd_core::OwnerExtent, bf16_sibling: bool) -> Result<usize> {
        if super::segment_rows(module, extent).is_none() { return Ok(0) }
        let info = module.info();
        let row = input_row_bytes(info);
        let sibling = bf16_sibling.then_some(info.hidden * 2);
        let input = cuteafd_core::spark_executions::segment_input_bytes(extent.capacity,
            extent.staged_rows().context("segmented owner has no staged rows")?, row.try_into()?,
            sibling.map(u32::try_from).transpose()?).context("segmented input bytes overflow")? as usize;
        let old = extent.capacity as usize * sibling.unwrap_or(row);
        Ok(WaveStaging::table_bytes(extent, info.topk).context("segment table overflow")? + input - old)
    }

    pub(crate) fn prepare_waves(&mut self, extent: cuteafd_core::OwnerExtent) -> Result<()> {
        if super::segment_rows(&self.experts.module, extent).is_none() { return Ok(()) }
        let info = self.experts.module.info();
        let row = input_row_bytes(info);
        let offset = if self.experts.bf16_module.is_some() { self.capacity * info.hidden * 2 } else { 0 };
        let staged = extent.staged_rows().context("segmented owner has no staged rows")? as usize;
        let plane = device_part(self.inputs[0].buffer, offset, staged * row)?;
        let minimum = self.experts.module.segment_recipe().map_or(1, |recipe| recipe.minimum_rows as usize);
        self.waves = WaveStaging::new(self.library, extent, plane, row, minimum)?;
        self.timing_units = std::collections::VecDeque::with_capacity(extent.buffered_units as usize);
        Ok(())
    }

    pub(crate) fn has_staged(&self) -> bool {
        self.waves.as_ref().is_some_and(WaveStaging::has_staged)
    }

    #[cfg(test)]
    pub(crate) fn executor_id(&self) -> u64 { self.executor_id }
    #[cfg(test)]
    pub(crate) fn pending_is_empty(&self) -> bool { self.pending.is_empty() }
    #[cfg(test)]
    pub(crate) fn wave_flushes(&self) -> u64 { self.waves.as_ref().map_or(0, |waves| waves.flushes) }
    #[cfg(test)]
    pub(crate) fn merged_waves(&self) -> u64 { self.waves.as_ref().map_or(0, |waves| waves.merged) }

    pub(crate) fn cancel_staged(&mut self, input_address: usize) -> Result<bool> {
        let Some(waves) = &mut self.waves else { return Ok(false) };
        waves.cancel(&mut self.pending, input_address)
    }

    pub(crate) fn flush_staged(&mut self) -> Result<()> {
        let Some(waves) = &mut self.waves else { return Ok(()) };
        let (experts, reply, timing, timing_units) = (&self.experts, &self.reply, &self.timing, &mut self.timing_units);
        let stream = self.stream.raw;
        waves.flush(&mut self.pending, &self.route_staging, stream,
            &mut |wave| launch_wave(experts, reply, timing, timing_units, stream, wave))
    }

    fn stage(&mut self, request: &BackboneRequest<'_>, input: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer, executor_id: u64) -> Result<bool> {
        let Some(waves) = &mut self.waves else { return Ok(false) };
        ensure!(request.output_dtype() == ExpertV2Dtype::Bf16 || request.partial_plan()?.is_none(),
            "fused FP8 reply requires whole combined rows");
        if request.input_dtype() != segment_input_dtype(self.experts.module.info()) {
            waves.bypass(request.rows(), super::super::execution::waves::Bypass::Ingress);
            return Ok(false)
        }
        ensure!(executor_id == self.executor_id && request.layer() as usize == self.experts.layers[self.layer].layer,
            "staged expert identity mismatch");
        let (experts, reply, timing, timing_units) = (&self.experts, &self.reply, &self.timing, &mut self.timing_units);
        let stream = self.stream.raw;
        waves.stage(Arrival { request, input, output, layer: self.layer, executor_id }, &mut self.pending,
            &mut self.route_staging, stream, &mut |wave| launch_wave(experts, reply, timing, timing_units, stream, wave))
    }

    /// Exact retained input/routes/output bytes (scratch is admitted separately).
    pub(crate) fn io_bytes(info: &cuteafd_ffi::fp8_moe::Fp8MoeInfo, bf16_sibling: bool,
        capacity: usize) -> Result<usize> {
        if info.nvfp4_input {
            return cuteafd_ffi::fp8_moe::fp8_slot_io_bytes_for_input(info.hidden, info.topk,
                info.input_row_bytes(), capacity);
        }
        cuteafd_ffi::fp8_moe::fp8_slot_io_bytes(info.hidden, info.topk, info.wire_input,
            bf16_sibling, capacity)
    }

    pub(crate) fn workspace_bytes(experts: &Fp8Weights<'_>, capacity: usize) -> Result<usize> {
        Self::io_bytes(experts.module.info(), experts.bf16_module.is_some(), capacity)?
            .checked_add(experts.slot_scratch_bytes(capacity)?)
            .context("native slot workspace size overflow")
    }

    pub(crate) fn new(library: &'a NativeLibrary, experts: Rc<Fp8Weights<'a>>, capacity: u32) -> Result<Self> {
        Self::with_scratch(library, experts, capacity, None)
    }

    pub(crate) fn with_scratch(library: &'a NativeLibrary, experts: Rc<Fp8Weights<'a>>, capacity: u32,
        scratch: Option<Rc<DeviceAllocation<'a>>>) -> Result<Self> {
        Self::with_extent(library, experts, capacity, scratch, None)
    }

    pub(crate) fn with_extent(library: &'a NativeLibrary, experts: Rc<Fp8Weights<'a>>, capacity: u32,
        scratch: Option<Rc<DeviceAllocation<'a>>>, extent: Option<cuteafd_core::OwnerExtent>) -> Result<Self> {
        let geometry = cuteafd_core::expert_geometry();
        let (hidden, topk, capacity) = (geometry.hidden as usize, geometry.topk as usize, capacity as usize);
        validate_worker_layers(&experts.layers.iter().map(|layer| layer.layer).collect::<Vec<_>>())?;
        ensure!(experts.module.info().capacity_for(capacity).is_some(), "FP8 package lacks capacity {capacity}");
        ensure!(experts.wire_input() || experts.module.info().weights.requires_bf16_input(),
            "Spark workers need a wire-input package, or native BF16/NVFP4-input weights");
        let experts = match scratch {
            Some(scratch) => Fp8Experts::from_scratch(experts, capacity, scratch)?,
            None => Fp8Experts::from_shared(library, experts, capacity)?,
        };
        let input_bytes = match extent.filter(|&e| super::segment_rows(&experts.module, e).is_some()) {
            Some(extent) => cuteafd_core::spark_executions::segment_input_bytes(capacity as u32,
                extent.staged_rows().context("segmented owner has no staged rows")?,
                input_row_bytes(experts.module.info()).try_into()?,
                experts.bf16_module.is_some().then_some(u32::try_from(hidden * 2)?))
                .context("segmented input extent overflow")? as usize,
            None => capacity * if experts.module.info().nvfp4_input { cuteafd_core::nvfp4_wire_row_bytes(hidden) }
                else if !experts.wire_input() || experts.bf16_module.is_some() { hidden * 2 }
                else { hidden + hidden / 32 },
        };
        let inputs = [
            DeviceAllocation::new(library, input_bytes, Tag::workspace("experts/fp8-inputs"))?,
            DeviceAllocation::new(library, capacity * topk * 4, Tag::workspace("experts/fp8-inputs"))?,
            DeviceAllocation::new(library, capacity * topk * 4, Tag::workspace("experts/fp8-inputs"))?,
        ];
        let output = DeviceAllocation::new(library, capacity * hidden * 2, Tag::workspace("experts/fp8-output"))?;
        let executor_id = cuteafd_transport::expert::SparkWorld::canonical(experts.tp)?.topology().executor_id(experts.rank)?;
        let timing = super::super::execution::timing::WorkerTiming::from_env(library, experts.rank)?
            .map(|timing| Rc::new(std::cell::RefCell::new(timing)));
        Ok(Self {
            stream: Rc::new(LoadStream { library, raw: library.cuda_stream_create()? }),
            experts,
            library,
            inputs,
            output,
            reply: super::super::reply::Fp8Reply::load(library)?,
            capacity,
            layer: 0,
            executor_id,
            timing,
            deferred: false,
            waves: None,
            timing_units: Default::default(),
            pending: super::super::execution::timing::PendingUnits::new(library, 1)?,
            clock: Default::default(),
            route_staging: super::super::service::routes::RouteStaging::new(library, 1, capacity)?,
        })
    }

    pub(crate) fn installed_grouping(&self, loaded: cuteafd_transport::worker_selection::WorkerSelection,
        rank: usize) -> Result<cuteafd_transport::worker_selection::InstalledExpertGrouping> {
        self.experts.installed_grouping(loaded, rank, self.capacity)
    }

    /// All homogeneous groups in one slot run serially; their gap clock must
    /// follow the slot, not skip compute from a different weight group.
    pub(crate) fn share_owner(&mut self, other: &Self) {
        self.stream = other.stream.clone();
        if let (Some(waves), Some(shared)) = (&mut self.waves, &other.waves) { waves.share_copy_stream(shared); }
        self.timing = other.timing.clone();
        self.clock = other.clock.clone();
    }

    /// Leave mapped requests queued at return (see `deferred`).
    pub(crate) fn defer(&mut self, units: usize, staging_rows: usize, owner: cuteafd_core::ExecutionOwner) -> Result<()> {
        ensure!(self.pending.is_empty(), "cannot resize an active FP8 queue");
        ensure!(staging_rows > 0 && staging_rows <= self.capacity, "FP8 staging exceeds owner rows");
        self.pending = super::super::execution::timing::PendingUnits::with_clock(self.library, units, owner,
            self.clock.clone())?;
        self.route_staging.resize(self.library, units, staging_rows)?;
        self.deferred = true;
        Ok(())
    }

    /// Publish deferred diagnostics after the owner's existing completion check.
    pub(crate) fn complete(&mut self) -> Result<Option<super::super::execution::timing::UnitPace>> {
        let Some(((request_id, layer, rows, executor_id), pace)) = self.pending.pop()? else { return Ok(None) };
        let timed = self.timing_units.pop_front().unwrap_or(true);
        if let Some(timing) = &mut self.timing.as_mut().filter(|_| timed) {
            // SAFETY: the caller observed this stream drained after both records.
            unsafe { timing.borrow_mut().publish(request_id, layer, rows, executor_id, "fp8_moe")?; }
        }
        Ok(Some(pace))
    }

    pub(crate) fn unit_ready(&self) -> Result<bool> { self.pending.ready() }

    pub(crate) fn synchronize(&self) -> Result<()> {
        // SAFETY: this owner retains its stream and every queued buffer.
        unsafe {
            if let Some(waves) = &self.waves { self.library.cuda_stream_synchronize(waves.copy_stream())?; }
            self.library.cuda_stream_synchronize(self.stream.raw)?;
        }
        Ok(())
    }

    /// Whether every request queued on this execution's stream has completed.
    pub(crate) fn stream_idle(&self) -> Result<bool> {
        // SAFETY: the owned stream is live for the worker's lifetime.
        unsafe { self.library.cuda_stream_query(self.stream.raw) }
    }

    pub(crate) fn bind_layer(&mut self, layer: usize) -> Result<()> {
        ensure!(layer < self.experts.layers.len(), "requested FP8 layer is not resident");
        self.layer = layer;
        Ok(())
    }

    fn execute(&mut self, request: &BackboneRequest<'_>, executor_id: u64, exchange: &mut HostExpertExchange,
        destination: Option<CuteafdDeviceBuffer>, hidden_view: Option<CuteafdDeviceBuffer>)
        -> Result<CuteafdDeviceBuffer> {
        ensure!(request.layer() as usize == self.experts.layers[self.layer].layer,
            "request does not match the selected FP8 layer");
        ensure!(executor_id == self.executor_id, "FP8 response executor identity mismatch");
        if self.deferred {
            if let (Some(input), Some(output)) = (hidden_view, destination) {
                if self.stage(request, input, output, executor_id)? { return Ok(output) }
            }
        }
        self.flush_staged()?;
        // Host-only fallback can alias the wire arena; retire prior consumers.
        if self.waves.is_some() && (hidden_view.is_none() || self.experts.bf16_module.is_none()) { self.synchronize()?; }
        let rows = request.rows() as usize;
        ensure!(rows > 0 && rows <= self.capacity, "FP8 request exceeds capacity");
        // FP8 K32 wire rows, or BF16 rows when the BF16-input package is loaded,
        // or the exact pre-quantized NVFP4 rows the NVFP4-wire package reads.
        let nvfp4 = self.experts.module.info().nvfp4_input;
        if nvfp4 { request.require_input_dtype(ExpertV2Dtype::Nvfp4E2m1E4m3K16Amax as u32)?; }
        let bf16 = !nvfp4 && request.require_input_dtype(ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32).is_err();
        if bf16 {
            ensure!(!self.experts.wire_input() || self.experts.bf16_module.is_some(),
                "BF16 expert input needs the BF16-input FP8 package (fp8-<family>-bf16) on this rank");
            request.require_input_dtype(ExpertV2Dtype::Bf16 as u32)?;
        }
        ensure!(bf16 || nvfp4 || self.experts.wire_input(),
            "native BF16/NVFP4 Spark input must be BF16 or exact NVFP4 rows, not FP8 K32");
        ensure!(request.hidden().len() <= self.inputs[0].buffer.bytes, "FP8 request rows exceed the input buffer");
        let bytes = request.plane_bytes()?;
        let routes = rows * cuteafd_core::expert_geometry().topk as usize;
        let stage = self.pending.next_slot()?;
        self.route_staging.pack(stage, request)?;
        ensure!(cfg!(target_endian = "little"), "native exchange requires little-endian storage");
        // Every copy follows the previous unit's consumers on this owner stream;
        // the transport retains each mapped source independently until its reply.
        let hidden_bytes = request.hidden().len();
        // SAFETY: the owner stream outlives every queued unit (#762 pace).
        unsafe { self.pending.arrive(self.stream.raw)?; }
        let device = self.inputs[0].buffer.device_id;
        let mut hidden = self.inputs[0].buffer.ptr;
        match hidden_view.filter(|view| view.bytes >= hidden_bytes) {
            // The kernel reads the rows from its input pointer exactly as it
            // would read the resident copy, and the transport retains the
            // mapped frame until this unit's fence, so an aligned same-device
            // view is input storage: read it where it landed and skip the
            // LPDDR-to-LPDDR copy pass.
            Some(view) if view.ptr as usize % 16 == 0 && view.device_id == device => hidden = view.ptr,
            // SAFETY: the view is device-visible request storage of at least
            // `hidden_bytes`, retained by the transport until this request's
            // response is emitted, which follows the stream synchronize below.
            Some(view) => unsafe {
                self.library.copy_d2d_async(self.inputs[0].buffer,
                    CuteafdDeviceBuffer { bytes: hidden_bytes, ..view }, hidden_bytes, self.stream.raw)?
            },
            None => self.library.copy_h2d(self.inputs[0].buffer, request.hidden())?,
        }
        // SAFETY: this unit retains its packed slot through the FIFO completion fence.
        unsafe { self.route_staging.upload(stage, 0, routes, self.inputs[1].buffer,
            self.inputs[2].buffer, self.stream.raw)?; }
        let compressed = request.output_dtype() != ExpertV2Dtype::Bf16;
        ensure!(!compressed || destination.is_some(), "compressed replies require a mapped response slot");
        let output = match destination {
            Some(output) => output,
            None => self.output.buffer,
        };
        ensure!(output.bytes >= if compressed { rows * request.output_row_bytes()? as usize } else { bytes },
            "FP8 response slot is too small");
        // SAFETY: inputs, output and the resident layer are live device memory of
        // the documented extents; the stream is synchronized below.
        unsafe {
            if let Some(timing) = &self.timing { timing.borrow().record(0, self.stream.raw)?; }
            if compressed {
                ensure!(request.partial_plan()?.is_none(), "FP8 fused reply requires combined rows");
                self.experts.run_reply(&self.reply, self.layer, rows,
                    [CuteafdDeviceBuffer { ptr: hidden, bytes: hidden_bytes, ..self.inputs[0].buffer },
                     self.inputs[1].buffer, self.inputs[2].buffer, output], bf16, false,
                    request.output_dtype(), self.stream.raw)?;
            } else if bf16 {
                self.experts.run_bf16(self.layer, rows, hidden, self.inputs[1].buffer.ptr,
                    self.inputs[2].buffer.ptr, output.ptr, self.stream.raw)?;
            } else {
                self.experts.run(self.layer, rows, hidden, self.inputs[1].buffer.ptr,
                    self.inputs[2].buffer.ptr, output.ptr, self.stream.raw)?;
            }
            if let Some(timing) = &self.timing { timing.borrow().record(1, self.stream.raw)?; }
            // The transport holds both mapped slots until this unit's fence,
            // independent of later work queued on the owner's stream.
            if self.deferred && destination.is_some() {
                self.pending.record((request.request_id(), request.layer(), request.rows(), executor_id), self.stream.raw)?;
                if self.waves.is_some() { self.timing_units.push_back(true); }
                return Ok(output);
            }
            self.library.cuda_stream_synchronize(self.stream.raw)?;
        }
        if let Some(timing) = &mut self.timing {
            // SAFETY: the existing worker barrier completed both event records.
            unsafe { timing.borrow_mut().publish(request.request_id(), request.layer(), request.rows(), executor_id, "fp8_moe")?; }
        }
        if destination.is_none() {
            ensure!(exchange.partials.len() >= bytes, "FP8 host exchange is too small");
            self.stream.copy_d2h(&mut exchange.partials[..bytes], output)?;
        }
        Ok(output)
    }

    /// # Safety
    /// The transport exclusively owns a GPU-accessible send slot on this device
    /// and retains it through the response send completion.
    pub(crate) unsafe fn execute_mapped_request(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, slot: CuteafdDeviceBuffer, hidden: Option<CuteafdDeviceBuffer>)
        -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        if !request.permits_device_response() || slot.bytes < prefix + bytes {
            return Ok(None);
        }
        ensure!(!slot.ptr.is_null(), "null FP8 response slot");
        let output = CuteafdDeviceBuffer { ptr: slot.ptr.cast::<u8>().add(prefix).cast(), bytes, ..slot };
        let response = request.response_device(executor_id, output)?;
        self.execute(request, executor_id, exchange, Some(output), hidden)?;
        Ok(Some(response))
    }

    /// # Safety
    /// Transport retains all mapped frames through this unit's completion and send.
    pub(crate) unsafe fn execute_mapped_payload(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, mapped: cuteafd_transport::ProtocolV2RequestDevicePayload)
        -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        let Some(slot) = mapped.response_slot else { return Ok(None) };
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        if !request.permits_device_response() || slot.bytes < prefix + bytes { return Ok(None); }
        // SAFETY: checked prefix/plane extent stays in the exclusively retained send slot.
        let output = CuteafdDeviceBuffer { ptr: unsafe { slot.ptr.cast::<u8>().add(prefix).cast() }, bytes, ..slot };
        let response = request.response_device(executor_id, output)?;
        self.execute(request, executor_id, exchange, Some(output), Some(mapped.hidden_payload))?;
        Ok(Some(response))
    }

    pub(crate) fn execute_host_chunks<F>(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, row_indices: &mut [u32], max_frame_bytes: usize, mut sink: F) -> Result<()>
    where
        F: FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    {
        let chunk_rows = request.response_chunk_rows(max_frame_bytes)?;
        ensure!(row_indices.len() >= chunk_rows as usize, "response row-index scratch is too short");
        self.execute(request, executor_id, exchange, None, None)?;
        let stride = cuteafd_core::expert_geometry().row_bytes() as usize;
        for start in (0..request.rows()).step_by(chunk_rows as usize) {
            let end = start.saturating_add(chunk_rows).min(request.rows());
            sink(request.response_chunk(executor_id, start,
                &exchange.partials[start as usize * stride..end as usize * stride], row_indices, max_frame_bytes)
                .context("FP8 response chunk")?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixture allocations share device 7 in the process-wide memory ledger.
    static FIXTURE_LEDGER: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn native_slots_own_distinct_capacity_sized_scratch_and_charge_workspace_once() -> Result<()> {
        use cuteafd_ffi::{fp8_moe::Fp8MoeModule, memory_ledger::{self, Space},
            native_library_lifetime_fixture::Fixture};
        let _ledger = FIXTURE_LEDGER.lock().unwrap_or_else(|error| error.into_inner());
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 30)?;
        std::fs::hard_link(fixture.directory().join("liblifetime_fixture.so"),
            fixture.directory().join("libcuteafd_fp8moe.so"))?;
        let geometry = cuteafd_core::expert_geometry();
        let baseline = memory_ledger::snapshot().total(Space::Device, 7);
        for format in [1u32, 2, 4, 5] {
            for capacity in [1usize, 17] {
                let wire = matches!(format, 1 | 2);
                let mut words = [6 | format << 8 | 1 << 16 | 1 << 24, geometry.hidden,
                    128, 2, geometry.topk, 256, 2, if wire { 7 } else { 1 }, 0,
                    2, 16, 4096, 0, 0, 0, 0, 1.0f32.to_bits(), 0];
                fixture.configure_fp8(&library, &words, &[1024, 4096, 0, 0, 0, 0])?;
                // SAFETY: this test compiled the CPU-only native ABI above.
                let module = unsafe { Fp8MoeModule::load(fixture.directory())? };
                let bf16_module = if wire {
                    words[7] = 1;
                    fixture.configure_fp8(&library, &words, &[2048, 8192, 0, 0, 0, 0])?;
                    // SAFETY: the same trusted CPU ABI snapshots independent metadata per module.
                    Some(unsafe { Fp8MoeModule::load(fixture.directory())? })
                } else { None };
                let resident = {
                    let _scope = memory_ledger::scope("experts/weights");
                    DeviceAllocation::new(&library, 512, crate::shared::memory::TEST)?
                };
                let weights = Rc::new(Fp8Weights {
                    layers: vec![super::super::Fp8Layer { layer: 3,
                        slice: cuteafd_core::SparkSlice { unit_map: None, layer: 3, first: 0,
                            width: 128, stored_width: 128, tiers: vec![], projection_counts: vec![] },
                        regions: vec![Some(resident.into())] }],
                    module, bf16_module, tp: 2, rank: 0, weight_read_bytes: 256,
                });
                let scratch = match (wire, capacity) { (true, 1) => 2048, (true, _) => 8192,
                    (false, 1) => 1024, (false, _) => 4096 };
                assert_eq!(weights.resident_bytes(), 512);
                assert_eq!(weights.slot_scratch_bytes(capacity)?, scratch);
                let io = Fp8Worker::io_bytes(weights.module.info(), wire, capacity)?;
                let workspace = Fp8Worker::workspace_bytes(&weights, capacity)?;
                assert_eq!(workspace, io + scratch);
                let workers = {
                    let _scope = memory_ledger::scope("experts/workspace/slot");
                    [Fp8Worker::new(&library, weights.clone(), capacity as u32)?,
                        Fp8Worker::new(&library, weights.clone(), capacity as u32)?]
                };
                assert!(Rc::ptr_eq(&workers[0].experts.weights, &workers[1].experts.weights));
                assert_ne!(workers[0].experts.scratch.buffer.ptr, workers[1].experts.scratch.buffer.ptr);
                for worker in &workers {
                    assert_eq!(worker.experts.scratch_bytes(), scratch);
                    assert_eq!(worker.inputs.iter().map(|input| input.buffer.bytes).sum::<usize>()
                        + worker.output.buffer.bytes, io);
                }
                let ledger = memory_ledger::snapshot();
                let scopes = ledger.by_scope(Space::Device, 7);
                assert_eq!(scopes["experts/weights"], 512);
                assert_eq!(scopes["experts/workspace/native_scratch"], 2 * scratch);
                assert_eq!(scopes["experts/workspace/slot"], 2 * io);
                assert_eq!(ledger.total(Space::Device, 7) - baseline, 512 + 2 * workspace);
                let row = ledger.rows.iter().find(|row| row.key.device == 7
                    && row.key.scope == "experts/workspace/native_scratch").unwrap();
                assert_eq!(row.allocations, 2);
                drop(workers);
                assert_eq!(memory_ledger::snapshot().total(Space::Device, 7) - baseline, 512);
                drop(weights);
                assert_eq!(memory_ledger::snapshot().total(Space::Device, 7), baseline);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn selected_compressed_replies_dispatch_core_then_epilogue_including_staged_pairs() -> Result<()> {
        use crate::shared::memory::HostAllocation;
        use cuteafd_ffi::{fp8_moe::Fp8MoeModule, native_library_lifetime_fixture::Fixture};
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RequestView,
            ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry, ExpertV2SourceKind,
            ProtocolV2RequestDevicePayload, EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_UE8M0_K32,
            EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3};
        let _ledger = FIXTURE_LEDGER.lock().unwrap_or_else(|error| error.into_inner());
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        std::fs::hard_link(fixture.directory().join("liblifetime_fixture.so"),
            fixture.directory().join("libcuteafd_fp8moe.so"))?;
        let geometry = cuteafd_core::expert_geometry();
        let words = [6 | 1 << 8 | 1 << 16 | 1 << 24, geometry.hidden,
            128, geometry.experts, geometry.topk, 256, 2, 7, 0,
            2, 16, 4096, 0, 0, 0, 0, 1.0f32.to_bits(), 0];
        let scratch_bytes = (1 << 20) + 2 * 769 * geometry.topk as u64 * geometry.hidden as u64 * 2;
        fixture.configure_fp8(&library, &words, &[scratch_bytes, scratch_bytes, 0, 0, 0, 0])?;
        let replies = (0..4).map(|index| serde_json::json!({
            "name": format!("expert_reply_fp8_h{}_k{}_{}{}", geometry.hidden, geometry.topk,
                index % 2 + 1, if index >= 2 { "_segmented" } else { "" }),
            "pointers": ["y", "positions", "routes", "reply"],
            "scalars": [{"type": "int32"}]
        })).collect::<Vec<_>>();
        fixture.register_programs(&library, &replies)?;
        for index in 0..4 { fixture.register_reply(&library, index, 0x1166, b'1' + index as u8)?; }
        fixture.configure_fp8_segments(&library)?;
        // SAFETY: this test built the CPU-only ABI and owns its buffers below.
        let module = unsafe { Fp8MoeModule::load(fixture.directory())? };
        let source_scratch = DeviceAllocation::new(&library, 1 << 20, crate::shared::memory::TEST)?;
        let (y, positions) = module.reply_sources(source_scratch.buffer, 2, false)?;
        assert_eq!(y.bytes, 2 * geometry.topk as usize * geometry.hidden as usize * 2);
        assert_eq!(positions.bytes, 2 * geometry.topk as usize * 4);
        for rows in [0, 4097] { assert!(module.reply_sources(source_scratch.buffer, rows, false).is_err()); }
        assert!(module.reply_sources(CuteafdDeviceBuffer { bytes: 1024, ..source_scratch.buffer }, 2, false).is_err());
        assert!(module.reply_sources(CuteafdDeviceBuffer { ptr: std::ptr::null_mut(), ..source_scratch.buffer }, 2, false).is_err());
        let weights = Rc::new(Fp8Weights {
            layers: vec![super::super::Fp8Layer { layer: 3,
                slice: cuteafd_core::SparkSlice { unit_map: None, layer: 3, first: 0,
                    width: 128, stored_width: 128, tiers: vec![], projection_counts: vec![] },
                regions: vec![None, None, None, None, None, None] }],
            module, bf16_module: None, tp: 2, rank: 0, weight_read_bytes: 256,
        });
        for (format, flag) in [EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_UE8M0_K32,
            EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3].into_iter().enumerate() {
            for staged in [false, true] {
                let extent = cuteafd_core::OwnerExtent { owner: cuteafd_core::ExecutionOwner::Prefill,
                    capacity: 4096, compiled: 4096, staging_rows: 769, buffered_units: 3 };
                let mut worker = Fp8Worker::with_extent(&library, weights.clone(), 4096, None,
                    staged.then_some(extent))?;
                if staged {
                    worker.prepare_waves(extent)?;
                    worker.defer(3, 769, cuteafd_core::ExecutionOwner::Prefill)?;
                }
                let rows = if staged { 769 } else { 2 };
                let mut exchange = HostExpertExchange::new(4096)?;
                let mut frames = Vec::new();
                let mut replies = Vec::new();
                let before = fixture.events()?.len();
                for unit in 0..if staged { 2 } else { 1 } {
                    let mut owned = ExpertProtocolV2Request::new(91 + unit, 17, 3, geometry.hidden,
                        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                        (0..rows).map(|row| ExpertProtocolV2RowDescriptor { row_id: row as u64,
                            source_kind: ExpertV2SourceKind::Prefill, source_request_id: unit,
                            token_position: row as u64, route_offset: row * geometry.topk,
                            route_count: geometry.topk }).collect(),
                        (0..rows * geometry.topk).map(|route| ExpertProtocolV2RouteEntry {
                            row_index: route / geometry.topk, expert_id: route % geometry.experts,
                            gate_weight: 0.25 }).collect(),
                        vec![7; rows as usize * geometry.hidden as usize * 33 / 32])?;
                    owned.header.flags |= flag
                        | cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
                    let encoded = owned.encode()?;
                    let mut frame = HostAllocation::new(&library, encoded.len(), crate::shared::memory::TEST)?;
                    frame.bytes_mut().copy_from_slice(&encoded);
                    let request = BackboneRequest::parse(frame.bytes(), 4096)?;
                    let view = ExpertProtocolV2RequestView::parse(frame.bytes())?;
                    let reply = HostAllocation::new(&library,
                        EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + request.plane_bytes()?, crate::shared::memory::TEST)?;
                    let alias = library.cuda_host_buffer_device_alias(frame.buffer)?;
                    let mapped = ProtocolV2RequestDevicePayload {
                        hidden_payload: CuteafdDeviceBuffer {
                            ptr: view.hidden_payload().as_ptr().cast_mut().cast(),
                            bytes: view.hidden_payload().len(), ..alias }, execution_lane: 0,
                        response_slot: Some(library.cuda_host_buffer_device_alias(reply.buffer)?) };
                    // SAFETY: frame and reply owners survive the explicit drain below.
                    unsafe { worker.execute_mapped_payload(&request, worker.executor_id, &mut exchange, mapped)?; }
                    frames.push(frame);
                    replies.push(reply);
                }
                worker.flush_staged()?;
                worker.synchronize()?;
                let events = fixture.events()?;
                let dispatch: String = events[before..].chars()
                    .filter(|c| matches!(c, 'c' | 's' | '1'..='4')).collect();
                assert_eq!(dispatch, format!("{}{}", if staged { 's' } else { 'c' },
                    char::from(b'1' + format as u8 + if staged { 2 } else { 0 })));
                if staged {
                    assert_eq!(worker.waves.as_ref().unwrap().flushes, 1);
                    while !worker.pending.is_empty() { assert!(worker.unit_ready()?); worker.complete()?; }
                }
                drop(worker);
                drop((frames, replies));
            }
        }
        Ok(())
    }

    /// Immediate mapped BF16 replies read the mapped input in place, skipping its copy.
    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn three_queued_prefill_units_preserve_distinct_routes_inputs_and_replies() -> Result<()> {
        queued_prefill_units(false, false, false)
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn grouped_prefill_orders_descriptors_before_launch_and_cancels_without_launching() -> Result<()> {
        queued_prefill_units(true, false, false)
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn two_received_units_copy_before_one_flush_and_ignore_retired_source_bytes() -> Result<()> {
        queued_prefill_units(true, true, false)
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn bf16_fallback_plane_cannot_overwrite_retained_wire_units() -> Result<()> {
        queued_prefill_units(true, true, true)
    }

    fn queued_prefill_units(segmented: bool, received_pair: bool, bf16_sibling: bool) -> Result<()> {
        use crate::shared::memory::HostAllocation;
        use cuteafd_ffi::{fp8_moe::Fp8MoeModule, native_library_lifetime_fixture::Fixture};
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RequestView,
            ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry, ExpertV2SourceKind,
            ProtocolV2RequestDevicePayload, expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16};
        let _ledger = FIXTURE_LEDGER.lock().unwrap_or_else(|error| error.into_inner());
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 30)?;
        fixture.queued_experts(&library)?;
        std::fs::hard_link(fixture.directory().join("liblifetime_fixture.so"),
            fixture.directory().join("libcuteafd_fp8moe.so"))?;
        let geometry = cuteafd_core::expert_geometry();
        let words = [6 | 1 << 8 | 1 << 16 | 1 << 24, geometry.hidden,
            128, geometry.experts, geometry.topk, 256, 2, 7, 0,
            2, 16, 4096, 0, 0, 0, 0, 1.0f32.to_bits(), 0];
        let bf16_module = if bf16_sibling {
            let mut bf16_words = words;
            bf16_words[7] = 1;
            fixture.configure_fp8(&library, &bf16_words, &[1024, 4096, 0, 0, 0, 0])?;
            // SAFETY: the fixture snapshots the BF16 sibling's metadata independently.
            Some(unsafe { Fp8MoeModule::load(fixture.directory())? })
        } else { None };
        fixture.configure_fp8(&library, &words, &[1024, 4096, 0, 0, 0, 0])?;
        if segmented { fixture.configure_fp8_segments(&library)?; }
        // SAFETY: this test built the trusted CPU ABI immediately above.
        let module = unsafe { Fp8MoeModule::load(fixture.directory())? };
        let layers = [3, 4].into_iter().map(|layer| super::super::Fp8Layer { layer,
            slice: cuteafd_core::SparkSlice { unit_map: None, layer: layer as u32, first: 0,
                width: 128, stored_width: 128, tiers: vec![], projection_counts: vec![] },
            regions: vec![None, None, None, None, None, None] }).collect();
        let weights = Rc::new(Fp8Weights { layers, module, bf16_module,
            tp: 2, rank: 0, weight_read_bytes: 256 });
        let extent = cuteafd_core::OwnerExtent { owner: cuteafd_core::ExecutionOwner::Prefill,
            capacity: 4096, compiled: 4096, staging_rows: 769, buffered_units: 3 };
        let scratch = Rc::new(DeviceAllocation::new(&library, weights.owner_scratch_bytes(extent)?, crate::shared::memory::TEST)?);
        let mut worker = Fp8Worker::with_extent(&library, weights, 4096, Some(scratch), segmented.then_some(extent))?;
        if segmented { worker.prepare_waves(extent)?; }
        worker.defer(3, if segmented { 769 } else { 4096 }, cuteafd_core::ExecutionOwner::Prefill)?;
        let mut exchange = HostExpertExchange::new(4096)?;
        let rows = if segmented { 769u32 } else { 2u32 };
        let hidden = geometry.hidden as usize;
        let topk = geometry.topk as usize;
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let mut frames = Vec::new();
        let mut replies = Vec::new();
        let mut expected = Vec::new();
        for unit in 0..4usize {
            let layer = if segmented { 3 } else { 3 + unit % 2 };
            let input = 7 + unit as u8 * 4;
            let gate = (unit + 1) as f32;
            let ids = (0..rows as usize * topk).map(|route|
                ((unit * 23 + route * 3) % geometry.experts as usize) as u32).collect::<Vec<_>>();
            let mut owned = ExpertProtocolV2Request::new(91 + unit as u64, 17, layer as u32, geometry.hidden,
                ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                (0..rows).map(|row| ExpertProtocolV2RowDescriptor { row_id: row as u64,
                    source_kind: ExpertV2SourceKind::Prefill, source_request_id: unit as u64,
                    token_position: row as u64, route_offset: row * geometry.topk,
                    route_count: geometry.topk }).collect(),
                ids.iter().enumerate().map(|(route, &expert_id)| ExpertProtocolV2RouteEntry {
                    row_index: (route / topk) as u32, expert_id, gate_weight: gate }).collect(),
                vec![input; rows as usize * (hidden + hidden / 32)])?;
            owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
            let encoded = owned.encode()?;
            let mut frame = HostAllocation::new(&library, encoded.len(), crate::shared::memory::TEST)?;
            frame.bytes_mut().copy_from_slice(&encoded);
            let mut reply = HostAllocation::new(&library, prefix + rows as usize * hidden * 2 + 32, crate::shared::memory::TEST)?;
            reply.bytes_mut().fill(0xa5);
            expected.push((0..rows as usize).flat_map(|row| {
                let value = input as u16 + ids[row * topk..(row + 1) * topk].iter()
                    .map(|id| ((id + 1) as usize * (unit + 1)) as u16).sum::<u16>();
                std::iter::repeat_n(value.to_le_bytes(), hidden).flatten()
            }).collect::<Vec<_>>());
            frames.push(frame);
            replies.push(reply);
        }
        if received_pair {
            use crate::shared::experts::service::local::continue_intake;
            // Drive the production nonblocking sweep policy with two already
            // received frames, then an empty receive (never a timer or wait).
            for unit in 0..2 {
                let view = ExpertProtocolV2RequestView::parse(frames[unit].bytes())?;
                let request = BackboneRequest::parse(frames[unit].bytes(), 4096)?;
                let alias = library.cuda_host_buffer_device_alias(frames[unit].buffer)?;
                let mapped = ProtocolV2RequestDevicePayload {
                    hidden_payload: CuteafdDeviceBuffer { ptr: view.hidden_payload().as_ptr().cast_mut().cast(),
                        bytes: view.hidden_payload().len(), ..alias }, execution_lane: 0,
                    response_slot: Some(library.cuda_host_buffer_device_alias(replies[unit].buffer)?) };
                worker.bind_layer(0)?;
                // SAFETY: frames/replies remain owned until both completion fences retire.
                unsafe { worker.execute_mapped_payload(&request, worker.executor_id, &mut exchange, mapped)?; }
                if let Some(staged) = worker.waves.as_ref().unwrap().staged.last() {
                    assert_ne!(staged.input.ptr, mapped.hidden_payload.ptr, "FC1 must not gather mapped host rows");
                }
                assert_eq!(continue_intake(true, worker.has_staged(), unit, 3), unit == 0);
            }
            assert!(!continue_intake(false, worker.has_staged(), 2, 3));
            assert_eq!(worker.waves.as_ref().unwrap().staged.len(), 0);
            assert_eq!(worker.waves.as_ref().unwrap().flushes, 1, "the second unit forces a pair flush");
            fixture.advance_queued_experts(&library, 2)?; // both input copies, no FC1
            if bf16_sibling {
                let bf16_bytes = worker.capacity * hidden * 2;
                assert_eq!(worker.waves.as_ref().unwrap().plane().ptr as usize,
                    worker.inputs[0].buffer.ptr as usize + bf16_bytes);
                assert_eq!(worker.inputs[0].buffer.bytes, bf16_bytes + 3 * 769 * (hidden + hidden / 32));
                // Overwrite the entire ordinary BF16 fallback plane while the
                // wire units still await FC1. An aliased max(old, wire) fails.
                library.copy_h2d(device_part(worker.inputs[0].buffer, 0, bf16_bytes)?, &vec![0xee; bf16_bytes])?;
            }
            // Negative control: direct mapped gather would now produce different bytes.
            for frame in &mut frames[..2] { frame.bytes_mut().fill(0xff); }
            worker.flush_staged()?;
            assert_eq!(worker.waves.as_ref().unwrap().flushes, 1);
            fixture.advance_queued_experts(&library, 8)?; // packed routes + addresses
            assert!(!worker.unit_ready()?);
            fixture.advance_queued_experts(&library, 1)?; // one indirect FC1/down/combine fixture
            for unit in 0..2 {
                assert!(worker.unit_ready()?);
                worker.complete()?;
                assert_eq!(&replies[unit].bytes()[prefix..prefix + expected[unit].len()], expected[unit]);
            }
            assert!(worker.pending.is_empty());
            return Ok(());
        }
        for unit in 0..4 {
            if unit == 1 && segmented {
                // Cancel the first peer before a second unit forces the pair
                // to launch. Later units reuse its slot without publishing it.
                let cancelled = worker.waves.as_ref().unwrap().staged[0].source_address;
                assert!(worker.cancel_staged(cancelled)?);
                assert!(!worker.cancel_staged(cancelled)?);
                assert!(worker.unit_ready()?, "cancelling the lone staged unit leaves no fence");
            }
            if unit == 3 && !segmented {
                fixture.advance_queued_experts(&library, 1)?;
                assert!(worker.unit_ready()? && !worker.stream_idle()?);
                worker.complete()?;
                assert_eq!(worker.pending.next_slot()?, 0);
                assert_eq!(&replies[0].bytes()[prefix..prefix + expected[0].len()], expected[0]);
            }
            let view = ExpertProtocolV2RequestView::parse(frames[unit].bytes())?;
            let request = BackboneRequest::parse(frames[unit].bytes(), 4096)?;
            let alias = library.cuda_host_buffer_device_alias(frames[unit].buffer)?;
            let field = |bytes: &[u8]| CuteafdDeviceBuffer {
                ptr: bytes.as_ptr().cast_mut().cast(), bytes: bytes.len(), ..alias };
            let mapped = ProtocolV2RequestDevicePayload {
                hidden_payload: field(view.hidden_payload()), execution_lane: 0,
                response_slot: Some(library.cuda_host_buffer_device_alias(replies[unit].buffer)?) };
            worker.bind_layer(if segmented { 0 } else { unit % 2 })?;
            // SAFETY: every frame and reply is retained until all queued unit fences complete.
            let response = unsafe { worker.execute_mapped_payload(&request, worker.executor_id,
                &mut exchange, mapped)? }.context("mapped queue request rejected")?;
            assert_eq!(response.header.request_id, 91 + unit as u64);
            assert!(!worker.unit_ready()?);
            if unit == 0 && !segmented {
                // The mapped input is read in place; only the two route copies
                // precede the expert read. Submit different routes before that read.
                fixture.advance_queued_experts(&library, 2)?;
                assert!(!worker.unit_ready()?);
            }
        }
        if segmented {
            assert!(worker.has_staged(), "the third surviving unit waits behind the flushed pair");
            assert_eq!(worker.waves.as_ref().unwrap().flushes, 1);
            assert_eq!(worker.pending.len(), 2);
            assert!(worker.pending.next_slot().is_err());
            let segments = worker.waves.as_ref().unwrap();
            assert_eq!(segments.rows(), 2 * 769, "one merged launch holds two units");
            assert_eq!(worker.inputs[0].buffer.bytes, cuteafd_core::spark_executions::segment_input_bytes(
                4096, 3 * 769, geometry.hidden + geometry.hidden / 32, None).unwrap() as usize);
            for unit in &segments.staged {
                assert_ne!(unit.input.ptr as usize, unit.source_address);
            }
            assert_eq!(segments.table_bytes_allocated(), 2 * 769 * (16 + 8 * topk));
            assert_eq!(segments.descriptor_bytes_allocated(), 3 * 769 * 16);
            worker.flush_staged()?;
            assert!(!worker.unit_ready()? && worker.pending.next_slot().is_err());
            // The pair and trailing singleton retain separate completion fences.
            worker.synchronize()?;
            for unit in [1, 2, 3] {
                assert!(worker.unit_ready()?);
                worker.complete()?;
                assert_eq!(&replies[unit].bytes()[prefix..prefix + expected[unit].len()], expected[unit]);
                assert!(replies[unit].bytes()[..prefix].iter().all(|&b| b == 0xa5));
                assert!(replies[unit].bytes()[prefix + expected[unit].len()..].iter().all(|&b| b == 0xa5));
            }
            assert!(replies[0].bytes().iter().all(|&b| b == 0xa5), "cancelled peer must not launch");
            assert!(worker.stream_idle()? && worker.pending.is_empty());
            // Reuse the cancelled peer as a lone unit. It launches the ordinary
            // recipe, with one input copy/two route uploads and no descriptors.
            let view = ExpertProtocolV2RequestView::parse(frames[0].bytes())?;
            let request = BackboneRequest::parse(frames[0].bytes(), 4096)?;
            let alias = library.cuda_host_buffer_device_alias(frames[0].buffer)?;
            let mapped = ProtocolV2RequestDevicePayload {
                hidden_payload: CuteafdDeviceBuffer { ptr: view.hidden_payload().as_ptr().cast_mut().cast(),
                    bytes: view.hidden_payload().len(), ..alias }, execution_lane: 0,
                response_slot: Some(library.cuda_host_buffer_device_alias(replies[0].buffer)?) };
            // SAFETY: the source and reply remain retained through the completion below.
            unsafe { worker.execute_mapped_payload(&request, worker.executor_id, &mut exchange, mapped)?; }
            worker.flush_staged()?;
            fixture.advance_queued_experts(&library, 3)?;
            assert!(!worker.unit_ready()?);
            assert!(replies[0].bytes().iter().all(|&b| b == 0xa5));
            fixture.advance_queued_experts(&library, 1)?;
            assert!(worker.unit_ready()?);
            worker.complete()?;
            assert_eq!(&replies[0].bytes()[prefix..prefix + expected[0].len()], expected[0]);
            assert!(worker.stream_idle()? && worker.pending.is_empty());
            return Ok(());
        }
        assert_eq!(worker.pending.len(), 3);
        assert!(!worker.stream_idle()?);
        for unit in 1..4 {
            fixture.advance_queued_experts(&library, 4)?;
            assert!(worker.unit_ready()?);
            if unit < 3 { assert!(!worker.stream_idle()?); }
            worker.complete()?;
            assert_eq!(&replies[unit].bytes()[prefix..prefix + expected[unit].len()], expected[unit]);
            assert!(replies[unit].bytes()[..prefix].iter().all(|byte| *byte == 0xa5));
            assert!(replies[unit].bytes()[prefix + expected[unit].len()..].iter().all(|byte| *byte == 0xa5));
        }
        assert!(worker.stream_idle()? && worker.pending.is_empty());
        Ok(())
    }

    #[test]
    fn width_group_accepts_sparse_layers_but_not_empty_or_duplicate_ids() {
        for layers in [&[3][..], &[3, 4, 5][..], &[3, 6, 7, 8, 9, 12][..]] {
            super::validate_worker_layers(layers).unwrap();
        }
        for layers in [&[][..], &[3, 3][..], &[6, 3][..]] {
            assert!(super::validate_worker_layers(layers).is_err());
        }
    }



}
