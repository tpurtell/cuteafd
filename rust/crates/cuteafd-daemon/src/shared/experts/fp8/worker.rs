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

pub(crate) struct Fp8Worker<'a> {
    // Drain the stream before dropping the experts or the inputs.
    stream: Rc<LoadStream<'a>>,
    copy_stream: Option<Rc<LoadStream<'a>>>,
    experts: Fp8Experts<'a, Rc<Fp8Weights<'a>>>,
    library: &'a NativeLibrary,
    inputs: [DeviceAllocation<'a>; 3],
    output: DeviceAllocation<'a>,
    reply_codec: super::super::reply::ReplyCodec<'a>,
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
    segments: Option<Segments<'a>>,
    timing_units: std::collections::VecDeque<bool>,
}

impl Drop for Fp8Worker<'_> {
    fn drop(&mut self) {
        // Shared stream owners may outlive this group's allocations.
        crate::shared::decode_graph::fatal_drain(self.synchronize(), "FP8 worker inputs");
    }
}

struct StagedUnit {
    slot: usize,
    layer: usize,
    identity: (u64, u32, u32, u64),
    source_address: usize,
    input: CuteafdDeviceBuffer,
    output: CuteafdDeviceBuffer,
}
struct Segments<'a> {
    copied: Vec<crate::shared::memory::device::Event<'a>>,
    input_offset: usize,
    staged_units: u64,
    flushes: u64,
    rows: usize,
    unit_rows: usize,
    descriptors: Vec<crate::shared::memory::HostAllocation<'a>>,
    device: [DeviceAllocation<'a>; 4],
    staged: Vec<StagedUnit>,
}

fn device_part(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes), "segment device extent overflow");
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

fn validate_worker_layers(layers: &[usize]) -> Result<()> {
    ensure!(!layers.is_empty(), "FP8 worker has no layers");
    super::validate_layer_ids(layers)
}

impl<'a> Fp8Worker<'a> {
    pub(crate) fn capacity(&self) -> usize { self.capacity }
    pub(crate) fn segment_io_bytes(module: &cuteafd_ffi::fp8_moe::Fp8MoeModule,
        extent: cuteafd_core::OwnerExtent, bf16_sibling: bool) -> Result<usize> {
        let Some(rows) = super::segment_rows(module, extent) else { return Ok(0) };
        let info = module.info();
        let input = cuteafd_core::spark_executions::segment_input_bytes(extent.capacity, rows,
            info.hidden as u32, bf16_sibling).context("segmented input bytes overflow")? as usize;
        let old = extent.capacity as usize * if bf16_sibling { info.hidden * 2 } else { info.wire_row_bytes() };
        Ok(rows as usize * (16 + info.topk * 8) + input - old)
    }
    pub(crate) fn prepare_segments(&mut self, extent: cuteafd_core::OwnerExtent) -> Result<()> {
        let Some(rows) = super::segment_rows(&self.experts.module, extent) else { return Ok(()) };
        let rows = rows as usize;
        let topk = self.experts.module.info().topk;
        let descriptors = (0..extent.buffered_units).map(|_| crate::shared::memory::HostAllocation::named(
            self.library, extent.staging_rows as usize * 16, "experts/workspace/row_addresses"))
            .collect::<Result<Vec<_>>>()?;
        self.copy_stream = Some(Rc::new(LoadStream { library: self.library, raw: self.library.cuda_stream_create()? }));
        let device = crate::shared::memory::device::Device { library: self.library, id: self.library.cuda_get_device()? };
        self.timing_units = std::collections::VecDeque::with_capacity(extent.buffered_units as usize);
        self.segments = Some(Segments {
            copied: (0..extent.buffered_units).map(|_| crate::shared::memory::device::Event::new(device)).collect::<Result<_>>()?,
            input_offset: if self.experts.bf16_module.is_some() { self.capacity * self.experts.module.info().hidden * 2 } else { 0 },
            staged_units: 0, flushes: 0, rows, unit_rows: extent.staging_rows as usize, descriptors,
            device: [DeviceAllocation::new(self.library, rows * 8)?, DeviceAllocation::new(self.library, rows * topk * 4)?,
                DeviceAllocation::new(self.library, rows * topk * 4)?, DeviceAllocation::new(self.library, rows * 8)?],
            staged: Vec::with_capacity(extent.buffered_units as usize) });
        Ok(())
    }

    pub(crate) fn has_staged(&self) -> bool {
        self.segments.as_ref().is_some_and(|segments| !segments.staged.is_empty())
    }

    pub(crate) fn cancel_staged(&mut self, input_address: usize) -> Result<bool> {
        let Some(segments) = &mut self.segments else { return Ok(false) };
        let Some(index) = segments.staged.iter().position(|unit| unit.source_address == input_address) else { return Ok(false) };
        let unit = &segments.staged[index];
        // SAFETY: cancellation must retire the already-issued copy before the
        // caller can release the mapped frame or reuse this device slot.
        unsafe { self.library.cuda_event_synchronize(segments.copied[unit.slot].raw)?; }
        let unit = segments.staged.remove(index);
        self.pending.cancel_reserved(unit.slot)?;
        Ok(true)
    }

    pub(crate) fn flush_staged(&mut self) -> Result<()> {
        let Some(segments) = &mut self.segments else { return Ok(()) };
        if segments.staged.is_empty() { return Ok(()) }
        let rows: usize = segments.staged.iter().map(|unit| unit.identity.2 as usize).sum();
        ensure!(rows <= segments.rows, "staged group exceeds admitted rows");
        segments.flushes += 1;
        tracing::debug!(target: "cuteafd::spark_coalesce", staged_units = segments.staged_units,
            flushes = segments.flushes, segments = segments.staged.len(), rows, "Spark staged flush");
        let topk = self.experts.module.info().topk;
        let layer = segments.staged[0].layer;
        let mut first = 0;
        for unit in &segments.staged {
            ensure!(unit.layer == layer, "staged units have different layers");
            let count = unit.identity.2 as usize;
            // SAFETY: all per-unit staging and mapped rows remain retained until
            // the fences below. Device tables are overwritten only in stream order.
            unsafe {
                self.library.cuda_stream_wait_event(self.stream.raw, segments.copied[unit.slot].raw)?;
                self.route_staging.upload(unit.slot, 0, count * topk,
                    device_part(segments.device[1].buffer, first * topk * 4, count * topk * 4)?,
                    device_part(segments.device[2].buffer, first * topk * 4, count * topk * 4)?, self.stream.raw)?;
                if segments.staged.len() == 1 { first += count; continue; }
                let source = segments.descriptors[unit.slot].buffer;
                let outputs = cuteafd_ffi::CuteafdHostBuffer { ptr: source.ptr.cast::<u8>()
                    .wrapping_add(segments.unit_rows * 8).cast(), bytes: count * 8, ..source };
                self.library.copy_host_buffers_h2d_batch_async(&[
                    device_part(segments.device[0].buffer, first * 8, count * 8)?,
                    device_part(segments.device[3].buffer, first * 8, count * 8)?],
                    &[source, outputs], &[count * 8, count * 8], self.stream.raw)?;
            }
            first += count;
        }
        // SAFETY: descriptor copies precede FC1 on this stream. All addressed
        // receive/send slots, routes, weights and admitted scratch remain live.
        unsafe {
            if let Some(timing) = &self.timing { timing.borrow().record(0, self.stream.raw)?; }
            if segments.staged.len() == 1 {
                let unit = &segments.staged[0];
                self.experts.run(layer, rows, unit.input.ptr, segments.device[1].buffer.ptr,
                    segments.device[2].buffer.ptr, unit.output.ptr, self.stream.raw)?;
            } else {
                self.experts.run_segments(layer, rows, segments.device.each_ref().map(|allocation| allocation.buffer.ptr),
                    self.stream.raw)?;
            }
            if let Some(timing) = &self.timing { timing.borrow().record(1, self.stream.raw)?; }
            for (index, unit) in segments.staged.iter().enumerate() {
                self.pending.record_reserved(unit.slot, unit.identity, self.stream.raw)?;
                self.timing_units.push_back(index == 0);
            }
        }
        segments.staged.clear();
        Ok(())
    }

    fn stage(&mut self, request: &BackboneRequest<'_>, input: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer, executor_id: u64) -> Result<bool> {
        let rows = request.rows() as usize;
        let compatible = self.segments.as_ref().is_some_and(|segments| rows <= segments.unit_rows)
            && request.require_input_dtype(ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32).is_ok();
        if !compatible {
            tracing::debug!(target: "cuteafd::spark_coalesce", rows, reason = "incompatible_input_or_capacity", "Spark staging bypass");
            return Ok(false)
        }
        let singleton = self.experts.module.segment_recipe().is_some_and(|recipe| rows < recipe.minimum_rows as usize);
        if singleton {
            tracing::debug!(target: "cuteafd::spark_coalesce", rows, reason = "singleton_recipe", "Spark staged boundary");
            self.flush_staged()?;
        }
        if self.segments.as_ref().is_some_and(|segments| segments.staged.first()
            .is_some_and(|unit| unit.layer != self.layer)) {
            tracing::debug!(target: "cuteafd::spark_coalesce", reason = "different_layer", "Spark staged boundary");
            self.flush_staged()?;
        }
        ensure!(executor_id == self.executor_id && request.layer() as usize == self.experts.layers[self.layer].layer,
            "staged expert identity mismatch");
        if self.segments.as_ref().is_some_and(|segments| {
            let staged_rows: usize = segments.staged.iter().map(|unit| unit.identity.2 as usize).sum();
            staged_rows + rows > segments.rows.min(2 * segments.unit_rows)
        }) { self.flush_staged()?; }
        let segments = self.segments.as_mut().unwrap();
        let slot = self.pending.next_slot()?;
        self.route_staging.pack(slot, request)?;
        let bytes = segments.descriptors[slot].bytes_mut();
        let stride = self.experts.module.info().wire_row_bytes();
        ensure!((input.ptr as usize) % 16 == 0 && (output.ptr as usize) % 16 == 0 && stride % 16 == 0,
            "segmented mapped rows require 16-byte aligned input/output payloads and input stride");
        ensure!(input.bytes >= rows * stride && output.bytes >= rows * self.experts.module.info().hidden * 2,
            "staged mapped row extent is too small");
        let source_address = input.ptr as usize;
        let resident = device_part(self.inputs[0].buffer,
            segments.input_offset + slot * segments.unit_rows * stride, rows * stride)?;
        for row in 0..rows {
            let input_address = (resident.ptr as usize).checked_add(row * stride).context("input row address overflow")?;
            let output_address = (output.ptr as usize).checked_add(row * self.experts.module.info().hidden * 2)
                .context("output row address overflow")?;
            bytes[row * 8..row * 8 + 8].copy_from_slice(&(input_address as u64).to_le_bytes());
            let at = segments.unit_rows * 8 + row * 8;
            bytes[at..at + 8].copy_from_slice(&(output_address as u64).to_le_bytes());
        }
        ensure!(self.pending.reserve()? == slot, "staged slot reservation changed");
        segments.staged.push(StagedUnit { slot, layer: self.layer, source_address,
            identity: (request.request_id(), request.layer(), request.rows(), executor_id), input: resident, output });
        let copy = self.copy_stream.as_ref().unwrap().raw;
        // SAFETY: the reserved unit retains its mapped source, device slice and
        // event. Copy is host-triggered now, with no wait on future compute.
        unsafe {
            self.pending.arrive_reserved(slot, copy)?;
            self.library.copy_d2d_async(resident, input, rows * stride, copy)?;
            self.library.cuda_event_record(segments.copied[slot].raw, copy)?;
        }
        segments.staged_units += 1;
        tracing::debug!(target: "cuteafd::spark_coalesce", staged_units = segments.staged_units,
            staged = segments.staged.len(), rows, slot, "Spark unit copied and staged");
        let staged_rows: usize = segments.staged.iter().map(|unit| unit.identity.2 as usize).sum();
        let full = segments.staged.len() == segments.copied.len().min(2)
            || staged_rows >= segments.rows.min(2 * segments.unit_rows);
        if singleton || full { self.flush_staged()?; }
        Ok(true)
    }

    /// Exact retained input/routes/output bytes (scratch is admitted separately).
    pub(crate) fn io_bytes(info: &cuteafd_ffi::fp8_moe::Fp8MoeInfo, bf16_sibling: bool,
        capacity: usize) -> Result<usize> {
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
        ensure!(experts.wire_input() || matches!(experts.module.info().weights,
            cuteafd_ffi::fp8_moe::Fp8MoeWeights::Nvfp4 | cuteafd_ffi::fp8_moe::Fp8MoeWeights::Bf16),
            "Spark workers need a wire-input package, or native BF16-input weights");
        let experts = match scratch {
            Some(scratch) => Fp8Experts::from_scratch(experts, capacity, scratch)?,
            None => Fp8Experts::from_shared(library, experts, capacity)?,
        };
        let input_bytes = match extent.and_then(|e| super::segment_rows(&experts.module, e)) {
            Some(rows) => cuteafd_core::spark_executions::segment_input_bytes(capacity as u32, rows,
                hidden as u32, experts.bf16_module.is_some()).context("segmented input extent overflow")? as usize,
            None => capacity * if !experts.wire_input() || experts.bf16_module.is_some() { hidden * 2 }
                else { hidden + hidden / 32 },
        };
        let inputs = [
            DeviceAllocation::new(library, input_bytes)?,
            DeviceAllocation::new(library, capacity * topk * 4)?,
            DeviceAllocation::new(library, capacity * topk * 4)?,
        ];
        let output = DeviceAllocation::new(library, capacity * hidden * 2)?;
        let executor_id = cuteafd_transport::expert::SparkWorld::canonical(experts.tp)?.topology().executor_id(experts.rank)?;
        let timing = super::super::execution::timing::WorkerTiming::from_env(library, experts.rank)?
            .map(|timing| Rc::new(std::cell::RefCell::new(timing)));
        Ok(Self {
            stream: Rc::new(LoadStream { library, raw: library.cuda_stream_create()? }),
            copy_stream: None,
            experts,
            library,
            inputs,
            output,
            reply_codec: super::super::reply::ReplyCodec::load(library)?,
            capacity,
            layer: 0,
            executor_id,
            timing,
            deferred: false,
            segments: None,
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
        if self.copy_stream.is_some() && other.copy_stream.is_some() { self.copy_stream = other.copy_stream.clone(); }
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
            if let Some(copy) = &self.copy_stream { self.library.cuda_stream_synchronize(copy.raw)?; }
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
        if self.deferred && request.output_dtype() == ExpertV2Dtype::Bf16 {
            if let (Some(input), Some(output)) = (hidden_view, destination) {
                if self.stage(request, input, output, executor_id)? { return Ok(output) }
            }
        }
        self.flush_staged()?;
        // Host-only fallback can alias the wire arena; retire prior consumers.
        if self.segments.is_some() && (hidden_view.is_none() || self.experts.bf16_module.is_none()) { self.synchronize()?; }
        let rows = request.rows() as usize;
        ensure!(rows > 0 && rows <= self.capacity, "FP8 request exceeds capacity");
        // FP8 K32 wire rows, or BF16 rows when the BF16-input package is loaded.
        let bf16 = request.require_input_dtype(ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32).is_err();
        if bf16 {
            ensure!(!self.experts.wire_input() || self.experts.bf16_module.is_some(),
                "BF16 expert input needs the BF16-input FP8 package (fp8-<family>-bf16) on this rank");
            request.require_input_dtype(ExpertV2Dtype::Bf16 as u32)?;
        }
        ensure!(bf16 || self.experts.wire_input(),
            "native BF16/NVFP4 Spark input must be BF16, not FP8 K32");
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
        match hidden_view.filter(|view| view.bytes >= hidden_bytes) {
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
        let output = match destination.filter(|_| !compressed) {
            Some(output) => output,
            None => self.output.buffer,
        };
        ensure!(output.bytes >= bytes, "FP8 response slot is too small");
        // SAFETY: inputs, output and the resident layer are live device memory of
        // the documented extents; the stream is synchronized below.
        unsafe {
            if let Some(timing) = &self.timing { timing.borrow().record(0, self.stream.raw)?; }
            if bf16 {
                self.experts.run_bf16(self.layer, rows, self.inputs[0].buffer.ptr, self.inputs[1].buffer.ptr,
                    self.inputs[2].buffer.ptr, output.ptr, self.stream.raw)?;
            } else {
                self.experts.run(self.layer, rows, self.inputs[0].buffer.ptr, self.inputs[1].buffer.ptr,
                    self.inputs[2].buffer.ptr, output.ptr, self.stream.raw)?;
            }
            if let Some(timing) = &self.timing { timing.borrow().record(1, self.stream.raw)?; }
            if compressed {
                self.reply_codec.quantize(request, output,
                    destination.context("missing compressed reply slot")?, request.rows() as usize, self.stream.raw)?;
            }
            // The transport holds both mapped slots until this unit's fence,
            // independent of later work queued on the owner's stream.
            if self.deferred && destination.is_some() {
                self.pending.record((request.request_id(), request.layer(), request.rows(), executor_id), self.stream.raw)?;
                if self.segments.is_some() { self.timing_units.push_back(true); }
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
            self.library.copy_d2h(&mut exchange.partials[..bytes], output)?;
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

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn native_slots_own_distinct_capacity_sized_scratch_and_charge_workspace_once() -> Result<()> {
        use cuteafd_ffi::{fp8_moe::Fp8MoeModule, memory_ledger::{self, Space},
            native_library_lifetime_fixture::Fixture};
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
                let mut words = [5 | format << 8 | 1 << 16 | 1 << 24, geometry.hidden,
                    128, 2, geometry.topk, 256, 2, if wire { 7 } else { 1 }, 0,
                    2, 16, 4096, 0, 0, 0, 0];
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
                    DeviceAllocation::new(&library, 512)?
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
    fn staging_boundaries_flush_without_waiting_for_stream_idle() {
        let stage = include_str!("worker.rs").split("    fn stage(").nth(1).unwrap()
            .split("    /// Exact retained").next().unwrap();
        assert!(!stage.contains("stream_idle"));
        assert!(stage.contains("if singleton {\n            tracing::debug!"));
        assert!(stage.contains("unit.layer != self.layer"));
        assert!(stage.contains("staged_rows + rows > segments.rows.min(2 * segments.unit_rows)"));
        assert!(stage.contains("segments.staged.len() == segments.copied.len().min(2)"));
        assert!(stage.contains("staged_rows >= segments.rows.min(2 * segments.unit_rows)"));
        assert!(stage.contains("if singleton || full { self.flush_staged()?; }"));
    }

    #[test]
    fn staged_input_copy_is_host_triggered_and_compute_waits_on_recorded_event() {
        let source = include_str!("worker.rs");
        let stage = source.split("    fn stage(").nth(1).unwrap().split("    /// Exact retained").next().unwrap();
        assert!(stage.find("copy_d2d_async(resident").unwrap() < stage.find("cuda_event_record(segments.copied").unwrap());
        assert!(!stage.contains("cuda_stream_wait_event"));
        let flush = source.split("    pub(crate) fn flush_staged(").nth(1).unwrap().split("    fn stage(").next().unwrap();
        assert!(flush.find("cuda_stream_wait_event").unwrap() < flush.find("self.experts.run(").unwrap());
        assert!(flush.contains("unit.input.ptr"));
        assert!(stage.contains("(resident.ptr as usize)"));
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
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 30)?;
        fixture.queued_experts(&library)?;
        std::fs::hard_link(fixture.directory().join("liblifetime_fixture.so"),
            fixture.directory().join("libcuteafd_fp8moe.so"))?;
        let geometry = cuteafd_core::expert_geometry();
        let words = [5 | 1 << 8 | 1 << 16 | 1 << 24, geometry.hidden,
            128, geometry.experts, geometry.topk, 256, 2, 7, 0,
            2, 16, 4096, 0, 0, 0, 0];
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
        let scratch = Rc::new(DeviceAllocation::new(&library, weights.owner_scratch_bytes(extent)?)?);
        let mut worker = Fp8Worker::with_extent(&library, weights, 4096, Some(scratch), segmented.then_some(extent))?;
        if segmented { worker.prepare_segments(extent)?; }
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
            let mut frame = HostAllocation::new(&library, encoded.len())?;
            frame.bytes_mut().copy_from_slice(&encoded);
            let mut reply = HostAllocation::new(&library, prefix + rows as usize * hidden * 2 + 32)?;
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
                if let Some(staged) = worker.segments.as_ref().unwrap().staged.last() {
                    assert_ne!(staged.input.ptr, mapped.hidden_payload.ptr, "FC1 must not gather mapped host rows");
                }
                assert_eq!(continue_intake(true, worker.has_staged(), unit, 3), unit == 0);
            }
            assert!(!continue_intake(false, worker.has_staged(), 2, 3));
            assert_eq!(worker.segments.as_ref().unwrap().staged.len(), 0);
            assert_eq!(worker.segments.as_ref().unwrap().flushes, 1, "the second unit forces a pair flush");
            fixture.advance_queued_experts(&library, 2)?; // both input copies, no FC1
            if bf16_sibling {
                let bf16_bytes = worker.capacity * hidden * 2;
                assert_eq!(worker.segments.as_ref().unwrap().input_offset, bf16_bytes);
                assert_eq!(worker.inputs[0].buffer.bytes, bf16_bytes + 3 * 769 * (hidden + hidden / 32));
                // Overwrite the entire ordinary BF16 fallback plane while the
                // wire units still await FC1. An aliased max(old, wire) fails.
                library.copy_h2d(device_part(worker.inputs[0].buffer, 0, bf16_bytes)?, &vec![0xee; bf16_bytes])?;
            }
            // Negative control: direct mapped gather would now produce different bytes.
            for frame in &mut frames[..2] { frame.bytes_mut().fill(0xff); }
            worker.flush_staged()?;
            assert_eq!(worker.segments.as_ref().unwrap().flushes, 1);
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
                let cancelled = worker.segments.as_ref().unwrap().staged[0].source_address;
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
                // Input/routes have arrived; the first expert read is still queued.
                // Submit unit 1's different routes before allowing that read to run.
                fixture.advance_queued_experts(&library, 3)?;
                assert!(!worker.unit_ready()?);
            }
        }
        if segmented {
            assert!(worker.has_staged(), "the third surviving unit waits behind the flushed pair");
            assert_eq!(worker.segments.as_ref().unwrap().flushes, 1);
            assert_eq!(worker.pending.len(), 2);
            assert!(worker.pending.next_slot().is_err());
            let segments = worker.segments.as_ref().unwrap();
            assert_eq!(segments.rows, 3 * 769);
            assert_eq!(worker.inputs[0].buffer.bytes, cuteafd_core::spark_executions::segment_input_bytes(
                4096, 3 * 769, geometry.hidden, false).unwrap() as usize);
            for unit in &segments.staged {
                assert_ne!(unit.input.ptr as usize, unit.source_address);
            }
            assert_eq!(segments.device.iter().map(|a| a.buffer.bytes).sum::<usize>(),
                3 * 769 * (16 + 8 * topk));
            assert_eq!(segments.descriptors.iter().map(|a| a.buffer.bytes).sum::<usize>(), 3 * 769 * 16);
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
