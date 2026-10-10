//! Spark request adapter for the FP8 family: FP8 K32 wire rows and routes in,
//! the compact BF16 rank partial out, written straight into the registered
//! send slot when the transport permits it.
use super::Fp8Experts;
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
    stream: LoadStream<'a>,
    experts: Rc<Fp8Experts<'a>>,
    library: &'a NativeLibrary,
    inputs: [DeviceAllocation<'a>; 3],
    output: DeviceAllocation<'a>,
    capacity: usize,
    layer: usize,
    executor_id: u64,
}

impl<'a> Fp8Worker<'a> {
    /// Device bytes the worker adds to its resident layers for `capacity`.
    /// (Input rows are sized for BF16, the larger of the two representations.)
    pub(crate) fn workspace_bytes(capacity: usize) -> usize {
        let geometry = cuteafd_core::expert_geometry();
        let (hidden, topk) = (geometry.hidden as usize, geometry.topk as usize);
        capacity * (hidden * 2 + topk * 8 + hidden * 2)
    }

    pub(crate) fn new(library: &'a NativeLibrary, experts: Rc<Fp8Experts<'a>>, capacity: u32) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/workspace");
        let geometry = cuteafd_core::expert_geometry();
        let (hidden, topk, capacity) = (geometry.hidden as usize, geometry.topk as usize, capacity as usize);
        ensure!(!experts.layers.is_empty(), "FP8 worker has no layers");
        ensure!(experts.layers.windows(2).all(|w| w[1].layer == w[0].layer + 1), "FP8 worker layers must be contiguous");
        ensure!(experts.module.info().capacity_for(capacity).is_some(), "FP8 package lacks capacity {capacity}");
        ensure!(experts.wire_input(), "Spark workers need an FP8 package built for wire rows (the spark role)");
        let inputs = [
            DeviceAllocation::new(library, capacity * if experts.bf16_module.is_some() { hidden * 2 }
                else { hidden + hidden / 32 })?,
            DeviceAllocation::new(library, capacity * topk * 4)?,
            DeviceAllocation::new(library, capacity * topk * 4)?,
        ];
        let output = DeviceAllocation::new(library, capacity * hidden * 2)?;
        let executor_id = cuteafd_transport::expert::v41_spark_executor_id(experts.tp, experts.rank)?;
        Ok(Self {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            experts,
            library,
            inputs,
            output,
            capacity,
            layer: 0,
            executor_id,
        })
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
        let rows = request.rows() as usize;
        ensure!(rows > 0 && rows <= self.capacity, "FP8 request exceeds capacity");
        ensure!(!request.is_paired(), "FP8 experts have no paired layout");
        // FP8 K32 wire rows, or BF16 rows when the BF16-input package is loaded.
        let bf16 = request.require_input_dtype(ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32).is_err();
        if bf16 {
            ensure!(self.experts.bf16_module.is_some(),
                "BF16 expert input needs the BF16-input FP8 package (fp8-<family>-bf16) on this rank");
            request.require_input_dtype(ExpertV2Dtype::Bf16 as u32)?;
        }
        ensure!(request.hidden().len() <= self.inputs[0].buffer.bytes, "FP8 request rows exceed the input buffer");
        let bytes = request.plane_bytes()?;
        let routes = rows * cuteafd_core::expert_geometry().topk as usize;
        request.copy_routes_into(&mut exchange.ids, &mut exchange.routing)?;
        ensure!(cfg!(target_endian = "little"), "native exchange requires little-endian storage");
        // Every previous response completed this stream before returning. The
        // hidden rows come as a device copy from the mapped request frame when
        // the transport exposes one, else as a host upload.
        let hidden_bytes = request.hidden().len();
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
        // SAFETY: plain i32/f32 slices of `routes` elements viewed as bytes.
        unsafe {
            self.library.copy_h2d(self.inputs[1].buffer,
                std::slice::from_raw_parts(exchange.ids.as_ptr().cast::<u8>(), routes * 4))?;
            self.library.copy_h2d(self.inputs[2].buffer,
                std::slice::from_raw_parts(exchange.routing.as_ptr().cast::<u8>(), routes * 4))?;
        }
        let output = match destination {
            Some(output) => output,
            None => CuteafdDeviceBuffer { bytes, ..self.output.buffer },
        };
        ensure!(output.bytes >= bytes, "FP8 response slot is too small");
        // SAFETY: inputs, output and the resident layer are live device memory of
        // the documented extents; the stream is synchronized below.
        unsafe {
            if bf16 {
                self.experts.run_bf16(self.layer, rows, self.inputs[0].buffer.ptr, self.inputs[1].buffer.ptr,
                    self.inputs[2].buffer.ptr, output.ptr, self.stream.raw)?;
            } else {
                self.experts.run(self.layer, rows, self.inputs[0].buffer.ptr, self.inputs[1].buffer.ptr,
                    self.inputs[2].buffer.ptr, output.ptr, self.stream.raw)?;
            }
            self.library.cuda_stream_synchronize(self.stream.raw)?;
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
