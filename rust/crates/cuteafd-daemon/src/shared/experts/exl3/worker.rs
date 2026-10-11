//! Spark request adapter: preserve the compact BF16 wire response and write
//! directly into registered send storage when the transport permits it.
use super::{
    execution::{Exl3Execution, Exl3InputFormat, Exl3RowPolicy, Exl3Schedule, Exl3Workspace},
    Exl3Weights,
};
use crate::shared::experts::execution::HostExpertExchange;
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_transport::{
    expert::BackboneRequest, ExpertProtocolV2DeviceResponseRef, ExpertProtocolV2ResponseRef,
    ExpertV2Dtype, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};
use std::{io::Write, path::Path, rc::Rc};

/// How a call uploads its routes and waits for its GPU work
/// (`CUTEAFD_EXL3_WORKER_PATH`, read once when the worker is built).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostPath {
    /// `async` (default): the routes are written straight into pinned
    /// staging and uploaded with one batched asynchronous copy, so the host
    /// goes on to launch while the copy runs; the wire decode reads the
    /// hidden rows in the mapped request frame instead of a copy of them; the
    /// worker thread then polls the stream, which returns within a query of
    /// the top-k sum's completion.
    Async,
    /// `blocking`: the hidden rows copied on the stream, two synchronous route
    /// copies, then a blocking stream synchronize.
    Blocking,
}

impl HostPath {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            None | Some("") | Some("async") => Ok(Self::Async),
            Some("blocking") => Ok(Self::Blocking),
            Some(other) => anyhow::bail!("CUTEAFD_EXL3_WORKER_PATH must be async or blocking, not {other:?}"),
        }
    }

    fn from_env() -> Result<Self> {
        Self::parse(std::env::var("CUTEAFD_EXL3_WORKER_PATH").ok().as_deref())
    }
}

/// Opt-in capture of every call's routes for kernel benchmarks
/// (`CUTEAFD_EXL3_ROUTE_DUMP=<path prefix>`, written to
/// `<prefix>.<executor id>.bin`). Records: four little-endian u32 (magic
/// 0x31455452, layer, rows, top-k), then the int32 expert ids and the FP32
/// gate weights of the call's `rows * top-k` routes, in request order. At most
/// `CUTEAFD_EXL3_ROUTE_DUMP_CALLS` calls (default 200,000) are kept.
struct RouteDump {
    writer: std::io::BufWriter<std::fs::File>,
    remaining: u64,
}

impl RouteDump {
    const MAGIC: u32 = 0x3145_5452;

    fn from_env(executor_id: u64) -> Result<Option<Self>> {
        let Some(prefix) = std::env::var_os("CUTEAFD_EXL3_ROUTE_DUMP").filter(|p| !p.is_empty()) else {
            return Ok(None);
        };
        let remaining = match std::env::var("CUTEAFD_EXL3_ROUTE_DUMP_CALLS") {
            Ok(value) => value.parse().context("CUTEAFD_EXL3_ROUTE_DUMP_CALLS must be a call count")?,
            Err(_) => 200_000,
        };
        let mut path = prefix;
        path.push(format!(".{executor_id}.bin"));
        let path = std::path::PathBuf::from(path);
        tracing::info!(path = %path.display(), calls = remaining, "EXL3 worker route dump");
        Self::open(&path, remaining).map(Some)
    }

    fn open(path: &Path, calls: u64) -> Result<Self> {
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path)
            .with_context(|| format!("opening the EXL3 route dump {}", path.display()))?;
        Ok(Self { writer: std::io::BufWriter::with_capacity(1 << 20, file), remaining: calls })
    }

    fn record(&mut self, layer: u32, rows: u32, topk: u32, ids: &[i32], weights: &[f32]) -> Result<()> {
        if self.remaining == 0 {
            return Ok(());
        }
        self.remaining -= 1;
        for word in [Self::MAGIC, layer, rows, topk] {
            self.writer.write_all(&word.to_le_bytes())?;
        }
        for id in ids {
            self.writer.write_all(&id.to_le_bytes())?;
        }
        for weight in weights {
            self.writer.write_all(&weight.to_le_bytes())?;
        }
        if self.remaining == 0 {
            self.writer.flush()?;
        }
        Ok(())
    }
}

impl Drop for RouteDump {
    fn drop(&mut self) {
        if let Err(error) = self.writer.flush() {
            tracing::warn!(%error, "flushing the EXL3 route dump");
        }
    }
}

pub(crate) struct Exl3Worker<'a> {
    // Drain the stream before dropping kernels, weights or input allocations.
    stream: LoadStream<'a>,
    // Opt-in diagnostics: allocate events once; the default path never records them.
    timing: Option<[crate::shared::memory::device::Event<'a>; 2]>,
    executions: Vec<Exl3Execution<'a>>,
    row_policy: Exl3RowPolicy,
    capacity: usize,
    inputs: [DeviceAllocation<'a>; 3],
    /// Pinned staging of a call's routes, laid out as `inputs[1]` and
    /// `inputs[2]` receive them: int32 ids (then a paired layout's ownership
    /// words) from the start, FP32 gate weights from `weights_offset`. Every
    /// call completes its stream before returning, so the next call may
    /// overwrite it.
    staging: HostAllocation<'a>,
    weights_offset: usize,
    paired: bool,
    host_path: HostPath,
    route_dump: Option<RouteDump>,
    ownership_words: usize,
    library: &'a NativeLibrary,
    first_layer: usize,
    layer_count: usize,
    layer: usize,
    executor_id: u64,
}

impl<'a> Exl3Worker<'a> {
    fn capacities(capacity: u32) -> Result<Vec<u32>> {
        Exl3RowPolicy::active().capacities(capacity as usize)
    }

    /// Validate every capacity before allocating or reading resident weights,
    /// including that each export carries the requested decode schedule.
    pub(crate) fn partition(directory: &Path, capacity: u32, rank: usize, schedule: Exl3Schedule)
        -> Result<cuteafd_loader::V41Exl3Partition> {
        use cuteafd_ffi::V41Exl3Layout;
        use cuteafd_loader::V41Exl3Partition;
        ensure!(rank < 6, "EXL3 worker rank must be 0..5");
        schedule.validate(Exl3RowPolicy::active())?;
        let mut selected = None;
        for c in Self::capacities(capacity)? {
            schedule.check_export(directory, c)?;
            let layout = Exl3Execution::artifact_layout(&schedule.directory(directory, c))?;
            ensure!(selected.is_none_or(|previous| previous == layout), "EXL3 capacity artifacts disagree on partition");
            ensure!(layout == V41Exl3Layout::Disjoint || layout == if rank % 2 == 0 {
                V41Exl3Layout::PairedLast
            } else { V41Exl3Layout::PairedFirst }, "EXL3 artifact boundary does not match worker rank");
            selected = Some(layout);
        }
        Ok(if selected == Some(V41Exl3Layout::Disjoint) { V41Exl3Partition::Disjoint }
            else { V41Exl3Partition::PairedTp4 })
    }

    /// FP8 E4M3 wire row: hidden values then hidden/32 UE8M0 K32 scales.
    fn wire_row_bytes() -> usize {
        let hidden = cuteafd_core::expert_geometry().hidden as usize;
        hidden + hidden / 32
    }

    fn topk() -> usize {
        cuteafd_core::expert_geometry().topk as usize
    }

    pub(crate) fn plan(directory: &Path, capacity: u32, schedule: Exl3Schedule) -> Result<usize> {
        let directories: Vec<_> = Self::capacities(capacity)?.into_iter()
            .map(|c| schedule.directory(directory, c)).collect();
        let ownership_bytes = Exl3Execution::ownership_bytes(&directories[0])?;
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)?
            .checked_add(capacity as usize * (Self::wire_row_bytes() + Self::topk() * 8))
            .and_then(|bytes| bytes.checked_add(ownership_bytes))
            .context("EXL3 worker workspace budget overflow")
    }

    fn validate_rank(world: usize, rank: usize) -> Result<()> {
        ensure!((1..=8).contains(&world) && rank < world,
            "EXL3 worker requires implicit Spark TP1..8 weights and rank below world");
        Ok(())
    }

    pub(crate) fn new(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
        capacity: u32,
        available_bytes: usize,
        schedule: Exl3Schedule,
    ) -> Result<Self> {
        let first = weights.first().context("EXL3 worker has no layers")?;
        let cuteafd_loader::V41Exl3Layer::Backbone(first_layer) = first.layout.layer else {
            anyhow::bail!("EXL3 Spark worker requires backbone layers");
        };
        let rank = first.layout.rank;
        Self::validate_rank(first.layout.world, rank)?;
        for (index, weight) in weights.iter().enumerate() {
            ensure!(
                matches!(weight.layout.layer, cuteafd_loader::V41Exl3Layer::Backbone(layer)
                if layer == first_layer + index),
                "EXL3 worker layers must be contiguous"
            );
        }
        let layer_count = weights.len();
        let budget = Self::plan(directory, capacity, schedule)?;
        ensure!(
            budget <= available_bytes,
            "EXL3 worker workspace exceeds device budget"
        );
        let mut executions = Vec::new();
        let row_policy = Exl3RowPolicy::active();
        let capacities = Self::capacities(capacity)?;
        let directories: Vec<_> = capacities.iter().map(|&c| schedule.directory(directory, c)).collect();
        let arena = Exl3Workspace::new(library, &directories)?;
        for c in capacities {
            let execution = unsafe {
                Exl3Execution::with_shared_workspace(
                    library,
                    weights.clone(),
                    &schedule.directory(directory, c),
                    Exl3InputFormat::Fp8K32,
                    Some(arena.clone()),
                )?
            };
            ensure!(
                execution.capacity() == c as usize && execution.output_element_bytes() == 2,
                "EXL3 Spark export must match planned capacity and BF16 response precision"
            );
            executions.push(execution);
        }
        let ownership_words = if first.layout.layout == cuteafd_loader::V41Exl3Partition::PairedTp4 {
            first.layout.experts * first.layout.tiers.len()
        } else { 0 };
        let topk = Self::topk();
        let weights_offset = (capacity as usize * topk + ownership_words) * 4;
        let staging = HostAllocation::new(library, weights_offset + capacity as usize * topk * 4)?;
        let host_path = HostPath::from_env()?;
        let executor_id = cuteafd_transport::expert::v41_spark_executor_id(first.layout.world, rank)?;
        let route_dump = RouteDump::from_env(executor_id)?;
        let inputs = [
            DeviceAllocation::new(library, capacity as usize * Self::wire_row_bytes())?,
            DeviceAllocation::new(library, (capacity as usize * topk + ownership_words) * 4)?,
            DeviceAllocation::new(library, capacity as usize * topk * 4)?,
        ];
        ensure!(
            executions
                .iter()
                .map(|e| e.workspace_bytes())
                .sum::<usize>()
                + inputs.iter().map(|b| b.buffer.bytes).sum::<usize>()
                + arena.bytes()
                == budget,
            "EXL3 worker workspace plan mismatch"
        );
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let timing = if std::env::var_os("CUTEAFD_EXL3_WORKER_TIMING").is_some() {
            let device = crate::shared::memory::device::Device {
                library,
                id: library.cuda_get_device()?,
            };
            Some([
                crate::shared::memory::device::Event::new(device)?,
                crate::shared::memory::device::Event::new(device)?,
            ])
        } else {
            None
        };
        Ok(Self {
            row_policy,
            stream,
            timing,
            executions,
            capacity: capacity as usize,
            inputs,
            staging,
            weights_offset,
            paired: ownership_words > 0,
            host_path,
            route_dump,
            ownership_words,
            library,
            first_layer,
            layer_count,
            layer: 0,
            executor_id,
        })
    }

    pub(crate) fn is_paired(&self) -> bool { self.paired }

    /// The staged ids, ownership words and weights of a call of `routes` routes.
    fn staged(&mut self, routes: usize) -> (&mut [i32], &mut [i32], &mut [f32]) {
        assert!(routes * 4 <= self.weights_offset - self.ownership_words * 4,
            "EXL3 staged routes exceed capacity");
        let base = self.staging.buffer.ptr.cast::<u8>();
        // SAFETY: the three ranges are disjoint, inside the pinned allocation
        // (ids + ownership end at or before `weights_offset`, the weights
        // region holds capacity * top-k floats), 4-byte aligned, and
        // exclusively borrowed through `self`.
        unsafe {
            (
                std::slice::from_raw_parts_mut(base.cast::<i32>(), routes),
                std::slice::from_raw_parts_mut(base.cast::<i32>().add(routes), self.ownership_words),
                std::slice::from_raw_parts_mut(base.add(self.weights_offset).cast::<f32>(), routes),
            )
        }
    }

    /// Read-only view of the staged ids and weights of a call of `routes` routes.
    fn staged_view(&self, routes: usize) -> (&[i32], &[f32]) {
        assert!(routes * 4 <= self.weights_offset - self.ownership_words * 4,
            "EXL3 staged routes exceed capacity");
        let base = self.staging.buffer.ptr.cast::<u8>();
        // SAFETY: as in `staged`; shared borrows of `self` exclude writers.
        unsafe {
            (
                std::slice::from_raw_parts(base.cast::<i32>(), routes),
                std::slice::from_raw_parts(base.add(self.weights_offset).cast::<f32>(), routes),
            )
        }
    }

    pub(crate) fn bind_layer(&mut self, layer: usize) -> Result<()> {
        ensure!(
            layer < self.layer_count,
            "requested EXL3 layer is not resident"
        );
        self.layer = layer;
        Ok(())
    }

    fn execute(
        &mut self,
        request: &BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        destination: Option<CuteafdDeviceBuffer>,
        hidden_view: Option<CuteafdDeviceBuffer>,
    ) -> Result<()> {
        ensure!(
            request.layer() as usize == self.first_layer + self.layer,
            "request does not match selected EXL3 layer"
        );
        ensure!(
            executor_id == self.executor_id,
            "EXL3 response executor identity mismatch"
        );
        ensure!(
            request.rows() > 0 && request.rows() as usize <= self.capacity,
            "EXL3 request exceeds capacity"
        );
        ensure!(request.is_paired() == self.is_paired(), "EXL3 request/resident layout mismatch");
        request.require_input_dtype(ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32)?;
        let bytes = request.plane_bytes()?;
        ensure!(
            exchange.partials.len() >= bytes,
            "EXL3 host exchange is too small"
        );
        let started = self.timing.as_ref().map(|_| std::time::Instant::now());
        let routes = request.rows() as usize * Self::topk();
        let (paired, rank) = (self.paired, self.executor_id.saturating_sub(1) as usize);
        {
            let (ids, ownership, weights) = self.staged(routes);
            if paired {
                request.copy_paired_routes_into(ids, weights, rank, ownership)?;
            } else {
                request.copy_routes_into(ids, weights)?;
            }
        }
        ensure!(
            cfg!(target_endian = "little"),
            "native exchange requires little-endian storage"
        );
        // Every previous response completed this stream before returning. The
        // hidden rows come from the mapped request frame when the transport
        // exposes one (it outlives this request's stream wait below), else as
        // a host upload. On the asynchronous path, an execution whose own
        // decode pass is the rows' only reader reads them in the frame (when
        // aligned for it); otherwise they are first copied on the stream.
        let hidden_bytes = request.hidden().len();
        ensure!(hidden_bytes <= self.inputs[0].buffer.bytes, "EXL3 request hidden rows exceed the input buffer");
        let required = self.row_policy.required_capacity(request.rows() as usize);
        let decodes = self.executions.iter().find(|e| e.capacity() >= required)
            .context("missing preloaded EXL3 worker capacity")?.decodes_wire_rows();
        let device = self.inputs[0].buffer.device_id;
        let mut hidden = self.inputs[0].buffer;
        match hidden_view.filter(|view| view.bytes >= hidden_bytes) {
            Some(view) if self.host_path == HostPath::Async && decodes && view.ptr as usize % 16 == 0
                && view.device_id == device => hidden = CuteafdDeviceBuffer { bytes: hidden_bytes, ..view },
            // SAFETY: the view is device-visible request storage of at least
            // `hidden_bytes`, retained by the transport until this request's
            // response is emitted, which follows the stream wait below.
            Some(view) => unsafe {
                self.library.copy_d2d_async(self.inputs[0].buffer,
                    CuteafdDeviceBuffer { bytes: hidden_bytes, ..view }, hidden_bytes, self.stream.raw)?
            },
            None => self.library.copy_h2d(self.inputs[0].buffer, request.hidden())?,
        }
        let id_bytes = (routes + self.ownership_words) * 4;
        let staged = self.staging.buffer;
        let id_source = CuteafdHostBuffer { bytes: id_bytes, ..staged };
        let weight_source = CuteafdHostBuffer {
            // SAFETY: the weights region starts inside the pinned allocation.
            ptr: unsafe { staged.ptr.cast::<u8>().add(self.weights_offset) }.cast(),
            bytes: routes * 4,
            ..staged
        };
        match self.host_path {
            // SAFETY: both sources are pinned staging written above and left
            // untouched until this call's stream completes (below); both
            // destinations are this worker's input allocations.
            HostPath::Async => unsafe {
                self.library.copy_host_buffers_h2d_batch_async(
                    &[self.inputs[1].buffer, self.inputs[2].buffer],
                    &[id_source, weight_source],
                    &[id_bytes, routes * 4],
                    self.stream.raw,
                )?
            },
            HostPath::Blocking => unsafe {
                self.library.copy_h2d(self.inputs[1].buffer,
                    std::slice::from_raw_parts(id_source.ptr.cast::<u8>(), id_bytes))?;
                self.library.copy_h2d(self.inputs[2].buffer,
                    std::slice::from_raw_parts(weight_source.ptr.cast::<u8>(), routes * 4))?;
            },
        }
        let uploaded_us = started.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        if let Some([start, _]) = &self.timing {
            unsafe { self.library.cuda_event_record(start.raw, self.stream.raw)?; }
        }
        let inputs = [hidden, self.inputs[1].buffer, self.inputs[2].buffer];
        // Modules and storage are resolved before accepting requests. This
        // bounded selection performs no allocation, loading or compilation.
        let execution = self
            .executions
            .iter_mut()
            .find(|e| e.capacity() >= required)
            .context("missing preloaded EXL3 worker capacity")?;
        let output = unsafe {
            if self.ownership_words > 0 {
                let ownership = CuteafdDeviceBuffer {
                    ptr: self.inputs[1].buffer.ptr.cast::<u8>().add(routes * 4).cast(),
                    bytes: self.ownership_words * 4,
                    ..self.inputs[1].buffer
                };
                execution.launch_paired_layer_into(self.layer, inputs, request.rows() as usize,
                    self.stream.raw, destination, ownership)?
            } else { match destination {
                Some(output) => execution.launch_layer_into(
                    self.layer,
                    inputs,
                    request.rows() as usize,
                    self.stream.raw,
                    output,
                )?,
                None => execution.launch_layer(
                    self.layer,
                    inputs,
                    request.rows() as usize,
                    self.stream.raw,
                )?,
            } }
        };
        let enqueued_us = started.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        if let Some([_, end]) = &self.timing {
            unsafe { self.library.cuda_event_record(end.raw, self.stream.raw)?; }
        }
        match self.host_path {
            // Poll instead of sleeping in the driver: a blocking synchronize
            // returned about 8 us after the GPU finished (nsys of a GB10
            // worker in service), and the worker thread has nothing else to do
            // until the response is out.
            HostPath::Async => while !unsafe { self.library.cuda_stream_query(self.stream.raw)? } {
                std::hint::spin_loop();
            },
            HostPath::Blocking => unsafe { self.library.cuda_stream_synchronize(self.stream.raw)? },
        }
        let completed_us = started.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        if destination.is_none() {
            self.library
                .copy_d2h(&mut exchange.partials[..bytes], output)?;
        }
        if let (Some(started), Some([start, end])) = (started, &self.timing) {
            // This interval overlaps host enqueue/wait and includes all stream work,
            // including routing and output reduction, not just the expert kernel.
            let gpu_us = unsafe { self.library.cuda_event_elapsed_ms(start.raw, end.raw)? } * 1000.;
            let mut seen = vec![false; cuteafd_core::expert_geometry().experts as usize];
            let (ids, _) = self.staged_view(routes);
            for &id in ids {
                if let Some(value) = seen.get_mut(id as usize) { *value = true; }
            }
            tracing::info!(target: "cuteafd::worker_timing", executor_id, layer=request.layer(), rows=request.rows(),
                distinct_experts=seen.iter().filter(|&&v| v).count(), mapped=destination.is_some(),
                upload_us=uploaded_us, enqueue_us=enqueued_us-uploaded_us,
                wait_us=completed_us-enqueued_us, gpu_us, total_us=started.elapsed().as_micros() as u64,
                "EXL3 worker execution");
        }
        if self.route_dump.is_some() {
            let (layer, rows, topk) = (request.layer(), request.rows(), Self::topk() as u32);
            let mut dump = self.route_dump.take();
            let (ids, weights) = self.staged_view(routes);
            if let Some(writer) = &mut dump {
                if let Err(error) = writer.record(layer, rows, topk, ids, weights) {
                    // A debug capture never fails a request: stop capturing.
                    tracing::warn!(%error, "EXL3 route dump stopped");
                    dump = None;
                }
            }
            self.route_dump = dump;
        }
        Ok(())
    }

    /// # Safety
    /// The transport exclusively owns a GPU-accessible send slot on this device
    /// and retains it through the response send completion.
    pub(crate) unsafe fn execute_mapped_request(
        &mut self,
        request: &BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        slot: CuteafdDeviceBuffer,
        hidden: Option<CuteafdDeviceBuffer>,
    ) -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        if !request.permits_device_response() || slot.bytes < prefix + bytes {
            return Ok(None);
        }
        ensure!(!slot.ptr.is_null(), "null EXL3 response slot");
        let output = CuteafdDeviceBuffer {
            ptr: slot.ptr.cast::<u8>().add(prefix).cast(),
            bytes,
            ..slot
        };
        let response = request.response_device(executor_id, output)?;
        self.execute(request, executor_id, exchange, Some(output), hidden)?;
        Ok(Some(response))
    }

    pub(crate) fn execute_host_chunks<F>(
        &mut self,
        request: &BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        row_indices: &mut [u32],
        max_frame_bytes: usize,
        mut sink: F,
    ) -> Result<()>
    where
        F: FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    {
        let chunk_rows = request.response_chunk_rows(max_frame_bytes)?;
        ensure!(
            row_indices.len() >= chunk_rows as usize,
            "response row-index scratch is too short"
        );
        self.execute(request, executor_id, exchange, None, None)?;
        let stride = cuteafd_core::expert_geometry().row_bytes() as usize;
        for start in (0..request.rows()).step_by(chunk_rows as usize) {
            let end = start.saturating_add(chunk_rows).min(request.rows());
            sink(request.response_chunk(
                executor_id,
                start,
                &exchange.partials[start as usize * stride..end as usize * stride],
                row_indices,
                max_frame_bytes,
            )?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
