//! Native FP8 K32 Spark adapter with distinct decode and grouped prefill arenas.
use super::*;
use crate::shared::experts::exlr8::{Exlr8Execution, Exlr8Weights};
use crate::shared::memory::LoadStream;
use cuteafd_ffi::{CuteafdDeviceBuffer, programs::Programs};
use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};
use cuteafd_transport::{ExpertProtocolV2DeviceResponseRef, ExpertProtocolV2ResponseRef,
    ExpertV2Dtype, ProtocolV2RequestDevicePayload};

const K1_ROWS: usize = cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS;

const PREFILL_ROWS: usize = 4096;

fn chunks(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    let capacity = if rows <= K1_ROWS { K1_ROWS } else { PREFILL_ROWS };
    (0..rows).step_by(capacity).map(move |first| (first, (rows - first).min(capacity)))
}

fn execution_bytes(programs: &Programs<'_>, capacity: usize, unit_limit: usize,
    activations: crate::shared::experts::exlr8::Exlr8PrefillActivations) -> Result<usize> {
    ensure!((1..=PREFILL_ROWS).contains(&capacity), "EXLR8 capacity must be 1..4096");
    let decode = Exlr8Execution::workspace_bytes(programs, capacity.min(K1_ROWS), ExpertSet::Routed)?;
    let prefill = if capacity > K1_ROWS {
        Exlr8Execution::workspace_bytes_with_limit(programs, capacity, ExpertSet::Routed, unit_limit, activations)?
    } else { 0 };
    decode.checked_add(prefill).context("EXLR8 retained execution workspace overflow")
}

// This is a traffic cost per routed occurrence, not a staging reservation.
fn expert_read_bound(experts: usize, units: impl Iterator<Item = (usize, usize)>) -> Result<usize> {
    let mut bytes = vec![0usize; experts];
    for (expert, extent) in units {
        let total = bytes.get_mut(expert).context("EXLR8 expert outside resident domain")?;
        *total = total.checked_add(extent).context("EXLR8 expert read extent overflow")?;
    }
    Ok(bytes.into_iter().max().unwrap_or(0))
}

pub(super) struct Worker<'w, 'a> {
    // Drain before execution storage or the borrowed weights can be released.
    stream: LoadStream<'a>,
    execution: Exlr8Execution<'a>,
    prefill: Option<Exlr8Execution<'a>>,
    weights: &'w [Exlr8Weights<'a>],
    library: &'a NativeLibrary,
    selected: usize,
    capacity: usize,
    executor_id: u64,
    timing: Option<super::super::execution::timing::WorkerTiming<'a>>,
}

pub(super) fn load<'a>(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig, programs: &'a Programs<'a>,
    push: Option<&mut std::net::TcpStream>)
    -> Result<(backend::Weights<'a>, usize)> {
    let manifest = config.exlr8_manifest.as_deref().context("EXLR8 worker requires a manifest")?;
    let model = config.model_admission.as_ref().context("EXLR8 requires model admission")?;
    ensure!(model.endpoints.iter().all(|endpoint|
        endpoint.ingress == cuteafd_core::SparkIngress::Fp8K32),
        "EXLR8 worker requires standard FP8 K32 ingress and BF16 partials");
    let rows = config.capacity as usize;
    let fallback_staging = rows.checked_mul({ let h = catalog.routed_experts().geometry()?.hidden as usize; h + h / 32 }).context("EXLR8 fallback staging overflow")?;
    let layers = config.resident_layers(catalog.routed_experts().layers)?;
    let plans = layers.iter()
        .map(|&layer| Residency::plan(&config.snapshot, manifest, layer, config.world,
            config.rank, ExpertSet::Routed).map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    let unit_limit = plans.iter().map(Residency::compact_unit_limit).max().context("EXLR8 has no resident plans")?;
    let k1_workspace_bytes = execution_bytes(programs, rows, unit_limit, config.exlr8_prefill_activations)?;
    let workspace = k1_workspace_bytes.checked_add(fallback_staging)
        .context("EXLR8 workspace admission overflow")?;
    let budgets = plans.iter().map(Exlr8Weights::plan).collect::<Result<Vec<_>>>()?;
    let sum = |values: Vec<usize>| values.into_iter().try_fold(0usize, |total, bytes|
        total.checked_add(bytes).context("EXLR8 admission overflow"));
    let resident = sum(plans.iter().map(|plan| plan.resident_bytes).collect())?;
    let retained_host = sum(plans.iter().map(Residency::host_bytes).collect())?;
    let pinned = budgets.iter().map(|budget| budget.pinned_host_bytes).max().unwrap_or(0);
    let descriptors = budgets.iter().zip(&plans)
        .map(|(budget, plan)| budget.read_scratch_bytes - plan.host_bytes()).max().unwrap_or(0);
    let scratch = retained_host.checked_add(descriptors).context("EXLR8 host admission overflow")?;
    let retained = workspace.checked_add(retained_host).context("EXLR8 workspace admission overflow")?;
    let peak = spark_admission_budget(config, resident, 0, pinned, scratch, retained)?;
    ensure!(peak.load_peak <= config.device_budget && peak.serve_peak <= config.device_budget,
        "EXLR8 admission exceeds budget: load {} serve {} budget {}",
        peak.load_peak, peak.serve_peak, config.device_budget);
    admit_worker_peak(library, peak.load_peak, peak.serve_peak)?;
    if let Some(stream) = push {
        if let Some(weights) = super::push_exlr8::load(library, config, &plans, &budgets,
            resident, workspace, retained_host, descriptors, stream)? {
            return Ok((backend::Weights::Exlr8(weights), config.device_budget - resident - retained_host));
        }
    }
    let mut plans: Vec<_> = plans.into_iter().map(Some).collect();
    let mut remaining = config.device_budget;
    let weights = crate::shared::experts::load_in_rank_order(&layers, config.rank, |index, _| {
        let plan = plans[index].take().context("EXLR8 resident plan loaded twice")?;
        let bytes = plan.resident_bytes;
        let weight = Exlr8Weights::load(library, plan, remaining - workspace)?;
        remaining = remaining.checked_sub(bytes).context("EXLR8 resident budget exhausted")?;
        Ok(weight)
    })?;
    tracing::info!(rank = config.rank, resident, workspace, k1_workspace_bytes, fallback_staging, retained_host,
        load_peak = peak.load_peak, serve_peak = peak.serve_peak, "EXLR8 worker admitted");
    Ok((backend::Weights::Exlr8(weights), remaining - retained_host))
}

impl<'w, 'a> Worker<'w, 'a> {
    pub(super) fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>,
        weights: &'w [Exlr8Weights<'a>], config: &NativeExpertServiceConfig,
        remaining: usize) -> Result<Self> {
        ensure!(!weights.is_empty(), "EXLR8 worker has no resident layers");
        let capacity = config.capacity as usize;
        let unit_limit = weights.iter().map(|weight| weight.layout.compact_unit_limit()).max()
            .context("EXLR8 has no resident layers")?;
        let workspace = execution_bytes(programs, capacity, unit_limit, config.exlr8_prefill_activations)?;
        ensure!(workspace <= remaining, "EXLR8 retained arenas exceed admitted bytes");
        let decode_bytes = Exlr8Execution::workspace_bytes(programs, capacity.min(K1_ROWS), ExpertSet::Routed)?;
        let execution = Exlr8Execution::new(library, programs,
            capacity.min(K1_ROWS), decode_bytes, ExpertSet::Routed)?;
        let prefill = if capacity > K1_ROWS {
            Some(Exlr8Execution::new_with_limit(library, programs, capacity,
                workspace - decode_bytes, ExpertSet::Routed, unit_limit, config.exlr8_prefill_activations)?)
        } else { None };
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        let timing = super::super::execution::timing::WorkerTiming::from_env(library, config.rank)?;
        Ok(Self { stream, execution, prefill, weights, library, selected: 0,
            capacity: config.capacity as usize,
            executor_id: config.spark_world()?.topology().executor_id(config.rank)?, timing })
    }
    pub(super) fn bind_layer(&mut self, index: usize) -> Result<()> {
        ensure!(index < self.weights.len(), "EXLR8 requested layer is not resident");
        self.stream.require_complete()?;
        self.selected = index;
        Ok(())
    }
    pub(super) fn grouping(&self, loaded: cuteafd_transport::worker_selection::WorkerSelection,
        rank: usize) -> Result<cuteafd_transport::worker_selection::InstalledExpertGrouping> {
        let read_bytes = self.weights.iter().map(|weight| expert_read_bound(weight.layout.experts,
            weight.layout.loads.iter().map(|unit| (unit.expert, unit.staging_bytes()))))
            .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
        backend::conservative_grouping(loaded, rank, self.capacity, read_bytes)
    }

    fn execute(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, destination: Option<CuteafdDeviceBuffer>,
        mapped: Option<ProtocolV2RequestDevicePayload>) -> Result<()> {
        ensure!(!request.is_native_group(), "EXLR8 requires canonical requests");
        request.require_input_dtype(ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32)?;
        ensure!(request.rows() > 0 && request.rows() as usize <= self.capacity,
            "EXLR8 request exceeds admitted capacity");
        ensure!(executor_id == self.executor_id && request.layer() as usize == self.weights[self.selected].layout.layer,
            "EXLR8 request executor or resident layer mismatch");
        ensure!(exchange.partials.len() >= request.plane_bytes()?, "EXLR8 BF16 exchange is too small");
        let geometry = cuteafd_core::expert_geometry();
        let input_stride = geometry.hidden as usize + geometry.hidden as usize / 32;
        let output_stride = geometry.hidden as usize * 2;
        let topk = geometry.topk as usize;
        if mapped.is_none() {
            request.copy_routes_into(&mut exchange.ids, &mut exchange.routing)?;
        }
        let execution = if request.rows() as usize > K1_ROWS {
            self.prefill.as_ref().context("EXLR8 grouped prefill was not admitted")?
        } else { &self.execution };
        let work = (|| -> Result<()> {
            for (first, rows) in chunks(request.rows() as usize) {
                let routes = rows * topk;
                if let Some(mapped) = mapped {
                    let hidden = super::super::exlr8::view(mapped.hidden_payload, first * input_stride, rows * input_stride)?;
                    let records = super::super::exlr8::view(mapped.routes, first * topk * 12, routes * 12)?;
                    // SAFETY: validated mapped request and exclusive arena remain live
                    // through the unconditional stream drain after this closure.
                    unsafe {
                        library_copy(self.library, execution.input()?, hidden, hidden.bytes, self.stream.raw)?;
                        if let Some(timing) = &self.timing { timing.record(0, self.stream.raw)?; }
                        super::routes::copy_route_records(self.library, records,
                            execution.ids()?, execution.route_weights()?, routes, self.stream.raw)?;
                    }
                } else {
                    self.library.copy_h2d(execution.input()?,
                        &request.hidden()[first * input_stride..(first + rows) * input_stride])?;
                    let range = first * topk..(first + rows) * topk;
                    // SAFETY: contiguous initialized i32/f32 arrays, read as their
                    // native little-endian wire bytes only during blocking copies.
                    unsafe {
                        let ids = std::slice::from_raw_parts(exchange.ids[range.clone()].as_ptr().cast(), routes * 4);
                        let gates = std::slice::from_raw_parts(exchange.routing[range].as_ptr().cast(), routes * 4);
                        self.library.copy_h2d(execution.ids()?, ids)?;
                        self.library.copy_h2d(execution.route_weights()?, gates)?;
                        if let Some(timing) = &self.timing { timing.record(0, self.stream.raw)?; }
                    }
                }
                // SAFETY: weights, arena, timing events and inputs remain owned through the drain.
                unsafe {
                    execution.run(&self.weights[self.selected], rows, self.stream.raw)?;
                    if let Some(timing) = &self.timing { timing.record(1, self.stream.raw)?; }
                }
                let output = super::super::exlr8::view(execution.output()?, 0, rows * output_stride)?;
                if let Some(destination) = destination {
                    let target = super::super::exlr8::view(destination, first * output_stride, output.bytes)?;
                    // SAFETY: transport owns the destination until response send completes.
                    unsafe { library_copy(self.library, target, output, output.bytes, self.stream.raw)?; }
                }
                // Finish arena readers before
                // any default-stream upload, download or mapped frame release.
                // SAFETY: this worker exclusively owns the stream and all queued views.
                unsafe { self.library.cuda_stream_synchronize(self.stream.raw)?; }
                if let Some(timing) = &mut self.timing {
                    // SAFETY: the existing completion barrier finished both event records.
                    unsafe { timing.publish(request.request_id(), request.layer(), rows as u32, executor_id, "exlr8")?; }
                }
                if destination.is_none() {
                    self.library.copy_d2h(&mut exchange.partials[first * output_stride..(first + rows) * output_stride], output)?;
                }
            }
            Ok(())
        })();
        // An earlier copy may have queued even if a later copy or launch failed.
        // SAFETY: all borrowed request/response views still outlive this drain.
        let drain = unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }.map_err(anyhow::Error::from);
        if drain.is_err() {
            crate::shared::decode_graph::fatal_drain(drain, "EXLR8 mapped request");
        }
        work
    }
    /// # Safety
    /// The transport retains exclusive mapped request/response slots until this
    /// method drains its stream and the emitted response finishes sending.
    pub(super) unsafe fn execute_mapped(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, mapped: ProtocolV2RequestDevicePayload)
        -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        let Some(slot) = mapped.response_slot else { return Ok(None); };
        let prefix = cuteafd_transport::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        if !request.permits_device_response() || slot.bytes < prefix + bytes { return Ok(None); }
        let output = super::super::exlr8::view(slot, prefix, bytes)?;
        let response = request.response_device(executor_id, output)?;
        self.execute(request, executor_id, exchange, Some(output), Some(mapped))?;
        Ok(Some(response))
    }
    pub(super) fn execute_host_chunks<F>(&mut self, request: &BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, row_indices: &mut [u32], max_frame_bytes: usize,
        mut sink: F) -> Result<()> where F: FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()> {
        let chunk_rows = request.response_chunk_rows(max_frame_bytes)? as usize;
        ensure!(row_indices.len() >= chunk_rows, "EXLR8 row-index scratch too short");
        self.execute(request, executor_id, exchange, None, None)?;
        let stride = cuteafd_core::expert_geometry().hidden as usize * 2;
        for first in (0..request.rows() as usize).step_by(chunk_rows) {
            let end = (first + chunk_rows).min(request.rows() as usize);
            sink(request.response_chunk(executor_id, first as u32,
                &exchange.partials[first * stride..end * stride], row_indices, max_frame_bytes)?)?;
        }
        Ok(())
    }
}

/// # Safety
/// Both buffers belong to this device and stay live until the stream drains.
unsafe fn library_copy(library: &NativeLibrary, target: CuteafdDeviceBuffer,
    source: CuteafdDeviceBuffer, bytes: usize, stream: *mut std::ffi::c_void) -> Result<()> {
    ensure!(target.device_id == source.device_id, "EXLR8 mapped input/output device mismatch");
    // SAFETY: caller retains the checked buffers through the stream drain.
    unsafe { library.copy_d2d_async(target, source, bytes, stream)?; }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exlr8_load_order_starts_after_peak_admission() {
        let source = include_str!("exlr8.rs").split("#[cfg(test)]").next().unwrap();
        let admission = source.find("admit_worker_peak(library, peak.load_peak, peak.serve_peak)?").unwrap();
        let order = source.find("crate::shared::experts::load_in_rank_order(&layers, config.rank").unwrap();
        let take = source.find("plans[index].take()").unwrap();
        let upload = source.find("Exlr8Weights::load(library, plan, remaining - workspace)?").unwrap();
        assert!(admission < order && order < take && take < upload);
    }

    #[test]
    fn exlr8_traffic_bound_charges_each_route_not_one_layer_per_route() -> Result<()> {
        use cuteafd_transport::worker_selection::{WorkerIdentity, WorkerSelection};
        let root = tempfile::tempdir()?;
        std::fs::write(root.path().join("config.json"), b"{}")?;
        std::fs::write(root.path().join("model.safetensors.index.json"), b"{}")?;
        let identity = WorkerIdentity::from_snapshot(root.path(), cuteafd_core::ExpertGeometry::GLM5,
            cuteafd_transport::expert::SparkWorld::canonical(4)?)?;
        let loaded = WorkerSelection::new(identity, &[3])?;
        let units = [(0, 7), (1, 9), (0, 5), (1, 11)];
        let largest = expert_read_bound(2, units.into_iter())?;
        assert_eq!(largest, 20);
        assert!(largest < units.iter().map(|(_, bytes)| bytes).sum::<usize>());
        assert!(expert_read_bound(2, [(2, 1)].into_iter()).is_err());
        assert!(expert_read_bound(2, [(0, usize::MAX), (0, 1)].into_iter()).is_err());
        let report = backend::conservative_grouping(loaded, 0, 4096, largest)?;
        for rows in [1, 16, 64, 65, 4096] {
            assert_eq!(report.group_rows(rows)?, 1);
            let occurrences = rows * 8;
            let charged = occurrences * report.weight_read_bytes()?;
            let actual = (0..occurrences).map(|route| if route % 2 == 0 { 12 } else { 20 }).sum::<usize>();
            assert!(charged >= actual);
        }
        Ok(())
    }

    #[test]
    fn exlr8_chunks_cover_every_row_once_with_distinct_wire_strides() {
        for rows in [1, 16, 63, 64, 65, 128, 4095, 4096] {
            let mut next = 0;
            for (first, count) in chunks(rows) {
                assert_eq!(first, next);
                assert!((1..=if rows <= K1_ROWS { K1_ROWS } else { PREFILL_ROWS }).contains(&count));
                assert_eq!(count, rows);
                let wire_stride = 6144 + 6144 / 32;
                let input = first * wire_stride..(first + count) * wire_stride;
                let output = first * 6144 * 2..(first + count) * 6144 * 2;
                let routes = first * 8 * 12..(first + count) * 8 * 12;
                assert_eq!(input.len(), count * wire_stride);
                assert_eq!(output.len(), count * 6144 * 2);
                assert_eq!(routes.len(), count * 8 * 12);
                next += count;
            }
            assert_eq!(next, rows);
        }
    }
}
