//! One coordinator wave owns TP route planes through final native reduction.
use super::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{
    replicated_expert_tie_seed_for, ReplicatedExpertCostModel, ReplicatedExpertScheduleConfig,
    ReplicatedExpertScheduler, ReplicatedExpertTieSeedMode, INACTIVE_REPLICATED_EXPERT_GROUP,
};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{
    expert::{SparkTopology, SparkExperts, V41_PARTIAL_ROW_BYTES, V41_ROUTED_EXPERTS},
    ExpertProtocolV2Request,
};

/// One replicated lane's whole-expert ownership planner. Preallocated once per
/// independent lane; every field is lane-local, so two lanes can never perturb
/// each other's assignment. The histogram and encoded owners are reused across
/// layers and requests without allocating in the request path.
pub(crate) struct ReplicatedGroupPlanner {
    topology: SparkTopology,
    tie_seed_mode: ReplicatedExpertTieSeedMode,
    scheduler: ReplicatedExpertScheduler,
    histogram: [u32; V41_ROUTED_EXPERTS],
    owners: [u8; V41_ROUTED_EXPERTS],
}

/// Opt-in tie-seed control. Unset selects `dispatch`, which reproduces the
/// historical per-dispatch-counter seed byte for byte. `layer` is the narrow
/// diagnostic mode that pins the request component so repeats of the same work
/// at the same layer share ownership when their histogram is unchanged.
pub(crate) const REPLICATED_EXPERT_TIE_SEED_ENV: &str = "CUTEAFD_REPLICATED_EXPERT_TIE_SEED";

/// Whole-expert weight-only profile: one positive weight unit per active expert
/// and no per-row activation term, so ownership depends only on which experts a
/// batch routed to. `tile_rows = 16` keeps the model valid without claiming a
/// calibrated tile cost; supplying calibrated costs is an explicit opt-in
/// environment override, never an invented profile.
fn parse_replicated_cost(value: &str) -> Result<ReplicatedExpertCostModel> {
    let parts = value.split(',').map(str::trim).collect::<Vec<_>>();
    ensure!(
        parts.len() == 3,
        "CUTEAFD_REPLICATED_EXPERT_COST expects weight,tile_cost,tile_rows"
    );
    let weight = parts[0].parse::<u64>().context("replicated expert weight cost")?;
    let tile_cost = parts[1].parse::<u64>().context("replicated expert tile cost")?;
    let tile_rows = parts[2].parse::<u32>().context("replicated expert tile rows")?;
    let model = ReplicatedExpertCostModel::new(weight, tile_cost, tile_rows);
    model
        .validate()
        .map_err(|error| anyhow::anyhow!("invalid replicated expert cost model: {error}"))?;
    Ok(model)
}

fn replicated_cost_model() -> Result<ReplicatedExpertCostModel> {
    match std::env::var_os("CUTEAFD_REPLICATED_EXPERT_COST") {
        None => Ok(ReplicatedExpertCostModel::new(1, 0, 16)),
        Some(value) => parse_replicated_cost(
            value
                .to_str()
                .context("CUTEAFD_REPLICATED_EXPERT_COST is not valid UTF-8")?,
        ),
    }
}

/// Resolve the opt-in tie-seed mode once, at planner construction. Missing
/// environment selects the historical `dispatch` mode; every present value must
/// be exactly `dispatch` or `layer`, so a typo fails closed before any request.
fn replicated_tie_seed_mode() -> Result<ReplicatedExpertTieSeedMode> {
    let value = match std::env::var_os(REPLICATED_EXPERT_TIE_SEED_ENV) {
        None => None,
        Some(raw) => Some(
            raw.into_string()
                .map_err(|_| anyhow::anyhow!("{REPLICATED_EXPERT_TIE_SEED_ENV} is not valid UTF-8"))?,
        ),
    };
    ReplicatedExpertTieSeedMode::from_env_value(value.as_deref())
        .map_err(|error| anyhow::anyhow!("{error}"))
}

impl ReplicatedGroupPlanner {
    pub(crate) fn new(topology: SparkTopology) -> Result<Self> {
        Self::new_with_seed_mode(topology, replicated_tie_seed_mode()?)
    }

    /// Constructor with an explicit mode, used by focused tests so they never
    /// mutate process environment. Logs the resolved mode once per process.
    fn new_with_seed_mode(
        topology: SparkTopology,
        tie_seed_mode: ReplicatedExpertTieSeedMode,
    ) -> Result<Self> {
        static TIE_SEED_MODE_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        TIE_SEED_MODE_LOGGED.get_or_init(|| {
            tracing::info!(
                env = REPLICATED_EXPERT_TIE_SEED_ENV,
                mode = tie_seed_mode.as_str(),
                "replicated expert tie-seed mode resolved"
            );
        });
        let config = ReplicatedExpertScheduleConfig::new(
            topology.group_count(),
            replicated_cost_model()?,
        );
        let scheduler = ReplicatedExpertScheduler::new(config, V41_ROUTED_EXPERTS)
            .map_err(|error| anyhow::anyhow!("replicated expert scheduler rejected: {error}"))?;
        Ok(Self {
            topology,
            tie_seed_mode,
            scheduler,
            histogram: [0; V41_ROUTED_EXPERTS],
            owners: [INACTIVE_REPLICATED_EXPERT_GROUP; V41_ROUTED_EXPERTS],
        })
    }

    #[cfg(test)]
    fn seed_mode(&self) -> ReplicatedExpertTieSeedMode {
        self.tie_seed_mode
    }

    pub(crate) fn topology(&self) -> SparkTopology {
        self.topology
    }

    /// Plan whole-expert ownership for one request from the router's host-side
    /// route IDs (no device read and no allocation), copy the encoded assignment
    /// into the request-owned route words, and prove the ownership contract
    /// before any bytes travel. Planning exactly once per request keeps the
    /// assignment reproducible from `(layer, request_id, histogram)`.
    pub(crate) fn encode(&mut self, request: &mut ExpertProtocolV2Request) -> Result<()> {
        ensure!(
            request.header.row_count > 0
                && request.header.row_count <= 4096
                && request.routes.len() == request.header.row_count as usize * 6,
            "replicated expert ownership needs one canonical bounded request"
        );
        self.histogram.fill(0);
        for route in &request.routes {
            let expert = route.expert_id as usize;
            ensure!(
                expert < V41_ROUTED_EXPERTS,
                "replicated expert route id out of range"
            );
            self.histogram[expert] += 1;
        }
        let seed = replicated_expert_tie_seed_for(
            self.tie_seed_mode,
            request.header.layer_id,
            request.header.request_id,
        );
        {
            let plan = self
                .scheduler
                .plan(&self.histogram, seed)
                .map_err(|error| anyhow::anyhow!("replicated expert schedule rejected: {error}"))?;
            ensure!(
                plan.group_count() == self.topology.group_count(),
                "replicated expert plan group count differs from the bound topology"
            );
            self.owners.fill(INACTIVE_REPLICATED_EXPERT_GROUP);
            for (expert, group) in plan.assignment().iter().enumerate() {
                // Every routed expert is active, so it carries an active owner;
                // the transport rejects inactive or out-of-range owners.
                self.owners[expert] = group.encoded();
            }
        }
        request.with_native_group_owners(&self.owners, self.topology)
    }
}

pub(crate) struct NativeTp4Wave<'a> {
    link: crate::shared::spark_intake::SparkLink<'a>,
    // Drop drains the stream before fields release any GPU allocations.
    stream: LoadStream<'a>,
    output: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    ready_rows: Option<u32>,
    local: Option<super::local::LocalExpertWave<'a>>,
    tp2: Option<Box<super::tp2_ffn::Wave<'a>>>,
    /// Present only for an explicit `TP×EP` topology; legacy transports stay on
    /// the canonical request contract. Shared with the exchange worker, which
    /// encodes each request's ownership in issue order.
    native: Option<(SparkTopology, std::sync::Arc<std::sync::Mutex<ReplicatedGroupPlanner>>)>,
}

impl<'a> NativeTp4Wave<'a> {
    pub(crate) fn spark_world(&self) -> usize { self.link.world_size() }
    /// Explicit replicated topology of this lane, or `None` for the legacy path.
    pub(crate) fn native_topology(&self) -> Option<SparkTopology> {
        self.native.as_ref().map(|(topology, _)| *topology)
    }

    pub fn install_local(&mut self, wave: super::local::LocalExpertWave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.local = Some(wave);
        Ok(())
    }
    pub fn has_local_layer(&self, layer: usize) -> bool {
        self.local.as_ref().is_some_and(|wave| wave.contains(layer)) || self.has_tp2_layer(layer)
    }
    pub fn install_tp2(&mut self, wave: super::tp2_ffn::Wave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.tp2 = Some(Box::new(wave));
        Ok(())
    }
    pub fn has_tp2_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains(layer))
    }
    pub fn has_tp2_shared_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains_shared(layer))
    }
    /// # Safety
    /// Input and router producers have completed. Both borrowed outputs remain
    /// immutable through this operation, including cancellation draining.
    pub async unsafe fn execute_tp2_ffn(&mut self, input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        ensure!(binding == input.binding() && input.layer == routed.layer
            && input.tokens.len() == routed.rows as usize && input.tokens == routed.tokens,
            "TP2 FFN input/router identity mismatch");
        let values = unsafe { self.tp2.as_mut().context("TP2 expert lane missing")?
            .execute(input.layer, routed.rows, input.values, routed.expert_input, routed.ids, routed.routing).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared buffers stay live and unmodified through drain.
    pub unsafe fn execute_local_ffn(&mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?.execute(routed, shared)? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding: routed.binding()?, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared inputs remain immutable until completion or drain.
    pub async unsafe fn execute_local_ffn_cooperative(&mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?
            .execute_cooperative(routed, shared).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// Admission invalidates prior output; completed RoCE sessions remain reusable.
    pub fn begin_request(&mut self) {
        self.ready_rows = None;
    }
    /// After a cancelled or failed pass whose streams drained: stops the
    /// exchange, resets the connections and starts it again.
    pub fn reset_connections(&mut self) -> Result<()> {
        self.ready_rows = None;
        self.link.reset_connections()
    }
    /// Legacy TP4 upper-bound reservation (four rank planes), also sufficient
    /// for a legacy TP2 transport.
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        Self::device_bytes_for(capacity, 4)
    }
    /// Reservation for exactly `ranks` physical rank planes plus the final
    /// output plane. Six-rank layouts need their own, larger reservation.
    pub fn device_bytes_for(capacity: u32, ranks: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP wave capacity"
        );
        ensure!(
            (1..=8).contains(&ranks),
            "native TP wave requires one through eight ranks"
        );
        (capacity as usize)
            .checked_mul(ranks * V41_PARTIAL_ROW_BYTES as usize + 2 * 5120 * 2)
            .context("native TP wave budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        transport: SparkExperts,
        available_bytes: usize,
    ) -> Result<Self> {
        let capacity = transport.capacity();
        let world_size = transport.world_size();
        ensure!(
            (1..=8).contains(&world_size),
            "native TP wave requires one through eight ranks"
        );
        ensure!(
            Self::device_bytes_for(capacity, world_size)? <= available_bytes,
            "native TP wave exceeds device budget"
        );
        let reducer = library.v41_compact_reducer()?;
        // Prove the exact physical-rank reduction is available before any plane
        // or output allocation. Legacy two-rank transports also fail here rather
        // than at their first request.
        reducer.require_rank_count(world_size as u32)?;
        let native = transport
            .ownership_topology()
            .map(|topology| ReplicatedGroupPlanner::new(topology)
                .map(|planner| (topology, std::sync::Arc::new(std::sync::Mutex::new(planner)))))
            .transpose()?;
        let timing = crate::families::deepseek_v41::v41_native_serve::protocol_v2_timing();
        let hidden_bytes = capacity as usize * 5120 * 2;
        Ok(Self {
            link: crate::shared::spark_intake::SparkLink::from_transport(library, transport,
                V41_PARTIAL_ROW_BYTES as usize, crate::shared::spark_intake::IntakePreference::Probed, timing)?,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            output: DeviceAllocation::new(library, hidden_bytes)?,
            library,
            ready_rows: None,
            local: None,
            tp2: None,
            native,
        })
    }

    /// Opens every rank's session once the workers serve `layer` (a remote
    /// layer) and starts the link's exchange worker. The widest request (the
    /// transport's capacity, in the checkpoint's wire format) sizes the rings;
    /// only the control bootstrap runs, nothing is posted.
    pub(crate) fn bootstrap(&mut self, layer: u32, nvfp4: bool) -> Result<()> {
        use cuteafd_transport::{ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind};
        let rows = self.link.capacity();
        let (dtype, width) = if nvfp4 { (ExpertV2Dtype::Bf16, 10240) } else { (ExpertV2Dtype::Fp8E4m3Ue8m0K32, 5280) };
        let descriptors = (0..rows).map(|row| ExpertProtocolV2RowDescriptor { row_id: row.into(),
            source_kind: ExpertV2SourceKind::Prefill, source_request_id: 1, token_position: row.into(),
            route_offset: row * 6, route_count: 6 }).collect();
        let routes = (0..rows * 6).map(|i| ExpertProtocolV2RouteEntry { row_index: i / 6,
            expert_id: i % V41_ROUTED_EXPERTS as u32, gate_weight: 0.0 }).collect();
        let mut request = ExpertProtocolV2Request::new(1, 0, layer, 5120, dtype, descriptors, routes,
            vec![0; rows as usize * width])?;
        request.header.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        if let Some((_, planner)) = &self.native {
            planner.lock().map_err(|_| anyhow::anyhow!("replicated expert planner poisoned"))?.encode(&mut request)?;
        }
        self.link.bootstrap(&request)
    }

    /// Sends a remote layer's wave on the link's exchange: the router's route
    /// ids, gate weights and expert input rows copy into the wave's staging on
    /// the router stream (joined to the chain), and the exchange worker builds,
    /// posts and lands it. Nothing waits on the host. Finish with
    /// [`NativePendingFfn::finish`] (or its TP2 form).
    ///
    /// # Safety
    /// The router output was produced in stream order on its own stream (the
    /// chain head or `router_stream`) and stays unchanged until the wave's
    /// staging copies ran, which later work on `router_stream` orders.
    pub(crate) unsafe fn dispatch_ffn<'w>(&'w mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        router_stream: *mut std::ffi::c_void, placement: u64,
        rows: &[crate::families::deepseek_v41::v41_backbone_router::ExpertRow]) -> Result<NativePendingFfn<'w, 'a>> {
        self.ready_rows = None;
        let capacity = self.link.capacity();
        ensure!(routed.rows > 0 && routed.rows <= capacity, "native dispatched FFN rows exceed capacity");
        let (binding, build) = routed.request_builder(placement, rows,
            self.native.as_ref().map(|(_, planner)| std::sync::Arc::clone(planner)))?;
        let (layout, sources) = routed.wave_sources()?;
        // Stage on this wave's own stream when the router shares its GPU: the
        // lane's owners (the router among them) then carry nothing past their
        // per-layer host waits, which a host-ordered pass relies on when it
        // rebinds them (`LoadStream::require_complete`). A router on the other
        // GPU (TP2 shared, device-ordered passes) stages from its own stream.
        let stage = if sources[0].device_id == self.link.device() { self.stream.raw } else { router_stream };
        let exchange = self.link.exchange()?;
        // SAFETY: the sources are the router's live device outputs, complete in
        // chain order: `stage` joins the chain head, which the router's stage
        // finished (caller contract).
        let ticket = unsafe {
            crate::shared::memory::chain::join(self.library, stage)?;
            exchange.send(routed.rows as usize, layout, sources, stage, build)?
        };
        Ok(NativePendingFfn { ticket, binding, layer: routed.layer, rows: routed.rows, library: self.library,
            stream: &self.stream, link: &mut self.link, output: self.output.buffer, ready_rows: &mut self.ready_rows,
            tp2: self.tp2.as_deref_mut() })
    }
    /// Borrowed device view; never free it or retain it across wave reuse/drop.
    pub fn output(&self) -> Result<CuteafdDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("native TP wave output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * 5120 * 2;
        Ok(output)
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }
}
impl Drop for NativeTp4Wave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native coordinator TP wave");
        }
    }
}

/// Complete ordered TP reduction plus the shared expert, borrowed until consumed.
pub(crate) struct NativeFfnOutput<'a> {
    pub values: CuteafdDeviceBuffer,
    binding: crate::families::deepseek_v41::v41_attention_binding::QueryBinding,
    _owner: std::marker::PhantomData<&'a ()>,
}
impl NativeFfnOutput<'_> {
    pub fn binding(&self) -> crate::families::deepseek_v41::v41_attention_binding::QueryBinding {
        self.binding
    }
}

/// A remote layer's wave on the exchange, borrowed until its reduction is
/// queued. Dropping it unfinished leaves the wave unlanded: the owner's pass
/// fails and resets the link (stopping the exchange) after its streams drain.
pub(crate) struct NativePendingFfn<'w, 'a> {
    ticket: crate::shared::spark_waves::WaveTicket,
    binding: crate::families::deepseek_v41::v41_attention_binding::QueryBinding,
    layer: usize,
    rows: u32,
    library: &'a NativeLibrary,
    stream: &'w LoadStream<'a>,
    link: &'w mut crate::shared::spark_intake::SparkLink<'a>,
    output: CuteafdDeviceBuffer,
    ready_rows: &'w mut Option<u32>,
    tp2: Option<&'w mut super::tp2_ffn::Wave<'a>>,
}
impl<'w> NativePendingFfn<'w, '_> {
    /// # Safety
    /// Shared values hold the completed contribution for this exact request and
    /// remain immutable until the final reduction completes on the stream.
    pub async unsafe fn finish(self,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'w>> {
        unsafe { self.finish_inner(shared, false).await }
    }
    /// # Safety
    /// Same input retention as finish; cancellation drains before releasing frames.
    pub async unsafe fn finish_cooperative(self,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'w>> {
        unsafe { self.finish_inner(shared, true).await }
    }
    /// Run shared TP2 after the Spark dispatch, reduce on the transport GPU, and
    /// return the result to the block's GPU. The pending owner retains every lane workspace.
    /// # Safety
    /// Input is the completed normalized FFN input for the dispatched request;
    /// its storage remains immutable through completion or cancellation drain.
    pub async unsafe fn finish_tp2(mut self, input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>) -> Result<NativeFfnOutput<'w>> {
        ensure!(self.binding == input.binding() && self.layer == input.layer
            && self.rows as usize == input.tokens.len() && matches!(input.values.device_id, 0 | 1),
            "TP2 shared input/request/device differs");
        let values = unsafe { self.tp2.as_mut().context("TP2 shared workspace missing")?
            .execute_shared_on(input.layer, self.rows, input.values, self.output.device_id as usize).await? };
        let destination = input.values.device_id;
        if destination != self.output.device_id {
            let device = crate::shared::memory::device::Device { library: self.library, id: self.output.device_id };
            device.future(unsafe { self.finish_values(values, true, Some(destination)) }).await
        } else {
            unsafe { self.finish_values(values, true, Some(destination)).await }
        }
    }
    async unsafe fn finish_inner(self,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>, cooperative: bool) -> Result<NativeFfnOutput<'w>> {
        ensure!(self.binding == shared.binding()? && self.layer == shared.layer && self.rows == shared.rows
            && shared.values.bytes == self.rows as usize * 10240 && shared.values.device_id == self.output.device_id,
            "native TP shared contribution differs from routed request");
        unsafe { self.finish_values(shared.values, cooperative, None).await }
    }
    async unsafe fn finish_values(mut self, values: CuteafdDeviceBuffer, cooperative: bool,
        destination: Option<i32>) -> Result<NativeFfnOutput<'w>> {
        ensure!(values.device_id == self.output.device_id && values.bytes == self.rows as usize * 10240,
            "shared reduction device/extent differs");
        let timing = std::time::Instant::now();
        let rows = self.rows;
        let slot = self.ticket.slot();
        let exchange = self.link.exchange()?;
        // A host-ordered pass (1 RTX) collects as before the exchange: the
        // host waits for the reply here, where everything this layer can
        // issue (router, staging, shared FFN, L2 prefetch) is already queued,
        // and only then queues the landing and reduce, so no stream sits
        // blocked on a device wait through the round trip. A device-ordered
        // pass queues them at once (the device waits).
        let host_ordered = !crate::shared::memory::chain::deferred();
        if host_ordered {
            // `--host-wait block` spins briefly before it sleeps: a reply
            // usually lands within that, and this wait is the critical path.
            let mut backoff = cuteafd_transport::wait_policy::Backoff::new();
            while !exchange.landed(&self.ticket) {
                crate::shared::spark_waves::healthy()?;
                if cooperative { tokio::task::yield_now().await } else { backoff.pause() }
            }
        }
        // SAFETY: the reduce stream joins the chain (the shared contribution's
        // producers), then waits on the device for this wave's planes; the
        // intake planes, shared values and output stay live until the stream
        // completes (the pass drains before releasing them).
        unsafe {
            crate::shared::memory::chain::join(self.library, self.stream.raw)?;
            exchange.land(self.ticket, self.stream.raw)?;
            exchange.reduce(slot, values.ptr.cast(), self.output.ptr.cast(), rows as usize, self.stream.raw)?;
        }
        if cooperative {
            // SAFETY: the pending owner retains every input until completion or cancellation.
            unsafe { crate::shared::memory::chain::finish_cooperative(self.stream).await? };
        } else {
            // SAFETY: the same stream owns the reduction and its inputs.
            unsafe { crate::shared::memory::chain::finish(self.library, self.stream.raw)? };
        }
        if host_ordered && crate::shared::memory::chain::active() {
            // The layer's one host wait for its streams: the reduce joined the
            // chain head (attention, router, shared FFN), so its completion is
            // what the next layer's rebinds check (`require_complete`).
            if cooperative { self.stream.wait().await?; }
            // SAFETY: this owner's live stream.
            else { unsafe { self.library.cuda_stream_synchronize(self.stream.raw)? }; }
        }
        tracing::debug!(target: "cuteafd::timing", layer = self.layer, rows,
            collect_us = timing.elapsed().as_micros() as u64, "target collection");
        *self.ready_rows = Some(rows);
        let mut values = self.output;
        values.bytes = rows as usize * 10240;
        if let Some(device) = destination.filter(|&device| device != values.device_id) {
            values = unsafe { self.tp2.as_mut().context("TP2 return workspace missing")?
                .return_result(values, device as usize, rows).await? };
        }
        Ok(NativeFfnOutput {
            values,
            binding: self.binding,
            _owner: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod replicated_tests {
    use super::*;
    use cuteafd_transport::{
        expert::{BackboneRequest, V41NativeOwnerRouteWord, V41_NATIVE_GROUP_REQUEST_FLAG},
        ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry, ExpertV2Dtype, ExpertV2SourceKind,
    };

    fn request(layer: u32, request_id: u64, experts: &[u32]) -> ExpertProtocolV2Request {
        let rows = experts.len() / 6;
        assert_eq!(rows * 6, experts.len());
        let mut request = ExpertProtocolV2Request::new(
            request_id,
            7,
            layer,
            5120,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..rows).map(|r| ExpertProtocolV2RowDescriptor {
                row_id: r as u64,
                source_kind: ExpertV2SourceKind::Decode,
                source_request_id: request_id,
                token_position: r as u64,
                route_offset: r as u32 * 6,
                route_count: 6,
            }).collect(),
            experts.iter().enumerate().map(|(i, &expert_id)| ExpertProtocolV2RouteEntry {
                row_index: (i / 6) as u32,
                expert_id,
                gate_weight: 1.0 / 6.0,
            }).collect(),
            vec![0; rows * 5280],
        ).unwrap();
        request.header.flags |=
            cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        request
    }

    fn owners(request: &ExpertProtocolV2Request, topology: SparkTopology) -> Vec<u8> {
        request.routes.iter().map(|route| {
            let word = V41NativeOwnerRouteWord::decode(route.expert_id, topology.group_count()).unwrap();
            assert!(word.expert_id < 384);
            word.owner
        }).collect()
    }

    #[test]
    fn six_route_batches_split_evenly_across_replicated_groups() {
        let unique = [0u32, 7, 19, 88, 200, 383];
        for (topology, expected) in [
            (SparkTopology::new(2, 2).unwrap(), 3usize),
            (SparkTopology::new(3, 2).unwrap(), 3),
            (SparkTopology::new(2, 3).unwrap(), 2),
        ] {
            let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
            let mut request = request(3, 11, &unique);
            planner.encode(&mut request).unwrap();
            assert_ne!(request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG, 0);
            BackboneRequest::validate_owned_native_group(&request, 1, topology).unwrap();
            let owner = owners(&request, topology);
            let mut per_group = [0usize; 3];
            for group in &owner {
                per_group[*group as usize] += 1;
            }
            assert!(
                per_group[..topology.group_count() as usize]
                    .iter()
                    .all(|&count| count == expected),
                "{:?}",
                &per_group[..topology.group_count() as usize]
            );
            // Every route keeps its true expert id and exact gate weight.
            for (route, &expert) in request.routes.iter().zip(&unique) {
                let word = V41NativeOwnerRouteWord::decode(route.expert_id, topology.group_count()).unwrap();
                assert_eq!(word.expert_id, expert);
                assert_eq!(route.gate_weight, 1.0 / 6.0);
            }
        }
    }

    #[test]
    fn seed_mode_is_wired_and_layer_is_stable_across_request_ids() {
        // Uniform-cost experts make every ordering decision a tie, so only the
        // seed can change the assignment.
        let topology = SparkTopology::new(2, 2).unwrap();
        let experts = [0u32, 7, 19, 88, 200, 383];

        let mut layer = ReplicatedGroupPlanner::new_with_seed_mode(
            topology,
            ReplicatedExpertTieSeedMode::Layer,
        )
        .unwrap();
        assert_eq!(layer.seed_mode(), ReplicatedExpertTieSeedMode::Layer);
        let mut base = request(3, 11, &experts);
        layer.encode(&mut base).unwrap();
        let base_owners = owners(&base, topology);
        for request_id in [12_u64, 99, u64::MAX] {
            let mut repeat = request(3, request_id, &experts);
            layer.encode(&mut repeat).unwrap();
            assert_eq!(
                owners(&repeat, topology),
                base_owners,
                "layer mode must ignore the dispatch request id"
            );
        }

        let mut dispatch = ReplicatedGroupPlanner::new_with_seed_mode(
            topology,
            ReplicatedExpertTieSeedMode::Dispatch,
        )
        .unwrap();
        assert_eq!(dispatch.seed_mode(), ReplicatedExpertTieSeedMode::Dispatch);
        let mut dispatch_base = request(3, 11, &experts);
        dispatch.encode(&mut dispatch_base).unwrap();
        let dispatch_owners = owners(&dispatch_base, topology);
        let mut varied = false;
        for request_id in 2_u64..512 {
            let mut repeat = request(3, request_id, &experts);
            dispatch.encode(&mut repeat).unwrap();
            if owners(&repeat, topology) != dispatch_owners {
                varied = true;
                break;
            }
        }
        assert!(
            varied,
            "dispatch mode must still depend on the request id on tie-heavy routes"
        );
    }

    #[test]
    fn single_group_seed_modes_have_no_effect() {
        // EP1 negative: group_count == 1 makes the seed irrelevant.
        let topology = SparkTopology::new(4, 1).unwrap();
        let experts = [0u32, 7, 19, 88, 200, 383];
        for mode in [
            ReplicatedExpertTieSeedMode::Dispatch,
            ReplicatedExpertTieSeedMode::Layer,
        ] {
            let mut planner = ReplicatedGroupPlanner::new_with_seed_mode(topology, mode).unwrap();
            for request_id in [1_u64, 42, u64::MAX] {
                let mut request = request(3, request_id, &experts);
                planner.encode(&mut request).unwrap();
                assert!(
                    owners(&request, topology).iter().all(|&group| group == 0),
                    "single-group ownership must always be group 0"
                );
            }
        }
    }

    #[test]
    fn duplicate_experts_in_one_row_are_rejected_by_the_protocol() {
        // A real router emits six distinct experts per token. This one-row
        // fixture repeats expert 0 six times: it is an adversarial protocol case,
        // and the canonical admission contract rejects it before ownership is
        // ever encoded. Duplicate expert ids *across* rows are legitimate and are
        // covered by the reuse fixture below.
        let topology = SparkTopology::new(2, 2).unwrap();
        let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
        let mut encoded = request(5, 100, &[0u32; 6]);
        let error = planner.encode(&mut encoded).unwrap_err().to_string();
        assert!(error.contains("duplicate native expert route"), "{error}");
    }

    #[test]
    fn real_row_reuse_keeps_one_group_per_expert_and_masks_weights_exactly() {
        // Production-shaped fixture: each row routes six distinct experts; rows
        // repeat a base set and frequently reuse one hot six-expert subset. Every
        // occurrence of an expert must carry the same owner, and only owned routes
        // may keep the exact FP32 gate weight; every other route is masked to the
        // 384 sentinel with weight zero.
        let topology = SparkTopology::new(2, 2).unwrap();
        let base = [0u32, 1, 2, 3, 4, 5];
        let hot = [200u32, 201, 202, 203, 204, 205];
        let mut routes = Vec::new();
        for row in 0..12usize {
            if row % 4 == 3 { routes.extend_from_slice(&hot); }
            else {
                let offset = row % 6;
                routes.extend((0..6).map(|slot| base[(slot + offset) % 6]));
            }
        }
        let rows = routes.len() / 6;
        let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
        let mut encoded = request(6, 777, &routes);
        planner.encode(&mut encoded).unwrap();
        BackboneRequest::validate_owned_native_group(&encoded, rows as u32, topology).unwrap();
        // Ownership is a pure function of the expert id, stable across rows.
        let mut expert_owner = std::collections::BTreeMap::new();
        for (route, &expert) in encoded.routes.iter().zip(&routes) {
            let word = V41NativeOwnerRouteWord::decode(route.expert_id, topology.group_count()).unwrap();
            assert_eq!(word.expert_id, expert);
            if let Some(previous) = expert_owner.insert(expert, word.owner) {
                assert_eq!(previous, word.owner, "expert {expert} changed group");
            }
        }
        // Duplicate reuse actually happened, and both hot/base experts are active.
        assert_eq!(routes.len(), 72);
        assert!(expert_owner.contains_key(&0) && expert_owner.contains_key(&200));
        // The hot six-expert subset appears in rows 3, 7 and 11.
        assert_eq!(routes.iter().filter(|&&expert| expert == 200).count(), 3);
        // The exact gate weights survive the ownership encode.
        for route in &encoded.routes {
            assert_eq!(route.gate_weight, 1.0 / 6.0);
        }
        // Unpacking each group preserves owned weights and masks the rest to the
        // kernel's unassigned sentinel with a zero weight, exactly.
        let frame = encoded.encode().unwrap();
        for group in 0..topology.group_count() {
            let mut ids = vec![0i32; encoded.routes.len()];
            let mut weights = vec![0f32; encoded.routes.len()];
            let view = BackboneRequest::parse_native_group(
                &frame, rows as u32, topology).unwrap();
            view.copy_native_group_routes_into(&mut ids, &mut weights, group).unwrap();
            for (index, (route, &expert)) in encoded.routes.iter().zip(&routes).enumerate() {
                let word = V41NativeOwnerRouteWord::decode(route.expert_id, topology.group_count()).unwrap();
                if word.owner == group {
                    assert_eq!(ids[index], expert as i32);
                    assert_eq!(weights[index], 1.0 / 6.0);
                } else {
                    assert_eq!(ids[index], cuteafd_transport::expert::V41_NATIVE_UNASSIGNED_EXPERT_ID);
                    assert_eq!(weights[index], 0.0);
                }
            }
        }
    }

    #[test]
    fn an_empty_group_masks_every_route_and_returns_a_zero_plane() {
        // The baseline scheduler balances any batch with at least two active
        // experts, so an empty group is not reachable through whole-expert LPT on
        // a canonical histogram. It is still a real transport state (for example
        // a heavily skewed calibration), so this fixture drives the ownership
        // contract directly: group 0 owns every expert, group 1 owns none, and
        // group 1 must mask all six routes to the sentinel/zero plane.
        let topology = SparkTopology::new(2, 2).unwrap();
        let experts = [10u32, 11, 12, 13, 14, 15];
        let mut encoded = request(5, 101, &experts);
        let mut all_zero_owners = [0u8; 384];
        all_zero_owners[10] = 0; // explicit: group 0 owns the whole batch
        encoded.with_native_group_owners(&all_zero_owners, topology).unwrap();
        let frame = encoded.encode().unwrap();
        BackboneRequest::validate_owned_native_group(&encoded, 1, topology).unwrap();
        for group in 0..topology.group_count() {
            let mut ids = vec![0i32; 6];
            let mut weights = vec![0f32; 6];
            let view = BackboneRequest::parse_native_group(&frame, 1, topology).unwrap();
            view.copy_native_group_routes_into(&mut ids, &mut weights, group).unwrap();
            if group == 0 {
                assert_eq!(ids, experts.map(|expert| expert as i32));
            } else {
                assert_eq!(
                    ids,
                    vec![cuteafd_transport::expert::V41_NATIVE_UNASSIGNED_EXPERT_ID; 6]
                );
                assert!(weights.iter().all(|&weight| weight == 0.0));
            }
        }
    }

    #[test]
    fn planning_is_reproducible_for_a_fixed_layer_and_request_id() {
        // Same (histogram, layer, request_id) must reproduce the same assignment
        // so a graph replay with unchanged input keeps identical ownership.
        let topology = SparkTopology::new(2, 3).unwrap();
        let experts = [3u32, 9, 12, 40, 77, 300];
        let mut replan = || {
            let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
            let mut request = request(9, 4242, &experts);
            planner.encode(&mut request).unwrap();
            owners(&request, topology)
        };
        let first = replan();
        let second = replan();
        assert_eq!(first, second);
        // A distinct request identity is allowed to break ties differently, but
        // must still be a valid exactly-once assignment.
        let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
        let mut other = request(9, 4243, &experts);
        planner.encode(&mut other).unwrap();
        let mut per_group = [0usize; 3];
        for group in owners(&other, topology) {
            per_group[group as usize] += 1;
        }
        assert!(per_group.iter().all(|&count| count == 2));
    }

    #[test]
    fn single_group_topology_owns_every_active_expert() {
        let topology = SparkTopology::new(2, 1).unwrap();
        let mut planner = ReplicatedGroupPlanner::new(topology).unwrap();
        let mut request = request(0, 1, &[1, 2, 3, 4, 5, 6]);
        planner.encode(&mut request).unwrap();
        assert!(owners(&request, topology).iter().all(|&group| group == 0));
    }

    #[test]
    fn wave_reservation_scales_with_the_physical_rank_count() {
        for capacity in [1u32, 16, 80, 4096] {
            let legacy = NativeTp4Wave::device_bytes(capacity).unwrap();
            assert_eq!(legacy, NativeTp4Wave::device_bytes_for(capacity, 4).unwrap());
            assert_eq!(legacy, capacity as usize * 6 * 10240);
            for ranks in 1usize..=8 {
                let expected = capacity as usize * (ranks * 10240 + 20480);
                assert_eq!(NativeTp4Wave::device_bytes_for(capacity, ranks).unwrap(), expected);
                assert!(expected >= capacity as usize * ranks * 10240);
            }
            // Six ranks reserve two more planes than the legacy TP4 forecast.
            assert!(
                NativeTp4Wave::device_bytes_for(capacity, 6).unwrap() > legacy
            );
        }
        for (capacity, ranks) in [(0u32, 6usize), (4097, 6), (80, 0), (80, 9)] {
            assert!(NativeTp4Wave::device_bytes_for(capacity, ranks).is_err(), "{capacity}/{ranks}");
        }
    }

    #[test]
    fn replicated_cost_override_is_validated_or_rejected() {
        assert_eq!(parse_replicated_cost("1,0,16").unwrap(), ReplicatedExpertCostModel::new(1, 0, 16));
        assert_eq!(parse_replicated_cost(" 4 , 2 , 32 ").unwrap(), ReplicatedExpertCostModel::new(4, 2, 32));
        for invalid in ["", "1,0", "1,0,16,2", "1,0,0", "0,0,16", "x,0,16", "1,2"] {
            assert!(parse_replicated_cost(invalid).is_err(), "{invalid:?}");
        }
        // The default profile is the documented whole-expert weight-only model.
        assert_eq!(ReplicatedExpertCostModel::new(1, 0, 16).expert_cost(6).unwrap(), 1);
        assert_eq!(ReplicatedExpertCostModel::new(1, 0, 16).expert_cost(0).unwrap(), 0);
    }
}

#[cfg(test)]
mod upload_tests {
    use super::*;

    #[test]
    fn tp_wave_budget_remains_tp4_upper_bound() -> Result<()> {
        assert!(NativeTp4Wave::device_bytes(0).is_err());
        assert!(NativeTp4Wave::device_bytes(4097).is_err());
        for capacity in [1, 16, 80, 4096] {
            let reserved = NativeTp4Wave::device_bytes(capacity)?;
            assert_eq!(reserved, capacity as usize * 6 * 10240);
            assert!(reserved >= capacity as usize * (2 * V41_PARTIAL_ROW_BYTES as usize + 2 * 10240));
        }
        Ok(())
    }

}
