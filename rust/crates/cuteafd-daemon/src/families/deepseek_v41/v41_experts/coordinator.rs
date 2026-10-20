//! One coordinator wave owns TP route planes through final native reduction.
use super::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{
    replicated_expert_tie_seed_for, ReplicatedExpertCostModel, ReplicatedExpertScheduleConfig,
    ReplicatedExpertScheduler, ReplicatedExpertTieSeedMode, INACTIVE_REPLICATED_EXPERT_GROUP,
};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{
    expert::{SparkTopology, SparkExpertWave, SparkExperts, V41_PARTIAL_ROW_BYTES,
        V41_ROUTED_EXPERTS},
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
    shared: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    ready_rows: Option<u32>,
    local: Option<super::local::LocalExpertWave<'a>>,
    tp2: Option<Box<super::tp2_ffn::Wave<'a>>>,
    /// Present only for an explicit `TP×EP` topology; legacy transports stay on
    /// the canonical request contract.
    native: Option<ReplicatedGroupPlanner>,
    /// The device-driven exchange for verification waves (`CUTEAFD_V41_DEVICE=1`).
    device: Option<DeviceExchange<'a>>,
}

/// Request metadata the proxy thread stamps on this lane's device waves: the
/// pass's placement and its rows in block order (identical for every layer).
#[derive(Default)]
struct DeviceContext {
    placement: u64,
    rows: Vec<cuteafd_transport::ExpertProtocolV2RowDescriptor>,
}

struct DeviceExchange<'a> {
    link: crate::shared::spark_intake::SparkDeviceLink<'a>,
    context: std::sync::Arc<std::sync::Mutex<DeviceContext>>,
    capacity: usize,
}

/// A V4.1 Spark request from a device wave (FP8 wire rows, top-6 routes).
fn device_request(context: &DeviceContext, layer: u32, rows: usize,
    routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, wire: bytes::Bytes) -> Result<ExpertProtocolV2Request> {
    use std::sync::atomic::{AtomicU64, Ordering};
    // Distinct from the router's ids (which count up from 1).
    static NEXT: AtomicU64 = AtomicU64::new(1 << 62);
    ensure!(context.rows.len() == rows, "device wave of {rows} rows for a pass of {}", context.rows.len());
    let mut request = ExpertProtocolV2Request::new_bytes(NEXT.fetch_add(1, Ordering::Relaxed), context.placement,
        layer, 5120, cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32, context.rows.clone(), routes, wire)?;
    request.header.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    cuteafd_transport::expert::BackboneRequest::validate_owned(&request, rows as u32)?;
    Ok(request)
}
impl<'a> NativeTp4Wave<'a> {
    pub(crate) fn spark_world(&self) -> usize { self.link.transport.world_size() }
    /// Explicit replicated topology of this lane, or `None` for the legacy path.
    pub(crate) fn native_topology(&self) -> Option<SparkTopology> {
        self.native.as_ref().map(ReplicatedGroupPlanner::topology)
    }
    /// Apply the lane's whole-expert ownership to one remote request. Legacy
    /// lanes are a no-op; replicated lanes re-encode the route words and set the
    /// native group flag, so every remote path that dispatches through this wave
    /// carries the same contract.
    pub(crate) fn prepare_remote_request(&mut self, request: &mut crate::families::deepseek_v41::v41_backbone_router::BoundExpertRequest) -> Result<()> {
        if let Some(native) = &mut self.native {
            request.assign_native(native)?;
        }
        Ok(())
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
    pub fn reset_connections(&mut self) {
        self.ready_rows = None;
        self.link.transport.reset_connections();
    }
    /// Legacy TP4 upper-bound reservation (four rank planes), also sufficient
    /// for a legacy TP2 transport.
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        Self::device_bytes_for(capacity, 4)
    }
    /// Reservation for exactly `ranks` physical rank planes plus the shared
    /// input and final output planes. Six-rank layouts need their own, larger
    /// reservation; reusing the four-rank reserve undercounts by two planes.
    pub fn device_bytes_for(capacity: u32, ranks: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP wave capacity"
        );
        ensure!(
            matches!(ranks, 2 | 3 | 4 | 6),
            "native TP wave requires two, three, four or six ranks"
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
        // V4.1's receive uploader already supports pinned-host intake.
        let choice = crate::shared::spark_intake::choose_preferred_mode(library, crate::shared::spark_intake::IntakePreference::Pinned)?;
        tracing::info!(mode = choice.mode.name(), reason = %choice.reason,
            "V4.1 Spark intake probe (host lane uses pinned uploads)");
        ensure!(
            matches!(world_size, 2 | 3 | 4 | 6),
            "native TP wave requires two, three, four or six ranks"
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
            .map(ReplicatedGroupPlanner::new)
            .transpose()?;
        let hidden_bytes = capacity as usize * 5120 * 2;
        Ok(Self {
            link: crate::shared::spark_intake::SparkLink::from_transport(library, transport,
                V41_PARTIAL_ROW_BYTES as usize, crate::shared::spark_intake::IntakePreference::Pinned)?,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            shared: DeviceAllocation::new(library, hidden_bytes)?,
            output: DeviceAllocation::new(library, hidden_bytes)?,
            library,
            ready_rows: None,
            local: None,
            tp2: None,
            native,
            device: None,
        })
    }
    /// Connects this lane's device-driven exchange (`capacity` rows per wave,
    /// on the current GPU, which must be the transport's reduction GPU), for
    /// legacy TP4 FP8 transports; `layer` is a Spark layer for the warm-up.
    pub(crate) fn install_device_link(&mut self, peers: &[std::net::SocketAddr], capacity: usize, layer: u32,
        config: cuteafd_transport::TcpTransportConfig) -> Result<()> {
        if !crate::shared::spark_intake::preferred_device_exchange_available(self.library, crate::shared::spark_intake::IntakePreference::Pinned)? {
            return Ok(());
        }
        ensure!(self.native.is_none(),
            "the device exchange serves the legacy TP contract only");
        let spark_world = self.link.transport.world();
        let warm_context = DeviceContext { placement: 0, rows: (0..capacity as u64).map(|row|
            cuteafd_transport::ExpertProtocolV2RowDescriptor { row_id: row,
                source_kind: cuteafd_transport::ExpertV2SourceKind::MtpVerify, source_request_id: 1,
                token_position: row, route_offset: row as u32 * 6, route_count: 6 }).collect() };
        let routes = (0..capacity * 6).map(|i| cuteafd_transport::ExpertProtocolV2RouteEntry {
            row_index: (i / 6) as u32, expert_id: (i % V41_ROUTED_EXPERTS) as u32, gate_weight: 0.0 }).collect();
        let warm = device_request(&warm_context, layer, capacity, routes, vec![0u8; capacity * 5280].into())?;
        let context = std::sync::Arc::new(std::sync::Mutex::new(DeviceContext::default()));
        let shared = std::sync::Arc::clone(&context);
        let build: cuteafd_transport::expert::DeviceBuild = Box::new(move |wave, routes, wire| {
            let context = shared.lock().map_err(|_| anyhow::anyhow!("device context poisoned"))?;
            device_request(&context, wave.layer, wave.rows as usize, routes, wire)
        });
        let device = self.library.cuda_get_device()?;
        let link = crate::shared::spark_intake::SparkDeviceLink::new(self.library, device, spark_world, peers,
            capacity, 6, 5280, V41_PARTIAL_ROW_BYTES as usize,
            cuteafd_transport::TcpTransportConfig { timing: false, ..config }, Some(warm), build)?;
        // The other RTX publishes waves too when the router runs there.
        if self.tp2.is_some() {
            link.initialize_on(1 - device)?;
        }
        self.device = Some(DeviceExchange { link, context, capacity });
        Ok(())
    }
    /// Whether a remote wave of `rows` goes through the device exchange.
    pub(crate) fn device_wave(&self, layer: usize, rows: usize) -> bool {
        self.device.as_ref().is_some_and(|d| rows <= d.capacity)
            && (self.tp2.is_none() || self.has_tp2_shared_layer(layer)) && crate::shared::memory::chain::deferred()
    }
    /// The device exchange's sequences (stuck-step reports).
    pub(crate) fn device_state(&self) -> Option<String> {
        self.device.as_ref().map(|d| d.link.state())
    }
    /// After the pass drained: any device wave's error.
    pub(crate) fn check_device(&self) -> Result<()> {
        self.device.as_ref().map_or(Ok(()), |d| d.link.check())
    }
    /// Publishes a remote layer's wave from the device: the routes and wire
    /// rows leave from a stream of the router's GPU once the chain reaches it.
    ///
    /// # Safety
    /// [`Self::device_wave`] holds; the router output finished into the chain
    /// and stays unchanged until the matching [`Self::device_collect`].
    pub(crate) unsafe fn device_dispatch(&mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        placement: u64, rows: &[crate::families::deepseek_v41::v41_backbone_router::ExpertRow])
        -> Result<crate::families::deepseek_v41::v41_attention_binding::QueryBinding> {
        self.ready_rows = None;
        let binding = routed.validate_request_rows(rows)?;
        ensure!(!routed.request_nvfp4(), "the device exchange carries FP8 wire rows");
        let device = self.device.as_ref().context("device exchange absent")?;
        // Wakes a parked proxy (and surfaces an earlier wave's error).
        device.link.arm()?;
        {
            let mut context = device.context.lock().map_err(|_| anyhow::anyhow!("device context poisoned"))?;
            context.placement = placement;
            context.rows.clear();
            context.rows.extend(rows.iter().enumerate().map(|(i, r)| cuteafd_transport::ExpertProtocolV2RowDescriptor {
                row_id: i as u64, source_kind: r.kind, source_request_id: r.request_id, token_position: r.position,
                route_offset: i as u32 * 6, route_count: 6 }));
        }
        let count = routed.rows as usize;
        let (origin, device_id) = match self.tp2.as_deref() {
            Some(tp2) => { let s = tp2.stream(routed.ids.device_id as usize); (s.raw, s.device) }
            None => (self.stream.raw, crate::shared::memory::device::Device { library: self.library,
                id: self.output.buffer.device_id }),
        };
        ensure!(device_id.id == routed.ids.device_id, "device wave origin GPU differs from the router's");
        device.link.expect(1);
        device_id.run(|| unsafe {
            crate::shared::memory::chain::join(self.library, origin)?;
            device.link.dispatch(routed.layer, count, 0, routed.ids, routed.routing, routed.expert_input, origin)
        })?;
        Ok(binding)
    }
    /// Waits (on the transport GPU's stream) for the dispatched wave's
    /// partials, reduces them with `shared` (complete in the chain) and, for a
    /// block on the other GPU, returns the result there.
    ///
    /// # Safety
    /// Follows this lane's [`Self::device_dispatch`] of `rows` rows.
    pub(crate) async unsafe fn device_collect(&mut self, binding: crate::families::deepseek_v41::v41_attention_binding::QueryBinding,
        shared: CuteafdDeviceBuffer, rows: u32, destination: i32) -> Result<NativeFfnOutput<'_>> {
        let device = self.device.as_ref().context("device exchange absent")?;
        ensure!(shared.device_id == self.output.buffer.device_id && shared.bytes >= rows as usize * 10240,
            "device reduction shared input differs");
        let stream = &self.stream;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * 10240;
        unsafe {
            crate::shared::memory::chain::join(self.library, stream.raw)?;
            device.link.collect(stream.raw)?;
            device.link.reduce(shared.ptr.cast(), output.ptr.cast(), rows as usize, stream.raw)?;
            crate::shared::memory::chain::finish(self.library, stream.raw)?;
        }
        let values = if destination != output.device_id {
            unsafe { self.tp2.as_mut().context("TP2 return workspace missing")?
                .return_result(output, destination as usize, rows).await? }
        } else { output };
        self.ready_rows = Some(rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// A remote layer on the device exchange with the TP2 shared expert: the
    /// wave goes out, both GPUs' shared halves run meanwhile, the transport
    /// GPU waits for the partials and reduces, all queued without a host wait.
    ///
    /// # Safety
    /// As [`Self::execute_tp2_ffn`]; [`Self::device_wave`] holds.
    pub(crate) async unsafe fn execute_device_tp2(&mut self,
        input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        placement: u64, rows: &[crate::families::deepseek_v41::v41_backbone_router::ExpertRow])
        -> Result<NativeFfnOutput<'_>> {
        let binding = unsafe { self.device_dispatch(routed, placement, rows)? };
        ensure!(binding == input.binding() && input.layer == routed.layer, "device TP2 input/router identity mismatch");
        let reduction = self.output.buffer.device_id as usize;
        let tp2 = self.tp2.as_mut().context("TP2 shared workspace missing")?;
        let values = unsafe { tp2.execute_shared_on(input.layer, routed.rows, input.values, reduction).await? };
        unsafe { self.device_collect(binding, values, routed.rows, input.values.device_id).await }
    }
    /// RoCE execution with optional host BF16 shared-expert contribution.
    /// All GPU copies finish before frame storage can be reused, and reduction
    /// finishes before the borrowed output view is exposed. Cancellation leaves
    /// output unavailable until an entirely successful subsequent execution.
    pub async fn execute(
        &mut self,
        request: &ExpertProtocolV2Request,
        shared: Option<&[u8]>,
    ) -> Result<CuteafdDeviceBuffer> {
        self.ready_rows = None;
        self.synchronize()?;
        let rows = request.header.row_count;
        ensure!(
            rows > 0 && rows <= self.link.transport.capacity(),
            "native wave exceeds capacity"
        );
        let hidden_bytes = rows as usize * 5120 * 2;
        if let Some(shared) = shared {
            ensure!(
                shared.len() == hidden_bytes,
                "native shared output has wrong BF16 geometry"
            );
            self.library.copy_h2d(self.shared.buffer, shared)?;
        }
        self.execute_prepared(request, shared.is_some()).await
    }
    /// # Safety
    /// Shared device values are complete and immutable through the copy. The
    /// request and shared result must derive from the same actual block input.
    pub async unsafe fn execute_ffn<'w>(
        &'w mut self,
        request: &crate::families::deepseek_v41::v41_backbone_router::BoundExpertRequest,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>,
    ) -> Result<NativeFfnOutput<'w>> {
        self.ready_rows = None;
        validate_shared(
            request,
            shared,
            self.shared.buffer,
            self.link.transport.capacity(),
        )?;
        unsafe { self.dispatch_ffn(request).await?.finish(shared).await }
    }
    /// Enqueue all TP rank expert requests before returning. The caller can then run
    /// shared FFN work on RTX while the Spark workers execute the routed experts.
    pub async fn dispatch_ffn<'w, 'r>(
        &'w mut self,
        request: &'r crate::families::deepseek_v41::v41_backbone_router::BoundExpertRequest,
    ) -> Result<NativePendingFfn<'w, 'a, 'r>> {
        self.ready_rows = None;
        // Previous output or cancellation cleanup completed this wave.
        self.stream.require_complete()?;
        let header = &request.request().header;
        ensure!(
            header.layer_id as usize == request.binding().layer()
                && header.row_count > 0
                && header.row_count <= self.link.transport.capacity(),
            "native dispatched FFN identity or rows differ"
        );
        let capacity = self.link.transport.capacity();
        let pending = self.link.dispatch(request.request())?;
        Ok(NativePendingFfn {
            pending,
            request,
            capacity,
            library: self.library,
            stream: &self.stream,
            link: &mut self.link,
            shared: self.shared.buffer,
            output: self.output.buffer,
            ready_rows: &mut self.ready_rows,
            tp2: self.tp2.as_deref_mut(),
        })
    }

    async fn execute_prepared(
        &mut self,
        request: &ExpertProtocolV2Request,
        has_shared: bool,
    ) -> Result<CuteafdDeviceBuffer> {
        let rows = request.header.row_count;
        let wave = self.link.dispatch(request)?;
        // SAFETY: this wave owns the intake planes and output until its stream drains.
        unsafe { crate::shared::memory::chain::join(self.library, self.stream.raw)? };
        self.link.receive(wave, rows as usize, self.stream.raw).await?;
        // SAFETY: receive orders the same stream after all partials, and the optional
        // shared plane was initialized by execute before dispatch.
        unsafe {
            self.link.reduce(if has_shared { self.shared.buffer.ptr.cast() } else { std::ptr::null() },
                self.output.buffer.ptr.cast(), rows as usize, self.stream.raw)?;
            crate::shared::memory::chain::finish(self.library, self.stream.raw)?;
        }
        self.ready_rows = Some(rows);
        self.output()
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

fn validate_shared(
    request: &crate::families::deepseek_v41::v41_backbone_router::BoundExpertRequest,
    shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>,
    destination: CuteafdDeviceBuffer,
    capacity: u32,
) -> Result<()> {
    let header = &request.request().header;
    ensure!(
        request.binding() == shared.binding()?
            && header.layer_id as usize == shared.layer
            && header.row_count == shared.rows
            && header.row_count > 0
            && header.row_count <= capacity
            && shared.values.bytes == header.row_count as usize * 10240
            && shared.values.device_id == destination.device_id,
        "native TP shared contribution differs from routed request"
    );
    Ok(())
}
/// Borrows every mutable reduction buffer and owns all unread response sockets.
/// Dropping before completion leaves the wave unpublished and closes the sockets.
pub(crate) struct NativePendingFfn<'w, 'a, 'r> {
    pending: SparkExpertWave,
    request: &'r crate::families::deepseek_v41::v41_backbone_router::BoundExpertRequest,
    capacity: u32,
    library: &'a NativeLibrary,
    stream: &'w LoadStream<'a>,
    link: &'w mut crate::shared::spark_intake::SparkLink<'a>,
    shared: CuteafdDeviceBuffer,
    output: CuteafdDeviceBuffer,
    ready_rows: &'w mut Option<u32>,
    tp2: Option<&'w mut super::tp2_ffn::Wave<'a>>,
}
impl<'w> NativePendingFfn<'w, '_, '_> {
    /// # Safety
    /// Shared values hold the completed contribution for this exact request and
    /// remain immutable until the final reduction drains. Producers must be drained.
    pub async unsafe fn finish(
        self,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>,
    ) -> Result<NativeFfnOutput<'w>> {
        unsafe { self.finish_inner(shared, false).await }
    }
    /// # Safety
    /// Same input retention as finish; cancellation drains before releasing frames.
    pub async unsafe fn finish_cooperative(self,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'w>> {
        unsafe { self.finish_inner(shared, true).await }
    }
    /// Run shared TP2 after Spark dispatch, reduce on the transport GPU, and
    /// return the result to the block's GPU. The pending owner retains every lane workspace.
    /// # Safety
    /// Input is the completed normalized FFN input for the dispatched request;
    /// its storage remains immutable through completion or cancellation drain.
    pub async unsafe fn finish_tp2(mut self, input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>) -> Result<NativeFfnOutput<'w>> {
        let header = &self.request.request().header;
        ensure!(self.request.binding() == input.binding() && header.layer_id as usize == input.layer
            && header.row_count as usize == input.tokens.len()
            && matches!(input.values.device_id, 0 | 1),
            "TP2 shared input/request/device differs");
        let values = unsafe { self.tp2.as_mut().context("TP2 shared workspace missing")?
            .execute_shared_on(input.layer,header.row_count,input.values,self.output.device_id as usize).await? };
        let destination = input.values.device_id;
        if destination != self.output.device_id {
            let device = crate::shared::memory::device::Device { library: self.library, id: self.output.device_id };
            device.future(unsafe { self.finish_values(values,true,Some(destination)) }).await
        } else {
            unsafe { self.finish_values(values,true,Some(destination)).await }
        }
    }
    async unsafe fn finish_inner(self,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>, cooperative: bool) -> Result<NativeFfnOutput<'w>> {
        validate_shared(self.request, shared, self.shared, self.capacity)?;
        unsafe { self.finish_values(shared.values,cooperative,None).await }
    }
    async unsafe fn finish_values(mut self, values: CuteafdDeviceBuffer, cooperative: bool,
        destination: Option<i32>) -> Result<NativeFfnOutput<'w>> {
        ensure!(values.device_id == self.output.device_id
            && values.bytes == self.request.request().header.row_count as usize * 10240,
            "shared reduction device/extent differs");
        let timing = std::time::Instant::now();
        // The shared owner remains borrowed until reduction drains, so consume
        // its completed device output directly instead of copying it first.
        let shared_copy_us = 0u64;
        let rows = self.request.request().header.row_count;
        // SAFETY: shared producers have drained or published their chain dependency;
        // all receive/reduction storage is borrowed by this pending owner.
        unsafe { crate::shared::memory::chain::join(self.library, self.stream.raw)? };
        self.link.receive(self.pending, rows as usize, self.stream.raw).await?;
        let received_us = timing.elapsed().as_micros() as u64;
        // SAFETY: the intake, shared values and output live through the stream fence.
        unsafe { self.link.reduce(values.ptr.cast(), self.output.ptr.cast(), rows as usize, self.stream.raw)? };
        if cooperative {
            // SAFETY: the pending owner retains every input until completion or cancellation.
            unsafe { crate::shared::memory::chain::finish_cooperative(self.stream).await? };
        } else {
            // SAFETY: the same stream owns the reduction and its inputs.
            unsafe { crate::shared::memory::chain::finish(self.library, self.stream.raw)? };
        }
        tracing::debug!(target: "cuteafd::timing", layer=self.request.request().header.layer_id, rows,
            shared_copy_us, receive_us=received_us, reduce_us=timing.elapsed().as_micros() as u64-received_us,
            "target collection");
        *self.ready_rows = Some(rows);
        let mut values = self.output;
        values.bytes = rows as usize * 10240;
        if let Some(device) = destination.filter(|&device| device != values.device_id) {
            values = unsafe { self.tp2.as_mut().context("TP2 return workspace missing")?
                .return_result(values, device as usize, rows).await? };
        }
        Ok(NativeFfnOutput {
            values,
            binding: self.request.binding(),
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
            for ranks in [2usize, 3, 4, 6] {
                let expected = capacity as usize * (ranks * 10240 + 20480);
                assert_eq!(NativeTp4Wave::device_bytes_for(capacity, ranks).unwrap(), expected);
                assert!(expected >= capacity as usize * ranks * 10240);
            }
            // Six ranks reserve two more planes than the legacy TP4 forecast.
            assert!(
                NativeTp4Wave::device_bytes_for(capacity, 6).unwrap() > legacy
            );
        }
        for (capacity, ranks) in [(0u32, 6usize), (4097, 6), (80, 0), (80, 1), (80, 5), (80, 8)] {
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
