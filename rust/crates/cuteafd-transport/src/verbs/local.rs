//! Persistent QP sessions polled directly by a thread-local GPU owner.
use super::*;
use std::sync::atomic::AtomicUsize;

/// Shared byte budget for the mapped RDMA rings pinned by accepted persistent
/// sessions. The limit is the admitted model's ring allowance, so a peer that
/// advertises wire capacities larger than the admission planned fails closed
/// instead of pinning more unified memory than was reserved for it.
///
/// The counter is atomic because the bootstrap thread may admit several
/// connections concurrently; a reservation is charged before the native mapped
/// allocation and released by [`RingReservation`]'s `Drop`.
pub struct RingBudget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
}

impl RingBudget {
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })
    }

    /// Total bytes this budget admits across all live and pending endpoints.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes currently reserved by live or pending endpoints.
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// High-water mark of concurrently reserved bytes.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }

    /// Charge `bytes` to this budget, returning an RAII credit that releases
    /// them on drop. Fails closed when the reservation would exceed the limit or
    /// overflow; the shared counter is updated with a CAS so concurrent
    /// admissions cannot overshoot.
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Result<RingReservation> {
        let mut current = self.used.load(Ordering::Acquire);
        loop {
            let next = current
                .checked_add(bytes)
                .context("mapped RDMA ring budget byte count overflow")?;
            anyhow::ensure!(
                next <= self.limit,
                "mapped RDMA ring budget exceeded: {next} bytes requested in total, limit is {}",
                self.limit
            );
            match self.used.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::AcqRel);
                    return Ok(RingReservation {
                        budget: Arc::clone(self),
                        bytes,
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

/// Pending mapped-ring credit. After native allocation this moves into the
/// endpoint, where confirmed teardown releases it. Failed quiesce retains both
/// registered memory and its credit, including across retained response owners.
pub struct RingReservation {
    budget: Arc<RingBudget>,
    bytes: usize,
}

impl RingReservation {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for RingReservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

pub struct LocalVerbsExpertConnection {
    library: Arc<NativeLibrary>,
    endpoint: NativeRdmaEndpoint,
    start: VerbsHostProtocolV2PersistentStart,
    request_ring: VerbsHostRdmaRing,
    response_ring: VerbsHostRdmaRing,
    request_recv_view: CuteafdRdmaRcEndpointBufferView,
    response_send_view: CuteafdRdmaRcEndpointBufferView,
    response_frame: ExpertProtocolV2FrameBuffer,
    request_recv_sequence: usize,
    response_send_sequence: usize,
    /// Reply frames posted and not yet fully completed (WRs per frame).
    response_sends: request_write::SendQueue,
    response_copy_stream: Option<VerbsHostCudaStream>,
    next_liveness: Instant,
    liveness_polls: u8,
    /// Startup-resolved diagnostics flag; see `protocol_v2_timing_from_env`.
    timing: bool,
    /// Requests received by `begin_deferred` and not yet finished, oldest
    /// first: their receive start times (timing only). Request `k` holds
    /// receive slot `request_recv_sequence + k` and send slot
    /// `response_send_sequence + k`; they finish in this order.
    deferred: std::collections::VecDeque<Option<Instant>>,
    /// Where reply frames above one chunk WRITE their tail (none: write mode,
    /// or every reply fits one chunk).
    reply_target: Option<request_write::ReplyTarget>,
    /// Newest coordinator receives-posted watermark seen on a request.
    reply_watermark: Option<u32>,
    /// Large reply frames sent whole because their receive was not yet
    /// covered by a watermark (RC SEND waits for the receive instead).
    uncovered_replies: u64,
    // EOF acknowledges teardown only after native rings and their credit are released.
    stream: TcpStream,
}
// No operation can race: polling requires &mut self, and all registered views
// remain owned by the endpoint. Like VerbsHostMappedRdmaRing, a session may move
// from its bootstrap thread to the GPU owner, but is never shared concurrently.
unsafe impl Send for LocalVerbsExpertConnection {}

/// The coordinator's last bootstrap message, after it connected its QP and
/// posted its receives: where reply tails land. Required exactly when replies
/// can exceed one chunk and are not write-mode.
fn read_reply_target(reader: &mut BufReader<TcpStream>, start: &VerbsHostProtocolV2PersistentStart,
    response_ring: VerbsHostRdmaRing) -> Result<Option<request_write::ReplyTarget>> {
    let reply: VerbsHostProtocolV2ReplyTarget = decode_persistent_control(read_control_value(reader)?)?;
    anyhow::ensure!(reply.message == "protocol_v2_reply_target", "expected protocol_v2_reply_target, got {}",
        reply.message);
    anyhow::ensure!(reply.target.is_some() == (start.write_target.is_none()
        && request_write::needs_target(start.response_capacity_wire_bytes)),
        "coordinator reply target does not match response capacity {}", start.response_capacity_wire_bytes);
    if let Some(target) = &reply.target {
        target.validate(response_ring.slot_capacity_bytes)?;
        anyhow::ensure!(target.depth == response_ring.depth, "coordinator reply ring depth differs from the start");
    }
    Ok(reply.target)
}

/// Raises the newest coordinator watermark from a received request header
/// (wrapping 32-bit; a request never lowers it).
fn observe_watermark(watermark: &mut Option<u32>, frame: &[u8], flags: u32) -> Result<()> {
    let seen = request_write::read_watermark(frame, flags)?;
    *watermark = Some(match *watermark {
        Some(current) if (seen.wrapping_sub(current) as i32) < 0 => current,
        _ => seen,
    });
    Ok(())
}

/// A reply frame already staged contiguously at a response send slot.
enum Reply {
    /// To the coordinator's posted receive (SEND, or tail WRITEs + head SEND).
    Send { offset: usize, bytes: usize, wr_id: u64 },
    /// Write mode: `bytes` of rows to the client's plane, then `flag`.
    Written { target: VerbsHostWriteTarget, offset: usize, bytes: usize, flag: u64, wr_id: u64 },
}

/// Posts reply frames on the worker QP with WR-level send-queue accounting.
/// Every frame posts its WRs back to back on one QP, the head (SEND or flag
/// WRITE) last, so the coordinator never observes a frame before its tail.
struct ReplyPoster<'a> {
    endpoint: &'a NativeRdmaEndpoint,
    sends: &'a mut request_write::SendQueue,
    target: Option<request_write::ReplyTarget>,
    watermark: Option<u32>,
    capacity: usize,
    uncovered: &'a mut u64,
}
impl ReplyPoster<'_> {
    /// Waits for one send completion and retires it.
    fn reap_one(&mut self) -> Result<()> {
        let stats = self.endpoint.poll_stats(1, 0, default_control_timeout())?;
        self.sends.completed(stats.send_completions as usize)
    }
    /// Frees a response send slot: fewer than `depth - held` frames own one.
    fn reserve_slot(&mut self, depth: usize, held: usize) -> Result<()> {
        while self.sends.requests_in_flight() + held >= depth { self.reap_one()?; }
        Ok(())
    }
    /// Send-queue room for one more WR. Every reply WR is signaled, so each
    /// completion frees exactly one entry.
    fn room(&mut self) -> Result<()> {
        while self.sends.in_flight >= self.endpoint.info.max_send_wr as usize { self.reap_one()?; }
        Ok(())
    }
    /// Posts `reply`, the `sequence`-th frame that consumes a coordinator
    /// receive (write mode never does; its sequence is unused).
    fn post(&mut self, reply: Reply, sequence: u64) -> Result<()> {
        self.sends.posted_request(None)?;
        match reply {
            Reply::Send { offset, bytes, wr_id } => {
                let (parts, chunked) = request_write::reply_parts(self.target.as_ref(), self.watermark, sequence,
                    bytes, self.capacity)?;
                if !chunked && bytes > request_write::CHUNK_BYTES && self.target.is_some() {
                    *self.uncovered += 1;
                    if self.uncovered.is_power_of_two() {
                        tracing::debug!(uncovered = *self.uncovered, sequence, bytes,
                            "reply receive not yet covered by a watermark; sent as one SEND");
                    }
                }
                for part in parts {
                    self.room()?;
                    match part {
                        request_write::ReplyPart::Write { offset: at, bytes: len, remote } => {
                            // SAFETY: the frame is staged at `offset` in this
                            // endpoint's registered send ring and stays unmodified
                            // until its WRs complete (`reserve_slot`); `remote` lies
                            // in the coordinator receive this frame will complete,
                            // which the watermark shows is posted and unread.
                            unsafe { self.endpoint.library.rdma_rc_endpoint_post_request_part(self.endpoint.info.handle,
                                offset, 0, None, at, len, Some(remote), wr_id)?; }
                            self.endpoint.traffic.written(len);
                            self.sends.posted_part(false)?;
                        }
                        request_write::ReplyPart::Send { bytes: len } => {
                            self.endpoint.post_send_at(offset, len, wr_id)?;
                            self.sends.posted_part(true)?;
                        }
                    }
                }
            }
            Reply::Written { target, offset, bytes, flag, wr_id } => {
                anyhow::ensure!(bytes as u64 <= target.plane_bytes, "written response exceeds the client's plane");
                // Rows to the plane in chunks, then the flag alone, last: RC
                // orders it behind every row WRITE on this QP.
                for at in (0..bytes).step_by(request_write::CHUNK_BYTES) {
                    let len = request_write::CHUNK_BYTES.min(bytes - at);
                    self.room()?;
                    // SAFETY: the rows are staged at `offset` in this endpoint's
                    // registered send ring and stay unmodified until their WRs
                    // complete (`reserve_slot`); the plane range was exposed by
                    // the client for this connection (checked against
                    // `plane_bytes` above).
                    unsafe { self.endpoint.library.rdma_rc_endpoint_post_request_part(self.endpoint.info.handle,
                        offset, 0, None, at, len, Some((target.plane_addr + at as u64, target.rkey)), wr_id)?; }
                    self.endpoint.traffic.written(len);
                    self.sends.posted_part(false)?;
                }
                self.room()?;
                self.endpoint.post_write_flagged(&target, offset, 0, flag, wr_id)?;
                self.sends.posted_part(true)?;
            }
        }
        Ok(())
    }
}

impl LocalVerbsExpertConnection {
    /// An opt-in placement worker rejects unsealed or different later lanes
    /// before any mapped ring/native endpoint allocation.
    pub fn accept_with_selection(
        stream: TcpStream, max_frame_bytes: usize, timing: bool,
        budget: Arc<RingBudget>, selection: Option<&crate::worker_selection::InstalledModelAdmission>,
    ) -> Result<Self> {
        Self::accept_with_grouping(stream, max_frame_bytes, timing, budget, selection, None)
    }

    /// Publish the immutable report obtained from this worker's loaded executions.
    pub fn accept_with_grouping(
        stream: TcpStream, max_frame_bytes: usize, timing: bool,
        budget: Arc<RingBudget>, selection: Option<&crate::worker_selection::InstalledModelAdmission>,
        grouping: Option<&crate::worker_selection::InstalledExpertGrouping>,
    ) -> Result<Self> {
        if selection.is_none() { verbs_host_preflight()?; }
        let start = match Self::read_persistent_start(&stream, max_frame_bytes, selection) {
            Ok(start) => start,
            Err(error) => {
                if !error.is::<crate::worker_selection::WorkerSelectionOnly>() && !error.is::<FlowProbesOnly>()
                    && !error.is::<crate::traffic::TrafficProbeOnly>() {
                    let mut stream = stream;
                    let _ = crate::worker_selection::reject_worker_selection(&mut stream, &error);
                }
                return Err(error);
            }
        };
        let endpoint = selection.map(|installed| installed.reserve_endpoint(start.model_endpoint)).transpose()?.flatten();
        if selection.is_some() { verbs_host_preflight()?; }
        // Validate the wire ring geometry first so the reserved byte count is
        // the authoritative span that the native allocation will pin.
        let (request_ring, response_ring) = Self::validated_rings(&start)?;
        let bytes = request_ring
            .registered_span_bytes
            .checked_add(response_ring.registered_span_bytes)
            .context("mapped RDMA ring reservation byte count overflow")?;
        let reservation = match budget.reserve(bytes) {
            Ok(reservation) => reservation,
            Err(error) => {
                let error = error.context(format!(
                    "expert endpoint execution_lane={} request_capacity={} response_capacity={} ring_bytes={} budget_used={} budget_limit={}",
                    start.execution_lane, start.request_capacity_wire_bytes,
                    start.response_capacity_wire_bytes, bytes, budget.used(), budget.limit()));
                let mut stream = stream;
                write_control(&mut stream, &serde_json::json!({
                    "message": "protocol_v2_bootstrap_error", "error": format!("{error:#}")
                }))?;
                return Err(error);
            }
        };
        eprintln!(
            "protocol_v2_verbs_persistent_server_ring_budget execution_lane={} request_capacity={} request_span={} response_capacity={} response_span={} depth={} reserved_bytes={} budget_limit={} budget_used={} budget_peak={}",
            start.execution_lane,
            start.request_capacity_wire_bytes,
            request_ring.registered_span_bytes,
            start.response_capacity_wire_bytes,
            response_ring.registered_span_bytes,
            request_ring.depth,
            bytes,
            budget.limit(),
            budget.used(),
            budget.peak()
        );
        let path =
            verbs_host_native_library_path().context("native RoCE library not configured")?;
        let library = Arc::new(unsafe { NativeLibrary::load(&path) }?);
        Self::initialize_with_rings(stream, library, start, request_ring, response_ring, timing, grouping, Some((reservation, endpoint)))
    }

    /// Negotiated wire ring geometry from the persistent handshake, as
    /// `(request_capacity, request_span, response_capacity, response_span, depth)`.
    /// Read-only and side-effect free; the spans are exactly the bytes this
    /// endpoint pinned and `RingBudget` charged. Used by the large-frame
    /// loopback test and diagnostics so callers do not have to parse logs.
    pub fn negotiated_ring_geometry(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.start.request_capacity_wire_bytes,
            self.start.request_registered_span_bytes,
            self.start.response_capacity_wire_bytes,
            self.start.response_registered_span_bytes,
            self.start.ring_depth,
        )
    }

    fn read_persistent_start(
        stream: &TcpStream,
        max_frame_bytes: usize,
        selection: Option<&crate::worker_selection::InstalledModelAdmission>,
    ) -> Result<VerbsHostProtocolV2PersistentStart> {
        configure_control_stream(stream, default_control_timeout())?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut value = read_control_value(&mut reader)?;
        if value["message"] == "fabric_traffic" {
            crate::traffic::control_phase(&value);
            write_control(&mut stream.try_clone()?, &serde_json::json!({
                "message": "fabric_traffic", "sample": crate::traffic::local()
            }))?;
            return Err(crate::traffic::TrafficProbeOnly.into());
        }
        if flows::is_flow_probe_start(&value) {
            // A coordinator measuring which bond member a flow label reaches:
            // serve its probes, and admit a session only if one follows.
            let library = load_verbs_host_native_library()?;
            let mut stream = stream.try_clone()?;
            value = flows::serve_flow_probes(&mut stream, &mut reader, &library, value)?.ok_or(FlowProbesOnly)?;
        }
        if value["message"] == "worker_selection_start" {
            let installed = selection.context("worker is not placement-handshake enabled")?;
            installed.acknowledge_repeat(&mut stream.try_clone()?, value)?;
            return Err(crate::worker_selection::WorkerSelectionOnly.into());
        }
        let start: VerbsHostProtocolV2PersistentStart = decode_persistent_control(value)?;
        let digest = selection.map(|installed| installed.selection().digest()).transpose()?;
        anyhow::ensure!(start.worker_selection_digest == digest,
            "worker connection does not match its loaded placement");
        anyhow::ensure!(
            start.message == "protocol_v2_persistent_start",
            "native owner requires persistent RoCE bootstrap"
        );
        anyhow::ensure!(
            start.request_capacity_wire_bytes <= max_frame_bytes
                && start.response_capacity_wire_bytes <= max_frame_bytes,
            "peer registration exceeds native frame budget"
        );
        anyhow::ensure!(
            start.ring_depth <= 8,
            "native peer ring depth exceeds admission budget"
        );
        if let Some(installed) = selection {
            if let Some(spec) = installed.endpoint_spec(start.model_endpoint)? {
                let model = installed.selection().model().expect("typed endpoint");
                let rings = spec.resolved_rings(model.rings)
                    .context("invalid admitted endpoint ring depth")?;
                let (request, response) = spec.wire_bytes(installed.selection().geometry())
                    .context("invalid admitted endpoint wire geometry")?;
                let slot = |bytes: usize| bytes.max(rings.slot_bytes).min(rings.max_frame_bytes);
                let expected_request = VerbsHostRdmaRing::new(slot(request), rings.alignment, rings.depth)?;
                let expected_response = VerbsHostRdmaRing::new(slot(response), rings.alignment, rings.depth)?;
                let (request, response) = Self::validated_rings(&start)?;
                anyhow::ensure!(request == expected_request && response == expected_response,
                    "endpoint {:?} ring geometry differs from ModelAdmission", start.model_endpoint);
            }
        } else {
            anyhow::ensure!(start.model_endpoint.is_none(), "unsealed worker cannot admit a typed endpoint");
        }
        Ok(start)
    }

    fn validated_rings(
        start: &VerbsHostProtocolV2PersistentStart,
    ) -> Result<(VerbsHostRdmaRing, VerbsHostRdmaRing)> {
        Self::validated_ring_geometry(
            start.request_capacity_wire_bytes,
            start.request_slot_stride_bytes,
            start.request_registered_span_bytes,
            start.response_capacity_wire_bytes,
            start.response_slot_stride_bytes,
            start.response_registered_span_bytes,
            start.ring_depth,
        )
    }

    /// Validate the peer's advertised ring geometry with the transport's own
    /// `from_wire` rules. This runs before any budget charge or native mapped
    /// allocation, so a malformed advertisement is rejected up front.
    fn validated_ring_geometry(
        request_capacity_wire_bytes: usize,
        request_slot_stride_bytes: usize,
        request_registered_span_bytes: usize,
        response_capacity_wire_bytes: usize,
        response_slot_stride_bytes: usize,
        response_registered_span_bytes: usize,
        ring_depth: usize,
    ) -> Result<(VerbsHostRdmaRing, VerbsHostRdmaRing)> {
        let request_ring = VerbsHostRdmaRing::from_wire(
            request_capacity_wire_bytes,
            request_slot_stride_bytes,
            ring_depth,
            request_registered_span_bytes,
        )?;
        let response_ring = VerbsHostRdmaRing::from_wire(
            response_capacity_wire_bytes,
            response_slot_stride_bytes,
            ring_depth,
            response_registered_span_bytes,
        )?;
        Ok((request_ring, response_ring))
    }

    pub(super) fn initialize(
        stream: TcpStream,
        library: Arc<NativeLibrary>,
        start: VerbsHostProtocolV2PersistentStart,
        timing: bool,
    ) -> Result<Self> {
        let (request_ring, response_ring) = Self::validated_rings(&start)?;
        Self::initialize_with_rings(stream, library, start, request_ring, response_ring, timing, None, None)
    }

    fn initialize_with_rings(
        mut stream: TcpStream,
        library: Arc<NativeLibrary>,
        start: VerbsHostProtocolV2PersistentStart,
        request_ring: VerbsHostRdmaRing,
        response_ring: VerbsHostRdmaRing,
        timing: bool,
        grouping: Option<&crate::worker_selection::InstalledExpertGrouping>,
        pending: Option<(RingReservation, Option<crate::worker_selection::EndpointReservation>)>,
    ) -> Result<Self> {
        // Allocate the per-QP prefix only after selection budget admission and
        // before native rings. Bulk payloads stay in the mapped send ring.
        let response_frame = match pending.as_ref().and_then(|(_, endpoint)| endpoint.as_ref()) {
            Some(endpoint) => ExpertProtocolV2FrameBuffer::for_endpoint_response(endpoint.contract().0)?,
            None => ExpertProtocolV2FrameBuffer::new(),
        };
        let rdma_device = verbs_host_rdma_device_for_stream(&stream)?;
        let mut endpoint = NativeRdmaEndpoint::create_from_wire_bytes_mapped_on_device(
            Arc::clone(&library),
            "server",
            start.request_capacity_wire_bytes,
            start.response_capacity_wire_bytes,
            start.request_registered_span_bytes,
            start.response_registered_span_bytes,
            next_local_psn("server"),
            rdma_device.as_ref(),
        )?;
        if let Some((rings, admitted)) = pending {
            endpoint.credits = EndpointCredits { rings: Some(rings), endpoint: admitted };
        }
        let request_recv_view = endpoint.recv_buffer_view()?;
        let response_send_view = endpoint.send_buffer_view()?;
        validate_mapped_endpoint_buffer_view(
            request_recv_view,
            request_ring,
            "persistent receive",
        )?;
        validate_mapped_endpoint_buffer_view(response_send_view, response_ring, "persistent send")?;
        let (_client_host, server_host) = distinct_endpoint_hosts(
            start.client_endpoint.host.clone(),
            local_control_host("server"),
        );
        let server_endpoint = endpoint.verbs_descriptor("server", &server_host);
        validate_persistent_endpoint_capacity(
            &start.client_endpoint,
            "client",
            start.request_capacity_wire_bytes,
            start.response_capacity_wire_bytes,
            start.request_registered_span_bytes,
            start.response_registered_span_bytes,
            request_ring.depth,
        )?;
        validate_persistent_endpoint_capacity(
            &server_endpoint,
            "server",
            start.response_capacity_wire_bytes,
            start.request_capacity_wire_bytes,
            start.response_registered_span_bytes,
            start.request_registered_span_bytes,
            response_ring.depth,
        )?;
        endpoint.connect_with_flow_label(&start.client_native_endpoint, start.flow_label)?;
        // Frames above one chunk arrive as WRITE tails into the request ring.
        let request_target = if request_write::needs_target(start.request_capacity_wire_bytes) {
            endpoint.traffic.receives_writes();
            let (address, rkey) = library.rdma_rc_endpoint_request_target(endpoint.info.handle)?;
            Some(request_write::Target { address, rkey })
        } else { None };
        for slot in 0..request_ring.depth {
            endpoint.post_recv_at(
                request_ring.slot_offset(slot),
                request_ring.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + slot as u64,
            )?;
        }
        if timing {
            let server_native_endpoint = endpoint.native_descriptor();
            eprintln!(
            "protocol_v2_verbs_persistent_server_connect ring_depth={} request_capacity={} request_stride={} request_span={} response_capacity={} response_stride={} response_span={} server_device={} server_gid={} server_status=\"{}\" client_device={} client_gid={} client_status=\"{}\"",
            request_ring.depth,
            start.request_capacity_wire_bytes,
            request_ring.slot_stride_bytes,
            request_ring.registered_span_bytes,
            start.response_capacity_wire_bytes,
            response_ring.slot_stride_bytes,
            response_ring.registered_span_bytes,
            server_native_endpoint.device_name,
            server_native_endpoint.gid_hex,
            server_native_endpoint.status,
            start.client_native_endpoint.device_name,
            start.client_native_endpoint.gid_hex,
            start.client_native_endpoint.status
        );
        }
        write_control(
            &mut stream,
            &VerbsHostProtocolV2PersistentReady {
                message: "protocol_v2_persistent_ready".to_owned(),
                wire_version: PERSISTENT_CONTROL_VERSION,
                expert_grouping: grouping.cloned(),
                server_endpoint,
                server_native_endpoint: endpoint.native_descriptor(),
                request_target,
                flow_label: start.flow_label,
            },
        )?;
        let reply = read_reply_target(&mut BufReader::new(stream.try_clone()?), &start, response_ring)?;

        Ok(Self {
            stream,
            library,
            endpoint,
            start,
            request_ring,
            response_ring,
            request_recv_view,
            response_send_view,
            response_frame,
            request_recv_sequence: 0,
            response_send_sequence: 0,
            response_sends: Default::default(),
            response_copy_stream: None,
            next_liveness: Instant::now(),
            liveness_polls: 0,
            deferred: std::collections::VecDeque::new(),
            reply_target: reply,
            reply_watermark: None,
            uncovered_replies: 0,
            timing,
        })
    }

    /// Execute at most one request. The callback must finish all GPU access to
    /// request/response slots before returning, including on errors. Idle returns
    /// immediately so other QPs can progress on the same GPU owner. With
    /// `wait = Some(timeout)` and no completion already pending, the call first
    /// blocks on the endpoint's completion channel (armed with
    /// `ibv_req_notify_cq`) for up to `timeout`, so an idle peer is woken by the
    /// QP event rather than polled. `timeout` is the whole wait (`wait_stats`):
    /// it is not raised to the 30 s floor of an exchange in flight, because the
    /// owner has other connections to sweep when it ends.
    pub fn poll<F>(&mut self, wait: Option<Duration>, execute: F) -> Result<bool>
    where
        F: FnMut(
            &ExpertProtocolV2RequestView<'_>,
            ProtocolV2RequestDevicePayload,
            &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
        ) -> Result<()>,
    {
        // Clock/peek checks are amortized while the owner spins on completions.
        if self.liveness_polls == 0 || wait.is_some() {
            check_peer_liveness(&self.stream, &mut self.next_liveness)?;
        }
        self.liveness_polls = self.liveness_polls.wrapping_add(1);
        let timing_enabled = self.timing;
        let total_started = timing_enabled.then(Instant::now);
        let poll_recv_started = timing_enabled.then(Instant::now);
        let stats = match wait {
            Some(timeout) => match self.endpoint.wait_stats(0, 1, timeout) {
                Ok(stats) => stats,
                // The event wait reports a window with no completion as an RDMA
                // error; on an idle connection that is the expected outcome and
                // not a peer failure.
                Err(error) if is_verbs_host_rdma_poll_timeout(&error) => {
                    CuteafdRdmaRcCompletionStats::default()
                }
                Err(error) => return Err(error),
            },
            None => self.endpoint.try_poll(0, 1)?,
        };
        if stats.recv_completions == 0 {
            if wait.is_some() {
                check_peer_liveness(&self.stream, &mut self.next_liveness)?;
            }
            return Ok(false);
        }
        self.serve_received(poll_recv_started, total_started, execute).map(|()| true)
    }

    /// Serve the request whose receive completion was just consumed (or was
    /// held by [`Self::begin_deferred`]): execute, emit, then repost its slot.
    fn serve_received<F>(&mut self, poll_recv_started: Option<Instant>, total_started: Option<Instant>,
        mut execute: F) -> Result<()>
    where
        F: FnMut(
            &ExpertProtocolV2RequestView<'_>,
            ProtocolV2RequestDevicePayload,
            &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
        ) -> Result<()>,
    {
        let timing_enabled = self.timing;
        // Execution can take much longer than a poll; check promptly when it returns.
        self.liveness_polls = 0;
        let endpoint = &self.endpoint;
        let library = &self.library;
        let start = &self.start;
        let request_ring = self.request_ring;
        let response_ring = self.response_ring;
        let request_recv_view = self.request_recv_view;
        let response_send_view = self.response_send_view;
        let response_frame = &mut self.response_frame;
        let response_copy_stream = &mut self.response_copy_stream;
        let write_target = self.start.write_target;
        let mut request_recv_sequence = self.request_recv_sequence;
        let mut response_send_sequence = self.response_send_sequence;
        let mut poster = ReplyPoster { endpoint, sends: &mut self.response_sends, target: self.reply_target,
            watermark: self.reply_watermark, capacity: start.response_capacity_wire_bytes,
            uncovered: &mut self.uncovered_replies };
        let reply_watermark = &mut self.reply_watermark;
        let result = (|| -> Result<()> {
            let poll_recv_ms = elapsed_ms_optional(poll_recv_started);
            let request_recv_slot = mapped_ring_slot(
                request_recv_view,
                request_ring,
                request_recv_sequence as u64,
            )?;
            let request_storage = unsafe {
                std::slice::from_raw_parts(
                    request_recv_slot.host_ptr.cast_const(),
                    request_recv_slot.capacity_bytes,
                )
            };
            let copy_header_ms = 0.0_f64;
            let request_wire_bytes = persistent_protocol_v2_request_wire_bytes_from_header(
                &request_storage[..EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN],
                start.request_capacity_wire_bytes,
            )?;
            let resize_ms = 0.0_f64;
            let copy_request_ms = 0.0_f64;
            // The head SEND only: the client counts its WRITE tail as written
            // bytes, which the fabric report adds for a `receives_writes` QP.
            endpoint.traffic.received(request_wire_bytes.min(request_write::CHUNK_BYTES));
            let mut post_recv_ms = 0.0_f64;
            let parse_started = timing_enabled.then(Instant::now);
            let request =
                ExpertProtocolV2RequestView::parse(&request_storage[..request_wire_bytes])
                    .context("parsing persistent verbs-host ProtocolV2 request frame")?;
            if let Some(admitted) = &endpoint.credits.endpoint {
                admitted.validate_request(&request.header)?;
            }
            observe_watermark(reply_watermark, &request_storage[..request_wire_bytes], request.header.flags)?;
            poster.watermark = *reply_watermark;
            let parse_ms = elapsed_ms_optional(parse_started);
            let request_id = request.header.request_id;
            let layer_id = request.header.layer_id;
            let row_count = request.header.row_count;
            let route_count = request.header.route_count;
            if timing_enabled {
                eprintln!(
                "protocol_v2_verbs_persistent_server_received execution_lane={} request_id={} layer_id={} rows={} routes={} request_wire_bytes={} response_capacity={} poll_recv_ms={:.3} copy_header_ms={:.3} resize_ms={:.3} copy_request_ms={:.3} post_recv_ms={:.3} parse_ms={:.3}",
                start.execution_lane,
                request_id,
                layer_id,
                row_count,
                route_count,
                request_wire_bytes,
                start.response_capacity_wire_bytes,
                poll_recv_ms,
                copy_header_ms,
                resize_ms,
                copy_request_ms,
                post_recv_ms,
                parse_ms
            );
            }
            let mut response_frames = 0_usize;
            let mut response_wire_bytes = 0_usize;
            let mut encode_ms = 0.0_f64;
            let mut send_ms = 0.0_f64;
            let mut poll_send_ms = 0.0_f64;
            let mut direct_device_response_frames = 0_usize;
            let mut final_response_emitted = false;
            let poll_send_started = timing_enabled.then(Instant::now);
            poster.reserve_slot(response_ring.depth, 0)?;
            poll_send_ms += elapsed_ms_optional(poll_send_started);
            let first_response_slot = mapped_ring_slot(
                response_send_view,
                response_ring,
                response_send_sequence as u64,
            )?;
            let device_payload = protocol_v2_request_device_payload(
                &request_storage[..request_wire_bytes],
                &request,
                request_recv_slot.device_buffer,
                Some(first_response_slot.device_buffer),
                start.execution_lane,
            )?;
            let execute_started = timing_enabled.then(Instant::now);
            // The final reply is the client's credit to WRITE into this request
            // slot again, so it is posted only after execute has returned and
            // the slot's receive is reposted (below).
            let mut deferred_final: Option<(Reply, u64)> = None;
            execute(&request, device_payload, &mut |response| {
                if final_response_emitted {
                    bail!(
                        "persistent verbs-host executor emitted a response after the final chunk"
                    );
                }
                let poll_send_started = timing_enabled.then(Instant::now);
                poster.reserve_slot(response_ring.depth, 0)?;
                poll_send_ms += elapsed_ms_optional(poll_send_started);
                let reply_sequence = response_send_sequence as u64;
                let response_has_more = response.more_chunks();
                let response_send_slot = response_send_sequence % response_ring.depth;
                let response_send_offset = response_ring.slot_offset(response_send_sequence);
                let encode_started = timing_enabled.then(Instant::now);
                let response_wire_bytes_for_chunk = match response {
                    ProtocolV2ExecutorResponseRef::Host(response) if write_target.is_some() => {
                        // Write mode: the rows go from the send slot to the client's plane.
                        let target = write_target.expect("write target checked above");
                        let bytes = response.partial_output_payload.len();
                        let slot = mapped_ring_slot(response_send_view, response_ring, response_send_sequence as u64)?;
                        anyhow::ensure!(bytes <= slot.capacity_bytes, "written response exceeds its send slot");
                        // SAFETY: the slot lies in this endpoint's mapped send ring and
                        // its previous send completed (the in-flight check above).
                        unsafe { std::ptr::copy_nonoverlapping(response.partial_output_payload.as_ptr(), slot.host_ptr,
                            bytes); }
                        let flag = written_response_flag(&response.header, response_has_more || response.row_indices.is_some());
                        let send_started = timing_enabled.then(Instant::now);
                        let reply = Reply::Written { target, offset: response_send_offset,
                            bytes: if flag & VERBS_HOST_WRITE_FLAG_ERROR != 0 { 0 } else { bytes }, flag,
                            wr_id: VERBS_HOST_SEND_WR_ID + response_send_slot as u64 };
                        if response_has_more { poster.post(reply, reply_sequence)? }
                        else { deferred_final = Some((reply, reply_sequence)) }
                        send_ms += elapsed_ms_optional(send_started);
                        bytes
                    }
                    ProtocolV2ExecutorResponseRef::Host(response) => {
                        let response_prefix =
                            response_frame.encode_borrowed_response_prefix(&response)?;
                        let wire_bytes = response_prefix
                            .len()
                            .checked_add(response.partial_output_payload.len())
                            .context("persistent verbs-host response byte count overflow")?;
                        encode_ms += elapsed_ms_optional(encode_started);
                        anyhow::ensure!(
                            wire_bytes <= start.response_capacity_wire_bytes,
                            "persistent verbs-host ProtocolV2 executor response frame bytes {wire_bytes} exceeded response capacity {}",
                            start.response_capacity_wire_bytes
                        );
                        let send_started = timing_enabled.then(Instant::now);
                        // Staged in the send slot (the borrowed payload ends with
                        // this callback); a final frame is posted once the
                        // request slot is reposted.
                        let slot = mapped_ring_slot(response_send_view, response_ring, response_send_sequence as u64)?;
                        anyhow::ensure!(wire_bytes <= slot.capacity_bytes, "response exceeds its send slot");
                        // SAFETY: the range lies in this endpoint's mapped send
                        // ring (capacity checked above), and every WR of the frame
                        // that last used this slot completed (`reserve_slot`).
                        let dst = unsafe { std::slice::from_raw_parts_mut(slot.host_ptr, wire_bytes) };
                        dst[..response_prefix.len()].copy_from_slice(response_prefix);
                        dst[response_prefix.len()..].copy_from_slice(response.partial_output_payload);
                        let reply = Reply::Send { offset: response_send_offset, bytes: wire_bytes,
                            wr_id: VERBS_HOST_SEND_WR_ID + response_send_slot as u64 };
                        if response_has_more { poster.post(reply, reply_sequence)? }
                        else { deferred_final = Some((reply, reply_sequence)) }
                        send_ms += elapsed_ms_optional(send_started);
                        wire_bytes
                    }
                    ProtocolV2ExecutorResponseRef::Device(response) => {
                        let response_prefix =
                            response_frame.encode_device_response_prefix(&response)?;
                        let wire_bytes = response_prefix
                            .len()
                            .checked_add(response.partial_output_payload.bytes)
                            .context("persistent verbs-host device response byte count overflow")?;
                        encode_ms += elapsed_ms_optional(encode_started);
                        anyhow::ensure!(
                            wire_bytes <= start.response_capacity_wire_bytes,
                            "persistent verbs-host ProtocolV2 executor device response frame bytes {wire_bytes} exceeded response capacity {}",
                            start.response_capacity_wire_bytes
                        );
                        let slot = mapped_ring_slot(
                            response_send_view,
                            response_ring,
                            response_send_sequence as u64,
                        )?;
                        let send_started = timing_enabled.then(Instant::now);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                response_prefix.as_ptr(),
                                slot.host_ptr,
                                response_prefix.len(),
                            );
                        }
                        let destination = protocol_v2_device_buffer_slice(
                            slot.device_buffer,
                            response_prefix.len(),
                            response.partial_output_payload.bytes,
                            "persistent verbs-host response payload",
                        )?;
                        if device_buffers_match(destination, response.partial_output_payload) {
                            direct_device_response_frames += 1;
                        } else {
                            if response_copy_stream.is_none() {
                                *response_copy_stream =
                                    Some(VerbsHostCudaStream::create(Arc::clone(&library))?);
                            }
                            let copy_stream = response_copy_stream
                                .as_ref()
                                .expect("response copy stream was created above")
                                .raw;
                            unsafe {
                                library.copy_d2d_async(
                                    destination,
                                    response.partial_output_payload,
                                    response.partial_output_payload.bytes,
                                    copy_stream,
                                )?;
                                library.cuda_stream_synchronize(copy_stream)?;
                            }
                        }
                        let wr_id = VERBS_HOST_SEND_WR_ID + response_send_slot as u64;
                        let reply = match write_target {
                            Some(target) => {
                                // Write mode: rows to the client's plane, then the flag.
                                let flag = written_response_flag(&response.header, response_has_more);
                                Reply::Written { target, offset: response_send_offset + response_prefix.len(),
                                    bytes: if flag & VERBS_HOST_WRITE_FLAG_ERROR != 0 { 0 } else {
                                        response.partial_output_payload.bytes }, flag, wr_id }
                            }
                            None => Reply::Send { offset: response_send_offset, bytes: wire_bytes, wr_id },
                        };
                        if response_has_more { poster.post(reply, reply_sequence)? }
                        else { deferred_final = Some((reply, reply_sequence)) }
                        send_ms += elapsed_ms_optional(send_started);
                        wire_bytes
                    }
                };
                // One send slot per frame; in SEND mode it is also the coordinator
                // receive the frame completes (write mode posts no receives and
                // has no reply target, so its sequence is never a landing).
                response_send_sequence = response_send_sequence.wrapping_add(1);
                response_frames += 1;
                response_wire_bytes = response_wire_bytes
                    .checked_add(response_wire_bytes_for_chunk)
                    .context("persistent verbs-host response wire byte count overflow")?;
                final_response_emitted = !response_has_more;
                Ok(())
            })?;
            let execute_ms = elapsed_ms_optional(execute_started);
            if response_frames == 0 || !final_response_emitted {
                bail!("persistent verbs-host executor did not emit a final response chunk");
            }
            let post_recv_started = timing_enabled.then(Instant::now);
            // execute has returned, so every host and GPU consumer of the request
            // slot is done: repost it, then grant it back with the final reply.
            endpoint.post_recv_at(request_ring.slot_offset(request_recv_sequence), request_ring.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + request_recv_slot.slot_index as u64)?;
            let (reply, sequence) = deferred_final.context("persistent verbs-host final response was not staged")?;
            poster.post(reply, sequence)?;
            request_recv_sequence = request_recv_sequence.wrapping_add(1);
            post_recv_ms = elapsed_ms_optional(post_recv_started);
            if timing_enabled {
                eprintln!(
                "protocol_v2_verbs_persistent_server_roundtrip_timing execution_lane={} request_id={} layer_id={} rows={} routes={} request_wire_bytes={} response_frames={} direct_device_response_frames={} response_wire_bytes={} executor={} poll_recv_ms={:.3} copy_header_ms={:.3} resize_ms={:.3} copy_request_ms={:.3} post_recv_ms={:.3} parse_ms={:.3} execute_ms={:.3} encode_ms={:.3} send_ms={:.3} poll_send_ms={:.3} total_ms={:.3} end_epoch_us={}",
                start.execution_lane,
                request_id,
                layer_id,
                row_count,
                route_count,
                request_wire_bytes,
                response_frames,
                direct_device_response_frames,
                response_wire_bytes,
                "local-owner",
                poll_recv_ms,
                copy_header_ms,
                resize_ms,
                copy_request_ms,
                post_recv_ms,
                parse_ms,
                execute_ms,
                encode_ms,
                send_ms,
                poll_send_ms,
                elapsed_ms_optional(total_started),
                // Wall-clock end, so per-rank busy intervals and idle gaps can be
                // rebuilt across requests: [end - (total - poll_recv), end].
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_micros())
            );
            }
            Ok(())
        })();
        self.request_recv_sequence = request_recv_sequence;
        self.response_send_sequence = response_send_sequence;
        result
    }

    /// Requests launched by [`Self::begin_deferred`] that await
    /// [`Self::finish_deferred`]; their request and response slots stay held.
    pub fn deferred(&self) -> usize { self.deferred.len() }

    /// Receive at most one request and hand it to `launch`, which may leave GPU
    /// work queued on the mapped request and response slots. The request slot
    /// is not reposted and no response is sent until [`Self::finish_deferred`]
    /// finishes it, so both slots stay owned by the queued work. Up to `limit`
    /// requests may be deferred on one connection (bounded by its rings);
    /// at the limit nothing is received.
    pub fn begin_deferred<T, F>(&mut self, wait: Option<Duration>, limit: usize, launch: F) -> Result<Option<T>>
    where
        F: FnOnce(&ExpertProtocolV2RequestView<'_>, ProtocolV2RequestDevicePayload) -> Result<T>,
    {
        let held = self.deferred.len();
        if !deferred_admits(held, limit, self.request_ring.depth, self.response_ring.depth) { return Ok(None); }
        if self.liveness_polls == 0 || wait.is_some() {
            check_peer_liveness(&self.stream, &mut self.next_liveness)?;
        }
        self.liveness_polls = self.liveness_polls.wrapping_add(1);
        let started = self.timing.then(Instant::now);
        let stats = match wait {
            Some(timeout) => match self.endpoint.wait_stats(0, 1, timeout) {
                Ok(stats) => stats,
                Err(error) if is_verbs_host_rdma_poll_timeout(&error) => CuteafdRdmaRcCompletionStats::default(),
                Err(error) => return Err(error),
            },
            None => self.endpoint.try_poll(0, 1)?,
        };
        if stats.recv_completions == 0 {
            if wait.is_some() { check_peer_liveness(&self.stream, &mut self.next_liveness)?; }
            return Ok(None);
        }
        self.liveness_polls = 0;
        // The completion is consumed: from here the request is owned by this
        // connection until finish_deferred reposts it, even if launch fails.
        self.deferred.push_back(started);
        let request_slot = mapped_ring_slot(self.request_recv_view, self.request_ring,
            (self.request_recv_sequence + held) as u64)?;
        // SAFETY: the received slot lies inside this endpoint's mapped receive
        // ring and is not reposted until finish_deferred.
        let storage = unsafe { std::slice::from_raw_parts(request_slot.host_ptr.cast_const(),
            request_slot.capacity_bytes) };
        let wire_bytes = persistent_protocol_v2_request_wire_bytes_from_header(
            &storage[..EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN], self.start.request_capacity_wire_bytes)?;
        let request = ExpertProtocolV2RequestView::parse(&storage[..wire_bytes])
            .context("parsing deferred verbs-host ProtocolV2 request frame")?;
        if let Some(admitted) = &self.endpoint.credits.endpoint { admitted.validate_request(&request.header)?; }
        // The response goes into send slot `response_send_sequence + held`;
        // drain send completions until that slot (and every earlier deferred
        // request's) is free, exactly as the emit path does for one.
        observe_watermark(&mut self.reply_watermark, &storage[..wire_bytes], request.header.flags)?;
        ReplyPoster { endpoint: &self.endpoint, sends: &mut self.response_sends, target: self.reply_target,
            watermark: self.reply_watermark, capacity: self.start.response_capacity_wire_bytes,
            uncovered: &mut self.uncovered_replies }.reserve_slot(self.response_ring.depth, held)?;
        let response_slot = mapped_ring_slot(self.response_send_view, self.response_ring,
            (self.response_send_sequence + held) as u64)?;
        let payload = protocol_v2_request_device_payload(&storage[..wire_bytes], &request,
            request_slot.device_buffer, Some(response_slot.device_buffer), self.start.execution_lane)?;
        launch(&request, payload).map(Some)
    }

    /// Complete the oldest request [`Self::begin_deferred`] launched:
    /// `complete` receives the same request and payload, waits for nothing it
    /// did not queue (callers poll their own completion first), and emits the
    /// response; the request slot is then reposted.
    pub fn finish_deferred<F>(&mut self, mut complete: F) -> Result<()>
    where
        F: FnMut(
            &ExpertProtocolV2RequestView<'_>,
            ProtocolV2RequestDevicePayload,
            &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
        ) -> Result<()>,
    {
        let started = self.deferred.pop_front().context("no deferred request to finish")?;
        self.serve_received(started, started, &mut complete)
    }
}

/// Whether a connection holding `held` deferred requests may receive another:
/// each holds one receive and one send slot until it finishes.
fn deferred_admits(held: usize, limit: usize, request_depth: usize, response_depth: usize) -> bool {
    held < limit.min(request_depth).min(response_depth)
}


/// A final Error must never publish a successful flag for an untouched plane.
fn written_response_flag(header: &ExpertProtocolV2ResponseHeader, unwritable: bool) -> u64 {
    let failed = unwritable || header.status == crate::protocol_v2::ExpertProtocolV2Status::Error;
    (header.request_id & !VERBS_HOST_WRITE_FLAG_ERROR)
        | if failed { VERBS_HOST_WRITE_FLAG_ERROR } else { 0 }
}

fn check_peer_liveness(stream: &TcpStream, next: &mut Instant) -> Result<()> {
    let now = Instant::now();
    if now >= *next {
        ensure_peer_open(stream)?;
        *next = now + Duration::from_millis(1);
    }
    Ok(())
}

fn ensure_peer_open(stream: &TcpStream) -> Result<()> {
    anyhow::ensure!(!verbs_host_control_plane_closed(stream)?, "native RoCE peer closed");
    Ok(())
}

#[cfg(test)]
mod deferred_tests {
    use super::deferred_admits;
    use crate::verbs::request_write::SendQueue;

    /// N deferred requests per connection, bounded by the slot count and both
    /// rings; each request's send slot is reserved before the next is received.
    #[test]
    fn deferred_requests_are_bounded_by_slots_and_rings() {
        assert!(deferred_admits(0, 1, 8, 8));
        assert!(!deferred_admits(1, 1, 8, 8));
        assert!(deferred_admits(2, 3, 8, 8));
        assert!(!deferred_admits(3, 3, 8, 8));
        // A depth-one decode ring holds one request in flight whatever the pool size.
        assert!(!deferred_admits(1, 4, 1, 1));
        // Send slots count reply frames, not WRs: a chunked frame of many WRs
        // owns one slot until its head is posted and every WR completed.
        let mut sends = SendQueue::default();
        for parts in [101, 1] {
            sends.posted_request(None).unwrap();
            for part in 0..parts { sends.posted_part(part + 1 == parts).unwrap(); }
        }
        let free = |sends: &SendQueue, held: usize, depth: usize| sends.requests_in_flight() + held < depth;
        assert!(free(&sends, 1, 4) && !free(&sends, 2, 4));
        sends.completed(100).unwrap();
        assert!(!free(&sends, 2, 4), "a chunked frame freed its slot before its head completed");
        sends.completed(1).unwrap();
        assert!(free(&sends, 2, 4));
        assert!(!free(&SendQueue::default(), 1, 1));
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    fn typed_admission() -> Result<crate::worker_selection::InstalledModelAdmission> {
        typed_admission_with_depths(&[None])
    }

    fn typed_admission_with_depths(depths: &[Option<usize>]) -> Result<crate::worker_selection::InstalledModelAdmission> {
        use crate::worker_selection::{InstalledModelAdmission, ModelAdmission};
        use cuteafd_core::{ModelAdmissionResponse, RuntimeSample, SparkEndpoint, SparkIngress,
            SparkRings, SparkRuntimeInventory, SparkSlice};
        let selection = placement_selection(&[13]).with_model(ModelAdmission {
            capacity: 80, compiled_capacity: 80,
            endpoints: depths.iter().map(|&ring_depth| SparkEndpoint {
                capacity: 80, ingress: SparkIngress::Bf16, ring_depth }).collect(),
            rings: SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 }, execution_slots: 1
        })?;
        let report = ModelAdmissionResponse {
            rank: 0, resident_bytes: 1,
            slices: vec![SparkSlice { unit_map: None, layer: 13, first: 0, width: 1024, stored_width: 1024,
                tiers: vec![], projection_counts: vec![] }],
            runtime: SparkRuntimeInventory { detail: Default::default(), sample: RuntimeSample {
                device: 0, total_bytes: 121 << 30, free_bytes: 100 << 30, context_bytes: 1,
                module_bytes: 1, sms: 48, arch: "sm_121".into(), driver: "test".into(),
            }, workspace_bytes: 1, pinned_workspace_bytes: 0 },
            mapped_ring_bytes: selection.model().unwrap().ring_bytes(selection.geometry())? as u64,
            host_exchange_bytes: 0, row_indices_bytes: 0,
        };
        InstalledModelAdmission::new(0, selection, Some(report))
    }

    #[test]
    fn typed_mixed_depth_handshake_matches_each_endpoint_before_allocation() -> Result<()> {
        let installed = typed_admission_with_depths(&[None, None, Some(2)])?;
        for (index, expected_depth) in [8, 8, 2].into_iter().enumerate() {
            for depth in [0, 1, 2, 4, 8, 9] {
                let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
                let mut client = TcpStream::connect(listener.local_addr()?)?;
                let (server, _) = listener.accept()?;
                let lease = installed.reserve_endpoint(Some(index as u32))?.unwrap();
                let (endpoint, rings) = lease.contract();
                assert_eq!(rings.depth, expected_depth);
                let (request, response) = endpoint.wire_bytes(installed.selection().geometry()).unwrap();
                let (request, response) = persistent_ring_geometry(rings.max_frame_bytes, request,
                    response, rings.slot_bytes, rings.alignment, rings.depth)?;
                let mut start = placement_start(Some(installed.selection().digest()?), 0);
                start["model_endpoint"] = index.into();
                start["ring_depth"] = depth.into();
                start["request_capacity_wire_bytes"] = request.slot_capacity_bytes.into();
                start["response_capacity_wire_bytes"] = response.slot_capacity_bytes.into();
                start["request_slot_stride_bytes"] = request.slot_stride_bytes.into();
                start["response_slot_stride_bytes"] = response.slot_stride_bytes.into();
                start["request_registered_span_bytes"] = (request.slot_stride_bytes * depth).into();
                start["response_registered_span_bytes"] = (response.slot_stride_bytes * depth).into();
                write_control(&mut client, &start)?;
                let result = LocalVerbsExpertConnection::read_persistent_start(&server, 64 << 20, Some(&installed));
                assert_eq!(result.is_ok(), depth == expected_depth,
                    "endpoint={index} depth={depth}: {:?}", result.err());
            }
        }
        for depth in [0, 9, usize::MAX] {
            assert!(typed_admission_with_depths(&[None, Some(depth)]).is_err());
        }
        Ok(())
    }

    #[test]
    fn typed_endpoint_rejects_wrong_ring_contract_before_allocation() -> Result<()> {
        let installed = typed_admission()?;
        for (field, changed) in [("model_endpoint", 1), ("request_capacity_wire_bytes", 4096),
            ("request_slot_stride_bytes", 4096), ("ring_depth", 4)] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let mut client = TcpStream::connect(listener.local_addr()?)?;
            let (server, _) = listener.accept()?;
            let mut start = placement_start(Some(installed.selection().digest()?), 0);
            start["model_endpoint"] = 0.into();
            start["request_capacity_wire_bytes"] = (8 << 20).into();
            start["response_capacity_wire_bytes"] = (8 << 20).into();
            start["request_slot_stride_bytes"] = (8 << 20).into();
            start["response_slot_stride_bytes"] = (8 << 20).into();
            start["request_registered_span_bytes"] = (64 << 20).into();
            start["response_registered_span_bytes"] = (64 << 20).into();
            start[field] = changed.into();
            write_control(&mut client, &start)?;
            assert!(LocalVerbsExpertConnection::read_persistent_start(&server, 64 << 20, Some(&installed)).is_err());
            assert!(installed.reserve_endpoint(Some(0))?.is_some());
        }
        Ok(())
    }

    #[test]
    fn failed_quiesce_drop_keeps_registered_ring_credit_charged() -> Result<()> {
        let budget = RingBudget::new(100);
        let installed = typed_admission()?;
        let mut credits = EndpointCredits { rings: Some(budget.reserve(100)?), endpoint: installed.reserve_endpoint(Some(0))? };
        let mut destroyed = false;
        assert!(credits.retire(|| {
            anyhow::ensure!(false, "simulated failed QP quiesce");
            destroyed = true;
            Ok(())
        }).is_err());
        assert!(!destroyed);
        drop(credits);
        assert_eq!(budget.used(), 100);
        assert!(budget.reserve(1).is_err());
        assert!(installed.reserve_endpoint(Some(0)).is_err());
        Ok(())
    }

    #[test]
    fn retained_endpoint_owns_credit_until_successful_teardown() -> Result<()> {
        struct Owner(EndpointCredits);
        impl Drop for Owner {
            fn drop(&mut self) { self.0.retire(|| Ok(())).unwrap(); }
        }
        let budget = RingBudget::new(100);
        let installed = typed_admission()?;
        let endpoint = Arc::new(Owner(EndpointCredits { rings: Some(budget.reserve(100)?), endpoint: installed.reserve_endpoint(Some(0))? }));
        let retained_response = Arc::clone(&endpoint);
        drop(endpoint);
        assert_eq!(budget.used(), 100);
        assert!(installed.reserve_endpoint(Some(0)).is_err());
        drop(retained_response);
        assert_eq!(budget.used(), 0);
        assert!(installed.reserve_endpoint(Some(0))?.is_some());
        let pending = budget.reserve(100)?;
        drop(pending);
        assert_eq!(budget.used(), 0);
        Ok(())
    }

    #[test]
    fn write_mode_error_response_sets_error_flag_without_losing_request_identity() {
        let mut header = ExpertProtocolV2ResponseHeader {
            request_id: 42, placement_version: 7, layer_id: 13, row_count: 0,
            output_dim: 80, output_dtype: crate::ExpertV2Dtype::Bf16,
            output_row_stride_bytes: 160, output_payload_bytes: 0,
            status: crate::ExpertProtocolV2Status::Error, flags: 0, executor_id: 1,
        };
        assert_eq!(written_response_flag(&header, false), 42 | VERBS_HOST_WRITE_FLAG_ERROR);
        header.status = crate::ExpertProtocolV2Status::Ok;
        assert_eq!(written_response_flag(&header, false), 42);
        assert_eq!(written_response_flag(&header, true), 42 | VERBS_HOST_WRITE_FLAG_ERROR);
    }

    #[test]
    fn busy_peer_eof_releases_rings_without_an_idle_window() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let mut client = TcpStream::connect(listener.local_addr()?)?;
        let (server, _) = listener.accept()?;
        let budget = RingBudget::new(100);
        let credit = budget.reserve(100)?;
        let (ready, busy) = std::sync::mpsc::channel();
        let owner = std::thread::spawn(move || -> Result<()> {
            let mut next = Instant::now();
            check_peer_liveness(&server, &mut next)?;
            ready.send(())?;
            let started = Instant::now();
            // Simulate continuous completions; there is never a one-second idle window.
            let mut polls = 0u8;
            loop {
                if polls == 0 && check_peer_liveness(&server, &mut next).is_err() { break; }
                polls = polls.wrapping_add(1);
                anyhow::ensure!(started.elapsed() < Duration::from_millis(500), "peer EOF not observed promptly");
                std::thread::yield_now();
            }
            drop(credit);
            drop(server);
            Ok(())
        });
        busy.recv_timeout(Duration::from_secs(1))?;
        let started = Instant::now();
        client.shutdown(std::net::Shutdown::Write)?;
        client.set_read_timeout(Some(Duration::from_millis(500)))?;
        let mut byte = [0u8; 1];
        assert_eq!(std::io::Read::read(&mut client, &mut byte)?, 0);
        owner.join().unwrap()?;
        assert!(started.elapsed() < Duration::from_millis(500));
        assert_eq!(budget.used(), 0);
        Ok(())
    }

    fn placement_selection(layers: &[usize]) -> crate::worker_selection::WorkerSelection {
        let identity = serde_json::from_value(serde_json::json!({
            "checkpoint": "0".repeat(64), "geometry": [6144, 384, 8, 2048, 70],
            "world": 2, "topology": [2, 1], "ownership_wire": false
        })).unwrap();
        crate::worker_selection::WorkerSelection::new(identity, layers).unwrap()
    }

    fn placement_start(digest: Option<String>, lane: u32) -> serde_json::Value {
        let native = serde_json::json!({
            "port_num": 1, "qp_num": 2, "psn": 3, "lid": 0, "active_mtu": 5,
            "gid_hex": "0".repeat(32), "send_frame_bytes": 4096, "recv_frame_bytes": 4096,
            "send_registered_span_bytes": 32768, "recv_registered_span_bytes": 32768,
            "max_send_wr": 8, "max_recv_wr": 8, "max_sge": 2,
            "device_name": "test", "status": "test"
        });
        let mut endpoint = native.clone();
        endpoint["role"] = "client".into();
        endpoint["host"] = "coordinator".into();
        let mut start = serde_json::json!({
            "message": "protocol_v2_persistent_start", "wire_version": PERSISTENT_CONTROL_VERSION,
            "execution_lane": lane,
            "request_capacity_wire_bytes": 4096, "response_capacity_wire_bytes": 4096,
            "request_registered_span_bytes": 32768, "response_registered_span_bytes": 32768,
            "ring_depth": 8, "request_slot_stride_bytes": 4096, "response_slot_stride_bytes": 4096,
            "client_endpoint": endpoint, "client_native_endpoint": native
        });
        if let Some(digest) = digest { start["worker_selection_digest"] = digest.into(); }
        start
    }

    /// The persistent bootstrap with reply targets, over a real TCP pair (no
    /// NIC): the worker reads the coordinator's start, writes ready, then
    /// reads `protocol_v2_reply_target`, which the coordinator sends only
    /// after ready. A large-reply session must get a valid target for its
    /// ring; a small-reply or write-mode one must get none; anything else,
    /// a wrong message or a stale control version, fails closed.
    #[test]
    fn bootstrap_reads_the_reply_target_after_ready_and_fails_closed() -> Result<()> {
        let small = 64 << 10;
        let large = 33_816_672;
        let ring = |capacity: usize| VerbsHostRdmaRing::new(capacity, 4096, 8);
        let target = |ring: VerbsHostRdmaRing| request_write::ReplyTarget { address: 1 << 40, rkey: 3,
            slot_stride_bytes: ring.slot_stride_bytes, depth: ring.depth, landing: None };
        let cases: Vec<(usize, bool, serde_json::Value, bool)> = vec![
            (large, false, serde_json::to_value(VerbsHostProtocolV2ReplyTarget { message: "protocol_v2_reply_target".into(),
                wire_version: PERSISTENT_CONTROL_VERSION, target: Some(target(ring(large)?)) })?, true),
            (small, false, serde_json::to_value(VerbsHostProtocolV2ReplyTarget { message: "protocol_v2_reply_target".into(),
                wire_version: PERSISTENT_CONTROL_VERSION, target: None })?, true),
            // Write mode: replies are flagged writes to the client's plane.
            (large, true, serde_json::to_value(VerbsHostProtocolV2ReplyTarget { message: "protocol_v2_reply_target".into(),
                wire_version: PERSISTENT_CONTROL_VERSION, target: None })?, true),
            (large, false, serde_json::to_value(VerbsHostProtocolV2ReplyTarget { message: "protocol_v2_reply_target".into(),
                wire_version: PERSISTENT_CONTROL_VERSION, target: None })?, false),
            (small, false, serde_json::to_value(VerbsHostProtocolV2ReplyTarget { message: "protocol_v2_reply_target".into(),
                wire_version: PERSISTENT_CONTROL_VERSION, target: Some(target(ring(small)?)) })?, false),
            (large, false, serde_json::to_value(VerbsHostProtocolV2ReplyTarget { message: "protocol_v2_reply_target".into(),
                wire_version: PERSISTENT_CONTROL_VERSION, target: Some(request_write::ReplyTarget { depth: 2,
                    ..target(ring(large)?) }) })?, false),
            (large, false, serde_json::to_value(VerbsHostProtocolV2ReplyTarget { message: "protocol_v2_reply_target".into(),
                wire_version: PERSISTENT_CONTROL_VERSION - 1, target: Some(target(ring(large)?)) })?, false),
            (large, false, serde_json::json!({"message": "protocol_v2_persistent_ready",
                "wire_version": PERSISTENT_CONTROL_VERSION}), false),
        ];
        for (capacity, write_mode, reply, accepted) in cases {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let mut coordinator = TcpStream::connect(listener.local_addr()?)?;
            let (worker, _) = listener.accept()?;
            let mut start = placement_start(None, 0);
            start["response_capacity_wire_bytes"] = capacity.into();
            if write_mode {
                start["write_target"] = serde_json::json!({"plane_addr": 1, "plane_bytes": capacity, "flag_addr": 2, "rkey": 4});
            }
            write_control(&mut coordinator, &start)?;
            let start: VerbsHostProtocolV2PersistentStart = decode_persistent_control(
                read_control_value(&mut BufReader::new(worker.try_clone()?))?)?;
            // The worker answers ready; the coordinator reads it before sending
            // its reply target, so the start reader cannot have buffered it.
            write_control(&mut worker.try_clone()?, &serde_json::json!({"message": "protocol_v2_persistent_ready"}))?;
            read_control_value(&mut BufReader::new(coordinator.try_clone()?))?;
            write_control(&mut coordinator, &reply)?;
            let result = read_reply_target(&mut BufReader::new(worker), &start, ring(capacity)?);
            assert_eq!(result.is_ok(), accepted, "capacity {capacity} write {write_mode}: {reply} -> {result:?}");
            if let Ok(target) = result { assert_eq!(target.is_some(), capacity == large && !write_mode); }
        }
        Ok(())
    }

    #[test]
    fn worker_selection_lanes_reject_mismatch_before_ring_reservation() -> Result<()> {
        let selection = placement_selection(&[13, 14]);
        let digest = selection.digest()?;
        for (wire, expected, accepted) in [
            (None, None, true), (None, Some(&selection), false),
            (Some(digest.clone()), None, false),
            (Some("1".repeat(64)), Some(&selection), false),
            (Some(digest), Some(&selection), true),
        ] {
            for lane in [0, 1, 2] {
                let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
                let mut client = TcpStream::connect(listener.local_addr()?)?;
                let (server, _) = listener.accept()?;
                write_control(&mut client, &placement_start(wire.clone(), lane))?;
                let budget = RingBudget::new(0);
                let installed = expected.map(|selection|
                    crate::worker_selection::InstalledModelAdmission::new(0, selection.clone(), None)).transpose()?;
                let start = LocalVerbsExpertConnection::read_persistent_start(
                    &server, 4096, installed.as_ref());
                assert_eq!(start.is_ok(), accepted);
                assert_eq!(budget.used(), 0);
                if let Ok(start) = start { assert_eq!(start.execution_lane, lane); }
            }
        }
        Ok(())
    }

    #[test]
    fn worker_selection_repeat_ack_is_control_only_and_immutable() -> Result<()> {
        let selection = placement_selection(&[13, 14]);
        let changed = placement_selection(&[14]);
        for (sent, rank, accepted) in [(&selection, 0, true), (&changed, 0, false), (&selection, 1, false)] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let mut client = TcpStream::connect(listener.local_addr()?)?;
            let (server, _) = listener.accept()?;
            write_control(&mut client, &serde_json::json!({
                "message": "worker_selection_start", "rank": rank,
                "selection": sent, "digest": sent.digest()?, "admission_timeout_ms": 3000
            }))?;
            let installed = crate::worker_selection::InstalledModelAdmission::new(0, selection.clone(), None)?;
            let result = LocalVerbsExpertConnection::accept_with_selection(
                server, 4096, false, RingBudget::new(0), Some(&installed));
            let error = result.err().context("control-only selection allocated a native session")?;
            assert_eq!(error.is::<crate::worker_selection::WorkerSelectionOnly>(), accepted);
            client.set_read_timeout(Some(Duration::from_secs(3)))?;
            let reply = read_control_value(&mut BufReader::new(client));
            if accepted {
                let reply = reply?;
                assert_eq!(reply["message"], "worker_selection_ready");
                assert_eq!(reply["selection"], serde_json::to_value(&selection)?);
                assert_eq!(reply["digest"], selection.digest()?);
            } else { assert!(reply.is_err()); }
        }
        Ok(())
    }

    #[test]
    fn ring_budget_reserves_releases_and_tracks_peak() -> Result<()> {
        let budget = RingBudget::new(100);
        assert_eq!((budget.limit(), budget.used(), budget.peak()), (100, 0, 0));
        let first = budget.reserve(60)?;
        assert_eq!(first.bytes(), 60);
        assert_eq!((budget.used(), budget.peak()), (60, 60));
        let second = budget.reserve(40)?;
        assert_eq!((budget.used(), budget.peak()), (100, 100));
        assert!(budget.reserve(1).is_err());
        assert_eq!(budget.used(), 100);
        drop(first);
        assert_eq!((budget.used(), budget.peak()), (40, 100));
        let third = budget.reserve(60)?;
        assert_eq!((budget.used(), budget.peak()), (100, 100));
        drop(second);
        drop(third);
        assert_eq!(budget.used(), 0);
        // A request above the limit fails without holding credit; a genuine
        // checked_add overflow is exercised by holding one byte of a
        // usize::MAX budget and then requesting usize::MAX.
        assert!(budget.reserve(101).is_err());
        assert_eq!(budget.used(), 0);
        let huge = RingBudget::new(usize::MAX);
        let one = huge.reserve(1)?;
        assert!(huge.reserve(usize::MAX).is_err(), "checked_add overflow must be rejected");
        assert_eq!(huge.used(), 1);
        drop(one);
        assert_eq!(huge.used(), 0);
        Ok(())
    }

    /// Unit-level RAII contract: a rejected reservation never charges the
    /// budget, and the credit owned by a dropped connection is returned so the
    /// next admission can succeed. The full socket accept path additionally
    /// requires RoCE preflight and a GPU, so it is covered by the live tests.
    #[test]
    fn unit_raii_credit_release_recovers_after_failure() -> Result<()> {
        let budget = RingBudget::new(10);
        assert!(budget.reserve(11).is_err());
        assert_eq!(budget.used(), 0);
        let held = budget.reserve(10)?;
        assert!(budget.reserve(1).is_err());
        assert_eq!(budget.used(), 10);
        drop(held);
        let reacquired = budget.reserve(10)?;
        assert_eq!(budget.used(), 10);
        drop(reacquired);
        assert_eq!(budget.used(), 0);
        Ok(())
    }

    /// The advertised wire geometry is validated with the transport's own
    /// `from_wire` rules before any budget charge or native allocation.
    #[test]
    fn validated_ring_geometry_rejects_before_any_reservation() -> Result<()> {
        let alignment = crate::verbs_host_capabilities().preferred_alignment;
        let depth = 8;
        let capacity = 1 << 20;
        let stride = VerbsHostRdmaRing::new_with_max_depth(capacity, alignment, depth, 8)?
            .slot_stride_bytes;
        let span = stride * depth;
        let (request, response) =
            LocalVerbsExpertConnection::validated_ring_geometry(
                capacity, stride, span, capacity, stride, span, depth,
            )?;
        assert_eq!(request.registered_span_bytes, span);
        assert_eq!(response.registered_span_bytes, span);
        assert_eq!(request.depth, depth);
        // A mismatched stride, a mismatched span, a zero capacity and an
        // excessive depth are all rejected by from_wire before allocation.
        assert!(LocalVerbsExpertConnection::validated_ring_geometry(
            capacity, stride + alignment, span, capacity, stride, span, depth,
        )
        .is_err());
        assert!(LocalVerbsExpertConnection::validated_ring_geometry(
            capacity, stride, span + alignment, capacity, stride, span, depth,
        )
        .is_err());
        assert!(LocalVerbsExpertConnection::validated_ring_geometry(
            0, stride, span, capacity, stride, span, depth,
        )
        .is_err());
        assert!(LocalVerbsExpertConnection::validated_ring_geometry(
            capacity, stride, span, capacity, stride, span, 9,
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_limit() {
        use std::sync::Barrier;
        // 8 threads each want 16 bytes from a 64-byte budget: they contend for
        // the limit, so some must be rejected and the peak can never exceed it.
        let budget = RingBudget::new(64);
        let barrier = Arc::new(Barrier::new(8));
        let rejections = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let budget = Arc::clone(&budget);
            let barrier = Arc::clone(&barrier);
            let rejections = Arc::clone(&rejections);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let held = budget.reserve(16);
                if held.is_err() {
                    rejections.fetch_add(1, Ordering::AcqRel);
                }
                assert!(budget.used() <= budget.limit());
                // Hold any credit until every thread has attempted its reserve,
                // then release it so the peak is observed deterministically.
                barrier.wait();
                drop(held);
            }));
        }
        for handle in handles {
            handle.join().expect("budget worker");
        }
        assert!(budget.peak() <= budget.limit());
        assert!(budget.peak() > 0);
        assert!(rejections.load(Ordering::Acquire) > 0);
        assert_eq!(budget.used(), 0);
    }
}
