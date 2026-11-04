//! Every persistent Spark frame, request or reply, is one registered RC SEND
//! into the peer's posted receive. A copied request is staged whole in its
//! session's send slot; a request whose payload views registered egress
//! gathers its header from the slot and its payload from the egress region.
//! RC holds a SEND until its receive is posted (RNR retry), so neither side
//! needs a credit for the peer's ring beyond its own ring depth.
use super::*;

/// Send-queue entries per frame, oldest first. A frame owns its source slot
/// (and, for a gathered request, its registered egress slot) from staging
/// until its SEND completes; only then is the slot restaged or rewritten.
#[derive(Debug, Default)]
pub(super) struct SendQueue {
    /// Per posted frame, oldest first: the registered egress slot it reads.
    frames: VecDeque<Option<usize>>,
}
impl SendQueue {
    /// One SEND posted; `egress` is the registered slot its payload reads.
    pub fn posted(&mut self, egress: Option<usize>) { self.frames.push_back(egress); }
    /// Send CQEs arrive in post order on one QP: they retire the oldest frames.
    pub fn completed(&mut self, count: usize) -> Result<()> {
        anyhow::ensure!(count <= self.frames.len(), "{count} send completions with {} in flight", self.frames.len());
        self.frames.drain(..count);
        Ok(())
    }
    /// SENDs posted and not yet completed (one send-queue entry each).
    pub fn in_flight(&self) -> usize { self.frames.len() }
    /// Reserve no storage and never wait: reap at most once if posted SENDs
    /// plus deferred replies currently fill the response ring.
    pub fn try_slot(&mut self, depth: usize, held: usize,
        reap: impl FnOnce(usize) -> Result<usize>) -> Result<bool> {
        if self.in_flight() + held < depth { return Ok(true); }
        if self.in_flight() == 0 { return Ok(false); }
        let completed = reap(self.in_flight())?;
        self.completed(completed)?;
        Ok(self.in_flight() + held < depth)
    }
    /// Whether some uncompleted SEND still reads registered egress `slot`.
    pub fn reads_egress(&self, slot: usize) -> bool { self.frames.contains(&Some(slot)) }
}

impl VerbsHostProtocolV2PersistentClientSession {
    /// Waits for one send completion (bounded by `timeout`).
    pub(super) fn wait_request_send(&mut self, timeout: Duration) -> Result<()> {
        let stats = self.endpoint.poll_stats(1, 0, timeout).context("waiting for a request send completion")?;
        anyhow::ensure!(stats.recv_completions == 0, "request send wait consumed a response");
        self.request_sends.completed(stats.send_completions as usize)
    }

    /// Non-blocking send-CQ reap; never polls an idle CQ (native rejects it).
    pub(super) fn try_reap_request_sends(&mut self) -> Result<bool> {
        if self.request_sends.in_flight() == 0 { return Ok(false); }
        let stats = self.endpoint.try_poll(u32::try_from(self.request_sends.in_flight())?, 0)?;
        anyhow::ensure!(stats.recv_completions == 0, "request send poll consumed a response");
        self.request_sends.completed(stats.send_completions as usize)?;
        Ok(stats.send_completions > 0)
    }

    /// Validates `request`, stages its frame in this session's send slot (the
    /// payload stays in registered egress if it is there) and posts it as one
    /// SEND. A placed-landing session first posts the request's own receive at
    /// `landing` (its wave's slice of the registered range), so the reply has
    /// a receive before the worker can answer; other sessions take `None`.
    pub(super) fn post_request(&mut self, request: &ExpertProtocolV2Request, config: &TcpTransportConfig,
        landing: Option<LandingSlice>) -> Result<VerbsHostProtocolV2ChunkSubmissionTiming> {
        anyhow::ensure!(landing.is_some() == self.placed_landing,
            "a placed GPU landing needs every request's landing slice, and only it takes one");
        if let Some(lease) = &self.endpoint.credits.endpoint { lease.validate_request(&request.header)?; }
        if let Some(ring) = &mut self.retained_response_ring {
            ring.reclaim_before_request(&self.endpoint, self.response_ring)?;
        }
        let total_started = config.timing.then(Instant::now);
        let prefix_bytes = self.request_frame.encode_request_prefix(request)?.len();
        let bytes = prefix_bytes.checked_add(request.hidden_payload.len())
            .context("persistent verbs-host ProtocolV2 request byte count overflow")?;
        anyhow::ensure!(bytes <= config.max_frame_bytes && bytes <= self.request_capacity_wire_bytes,
            "persistent verbs-host ProtocolV2 request frame length {bytes} exceeds capacity {}",
            self.request_capacity_wire_bytes.min(config.max_frame_bytes));
        let expected_started = config.timing.then(Instant::now);
        let expected_response = verbs_host_expected_response_wire_bytes(request)?;
        let expected_response_ms = elapsed_ms_optional(expected_started);
        anyhow::ensure!(expected_response <= config.max_frame_bytes && expected_response <= self.response_capacity_wire_bytes,
            "persistent verbs-host ProtocolV2 expected response frame length {expected_response} exceeds capacity {}",
            self.response_capacity_wire_bytes.min(config.max_frame_bytes));
        let encode_ms = elapsed_ms_optional(total_started);
        if let Some(slice) = landing {
            self.post_placed_receive(slice, request)?;
        }
        // Source slot: the SEND that used it last has completed.
        while self.request_sends.in_flight() >= self.request_ring.depth {
            self.wait_request_send(config.timeout)?;
        }
        let send_started = config.timing.then(Instant::now);
        let slot_offset = self.request_ring.slot_offset(self.request_send_sequence);
        let wr_id = VERBS_HOST_SEND_WR_ID + (self.request_send_sequence % self.request_ring.depth) as u64;
        let payload_region = self.egress.as_ref()
            .and_then(|(buffer, region)| Some((*region, buffer.offset_of(&request.hidden_payload)?)))
            .filter(|_| !request.hidden_payload.is_empty());
        let staged = if payload_region.is_some() { prefix_bytes } else { bytes };
        let view = self.endpoint.send_buffer_view()?;
        anyhow::ensure!(!view.host_ptr.is_null() && slot_offset <= view.bytes && staged <= view.bytes - slot_offset,
            "persistent verbs-host request slot exceeds its send buffer");
        // SAFETY: the range lies inside this endpoint's registered send buffer
        // (checked above). No WR reads it: the loop above waited until the
        // SEND that last used this slot completed, and only this session
        // posts from its buffer.
        let slot = unsafe { std::slice::from_raw_parts_mut(view.host_ptr.cast::<u8>().add(slot_offset), staged) };
        let (head, payload) = slot.split_at_mut(prefix_bytes);
        head.copy_from_slice(self.request_frame.as_slice());
        let egress_slot = match payload_region {
            Some((region, at)) => {
                // Zero-copy: the header from this session's slot, the payload
                // straight from the shared registered request buffer.
                self.endpoint.library.rdma_rc_endpoint_post_send_slot_region(self.endpoint.info.handle, slot_offset,
                    prefix_bytes, region, at, request.hidden_payload.len(), wr_id)?;
                self.endpoint.traffic.sent(bytes);
                self.egress.as_ref().map(|(buffer, _)| buffer.slot_of(at))
            }
            None => {
                if payload.len() >= PARALLEL_REQUEST_COPY_BYTES {
                    copy_threads(payload, &request.hidden_payload);
                } else {
                    payload.copy_from_slice(&request.hidden_payload);
                }
                self.endpoint.post_send_at(slot_offset, bytes, wr_id)?;
                None
            }
        };
        self.request_sends.posted(egress_slot);
        self.request_send_sequence = self.request_send_sequence.wrapping_add(1);
        Ok(VerbsHostProtocolV2ChunkSubmissionTiming { total_started, encode_ms, expected_response_ms,
            send_ms: elapsed_ms_optional(send_started) })
    }
}

/// A placed receive uses this request's negotiated plane, not the reusable
/// ring's BF16 capacity bound. Fallback requests have their compressed flags cleared.
fn landed_payload_bytes(request: &ExpertProtocolV2Request) -> Result<usize> {
    let rows = request.header.row_count as usize;
    if let Some(plan) = &request.partial {
        anyhow::ensure!(rows > 0 && request.header.route_count as usize % rows == 0,
            "GPU-landed partial reply needs a uniform top-k per row");
        return Ok(plan.payload(request.header.route_count as usize / rows,
            request.header.hidden_dim as usize)?.3);
    }
    let hidden = request.header.hidden_dim as usize;
    let row_bytes = if request.nvfp4_e2m1_fp8_e4m3_response_enabled() {
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(hidden)?
    } else if request.header.flags & crate::EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_UE8M0_K32 != 0 {
        ExpertV2Dtype::Fp8E4m3Ue8m0K32.row_bytes(hidden)?
    } else {
        // Packed ingress may still produce BF16 output without reply compression.
        (request.header.hidden_row_stride_bytes as usize).max(ExpertV2Dtype::Bf16.row_bytes(hidden)?)
    };
    rows.checked_mul(row_bytes).context("GPU-landed reply payload byte count overflow")
}

fn validate_placed_payload(request: &ExpertProtocolV2Request, slice: LandingSlice) -> Result<()> {
    let payload = landed_payload_bytes(request)?;
    anyhow::ensure!(payload <= slice.bytes,
        "a reply of {payload} payload bytes exceeds its {}-byte landing slice", slice.bytes);
    Ok(())
}

impl VerbsHostProtocolV2PersistentClientSession {
    /// One receive for the next reply: its header into the next header slot
    /// of the response ring (replies complete in post order on this QP), its
    /// payload into `slice` of the registered landing range. At most a ring of
    /// replies may be outstanding, so header slots are never reused early.
    fn post_placed_receive(&mut self, slice: LandingSlice, request: &ExpertProtocolV2Request) -> Result<()> {
        let outstanding = self.response_post_sequence.wrapping_sub(self.response_recv_sequence);
        anyhow::ensure!(outstanding < self.response_ring.depth,
            "placed GPU landing has {outstanding} replies outstanding, its header ring holds {}", self.response_ring.depth);
        validate_placed_payload(request, slice)?;
        let sequence = self.response_post_sequence;
        let wr_id = VERBS_HOST_RECV_WR_ID + (sequence % self.response_ring.depth) as u64;
        // SAFETY: the range was registered at connect and is kept allocated by
        // the landing's owner (`SparkExperts::set_gpu_landing`); the caller
        // places only this wave in the slice until its reply is consumed, and
        // native checks the slice against the registered range.
        unsafe {
            self.endpoint.library.rdma_rc_endpoint_post_recv_landing_at(self.endpoint.info.handle,
                self.response_ring.slot_offset(sequence), self.response_ring.slot_capacity_bytes, wr_id,
                slice.offset, slice.bytes)?;
        }
        self.response_post_sequence = sequence.wrapping_add(1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_reply_ring_tries_once_without_waiting_or_consuming_receives() -> Result<()> {
        let mut queue = SendQueue::default();
        for _ in 0..3 { queue.posted(None); }
        let mut polls = 0;
        assert!(!queue.try_slot(4, 1, |posted| {
            assert_eq!(posted, 3);
            polls += 1;
            Ok(0)
        })?);
        assert_eq!(polls, 1, "a full ring must return rather than retry");
        assert_eq!(queue.in_flight(), 3);
        assert!(queue.try_slot(4, 1, |_| Ok(1))?);
        assert!(queue.try_slot(4, 1, |_| panic!("free slot must not poll"))?);
        assert!(!SendQueue::default().try_slot(4, 4, |_| panic!("no SENDs to poll"))?);
        assert!(queue.try_slot(2, 0, |_| Ok(3)).is_err(), "invalid CQ counts fail closed");
        Ok(())
    }

    /// A full-capacity BF16 wave's landed payload is exactly its plane: the
    /// ring bound's row-index prefix does not count against the slice (the
    /// 2048-row loopback refused a 25,165,824-byte slice for 8 KiB of indices).
    #[test]
    fn landed_payload_bound_is_the_unindexed_plane() -> Result<()> {
        let (rows, hidden) = (2048usize, 6144u32);
        let request = ExpertProtocolV2Request::new_bytes(1, 1, 1, hidden, ExpertV2Dtype::Bf16,
            (0..rows).map(|row| crate::ExpertProtocolV2RowDescriptor { row_id: row as u64,
                source_kind: crate::ExpertV2SourceKind::Prefill, source_request_id: 1, token_position: row as u64,
                route_offset: 0, route_count: 0 }).collect(), vec![], bytes::Bytes::from(vec![0u8; rows * hidden as usize * 2]))?;
        let payload = rows * hidden as usize * 2;
        assert_eq!(landed_payload_bytes(&request)?, payload);
        validate_placed_payload(&request, LandingSlice { offset: 0, bytes: payload })?;
        assert!(validate_placed_payload(&request, LandingSlice { offset: 0, bytes: payload - 1 }).is_err());
        Ok(())
    }

    #[test]
    fn hybrid_warmup_placed_landing_matches_worker_reply_and_fallback() -> Result<()> {
        const CHILD: &str = "CUTEAFD_TEST_HYBRID_REPLY_LANDING";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "verbs::send_queue::tests::hybrid_warmup_placed_landing_matches_worker_reply_and_fallback", "--nocapture"])
                .env(CHILD, "1").status()?;
            anyhow::ensure!(status.success(), "GLM Flash hybrid landing child failed");
            return Ok(());
        }
        use crate::{ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
            ExpertV2SourceKind::{Decode, MtpVerify, Prefill}};
        use cuteafd_core::SparkReplyFormat::{Bf16, Fp8, Nvfp4};
        let geometry = cuteafd_core::ExpertGeometry::GLM5_FLASH;
        cuteafd_core::set_expert_geometry(geometry).unwrap();
        let onboarded = 3..24;
        let spark_layers = 24..45;
        assert!(onboarded.end == spark_layers.start && spark_layers.end == geometry.layers);
        for selected in [Bf16, Fp8, Nvfp4] {
            for (sources, eligible) in [
                (vec![(2752, Prefill)], true),
                (vec![(65, Prefill)], true),
                (vec![(64, Prefill)], true),
                (vec![(65, Prefill), (64, Prefill)], true),
                (vec![(65, Prefill), (1, Decode)], false),
                (vec![(65, Prefill), (1, MtpVerify)], false),
            ] {
                let origins: Vec<_> = sources.iter().enumerate().flat_map(|(id, &(rows, kind))|
                    std::iter::repeat_n((id as u64 + 1, kind), rows)).collect();
                let rows = origins.len();
                let mut request = ExpertProtocolV2Request::new_bytes(1, 1, spark_layers.start,
                    geometry.hidden, ExpertV2Dtype::Bf16,
                    origins.iter().enumerate().map(|(row, &(source_request_id, source_kind))|
                        ExpertProtocolV2RowDescriptor { row_id: row as u64, source_kind, source_request_id,
                            token_position: row as u64, route_offset: row as u32 * geometry.topk,
                            route_count: geometry.topk }).collect(),
                    (0..rows as u32 * geometry.topk).map(|route| ExpertProtocolV2RouteEntry {
                        row_index: route / geometry.topk, expert_id: route % geometry.topk,
                        gate_weight: 1.0 / geometry.topk as f32 }).collect(),
                    bytes::Bytes::from(vec![0; rows * geometry.row_bytes() as usize]))?;
                request.header.flags |= crate::protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
                let format = if eligible { selected } else { Bf16 };
                assert_eq!(request.select_combined_reply(cuteafd_core::spark_reply::SparkReplyFormats { prefill: Some(selected), decode: None })?.format, format);
                let encoded = request.encode()?;
                let worker = crate::expert::BackboneRequest::parse(&encoded, 2752)?;
                assert!(spark_layers.contains(&worker.layer()) && !onboarded.contains(&worker.layer()));
                let payload_bytes = rows * format.row_bytes(geometry.hidden as usize).unwrap();
                assert_eq!(worker.plane_bytes()?, payload_bytes);
                assert_eq!(landed_payload_bytes(&request)?, payload_bytes);
                let slice = LandingSlice { offset: 2 * 2752 * geometry.row_bytes() as usize,
                    bytes: payload_bytes };
                validate_placed_payload(&request, slice)?;
                assert!(validate_placed_payload(&request, LandingSlice { bytes: payload_bytes - 1, ..slice }).is_err());
                // The session still reserves enough for a later BF16 fallback.
                let ring_bound = verbs_host_expected_response_wire_bytes(&request)?;
                assert_eq!(ring_bound, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
                    + rows * (4 + geometry.row_bytes() as usize));
                if !eligible && selected != Bf16 {
                    let compressed_bytes = rows * selected.row_bytes(geometry.hidden as usize).unwrap();
                    assert!(validate_placed_payload(&request, LandingSlice { bytes: compressed_bytes, ..slice }).is_err());
                }
                let payload = vec![0; payload_bytes];
                let response = worker.response(11, &payload)?;
                let mut frame = ExpertProtocolV2FrameBuffer::default();
                let prefix = frame.encode_borrowed_response_prefix(&response)?;
                let header = ExpertProtocolV2ResponseView::parse_landed_header(prefix)?;
                assert_eq!(header.output_payload_bytes as usize, slice.bytes);
                let mut assembler = ProtocolV2ResponseChunkAssembler::validation_only(&request);
                assembler.accept_landed(&request, &header)?;
                assembler.finish_validation()?;
                assert_eq!(landed_payload_bytes(&request.clone().with_debug_checksum())?, payload_bytes);
            }
        }
        Ok(())
    }

    #[test]
    fn source_slots_free_in_post_order_one_completion_per_frame() -> Result<()> {
        let mut queue = SendQueue::default();
        for egress in [Some(0), None, Some(1)] { queue.posted(egress); }
        assert_eq!(queue.in_flight(), 3);
        queue.completed(1)?;
        assert_eq!(queue.in_flight(), 2);
        assert!(!queue.reads_egress(0) && queue.reads_egress(1));
        assert!(queue.completed(3).is_err(), "more completions than SENDs in flight");
        queue.completed(2)?;
        assert_eq!(queue.in_flight(), 0);
        Ok(())
    }

    /// Rewriting egress slot `w % N` waits only for the request that reads
    /// that slot (wave `w - N`), not for every wave still on the wire.
    #[test]
    fn egress_slot_reuse_waits_only_for_its_own_readers() -> Result<()> {
        let mut queue = SendQueue::default();
        for slot in [0, 1, 2] { queue.posted(Some(slot)); }
        assert!(queue.reads_egress(0) && queue.reads_egress(2));
        queue.completed(1)?;
        assert!(!queue.reads_egress(0), "slot 0 is free once its own wave completed");
        assert!(queue.reads_egress(1) && queue.reads_egress(2));
        queue.posted(None); // a decode-sized copied frame reads no egress
        assert!(!queue.reads_egress(0));
        Ok(())
    }

    /// N waves rotating over N registered egress slots: rewriting slot `w % N`
    /// (`drain_egress_reads`, which retires completions oldest first while
    /// the slot is still read) retires exactly wave `w - N` (and any copied
    /// frame posted before it) and leaves every other wave's SEND on the wire.
    #[test]
    fn egress_slot_rewrite_retires_only_its_own_wave_while_others_stay_in_flight() -> Result<()> {
        for slots in [2usize, 3, 4] {
            let mut queue = SendQueue::default();
            let mut wire: VecDeque<Option<usize>> = VecDeque::new();
            for wave in 0..1024usize {
                let slot = wave % slots;
                let before = wire.len();
                let readers_before = wire.iter().filter(|reads| reads.is_some()).count();
                while queue.reads_egress(slot) {
                    queue.completed(1)?;
                    wire.pop_front();
                }
                let retired = before - wire.len();
                assert!(retired <= 2, "{slots} slots: wave {wave} retired {retired} frames");
                assert_eq!(readers_before - wire.iter().filter(|reads| reads.is_some()).count(),
                    usize::from(wave >= slots), "{slots} slots: wave {wave}");
                assert!(wire.iter().all(|&reads| reads != Some(slot)));
                assert_eq!(wire.iter().filter(|reads| reads.is_some()).count(), wave.min(slots - 1),
                    "the other waves' requests stay in flight");
                assert_eq!(queue.in_flight(), wire.len());
                queue.posted(Some(slot));
                wire.push_back(Some(slot));
                if wave % 3 == 1 {
                    // A decode-sized copied frame between prefill waves.
                    queue.posted(None);
                    wire.push_back(None);
                }
            }
        }
        Ok(())
    }
}
