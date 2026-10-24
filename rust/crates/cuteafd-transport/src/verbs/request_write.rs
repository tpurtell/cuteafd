//! Every persistent request frame lands in the worker's request slot for its
//! sequence. Bytes past the first [`CHUNK_BYTES`] go as RDMA WRITEs of at most
//! one chunk each; the frame's head (its first chunk, holding the header) goes
//! last on the same QP as the SEND that consumes the slot's posted receive. RC
//! ordering makes that receive completion follow every earlier WRITE, so the
//! worker's receive path is unchanged. A frame of at most one chunk is that one
//! SEND: small decode/verify requests post exactly what they always did.
//!
//! Credit: a WRITE may target slot `n % depth` only once the worker has
//! finished with request `n - depth` there. The worker posts a request's final
//! reply only after its executor returned and that slot's receive is reposted
//! (`local.rs`), so a received final reply is that credit; the client keeps at
//! most `depth` requests without a final reply ([`Credits`]).
use super::*;

/// Largest single request WR. Also the largest request that needs no WRITE.
pub(super) const CHUNK_BYTES: usize = 256 << 10;

/// The worker's request ring, exposed for remote WRITE. Present exactly when
/// the session's request capacity exceeds one chunk.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Target {
    pub address: u64,
    pub rkey: u32,
}

/// Whether a session of this request capacity can carry a WRITE tail.
pub(super) fn needs_target(request_capacity_wire_bytes: usize) -> bool {
    request_capacity_wire_bytes > CHUNK_BYTES
}

/// The frame ranges one request posts, in post order: every tail chunk as a
/// WRITE (`true`), then the head as the SEND (`false`). [`Upload`] walks the
/// same sequence one WR at a time.
#[cfg(test)]
fn parts(bytes: usize) -> impl Iterator<Item = (usize, usize, bool)> {
    let head = bytes.min(CHUNK_BYTES);
    (head..bytes).step_by(CHUNK_BYTES).map(move |offset| (offset, CHUNK_BYTES.min(bytes - offset), true))
        .chain(std::iter::once((0, head, false)))
}

/// Requests posted to one worker and not yet answered by a final reply. The
/// worker's slot for the next request is free only below `depth`.
#[derive(Debug, Default)]
pub(super) struct Credits {
    outstanding: usize,
}
impl Credits {
    pub fn acquire(&mut self, depth: usize) -> Result<()> {
        anyhow::ensure!(self.outstanding < depth,
            "request slot still owned by the worker: {} requests await their final reply", self.outstanding);
        self.outstanding += 1;
        Ok(())
    }
    pub fn final_reply(&mut self) -> Result<()> {
        self.outstanding = self.outstanding.checked_sub(1).context("final reply without an outstanding request")?;
        Ok(())
    }
    /// Write mode has no reply on this QP: the device lane posts a wave only
    /// after the GPU observed the previous wave's final flag on every rank
    /// (`SparkExperts::set_write_targets`), which the worker writes after
    /// reposting that slot. The post itself is therefore the credit.
    pub fn written_reply_observed(&mut self) { self.outstanding = 0; }
}

/// Send-queue entries per request, oldest first. A request owns its source
/// slot from staging until its head SEND is posted (`sealed`) and every one
/// of its WRs has completed; only then is the slot restaged. Completions can
/// arrive between part posts, so an unsealed request with zero WRs in flight
/// still owns its slot and still receives the parts posted after it.
#[derive(Debug, Default)]
pub(super) struct SendQueue {
    /// (WRs posted and not completed, head SEND posted), oldest first.
    requests: VecDeque<(usize, bool)>,
    pub in_flight: usize,
}
impl SendQueue {
    pub fn posted_request(&mut self) -> Result<()> {
        anyhow::ensure!(self.requests.back().is_none_or(|&(_, sealed)| sealed),
            "request staged before the previous request's head SEND was posted");
        self.requests.push_back((0, false));
        Ok(())
    }
    /// One WR of the newest request; `head` marks its final (head SEND) part.
    pub fn posted_part(&mut self, head: bool) -> Result<()> {
        let (parts, sealed) = self.requests.back_mut().context("request part without a request")?;
        anyhow::ensure!(!*sealed, "request part posted after its head SEND");
        *parts += 1;
        *sealed = head;
        self.in_flight += 1;
        Ok(())
    }
    /// Send CQEs arrive in post order on one QP: they retire the oldest WRs.
    pub fn completed(&mut self, mut count: usize) -> Result<()> {
        anyhow::ensure!(count <= self.in_flight, "{count} request send completions with {} in flight", self.in_flight);
        self.in_flight -= count;
        let mut index = 0;
        while count > 0 {
            let (parts, _) = self.requests.get_mut(index).context("request send completion without a request")?;
            let taken = count.min(*parts);
            *parts -= taken;
            count -= taken;
            index += 1;
        }
        while self.requests.front().is_some_and(|&(parts, sealed)| parts == 0 && sealed) {
            self.requests.pop_front();
        }
        Ok(())
    }
    /// Requests that still own their source slot.
    pub fn requests_in_flight(&self) -> usize { self.requests.len() }
}

/// A staged request whose parts are being posted.
pub(super) struct Upload {
    wr_id: u64,
    slot_offset: usize,
    remote: Option<u64>,
    prefix_bytes: usize,
    payload_region: Option<(u32, usize)>,
    bytes: usize,
    /// Next tail offset to WRITE; `bytes` once only the head SEND remains.
    cursor: usize,
    sent: bool,
    pub timing: VerbsHostProtocolV2ChunkSubmissionTiming,
}
impl Upload {
    pub fn sent(&self) -> bool { self.sent }
}

impl VerbsHostProtocolV2PersistentClientSession {
    fn reap_request_sends(&mut self, count: usize) -> Result<()> { self.request_sends.completed(count) }

    /// Waits for one send completion (bounded by `timeout`).
    pub(super) fn wait_request_send(&mut self, timeout: Duration) -> Result<()> {
        let stats = self.endpoint.poll_stats(1, 0, timeout).context("waiting for a request send completion")?;
        anyhow::ensure!(stats.recv_completions == 0, "request send wait consumed a response");
        self.reap_request_sends(stats.send_completions as usize)
    }

    /// Non-blocking send-CQ reap; never polls an idle CQ (native rejects it).
    pub(super) fn try_reap_request_sends(&mut self) -> Result<bool> {
        if self.request_sends.in_flight == 0 { return Ok(false); }
        let stats = self.endpoint.try_poll(u32::try_from(self.request_sends.in_flight)?, 0)?;
        anyhow::ensure!(stats.recv_completions == 0, "request send poll consumed a response");
        self.reap_request_sends(stats.send_completions as usize)?;
        Ok(stats.send_completions > 0)
    }

    /// Validates `request`, claims its worker slot and stages its frame in this
    /// session's send slot (the payload stays in registered egress if it is
    /// there). Nothing is posted yet.
    pub(super) fn stage_request(&mut self, request: &ExpertProtocolV2Request, config: &TcpTransportConfig)
        -> Result<Upload> {
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
        let remote = if bytes > CHUNK_BYTES {
            let target = self.request_target.context("request exceeds one chunk but the worker exposed no target")?;
            Some(target.address.checked_add(self.request_ring.slot_offset(self.request_send_sequence) as u64)
                .context("request WRITE remote address overflow")?)
        } else { None };
        // Source slot: every WR of the request that used it last has completed.
        while self.request_sends.requests_in_flight() >= self.request_ring.depth {
            self.wait_request_send(config.timeout)?;
        }
        let slot_offset = self.request_ring.slot_offset(self.request_send_sequence);
        let payload_region = self.egress.as_ref()
            .and_then(|(buffer, region)| Some((*region, buffer.offset_of(&request.hidden_payload)?)))
            .filter(|_| !request.hidden_payload.is_empty());
        let staged = if payload_region.is_some() { prefix_bytes } else { bytes };
        let view = self.endpoint.send_buffer_view()?;
        anyhow::ensure!(!view.host_ptr.is_null() && slot_offset <= view.bytes && staged <= view.bytes - slot_offset,
            "persistent verbs-host request slot exceeds its send buffer");
        // SAFETY: the range lies inside this endpoint's registered send buffer
        // (checked above). No WR reads it: the loop above waited until every
        // WR of the request that last used this slot completed, and only this
        // session posts from its buffer.
        let slot = unsafe { std::slice::from_raw_parts_mut(view.host_ptr.cast::<u8>().add(slot_offset), staged) };
        let (head, payload) = slot.split_at_mut(prefix_bytes);
        head.copy_from_slice(self.request_frame.as_slice());
        if payload_region.is_none() {
            if payload.len() >= PARALLEL_REQUEST_COPY_BYTES {
                copy_threads(payload, &request.hidden_payload);
            } else {
                payload.copy_from_slice(&request.hidden_payload);
            }
        }
        let wr_id = VERBS_HOST_SEND_WR_ID + (self.request_send_sequence % self.request_ring.depth) as u64;
        // The worker's slot for this sequence; released by its final reply.
        self.request_credits.acquire(self.request_ring.depth)?;
        self.request_sends.posted_request()?;
        self.request_send_sequence = self.request_send_sequence.wrapping_add(1);
        Ok(Upload { wr_id, slot_offset, remote, prefix_bytes, payload_region, bytes, cursor: bytes.min(CHUNK_BYTES), sent: false,
            timing: VerbsHostProtocolV2ChunkSubmissionTiming { total_started,
                encode_ms: elapsed_ms_optional(total_started), expected_response_ms, send_ms: 0.0 } })
    }

    /// Posts the upload's next part if the send queue has room; returns whether
    /// it posted one. The head SEND goes only after every tail WRITE.
    pub(super) fn post_request_part(&mut self, upload: &mut Upload) -> Result<bool> {
        if upload.sent || self.request_sends.in_flight >= self.endpoint.info.max_send_wr as usize { return Ok(false); }
        let started = upload.timing.total_started.map(|_| Instant::now());
        let (offset, bytes, remote) = if upload.cursor < upload.bytes {
            let bytes = CHUNK_BYTES.min(upload.bytes - upload.cursor);
            let address = upload.remote.context("request tail without a WRITE target")?
                .checked_add(upload.cursor as u64).context("request WRITE remote address overflow")?;
            let rkey = self.request_target.context("request tail without a WRITE target")?.rkey;
            (upload.cursor, bytes, Some((address, rkey)))
        } else {
            (0, upload.bytes.min(CHUNK_BYTES), None)
        };
        // SAFETY: the staged slot and any registered egress region stay
        // unmodified until this WR completes (`stage_request` waits for it
        // before restaging; `egress_target` drains sends before a rewrite)
        // or the QP is quiesced; the remote range lies in the worker's
        // exposed slot for this request (checked against the ring geometry
        // at connect), which it does not read until the head SEND arrives.
        unsafe {
            self.endpoint.library.rdma_rc_endpoint_post_request_part(self.endpoint.info.handle, upload.slot_offset,
                upload.prefix_bytes, upload.payload_region, offset, bytes, remote, upload.wr_id)?;
        }
        self.request_sends.posted_part(remote.is_none())?;
        match remote {
            Some(_) => { self.endpoint.traffic.written(bytes); upload.cursor += bytes; }
            None => { self.endpoint.traffic.sent(bytes); upload.sent = true; }
        }
        upload.timing.send_ms += elapsed_ms_optional(started);
        Ok(true)
    }

    /// Posts every part of `upload`, waiting for send-queue room as needed.
    pub(super) fn post_request_parts(&mut self, upload: &mut Upload, timeout: Duration) -> Result<()> {
        while !upload.sent {
            if !self.post_request_part(upload)? { self.wait_request_send(timeout)?; }
        }
        Ok(())
    }

    /// Stages and posts `request` on this session alone.
    pub(super) fn post_request(&mut self, request: &ExpertProtocolV2Request, config: &TcpTransportConfig)
        -> Result<VerbsHostProtocolV2ChunkSubmissionTiming> {
        let mut upload = self.stage_request(request, config)?;
        self.post_request_parts(&mut upload, config.timeout)?;
        Ok(upload.timing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_frames_are_one_send_and_large_frames_end_with_the_head_send() {
        assert_eq!(parts(96).collect::<Vec<_>>(), [(0, 96, false)]);
        assert_eq!(parts(CHUNK_BYTES).collect::<Vec<_>>(), [(0, CHUNK_BYTES, false)]);
        assert_eq!(parts(CHUNK_BYTES + 1).collect::<Vec<_>>(), [(CHUNK_BYTES, 1, true), (0, CHUNK_BYTES, false)]);
        // A full MiMo FP8 4096-row frame: every byte exactly once, WRITEs in
        // ascending order, the header-bearing head last.
        let bytes = 26_509_408;
        let parts: Vec<_> = parts(bytes).collect();
        assert_eq!(parts.len(), bytes.div_ceil(CHUNK_BYTES));
        assert_eq!(*parts.last().unwrap(), (0, CHUNK_BYTES, false));
        let mut next = CHUNK_BYTES;
        for &(offset, len, write) in &parts[..parts.len() - 1] {
            assert!(write && offset == next && (1..=CHUNK_BYTES).contains(&len));
            next += len;
        }
        assert_eq!(next, bytes);
        assert!(!needs_target(CHUNK_BYTES) && needs_target(CHUNK_BYTES + 1));
    }

    #[test]
    fn worker_slot_credit_follows_final_replies_only() -> Result<()> {
        for depth in [1, 2, 3, 8] {
            let mut credits = Credits::default();
            for _ in 0..depth { credits.acquire(depth)?; }
            assert!(credits.acquire(depth).is_err(), "slot n % depth reused before request n - depth replied");
            for _ in 0..4096 {
                credits.final_reply()?;
                credits.acquire(depth)?;
                assert!(credits.acquire(depth).is_err());
            }
            for _ in 0..depth { credits.final_reply()?; }
            assert!(credits.final_reply().is_err());
            credits.acquire(depth)?;
            credits.written_reply_observed();
            assert_eq!(credits.outstanding, 0);
        }
        Ok(())
    }

    #[test]
    fn source_slots_free_only_after_every_part_completes() -> Result<()> {
        let mut queue = SendQueue::default();
        for parts in [102, 1, 3] {
            queue.posted_request()?;
            for part in 0..parts { queue.posted_part(part + 1 == parts)?; }
        }
        assert_eq!((queue.in_flight, queue.requests_in_flight()), (106, 3));
        queue.completed(101)?;
        assert_eq!(queue.requests_in_flight(), 3, "the 102-part request still reads its slot");
        queue.completed(1)?;
        assert_eq!(queue.requests_in_flight(), 2);
        queue.completed(3)?;
        assert_eq!(queue.requests_in_flight(), 1);
        assert!(queue.completed(2).is_err());
        queue.completed(1)?;
        assert_eq!((queue.in_flight, queue.requests_in_flight()), (0, 0));
        // A staged request owns its slot before any part is posted.
        queue.posted_request()?;
        assert_eq!(queue.requests_in_flight(), 1);
        assert!(queue.posted_request().is_err(), "second request staged before the first was sealed");
        queue.posted_part(true)?;
        assert!(queue.posted_part(false).is_err(), "part after the head SEND");
        queue.completed(1)?;
        assert_eq!(queue.requests_in_flight(), 0);
        Ok(())
    }

    /// The serving sequence that failed on GLM K4 max (`request part without
    /// a request`): `LocalTp4Client::post` reaps each rank's CQ between that
    /// rank's part posts, so every earlier WRITE can complete while later
    /// tail chunks are still unposted. Drives several ranks through bootstrap
    /// (one small SEND), decode-sized frames, then N-wave prefill frames with
    /// a CQ that completes everything posted so far before each post.
    #[test]
    fn completions_between_part_posts_keep_the_request_open_until_its_head() -> Result<()> {
        let max_send_wr = 16;
        for depth in [1usize, 2, 3, 8] {
            let mut ranks: Vec<SendQueue> = (0..6).map(|_| SendQueue::default()).collect();
            // (frame bytes) per wave: bootstrap, decode, prefill N waves, decode again.
            let waves = [96, 40_000, 26_509_408, 26_509_408, 26_509_408, 8_000, CHUNK_BYTES + 1];
            for (wave, &bytes) in waves.iter().enumerate() {
                let mut uploads: Vec<std::iter::Peekable<_>> = Vec::new();
                for queue in &mut ranks {
                    // `stage_request`: wait until a source slot is free.
                    while queue.requests_in_flight() >= depth {
                        let all = queue.in_flight;
                        queue.completed(all)?;
                    }
                    queue.posted_request()?;
                    uploads.push(parts(bytes).peekable());
                }
                let mut round = 0;
                while uploads.iter_mut().any(|parts| parts.peek().is_some()) {
                    for (rank, (queue, parts)) in ranks.iter_mut().zip(&mut uploads).enumerate() {
                        // `try_reap_request_sends` before every post: alternate
                        // between completing everything and completing nothing.
                        if (round + rank + wave) % 2 == 0 { let all = queue.in_flight; queue.completed(all)?; }
                        if queue.in_flight >= max_send_wr { continue; }
                        if let Some((_, _, write)) = parts.next() {
                            queue.posted_part(!write)?;
                            assert!(queue.requests_in_flight() >= 1);
                        }
                    }
                    round += 1;
                }
                for queue in &mut ranks {
                    assert!(queue.requests_in_flight() <= depth);
                    assert!(queue.requests.back().is_some_and(|&(_, sealed)| sealed));
                }
            }
            for queue in &mut ranks {
                let all = queue.in_flight;
                queue.completed(all)?;
                assert_eq!((queue.in_flight, queue.requests_in_flight()), (0, 0));
            }
        }
        Ok(())
    }

    /// Explicit hardware gate only. Compiling this test never opens a NIC.
    /// One loopback QP pair carries a WRITE tail plus head SEND into the
    /// receiver's posted slot without draining WRITE completions first.
    #[test]
    #[ignore = "requires approved native RDMA loopback hardware gate"]
    fn native_loopback_tail_writes_then_head_send_land_one_frame() -> Result<()> {
        const CHUNK: usize = CHUNK_BYTES;
        let frame: Vec<u8> = (0..3 * CHUNK + 193).map(|i| (i % 251) as u8).collect();
        let span = frame.len().next_multiple_of(4096);
        let library = load_verbs_host_native_library()?;
        let client = library.rdma_rc_endpoint_create(1, 111, frame.len(), frame.len(), span, span, 8, 8, 2)?;
        let server = match library.rdma_rc_endpoint_create(1, 222, frame.len(), frame.len(), span, span, 8, 8, 2) {
            Ok(server) => server,
            Err(error) => { library.rdma_rc_endpoint_destroy(client.handle)?; return Err(error); }
        };
        let result = (|| -> Result<()> {
            library.rdma_rc_endpoint_connect(client.handle, server.qp_num, server.psn, server.lid, &c_char_array_to_string(&server.gid_hex))?;
            library.rdma_rc_endpoint_connect(server.handle, client.qp_num, client.psn, client.lid, &c_char_array_to_string(&client.gid_hex))?;
            let (address, rkey) = library.rdma_rc_endpoint_request_target(server.handle)?;
            library.rdma_rc_endpoint_post_recv_at(server.handle, 0, frame.len(), 1)?;
            let view = library.rdma_rc_endpoint_buffer_view(client.handle, false)?;
            // SAFETY: no WR exists yet and the frame fits the owned send span.
            unsafe { std::ptr::copy_nonoverlapping(frame.as_ptr(), view.host_ptr.cast::<u8>(), frame.len()); }
            let mut posted = 0;
            for (offset, bytes, write) in parts(frame.len()) {
                // SAFETY: the source span stays untouched until both QPs are
                // destroyed below; the remote range is the server's slot 0.
                unsafe {
                    library.rdma_rc_endpoint_post_request_part(client.handle, 0, 0, None, offset, bytes,
                        write.then_some((address + offset as u64, rkey)), 2)?;
                }
                posted += 1;
            }
            library.rdma_rc_endpoint_poll(server.handle, 0, 1, 1_000_000, 1000)?;
            let mut received = vec![0; frame.len()];
            library.rdma_rc_endpoint_copy_recv_at(server.handle, &mut received, 0, frame.len())?;
            assert_eq!(received, frame);
            library.rdma_rc_endpoint_poll(client.handle, posted, 0, 1_000_000, 1000)?;
            Ok(())
        })();
        // Native destroy stops the QP before registrations/storage are freed.
        let a = library.rdma_rc_endpoint_destroy(client.handle);
        let b = library.rdma_rc_endpoint_destroy(server.handle);
        result.and(a).and(b)
    }
}
