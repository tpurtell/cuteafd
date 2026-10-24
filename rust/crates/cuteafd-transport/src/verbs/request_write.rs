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

/// Send-queue entries for `depth` frames of up to `frame_bytes` each, every
/// frame split into one WR per chunk. Posting a full ring of chunked frames
/// then never waits for a completion: the posting thread is the inference
/// owner, and a queue sized to the ring depth alone would hold it inside
/// `post` for nearly the whole transfer of a large frame.
pub(super) fn send_queue_depth(depth: u32, frame_bytes: usize) -> Result<u32> {
    // Chunks per frame (its head is one of them) plus one spare entry.
    let parts = u32::try_from(frame_bytes.div_ceil(CHUNK_BYTES).max(1) + 1)?;
    depth.checked_mul(parts).context("send queue depth overflow")
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

/// Where the coordinator's reply frame `s` lands: it consumes the `s`-th
/// receive the coordinator posted on this QP. Receives are posted in slot
/// order (initial `0..depth`, then reposts in consumption order, which
/// `RepostOrder` keeps), so receive `s` is host slot `s % depth`; with GPU
/// landing its payload goes to device slot `s % landing.slots` (native
/// rotates one slot per post). Sent once over the control stream after the
/// coordinator has attached landing and posted its initial receives.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct ReplyTarget {
    /// Coordinator response ring (host slots), exposed for remote WRITE.
    pub address: u64,
    pub rkey: u32,
    pub slot_stride_bytes: usize,
    pub depth: usize,
    /// GPU landing: frame bytes past `header_bytes` land here instead.
    pub landing: Option<ReplyLanding>,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct ReplyLanding {
    pub address: u64,
    pub rkey: u32,
    pub slot_bytes: usize,
    pub slots: usize,
    pub header_bytes: usize,
}
impl ReplyTarget {
    pub fn validate(&self, slot_capacity_bytes: usize) -> Result<()> {
        anyhow::ensure!(self.address != 0 && self.depth > 0 && self.slot_stride_bytes >= slot_capacity_bytes,
            "invalid coordinator reply target");
        if let Some(landing) = self.landing {
            anyhow::ensure!(landing.address != 0 && (1..=4).contains(&landing.slots) && landing.slot_bytes > 0
                && (1..CHUNK_BYTES).contains(&landing.header_bytes), "invalid coordinator GPU landing target");
        }
        Ok(())
    }
    /// Remote address and key for reply `sequence`'s frame bytes starting at
    /// `offset` (`offset >= CHUNK_BYTES`, so past any landing header).
    pub fn tail(&self, sequence: u64, offset: usize, bytes: usize, capacity: usize) -> Result<(u64, u32)> {
        anyhow::ensure!(offset >= CHUNK_BYTES && offset.checked_add(bytes).is_some_and(|end| end <= capacity),
            "reply tail outside its frame");
        match self.landing {
            Some(landing) => {
                let slot = (sequence % landing.slots as u64) as usize;
                let at = offset - landing.header_bytes;
                anyhow::ensure!(at + bytes <= landing.slot_bytes, "reply tail exceeds its GPU landing slot");
                Ok((landing.address + (slot * landing.slot_bytes + at) as u64, landing.rkey))
            }
            None => {
                let slot = (sequence % self.depth as u64) as usize;
                Ok((self.address + (slot * self.slot_stride_bytes + offset) as u64, self.rkey))
            }
        }
    }
}

/// Reply frame `sequence` may WRITE into its receive only once that receive is
/// posted: `sequence < posted`, compared on the 32-bit wrapping watermark the
/// coordinator stamps into every request (exact while the gap is < 2^31).
pub(super) fn reply_receive_posted(sequence: u64, watermark: u32) -> bool {
    (watermark.wrapping_sub(sequence as u32) as i32) > 0
}

/// Stamps the coordinator's receives-posted count into a staged request header.
pub(super) fn stamp_watermark(header: &mut [u8], flags: u32, posted: u64) -> Result<()> {
    let at = crate::protocol_v2::request_reply_watermark_offset(flags);
    header.get_mut(at..at + 4).context("request header too short for its watermark")?
        .copy_from_slice(&(posted as u32).to_le_bytes());
    Ok(())
}
pub(super) fn read_watermark(header: &[u8], flags: u32) -> Result<u32> {
    let at = crate::protocol_v2::request_reply_watermark_offset(flags);
    Ok(u32::from_le_bytes(header.get(at..at + 4).context("request header too short for its watermark")?
        .try_into().unwrap()))
}

/// One WR of a reply frame staged at a worker send slot, in post order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReplyPart {
    /// Frame bytes `[offset, offset + bytes)` RDMA-written to `remote`.
    Write { offset: usize, bytes: usize, remote: (u64, u32) },
    /// The head: the frame's first `bytes`, as the SEND that completes the
    /// coordinator's receive (the whole frame when it is not chunked).
    Send { bytes: usize },
}

/// How reply frame `sequence` of `bytes` goes out: tail WRITEs into the
/// receive it will complete, then the head SEND, when it exceeds one chunk and
/// that receive is covered by `watermark`; otherwise one SEND, which RC holds
/// until the receive is posted. Returns whether the frame was chunked.
pub(super) fn reply_parts(target: Option<&ReplyTarget>, watermark: Option<u32>, sequence: u64, bytes: usize,
    capacity: usize) -> Result<(Vec<ReplyPart>, bool)> {
    let covered = watermark.is_some_and(|posted| reply_receive_posted(sequence, posted));
    match target {
        Some(target) if bytes > CHUNK_BYTES && covered => {
            let mut out = Vec::with_capacity(bytes.div_ceil(CHUNK_BYTES));
            for offset in (CHUNK_BYTES..bytes).step_by(CHUNK_BYTES) {
                let len = CHUNK_BYTES.min(bytes - offset);
                out.push(ReplyPart::Write { offset, bytes: len, remote: target.tail(sequence, offset, len, capacity)? });
            }
            out.push(ReplyPart::Send { bytes: CHUNK_BYTES });
            Ok((out, true))
        }
        _ => Ok((vec![ReplyPart::Send { bytes }], false)),
    }
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
    /// Per request, oldest first: WRs posted and not completed, whether its
    /// head SEND is posted, and the registered egress slot its payload reads.
    requests: VecDeque<Pending>,
    pub in_flight: usize,
}
#[derive(Debug, Clone, Copy)]
struct Pending { parts: usize, sealed: bool, egress: Option<usize> }
impl SendQueue {
    pub fn posted_request(&mut self, egress: Option<usize>) -> Result<()> {
        anyhow::ensure!(self.requests.back().is_none_or(|p| p.sealed),
            "request staged before the previous request's head SEND was posted");
        self.requests.push_back(Pending { parts: 0, sealed: false, egress });
        Ok(())
    }
    /// One WR of the newest request; `head` marks its final (head SEND) part.
    pub fn posted_part(&mut self, head: bool) -> Result<()> {
        let pending = self.requests.back_mut().context("request part without a request")?;
        anyhow::ensure!(!pending.sealed, "request part posted after its head SEND");
        pending.parts += 1;
        pending.sealed = head;
        self.in_flight += 1;
        Ok(())
    }
    /// Send CQEs arrive in post order on one QP: they retire the oldest WRs.
    pub fn completed(&mut self, mut count: usize) -> Result<()> {
        anyhow::ensure!(count <= self.in_flight, "{count} request send completions with {} in flight", self.in_flight);
        self.in_flight -= count;
        let mut index = 0;
        while count > 0 {
            let pending = self.requests.get_mut(index).context("request send completion without a request")?;
            let taken = count.min(pending.parts);
            pending.parts -= taken;
            count -= taken;
            index += 1;
        }
        while self.requests.front().is_some_and(|p| p.parts == 0 && p.sealed) {
            self.requests.pop_front();
        }
        Ok(())
    }
    /// Requests that still own their source slot.
    pub fn requests_in_flight(&self) -> usize { self.requests.len() }
    /// Whether some unretired request still reads registered egress `slot`.
    pub fn reads_egress(&self, slot: usize) -> bool {
        self.requests.iter().any(|p| p.egress == Some(slot))
    }
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
        // Every receive posted so far may take a reply tail WRITE. Read after
        // the waits above: a later post only raises it.
        stamp_watermark(head, request.header.flags,
            self.endpoint.recvs_posted.load(std::sync::atomic::Ordering::Relaxed))?;
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
        let egress_slot = payload_region.and_then(|(_, at)| self.egress.as_ref().map(|(buffer, _)| buffer.slot_of(at)));
        self.request_sends.posted_request(egress_slot)?;
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
    fn send_queue_holds_a_full_ring_of_chunked_frames() -> Result<()> {
        assert_eq!(send_queue_depth(8, 96)?, 16);
        assert_eq!(send_queue_depth(8, CHUNK_BYTES + 1)?, 24);
        assert_eq!(send_queue_depth(8, 26_509_408)?, 8 * 103);
        // A full ring of full-capacity MiMo requests (26,509,408 B, 102 WRs
        // each) or worker replies (33,816,672 B, 130 WRs each, the worker QP's
        // send frame) posts in one pass: no part ever finds the queue full.
        for (depth, frame) in [(1u32, 26_509_408usize), (2, 26_509_408), (8, 26_509_408), (8, 33_816_672), (8, 96)] {
            let max = send_queue_depth(depth, frame)? as usize;
            let mut queue = SendQueue::default();
            for _ in 0..depth {
                queue.posted_request(None)?;
                let parts = parts(frame).count();
                for part in 0..parts {
                    assert!(queue.in_flight < max, "posting would wait for a completion");
                    queue.posted_part(part + 1 == parts)?;
                }
            }
            assert!(queue.in_flight < max);
        }
        Ok(())
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
            queue.posted_request(None)?;
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
        queue.posted_request(None)?;
        assert_eq!(queue.requests_in_flight(), 1);
        assert!(queue.posted_request(None).is_err(), "second request staged before the first was sealed");
        queue.posted_part(true)?;
        assert!(queue.posted_part(false).is_err(), "part after the head SEND");
        queue.completed(1)?;
        assert_eq!(queue.requests_in_flight(), 0);
        Ok(())
    }

    /// Rewriting egress slot `w % N` waits only for the requests that read
    /// that slot (wave `w - N`), not for every wave still on the wire.
    #[test]
    fn egress_slot_reuse_waits_only_for_its_own_readers() -> Result<()> {
        let mut queue = SendQueue::default();
        for slot in [0, 1, 2] {
            queue.posted_request(Some(slot))?;
            for part in 0..102 { queue.posted_part(part == 101)?; }
        }
        assert!(queue.reads_egress(0) && queue.reads_egress(2));
        queue.completed(102)?;
        assert!(!queue.reads_egress(0), "slot 0 is free once its own wave completed");
        assert!(queue.reads_egress(1) && queue.reads_egress(2));
        queue.posted_request(None)?; // a decode-sized copied frame reads no egress
        queue.posted_part(true)?;
        assert!(!queue.reads_egress(0));
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
                    queue.posted_request(None)?;
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
                    assert!(queue.requests.back().is_some_and(|p| p.sealed));
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

    fn host_target(depth: usize) -> ReplyTarget {
        ReplyTarget { address: 1 << 40, rkey: 7, slot_stride_bytes: 64 << 20, depth, landing: None }
    }

    #[test]
    fn small_replies_stay_one_send_and_large_covered_replies_end_with_the_head() -> Result<()> {
        let target = host_target(8);
        let capacity = 50 << 20;
        // At or under one chunk: one SEND everywhere (decode, verify).
        for bytes in [96, 40_000, CHUNK_BYTES] {
            assert_eq!(reply_parts(Some(&target), Some(100), 3, bytes, capacity)?,
                (vec![ReplyPart::Send { bytes }], false));
        }
        // A 33,816,672-byte MiMo N3 reply into receive 11 (slot 3, covered).
        let bytes = 33_816_672;
        let (parts, chunked) = reply_parts(Some(&target), Some(12), 11, bytes, capacity)?;
        assert!(chunked);
        assert_eq!(parts.len(), bytes.div_ceil(CHUNK_BYTES));
        assert_eq!(*parts.last().unwrap(), ReplyPart::Send { bytes: CHUNK_BYTES });
        let mut next = CHUNK_BYTES;
        for part in &parts[..parts.len() - 1] {
            let ReplyPart::Write { offset, bytes: len, remote } = *part else { panic!("head before tail") };
            assert_eq!(offset, next);
            assert_eq!(remote, (target.address + (3 * target.slot_stride_bytes + offset) as u64, 7));
            next += len;
        }
        assert_eq!(next, bytes);
        // No target (write mode or small capacity): one SEND.
        assert_eq!(reply_parts(None, Some(12), 11, bytes, capacity)?.1, false);
        Ok(())
    }

    #[test]
    fn uncovered_receives_fall_back_to_one_send_and_the_watermark_wraps() -> Result<()> {
        let target = host_target(8);
        let bytes = 3 * CHUNK_BYTES;
        // Receive 12 is not posted yet at watermark 12: RC SEND waits for it.
        assert_eq!(reply_parts(Some(&target), Some(12), 12, bytes, 64 << 20)?, (vec![ReplyPart::Send { bytes }], false));
        assert_eq!(reply_parts(Some(&target), None, 0, bytes, 64 << 20)?.1, false);
        assert!(reply_parts(Some(&target), Some(13), 12, bytes, 64 << 20)?.1);
        // 32-bit wrap: the sequence just below 2^32 is covered by a small watermark.
        assert!(reply_receive_posted(u32::MAX as u64, 3));
        assert!(reply_receive_posted((1u64 << 32) + 2, 3));
        assert!(!reply_receive_posted((1u64 << 32) + 3, 3));
        let mut header = vec![0u8; 128];
        for flags in [0, crate::EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM] {
            stamp_watermark(&mut header, flags, (1u64 << 32) + 5)?;
            assert_eq!(read_watermark(&header, flags)?, 5);
        }
        Ok(())
    }

    #[test]
    fn gpu_landing_tails_skip_the_host_header_and_rotate_landing_slots() -> Result<()> {
        let landing = ReplyLanding { address: 1 << 41, rkey: 9, slot_bytes: 4096 * 6144 * 2, slots: 3, header_bytes: 96 };
        let target = ReplyTarget { landing: Some(landing), ..host_target(8) };
        target.validate(64 << 20)?;
        let capacity = 96 + landing.slot_bytes;
        // Receive 4 lands in device slot 1; frame offset o goes to o - 96 there.
        assert_eq!(target.tail(4, CHUNK_BYTES, CHUNK_BYTES, capacity)?,
            (landing.address + (landing.slot_bytes + CHUNK_BYTES - 96) as u64, 9));
        // The last byte of a full plane is the last byte of its device slot.
        assert!(target.tail(5, capacity - 1, 1, capacity).is_ok());
        assert!(target.tail(5, capacity - 1, 2, capacity).is_err());
        assert!(target.tail(5, CHUNK_BYTES - 1, 1, capacity).is_err(), "header bytes are never written");
        assert!(ReplyTarget { landing: Some(ReplyLanding { slots: 5, ..landing }), ..target }.validate(64 << 20).is_err());
        Ok(())
    }

    /// A post that fails mid-frame (tail WRITE posted, head not) leaves the
    /// frame unsealed: its slot stays owned through any completions, and the
    /// next frame is refused, so the connection drops and quiesces the QP
    /// before the slot can be restaged under a live WRITE.
    #[test]
    fn failed_mid_frame_post_keeps_the_slot_and_refuses_the_next_frame() -> Result<()> {
        let mut sends = SendQueue::default();
        sends.posted_request(None)?;
        sends.posted_part(false)?;
        sends.posted_part(false)?; // the head SEND then fails to post
        sends.completed(2)?;
        assert_eq!(sends.requests_in_flight(), 1, "an unsealed frame released its slot");
        assert!(sends.posted_request(None).is_err());
        Ok(())
    }

    /// N deferred requests: each final reply consumes the next coordinator
    /// receive in emit order; with the coordinator reposting every consumed
    /// receive before its next dispatch, every reply frame of every request is
    /// covered and chunks, multi-frame replies included.
    #[test]
    fn deferred_and_multi_frame_replies_stay_covered_by_dispatch_watermarks() -> Result<()> {
        for (depth, deferred, frames) in [(8usize, 1usize, 1usize), (8, 3, 1), (8, 2, 3), (2, 2, 1)] {
            let target = host_target(depth);
            let mut posted = depth as u64; // initial receives
            let mut consumed = 0u64;
            let mut sent = 0u64;
            let mut watermark = None;
            for _wave in 0..64 {
                // Coordinator dispatches `deferred` requests; each stamps P.
                for _ in 0..deferred { watermark = Some(posted as u32); }
                // Worker replies: `frames` frames per request.
                for _ in 0..deferred * frames {
                    let (parts, chunked) = reply_parts(Some(&target), watermark, sent, 4 * CHUNK_BYTES, 64 << 20)?;
                    assert!(sent < posted, "a frame would consume an unposted receive");
                    assert_eq!(chunked, sent < watermark.unwrap() as u64);
                    // Every frame fits the ring here, so every frame chunks.
                    assert!(chunked || deferred * frames > depth, "depth {depth}: frame {sent} sent whole");
                    assert!(matches!(parts.last(), Some(ReplyPart::Send { .. })));
                    sent += 1;
                }
                // Coordinator consumes and reposts each received frame.
                while consumed < sent { consumed += 1; posted += 1; }
            }
            assert!(sent > 0);
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

    /// Echoes the request's hidden rows in `frames` row-indexed host frames.
    struct ChunkedEcho { frames: u32 }
    impl crate::synthetic::ProtocolV2ExpertExecutor for ChunkedEcho {
        fn name(&self) -> &'static str { "chunked-echo" }
        fn execute(&self, _: &ExpertProtocolV2RequestView<'_>) -> Result<crate::ExpertProtocolV2Response> {
            bail!("chunked echo streams")
        }
        fn execute_streaming(&self, request: &ExpertProtocolV2RequestView<'_>,
            emit: &mut dyn FnMut(crate::ExpertProtocolV2ResponseRef<'_>) -> Result<()>) -> Result<()> {
            let h = &request.header;
            let stride = h.hidden_row_stride_bytes as usize;
            let rows = h.row_count;
            let per = rows.div_ceil(self.frames).max(1);
            let indices: Vec<u32> = (0..rows).collect();
            for first in (0..rows).step_by(per as usize) {
                let end = (first + per).min(rows);
                let payload = &request.hidden_payload()[first as usize * stride..end as usize * stride];
                let response = crate::ExpertProtocolV2ResponseRef::new_with_output_stride(h.request_id,
                    h.placement_version, h.layer_id, end - first, h.hidden_dim, h.hidden_dtype, stride as u32,
                    crate::ExpertProtocolV2Status::Ok, payload)?.with_executor_name(self.name());
                let response = if self.frames > 1 {
                    response.with_row_indices(&indices[first as usize..end as usize], end < rows)?
                } else { response };
                emit(response)?;
            }
            Ok(())
        }
    }

    fn bf16_request(id: u64, rows: u32, hidden: u32) -> Result<ExpertProtocolV2Request> {
        let payload: Vec<u8> = (0..rows as usize * hidden as usize * 2).map(|i| ((i * 7 + id as usize) % 253) as u8).collect();
        Ok(ExpertProtocolV2Request::new(id, 1, 3, hidden, crate::ExpertV2Dtype::Bf16,
            (0..rows).map(|row| crate::ExpertProtocolV2RowDescriptor { row_id: row.into(),
                source_kind: crate::ExpertV2SourceKind::Prefill, source_request_id: id, token_position: row.into(),
                route_offset: row, route_count: 1 }).collect(),
            (0..rows).map(|row| crate::ExpertProtocolV2RouteEntry { row_index: row, expert_id: 0, gate_weight: 1.0 })
                .collect(), payload)?)
    }

    /// Explicit hardware gate only (golden-run.py --rdma --native-test).
    /// A real worker (`serve_protocol_v2_verbs_host_with_executor`) and a real
    /// coordinator transport on one host: request tails, watermarks and reply
    /// tails end to end. `CUTEAFD_LOOPBACK_ADDR` is a local RoCE address.
    /// Pinned-host intake: 1/2/3-wave dispatch, single- and four-frame
    /// replies, small (SEND-only) and large (chunked) frames; GPU landing
    /// (`CUTEAFD_LOOPBACK_GPU=1`): single-frame replies into a device plane.
    #[test]
    #[ignore = "requires approved native RDMA loopback hardware gate"]
    fn native_loopback_reply_tails_land_byte_exact_on_pinned_and_gpu_intake() -> Result<()> {
        let ip = std::env::var("CUTEAFD_LOOPBACK_ADDR").context("set CUTEAFD_LOOPBACK_ADDR to a local RoCE address")?;
        let gpu = std::env::var("CUTEAFD_LOOPBACK_GPU").as_deref() == Ok("1");
        let hidden = 6144u32;
        let capacity = 2048u32;
        for frames in if gpu { vec![1] } else { vec![1, 4] } {
            let listener = std::net::TcpListener::bind(format!("{ip}:0"))?;
            let addr = listener.local_addr()?;
            drop(listener);
            let executor: Arc<dyn crate::synthetic::ProtocolV2ExpertExecutor> = Arc::new(ChunkedEcho { frames });
            std::thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
                let _ = runtime.block_on(super::super::serve_protocol_v2_verbs_host_with_executor(&addr.to_string(), executor));
            });
            std::thread::sleep(Duration::from_millis(300));
            for waves in [1usize, 2, 3] {
                let library = load_verbs_host_native_library()?;
                let mut client = super::super::LocalTp4Client::new_ranks(vec![addr], TcpTransportConfig {
                    timing: false, timeout: Duration::from_secs(30), max_frame_bytes: 64 << 20 });
                client.set_max_waves(waves)?;
                let plane_bytes = capacity as usize * hidden as usize * 2;
                let mut plane = None;
                if gpu {
                    let buffer = library.alloc_device_buffer(plane_bytes * waves)?;
                    client.set_landing(vec![Some(DeviceLanding::with_slots(buffer, waves)?)])?;
                    plane = Some(buffer);
                }
                let result = (|| -> Result<()> {
                    // No `bootstrap`: it requires a placement worker's installed
                    // grouping, which this generic echo server does not report.
                    // The first dispatch is full-capacity, so it sizes the rings
                    // for every later wave, as serving's warm-up does.
                    let written_before = crate::traffic::written_total();
                    let mut large = 0usize;
                    let mut request_tails = 0u64;
                    // Small (one SEND each way), then large frames: 21 rows is
                    // 258,048 payload bytes, just under one chunk with headers.
                    let sizes = [2048u32, 1, 21, 64, 1500, 2048, 7, 2048];
                    let mut sent = std::collections::VecDeque::new();
                    for (index, &rows) in sizes.iter().cycle().take(sizes.len() * 3).enumerate() {
                        while client.open_waves() == waves {
                            let (request, slot): (ExpertProtocolV2Request, usize) = sent.pop_front().unwrap();
                            let mut got = vec![0u8; request.hidden_payload.len()];
                            while !client.poll(|chunk| {
                                if chunk.landed { return Ok(()); }
                                let first = chunk.row_indices.as_ref().map_or(0, |rows| rows[0]) as usize;
                                let at = first * hidden as usize * 2;
                                got[at..at + chunk.partial_output_payload.as_ref().len()]
                                    .copy_from_slice(chunk.partial_output_payload.as_ref());
                                Ok(())
                            })? {}
                            if let Some(buffer) = plane {
                                let view = CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>()
                                    .add(slot * plane_bytes) }.cast(), bytes: got.len(), ..buffer };
                                library.copy_d2h(&mut got, view)?;
                            }
                            anyhow::ensure!(got == request.hidden_payload.as_ref(), "request {} echo differs",
                                request.header.request_id);
                        }
                        let request = bf16_request(index as u64 + 1, rows, hidden)?;
                        let wire = request.wire_stats().wire_bytes;
                        if wire > CHUNK_BYTES { large += 1; request_tails += (wire - CHUNK_BYTES) as u64; }
                        client.dispatch(&request)?;
                        sent.push_back((request, index % waves));
                    }
                    while let Some((request, slot)) = sent.pop_front() {
                        let mut got = vec![0u8; request.hidden_payload.len()];
                        while !client.poll(|chunk| {
                            if chunk.landed { return Ok(()); }
                            let first = chunk.row_indices.as_ref().map_or(0, |rows| rows[0]) as usize;
                            let at = first * hidden as usize * 2;
                            got[at..at + chunk.partial_output_payload.as_ref().len()]
                                .copy_from_slice(chunk.partial_output_payload.as_ref());
                            Ok(())
                        })? {}
                        if let Some(buffer) = plane {
                            let view = CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>()
                                .add(slot * plane_bytes) }.cast(), bytes: got.len(), ..buffer };
                            library.copy_d2h(&mut got, view)?;
                        }
                        anyhow::ensure!(got == request.hidden_payload.as_ref(), "request {} echo differs",
                            request.header.request_id);
                    }
                    let written = crate::traffic::written_total() - written_before;
                    // Request tails are exact; everything above them is reply tails.
                    let reply_tails = written.checked_sub(request_tails).context("request tails not all written")?;
                    anyhow::ensure!(large > 0 && reply_tails > 0, "no reply tail was written");
                    eprintln!("loopback gpu={gpu} frames={frames} waves={waves}: {large} large requests, \
                        request tails {request_tails} B, reply tails {reply_tails} B");
                    Ok(())
                })();
                client.reset();
                if let Some(mut buffer) = plane { library.free_device_buffer(&mut buffer)?; }
                result?;
            }
        }
        Ok(())
    }
}
