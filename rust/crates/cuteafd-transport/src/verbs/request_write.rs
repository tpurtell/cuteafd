//! Request-only WRITE -> same-QP SEND. WRITE CQEs return byte-window capacity,
//! never receiver leases. Only a validated final reply returns a lease: the
//! worker posts it after execute has drained consumers and reposted the WQE.
use super::*;

pub(super) const DOORBELL_BYTES: usize = 40;

pub(super) fn next_epoch() -> Result<u64> {
    static EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos())?;
    let previous = EPOCH.fetch_update(Ordering::AcqRel, Ordering::Acquire,
        |last| last.checked_add(1).map(|next| next.max(now))).map_err(|_| anyhow::anyhow!("request WRITE epoch exhausted"))?;
    Ok(now.max(previous + 1))
}

/// The existing final reply becomes a credit only after this transition. An
/// execute error or failed repost leaves it unpublishable, including when the
/// callback had already prepared a final response.
#[derive(Default)]
pub(super) struct FinalCredit {
    ready: bool,
    drained: bool,
    reposted: bool,
}
impl FinalCredit {
    pub fn ready(&mut self) -> Result<()> {
        anyhow::ensure!(!self.ready, "duplicate final request WRITE response");
        self.ready = true;
        Ok(())
    }
    pub fn drained(&mut self) { self.drained = true; }
    pub fn reposted(&mut self) { self.reposted = true; }
    pub fn publish(&self) -> Result<()> {
        anyhow::ensure!(self.ready && self.drained && self.reposted,
            "request WRITE final credit precedes consumer drain or receive repost");
        Ok(())
    }
}
const MAGIC: u64 = 0x3151455257444643;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Settings {
    pub chunk_bytes: usize,
    pub destination_bytes: Option<usize>,
    pub aggregate_bytes: Option<usize>,
}
impl Default for Settings {
    fn default() -> Self {
        Self { chunk_bytes: 256 << 10, destination_bytes: None, aggregate_bytes: None }
    }
}
impl Settings {
    pub fn validate(self) -> Result<Self> {
        anyhow::ensure!(self.chunk_bytes > 0 && self.chunk_bytes <= u32::MAX as usize
            && self.destination_bytes.is_none_or(|bytes| bytes >= self.chunk_bytes)
            && self.aggregate_bytes.is_none_or(|bytes| bytes >= self.chunk_bytes),
            "request WRITE needs 0 < chunk <= destination window and aggregate window");
        Ok(self)
    }
    pub fn from_env() -> Result<Option<Self>> {
        match std::env::var("CUTEAFD_SPARK_REQUEST_WRITE").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("off") => return Ok(None),
            Ok("on") => {},
            _ => bail!("CUTEAFD_SPARK_REQUEST_WRITE must be on or off"),
        }
        let mut value = Self::default();
        if let Ok(raw) = std::env::var("CUTEAFD_SPARK_REQUEST_CHUNK_BYTES") {
            value.chunk_bytes = raw.parse().context("invalid request WRITE chunk bytes")?;
        }
        for (name, target) in [
            ("CUTEAFD_SPARK_REQUEST_DESTINATION_BYTES", &mut value.destination_bytes),
            ("CUTEAFD_SPARK_REQUEST_AGGREGATE_BYTES", &mut value.aggregate_bytes),
        ] {
            if let Ok(raw) = std::env::var(name) {
                *target = if raw == "off" { None } else { Some(raw.parse().with_context(|| format!("invalid {name}"))?) };
            }
        }
        Ok(Some(value.validate()?))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct Target {
    pub address: u64,
    pub rkey: u32,
    pub epoch: u64,
}

#[derive(Debug)]
pub(super) struct Window {
    settings: Settings,
    destinations: Vec<usize>,
    aggregate: usize,
    poisoned: bool,
}
impl Window {
    pub fn new(settings: Settings, ranks: usize) -> Self {
        Self { settings, destinations: vec![0; ranks], aggregate: 0, poisoned: false }
    }
    pub fn reserve(&mut self, rank: usize, bytes: usize) -> Result<bool> {
        anyhow::ensure!(!self.poisoned, "request WRITE window poisoned; quiesce required");
        anyhow::ensure!(bytes > 0 && bytes <= self.settings.chunk_bytes, "invalid WRITE chunk");
        if self.settings.destination_bytes.is_some_and(|cap| bytes > cap - self.destinations[rank])
            || self.settings.aggregate_bytes.is_some_and(|cap| bytes > cap - self.aggregate) { return Ok(false); }
        self.destinations[rank] += bytes;
        self.aggregate += bytes;
        Ok(true)
    }
    pub fn completed(&mut self, rank: usize, bytes: usize) -> Result<()> {
        anyhow::ensure!(bytes <= self.destinations[rank] && bytes <= self.aggregate, "WRITE completion accounting underflow");
        self.destinations[rank] -= bytes;
        self.aggregate -= bytes;
        Ok(())
    }
    pub fn poison(&mut self) { self.poisoned = true; }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Lease { sequence: u64, request: u64 }
#[derive(Debug)]
pub(super) struct Leases {
    slots: Vec<Option<Lease>>,
    pending: VecDeque<Lease>,
    next: u64,
    poisoned: bool,
}
impl Leases {
    pub fn new(depth: usize) -> Self { Self { slots: vec![None; depth], pending: VecDeque::new(), next: 0, poisoned: false } }
    pub fn acquire(&mut self, request: u64) -> Result<u64> {
        anyhow::ensure!(!self.poisoned, "request lease poisoned; quiesce required");
        let sequence = self.next;
        let slot = sequence as usize % self.slots.len();
        anyhow::ensure!(self.slots[slot].is_none(), "request WRITE slot still leased until final reply");
        self.next = self.next.checked_add(1).context("request WRITE generation exhausted")?;
        let lease = Lease { sequence, request };
        self.slots[slot] = Some(lease);
        self.pending.push_back(lease);
        Ok(sequence)
    }
    pub fn final_reply(&mut self, request: u64) -> Result<()> {
        anyhow::ensure!(!self.poisoned, "request lease poisoned; quiesce required");
        let lease = *self.pending.front().context("final reply without request lease")?;
        anyhow::ensure!(lease.request == request, "out-of-order request WRITE final reply");
        let slot = lease.sequence as usize % self.slots.len();
        anyhow::ensure!(self.slots[slot] == Some(lease), "request WRITE generation mismatch");
        self.slots[slot] = None;
        self.pending.pop_front();
        Ok(())
    }
    pub fn poison(&mut self) { self.poisoned = true; }
}

pub(super) fn doorbell(epoch: u64, sequence: u64, request: u64, bytes: usize) -> [u8; DOORBELL_BYTES] {
    let mut out = [0; DOORBELL_BYTES];
    for (dst, value) in out.chunks_exact_mut(8).zip([MAGIC, epoch, sequence, request, bytes as u64]) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    out
}
pub(super) fn validate_doorbell(raw: &[u8; DOORBELL_BYTES], epoch: u64, sequence: u64, capacity: usize) -> Result<(u64, usize)> {
    let mut words = raw.chunks_exact(8).map(|v| u64::from_le_bytes(v.try_into().unwrap()));
    anyhow::ensure!(words.next() == Some(MAGIC) && words.next() == Some(epoch) && words.next() == Some(sequence),
        "stale or out-of-order request WRITE doorbell");
    let request = words.next().unwrap();
    let bytes = usize::try_from(words.next().unwrap()).context("request WRITE length overflow")?;
    anyhow::ensure!((EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN..=capacity).contains(&bytes), "request WRITE doorbell length exceeds slot");
    Ok((request, bytes))
}

pub(super) struct State {
    pub settings: Settings,
    pub target: Target,
    pub rank: usize,
    pub window: Arc<Mutex<Window>>,
    pub leases: Leases,
    /// Every new WR signals. Same-QP CQ order gives exact WRITE byte credits;
    /// a zero entry is the single SEND doorbell (no payload byte credit).
    pub completions: VecDeque<usize>,
}
impl State {
    pub fn complete(&mut self, count: usize) -> Result<()> {
        for _ in 0..count {
            let bytes = self.completions.pop_front().context("request WRITE unexpected CQE")?;
            if bytes != 0 { self.window.lock().unwrap().completed(self.rank, bytes)?; }
        }
        Ok(())
    }
    pub fn poison(&mut self) {
        self.leases.poison();
        self.window.lock().unwrap().poison();
    }
}

pub(super) struct Upload {
    pub sequence: u64,
    pub prefix_bytes: usize,
    pub payload_region: Option<(u32, usize)>,
    pub bytes: usize,
    pub posted: usize,
    pub request: u64,
    pub timing: VerbsHostProtocolV2ChunkSubmissionTiming,
}

impl VerbsHostProtocolV2PersistentClientSession {
    pub(super) fn prepare_written_request(&mut self, request: &ExpertProtocolV2Request, config: &TcpTransportConfig) -> Result<Upload> {
        if let Some(lease) = &self.endpoint.credits.endpoint { lease.validate_request(&request.header)?; }
        if let Some(ring) = &mut self.retained_response_ring { ring.reclaim_before_request(&self.endpoint, self.response_ring)?; }
        let total_started = config.timing.then(Instant::now);
        let prefix = self.request_frame.encode_request_prefix(request)?;
        let bytes = prefix.len().checked_add(request.hidden_payload.len()).context("request WRITE length overflow")?;
        anyhow::ensure!(bytes <= self.request_capacity_wire_bytes && bytes <= config.max_frame_bytes
            && verbs_host_expected_response_wire_bytes(request)? <= self.response_capacity_wire_bytes,
            "request WRITE exceeds prewarmed geometry");
        let state = self.request_write.as_mut().context("request WRITE not negotiated")?;
        let sequence = state.leases.acquire(request.header.request_id)?;
        let offset = self.request_ring.slot_offset(sequence as usize);
        let view = self.endpoint.send_buffer_view()?;
        anyhow::ensure!(offset <= view.bytes && bytes <= view.bytes - offset, "request WRITE source outside ring");
        let payload_region = self.egress.as_ref().and_then(|(buffer, region)| buffer.offset_of(&request.hidden_payload).map(|offset| (*region, offset)))
            .filter(|_| !request.hidden_payload.is_empty());
        // SAFETY: acquired a fresh lease for this ring slot. No earlier source
        // WR remains: final response follows the request's SEND and all WRITEs.
        let storage = unsafe { std::slice::from_raw_parts_mut(view.host_ptr.cast::<u8>().add(offset), bytes) };
        storage[..prefix.len()].copy_from_slice(prefix);
        if payload_region.is_none() { copy_threads(&mut storage[prefix.len()..], &request.hidden_payload); }
        Ok(Upload { sequence, prefix_bytes: prefix.len(), payload_region, bytes, posted: 0,
            request: request.header.request_id, timing: VerbsHostProtocolV2ChunkSubmissionTiming {
                total_started, encode_ms: elapsed_ms_optional(total_started), expected_response_ms: 0.0, send_ms: 0.0,
            } })
    }

    /// One WR per call permits round-robin posts across rank QPs. A failed post
    /// keeps its reserved bytes and lease until all QPs quiesce; never retry it.
    pub(super) fn post_written_chunk(&mut self, upload: &mut Upload) -> Result<bool> {
        let state = self.request_write.as_mut().context("request WRITE not negotiated")?;
        if self.request_send_in_flight >= self.endpoint.info.max_send_wr as usize { return Ok(false); }
        let offset = self.request_ring.slot_offset(upload.sequence as usize);
        let slot = upload.sequence as usize % self.request_ring.depth;
        if upload.posted == upload.bytes {
            let bell = doorbell(state.target.epoch, upload.sequence, upload.request, upload.bytes);
            if let Err(error) = self.endpoint.library.rdma_rc_endpoint_request_doorbell(self.endpoint.info.handle, slot, &bell,
                VERBS_HOST_SEND_WR_ID + slot as u64) { state.poison(); return Err(error); }
            state.completions.push_back(0);
            self.request_send_in_flight += 1;
            self.request_send_sequence = self.request_send_sequence.checked_add(1).context("request WRITE sequence exhausted")?;
            return Ok(true);
        }
        let bytes = state.settings.chunk_bytes.min(upload.bytes - upload.posted);
        // Keep room for the doorbell before posting the final WRITE. It must
        // follow immediately, not wait for a WRITE completion to free a WR.
        if bytes == upload.bytes - upload.posted
            && self.endpoint.info.max_send_wr as usize - self.request_send_in_flight < 2 {
            return Ok(false);
        }
        if !state.window.lock().unwrap().reserve(state.rank, bytes)? { return Ok(false); }
        let remote = state.target.address.checked_add((offset + upload.posted) as u64).context("request WRITE remote address overflow")?;
        let result = self.endpoint.library.rdma_rc_endpoint_request_write(self.endpoint.info.handle, offset,
            upload.prefix_bytes, upload.payload_region, upload.posted, bytes, remote, state.target.rkey,
            VERBS_HOST_SEND_WR_ID + slot as u64);
        if let Err(error) = result { state.poison(); return Err(error); }
        state.completions.push_back(bytes);
        self.request_send_in_flight += 1;
        upload.posted += bytes;
        // Post the SEND immediately after the final WRITE, without a CQ drain.
        if upload.posted == upload.bytes && self.request_send_in_flight < self.endpoint.info.max_send_wr as usize {
            return self.post_written_chunk(upload);
        }
        Ok(false)
    }

    pub(super) fn request_write_final_reply(&mut self, request: u64) -> Result<()> {
        if self.request_write.is_some() {
            // The same-QP SEND has reached the worker before it can reply, so
            // earlier WRs no longer read their sources. Unreaped CQEs retain
            // byte-window credit, not a remote-slot lease.
            self.request_write.as_mut().unwrap().leases.final_reply(request)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generation_credit_requires_final_reply_not_writes_or_doorbell() -> Result<()> {
        let mut leases = Leases::new(2);
        assert_eq!(leases.acquire(7)?, 0);
        assert_eq!(leases.acquire(8)?, 1);
        assert!(leases.acquire(9).is_err());
        assert!(leases.final_reply(8).is_err());
        leases.final_reply(7)?;
        assert_eq!(leases.acquire(9)?, 2);
        assert!(leases.final_reply(7).is_err());
        Ok(())
    }
    #[test]
    fn caps_are_shared_between_waves_and_destinations() -> Result<()> {
        let settings = Settings { chunk_bytes: 4, destination_bytes: Some(8), aggregate_bytes: Some(12) };
        let mut window = Window::new(settings, 2);
        assert!(window.reserve(0, 4)?); // wave 1
        assert!(window.reserve(0, 4)?); // wave 2
        assert!(!window.reserve(0, 1)?);
        assert!(window.reserve(1, 4)?);
        assert!(!window.reserve(1, 1)?);
        window.completed(0, 4)?;
        assert!(window.reserve(1, 4)?);
        assert_eq!(window.aggregate, 12);
        Ok(())
    }
    #[test]
    fn windows_default_off_and_each_limit_is_independent() -> Result<()> {
        let settings = Settings::default().validate()?;
        assert_eq!(settings.destination_bytes, None);
        assert_eq!(settings.aggregate_bytes, None);
        let mut window = Window::new(settings, 6);
        for _wave in 0..4 {
            for rank in 0..6 {
                for _chunk in 0..32 { assert!(window.reserve(rank, settings.chunk_bytes)?); }
            }
        }
        for (destination_bytes, aggregate_bytes) in [(Some(8), None), (None, Some(8))] {
            let settings = Settings { chunk_bytes: 4, destination_bytes, aggregate_bytes }.validate()?;
            let mut window = Window::new(settings, 2);
            assert!(window.reserve(0, 4)?);
            assert!(window.reserve(0, 4)?);
            assert!(!window.reserve(0, 1)?);
            assert_eq!(window.reserve(1, 4)?, aggregate_bytes.is_none());
        }
        assert!(Settings { destination_bytes: Some(0), ..settings }.validate().is_err());
        Ok(())
    }
    #[test]
    fn final_credit_requires_drain_and_successful_repost() -> Result<()> {
        assert!(next_epoch()? < next_epoch()?);
        let mut credit = FinalCredit::default();
        assert!(credit.publish().is_err());
        credit.ready()?;
        assert!(credit.ready().is_err());
        assert!(credit.publish().is_err()); // execute failed or cancelled
        credit.drained();
        assert!(credit.publish().is_err()); // receive repost failed
        credit.reposted();
        credit.publish()?;
        Ok(())
    }
    #[test]
    fn failed_post_poison_retains_window_and_lease_until_quiesce() -> Result<()> {
        let window = Arc::new(Mutex::new(Window::new(Settings::default(), 1)));
        let mut state = State { settings: Settings::default(), target: Target { address: 1, rkey: 2, epoch: 3 },
            rank: 0, window: window.clone(), leases: Leases::new(2), completions: VecDeque::new() };
        state.leases.acquire(9)?;
        assert!(window.lock().unwrap().reserve(0, 100)?);
        state.poison();
        assert!(state.leases.acquire(10).is_err());
        assert!(state.leases.final_reply(9).is_err());
        assert!(window.lock().unwrap().reserve(0, 1).is_err());
        assert_eq!(window.lock().unwrap().aggregate, 100);
        assert!(state.leases.slots[0].is_some());
        Ok(())
    }
    #[test]
    fn completion_fifo_never_releases_generation_credit() -> Result<()> {
        let settings = Settings { chunk_bytes: 4, destination_bytes: Some(8), aggregate_bytes: Some(12) };
        let window = Arc::new(Mutex::new(Window::new(settings, 2)));
        let mut state = State { settings, target: Target { address: 1, rkey: 2, epoch: 3 }, rank: 0,
            window: window.clone(), leases: Leases::new(2), completions: VecDeque::new() };
        for request in [10, 11] {
            state.leases.acquire(request)?;
            assert!(window.lock().unwrap().reserve(0, 4)?);
            state.completions.extend([4, 0]); // one WRITE, then its SEND
        }
        state.complete(3)?;
        assert_eq!(window.lock().unwrap().aggregate, 0);
        assert!(state.leases.acquire(12).is_err());
        state.leases.final_reply(10)?;
        assert_eq!(state.leases.acquire(12)?, 2);
        state.complete(1)?; // old SEND CQE cannot return the reused generation
        assert!(state.leases.acquire(13).is_err());
        assert!(state.complete(1).is_err());
        Ok(())
    }
    #[test]
    fn mocked_loopback_full_frames_rotate_and_cancellation_requires_quiesce() -> Result<()> {
        for depth in [2, 3] {
            let settings = Settings { chunk_bytes: 37, destination_bytes: Some(74), aggregate_bytes: Some(111) };
            let mut window = Window::new(settings, 3);
            let mut leases: Vec<_> = (0..3).map(|_| Leases::new(depth)).collect();
            let mut slots = vec![vec![vec![0u8; 193]; depth]; 3];
            for wave in 0..32u64 {
                let frame: Vec<u8> = (0..193).map(|i| (i as u64 + wave) as u8).collect();
                let mut sequences = Vec::new();
                for lease in &mut leases { sequences.push(lease.acquire(wave)?); }
                let mut events = Vec::new();
                for offset in (0..frame.len()).step_by(settings.chunk_bytes) {
                    for rank in 0..3 {
                        let end = (offset + settings.chunk_bytes).min(frame.len());
                        assert!(window.reserve(rank, end - offset)?);
                        slots[rank][sequences[rank] as usize % depth][offset..end].copy_from_slice(&frame[offset..end]);
                        events.push((rank, end - offset));
                    }
                    // Mock CQEs free bytes only, never request leases.
                    for (rank, bytes) in events.drain(..) { window.completed(rank, bytes)?; }
                    assert!(leases.iter().all(|l| l.pending.len() == 1));
                }
                for rank in 0..3 {
                    let sequence = sequences[rank];
                    let raw = doorbell(44, sequence, wave, frame.len());
                    assert_eq!(validate_doorbell(&raw, 44, wave, frame.len())?, (wave, frame.len()));
                    assert_eq!(slots[rank][sequence as usize % depth], frame);
                    // Worker callback may emit chunks before its tail finishes.
                    let mut credit = FinalCredit::default();
                    credit.ready()?;
                    assert!(credit.publish().is_err());
                    credit.drained();
                    assert!(credit.publish().is_err());
                    credit.reposted();
                    credit.publish()?;
                    leases[rank].final_reply(wave)?;
                }
                assert_eq!(window.aggregate, 0);
            }
            // Cancellation does not grant a generation. Quiescence is the
            // only way to retire poisoned state, not another acquire/reset.
            leases[0].acquire(99)?;
            leases[0].poison();
            assert!(leases[0].acquire(100).is_err());
            assert!(leases[0].slots.iter().any(Option::is_some));
        }
        Ok(())
    }

    /// Explicit hardware gate only. Compiling this test never opens a NIC;
    /// running it requires parent approval and the normal fair hardware lock.
    #[test]
    #[ignore = "requires approved native RDMA loopback hardware gate"]
    fn native_request_write_loopback_complete_frame_and_one_doorbell() -> Result<()> {
        let library = load_verbs_host_native_library()?;
        let client = library.rdma_rc_endpoint_create(1, 111, 4096, 4096, 8192, 8192, 8, 8, 2)?;
        let server = match library.rdma_rc_endpoint_create(1, 222, 4096, 4096, 8192, 8192, 8, 8, 2) {
            Ok(server) => server,
            Err(error) => { library.rdma_rc_endpoint_destroy(client.handle)?; return Err(error); }
        };
        let result = (|| -> Result<()> {
            library.rdma_rc_endpoint_connect(client.handle, server.qp_num, server.psn, server.lid, &c_char_array_to_string(&server.gid_hex))?;
            library.rdma_rc_endpoint_connect(server.handle, client.qp_num, client.psn, client.lid, &c_char_array_to_string(&client.gid_hex))?;
            let (address, rkey) = library.rdma_rc_endpoint_request_target(server.handle)?;
            library.rdma_rc_endpoint_request_recv(server.handle, 0, 1)?;
            let frame: Vec<u8> = (0..193).map(|i| i as u8).collect();
            let view = library.rdma_rc_endpoint_buffer_view(client.handle, false)?;
            // SAFETY: no WR exists yet and the checked frame fits the owned ring.
            unsafe { std::ptr::copy_nonoverlapping(frame.as_ptr(), view.host_ptr.cast::<u8>(), frame.len()); }
            for offset in (0..frame.len()).step_by(37) {
                library.rdma_rc_endpoint_request_write(client.handle, 0, 0, None, offset,
                    37.min(frame.len() - offset), address + offset as u64, rkey, 2)?;
            }
            // Deliberately no WRITE CQ drain before the SEND.
            library.rdma_rc_endpoint_request_doorbell(client.handle, 0, &doorbell(77, 0, 10, frame.len()), 3)?;
            library.rdma_rc_endpoint_poll(server.handle, 0, 1, 1_000_000, 1000)?;
            let raw = library.rdma_rc_endpoint_request_read_doorbell(server.handle, 0)?;
            assert_eq!(validate_doorbell(&raw, 77, 0, frame.len())?, (10, frame.len()));
            let mut received = vec![0; frame.len()];
            library.rdma_rc_endpoint_copy_recv_at(server.handle, &mut received, 0, frame.len())?;
            assert_eq!(received, frame);
            library.rdma_rc_endpoint_poll(client.handle, 7, 0, 1_000_000, 1000)?;
            Ok(())
        })();
        // Native destroy stops the QP before registrations/storage are freed.
        let a = library.rdma_rc_endpoint_destroy(client.handle);
        let b = library.rdma_rc_endpoint_destroy(server.handle);
        result.and(a).and(b)
    }

    #[test]
    fn stale_reordered_and_wrong_session_doorbells_fail_closed() -> Result<()> {
        let raw = doorbell(71, 9, 123, 100);
        assert_eq!(validate_doorbell(&raw, 71, 9, 100)?, (123, 100));
        assert!(validate_doorbell(&raw, 72, 9, 100).is_err());
        assert!(validate_doorbell(&raw, 71, 8, 100).is_err());
        assert!(validate_doorbell(&raw, 71, 10, 100).is_err());
        assert!(validate_doorbell(&raw, 71, 9, 99).is_err());
        Ok(())
    }
}
