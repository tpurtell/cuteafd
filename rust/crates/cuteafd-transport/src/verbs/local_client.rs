//! Persistent coordinator QPs progressed by the inference owner. The existing
//! response assembler/recycling code runs locally; no QP worker is spawned.
use super::*;

pub(crate) struct LocalTp4Client {
    peers: Vec<SocketAddr>,
    config: TcpTransportConfig,
    execution_lane: u32,
    sessions: Vec<Option<VerbsHostProtocolV2PersistentClientSession>>,
    /// Bootstrap seals survive a QP reconnect and are never queried per request.
    expert_grouping: Vec<Option<crate::worker_selection::InstalledExpertGrouping>>,
    /// Retired control streams awaiting the worker's ring-release EOF, with
    /// the peer each was connected to (a released socket no longer has one).
    retired: Vec<(TcpStream, SocketAddr)>,
    retirement_error: Option<String>,
    capacity: Option<u32>,
    /// Explicit immutable ModelAdmission endpoint; survives every QP reconnect.
    model_endpoint: Option<u32>,
    pending: Vec<VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>>,
    /// Dispatched waves, oldest first. Every rank serves its requests in post
    /// order, so the oldest wave completes first on each rank; `poll` serves
    /// only the front and later waves' chunks wait in their own channels.
    waves: VecDeque<OpenWave>,
    /// Most waves outstanding at once (1 unless raised by the owner).
    max_waves: usize,
    /// Per-rank device ranges response payloads land in (GPU landing).
    landing: Vec<Option<DeviceLanding>>,
    /// Shared registered request buffer (zero-copy egress).
    egress: Option<Arc<super::egress::EgressBuffer>>,
    egress_slot: usize,
    /// Optional terminal contract. A failed temporary/persistent endpoint
    /// publishes into this witness before its external landing owner drops.
    terminal_owner: Option<Arc<AtomicBool>>,
    terminal_failed: bool,
    terminal_quiesced: bool,
    terminal_released: bool,
    /// Each rank's RoCE flow label for this transport's sessions
    /// (`CUTEAFD_RDMA_BOND_BALANCE`; all 0 when off).
    flows: super::flows::FlowSlots,
}
impl LocalTp4Client {
    /// Generic peer set for the declared physical rank counts 1 through 6.
    pub(crate) fn new_ranks(peers: Vec<SocketAddr>, config: TcpTransportConfig, execution_lane: u32) -> Self {
        let world = peers.len();
        // Placed now, before any request is in flight on the fabric.
        let flows = super::flows::FlowSlots::prepare(&peers);
        Self {
            flows,
            peers,
            config,
            execution_lane,
            sessions: (0..world).map(|_| None).collect(),
            expert_grouping: vec![None; world],
            retired: Vec::new(),
            retirement_error: None,
            capacity: None,
            model_endpoint: None,
            pending: (0..world).map(|_| VecDeque::with_capacity(1)).collect(),
            waves: VecDeque::with_capacity(1),
            max_waves: 1,
            landing: vec![None; world],
            egress: None,
            egress_slot: 0,
            terminal_owner: None,
            terminal_failed: false,
            terminal_quiesced: false,
            terminal_released: false,
        }
    }
    pub(crate) fn execution_lane(&self) -> u32 { self.execution_lane }
    pub(crate) fn peers(&self) -> &[SocketAddr] { &self.peers }
    pub(crate) fn set_capacity(&mut self, capacity: u32) { self.capacity = Some(capacity); }
    pub(crate) fn bind_model_endpoint(&mut self, endpoint: u32) -> Result<()> {
        anyhow::ensure!(endpoint < 8, "ModelAdmission endpoint index must be below eight");
        anyhow::ensure!(self.model_endpoint.is_none(), "ModelAdmission endpoint already bound");
        anyhow::ensure!(self.sessions.iter().all(Option::is_none)
            && self.expert_grouping.iter().all(Option::is_none) && self.waves.is_empty(),
            "ModelAdmission endpoint must be bound before bootstrap or dispatch");
        self.model_endpoint = Some(endpoint);
        Ok(())
    }
    /// Admit up to `waves` outstanding waves. Raising it requires no open wave.
    pub(crate) fn set_max_waves(&mut self, waves: usize) -> Result<()> {
        anyhow::ensure!((1..=4).contains(&waves), "Spark transport admits 1..=4 outstanding waves");
        anyhow::ensure!(self.waves.is_empty(), "cannot change wave admission with waves in flight");
        anyhow::ensure!(self.egress.is_none() && self.sessions.iter().all(Option::is_none),
            "wave admission must precede egress allocation and bootstrap");
        self.max_waves = waves;
        Ok(())
    }
    pub(crate) fn max_waves(&self) -> usize { self.max_waves }
    pub(crate) fn timeout(&self) -> Duration { self.config.timeout }
    pub(crate) fn open_waves(&self) -> usize { self.waves.len() }

    pub(crate) fn installed_grouping(&self) -> Result<Vec<crate::worker_selection::InstalledExpertGrouping>> {
        let reports = self.expert_grouping.iter().enumerate().map(|(rank, report)|
            report.clone().with_context(|| match self.sessions[rank].is_some() {
                // A session reported nothing: an older worker's handshake.
                true => format!("worker rank {rank} did not report installed expert grouping"),
                // No session ever ran, so no worker was asked. Name the cause
                // instead of blaming a rank that never spoke.
                false => format!("rank {rank} has no live Spark session, so its installed expert grouping was never \
                    reported; bootstrap the transport (or dispatch a wave) before planning against it"),
            }))
            .collect::<Result<Vec<_>>>()?;
        crate::worker_selection::InstalledExpertGrouping::validate_ranks(&reports)?;
        Ok(reports)
    }

    /// Connects every rank's persistent session with `request` — which sizes
    /// the registered rings, so it must be the widest wave this transport will
    /// ever dispatch — and returns once each worker's immutable installed
    /// grouping has been published. Only the control bootstrap runs: nothing is
    /// dispatched and no worker sees the request's rows. A planner can read
    /// [`Self::installed_grouping`] afterwards without a wave having to open a
    /// session first, and the waves that follow reuse these sessions.
    pub(crate) fn bootstrap(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        for rank in 0..self.peers.len() {
            if self.sessions[rank].is_none() {
                self.drain_retired()?;
                self.open_session(rank, request, self.landing[rank], true)?;
            }
        }
        self.installed_grouping()?;
        Ok(())
    }

    /// Opens `rank`'s session with `request` (which sizes the registered rings)
    /// and retains the grouping its bootstrap reported.
    fn open_session(&mut self, rank: usize, request: &ExpertProtocolV2Request, landing: Option<DeviceLanding>,
        egress: bool) -> Result<()> {
        landing_phase_admits_open(landing, self.max_waves, self.egress_slot)?;
        let flow_label = self.flows.label(rank, self.peers[rank])?;
        // Only the compact-BF16 wire caps a session's rings by capacity; every
        // other family's rings are the request's own wire extents.
        let capacity = self.capacity.filter(|_| cuteafd_core::expert_geometry().family() == Some("v41")
            && request.header.flags & crate::protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 != 0);
        self.sessions[rank] = Some(VerbsHostProtocolV2PersistentClientSession::connect_local(
            self.peers[rank], &self.config, request, self.execution_lane(), landing,
            if egress { self.egress.as_ref() } else { None },
            self.terminal_owner.clone(), flow_label, capacity, self.model_endpoint)?);
        self.retain_grouping(rank)
    }

    fn retain_grouping(&mut self, rank: usize) -> Result<()> {
        let current = self.sessions[rank].as_ref().and_then(|session| session.expert_grouping.as_ref());
        if let Some(previous) = &self.expert_grouping[rank] {
            anyhow::ensure!(current == Some(previous), "worker rank {rank} changed installed grouping on reconnect");
        }
        if let Some(report) = current {
            report.validate_connection(rank, self.peers.len())?;
            self.expert_grouping[rank] = Some(report.clone());
        }
        crate::worker_selection::seal_installed_grouping(self.peers[rank], current)?;
        if self.expert_grouping.iter().all(Option::is_some) {
            self.installed_grouping()?;
        }
        Ok(())
    }

    fn retire_session(&mut self, rank: usize) -> Result<()> {
        if let Some(session) = self.sessions[rank].take() {
            let stream = session._stream.try_clone()?;
            stream.shutdown(std::net::Shutdown::Write)?;
            drop(session);
            self.retired.push((stream, self.peers[rank]));
        }
        Ok(())
    }

    fn drain_retired(&mut self) -> Result<()> {
        self.drain_retired_with_timeout(RING_RELEASE_TIMEOUT)
    }

    fn drain_retired_with_timeout(&mut self, timeout: Duration) -> Result<()> {
        if let Some(error) = &self.retirement_error { anyhow::bail!("{error}"); }
        for (stream, peer) in &mut self.retired {
            // Once the worker's EOF crossed ours the socket has no peer address
            // (ENOTCONN); the EOF it left is still the release to read.
            let result = wait_for_ring_release(stream, timeout)
                .with_context(|| format!("expert endpoint peer={peer} did not release its rings"));
            if let Err(error) = result {
                // EOF was not acknowledged: this transport must never re-admit a replacement.
                self.retirement_error = Some(format!("{error:#}"));
                return Err(error);
            }
        }
        self.retired.clear();
        Ok(())
    }

    /// Allocates the shared request buffer (`bytes`, pinned and device-mapped)
    /// and registers it on every session from the next connection on.
    pub(crate) fn enable_egress(&mut self, bytes: usize) -> Result<CuteafdHostBuffer> {
        anyhow::ensure!(self.waves.is_empty(), "cannot replace egress with outstanding waves");
        if self.sessions.iter().any(Option::is_some) {
            anyhow::ensure!(self.terminal_owner.is_none(), "terminal egress must precede bootstrap");
            self.reset();
        }
        let buffer = super::egress::EgressBuffer::new(bytes, self.max_waves)?;
        let host = buffer.host();
        self.egress = Some(buffer);
        Ok(host)
    }
    /// The shared request buffer, once no request payload views it and every
    /// session's request sends have left, so it may be rewritten.
    pub(crate) fn egress_target(&mut self) -> Result<CuteafdHostBuffer> {
        let buffer = self.egress.as_ref().context("this transport has no egress buffer")?;
        anyhow::ensure!(buffer.writable(self.egress_slot), "a request still views the egress buffer");
        let host = buffer.target(self.egress_slot)?;
        for session in self.sessions.iter_mut().flatten() {
            session.drain_egress_reads(self.egress_slot, &self.config)?;
        }
        Ok(host)
    }
    /// Every slot of the registered request ring as a staging arena: requests
    /// whose payload views one go out from it on every rank (no per-rank copy).
    pub(crate) fn egress_arenas(&self) -> Result<Vec<super::StagingArena>> {
        let Some(buffer) = self.egress.as_ref() else { return Ok(Vec::new()) };
        (0..self.max_waves).map(|slot| super::StagingArena::of(buffer, slot)).collect()
    }
    /// The first `len` bytes of the egress buffer as a request payload.
    pub(crate) fn egress_payload(&self, len: usize) -> Result<bytes::Bytes> {
        self.egress.as_ref().context("this transport has no egress buffer")?.payload(self.egress_slot, len)
    }
    /// Lands each rank's response payloads in its device range from the next
    /// connection on; drops current sessions so they reconnect that way.
    pub(crate) fn set_landing(&mut self, landing: Vec<Option<DeviceLanding>>) -> Result<()> {
        anyhow::ensure!(landing.len() == self.peers.len(), "one GPU landing entry per rank");
        self.reset();
        self.landing = landing;
        Ok(())
    }
    /// Ranks whose live session lands payloads in device memory.
    pub(crate) fn landing_ranks(&self) -> Vec<bool> {
        self.sessions.iter().map(|s| s.as_ref().is_some_and(|s| s.gpu_landing)).collect()
    }
    /// Per rank: how many final reply frames its live session can leave with
    /// consumers at once (its response ring depth; 0 without a retained ring).
    pub(crate) fn retained_reply_slots(&self) -> Vec<usize> {
        self.sessions.iter().map(|s| s.as_ref().map_or(0, |s| s.retained_reply_slots())).collect()
    }
    pub(crate) fn reset(&mut self) {
        if self.terminal_owner.is_some() {
            // submit's terminal path first quiesces every QP, then drains
            // compute/uploads; resetting here would release queued owners.
            self.terminal_failed = true;
            return;
        }
        self.reset_storage();
    }
    fn reset_storage(&mut self) {
        // Drop queued payloads before sessions unregister their receive rings.
        self.waves.clear();
        self.egress_slot = 0;
        for pending in &mut self.pending {
            pending.clear();
        }
        for rank in 0..self.sessions.len() {
            if let Err(error) = self.retire_session(rank) {
                crate::health::record_failure(format!("expert session teardown failed: {error:#}"));
            }
        }
    }
    /// Posts `request` to every rank. `landing` names, per rank, where this
    /// wave's payload lands in a placed GPU landing (ranks without one ignore
    /// their entry); it is required when any rank's landing is placed.
    pub(crate) fn dispatch(&mut self, request: &ExpertProtocolV2Request, landing: Option<&[LandingSlice]>)
        -> Result<()> {
        crate::health::ensure_available()?;
        anyhow::ensure!(!self.terminal_failed && !self.terminal_quiesced,
            "terminal-owned Spark transport cannot be reused after failure or quiescence");
        anyhow::ensure!(self.waves.len() < self.max_waves,
            "Spark transport already has {} of {} admitted waves pending", self.waves.len(), self.max_waves);
        if let Some(slices) = landing {
            anyhow::ensure!(slices.len() == self.peers.len(), "one landing slice per rank");
        }
        let result = self.post(request, landing);
        if let Err(error) = &result {
            crate::health::record_failure(format!("expert dispatch failed: {error:#}"));
            self.reset();
        }
        result
    }
    fn post(&mut self, request: &ExpertProtocolV2Request, landing: Option<&[LandingSlice]>) -> Result<()> {
        // Any rank still unplaced is measured before this wave's requests go out.
        self.flows.ensure_all(&self.peers)?;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut done = Vec::with_capacity(self.peers.len());
        for rank in 0..self.peers.len() {
            anyhow::ensure!(self.waves.is_empty() || self.sessions[rank].is_some(),
                "a Spark session must stay open while earlier waves are outstanding");
            if self.sessions[rank]
                .as_ref()
                .map(|s| s.fits(request))
                .transpose()?
                == Some(false)
            {
                anyhow::ensure!(self.terminal_owner.is_none() && self.waves.is_empty(),
                    "Spark request exceeds its prewarmed session capacity; reconnect is unsupported here");
                self.retire_session(rank)?;
            }
            if self.sessions[rank].is_none() {
                self.drain_retired()?;
                self.open_session(rank, request, self.landing[rank], true)?;
            }
        }
        if self.expert_grouping.iter().any(Option::is_some) { self.installed_grouping()?; }
        // Every rank's bootstrap is validated before any request enters a QP;
        // then each rank's request goes out as one SEND.
        let mut timings = Vec::with_capacity(self.peers.len());
        for (rank, session) in self.sessions.iter_mut().enumerate() {
            let session = session.as_mut().unwrap();
            let slice = if session.placed_landing {
                let slice = landing.map(|slices| slices[rank])
                    .context("a placed GPU landing needs the wave's landing slice")?;
                let range = self.landing[rank].context("placed session without its landing range")?;
                anyhow::ensure!(range.holds(slice), "rank {rank} landing slice {slice:?} lies outside its {}-byte range",
                    range.bytes);
                Some(slice)
            } else { None };
            timings.push(session.post_request(request, &self.config, slice)?);
        }
        for (rank, timing) in timings.into_iter().enumerate() {
            let (response_tx, response_rx) = tokio::sync::oneshot::channel();
            self.pending[rank].push_back(VerbsHostProtocolV2PendingChunkRoundtrip::new(
                VerbsHostProtocolV2QueuedChunkCommand {
                    request: request.clone(),
                    stream_id: rank,
                    chunk_tx: tx.clone(),
                    response_tx,
                },
                timing,
            ));
            done.push(response_rx);
        }
        // The same-thread channels adapt the existing assembler; they never
        // wake or hand off work to another thread. Each sink takes ownership of its chunk; retained pinned frames
        // return to the session pool only after the owner releases them.
        // Each wave's deadline counts from when it becomes the oldest, so a
        // queued wave is not charged for the waves ahead of it.
        self.waves.push_back(OpenWave { chunks: rx, done, deadline: None, polls: 0 });
        self.egress_slot = (self.egress_slot + 1) % self.max_waves;
        Ok(())
    }

    pub(crate) fn enable_terminal_ownership(&mut self) -> Result<()> {
        anyhow::ensure!(self.sessions.iter().all(Option::is_none) && self.waves.is_empty(),
            "terminal Spark ownership must precede connection/bootstrap");
        self.terminal_owner = Some(Arc::new(AtomicBool::new(false)));
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn terminal_ready_for_test(&self) -> bool {
        self.terminal_owned() && !self.terminal_failed && !self.terminal_quiesced
            && self.sessions.iter().all(Option::is_none)
    }
    pub(crate) fn terminal_owned(&self) -> bool { self.terminal_owner.is_some() }
    pub(crate) fn terminal_retained(&self) -> bool {
        self.terminal_owner.as_ref().is_some_and(|owner|owner.load(Ordering::Acquire))
    }
    pub(crate) fn terminal_released(&self) -> bool { self.terminal_released }

    /// Stop every QP before changing any pending/chunk/registration owner.
    /// A failure preserves the complete collection, including successful ranks.
    pub(crate) fn terminal_quiesce(&mut self) -> Result<()> {
        anyhow::ensure!(self.terminal_owner.is_some(),"transport has no terminal owner contract");
        self.terminal_failed = true;
        let result=quiesce_all(&self.sessions, |session|session.endpoint.terminal_quiesce());
        if result.is_err() || self.terminal_retained() {
            self.terminal_owner.as_ref().unwrap().store(true,Ordering::Release);
            return Err(result.err().unwrap_or_else(||anyhow::anyhow!("a temporary Spark endpoint requires owner retention")));
        }
        self.terminal_quiesced = true;
        Ok(())
    }

    /// Called only after every compute/copy consumer has drained successfully.
    pub(crate) fn terminal_release(&mut self) -> Result<()> {
        anyhow::ensure!(self.terminal_quiesced && !self.terminal_retained(),
            "Spark pending owners cannot be released before successful terminal quiescence");
        self.reset_storage();
        self.terminal_released = true;
        Ok(())
    }
    pub(crate) fn poll<F>(&mut self, sink: F) -> Result<bool>
    where F: FnMut(VerbsHostProtocolV2ResponseChunk) -> Result<()>, {
        let result = self.poll_inner(sink);
        if let Err(error) = &result {
            crate::health::record_failure(format!("expert response failed: {error:#}"));
        }
        result
    }
    fn poll_inner<F>(&mut self, mut sink: F) -> Result<bool>
    where
        F: FnMut(VerbsHostProtocolV2ResponseChunk) -> Result<()>,
    {
        let timeout = self.config.timeout;
        let wave = self.waves.front_mut().context("local TP4 has no pending request")?;
        // The deadline (seconds) is checked on every 64th poll: a busy poller
        // must not spend its time reading the clock.
        wave.polls = wave.polls.wrapping_add(1);
        if wave.deadline.is_none() || wave.polls % 64 == 0 {
            let now = Instant::now();
            let deadline = *wave.deadline.get_or_insert(now + timeout);
            anyhow::ensure!(now < deadline, "local TP4 response deadline expired");
        }
        for rank in 0..self.peers.len() {
            if !self.pending[rank].is_empty() {
                self.sessions[rank]
                    .as_mut()
                    .context("local TP4 session missing")?
                    .try_progress_chunk_requests(&mut self.pending[rank], &self.config)?;
            }
            let chunks = &mut self.waves.front_mut().context("local TP4 response queue missing")?.chunks;
            while let Ok(chunk) = chunks.try_recv() {
                sink(chunk)?;
            }
        }
        // The oldest wave is complete once every rank's completion arrived;
        // later waves' requests may still be pending on the same ranks.
        let wave = self.waves.front_mut().context("local TP4 response queue missing")?;
        let mut index = 0;
        while index < wave.done.len() {
            match wave.done[index].try_recv() {
                Ok(result) => { result?; wave.done.swap_remove(index); }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => index += 1,
                Err(error) => return Err(error).context("local TP4 completion missing"),
            }
        }
        if !wave.done.is_empty() { return Ok(false); }
        let mut wave = self.waves.pop_front().expect("front wave present");
        // Final validation and completion publication happen synchronously in
        // accept_chunk_response_frame before it removes a pending request, so
        // every chunk of this wave was already queued.
        anyhow::ensure!(wave.chunks.try_recv().is_err(), "local TP4 chunk arrived after its completion");
        Ok(true)
    }
}

/// One dispatched wave's response chunks and per-rank completions.
struct OpenWave {
    chunks: tokio::sync::mpsc::UnboundedReceiver<VerbsHostProtocolV2ResponseChunk>,
    done: Vec<tokio::sync::oneshot::Receiver<Result<VerbsHostProtocolV2ResponseStreamStats>>>,
    deadline: Option<Instant>,
    /// Polls of this wave while it is the oldest (paces the deadline check).
    polls: u32,
}

fn quiesce_all<T>(owners: &[Option<T>], mut quiesce: impl FnMut(&T)->Result<()>) -> Result<()> {
    let mut failures=Vec::new();
    for (rank,owner) in owners.iter().enumerate() {
        if let Some(owner)=owner {
            if let Err(error)=quiesce(owner) { failures.push(format!("rank {rank}: {error:#}")); }
        }
    }
    anyhow::ensure!(failures.is_empty(),"terminal QP quiescence failed; all owners retained: {}",failures.join("; "));
    Ok(())
}

impl Drop for LocalTp4Client {
    fn drop(&mut self) {
        if self.terminal_owned() && !self.terminal_released {
            if let Err(error)=self.terminal_quiesce() {
                // Keep native registrations, retained chunks, external egress
                // owners and their library loaded. SparkLink also retains its
                // external GPU landing/queued-upload owner via the witness.
                std::mem::forget(std::mem::take(&mut self.sessions));
                std::mem::forget(std::mem::take(&mut self.pending));
                std::mem::forget(std::mem::take(&mut self.waves));
                std::mem::forget(self.egress.take());
                tracing::error!(%error,"retaining terminal Spark client owners");
            }
        }
    }
}

/// How long a reconnect waits for each retired endpoint's ring release.
///
/// A worker notices our EOF within its 1 ms liveness poll, drains the
/// executions holding that connection's units, releases its rings and closes.
/// What it drains is bounded by its deferred admission (at most the ring
/// depth of requests per connection, one layer of one unit each: tens of ms
/// for a 2752-row prefill unit on a GB10), so a live worker releases in well
/// under a second. A worker that has not released by then is wedged or gone:
/// re-admitting would double-book its ring budget, so the reconnect fails and
/// that transport stays refused (sticky), and the error reaches the request
/// (the serving reset forwards it). A longer bound only delays that error
/// while every lane waits on the reset. Retired streams are drained in turn,
/// but every rank received its EOF at once, so the total is the slowest rank.
const RING_RELEASE_TIMEOUT: Duration = Duration::from_secs(3);

fn wait_for_ring_release(stream: &mut TcpStream, timeout: Duration) -> Result<()> {
    stream.set_read_timeout(Some(timeout))?;
    let mut byte = [0u8; 1];
    match std::io::Read::read(stream, &mut byte) {
        Ok(0) => Ok(()),
        Ok(_) => anyhow::bail!("expert endpoint ring-release acknowledgement contained unexpected data"),
        Err(error) => Err(error).context("expert endpoint ring-release acknowledgement timed out or failed before reconnect"),
    }
}

/// A session opens with its landing rotation at slot 0: native posts its
/// receive WQEs' landing slots in post order from 0 (`RdmaLandingSlots`), so
/// reply `sequence` lands in slot `sequence % slots` from the session's first
/// reply. The client's waves rotate one wave slot per
/// dispatch (`egress_slot`), and the exchange lands wave `k` in slot `k % N`.
/// A multi-slot GPU landing may therefore (re)open only in phase, at wave
/// slot 0: otherwise its next reply would land in another wave's planes.
/// A placed landing has no rotation (each request names its slice), so it may
/// (re)open at any slot.
fn landing_phase_admits_open(landing: Option<DeviceLanding>, max_waves: usize, next_slot: usize) -> Result<()> {
    let slots = landing.filter(|landing| !landing.placed).map_or(1, |landing| landing.bytes / landing.slot_bytes.max(1));
    anyhow::ensure!(slots <= 1 || next_slot % max_waves.max(1) == 0,
        "a GPU-landing Spark session cannot (re)connect at wave slot {next_slot} of {max_waves}: its landing \
        rotation restarts at slot 0 (reset the link first)");
    Ok(())
}

#[cfg(test)]
mod terminal_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    struct Owners { registered: Arc<AtomicUsize>, landing: Arc<AtomicUsize> }
    impl Drop for Owners { fn drop(&mut self) {
        self.registered.fetch_add(1,Ordering::SeqCst);
        self.landing.fetch_add(1,Ordering::SeqCst);
    } }

    #[test]
    fn failed_qp_quiescence_retains_registered_and_external_landing_owners() {
        let registered=Arc::new(AtomicUsize::new(0));
        let landing=Arc::new(AtomicUsize::new(0));
        let owners=(0..2).map(|_|Some(Owners { registered:registered.clone(),landing:landing.clone() })).collect::<Vec<_>>();
        let mut attempted=0;
        let result=quiesce_all(&owners, |_| { attempted+=1;
            if attempted==1 { anyhow::bail!("injected ibv_destroy_qp failure"); } Ok(()) });
        assert!(result.is_err());
        assert_eq!(attempted,2);
        assert_eq!(registered.load(Ordering::SeqCst),0);
        assert_eq!(landing.load(Ordering::SeqCst),0);
        assert!(owners.iter().all(Option::is_some));
        drop(owners);
        assert_eq!(registered.load(Ordering::SeqCst),2);
        assert_eq!(landing.load(Ordering::SeqCst),2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_local_transports_do_not_collide_on_one_decode_owner() -> Result<()> {
        use crate::worker_selection::{execution_owner, ExecutionOwner, SparkExecutions};
        let peers = vec!["127.0.0.1:19454".parse()?];
        let owners = SparkExecutions::admit(&[64, 64], 1024, 1024, 2, &[1, 80, 1024])?;
        let mut retained = [false; 2];
        for lane in 0..2 {
            let mut client = LocalTp4Client::new_ranks(peers.clone(), TcpTransportConfig::default(), lane as u32);
            for source in [crate::ExpertV2SourceKind::Decode, crate::ExpertV2SourceKind::MtpVerify] {
                let owner = execution_owner(&owners, source, client.execution_lane(), 62)?;
                assert_eq!(owner, ExecutionOwner::Decode(lane));
            }
            let ExecutionOwner::Decode(owner) = execution_owner(&owners,
                crate::ExpertV2SourceKind::MtpVerify, client.execution_lane(), 62)? else { unreachable!() };
            assert!(!std::mem::replace(&mut retained[owner], true),
                "Spark execution owner {owner} already retains its admitted 1 units");
            assert_eq!(execution_owner(&owners, crate::ExpertV2SourceKind::Prefill,
                client.execution_lane(), 1024)?, ExecutionOwner::Prefill);
            client.reset();
            assert_eq!(client.execution_lane(), lane as u32, "reconnect must retain lane identity");
        }
        assert_eq!(retained, [true; 2]);
        Ok(())
    }

    #[test]
    fn multi_slot_landing_reconnects_only_in_phase() {
        let landing = |slots: usize| Some(DeviceLanding { ptr: 4096, bytes: slots * 12288, slot_bytes: 12288, placed: false });
        let placed = Some(DeviceLanding { ptr: 4096, bytes: 4 * 12288, slot_bytes: 4 * 12288, placed: true });
        for waves in 1..=4 {
            for next in 0..2 * waves {
                // Host receives, a one-slot landing and a placed landing have no phase.
                assert!(landing_phase_admits_open(None, waves, next).is_ok());
                assert!(landing_phase_admits_open(landing(1), waves, next).is_ok());
                assert!(landing_phase_admits_open(placed, waves, next).is_ok());
                assert_eq!(landing_phase_admits_open(landing(waves), waves, next).is_ok(),
                    waves == 1 || next % waves == 0, "{waves} waves at slot {next}");
            }
        }
    }

    /// A GPU-landing session can only be (re)opened at wave slot 0: out of
    /// phase, `open_session` refuses before any connection is attempted; in
    /// phase (and for host receives) it goes on to connect.
    #[test]
    fn out_of_phase_gpu_landing_session_is_refused_before_connecting() -> Result<()> {
        // Nothing listens here: an admitted open fails at its connect instead.
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let peer = listener.local_addr()?;
        drop(listener);
        let request = ExpertProtocolV2Request::new(1, 1, 3, 64, crate::ExpertV2Dtype::Bf16,
            vec![crate::ExpertProtocolV2RowDescriptor { row_id: 0, source_kind: crate::ExpertV2SourceKind::Prefill,
                source_request_id: 1, token_position: 0, route_offset: 0, route_count: 1 }],
            vec![crate::ExpertProtocolV2RouteEntry { row_index: 0, expert_id: 0, gate_weight: 1.0 }], vec![0; 128])?;
        for waves in 2..=4usize {
            let landing = DeviceLanding { ptr: 1 << 32, bytes: waves * 12288, slot_bytes: 12288, placed: false };
            for slot in 0..waves {
                for gpu in [false, true] {
                    let mut client = LocalTp4Client::new_ranks(vec![peer], TcpTransportConfig { timing: false,
                        timeout: Duration::from_millis(200), max_frame_bytes: 1 << 20 }, 0);
                    client.set_max_waves(waves)?;
                    if gpu { client.set_landing(vec![Some(landing)])?; }
                    // As after `slot` dispatches on live sessions that then dropped.
                    client.egress_slot = slot;
                    let error = format!("{:#}", client.open_session(0, &request, client.landing[0], true).unwrap_err());
                    let refused = error.contains("cannot (re)connect at wave slot");
                    assert_eq!(refused, gpu && slot != 0, "{waves} waves, slot {slot}, gpu {gpu}: {error}");
                    assert!(client.sessions[0].is_none());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn replaced_endpoint_releases_rings_before_reconnect() -> Result<()> {
        let budget = crate::RingBudget::new(302_120_960);
        let old = budget.reserve(134_217_728)?;
        let other_lane = budget.reserve(151_060_480)?;
        assert!(budget.reserve(134_217_728).is_err());
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let mut client = TcpStream::connect(listener.local_addr()?)?;
        let (mut server, _) = listener.accept()?;
        let owner = std::thread::spawn(move || -> Result<()> {
            let mut byte = [0u8; 1];
            assert_eq!(std::io::Read::read(&mut server, &mut byte)?, 0);
            // The native endpoint's teardown precedes the reservation and TCP EOF.
            drop(old);
            drop(server);
            Ok(())
        });
        client.shutdown(std::net::Shutdown::Write)?;
        wait_for_ring_release(&mut client, Duration::from_secs(1))?;
        assert_eq!(budget.used(), 151_060_480);
        let replacement = budget.reserve(134_217_728)?;
        assert_eq!(budget.used(), 285_278_208);
        assert_eq!(budget.peak(), 285_278_208);
        owner.join().unwrap()?;
        drop(replacement);
        drop(other_lane);
        Ok(())
    }

    #[test]
    fn ring_release_wait_is_bounded_and_names_the_endpoint() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let mut client = TcpStream::connect(listener.local_addr()?)?;
        let (_server, _) = listener.accept()?;
        let error = wait_for_ring_release(&mut client, Duration::from_millis(10)).unwrap_err();
        assert!(format!("{error:#}").contains("expert endpoint ring-release acknowledgement"));
        Ok(())
    }

    #[test]
    fn retirement_timeout_is_sticky_without_repeated_waits() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let client = TcpStream::connect(listener.local_addr()?)?;
        let (_server, _) = listener.accept()?;
        let mut transport = LocalTp4Client::new_ranks(Vec::new(), TcpTransportConfig::default(), 0);
        let peer = client.peer_addr()?;
        transport.retired.push((client, peer));
        let first = transport.drain_retired_with_timeout(Duration::from_millis(10)).unwrap_err();
        let started = Instant::now();
        let second = transport.drain_retired_with_timeout(Duration::from_secs(3)).unwrap_err();
        assert!(started.elapsed() < Duration::from_millis(100));
        assert_eq!(format!("{first:#}"), format!("{second:#}"));
        assert_eq!(transport.retired.len(), 1);
        Ok(())
    }

    /// The reconnect bound fits a live worker that releases only after
    /// draining queued work (here ~10x a queued prefill unit-layer) and fails
    /// a peer that never releases within the bound, not after it.
    #[test]
    fn ring_release_bound_waits_for_a_draining_worker_and_fails_a_wedged_one() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let client = TcpStream::connect(listener.local_addr()?)?;
        let (mut server, _) = listener.accept()?;
        let peer = client.peer_addr()?;
        client.shutdown(std::net::Shutdown::Write)?;
        let worker = std::thread::spawn(move || -> Result<()> {
            assert_eq!(std::io::Read::read(&mut server, &mut [0u8; 1])?, 0);
            std::thread::sleep(Duration::from_millis(300));
            drop(server);
            Ok(())
        });
        let mut transport = LocalTp4Client::new_ranks(Vec::new(), TcpTransportConfig::default(), 0);
        transport.retired.push((client, peer));
        transport.drain_retired()?;
        worker.join().unwrap()?;
        // A peer that never releases: the reconnect fails at the bound.
        let client = TcpStream::connect(listener.local_addr()?)?;
        let (_wedged, _) = listener.accept()?;
        let peer = client.peer_addr()?;
        transport.retired.push((client, peer));
        let started = Instant::now();
        let error = transport.drain_retired().unwrap_err();
        let waited = started.elapsed();
        assert!(waited >= RING_RELEASE_TIMEOUT && waited < RING_RELEASE_TIMEOUT + Duration::from_secs(1), "{waited:?}");
        assert!(format!("{error:#}").contains("did not release its rings"), "{error:#}");
        Ok(())
    }

    #[test]
    fn an_endpoint_that_already_released_and_closed_drains() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let client = TcpStream::connect(listener.local_addr()?)?;
        let (mut server, _) = listener.accept()?;
        let peer = client.peer_addr()?;
        client.shutdown(std::net::Shutdown::Write)?;
        // The worker reads our EOF, releases its rings and closes before the
        // next connect drains: the FINs crossed and the socket has no peer.
        assert_eq!(std::io::Read::read(&mut server, &mut [0u8; 1])?, 0);
        drop(server);
        let deadline = Instant::now() + Duration::from_secs(1);
        while client.peer_addr().is_ok() {
            anyhow::ensure!(Instant::now() < deadline, "loopback close did not complete");
            std::thread::yield_now();
        }
        let mut transport = LocalTp4Client::new_ranks(Vec::new(), TcpTransportConfig::default(), 0);
        transport.retired.push((client, peer));
        transport.drain_retired_with_timeout(Duration::from_secs(1))?;
        assert!(transport.retired.is_empty() && transport.retirement_error.is_none());
        Ok(())
    }

    #[test]
    fn bootstrap_rejection_reaches_coordinator_with_budget_numbers() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let client = TcpStream::connect(listener.local_addr()?)?;
        let (mut server, _) = listener.accept()?;
        write_control(&mut server, &serde_json::json!({
            "message": "protocol_v2_bootstrap_error",
            "error": "expert endpoint execution_lane=0: mapped RDMA ring budget exceeded: 419495936 bytes requested in total, limit is 302120960"
        }))?;
        let error = read_control_value(&mut BufReader::new(client)).unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("expert endpoint") && error.contains("419495936") && error.contains("302120960"));
        assert!(!error.contains("control plane closed"));
        Ok(())
    }

    /// A transport whose sessions were never opened has no report to give, and
    /// the error says so: it names the missing bootstrap instead of blaming a
    /// rank whose worker was never asked (which is what a planner reading the
    /// grouping before any session looks like).
    #[test]
    fn grouping_without_a_bootstrap_names_the_missing_session() -> Result<()> {
        let peers = ["127.0.0.1:19451".parse()?, "127.0.0.1:19452".parse()?];
        let client = LocalTp4Client::new_ranks(peers.to_vec(), TcpTransportConfig::default(), 0);
        assert!(client.sessions.iter().all(Option::is_none));
        let error = format!("{:#}", client.installed_grouping().unwrap_err());
        assert!(error.contains("rank 0 has no live Spark session"), "{error}");
        assert!(error.contains("bootstrap the transport"), "{error}");
        assert!(!error.contains("did not report installed expert grouping"), "{error}");
        Ok(())
    }

    #[test]
    fn explicit_model_endpoint_is_immutable_and_survives_reset() -> Result<()> {
        let peer = "127.0.0.1:19453".parse()?;
        let mut client = LocalTp4Client::new_ranks(vec![peer], TcpTransportConfig::default(), 0);
        assert!(client.bind_model_endpoint(8).is_err());
        assert!(client.model_endpoint.is_none());
        client.bind_model_endpoint(2)?;
        assert!(client.bind_model_endpoint(0).is_err());
        client.reset();
        assert_eq!(client.model_endpoint, Some(2));
        assert!(client.bind_model_endpoint(1).is_err());
        let mut unbound = LocalTp4Client::new_ranks(vec![peer], TcpTransportConfig::default(), 0);
        let (_sender, chunks) = tokio::sync::mpsc::unbounded_channel();
        let (_done, completed) = tokio::sync::oneshot::channel();
        unbound.waves.push_back(OpenWave { chunks, done: vec![completed], deadline: None, polls: 0 });
        assert!(unbound.bind_model_endpoint(2).is_err());
        assert!(unbound.model_endpoint.is_none());
        Ok(())
    }

    #[test]
    fn configured_sim_capacity_fits_two_endpoints_on_first_connect() -> Result<()> {
        let (request, response) = crate::protocol_v2::compact_expert_wire_bytes(
            cuteafd_core::ExpertGeometry::DEEPSEEK_V41, 1024, false)?;
        assert_eq!((request, response), (5_521_504, 10_489_952));
        let alignment = crate::verbs_host_capabilities().preferred_alignment;
        let request = VerbsHostRdmaRing::new(request.max(8 << 20), alignment, 8)?;
        let response = VerbsHostRdmaRing::new(response.max(8 << 20), alignment, 8)?;
        let bytes = request.registered_span_bytes + response.registered_span_bytes;
        assert_eq!(bytes, 151_060_480);
        assert_eq!(bytes * 2, 302_120_960);
        Ok(())
    }

    const STATS: VerbsHostProtocolV2ResponseStreamStats =
        VerbsHostProtocolV2ResponseStreamStats { response_frames: 1, response_wire_bytes: 0, response_executor_id: 1 };

    #[test]
    fn waves_are_admitted_up_to_the_limit_and_served_oldest_first() -> Result<()> {
        let peers = ["127.0.0.1:19443".parse()?];
        let mut client = LocalTp4Client::new_ranks(peers.to_vec(), TcpTransportConfig { timing: false,
            timeout: std::time::Duration::from_secs(1), max_frame_bytes: 200_000 }, 0);
        assert!(client.set_max_waves(0).is_err() && client.set_max_waves(5).is_err());
        client.set_max_waves(2)?;
        let mut senders = Vec::new();
        for _ in 0..2 {
            let (chunks, receiver) = tokio::sync::mpsc::unbounded_channel();
            let (done, completed) = tokio::sync::oneshot::channel();
            client.waves.push_back(OpenWave { chunks: receiver, done: vec![completed], deadline: None, polls: 0 });
            senders.push((chunks, done));
        }
        assert_eq!(client.open_waves(), 2);
        assert!(client.set_max_waves(3).is_err(), "admission changed with waves in flight");
        // A third dispatch is refused before any session or request is touched.
        let request = crate::expert::test_request(1);
        let error = client.dispatch(&request, None).unwrap_err().to_string();
        assert!(error.contains("2 of 2 admitted waves"), "{error}");
        // The second wave's completion does not complete the first.
        let (first, second) = (senders.remove(0), senders.remove(0));
        second.1.send(Ok(STATS)).ok();
        assert!(!client.poll(|_| Ok(()))?, "a later wave completed the oldest");
        first.1.send(Ok(STATS)).ok();
        assert!(client.poll(|_| Ok(()))?);
        assert_eq!(client.open_waves(), 1);
        assert!(client.poll(|_| Ok(()))?);
        assert_eq!(client.open_waves(), 0);
        drop((first.0, second.0));
        Ok(())
    }

    #[test]
    fn tp2_reset_retains_only_two_actual_peer_slots() -> Result<()> {
        let peers = ["127.0.0.1:19441".parse()?, "127.0.0.1:19442".parse()?];
        let mut client = LocalTp4Client::new_ranks(peers.to_vec(), TcpTransportConfig { timing: false,
            timeout: std::time::Duration::from_secs(1), max_frame_bytes: 200_000,
        }, 0);
        let (_sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let (_completion, completed) = tokio::sync::oneshot::channel();
        client.waves.push_back(OpenWave { chunks: receiver, done: vec![completed], deadline: Some(Instant::now()), polls: 0 });
        client.reset();
        assert_eq!(client.peers, peers);
        assert_eq!(client.sessions.len(), 2);
        assert_eq!(client.pending.len(), 2);
        assert!(client.sessions.iter().all(Option::is_none));
        assert!(client.pending.iter().all(VecDeque::is_empty));
        assert!(client.waves.is_empty());
        Ok(())
    }

    /// Echoes each request's hidden rows in one unindexed frame (a native
    /// compact-BF16 worker's shape, which GPU landing takes).
    struct Echo;
    impl crate::synthetic::ProtocolV2ExpertExecutor for Echo {
        fn name(&self) -> &'static str { "multiwave-echo" }
        fn execute(&self, request: &crate::ExpertProtocolV2RequestView<'_>) -> Result<crate::ExpertProtocolV2Response> {
            let h = &request.header;
            crate::ExpertProtocolV2Response::new_with_output_stride(h.request_id, h.placement_version, h.layer_id,
                h.row_count, h.hidden_dim, h.hidden_dtype, h.hidden_row_stride_bytes, crate::ExpertProtocolV2Status::Ok,
                request.hidden_payload().to_vec())
        }
    }

    fn bf16_request(id: u64, rows: u32, hidden: u32, payload: bytes::Bytes) -> Result<ExpertProtocolV2Request> {
        ExpertProtocolV2Request::new_bytes(id, 1, 3, hidden, crate::ExpertV2Dtype::Bf16,
            (0..rows).map(|row| crate::ExpertProtocolV2RowDescriptor { row_id: row.into(),
                source_kind: crate::ExpertV2SourceKind::Prefill, source_request_id: id, token_position: row.into(),
                route_offset: row, route_count: 1 }).collect(),
            (0..rows).map(|row| crate::ExpertProtocolV2RouteEntry { row_index: row, expert_id: 0, gate_weight: 1.0 })
                .collect(), payload)
    }

    /// Explicit hardware gate only (`CUTEAFD_LOOPBACK_ADDR` = a local RoCE
    /// address; `CUTEAFD_LOOPBACK_GPU=1` adds GPU landing, which needs CUDA and
    /// dma-buf). A real verbs worker and a real coordinator client on one
    /// host, for 1..=4 outstanding waves: pinned host or N-slot GPU landing,
    /// copied or N-slot registered egress, small (decode-sized) and large
    /// (full-capacity) frames, each one SEND. Every echo is byte-exact in its own
    /// wave's landing slot, and a registered egress slot is rewritten
    /// (`egress_target`, which drains only that slot's readers) while the
    /// other waves are still outstanding without disturbing their payloads.
    #[test]
    #[ignore = "requires approved native RDMA loopback hardware gate"]
    fn native_loopback_multiwave_gpu_landing_and_registered_egress_are_byte_exact() -> Result<()> {
        let ip = std::env::var("CUTEAFD_LOOPBACK_ADDR").context("set CUTEAFD_LOOPBACK_ADDR to a local RoCE address")?;
        let gpu = std::env::var("CUTEAFD_LOOPBACK_GPU").as_deref() == Ok("1");
        let hidden = 6144u32;
        let capacity = 2048u32;
        let listener = std::net::TcpListener::bind(format!("{ip}:0"))?;
        let addr = listener.local_addr()?;
        drop(listener);
        let executor: Arc<dyn crate::synthetic::ProtocolV2ExpertExecutor> = Arc::new(Echo);
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            let _ = runtime.block_on(super::super::serve_protocol_v2_verbs_host_with_executor(&addr.to_string(), executor));
        });
        std::thread::sleep(Duration::from_millis(300));
        let library = load_verbs_host_native_library()?;
        let plane_bytes = capacity as usize * hidden as usize * 2;
        // Landing modes: host receives, a rotating N-slot landing, and a placed
        // landing whose slot width changes every pass (as a prefill link
        // re-partitions its reserve between prompts) without reconnecting.
        for (landing, placed) in if gpu { vec![(false, false), (true, false), (true, true)] } else { vec![(false, false)] } {
            for registered in [false, true] {
                for waves in 1..=4usize {
                    let mut client = LocalTp4Client::new_ranks(vec![addr], TcpTransportConfig {
                        timing: false, timeout: Duration::from_secs(30), max_frame_bytes: 64 << 20 }, 0);
                    client.set_max_waves(waves)?;
                    if registered { client.enable_egress(plane_bytes)?; }
                    let mut plane = None;
                    if landing {
                        let buffer = library.alloc_device_buffer(plane_bytes * waves)?;
                        client.set_landing(vec![Some(match placed {
                            true => DeviceLanding::placed(buffer)?,
                            false => DeviceLanding::with_slots(buffer, waves)?,
                        })])?;
                        plane = Some(buffer);
                    }
                    let row_bytes = hidden as usize * 2;
                    let result = (|| -> Result<()> {
                        // Placed: the slot width of pass `p` (rows), inside the
                        // same `waves` x capacity range; rotating: the capacity.
                        let widths = [2048u32, 1376, 1024];
                        // Full capacity first: it sizes the rings for every later wave.
                        let sizes = [2048u32, 1, 21, 64, 1500, 2048, 7, 2048];
                        let mut sent: VecDeque<(ExpertProtocolV2Request, Vec<u8>, usize)> = VecDeque::new();
                        let mut zero_copy = 0;
                        let check = |client: &mut LocalTp4Client, request: ExpertProtocolV2Request, expected: Vec<u8>,
                            offset: usize| -> Result<()> {
                            // The request's payload (an egress view) is still intact.
                            anyhow::ensure!(request.hidden_payload.as_ref() == expected.as_slice(),
                                "request {} payload changed while in flight", request.header.request_id);
                            let mut got = vec![0u8; expected.len()];
                            let mut landed = false;
                            while !client.poll(|chunk| {
                                landed |= chunk.landed;
                                if !chunk.landed {
                                    let bytes = chunk.partial_output_payload.as_ref();
                                    got[..bytes.len()].copy_from_slice(bytes);
                                }
                                Ok(())
                            })? {}
                            if let Some(buffer) = plane.filter(|_| landed) {
                                // SAFETY: the slot lies inside the allocated plane, whose wave landed.
                                let view = CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>()
                                    .add(offset) }.cast(), bytes: got.len(), ..buffer };
                                library.copy_d2h(&mut got, view)?;
                            }
                            anyhow::ensure!(landed == plane.is_some(), "request {} landed={landed}", request.header.request_id);
                            anyhow::ensure!(got == expected, "request {} echo differs (offset {offset})", request.header.request_id);
                            Ok(())
                        };
                        for (index, &rows) in sizes.iter().cycle().take(sizes.len() * 3).enumerate() {
                            let pass = index / sizes.len();
                            let width = if placed { widths[pass % widths.len()] } else { capacity };
                            // A geometry change waits for quiescence, as the exchange's does.
                            let reshape = placed && index % sizes.len() == 0;
                            while client.open_waves() == waves || (reshape && client.open_waves() > 0) {
                                let (request, expected, offset) = sent.pop_front().unwrap();
                                check(&mut client, request, expected, offset)?;
                            }
                            let rows = rows.min(width);
                            let expected: Vec<u8> = (0..rows as usize * hidden as usize * 2)
                                .map(|i| ((i * 7 + index * 13) % 253) as u8).collect();
                            let slot = index % waves;
                            let offset = slot * width as usize * row_bytes;
                            let payload: bytes::Bytes = match registered && expected.len() >= 1 << 20 {
                                true => {
                                    // Rewrites this wave's slot while the other waves' requests,
                                    // views of the other slots, are still outstanding.
                                    let target = client.egress_target()?;
                                    // SAFETY: egress_target returned a writable slot of `plane_bytes`
                                    // that no request views and no send still reads.
                                    unsafe { std::slice::from_raw_parts_mut(target.ptr.cast::<u8>(), expected.len()) }
                                        .copy_from_slice(&expected);
                                    zero_copy += 1;
                                    client.egress_payload(expected.len())?
                                }
                                false => expected.clone().into(),
                            };
                            let request = bf16_request(index as u64 + 1, rows, hidden, payload)?;
                            let slice = [LandingSlice { offset, bytes: width as usize * row_bytes }];
                            client.dispatch(&request, placed.then_some(&slice[..]))?;
                            sent.push_back((request, expected, offset));
                        }
                        while let Some((request, expected, offset)) = sent.pop_front() {
                            check(&mut client, request, expected, offset)?;
                        }
                        anyhow::ensure!((zero_copy > 0) == registered, "registered={registered}: {zero_copy} zero-copy waves");
                        eprintln!("loopback landing={landing} placed={placed} registered={registered} waves={waves}: \
                            {} waves byte-exact, {zero_copy} zero-copy", sizes.len() * 3);
                        Ok(())
                    })();
                    client.reset();
                    if let Some(mut buffer) = plane { library.free_device_buffer(&mut buffer)?; }
                    result?;
                }
            }
        }
        Ok(())
    }
    /// Explicit hardware gate only (`CUTEAFD_LOOPBACK_ADDR`, optional
    /// `CUTEAFD_LOOPBACK_GPU=1`). A full-capacity request (2048 BF16 rows of
    /// 6144, 25 MB, copied and from registered egress) and its 25 MB echo each
    /// go as exactly one SEND: the request posts one send-queue entry and the
    /// worker parses it whole from one receive; the reply arrives as one
    /// full-size receive completion, byte-exact.
    #[test]
    #[ignore = "requires approved native RDMA loopback hardware gate"]
    fn native_loopback_large_request_and_reply_are_one_send_each() -> Result<()> {
        let ip = std::env::var("CUTEAFD_LOOPBACK_ADDR").context("set CUTEAFD_LOOPBACK_ADDR to a local RoCE address")?;
        let gpu = std::env::var("CUTEAFD_LOOPBACK_GPU").as_deref() == Ok("1");
        let (hidden, rows) = (6144u32, 2048u32);
        let listener = std::net::TcpListener::bind(format!("{ip}:0"))?;
        let addr = listener.local_addr()?;
        drop(listener);
        let executor: Arc<dyn crate::synthetic::ProtocolV2ExpertExecutor> = Arc::new(Echo);
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            let _ = runtime.block_on(super::super::serve_protocol_v2_verbs_host_with_executor(&addr.to_string(), executor));
        });
        std::thread::sleep(Duration::from_millis(300));
        let library = load_verbs_host_native_library()?;
        let plane_bytes = rows as usize * hidden as usize * 2;
        for registered in [false, true] {
            let mut client = LocalTp4Client::new_ranks(vec![addr], TcpTransportConfig {
                timing: false, timeout: Duration::from_secs(30), max_frame_bytes: 64 << 20 }, 0);
            if registered { client.enable_egress(plane_bytes)?; }
            let plane = if gpu {
                let buffer = library.alloc_device_buffer(plane_bytes)?;
                client.set_landing(vec![Some(DeviceLanding::new(buffer))])?;
                Some(buffer)
            } else { None };
            let result = (|| -> Result<()> {
                for index in 0..3u64 {
                    let expected: Vec<u8> = (0..plane_bytes).map(|i| ((i * 7 + index as usize) % 253) as u8).collect();
                    let payload: bytes::Bytes = if registered {
                        let target = client.egress_target()?;
                        // SAFETY: egress_target returned a writable slot of `plane_bytes`
                        // that no request views and no send still reads.
                        unsafe { std::slice::from_raw_parts_mut(target.ptr.cast::<u8>(), plane_bytes) }
                            .copy_from_slice(&expected);
                        client.egress_payload(plane_bytes)?
                    } else { expected.clone().into() };
                    let request = bf16_request(index + 1, rows, hidden, payload)?;
                    let wire = request.wire_stats().wire_bytes;
                    anyhow::ensure!(wire > 16 << 20, "request of {wire} bytes is not large");
                    let sent_before = client.sessions[0].as_ref().map_or(0, |s| s.request_sends.in_flight());
                    client.dispatch(&request, None)?;
                    let session = client.sessions[0].as_ref().context("session opened by dispatch")?;
                    // Nothing reaps send completions between dispatch and here.
                    anyhow::ensure!(session.request_sends.in_flight() == sent_before + 1,
                        "a {wire}-byte request posted {} send-queue entries", session.request_sends.in_flight() - sent_before);
                    let mut got = vec![0u8; plane_bytes];
                    let (mut frames, mut landed) = (0usize, false);
                    while !client.poll(|chunk| {
                        frames += 1;
                        landed |= chunk.landed;
                        anyhow::ensure!(chunk.wire_bytes > 16 << 20, "a {} byte reply frame was split", chunk.wire_bytes);
                        if !chunk.landed { got.copy_from_slice(chunk.partial_output_payload.as_ref()); }
                        Ok(())
                    })? {}
                    if let Some(buffer) = plane.filter(|_| landed) { library.copy_d2h(&mut got, buffer)?; }
                    anyhow::ensure!(frames == 1, "the echo arrived as {frames} receive completions");
                    anyhow::ensure!(got == expected, "request {} echo differs", index + 1);
                }
                eprintln!("loopback gpu={gpu} registered={registered}: 3 x {plane_bytes} B one SEND each way");
                Ok(())
            })();
            client.reset();
            if let Some(mut buffer) = plane { library.free_device_buffer(&mut buffer)?; }
            result?;
        }
        Ok(())
    }
}
