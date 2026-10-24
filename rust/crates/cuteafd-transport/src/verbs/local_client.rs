//! Persistent coordinator QPs progressed by the inference owner. The existing
//! response assembler/recycling code runs locally; no QP worker is spawned.
use super::*;

pub(crate) struct LocalTp4Client {
    peers: Vec<SocketAddr>,
    config: TcpTransportConfig,
    sessions: Vec<Option<VerbsHostProtocolV2PersistentClientSession>>,
    /// Bootstrap seals survive a QP reconnect and are never queried per request.
    expert_grouping: Vec<Option<crate::worker_selection::InstalledExpertGrouping>>,
    retired: Vec<TcpStream>,
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
    /// Per-rank write targets: responses are RDMA-written there (write mode).
    write: Vec<Option<DeviceWriteTarget>>,
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
    pub(crate) fn new_ranks(peers: Vec<SocketAddr>, config: TcpTransportConfig) -> Self {
        Self::with_peers(peers, config)
    }
    fn with_peers(peers: Vec<SocketAddr>, config: TcpTransportConfig) -> Self {
        let world = peers.len();
        // Placed now, before any request is in flight on the fabric.
        let flows = super::flows::FlowSlots::prepare(&peers);
        Self {
            flows,
            peers,
            config,
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
            write: vec![None; world],
            egress: None,
            egress_slot: 0,
            terminal_owner: None,
            terminal_failed: false,
            terminal_quiesced: false,
            terminal_released: false,
        }
    }
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
        let flow_label = self.flows.label(rank, self.peers[rank])?;
        // Only the compact-BF16 wire caps a session's rings by capacity; every
        // other family's rings are the request's own wire extents.
        let capacity = self.capacity.filter(|_| cuteafd_core::expert_geometry().family() == Some("v41")
            && request.header.flags & crate::protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 != 0);
        self.sessions[rank] = Some(VerbsHostProtocolV2PersistentClientSession::connect_local(
            self.peers[rank], &self.config, request, landing,
            if egress { self.egress.as_ref() } else { None },
            self.write[rank], self.terminal_owner.clone(), flow_label, capacity, self.model_endpoint)?);
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
            self.retired.push(stream);
        }
        Ok(())
    }

    fn drain_retired(&mut self) -> Result<()> {
        self.drain_retired_with_timeout(Duration::from_secs(3))
    }

    fn drain_retired_with_timeout(&mut self, timeout: Duration) -> Result<()> {
        if let Some(error) = &self.retirement_error { anyhow::bail!("{error}"); }
        for stream in &mut self.retired {
            let result = (|| -> Result<()> {
                let peer = stream.peer_addr()?;
                wait_for_ring_release(stream, timeout)
                    .with_context(|| format!("expert endpoint peer={peer} did not release its rings"))
            })();
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
    /// Write mode from the next connection on: each rank's responses are
    /// RDMA-written into its target (no receives); drops current sessions.
    pub(crate) fn set_write_targets(&mut self, write: Vec<Option<DeviceWriteTarget>>) -> Result<()> {
        anyhow::ensure!(write.len() == self.peers.len(), "one write target entry per rank");
        self.reset();
        self.write = write;
        Ok(())
    }
    /// Write mode: posts `request` to every rank and returns; the responses
    /// arrive as RDMA writes (their flags tell the GPU). Send completions are
    /// reaped as send slots are reused.
    pub(crate) fn post_written(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        crate::health::ensure_available()?;
        anyhow::ensure!(self.waves.is_empty(), "a received wave is pending on this transport");
        self.flows.ensure_all(&self.peers)?;
        for rank in 0..self.peers.len() {
            anyhow::ensure!(self.write[rank].is_some(), "rank {rank} has no write target");
            if self.sessions[rank].as_ref().map(|s| s.fits(request)).transpose()? == Some(false) {
                self.retire_session(rank)?;
            }
            if self.sessions[rank].is_none() {
                self.drain_retired()?;
                self.open_session(rank, request, None, false)?;
            }
            let session = self.sessions[rank].as_mut().unwrap();
            anyhow::ensure!(session.write_mode, "rank {rank} session is not in write mode");
            // No reply arrives on this QP; the caller's GPU fence is the credit.
            session.request_credits.written_reply_observed();
            session.post_chunk_request(request, &self.config)?;
        }
        Ok(())
    }
    /// Ranks whose live session lands payloads in device memory.
    pub(crate) fn landing_ranks(&self) -> Vec<bool> {
        self.sessions.iter().map(|s| s.as_ref().is_some_and(|s| s.gpu_landing)).collect()
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
    pub(crate) fn dispatch(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        crate::health::ensure_available()?;
        anyhow::ensure!(!self.terminal_failed && !self.terminal_quiesced,
            "terminal-owned Spark transport cannot be reused after failure or quiescence");
        anyhow::ensure!(self.waves.len() < self.max_waves,
            "Spark transport already has {} of {} admitted waves pending", self.waves.len(), self.max_waves);
        let result = self.post(request);
        if let Err(error) = &result {
            crate::health::record_failure(format!("expert dispatch failed: {error:#}"));
            self.reset();
        }
        result
    }
    fn post(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
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
        // Every rank's bootstrap is validated before any request enters a QP.
        let mut uploads = Vec::with_capacity(self.peers.len());
        for session in self.sessions.iter_mut() {
            uploads.push(session.as_mut().unwrap().stage_request(request, &self.config)?);
        }
        // One chunk per rank per round: the coordinator port interleaves every
        // rank's tail instead of bursting whole frames rank after rank.
        let deadline = Instant::now() + self.config.timeout;
        while uploads.iter().any(|upload| !upload.sent()) {
            anyhow::ensure!(Instant::now() < deadline, "Spark request upload deadline expired");
            let mut progressed = false;
            for (session, upload) in self.sessions.iter_mut().zip(&mut uploads) {
                let session = session.as_mut().unwrap();
                progressed |= session.try_reap_request_sends()?;
                progressed |= session.post_request_part(upload)?;
            }
            if !progressed {
                // Every unsent rank's send queue is full: wait on one of them.
                let rank = uploads.iter().position(|upload| !upload.sent()).expect("an unsent upload remains");
                let session = self.sessions[rank].as_mut().unwrap();
                session.post_request_parts(&mut uploads[rank], deadline.saturating_duration_since(Instant::now()))?;
            }
        }
        for (rank, upload) in uploads.into_iter().enumerate() {
            let timing = upload.timing;
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
        self.waves.push_back(OpenWave { chunks: rx, done, deadline: None });
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
        let deadline = *wave.deadline.get_or_insert_with(|| Instant::now() + timeout);
        anyhow::ensure!(
            Instant::now() < deadline,
            "local TP4 response deadline expired"
        );
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

fn wait_for_ring_release(stream: &mut TcpStream, timeout: Duration) -> Result<()> {
    stream.set_read_timeout(Some(timeout))?;
    let mut byte = [0u8; 1];
    match std::io::Read::read(stream, &mut byte) {
        Ok(0) => Ok(()),
        Ok(_) => anyhow::bail!("expert endpoint ring-release acknowledgement contained unexpected data"),
        Err(error) => Err(error).context("expert endpoint ring-release acknowledgement timed out or failed before reconnect"),
    }
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
        let mut transport = LocalTp4Client::new_ranks(Vec::new(), TcpTransportConfig::default());
        transport.retired.push(client);
        let first = transport.drain_retired_with_timeout(Duration::from_millis(10)).unwrap_err();
        let started = Instant::now();
        let second = transport.drain_retired_with_timeout(Duration::from_secs(3)).unwrap_err();
        assert!(started.elapsed() < Duration::from_millis(100));
        assert_eq!(format!("{first:#}"), format!("{second:#}"));
        assert_eq!(transport.retired.len(), 1);
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
        let client = LocalTp4Client::new_ranks(peers.to_vec(), TcpTransportConfig::default());
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
        let mut client = LocalTp4Client::new_ranks(vec![peer], TcpTransportConfig::default());
        assert!(client.bind_model_endpoint(8).is_err());
        assert!(client.model_endpoint.is_none());
        client.bind_model_endpoint(2)?;
        assert!(client.bind_model_endpoint(0).is_err());
        client.reset();
        assert_eq!(client.model_endpoint, Some(2));
        assert!(client.bind_model_endpoint(1).is_err());
        let mut unbound = LocalTp4Client::new_ranks(vec![peer], TcpTransportConfig::default());
        let (_sender, chunks) = tokio::sync::mpsc::unbounded_channel();
        let (_done, completed) = tokio::sync::oneshot::channel();
        unbound.waves.push_back(OpenWave { chunks, done: vec![completed], deadline: None });
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
            timeout: std::time::Duration::from_secs(1), max_frame_bytes: 200_000 });
        assert!(client.set_max_waves(0).is_err() && client.set_max_waves(5).is_err());
        client.set_max_waves(2)?;
        let mut senders = Vec::new();
        for _ in 0..2 {
            let (chunks, receiver) = tokio::sync::mpsc::unbounded_channel();
            let (done, completed) = tokio::sync::oneshot::channel();
            client.waves.push_back(OpenWave { chunks: receiver, done: vec![completed], deadline: None });
            senders.push((chunks, done));
        }
        assert_eq!(client.open_waves(), 2);
        assert!(client.set_max_waves(3).is_err(), "admission changed with waves in flight");
        // A third dispatch is refused before any session or request is touched.
        let request = crate::expert::test_request(1);
        let error = client.dispatch(&request).unwrap_err().to_string();
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
        });
        let (_sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let (_completion, completed) = tokio::sync::oneshot::channel();
        client.waves.push_back(OpenWave { chunks: receiver, done: vec![completed], deadline: Some(Instant::now()) });
        client.reset();
        assert_eq!(client.peers, peers);
        assert_eq!(client.sessions.len(), 2);
        assert_eq!(client.pending.len(), 2);
        assert!(client.sessions.iter().all(Option::is_none));
        assert!(client.pending.iter().all(VecDeque::is_empty));
        assert!(client.waves.is_empty());
        Ok(())
    }
}
