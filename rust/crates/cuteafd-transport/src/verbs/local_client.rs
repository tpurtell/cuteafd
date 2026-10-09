//! Persistent coordinator QPs progressed by the inference owner. The existing
//! response assembler/recycling code runs locally; no QP worker is spawned.
use super::*;

/// Requests up to this size go out rank after rank under a staggered egress.
/// Posted alone, one rank's zero-copy request completes at ~12-14 GB/s up to
/// 13 MB but at ~3 GB/s at 26 MB (raptor to a Spark, 200 Gb); larger requests
/// go out all at once, where the ranks' queue pairs overlap that.
const STAGGER_MAX_BYTES: usize = 16 << 20;

pub(crate) struct LocalTp4Client {
    peers: Vec<SocketAddr>,
    config: TcpTransportConfig,
    sessions: Vec<Option<VerbsHostProtocolV2PersistentClientSession>>,
    retired: Vec<TcpStream>,
    retirement_error: Option<String>,
    capacity: Option<u32>,
    pending: Vec<VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>>,
    chunks: Option<tokio::sync::mpsc::UnboundedReceiver<VerbsHostProtocolV2ResponseChunk>>,
    done: Vec<tokio::sync::oneshot::Receiver<Result<VerbsHostProtocolV2ResponseStreamStats>>>,
    deadline: Option<Instant>,
    /// Per-rank device ranges response payloads land in (GPU landing).
    landing: Vec<Option<DeviceLanding>>,
    /// Per-rank write targets: responses are RDMA-written there (write mode).
    write: Vec<Option<DeviceWriteTarget>>,
    /// Shared registered request buffer (zero-copy egress).
    egress: Option<Arc<super::egress::EgressBuffer>>,
    /// Post the ranks' zero-copy requests one after another (each once the
    /// previous rank's send completed) rather than all at once.
    stagger: bool,
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
    pub(crate) fn new(peers: [SocketAddr; 4], config: TcpTransportConfig) -> Self {
        Self::with_peers(peers.to_vec(), config)
    }
    pub(crate) fn new_tp2(peers: [SocketAddr; 2], config: TcpTransportConfig) -> Self {
        Self::with_peers(peers.to_vec(), config)
    }
    /// Generic peer set for the validated physical rank counts 2, 3, 4 and 6.
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
            retired: Vec::new(),
            retirement_error: None,
            capacity: None,
            pending: (0..world).map(|_| VecDeque::with_capacity(1)).collect(),
            chunks: None,
            done: Vec::with_capacity(world),
            deadline: None,
            landing: vec![None; world],
            write: vec![None; world],
            egress: None,
            stagger: false,
            terminal_owner: None,
            terminal_failed: false,
            terminal_quiesced: false,
            terminal_released: false,
        }
    }
    pub(crate) fn set_capacity(&mut self, capacity: u32) { self.capacity = Some(capacity); }

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
    pub(crate) fn enable_egress(&mut self, bytes: usize, stagger: bool) -> Result<CuteafdHostBuffer> {
        self.reset();
        let buffer = super::egress::EgressBuffer::new(bytes)?;
        let host = buffer.host();
        self.egress = Some(buffer);
        self.stagger = stagger;
        Ok(host)
    }
    /// The shared request buffer, once no request payload views it and every
    /// session's request sends have left, so it may be rewritten.
    pub(crate) fn egress_target(&mut self) -> Result<CuteafdHostBuffer> {
        let buffer = self.egress.as_ref().context("this transport has no egress buffer")?;
        anyhow::ensure!(buffer.writable(), "a request still views the egress buffer");
        let host = buffer.host();
        for session in self.sessions.iter_mut().flatten() {
            session.drain_request_sends(&self.config)?;
        }
        Ok(host)
    }
    /// The first `len` bytes of the egress buffer as a request payload.
    pub(crate) fn egress_payload(&self, len: usize) -> Result<bytes::Bytes> {
        self.egress.as_ref().context("this transport has no egress buffer")?.payload(len)
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
        anyhow::ensure!(self.deadline.is_none(), "a received wave is pending on this transport");
        self.flows.ensure_all(&self.peers)?;
        for rank in 0..self.peers.len() {
            anyhow::ensure!(self.write[rank].is_some(), "rank {rank} has no write target");
            if self.sessions[rank].as_ref().map(|s| s.fits(request)).transpose()? == Some(false) {
                self.retire_session(rank)?;
            }
            if self.sessions[rank].is_none() {
                self.drain_retired()?;
                let flow_label = self.flows.label(rank, self.peers[rank])?;
                self.sessions[rank] = Some(VerbsHostProtocolV2PersistentClientSession::connect_local(
                    self.peers[rank], &self.config, request, None, None, self.write[rank], self.terminal_owner.clone(),
                    flow_label, self.capacity.filter(|_| cuteafd_core::expert_geometry().family() == Some("v41")
                        && request.header.flags & crate::protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 != 0))?);
            }
            let session = self.sessions[rank].as_mut().unwrap();
            anyhow::ensure!(session.write_mode, "rank {rank} session is not in write mode");
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
        self.chunks = None;
        self.done.clear();
        for pending in &mut self.pending {
            pending.clear();
        }
        for rank in 0..self.sessions.len() {
            if let Err(error) = self.retire_session(rank) {
                crate::health::record_failure(format!("expert session teardown failed: {error:#}"));
            }
        }
        self.deadline = None;
    }
    pub(crate) fn dispatch(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        crate::health::ensure_available()?;
        anyhow::ensure!(!self.terminal_failed && !self.terminal_quiesced,
            "terminal-owned Spark transport cannot be reused after failure or quiescence");
        anyhow::ensure!(self.deadline.is_none(), "local TP4 request already pending");
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
        self.chunks = Some(rx);
        for rank in 0..self.peers.len() {
            if self.sessions[rank]
                .as_ref()
                .map(|s| s.fits(request))
                .transpose()?
                == Some(false)
            {
                anyhow::ensure!(self.terminal_owner.is_none(),
                    "terminal-owned Spark request exceeds its prewarmed session capacity; reconnect is unsupported");
                self.retire_session(rank)?;
            }
            if self.sessions[rank].is_none() {
                self.drain_retired()?;
                let flow_label = self.flows.label(rank, self.peers[rank])?;
                self.sessions[rank] = Some(VerbsHostProtocolV2PersistentClientSession::connect_local(
                    self.peers[rank],
                    &self.config,
                    request,
                    self.landing[rank],
                    self.egress.as_ref(),
                    self.write[rank],
                    self.terminal_owner.clone(),
                    flow_label,
                    self.capacity.filter(|_| cuteafd_core::expert_geometry().family() == Some("v41")
                        && request.header.flags & crate::protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 != 0),
                )?);
            }
            let session = self.sessions[rank].as_mut().unwrap();
            let timing = session.post_chunk_request(request, &self.config)?;
            if self.stagger && rank + 1 < self.peers.len() && request.hidden_payload.len() <= STAGGER_MAX_BYTES
                && session.sends_from_egress(request) {
                // The next rank's request goes out once this one has left: each
                // rank gets the whole link in turn and starts computing as early
                // as it can, so the ranks finish, and their partials land,
                // staggered instead of all at once.
                session.drain_request_sends(&self.config)?;
            }
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
            self.done.push(response_rx);
        }
        // The same-thread channels adapt the existing assembler; they never
        // wake or hand off work to another thread. Each sink takes ownership of its chunk; retained pinned frames
        // return to the session pool only after the owner releases them.
        self.deadline = Some(
            Instant::now()
                .checked_add(self.config.timeout)
                .context("local TP4 deadline overflow")?,
        );
        Ok(())
    }

    pub(crate) fn enable_terminal_ownership(&mut self) -> Result<()> {
        anyhow::ensure!(self.sessions.iter().all(Option::is_none) && self.deadline.is_none(),
            "terminal Spark ownership must precede connection/bootstrap");
        self.terminal_owner = Some(Arc::new(AtomicBool::new(false)));
        Ok(())
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
        let deadline = self.deadline.context("local TP4 has no pending request")?;
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
            let chunks = self
                .chunks
                .as_mut()
                .context("local TP4 response queue missing")?;
            while let Ok(chunk) = chunks.try_recv() {
                sink(chunk)?;
            }
        }
        if self.pending.iter().any(|p| !p.is_empty()) {
            return Ok(false);
        }
        // Final validation and completion publication happen synchronously in
        // accept_chunk_response_frame before it removes a pending request.
        for done in &mut self.done {
            done.try_recv().context("local TP4 completion missing")??;
        }
        self.done.clear();
        self.chunks = None;
        self.deadline = None;
        Ok(true)
    }
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
                std::mem::forget(self.chunks.take());
                std::mem::forget(std::mem::take(&mut self.done));
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

    #[test]
    fn tp2_reset_retains_only_two_actual_peer_slots() -> Result<()> {
        let peers = ["127.0.0.1:19441".parse()?, "127.0.0.1:19442".parse()?];
        let mut client = LocalTp4Client::new_tp2(peers, TcpTransportConfig { timing: false,
            timeout: std::time::Duration::from_secs(1), max_frame_bytes: 200_000,
        });
        let (_sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let (_completion, completed) = tokio::sync::oneshot::channel();
        client.chunks = Some(receiver);
        client.done.push(completed);
        client.deadline = Some(Instant::now());
        client.reset();
        assert_eq!(client.peers, peers);
        assert_eq!(client.sessions.len(), 2);
        assert_eq!(client.pending.len(), 2);
        assert!(client.sessions.iter().all(Option::is_none));
        assert!(client.pending.iter().all(VecDeque::is_empty));
        assert!(client.chunks.is_none() && client.done.is_empty() && client.deadline.is_none());
        Ok(())
    }
}
