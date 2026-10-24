//! TP2/TP3/TP4/TP6 dispatch through persistent RoCE QPs; TCP is used only for bootstrap.
#[cfg(test)]
use super::BackboneRequest;
use super::{SparkTopology, SparkWorld, V41Tp4ChunkReceiver, V41_NATIVE_GROUP_REQUEST_FLAG};
use crate::verbs::LocalTp4Client;
use crate::{
    DeviceLanding, ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor, ExpertV2SourceKind,
    TcpTransportConfig, VerbsHostProtocolV2ResponsePayload,
};
use anyhow::{ensure, Result};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;

fn poll_quantum(rows: &[ExpertProtocolV2RowDescriptor]) -> std::time::Duration {
    let decode = !rows.is_empty()
        && rows.iter().all(|row| {
            matches!(
                row.source_kind,
                ExpertV2SourceKind::Decode | ExpertV2SourceKind::MtpVerify
            )
        });
    std::time::Duration::from_micros(if decode { 0 } else { 250 })
}

pub struct SparkExperts {
    clients: LocalTp4Client,
    executors: Vec<u64>,
    capacity: u32,
    max_frame_bytes: usize,
    world: SparkWorld,
    /// [`SparkExpertWave`]s dispatched and not yet received, oldest first by
    /// sequence; at most the clients' admitted wave count.
    open_waves: VecDeque<u64>,
    wave_sequence: u64,
    /// Detached receivers must be dropped before cancelled queues can be reset.
    wave_tickets: Arc<()>,
    /// The opt-in route log (`PROFILE=expert-routes`); `None` when off.
    route_tap: Option<crate::route_log::Tap>,
}

/// A dispatched wave that does not borrow its transport, so several
/// transports can have waves in flight at once; complete it with
/// [`SparkExperts::receive_wave`] on the transport that dispatched it. A wave
/// dropped without being received leaves its QPs mid-wave; the transport's
/// next dispatch resets them first.
pub struct SparkExpertWave {
    receiver: V41Tp4ChunkReceiver,
    poll_quantum: std::time::Duration,
    sequence: u64,
    _ticket: Arc<()>,
}

/// How a received wave's rank planes arrived.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WaveReceipt {
    /// Bit r: rank r's whole plane landed in its device range (GPU landing);
    /// the sink saw none of its rows. Other ranks' rows all went to the sink.
    pub landed: u8,
}

impl SparkExperts {
    /// Rank identity and request encoding are declared before connecting.
    pub fn new_world(
        world: SparkWorld,
        peers: &[SocketAddr],
        capacity: u32,
        config: TcpTransportConfig,
    ) -> Result<Self> {
        ensure!(peers.len() == world.topology().world_size(),
            "Spark world requires exactly its physical rank count");
        let executors = world.topology().executor_ids();
        let clients = LocalTp4Client::new_ranks(peers.to_vec(), config.clone());
        Self::with_clients(peers, &executors, capacity, &config, clients, world)
    }

    fn with_clients(
        peers: &[SocketAddr],
        executors: &[u64],
        capacity: u32,
        config: &TcpTransportConfig,
        mut clients: LocalTp4Client,
        world: SparkWorld,
    ) -> Result<Self> {
        ensure!(peers.len() == executors.len() && (1..=cuteafd_core::MAX_SPARK_WORLD).contains(&peers.len()),
            "native TP/EP requires 1..=8 matching peers and executors");
        if let Some(topology) = world.ownership_topology() {
            ensure!(
                peers.len() == topology.world_size(),
                "native group topology requires exactly its physical rank count"
            );
        }
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP capacity"
        );
        ensure!(
            !config.timeout.is_zero(),
            "native RoCE timeout must be positive"
        );
        super::validate_frame_budget(capacity, config.max_frame_bytes)?;
        for rank in 0..peers.len() {
            ensure!(
                executors[rank] != 0 && !executors[..rank].contains(&executors[rank]),
                "native TP executor identities must be distinct and nonzero"
            );
            ensure!(
                !peers[..rank].contains(&peers[rank]),
                "native TP endpoints must be distinct"
            );
        }
        clients.set_capacity(capacity);
        Ok(Self {
            clients,
            executors: executors.to_vec(),
            capacity,
            max_frame_bytes: config.max_frame_bytes,
            world,
            open_waves: VecDeque::new(),
            wave_sequence: 0,
            wave_tickets: Arc::new(()),
            route_tap: crate::route_log::Tap::from_global(),
        })
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn world_size(&self) -> usize {
        self.executors.len()
    }
    /// Bind this transport to one immutable ModelAdmission endpoint on every
    /// rank before bootstrap or dispatch. Reconnect retains the same identity;
    /// missing, incompatible or busy endpoints fail without automatic fallback.
    pub fn bind_model_endpoint(&mut self, endpoint: u32) -> Result<()> {
        ensure!(self.open_waves.is_empty(), "cannot bind endpoint with waves in flight");
        self.clients.bind_model_endpoint(endpoint)
    }
    /// Admit up to `waves` outstanding [`SparkExpertWave`]s, received oldest
    /// first. The workers already serve each QP's requests in post order.
    pub fn set_max_waves(&mut self, waves: usize) -> Result<()> {
        ensure!(self.open_waves.is_empty(), "cannot change wave admission with waves in flight");
        self.clients.set_max_waves(waves)
    }
    pub fn max_waves(&self) -> usize { self.clients.max_waves() }
    pub fn open_waves(&self) -> usize { self.open_waves.len() }
    /// Immutable metadata returned by the workers' completed persistent bootstrap.
    pub fn installed_grouping(&self) -> Result<Vec<crate::worker_selection::InstalledExpertGrouping>> {
        self.clients.installed_grouping()
    }
    /// Completes every rank's persistent bootstrap without dispatching a wave,
    /// so [`Self::installed_grouping`] answers before the first request and the
    /// rings every later request needs are already registered. Call it after
    /// [`Self::enable_terminal_ownership`]; the sessions stay open for the
    /// waves that follow.
    ///
    /// `request` sizes the registered rings, so it must be the widest wave this
    /// transport will dispatch — the same one its start-up warm-up sends. Only
    /// the control bootstrap reaches a worker; the rows are never sent.
    pub fn bootstrap(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        ensure!(self.open_waves.is_empty(), "cannot bootstrap a Spark transport with a wave in flight");
        ensure!(request.header.row_count <= self.capacity,
            "Spark bootstrap request of {} rows exceeds the transport's {}-row capacity",
            request.header.row_count, self.capacity);
        self.clients.bootstrap(request)
    }
    /// Physical ownership and request wire bound before any connection.
    pub fn world(&self) -> SparkWorld { self.world }
    pub fn ownership_topology(&self) -> Option<SparkTopology> {
        self.world.ownership_topology()
    }
    /// Reset persistent QPs. Detached wave queue entries are retained; reusable
    /// intake owners use `reset_drained` after dropping every detached ticket.
    pub fn reset_connections(&mut self) {
        self.clients.reset();
    }

    /// Checks that cancelled detached waves no longer have live receivers.
    /// Terminal-owned transports require terminal quiescence/release, not reuse.
    pub fn require_drained_reset(&self) -> Result<()> {
        ensure!(Arc::strong_count(&self.wave_tickets) == 1,
            "cannot reset a Spark transport with outstanding wave tickets");
        ensure!(!self.terminal_owned(), "terminal-owned Spark transport cannot be reset for reuse");
        Ok(())
    }

    /// Reset after all CPU/GPU consumers have drained. Dropped unread waves may
    /// leave queue entries, but a live ticket makes this a non-mutating error.
    pub fn reset_drained(&mut self) -> Result<()> {
        self.require_drained_reset()?;
        self.clients.reset();
        self.open_waves.clear();
        self.wave_sequence = 0;
        Ok(())
    }

    /// Opt-in owner contract for a terminal engine. Error paths retain pending
    /// slots instead of resetting them before queued CUDA consumers drain.
    pub fn enable_terminal_ownership(&mut self) -> Result<()> {
        self.clients.enable_terminal_ownership()
    }
    #[cfg(test)]
    pub(super) fn terminal_ready_for_test(&self) -> bool { self.clients.terminal_ready_for_test() }
    pub fn terminal_owned(&self) -> bool { self.clients.terminal_owned() }
    pub fn terminal_released(&self) -> bool { self.clients.terminal_released() }
    pub fn terminal_quiesce(&mut self) -> Result<()> { self.clients.terminal_quiesce() }
    pub fn terminal_release(&mut self) -> Result<()> { self.clients.terminal_release() }

    /// Receive each rank's response payloads straight into its device range
    /// (`planes[rank]`, registered with the NIC over dma-buf) from the next
    /// connection on; `None` returns to pinned host receives. A rank whose
    /// registration fails keeps host receives; [`WaveReceipt`] says which
    /// ranks landed. Only waves received with [`Self::receive_wave`] or
    /// [`Self::receive_wave_owned`] accept landed planes.
    ///
    /// # Safety
    /// Every range must stay allocated until this transport is dropped or its
    /// landing is replaced, and hold a whole wave (`capacity` rows). A wave's
    /// planes are written from its dispatch until its receive returns; the
    /// caller must not read them before that, and must finish every read
    /// (including queued GPU work) before dispatching the next wave.
    pub unsafe fn set_gpu_landing(&mut self, planes: Option<Vec<DeviceLanding>>) -> Result<()> {
        let world = self.world_size();
        let landing = match planes {
            Some(planes) => {
                ensure!(planes.len() == world, "GPU landing needs one device range per rank");
                planes.into_iter().map(Some).collect()
            }
            None => vec![None; world],
        };
        self.open_waves.clear();
        self.clients.set_landing(landing)
    }

    /// Write mode from the next connection on (`None` restores receives):
    /// rank r's responses are RDMA-written into `targets[r]` and flagged
    /// there; [`Self::post_written`] posts a wave and returns.
    ///
    /// # Safety
    /// Every target range stays allocated until this transport is dropped or
    /// its targets are replaced; the planes are written from each post until
    /// the rank's flag shows its request id.
    pub unsafe fn set_write_targets(&mut self, targets: Option<Vec<crate::DeviceWriteTarget>>) -> Result<()> {
        let world = self.world_size();
        let write = match targets {
            Some(targets) => {
                ensure!(targets.len() == world, "write mode needs one target per rank");
                targets.into_iter().map(Some).collect()
            }
            None => vec![None; world],
        };
        self.open_waves.clear();
        self.clients.set_write_targets(write)
    }

    /// Write mode: posts `request` to every rank without waiting for anything.
    pub fn post_written(&mut self, request: &ExpertProtocolV2Request) -> Result<()> {
        if !self.open_waves.is_empty() {
            self.open_waves.clear();
            self.reset_connections();
        }
        ensure!((request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0)
                == self.world.ownership_topology().is_some(),
            "Spark request ownership flag differs from its declared wire");
        crate::health::ensure_available()?;
        let result = self.clients.post_written(request);
        if let Err(error) = &result { crate::health::record_failure(format!("expert write failed: {error:#}")); }
        if result.is_err() {
            self.reset_connections();
        } else if let Some(tap) = &mut self.route_tap {
            tap.record(request, self.executors.len());
        }
        result
    }

    /// Zero-copy request egress: allocates a pinned, device-mapped buffer of
    /// `bytes` that every rank's session registers (from the next connection
    /// on). Requests whose hidden payload comes from [`Self::egress_payload`]
    /// are sent without copying it into each rank's send ring. Returns the
    /// buffer, for the engine's D2H target.
    pub fn enable_egress(&mut self, bytes: usize) -> Result<cuteafd_ffi::CuteafdHostBuffer> {
        self.open_waves.clear();
        self.clients.enable_egress(bytes)
    }

    /// The egress buffer to write the next wave's payload into: fails unless
    /// no request views it any more; waits for the previous requests' sends.
    pub fn egress_target(&mut self) -> Result<cuteafd_ffi::CuteafdHostBuffer> {
        self.clients.egress_target()
    }

    /// The first `len` bytes of the egress buffer as a request's hidden payload.
    pub fn egress_payload(&self, len: usize) -> Result<bytes::Bytes> {
        self.clients.egress_payload(len)
    }

    /// Ranks whose current connection lands payloads in device memory
    /// (connections open on the first dispatch after a reset).
    pub fn gpu_landing_ranks(&self) -> Vec<bool> {
        self.clients.landing_ranks()
    }

    /// Send the same canonical request to all ranks and accept every route row.
    /// The synchronous sink must consume/copy its frame slice before returning.
    /// It runs on the polling thread, allowing CUDA owners to stay thread-local.
    /// On error or cancellation the caller must discard partial destination state;
    /// the streaming QP path does not replay partially delivered requests.
    pub async fn execute<F>(&mut self, request: &ExpertProtocolV2Request, sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        self.dispatch(request).await?.receive(sink).await
    }

    /// Post every active rank's request directly from the inference owner. Remote work
    /// overlaps the shared FFN; dispatch completion is not send completion.
    ///
    /// Request flags must match the declared wire: ownership worlds require
    /// native group requests, while canonical worlds reject ownership flags.
    pub async fn dispatch<'c, 'r>(
        &'c mut self,
        request: &'r ExpertProtocolV2Request,
    ) -> Result<SparkExpertPending<'c, 'r>> {
        ensure!(self.open_waves.is_empty() || self.clients.max_waves() == 1,
            "a borrowed Spark dispatch cannot follow outstanding waves");
        if !self.open_waves.is_empty() { self.open_waves.clear(); self.reset_connections(); }
        let receiver = self.post(request)?;
        Ok(SparkExpertPending {
            receiver,
            poll_quantum: poll_quantum(&request.rows),
            owner: self,
            _request: std::marker::PhantomData,
            complete: false,
        })
    }

    /// [`Self::dispatch`] returning an owned [`SparkExpertWave`]. The clients
    /// keep their own copy of the request, so it need not outlive the wave.
    /// Up to [`Self::max_waves`] may be outstanding; receive them in dispatch order.
    pub fn dispatch_wave(&mut self, request: &ExpertProtocolV2Request) -> Result<SparkExpertWave> {
        let receiver = self.post(request)?;
        self.wave_sequence += 1;
        self.open_waves.push_back(self.wave_sequence);
        Ok(SparkExpertWave { receiver, poll_quantum: poll_quantum(&request.rows), sequence: self.wave_sequence,
            _ticket: self.wave_tickets.clone() })
    }

    /// Accepts every row of `wave`, which this transport dispatched; the sink
    /// contract is [`SparkExpertPending::receive`]'s. Ranks that landed in
    /// device memory ([`Self::set_gpu_landing`]) bypass the sink.
    pub async fn receive_wave<F>(&mut self, wave: SparkExpertWave, mut sink: F) -> Result<WaveReceipt>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        self.receive_wave_owned(wave, |rank, start, payload| sink(rank, start, payload.as_ref())).await
    }

    /// [`Self::receive_wave`] handing the sink each payload's owner, as
    /// [`SparkExpertPending::receive_owned`] does: a retained receive slot must
    /// be released before this transport's next dispatch.
    pub async fn receive_wave_owned<F>(&mut self, mut wave: SparkExpertWave, mut sink: F) -> Result<WaveReceipt>
    where
        F: FnMut(usize, u32, VerbsHostProtocolV2ResponsePayload) -> Result<()>,
    {
        ensure!(self.open_waves.front() == Some(&wave.sequence),
            "Spark waves must be received in dispatch order");
        // Keep the polling loop in this owned receive future: decode previously
        // lost ~3% when an extra drain future sat on this hot path. All shared
        // Spark intakes use this path, including device-landing receipts.
        let mut receipt = WaveReceipt::default();
        let mut quantum = std::time::Instant::now();
        let mut first_wait = true;
        let result = loop {
            let polled = self.clients.poll(|chunk| {
                if chunk.landed {
                    let rank = wave.receiver.push_landed(&chunk)?;
                    ensure!(rank == chunk.stream_id, "native executor identity does not match its RoCE peer");
                    receipt.landed |= 1 << rank;
                    return Ok(());
                }
                let mut location = None;
                wave.receiver.push_rdma(&chunk, |rank, start, _bytes| {
                    ensure!(rank == chunk.stream_id,
                        "native executor identity does not match its RoCE peer");
                    location = Some((rank, start));
                    Ok(())
                })?;
                let (rank, start) = location.expect("validated chunk has a location");
                sink(rank, start, chunk.partial_output_payload)
            });
            match polled {
                Err(error) => break Err(error),
                Ok(true) => break if wave.receiver.complete() { Ok(receipt) }
                    else { Err(anyhow::anyhow!("native TP RoCE response coverage is incomplete")) },
                Ok(false) => {}
            }
            // Let another lane run at the first wait; later polls keep this
            // wave's existing decode/prefill quantum. Ready responses never yield.
            if first_wait || quantum.elapsed() >= wave.poll_quantum {
                first_wait = false;
                tokio::task::yield_now().await;
                quantum = std::time::Instant::now();
            } else {
                std::hint::spin_loop();
            }
        };
        if let Err(error) = &result {
            crate::health::record_failure(format!("expert receive failed: {error:#}"));
            self.open_waves.clear();
            self.reset_connections();
        } else {
            self.open_waves.pop_front();
        }
        result
    }

    fn post(&mut self, request: &ExpertProtocolV2Request) -> Result<V41Tp4ChunkReceiver> {
        crate::health::ensure_available()?;
        // A dropped wave is not received; at single-wave admission the next
        // post resets its QPs as before. Past the admitted count, refuse.
        if !self.open_waves.is_empty() && self.open_waves.len() >= self.clients.max_waves() {
            if self.clients.max_waves() > 1 {
                anyhow::bail!("Spark transport has {} outstanding waves, the admitted maximum",
                    self.open_waves.len());
            }
            self.open_waves.clear();
            self.reset_connections();
        }
        let flagged = request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0;
        let receiver = match self.world.ownership_topology() {
            Some(topology) => {
                ensure!(
                    flagged,
                    "topology-bound native transport requires a native group request"
                );
                V41Tp4ChunkReceiver::from_owned_topology(
                    request,
                    self.capacity,
                    &self.executors,
                    topology,
                    self.max_frame_bytes,
                )?
            }
            None => {
                ensure!(
                    !flagged,
                    "native group request requires a topology-bound transport"
                );
                if self.executors.len() == 4 {
                    V41Tp4ChunkReceiver::from_owned(
                        request,
                        self.capacity,
                        self.executors.as_slice().try_into().expect("four executors"),
                        self.max_frame_bytes,
                    )?
                } else {
                    V41Tp4ChunkReceiver::from_owned_ranks(
                        request,
                        self.capacity,
                        &self.executors,
                        self.max_frame_bytes,
                    )?
                }
            }
        };
        self.clients.dispatch(request)?;
        if let Some(tap) = &mut self.route_tap {
            tap.record(request, self.executors.len());
        }
        Ok(receiver)
    }
}

/// Holds exclusive admission until every active rank plane has been consumed.
/// Cancellation resets the QPs before another wave can reuse them.
pub struct SparkExpertPending<'c, 'r> {
    receiver: V41Tp4ChunkReceiver,
    poll_quantum: std::time::Duration,
    owner: &'c mut SparkExperts,
    _request: std::marker::PhantomData<&'r ExpertProtocolV2Request>,
    complete: bool,
}
impl SparkExpertPending<'_, '_> {
    pub async fn receive<F>(self, mut sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, &[u8]) -> Result<()>,
    {
        self.receive_owned(|rank, start, payload| sink(rank, start, payload.as_ref())).await
    }
    /// Transfer validated payload ownership to the sink. Retain each payload
    /// until any asynchronous consumer completes, including on failure. A sink
    /// error abandons this whole wave and resets its QPs; it cannot be resumed.
    pub async fn receive_owned<F>(mut self, mut sink: F) -> Result<()>
    where
        F: FnMut(usize, u32, crate::VerbsHostProtocolV2ResponsePayload) -> Result<()>,
    {
        // Borrowed legacy callers retain their inline loop. Shared intake uses
        // receive_wave_owned's inline loop; neither adds a separate drain future
        // (that extra future previously cost ~3% at C1, 153.5 -> 148.4 tok/s).
        let mut quantum = std::time::Instant::now();
        let mut first_wait = true;
        loop {
            let receiver = &mut self.receiver;
            if self.owner.clients.poll(|chunk| {
                let mut location = None;
                receiver.push_rdma(&chunk, |rank, start, _bytes| {
                    ensure!(
                        rank == chunk.stream_id,
                        "native executor identity does not match its RoCE peer"
                    );
                    location = Some((rank, start));
                    Ok(())
                })?;
                let (rank, start) = location.expect("validated chunk has a location");
                sink(rank, start, chunk.partial_output_payload)
            })? {
                break;
            }
            if first_wait || quantum.elapsed() >= self.poll_quantum {
                first_wait = false;
                tokio::task::yield_now().await;
                quantum = std::time::Instant::now();
            } else {
                std::hint::spin_loop();
            }
        }
        if !self.receiver.complete() {
            crate::health::record_failure("native TP RoCE response coverage is incomplete");
            anyhow::bail!("native TP RoCE response coverage is incomplete");
        }
        self.complete = true;
        Ok(())
    }
}
impl Drop for SparkExpertPending<'_, '_> {
    fn drop(&mut self) {
        if !self.complete {
            self.owner.reset_connections();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VerbsHostProtocolV2ResponseChunk, VerbsHostProtocolV2ResponsePayload};

    #[test]
    fn tp2_constructor_validates_real_peers_and_identities() -> Result<()> {
        let peers = ["127.0.0.1:19441".parse()?, "127.0.0.1:19442".parse()?];
        let config = TcpTransportConfig { timing: false,
            timeout: std::time::Duration::from_secs(1), max_frame_bytes: 200_000,
        };
        let mut client = SparkExperts::new_world(crate::expert::SparkWorld::canonical(peers.len())?, &peers, 2, config.clone())?;
        assert_eq!(client.world_size(), 2);
        client.reset_connections();
        assert_eq!(client.world_size(), 2);
        assert!(SparkExperts::new_world(crate::expert::SparkWorld::canonical((&[peers[0]; 2]).len())?, &[peers[0]; 2], 2, config.clone()).is_err());
        assert_eq!(client.world().topology().executor_ids(), [5, 6]);
        Ok(())
    }

    #[test]
    fn drained_reset_refuses_live_waves_but_recovers_dropped_unread_waves() -> Result<()> {
        let peers = ["127.0.0.1:19441".parse()?, "127.0.0.1:19442".parse()?];
        let config = TcpTransportConfig { timing: false,
            timeout: std::time::Duration::from_secs(1), max_frame_bytes: 200_000 };
        let request = super::super::tests::request(2);
        for waves in 1..=4 {
            let mut client = SparkExperts::new_world(SparkWorld::canonical(2)?, &peers, 2, config.clone())?;
            client.set_max_waves(waves)?;
            let mut unread = Vec::new();
            for sequence in 1..=waves as u64 {
                // Build the real detached tickets without opening hardware sessions.
                let receiver = V41Tp4ChunkReceiver::from_owned_ranks(&request, 2, &client.executors, 200_000)?;
                client.open_waves.push_back(sequence);
                client.wave_sequence = sequence;
                unread.push(SparkExpertWave { receiver, poll_quantum: poll_quantum(&request.rows),
                    sequence, _ticket: client.wave_tickets.clone() });
            }
            assert!(client.require_drained_reset().is_err());
            assert!(client.reset_drained().is_err());
            assert_eq!(client.open_waves(), waves);
            assert_eq!(client.wave_sequence, waves as u64);
            let last = unread.pop().unwrap();
            drop(unread);
            assert!(client.reset_drained().is_err(), "one live ticket must still block reuse");
            drop(last);
            client.reset_drained()?;
            assert_eq!(client.open_waves(), 0);
            assert_eq!(client.wave_sequence, 0);
            assert_eq!(client.max_waves(), waves);
            assert_eq!(client.capacity(), 2);
            client.set_max_waves(waves)?;
            client.reset_drained()?;
        }
        let mut terminal = SparkExperts::new_world(SparkWorld::canonical(2)?, &peers, 2, config)?;
        terminal.enable_terminal_ownership()?;
        assert!(terminal.reset_drained().is_err(), "terminal owners must quiesce/release, never reset for reuse");
        assert!(terminal.terminal_ready_for_test());
        Ok(())
    }

    #[test]
    fn tp2_chunks_require_both_exact_rank_planes() -> Result<()> {
        let owned = super::super::tests::request(2);
        let frame = owned.encode()?;
        let native = BackboneRequest::parse(&frame, 2)?;
        let mut receiver = V41Tp4ChunkReceiver::new_tp2(&native, [11, 27], 200_000)?;
        assert!(!receiver.complete());
        let payload = vec![0; super::super::V41_PARTIAL_ROW_BYTES as usize];
        let mut indices = [0];
        let mut make = |id, row| -> Result<Vec<u8>> {
            native.response_chunk(id, row, &payload, &mut indices, 200_000)?.to_owned()?.encode()
        };
        let wrong = make(3, 0)?;
        assert!(receiver.push(&wrong, |_, _, _| panic!("unknown executor reached sink")).is_err());
        for (id, rank) in [(27, 1), (11, 0)] {
            for row in 0..2 {
                let chunk = make(id, row)?;
                let mut stale = crate::ExpertProtocolV2Response::decode(&chunk)?;
                stale.header.request_id += 1;
                assert!(receiver.push(&stale.encode()?, |_, _, _| panic!("stale response reached sink")).is_err());
                // A failed sink must not commit row coverage.
                assert!(receiver.push(&chunk, |_, _, _| anyhow::bail!("injected sink failure")).is_err());
                receiver.push(&chunk, |actual, start, _| {
                    assert_eq!((actual, start), (rank, row)); Ok(())
                })?;
                assert!(receiver.push(&chunk, |_, _, _| panic!("duplicate reached sink")).is_err());
                assert_eq!(receiver.complete(), id == 11 && row == 1);
            }
        }
        assert_eq!(receiver.received_rows(), [2, 2, 0, 0]);
        let mut reserved = owned;
        reserved.header.flags |= 1 << 17;
        assert!(V41Tp4ChunkReceiver::from_owned_ranks(&reserved, 2, &[11, 27], 200_000).is_err());
        Ok(())
    }

    #[test]
    fn prefill_or_benchmark_rows_keep_original_polling_in_mixed_waves() {
        let row = |kind| ExpertProtocolV2RowDescriptor {
            row_id: 0,
            source_kind: kind,
            source_request_id: 1,
            token_position: 0,
            route_offset: 0,
            route_count: 6,
        };
        use ExpertV2SourceKind::{Benchmark, Decode, MtpVerify, Prefill};
        assert_eq!(poll_quantum(&[row(Decode), row(MtpVerify)]).as_micros(), 0);
        for other in [Prefill, Benchmark] {
            assert_eq!(poll_quantum(&[row(other)]).as_micros(), 250);
            assert_eq!(
                poll_quantum(&[row(Decode), row(other), row(MtpVerify)]).as_micros(),
                250
            );
        }
        assert_eq!(poll_quantum(&[]).as_micros(), 250);
    }

    #[test]
    #[ignore = "requires four idle live V4.1 FP8 RoCE expert workers"]
    fn local_qps_replay_cancel_and_recover_live() -> Result<()> {
        replay_live(false)
    }

    #[test]
    #[ignore = "requires four idle ABI-3 V4.1 atomic-output RoCE expert workers"]
    fn local_qps_atomic_prefill_cancel_and_recover_live() -> Result<()> {
        replay_live(true)
    }

    // A separate numerical gate for atomic prefill outputs. Protocol identity,
    // row/rank coverage and small ordered outputs keep their exact checks.
    fn atomic_planes_close(previous: &[Vec<u8>], current: &[Vec<u8>]) -> Result<()> {
        ensure!(previous.len() == current.len(), "rank count changed");
        for (rank, (old, new)) in previous.iter().zip(current).enumerate() {
            ensure!(!old.is_empty() && old.len() == new.len() && old.len() % 2 == 0,
                "atomic plane extent changed");
            let (mut error, mut energy, mut changed) = (0.0_f64, 0.0_f64, 0_usize);
            for (a, b) in old.chunks_exact(2).zip(new.chunks_exact(2)) {
                let decode = |v: &[u8]| f32::from_bits((u16::from_le_bytes([v[0],v[1]]) as u32) << 16) as f64;
                let (a, b) = (decode(a), decode(b));
                ensure!(a.is_finite() && b.is_finite(), "nonfinite atomic replay");
                let delta = (a-b).abs();
                ensure!(delta <= 2e-5 + a.abs().max(b.abs())/128.0,
                    "atomic replay exceeds one BF16 rounding step plus absolute floor");
                error += delta*delta;
                energy += a*a;
                changed += usize::from(a != b);
            }
            ensure!(changed*1000 <= old.len()/2, "too many atomic replay values changed");
            ensure!(error <= energy.max(1e-30)*2.5e-7, "atomic replay relative L2 exceeds 5e-4");
            eprintln!("atomic rank={rank}: changed={changed}/{} rel_l2={}", old.len()/2,
                (error/energy.max(1e-30)).sqrt());
        }
        Ok(())
    }

    #[test]
    fn atomic_replay_gate_rejects_corruption_and_broad_changes() {
        let plane = |bits: u16| vec![bits.to_le_bytes();2000].concat();
        let baseline = vec![plane(0x3f80)];
        let mut rounded = baseline.clone();
        rounded[0][..2].copy_from_slice(&0x3f81_u16.to_le_bytes());
        assert!(atomic_planes_close(&baseline, &rounded).is_ok());
        rounded[0][..2].copy_from_slice(&0x4080_u16.to_le_bytes());
        assert!(atomic_planes_close(&baseline, &rounded).is_err());
        let mut broad = baseline.clone();
        broad[0][..6].copy_from_slice(&[0x81,0x3f,0x81,0x3f,0x81,0x3f]);
        assert!(atomic_planes_close(&baseline, &broad).is_err());
        assert!(atomic_planes_close(&baseline, &[plane(0x7fc0)]).is_err());
        assert!(atomic_planes_close(&baseline, &[vec![0;2]]).is_err());
    }

    fn replay_live(atomic: bool) -> Result<()> {
        let peers: [SocketAddr; 4] = std::env::var("CUTEAFD_LIVE_ROCE_PEERS")?
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("four peers required"))?;
        let capacity: u32 = std::env::var("CUTEAFD_LIVE_ROCE_CAPACITY")
            .unwrap_or_else(|_| if atomic { "4096" } else { "80" }.into()).parse()?;
        ensure!([80, 256, 1024, 4096].contains(&capacity), "unsupported fixture capacity");
        ensure!(!atomic || capacity >= 256, "atomic fixture requires prefill capacity");
        let mut shapes = vec![1, 6, 16, 80];
        shapes.extend([256, 1024, 4096].into_iter().filter(|&rows| rows <= capacity));
        shapes.extend([6, 1]);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            let mut client = SparkExperts::new_world(crate::expert::SparkWorld::canonical(peers.len())?, &peers, capacity, TcpTransportConfig { timing: false,
                    timeout: std::time::Duration::from_secs(10),
                    max_frame_bytes: 64 * 1024 * 1024,
                })?;
            for rows in shapes {
                let base = super::super::tests::request(rows);
                let mut payload = Vec::new();
                for row in 0..rows {
                    payload.extend(
                        (0..5120).map(|i| {
                            0x30 + ((i + row) % 8) as u8 + if i % 2 == 0 { 0x80 } else { 0 }
                        }),
                    );
                    payload.extend([120; 160]);
                }
                let mut request = ExpertProtocolV2Request::new(
                    1000 + rows as u64,
                    base.header.placement_version,
                    39,
                    5120,
                    crate::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                    base.rows,
                    base.routes,
                    payload,
                )?;
                request.header.flags = base.header.flags;
                let mut expected: Option<Vec<Vec<u8>>> = None;
                for repetition in 0..3 {
                    request.header.request_id += 1;
                    let mut planes = vec![vec![0; rows as usize * 10240]; 4];
                    client
                        .execute(&request, |rank, start, bytes| {
                            let offset = start as usize * 10240;
                            planes[rank][offset..offset + bytes.len()].copy_from_slice(bytes);
                            Ok(())
                        })
                        .await?;
                    for plane in &planes {
                        ensure!(
                            plane
                                .chunks_exact(2)
                                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7fff != 0),
                            "zero expert plane"
                        );
                        ensure!(
                            plane
                                .chunks_exact(2)
                                .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
                            "nonfinite expert plane"
                        );
                    }
                    if let Some(ref previous) = expected {
                        if atomic && rows > 80 {
                            atomic_planes_close(previous, &planes)?;
                        } else {
                            ensure!(previous == &planes, "replay changed rank planes");
                        }
                    }
                    expected = Some(planes);
                    if repetition == 0 {
                        request.header.request_id += 1;
                        // Abandon after posting all ranks, before receiving any.
                        drop(client.dispatch(&request).await?);
                    } else if repetition == 1 {
                        request.header.request_id += 1;
                        let failed = client
                            .execute(&request, |_, _, _| anyhow::bail!("injected sink failure"))
                            .await;
                        ensure!(failed.is_err(), "sink failure was swallowed");
                    }
                }
                eprintln!(
                    "local QPs rows={rows}: {} replay after abandoned dispatch and sink failure",
                    if atomic && rows > 80 { "bounded numerical" } else { "exact" }
                );
            }
            Ok(())
        })
    }

    #[test]
    fn gpu_landed_headers_complete_whole_rank_planes_only() -> Result<()> {
        let request = super::super::tests::request(2);
        let frame = request.encode()?;
        let native = BackboneRequest::parse(&frame, 2)?;
        let mut receiver = V41Tp4ChunkReceiver::new(&native, [1, 2, 3, 4], 200_000)?;
        let plane = vec![7u8; 2 * super::super::V41_PARTIAL_ROW_BYTES as usize];
        let header_len = crate::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let landed = |executor: u64| -> Result<VerbsHostProtocolV2ResponseChunk> {
            let encoded = native.response(executor, &plane)?.to_owned()?.encode()?;
            let header = crate::ExpertProtocolV2ResponseView::parse_landed_header(&encoded[..header_len])?;
            Ok(VerbsHostProtocolV2ResponseChunk {
                stream_id: executor as usize - 1,
                header,
                row_indices: None,
                partial_output_payload: VerbsHostProtocolV2ResponsePayload::from_owned(Vec::new()),
                wire_bytes: encoded.len(),
                landed: true,
            })
        };
        // A row-indexed (chunked) frame cannot land: its indices would be
        // scattered into device memory ahead of the rows.
        let mut indices = [0];
        let chunked = native.response_chunk(1, 0, &plane[..plane.len() / 2], &mut indices, 200_000)?
            .to_owned()?.encode()?;
        assert!(crate::ExpertProtocolV2ResponseView::parse_landed_header(&chunked[..header_len]).is_err());
        // A header whose payload extent disagrees with its rows is rejected.
        let mut bad = native.response(1, &plane)?.to_owned()?.encode()?;
        bad[44..52].copy_from_slice(&1u64.to_le_bytes());
        assert!(crate::ExpertProtocolV2ResponseView::parse_landed_header(&bad[..header_len]).is_err());
        for executor in [3, 1, 4, 2] {
            let chunk = landed(executor)?;
            assert!(receiver.push_rdma(&chunk, |_, _, _| panic!("landed rows reached the sink")).is_err());
            assert_eq!(receiver.push_landed(&chunk)?, executor as usize - 1);
            assert!(receiver.push_landed(&chunk).is_err(), "a rank lands once per wave");
        }
        assert!(receiver.complete());
        // A host chunk is not a landed one.
        let mut host = landed(1)?;
        host.landed = false;
        let mut fresh = V41Tp4ChunkReceiver::new(&native, [1, 2, 3, 4], 200_000)?;
        assert!(fresh.push_landed(&host).is_err());
        Ok(())
    }

    #[test]
    fn eight_rank_landed_receipt_uses_all_bits_without_overflow() -> Result<()> {
        let request = super::super::tests::request(2);
        let frame = request.encode()?;
        let native = BackboneRequest::parse(&frame, 2)?;
        let executors = super::SparkWorld::canonical(cuteafd_core::MAX_SPARK_WORLD)?.topology().executor_ids();
        let mut receiver = V41Tp4ChunkReceiver::new_ranks(&native, &executors, 200_000)?;
        let plane = vec![7u8; native.plane_bytes()?];
        let mut receipt = WaveReceipt::default();
        for rank in (0..executors.len()).rev() {
            let encoded = native.response(executors[rank], &plane)?.to_owned()?.encode()?;
            let header = crate::ExpertProtocolV2ResponseView::parse_landed_header(
                &encoded[..crate::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN])?;
            let chunk = VerbsHostProtocolV2ResponseChunk { stream_id: rank, header, row_indices: None,
                partial_output_payload: VerbsHostProtocolV2ResponsePayload::from_owned(Vec::new()),
                wire_bytes: encoded.len(), landed: true };
            let selected = receiver.push_landed(&chunk)?;
            assert_eq!(selected, rank);
            receipt.landed |= 1 << selected;
        }
        assert!(receiver.complete());
        assert_eq!(receipt.landed, u8::MAX);
        Ok(())
    }

    #[test]
    fn rdma_chunks_preserve_rank_rows_and_reject_stale_or_reordered_data() -> Result<()> {
        let request = super::super::tests::request(2);
        let frame = request.encode()?;
        let native = BackboneRequest::parse(&frame, 2)?;
        let mut receiver = V41Tp4ChunkReceiver::new(&native, [1, 2, 3, 4], 200_000)?;
        for row in 0..2 {
            for rank in [3, 1, 0, 2] {
                let bytes = vec![(rank + row) as u8; super::super::V41_PARTIAL_ROW_BYTES as usize];
                let mut indices = [0];
                let response = native
                    .response_chunk(rank as u64 + 1, row, &bytes, &mut indices, 200_000)?
                    .to_owned()?;
                let wire_bytes = response.encode()?.len();
                let mut chunk = VerbsHostProtocolV2ResponseChunk {
                    stream_id: rank as usize,
                    header: response.header,
                    row_indices: Some(vec![row]),
                    partial_output_payload: VerbsHostProtocolV2ResponsePayload::from_owned(
                        bytes.clone(),
                    ),
                    wire_bytes,
                    landed: false,
                };
                chunk.header.request_id += 1;
                assert!(receiver
                    .push_rdma(&chunk, |_, _, _| panic!("stale data reached sink"))
                    .is_err());
                chunk.header.request_id -= 1;
                chunk.row_indices = Some(vec![1 - row]);
                assert!(receiver
                    .push_rdma(&chunk, |_, _, _| panic!("reordered data reached sink"))
                    .is_err());
                chunk.row_indices = Some(vec![row]);
                receiver.push_rdma(&chunk, |r, start, payload| {
                    assert_eq!((r, start), (rank as usize, row));
                    assert_eq!(payload, bytes);
                    Ok(())
                })?;
                assert!(receiver
                    .push_rdma(&chunk, |_, _, _| panic!("duplicate data reached sink"))
                    .is_err());
            }
        }
        assert!(receiver.complete());
        Ok(())
    }
}
