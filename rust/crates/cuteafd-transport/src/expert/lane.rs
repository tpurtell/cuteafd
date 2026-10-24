//! A [`SparkExperts`] on a thread of its own, so a prefill lane's request
//! assembly, RDMA post, polling and partial copies overlap the inference
//! owner's GPU work. The transport is created on that thread and never
//! leaves it (its verbs endpoints are not `Send`); jobs carry `Send` closures
//! that build the request and consume the response rows. One job is in
//! flight per lane. The decode hot path keeps using [`SparkExperts`] inline.
use super::{SparkExperts, SparkWorld};
use crate::{DeviceLanding, ExpertProtocolV2Request, TcpTransportConfig, VerbsHostProtocolV2ResponsePayload};
use anyhow::{anyhow, ensure, Context, Result};
use std::net::SocketAddr;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Builds the wave's request on the lane thread.
pub type LaneBuild = Box<dyn FnOnce() -> Result<ExpertProtocolV2Request> + Send>;
/// Receives each `(rank, first row, payload)` of the wave on the lane thread;
/// the payload must be consumed before it returns.
pub type LaneSink = Box<dyn FnMut(usize, u32, &[u8]) -> Result<()> + Send>;
/// Receives each payload's owner on the lane thread (see
/// [`SparkExperts::receive_wave_owned`]); retained receive slots must be
/// released before the lane's next job.
pub type LaneOwnedSink = Box<dyn FnMut(usize, u32, VerbsHostProtocolV2ResponsePayload) -> Result<()> + Send>;

enum Sink {
    Borrowed(LaneSink),
    Owned(LaneOwnedSink),
}

struct Job {
    build: LaneBuild,
    sink: Sink,
}

enum Command {
    Wave(Job),
    Quiesce(mpsc::Sender<Result<()>>),
    Release(mpsc::Sender<Result<()>>),
}

/// Seconds a finished job spent building its request and posting it, and
/// from the post until its last row arrived.
#[derive(Debug, Clone, Copy, Default)]
pub struct LaneTimes {
    pub build_post: f64,
    pub receive: f64,
    /// [`super::WaveReceipt::landed`]: ranks whose plane landed in device memory.
    pub landed: u8,
}

pub struct SparkExpertLane {
    jobs: Option<mpsc::Sender<Command>>,
    done: mpsc::Receiver<Result<LaneTimes>>,
    world: usize,
    pending: bool,
    thread: Option<std::thread::JoinHandle<()>>,
    quiesced: bool,
}

// Landing installation resets the transport, so terminal ownership must be
// enabled afterward, but still before the first connection or bootstrap.
unsafe fn prepare_transport(transport: &mut SparkExperts, landing: Option<Vec<DeviceLanding>>) -> Result<()> {
    if landing.is_some() {
        // SAFETY: the lane's caller retains these ranges until terminal release.
        unsafe { transport.set_gpu_landing(landing)? };
    }
    transport.enable_terminal_ownership()
}

impl SparkExpertLane {
    /// Transport bound to a declared world whose rank payloads land in `landing`.
    ///
    /// # Safety
    /// [`SparkExperts::set_gpu_landing`]'s contract, with a lane's wave running
    /// from [`Self::submit`] until [`Self::wait`] returns.
    pub unsafe fn spawn_world(world: SparkWorld, peers: Vec<SocketAddr>, capacity: u32,
        config: TcpTransportConfig, landing: Option<Vec<DeviceLanding>>) -> Result<Self> {
        // SAFETY: forwarded from this constructor's contract.
        unsafe { Self::spawn_transport(move || SparkExperts::new_world(world, &peers, capacity, config), landing) }
    }

    unsafe fn spawn_transport(build: impl FnOnce() -> Result<SparkExperts> + Send + 'static,
        landing: Option<Vec<DeviceLanding>>) -> Result<Self> {
        let (jobs, inbox) = mpsc::channel::<Command>();
        let (report, done) = mpsc::channel::<Result<LaneTimes>>();
        let (ready_tx, ready) = mpsc::channel::<Result<usize>>();
        let thread = std::thread::Builder::new().name("v41-roce-lane".into()).spawn(move || {
            let setup = build().and_then(|mut transport| {
                // SAFETY: forwarded from this constructor's contract.
                unsafe { prepare_transport(&mut transport, landing)? };
                let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
                Ok((transport, runtime))
            });
            let (mut transport, runtime) = match setup {
                Ok(setup) => setup,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(transport.world_size()));
            while let Ok(command) = inbox.recv() {
                let Job { build, sink } = match command {
                    Command::Quiesce(reply) => {
                        let _ = reply.send(transport.terminal_quiesce());
                        continue;
                    }
                    Command::Release(reply) => {
                        let result = transport.terminal_release();
                        let released = result.is_ok();
                        let _ = reply.send(result);
                        if released { break; }
                        continue;
                    }
                    Command::Wave(job) => job,
                };
                let started = Instant::now();
                let result = build().and_then(|request| transport.dispatch_wave(&request)).and_then(|wave| {
                    let posted = Instant::now();
                    let receipt = match sink {
                        Sink::Borrowed(sink) => runtime.block_on(transport.receive_wave(wave, sink))?,
                        Sink::Owned(sink) => runtime.block_on(transport.receive_wave_owned(wave, sink))?,
                    };
                    Ok(LaneTimes { build_post: (posted - started).as_secs_f64(),
                        receive: posted.elapsed().as_secs_f64(), landed: receipt.landed })
                });
                if report.send(result).is_err() {
                    break;
                }
            }
        })?;
        let world = ready.recv().map_err(|_| anyhow!("RoCE lane thread exited during setup"))??;
        Ok(Self { jobs: Some(jobs), done, world, pending: false, thread: Some(thread), quiesced: false })
    }

    pub fn world_size(&self) -> usize {
        self.world
    }

    /// Queues one wave: `build` runs, the request is posted to every rank
    /// and each response row goes to `sink`, all on the lane thread.
    pub fn submit(&mut self, build: LaneBuild, sink: LaneSink) -> Result<()> {
        self.submit_job(Job { build, sink: Sink::Borrowed(sink) })
    }

    /// [`Self::submit`] with a sink that takes each payload's owner.
    pub fn submit_owned(&mut self, build: LaneBuild, sink: LaneOwnedSink) -> Result<()> {
        self.submit_job(Job { build, sink: Sink::Owned(sink) })
    }

    fn submit_job(&mut self, job: Job) -> Result<()> {
        ensure!(!self.pending, "a RoCE lane holds one wave at a time");
        ensure!(!self.quiesced, "RoCE lane is terminally quiesced");
        self.jobs.as_ref().context("RoCE lane closed")?.send(Command::Wave(job))
            .map_err(|_| anyhow!("RoCE lane thread exited"))?;
        self.pending = true;
        Ok(())
    }

    /// Waits for the submitted wave's last row (or its error).
    pub fn wait(&mut self, timeout: Duration) -> Result<LaneTimes> {
        ensure!(self.pending, "no RoCE lane wave in flight");
        self.pending = false;
        match self.done.recv_timeout(timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(anyhow!("RoCE lane wave timed out after {timeout:?}")),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(anyhow!("RoCE lane thread exited")),
        }
    }

    /// Checks QP destruction on the transport thread without releasing its
    /// pending payloads or registrations. External landing/copy owners must
    /// remain live on any error, including a lost thread or reply.
    pub fn terminal_quiesce(&mut self) -> Result<()> {
        if self.quiesced { return Ok(()); }
        let (reply, result) = mpsc::channel();
        self.jobs.as_ref().context("RoCE lane closed without quiescence proof")?
            .send(Command::Quiesce(reply)).map_err(|_| anyhow!("RoCE lane thread exited before quiescence"))?;
        result.recv().context("RoCE lane thread lost quiescence reply")??;
        self.quiesced = true;
        Ok(())
    }

    /// Releases transport owners only after external compute/copy consumers
    /// have drained. Successful release joins the now-retired lane thread.
    pub fn terminal_release(&mut self) -> Result<()> {
        ensure!(self.quiesced, "RoCE lane release requires checked quiescence");
        let Some(jobs) = &self.jobs else { return Ok(()); };
        let (reply, result) = mpsc::channel();
        jobs.send(Command::Release(reply)).map_err(|_| anyhow!("RoCE lane thread exited before release"))?;
        result.recv().context("RoCE lane thread lost release reply")??;
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| anyhow!("RoCE lane thread panicked during release"))?;
        }
        Ok(())
    }
}

impl Drop for SparkExpertLane {
    fn drop(&mut self) {
        // Closing the job channel ends the thread after its current wave.
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::SparkTopology;
    use super::*;

    #[test]
    fn terminal_ownership_follows_landing_setup_before_connection() -> Result<()> {
        let peers = vec!["127.0.0.1:1".parse()?];
        let mut transport = SparkExperts::new_world(SparkWorld::canonical(1)?, &peers, 1,
            TcpTransportConfig { timing: false, timeout: Duration::from_secs(1), max_frame_bytes: 64 << 20 })?;
        // SAFETY: setup never connects or dereferences these dummy ranges.
        unsafe { prepare_transport(&mut transport, Some(vec![DeviceLanding { ptr: 1, bytes: 10240, slot_bytes: 10240 }]))? };
        assert!(transport.terminal_ready_for_test());
        transport.terminal_quiesce()?;
        transport.terminal_release()?;
        Ok(())
    }

    #[test]
    fn terminal_control_requires_reply_and_preserves_failed_lane() -> Result<()> {
        let (jobs, inbox) = mpsc::channel();
        let (_report, done) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let mut attempts = 0;
            while let Ok(command) = inbox.recv() {
                match command {
                    Command::Quiesce(reply) => {
                        attempts += 1;
                        let result = if attempts == 1 { Err(anyhow!("injected QP destroy failure")) } else { Ok(()) };
                        reply.send(result).unwrap();
                    }
                    Command::Release(reply) => { reply.send(Ok(())).unwrap(); break; }
                    Command::Wave(_) => panic!("terminal lane dispatched a wave"),
                }
            }
        });
        let mut lane = SparkExpertLane { jobs: Some(jobs), done, world: 1,
            pending: false, thread: Some(thread), quiesced: false };
        assert!(lane.terminal_release().is_err());
        assert!(lane.terminal_quiesce().unwrap_err().to_string().contains("injected"));
        assert!(!lane.quiesced && lane.jobs.is_some() && lane.thread.is_some());
        lane.terminal_quiesce()?;
        lane.terminal_release()?;
        // Already-retired lanes are a no-op, not a second control message.
        assert!(lane.thread.is_none() && lane.jobs.is_none());
        lane.terminal_quiesce()?;
        lane.terminal_release()?;
        Ok(())
    }

    #[test]
    fn lost_lane_thread_is_not_quiescence_proof() {
        let (jobs, inbox) = mpsc::channel();
        let (_report, done) = mpsc::channel();
        drop(inbox);
        let mut lane = SparkExpertLane { jobs: Some(jobs), done, world: 1,
            pending: false, thread: None, quiesced: false };
        assert!(lane.terminal_quiesce().is_err());
        assert!(!lane.quiesced);
    }

    #[test]
    fn topology_lane_preserves_rank_contract_and_refuses_canonical_requests() -> Result<()> {
        let topology = SparkTopology::NATIVE_TP3_EP1;
        let peers = (1..=3).map(|port| format!("127.0.0.1:{port}").parse())
            .collect::<std::result::Result<Vec<SocketAddr>, _>>()?;
        let config = TcpTransportConfig { timing: false, timeout: Duration::from_secs(1), max_frame_bytes: 64 << 20 };
        // SAFETY: no landing buffers are installed.
        let mut lane = unsafe { SparkExpertLane::spawn_world(SparkWorld::owned(topology)?, peers.clone(), 1, config.clone(), None)? };
        assert_eq!(lane.world_size(), 3);
        lane.submit(Box::new(|| {
            let mut request = ExpertProtocolV2Request::new(1, 1, 0, 5120,
                crate::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                vec![crate::ExpertProtocolV2RowDescriptor { row_id: 0, source_kind: crate::ExpertV2SourceKind::Prefill,
                    source_request_id: 1, token_position: 0, route_offset: 0, route_count: 6 }],
                (0..6).map(|expert| crate::ExpertProtocolV2RouteEntry { row_index: 0, expert_id: expert, gate_weight: 0.125 }).collect(),
                vec![0; 5280])?;
            request.header.flags |= super::super::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
            Ok(request)
        }), Box::new(|_, _, _| Ok(())))?;
        let error = lane.wait(Duration::from_secs(2)).unwrap_err();
        assert!(error.to_string().contains("requires a native group request"), "{error:#}");
        // SAFETY: no landing buffers are installed; mismatched worlds fail setup.
        assert!(unsafe { SparkExpertLane::spawn_world(SparkWorld::owned(topology)?, peers[..2].to_vec(), 1, config, None) }.is_err());
        Ok(())
    }
}
