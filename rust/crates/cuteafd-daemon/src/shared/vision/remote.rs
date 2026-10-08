//! Bounded cold-path RGB8/BF16 TCP channel; no CUDA work runs on the scheduler.
use super::{EncodeJob as NativeJob, EncoderService};
use cuteafd_core::{AudioKey, ImageKey};
use cuteafd_engine::media::{EncodeJob, EncodeOutput, EncoderClient, EncoderTicket, MediaError};
use cuteafd_loader::media::EncoderId;
use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
const MAGIC: &[u8; 8] = b"CAFDVI01";
const MAX_PATCHES: u32 = 16_384;
const MAX_WIDTH: u32 = 16_384;
const MAX_ERROR: usize = 1024;
type Result<T> = std::result::Result<T, MediaError>;
fn error(value: impl std::fmt::Display) -> MediaError {
    MediaError::Encoder(value.to_string())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderHandshake {
    pub encoder_id: EncoderId,
    pub max_patches: u32,
    pub output_width: u32,
    pub patch_size: u32,
    pub merge_size: u32,
    /// The placement contract, shared by coordinator and every replica.
    pub plan_hash: [u8; 32],
}
impl EncoderHandshake {
    fn validate(&self) -> Result<()> {
        if self.max_patches == 0
            || self.max_patches > MAX_PATCHES
            || self.output_width == 0
            || self.output_width > MAX_WIDTH
            || ![14, 16].contains(&self.patch_size)
            || ![2, 3].contains(&self.merge_size)
            || (self.merge_size == 3 && (self.patch_size != 14 || self.output_width != 5120))
        {
            return Err(error("invalid encoder capacity/geometry"));
        }
        Ok(())
    }
    fn compatible(&self, expected: &Self) -> Result<()> {
        self.validate()?;
        expected.validate()?;
        if self.encoder_id != expected.encoder_id
            || self.plan_hash != expected.plan_hash
            || self.output_width != expected.output_width
            || self.patch_size != expected.patch_size
            || self.merge_size != expected.merge_size
            || self.max_patches < expected.max_patches
        {
            return Err(error("vision encoder identity/placement/capacity mismatch"));
        }
        Ok(())
    }
    fn write(&self, stream: &mut Wire) -> Result<()> {
        stream.write_all(if self.merge_size == 3 { b"CAFDV401" } else { MAGIC }).map_err(error)?;
        stream.write_all(&self.encoder_id.0).map_err(error)?;
        stream.write_all(&self.plan_hash).map_err(error)?;
        for value in [
            self.max_patches,
            self.output_width,
            self.patch_size,
            self.merge_size,
        ] {
            put_u32(stream, value)?;
        }
        Ok(())
    }
    fn tokens(&self, h: u32, w: u32) -> u64 {
        if self.merge_size == 3 { u64::from(h.div_ceil(3)) * (u64::from(w.div_ceil(3)) + 1) + 2 }
        else { u64::from(h) * u64::from(w) / 4 }
    }
    fn input_bytes(&self, patches: u64) -> u64 {
        patches.saturating_mul(u64::from(self.patch_size).pow(2)).saturating_mul(3)
            .saturating_mul(if self.merge_size == 3 { 2 } else { 1 })
    }
    fn validate_job(&self, job: &EncodeJob) -> Result<usize> {
        let ([t, h, w], rgb8) = job.image_input()?;
        let patches = u64::from(h) * u64::from(w);
        let rgb_bytes = self.input_bytes(patches);
        if t != 1
            || h == 0
            || w == 0
            || (self.merge_size == 2 && (h % 2 != 0 || w % 2 != 0))
            || patches > u64::from(self.max_patches)
            || job.tokens as u64 != self.tokens(h, w)
            || (self.merge_size == 3 && job.tokens > 1024)
            || job.hidden_width != self.output_width as usize
            || rgb8.len() as u64 != rgb_bytes
        {
            return Err(error("invalid image grid/RGB8/output geometry"));
        }
        job.feature_bytes()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioHandshake {
    pub encoder_id: EncoderId,
    pub plan_hash: [u8; 32],
    pub max_samples: u32,
    pub output_width: u32,
}
#[derive(Clone)]
enum Handshake {
    Image(EncoderHandshake),
    Audio(AudioHandshake),
}
impl From<EncoderHandshake> for Handshake {
    fn from(value: EncoderHandshake) -> Self {
        Self::Image(value)
    }
}
impl From<AudioHandshake> for Handshake {
    fn from(value: AudioHandshake) -> Self {
        Self::Audio(value)
    }
}
impl Handshake {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Image(value) => value.validate(),
            Self::Audio(value) => {
                cuteafd_loader::media::audio::AudioGeometry::for_samples(
                    value.max_samples as usize,
                )
                .map_err(error)?;
                if ![4096, 6144].contains(&value.output_width) {
                    return Err(error("invalid audio output width"));
                }
                Ok(())
            }
        }
    }
    fn compatible(&self, expected: &Self) -> Result<()> {
        self.validate()?;
        expected.validate()?;
        match (self, expected) {
            (Self::Image(actual), Self::Image(expected)) => actual.compatible(expected),
            (Self::Audio(actual), Self::Audio(expected))
                if actual.encoder_id == expected.encoder_id
                    && actual.plan_hash == expected.plan_hash
                    && actual.output_width == expected.output_width
                    && actual.max_samples >= expected.max_samples =>
            {
                Ok(())
            }
            _ => Err(error(
                "encoder modality/identity/placement/capacity mismatch",
            )),
        }
    }
    fn write(&self, stream: &mut Wire) -> Result<()> {
        match self {
            Self::Image(value) => value.write(stream),
            Self::Audio(value) => {
                stream.write_all(b"CAFDAU01").map_err(error)?;
                stream.write_all(&value.encoder_id.0).map_err(error)?;
                stream.write_all(&value.plan_hash).map_err(error)?;
                for n in [value.max_samples, value.output_width, 24000, 4] {
                    put_u32(stream, n)?;
                }
                Ok(())
            }
        }
    }
    fn read(stream: &mut Wire) -> Result<Self> {
        let magic = read_array::<8>(stream)?;
        if &magic != MAGIC && &magic != b"CAFDV401" && &magic != b"CAFDAU01" {
            return Err(error("unsupported encoder wire version"));
        }
        let encoder_id = EncoderId(read_array(stream)?);
        let plan_hash = read_array(stream)?;
        let [capacity, output_width, patch, merge] = [
            get_u32(stream)?,
            get_u32(stream)?,
            get_u32(stream)?,
            get_u32(stream)?,
        ];
        let value = if (&magic == MAGIC && merge == 2) || (&magic == b"CAFDV401" && merge == 3) {
            Self::Image(EncoderHandshake {
                encoder_id,
                plan_hash,
                max_patches: capacity,
                output_width,
                patch_size: patch,
                merge_size: merge,
            })
        } else if &magic == b"CAFDAU01" && patch == 24000 && merge == 4 {
            Self::Audio(AudioHandshake {
                encoder_id,
                plan_hash,
                max_samples: capacity,
                output_width,
            })
        } else {
            return Err(error("unsupported encoder wire version/geometry"));
        };
        value.validate()?;
        Ok(value)
    }
    fn validate_job(&self, job: &EncodeJob) -> Result<usize> {
        match self {
            Self::Image(value) => value.validate_job(job),
            Self::Audio(value) => {
                let cuteafd_engine::media::EncodeInput::Audio { pcm } = &job.input else {
                    return Err(error("audio peer refuses image"));
                };
                let geometry = cuteafd_loader::media::audio::AudioGeometry::for_samples(pcm.len())
                    .map_err(error)?;
                if !job.key.is_audio()
                    || pcm.len() > value.max_samples as usize
                    || pcm.iter().any(|v| !v.is_finite())
                    || job.tokens != geometry.tokens
                    || job.hidden_width != value.output_width as usize
                {
                    return Err(error("invalid canonical audio PCM/output geometry"));
                }
                job.feature_bytes()
            }
        }
    }
}
fn read_array<const N: usize>(s: &mut Wire) -> Result<[u8; N]> {
    let mut bytes = [0; N];
    s.read_exact(&mut bytes).map_err(error)?;
    Ok(bytes)
}
fn get_u32(s: &mut Wire) -> Result<u32> {
    Ok(u32::from_le_bytes(read_array(s)?))
}
fn get_u64(s: &mut Wire) -> Result<u64> {
    Ok(u64::from_le_bytes(read_array(s)?))
}
fn put_u32(s: &mut Wire, n: u32) -> Result<()> {
    s.write_all(&n.to_le_bytes()).map_err(error)
}
fn put_u64(s: &mut Wire, n: u64) -> Result<()> {
    s.write_all(&n.to_le_bytes()).map_err(error)
}
// Each handshake/frame has one absolute deadline, even when bytes trickle in.
struct Wire {
    socket: TcpStream,
    timeout: Duration,
    deadline: Instant,
}
impl Wire {
    fn new(socket: TcpStream, timeout: Duration) -> Result<Self> {
        configure(&socket, timeout)?;
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| error("vision timeout too large"))?;
        Ok(Self {
            socket,
            timeout,
            deadline,
        })
    }
    fn frame(&mut self) {
        self.deadline = Instant::now() + self.timeout;
    }
    fn remaining(&self) -> std::io::Result<Duration> {
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::TimedOut, "vision frame deadline")
            })
    }
}
impl Read for Wire {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.socket.set_read_timeout(Some(self.remaining()?))?;
        self.socket.read(bytes)
    }
}
impl Write for Wire {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.socket.set_write_timeout(Some(self.remaining()?))?;
        self.socket.write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.socket.flush()
    }
}
fn configure(s: &TcpStream, timeout: Duration) -> Result<()> {
    if timeout.is_zero() {
        return Err(error("vision timeout must be positive"));
    }
    s.set_nodelay(true).map_err(error)?;
    s.set_read_timeout(Some(timeout)).map_err(error)?;
    s.set_write_timeout(Some(timeout)).map_err(error)
}
struct Work {
    job: EncodeJob,
    reply: mpsc::SyncSender<Result<EncodeOutput>>,
    cancelled: Arc<AtomicBool>,
}
struct Replica {
    queue: Option<mpsc::SyncSender<Work>>,
    socket: TcpStream,
    failed: Arc<AtomicBool>,
    owner: Option<JoinHandle<()>>,
}
struct OwnerHealth(Arc<AtomicBool>);
impl Drop for OwnerHealth {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
struct Pending {
    result: mpsc::Receiver<Result<EncodeOutput>>,
    cancelled: Arc<AtomicBool>,
}
/// A persistent connection and bounded owner queue per replica. The constructor
/// is a readiness barrier: all EncoderIds are verified before API vision is enabled.
pub struct RemoteEncoder {
    healthy: Arc<AtomicBool>,
    handshake: Handshake,
    replicas: Vec<Replica>,
    pending: HashMap<EncoderTicket, Pending>,
    next: u64,
    round_robin: usize,
}
impl RemoteEncoder {
    pub fn connect(
        addresses: Vec<SocketAddr>,
        expected: EncoderHandshake,
        timeout: Duration,
    ) -> Result<Self> {
        Self::connect_peer(addresses, expected.into(), timeout)
    }
    pub fn connect_audio(
        addresses: Vec<SocketAddr>,
        expected: AudioHandshake,
        timeout: Duration,
    ) -> Result<Self> {
        Self::connect_peer(addresses, expected.into(), timeout)
    }
    fn connect_peer(
        addresses: Vec<SocketAddr>,
        expected: Handshake,
        timeout: Duration,
    ) -> Result<Self> {
        expected.validate()?;
        if addresses.is_empty() || addresses.len() > 6 {
            return Err(error("vision needs 1..6 replicas"));
        }
        let mut this = Self {
            healthy: Arc::new(AtomicBool::new(true)),
            handshake: expected.clone(),
            replicas: vec![],
            pending: HashMap::new(),
            next: 0,
            round_robin: 0,
        };
        for address in addresses {
            let mut stream = Wire::new(
                TcpStream::connect_timeout(&address, timeout).map_err(error)?,
                timeout,
            )?;
            Handshake::read(&mut stream)?.compatible(&expected)?;
            expected.write(&mut stream)?;
            if get_u32(&mut stream)? != 0 {
                return Err(error("vision encoder rejected handshake"));
            }
            let socket = stream.socket.try_clone().map_err(error)?;
            let failed = Arc::new(AtomicBool::new(false));
            let failure = failed.clone();
            let health = this.healthy.clone();
            let (queue, jobs) = mpsc::sync_channel::<Work>(128);
            let owner = thread::Builder::new()
                .name("remote-vision-owner".into())
                .spawn(move || {
                    // Any owner exit (including panic) permanently closes vision admission.
                    let _health = OwnerHealth(health);
                    loop {
                        let work = match jobs.recv_timeout(Duration::from_secs(2)) {
                            Ok(work) => work,
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                if !failure.load(Ordering::Acquire) && ping(&mut stream).is_err() {
                                    failure.store(true, Ordering::Release);
                                    _health.0.store(false, Ordering::Release);
                                }
                                continue;
                            }
                        };
                        if work.cancelled.load(Ordering::Acquire) {
                            continue;
                        }
                        let result = if failure.load(Ordering::Acquire) {
                            Err(error("vision encoder unavailable"))
                        } else {
                            exchange(&mut stream, &work.job)
                        };
                        if result.is_err() {
                            failure.store(true, Ordering::Release);
                            _health.0.store(false, Ordering::Release);
                        }
                        if !work.cancelled.load(Ordering::Acquire) {
                            let _ = work.reply.send(result);
                        }
                    }
                })
                .map_err(error)?;
            this.replicas.push(Replica {
                queue: Some(queue),
                socket,
                failed,
                owner: Some(owner),
            });
        }
        Ok(this)
    }
    /// Image requests fail closed after a wire error; text needs no encoder.
    pub fn healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }
    /// Live health without a scheduler lock, including while it sleeps.
    pub fn health_handle(&self) -> Arc<AtomicBool> {
        self.healthy.clone()
    }
    pub fn ready(&self) -> bool {
        !self.replicas.is_empty() && self.healthy()
    }
}
impl EncoderClient for RemoteEncoder {
    fn submit(&mut self, job: EncodeJob) -> Result<EncoderTicket> {
        if !self.healthy() {
            return Err(error("vision encoder unavailable"));
        }
        self.handshake.validate_job(&job)?;
        let replica = self.round_robin % self.replicas.len();
        self.round_robin = self.round_robin.wrapping_add(1);
        let replica = &self.replicas[replica];
        if replica.failed.load(Ordering::Acquire) {
            return Err(error("vision encoder unavailable"));
        }
        let ticket = EncoderTicket(self.next);
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| error("vision ticket exhausted"))?;
        let (reply, result) = mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        replica
            .queue
            .as_ref()
            .ok_or_else(|| error("vision encoder stopped"))?
            .try_send(Work {
                job,
                reply,
                cancelled: cancelled.clone(),
            })
            .map_err(|failure| match failure {
                mpsc::TrySendError::Full(_) => MediaError::QueueFull,
                mpsc::TrySendError::Disconnected(_) => error("vision encoder unavailable"),
            })?;
        self.pending.insert(ticket, Pending { result, cancelled });
        Ok(ticket)
    }
    fn poll(&mut self, ticket: EncoderTicket) -> Option<Result<EncodeOutput>> {
        let result = match self.pending.get(&ticket)?.result.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => Err(error("vision encoder unavailable")),
        };
        self.pending.remove(&ticket);
        Some(result)
    }
    fn cancel(&mut self, ticket: EncoderTicket) {
        if let Some(pending) = self.pending.remove(&ticket) {
            pending.cancelled.store(true, Ordering::Release);
        }
    }
}
impl Drop for RemoteEncoder {
    fn drop(&mut self) {
        for pending in self.pending.values() {
            pending.cancelled.store(true, Ordering::Release);
        }
        for r in &mut self.replicas {
            r.queue.take();
            let _ = r.socket.shutdown(Shutdown::Both);
        }
        for r in &mut self.replicas {
            if let Some(owner) = r.owner.take() {
                let _ = owner.join();
            }
        }
    }
}
fn ping(s: &mut Wire) -> Result<()> {
    s.frame();
    put_u32(s, 0)?;
    if get_u32(s)? != 0 {
        return Err(error("vision heartbeat failed"));
    }
    Ok(())
}
fn exchange(s: &mut Wire, job: &EncodeJob) -> Result<EncodeOutput> {
    s.frame();
    match &job.input {
        cuteafd_engine::media::EncodeInput::Image { grid, rgb8 } => {
            put_u32(s, 1)?;
            s.write_all(job.key.bytes()).map_err(error)?;
            for &n in grid {
                put_u32(s, n)?;
            }
            put_u64(s, job.tokens as u64)?;
            put_u64(s, rgb8.len() as u64)?;
            s.write_all(rgb8).map_err(error)?;
        }
        cuteafd_engine::media::EncodeInput::Audio { pcm } => {
            put_u32(s, 2)?;
            s.write_all(job.key.bytes()).map_err(error)?;
            put_u32(s, pcm.len() as u32)?;
            put_u64(s, job.tokens as u64)?;
            put_u64(s, pcm.len() as u64 * 4)?;
            let bytes = pcm.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
            s.write_all(&bytes).map_err(error)?;
        }
    }
    let status = get_u32(s)?;
    let bytes = read_array(s)?;
    let key: cuteafd_core::MediaKey = if job.key.is_audio() {
        AudioKey(bytes).into()
    } else {
        ImageKey(bytes).into()
    };
    let elapsed_ms = get_u64(s)? as f64 / 1_000_000.0;
    let len = get_u64(s)?;
    if key != job.key {
        return Err(error("vision reply key mismatch"));
    }
    let expected = job.feature_bytes()?;
    if status == 0 && len != expected as u64 || status != 0 && len > MAX_ERROR as u64 {
        return Err(error("vision reply length mismatch"));
    }
    let mut bytes = vec![0; len as usize];
    s.read_exact(&mut bytes).map_err(error)?;
    if status != 0 {
        return Err(error(String::from_utf8_lossy(&bytes)));
    }
    Ok(EncodeOutput {
        key,
        features: bytes.into(),
        elapsed_ms,
    })
}

/// Network owner is separate from the resident CUDA owner; same process/context.
/// Shutdown interrupts sockets, joins the network thread, then drains CUDA work.
pub struct EncoderServer {
    stop: Arc<AtomicBool>,
    socket: Arc<Mutex<Option<TcpStream>>>,
    owner: Option<JoinHandle<()>>,
    pub address: SocketAddr,
}
impl EncoderServer {
    pub fn start(
        address: SocketAddr,
        handshake: EncoderHandshake,
        service: EncoderService,
        lut: Arc<[f32; 768]>,
        timeout: Duration,
    ) -> Result<Self> {
        Self::start_shared(address, handshake, Arc::new(service), lut, timeout)
    }
    pub fn start_shared(address: SocketAddr, handshake: EncoderHandshake, service: Arc<EncoderService>,
        lut: Arc<[f32; 768]>, timeout: Duration) -> Result<Self> {
        handshake.validate()?;
        let health = service.clone();
        Self::start_backend(
            address,
            handshake.into(),
            timeout,
            move || health.healthy(),
            move |job| {
                let started = Instant::now();
                let (grid, rgb8) = job.image_input()?;
                let ticket = service
                    .submit(NativeJob {
                        rgb: rgb8.clone(),
                        grid: [grid[1] as usize, grid[2] as usize],
                        lut: lut.clone(),
                        output: vec![0; job.tokens * job.hidden_width],
                    })
                    .map_err(error)?;
                // The owner retains all buffers until its stream has drained, even
                // when a peer disappears. A deadline closes TCP, not CUDA lifetimes.
                let output = loop {
                    match ticket.poll().map_err(error)? {
                        Some(output) => break output,
                        None => thread::sleep(Duration::from_millis(1)),
                    }
                };
                let mut bytes = Vec::with_capacity(output.len() * 2);
                for n in output {
                    bytes.extend_from_slice(&n.to_le_bytes());
                }
                Ok(EncodeOutput {
                    key: job.key,
                    features: bytes.into(),
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                })
            },
        )
    }
    pub fn start_audio(
        address: SocketAddr,
        handshake: AudioHandshake,
        service: EncoderService,
        timeout: Duration,
    ) -> Result<Self> {
        Self::start_audio_shared(address, handshake, Arc::new(service), timeout)
    }
    pub fn start_audio_shared(address: SocketAddr, handshake: AudioHandshake, service: Arc<EncoderService>,
        timeout: Duration) -> Result<Self> {
        let health = service.clone();
        Self::start_backend(
            address,
            handshake.into(),
            timeout,
            move || health.healthy(),
            move |job| {
                let started = Instant::now();
                let cuteafd_engine::media::EncodeInput::Audio { pcm } = &job.input else {
                    return Err(error("audio peer refuses image"));
                };
                let rows = job.tokens * job.hidden_width;
                let ticket = service
                    .submit_audio(super::audio::AudioEncodeJob {
                        pcm: pcm.clone(),
                        output: vec![0; rows],
                        fp32_scratch: vec![0.0; rows],
                    })
                    .map_err(error)?;
                let output = loop {
                    match ticket.poll().map_err(error)? {
                        Some(output) => break output,
                        None => thread::sleep(Duration::from_millis(1)),
                    }
                };
                let bytes = output
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                Ok(EncodeOutput {
                    key: job.key,
                    features: bytes.into(),
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                })
            },
        )
    }
    /// Family edge backend; transport, heartbeat and bounded frames stay shared.
    pub fn start_image_backend<F, H>(address: SocketAddr, handshake: EncoderHandshake,
        timeout: Duration, healthy: H, encode: F) -> Result<Self>
    where F: FnMut(&EncodeJob) -> Result<EncodeOutput> + Send + 'static,
        H: Fn() -> bool + Send + 'static,
    { Self::start_backend(address, handshake.into(), timeout, healthy, encode) }

    fn start_backend<F, H>(
        address: SocketAddr,
        handshake: Handshake,
        timeout: Duration,
        healthy: H,
        mut encode: F,
    ) -> Result<Self>
    where
        F: FnMut(&EncodeJob) -> Result<EncodeOutput> + Send + 'static,
        H: Fn() -> bool + Send + 'static,
    {
        handshake.validate()?;
        let listener = TcpListener::bind(address).map_err(error)?;
        listener.set_nonblocking(true).map_err(error)?;
        let address = listener.local_addr().map_err(error)?;
        let stop = Arc::new(AtomicBool::new(false));
        let socket = Arc::new(Mutex::new(None));
        let stopped = stop.clone();
        let active = socket.clone();
        let owner = thread::Builder::new()
            .name("vision-tcp-server".into())
            .spawn(move || {
                while !stopped.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            if let Ok(copy) = stream.try_clone() {
                                *active.lock().unwrap() = Some(copy);
                            }
                            let result = Wire::new(stream, timeout).and_then(|mut stream| {
                                handshake
                                    .write(&mut stream)
                                    .and_then(|()| Handshake::read(&mut stream))
                                    .and_then(|expected| handshake.compatible(&expected))
                                    .and_then(|()| {
                                        if !healthy() {
                                            return Err(error("vision encoder unavailable"));
                                        }
                                        put_u32(&mut stream, 0)
                                    })
                                    .and_then(|()| {
                                        serve_connection(
                                            &mut stream,
                                            &handshake,
                                            &stopped,
                                            &healthy,
                                            &mut encode,
                                        )
                                    })
                            });
                            if let Err(e) = result {
                                tracing::debug!(%e, "vision peer closed/failed");
                            }
                            active.lock().unwrap().take();
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(2))
                        }
                        Err(e) => {
                            tracing::error!(%e, "vision listener failed");
                            break;
                        }
                    }
                }
            })
            .map_err(error)?;
        Ok(Self {
            stop,
            socket,
            owner: Some(owner),
            address,
        })
    }
}
impl Drop for EncoderServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(s) = self.socket.lock().unwrap().as_ref() {
            let _ = s.shutdown(Shutdown::Both);
        }
        if let Some(owner) = self.owner.take() {
            let _ = owner.join();
        }
    }
}
fn serve_connection<F, H>(
    s: &mut Wire,
    handshake: &Handshake,
    stop: &AtomicBool,
    healthy: &H,
    encode: &mut F,
) -> Result<()>
where
    F: FnMut(&EncodeJob) -> Result<EncodeOutput>,
    H: Fn() -> bool,
{
    while !stop.load(Ordering::Acquire) {
        s.frame();
        let opcode = get_u32(s)?;
        if !healthy() {
            return Err(error("vision encoder unavailable"));
        }
        if opcode == 0 {
            put_u32(s, 0)?;
            continue;
        }
        let job = match (opcode, handshake) {
            (1, Handshake::Image(handshake)) => {
                let key: cuteafd_core::MediaKey = ImageKey(read_array(s)?).into();
                let grid = [get_u32(s)?, get_u32(s)?, get_u32(s)?];
                let tokens = get_u64(s)?;
                let len = get_u64(s)?;
                let patches = u64::from(grid[1]) * u64::from(grid[2]);
                if grid[0] != 1
                    || patches == 0
                    || patches > u64::from(handshake.max_patches)
                    || (handshake.merge_size == 2 && (grid[1] % 2 != 0 || grid[2] % 2 != 0))
                    || tokens != handshake.tokens(grid[1], grid[2])
                    || (handshake.merge_size == 3 && tokens > 1024)
                    || len != handshake.input_bytes(patches)
                {
                    return Err(error("invalid vision frame before allocation"));
                }
                let mut rgb = vec![0; len as usize];
                s.read_exact(&mut rgb).map_err(error)?;
                let job = EncodeJob::image(
                    match key {
                        cuteafd_core::MediaKey::Image(key) => key,
                        _ => unreachable!(),
                    },
                    grid,
                    rgb.into(),
                    tokens as usize,
                    handshake.output_width as usize,
                );
                job
            }
            (2, Handshake::Audio(handshake)) => {
                let key = AudioKey(read_array(s)?);
                let samples = get_u32(s)? as usize;
                let tokens = get_u64(s)?;
                let len = get_u64(s)?;
                let geometry = cuteafd_loader::media::audio::AudioGeometry::for_samples(samples)
                    .map_err(error)?;
                if samples > handshake.max_samples as usize
                    || tokens != geometry.tokens as u64
                    || len != samples as u64 * 4
                {
                    return Err(error("invalid audio frame before allocation"));
                }
                let mut bytes = vec![0; len as usize];
                s.read_exact(&mut bytes).map_err(error)?;
                let pcm = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                EncodeJob::audio(
                    key,
                    pcm.into(),
                    tokens as usize,
                    handshake.output_width as usize,
                )
            }
            _ => return Err(error("encoder peer refuses job modality")),
        };
        handshake.validate_job(&job)?;
        let key = job.key;
        let result = encode(&job).and_then(|output| {
            if output.key != key
                || output.features.len() != job.feature_bytes()?
                || !output.elapsed_ms.is_finite()
                || output.elapsed_ms < 0.0
            {
                Err(error("invalid vision backend output"))
            } else {
                Ok(output)
            }
        });
        // Encoding drains independently of the wire deadline; sending gets a fresh bound.
        s.frame();
        let (status, elapsed, bytes) = match result {
            Ok(output) => (0, (output.elapsed_ms * 1_000_000.0) as u64, output.features),
            Err(e) => (
                1,
                0,
                Arc::from(
                    e.to_string()
                        .as_bytes()
                        .iter()
                        .take(MAX_ERROR)
                        .copied()
                        .collect::<Vec<_>>(),
                ),
            ),
        };
        put_u32(s, status)?;
        s.write_all(key.bytes()).map_err(error)?;
        put_u64(s, elapsed)?;
        put_u64(s, bytes.len() as u64)?;
        s.write_all(&bytes).map_err(error)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_engine::media::FakeEncoder;
    #[test]
    fn rgb_wire_handshake_and_payload_remain_identical() {
        // MiMo/GLM Flash use patch16; Qwen uses patch14. Both retain CAFDVI01.
        for patch in [16u32, 14] {
            let mut h = handshake();
            h.patch_size = patch;
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let socket = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let (mut peer, _) = listener.accept().unwrap();
            let mut wire = Wire::new(socket, Duration::from_secs(2)).unwrap();
            h.write(&mut wire).unwrap();
            let mut actual = vec![0; 88];
            peer.read_exact(&mut actual).unwrap();
            let mut expected = b"CAFDVI01".to_vec();
            expected.extend_from_slice(&[7;32]); expected.extend_from_slice(&[9;32]);
            for n in [16384u32, 4096, patch, 2] { expected.extend_from_slice(&n.to_le_bytes()); }
            assert_eq!(actual, expected);
            let job = EncodeJob::image(ImageKey([3;32]), [1,2,2], vec![11; (4*patch*patch*3) as usize].into(), 1, 4096);
            let mut expected = 1u32.to_le_bytes().to_vec();
            expected.extend_from_slice(&[3;32]);
            for n in [1u32,2,2] { expected.extend_from_slice(&n.to_le_bytes()); }
            expected.extend_from_slice(&1u64.to_le_bytes());
            expected.extend_from_slice(&(4u64*u64::from(patch).pow(2)*3).to_le_bytes());
            expected.extend_from_slice(&vec![11; (4*patch*patch*3) as usize]);
            let owner = thread::spawn(move || {
                let mut actual = vec![0; expected.len()]; peer.read_exact(&mut actual).unwrap(); assert_eq!(actual,expected);
                peer.write_all(&0u32.to_le_bytes()).unwrap(); peer.write_all(&[3;32]).unwrap();
                peer.write_all(&0u64.to_le_bytes()).unwrap(); peer.write_all(&8192u64.to_le_bytes()).unwrap();
                peer.write_all(&vec![0;8192]).unwrap();
            });
            exchange(&mut wire, &job).unwrap(); owner.join().unwrap();
        }
    }
    #[test]
    fn v41_bf16_patch_frames_count_complete_spans_and_refuse_1058() {
        let mut h = handshake(); h.merge_size=3; h.patch_size=14; h.output_width=5120; h.max_patches=9216;
        let job = EncodeJob::image(ImageKey([1;32]),[1,93,93],vec![0;93*93*588*2].into(),994,5120);
        assert_eq!(h.validate_job(&job).unwrap(),994*10240);
        let rejected = EncodeJob::image(ImageKey([1;32]),[1,96,96],vec![0;96*96*588*2].into(),1058,5120);
        assert!(h.validate_job(&rejected).is_err());
        let oversized = EncodeJob::image(ImageKey([2;32]), [1,u32::MAX,u32::MAX], vec![].into(), 1, 5120);
        assert!(h.validate_job(&oversized).is_err());
        let server = EncoderServer::start_image_backend("127.0.0.1:0".parse().unwrap(),h.clone(),Duration::from_secs(2),||true,
            |job| Ok(EncodeOutput { key:job.key, features:FakeEncoder::features(job)?, elapsed_ms:0.0 })).unwrap();
        let mut client = RemoteEncoder::connect(vec![server.address],h,Duration::from_secs(2)).unwrap();
        let expected = FakeEncoder::features(&job).unwrap();
        let ticket = client.submit(job).unwrap();
        let end = Instant::now()+Duration::from_secs(2);
        loop { if let Some(result)=client.poll(ticket) { assert_eq!(result.unwrap().features,expected); break; }
            assert!(Instant::now()<end); thread::sleep(Duration::from_millis(1)); }
    }
    fn handshake() -> EncoderHandshake {
        EncoderHandshake {
            encoder_id: EncoderId([7; 32]),
            max_patches: 16_384,
            output_width: 4096,
            patch_size: 16,
            merge_size: 2,
            plan_hash: [9; 32],
        }
    }
    fn job(value: u8) -> EncodeJob {
        EncodeJob::image(
            ImageKey([value; 32]),
            [1, 4, 4],
            vec![value; 4 * 4 * 768].into(),
            4,
            4096,
        )
    }
    fn server() -> EncoderServer {
        EncoderServer::start_backend(
            "127.0.0.1:0".parse().unwrap(),
            handshake().into(),
            Duration::from_secs(2),
            || true,
            |job| {
                Ok(EncodeOutput {
                    key: job.key,
                    features: FakeEncoder::features(job)?,
                    elapsed_ms: 1.0,
                })
            },
        )
        .unwrap()
    }
    fn wait(client: &mut RemoteEncoder, ticket: EncoderTicket) -> Result<EncodeOutput> {
        let started = Instant::now();
        loop {
            if let Some(result) = client.poll(ticket) {
                return result;
            }
            assert!(started.elapsed() < Duration::from_secs(3));
            thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn loopback_replicas_are_byte_exact_cancel_and_drain() {
        let a = server();
        let b = server();
        let mut client = RemoteEncoder::connect(
            vec![a.address, b.address],
            handshake(),
            Duration::from_secs(2),
        )
        .unwrap();
        assert!(client.ready());
        let first = client.submit(job(1)).unwrap();
        let second = client.submit(job(2)).unwrap();
        assert_eq!(
            wait(&mut client, first).unwrap().features,
            FakeEncoder::features(&job(1)).unwrap()
        );
        assert_eq!(
            wait(&mut client, second).unwrap().features,
            FakeEncoder::features(&job(2)).unwrap()
        );
        let cancel = client.submit(job(3)).unwrap();
        client.cancel(cancel);
        assert!(client.poll(cancel).is_none());
        let next = client.submit(job(4)).unwrap();
        assert_eq!(
            wait(&mut client, next).unwrap().features,
            FakeEncoder::features(&job(4)).unwrap()
        );
    }
    fn audio_handshake() -> AudioHandshake {
        AudioHandshake {
            encoder_id: EncoderId([3; 32]),
            plan_hash: [9; 32],
            max_samples: 7200000,
            output_width: 4096,
        }
    }
    fn audio_job() -> EncodeJob {
        EncodeJob::audio(AudioKey([4; 32]), vec![0.25; 24000].into(), 7, 4096)
    }
    #[test]
    fn audio_loopback_is_byte_exact_bounded_and_modality_checked() {
        let server = EncoderServer::start_backend(
            "127.0.0.1:0".parse().unwrap(),
            audio_handshake().into(),
            Duration::from_secs(2),
            || true,
            |job| {
                Ok(EncodeOutput {
                    key: job.key,
                    features: FakeEncoder::features(job)?,
                    elapsed_ms: 1.0,
                })
            },
        )
        .unwrap();
        let mut client = RemoteEncoder::connect_audio(
            vec![server.address],
            audio_handshake(),
            Duration::from_secs(2),
        )
        .unwrap();
        assert!(client.submit(job(1)).is_err());
        let mut invalid = audio_job();
        invalid.tokens += 1;
        assert!(client.submit(invalid).is_err());
        let mut invalid = audio_job();
        if let cuteafd_engine::media::EncodeInput::Audio { pcm } = &mut invalid.input {
            *pcm = vec![f32::NAN; 24000].into();
        }
        assert!(client.submit(invalid).is_err());
        let first = client.submit(audio_job()).unwrap();
        assert_eq!(
            wait(&mut client, first).unwrap().features,
            FakeEncoder::features(&audio_job()).unwrap()
        );
        let cancelled = client.submit(audio_job()).unwrap();
        client.cancel(cancelled);
        assert!(client.poll(cancelled).is_none());
        let next = client.submit(audio_job()).unwrap();
        assert_eq!(
            wait(&mut client, next).unwrap().key,
            AudioKey([4; 32]).into()
        );
        drop(client);
        assert!(
            RemoteEncoder::connect(vec![server.address], handshake(), Duration::from_secs(1))
                .is_err()
        );
        let image_server = self::server();
        assert!(RemoteEncoder::connect_audio(
            vec![image_server.address],
            audio_handshake(),
            Duration::from_secs(1)
        )
        .is_err());
    }
    #[test]
    fn audio_identity_capacity_and_placement_are_readiness_barriers() {
        for field in 0..4 {
            let server = EncoderServer::start_backend(
                "127.0.0.1:0".parse().unwrap(),
                audio_handshake().into(),
                Duration::from_secs(1),
                || true,
                |_| panic!("must not encode"),
            )
            .unwrap();
            let mut expected = audio_handshake();
            match field {
                0 => expected.encoder_id.0[0] ^= 1,
                1 => expected.plan_hash[0] ^= 1,
                2 => expected.output_width = 6144,
                _ => expected.max_samples = 7200001,
            }
            assert!(RemoteEncoder::connect_audio(
                vec![server.address],
                expected,
                Duration::from_secs(1)
            )
            .is_err());
        }
    }
    #[test]
    fn malformed_audio_frame_and_vision_only_audio_opcode_close_before_encoding() {
        for audio in [false, true] {
            let hs: Handshake = if audio {
                audio_handshake().into()
            } else {
                handshake().into()
            };
            let server = EncoderServer::start_backend(
                "127.0.0.1:0".parse().unwrap(),
                hs.clone(),
                Duration::from_secs(1),
                || true,
                |_| panic!("must not encode invalid frame"),
            )
            .unwrap();
            let mut wire = Wire::new(
                TcpStream::connect(server.address).unwrap(),
                Duration::from_secs(1),
            )
            .unwrap();
            Handshake::read(&mut wire).unwrap().compatible(&hs).unwrap();
            hs.write(&mut wire).unwrap();
            assert_eq!(get_u32(&mut wire).unwrap(), 0);
            put_u32(&mut wire, 2).unwrap();
            if audio {
                wire.write_all(&[0; 32]).unwrap();
                put_u32(&mut wire, 24000).unwrap();
                put_u64(&mut wire, 7).unwrap();
                put_u64(&mut wire, u64::MAX).unwrap();
            }
            assert!(get_u32(&mut wire).is_err());
        }
    }
    #[test]
    fn vision_handshake_and_job_frame_remain_byte_identical() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let peer = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut headers = [0; 88];
            socket.read_exact(&mut headers).unwrap();
            let mut expected = b"CAFDVI01".to_vec();
            expected.extend([7; 32]);
            expected.extend([9; 32]);
            for n in [16384u32, 4096, 16, 2] {
                expected.extend(n.to_le_bytes());
            }
            assert_eq!(headers.as_slice(), expected);
            let mut frame = vec![0; 64 + 4 * 4 * 768];
            socket.read_exact(&mut frame).unwrap();
            let mut expected = 1u32.to_le_bytes().to_vec();
            expected.extend([5; 32]);
            for n in [1u32, 4, 4] {
                expected.extend(n.to_le_bytes());
            }
            expected.extend(4u64.to_le_bytes());
            expected.extend(12288u64.to_le_bytes());
            expected.extend(vec![5; 12288]);
            assert_eq!(frame, expected);
            socket.write_all(&0u32.to_le_bytes()).unwrap();
            socket.write_all(&[5; 32]).unwrap();
            socket.write_all(&0u64.to_le_bytes()).unwrap();
            socket.write_all(&32768u64.to_le_bytes()).unwrap();
            socket.write_all(&vec![0; 32768]).unwrap();
        });
        let mut wire =
            Wire::new(TcpStream::connect(address).unwrap(), Duration::from_secs(2)).unwrap();
        Handshake::Image(handshake()).write(&mut wire).unwrap();
        assert_eq!(exchange(&mut wire, &job(5)).unwrap().features.len(), 32768);
        peer.join().unwrap();
    }
    #[test]
    fn trickle_reads_do_not_extend_frame_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let peer = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            for _ in 0..20 {
                if socket.write_all(&[0]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
            }
        });
        let socket = TcpStream::connect(address).unwrap();
        let mut wire = Wire::new(socket, Duration::from_millis(100)).unwrap();
        let started = Instant::now();
        assert!(read_array::<20>(&mut wire).is_err());
        assert!(started.elapsed() < Duration::from_millis(350));
        drop(wire);
        peer.join().unwrap();
    }
    #[test]
    fn identity_placement_and_capacity_mismatches_fail_readiness() {
        for field in 0..3 {
            let server = server();
            let mut expected = handshake();
            match field {
                0 => expected.encoder_id.0[0] ^= 1,
                1 => expected.plan_hash[0] ^= 1,
                _ => expected.output_width += 1,
            }
            assert!(
                RemoteEncoder::connect(vec![server.address], expected, Duration::from_secs(1))
                    .is_err()
            );
        }
    }
    #[test]
    fn idle_replica_failure_reaches_shared_health_and_fails_all_images() {
        let a = server();
        let b = server();
        let mut client = RemoteEncoder::connect(
            vec![a.address, b.address],
            handshake(),
            Duration::from_millis(500),
        )
        .unwrap();
        let health = client.health_handle();
        assert!(health.load(Ordering::Acquire));
        drop(b);
        let start = Instant::now();
        while health.load(Ordering::Acquire) {
            assert!(start.elapsed() < Duration::from_secs(4));
            thread::sleep(Duration::from_millis(10));
        }
        assert!(!client.ready());
        assert!(
            client.submit(job(1)).is_err(),
            "the surviving replica cannot reopen image admission"
        );
    }
    #[test]
    fn native_owner_failure_closes_idle_heartbeat() {
        let alive = Arc::new(AtomicBool::new(true));
        let server_alive = alive.clone();
        let server = EncoderServer::start_backend(
            "127.0.0.1:0".parse().unwrap(),
            handshake().into(),
            Duration::from_secs(2),
            move || server_alive.load(Ordering::Acquire),
            |job| {
                Ok(EncodeOutput {
                    key: job.key,
                    features: FakeEncoder::features(job)?,
                    elapsed_ms: 1.0,
                })
            },
        )
        .unwrap();
        let client = RemoteEncoder::connect(
            vec![server.address],
            handshake(),
            Duration::from_millis(500),
        )
        .unwrap();
        let health = client.health_handle();
        alive.store(false, Ordering::Release);
        let start = Instant::now();
        while health.load(Ordering::Acquire) {
            assert!(start.elapsed() < Duration::from_secs(4));
            thread::sleep(Duration::from_millis(10));
        }
        assert!(!client.ready());
    }
    #[test]
    fn owner_exit_and_panic_fail_closed() {
        for panic in [false, true] {
            let health = Arc::new(AtomicBool::new(true));
            let owner_health = health.clone();
            let result = thread::spawn(move || {
                let _health = OwnerHealth(owner_health);
                if panic {
                    panic!("injected owner failure");
                }
            })
            .join();
            assert_eq!(result.is_err(), panic);
            assert!(!health.load(Ordering::Acquire));
        }
    }
    #[test]
    fn saturated_owner_queue_is_transient_without_poisoning_health() {
        let server = server();
        let mut client = RemoteEncoder::connect(vec![server.address], handshake(), Duration::from_secs(1)).unwrap();
        // Detach a bounded queue from its worker so saturation is deterministic.
        let (queue, receiver) = mpsc::sync_channel(1);
        let original = client.replicas[0].queue.replace(queue);
        let ticket = client.submit(job(0)).unwrap();
        assert!(matches!(client.submit(job(1)), Err(MediaError::QueueFull)));
        assert!(client.healthy());
        client.cancel(ticket);
        assert!(client.pending.is_empty());
        drop(receiver);
        client.replicas[0].queue = original;
    }
    #[test]
    fn invalid_geometry_rejected_before_queue_and_rank_failure_visible() {
        let server = server();
        let mut client =
            RemoteEncoder::connect(vec![server.address], handshake(), Duration::from_secs(1))
                .unwrap();
        let mut invalid = job(0);
        if let cuteafd_engine::media::EncodeInput::Image { grid, .. } = &mut invalid.input {
            grid[1] = u32::MAX;
        }
        assert!(client
            .submit(EncodeJob::audio(
                cuteafd_core::AudioKey([0; 32]),
                vec![0.0; 481].into(),
                1,
                4096
            ))
            .is_err());
        assert!(client.submit(invalid).is_err());
        drop(server);
        let ticket = client.submit(job(1)).unwrap();
        assert!(wait(&mut client, ticket).is_err());
        assert!(!client.healthy());
        assert!(client.submit(job(2)).is_err());
    }
}
