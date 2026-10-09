//! Where the Spark ranks' routed-expert partials arrive on the coordinator.
//!
//! Each prefill/decode transport owns a [`SparkIntake`]: one device plane per
//! rank that the compact reducer sums (with the shared expert) after a wave.
//! How the rows reach those planes is chosen once per process
//! ([`IntakeMode`], `CUTEAFD_SPARK_INTAKE`):
//!
//! - `gpu`: the NIC writes payloads straight into the planes (GPUDirect RDMA
//!   over dma-buf; see `cuteafd_transport::DeviceLanding`). No host copy, no
//!   upload.
//! - `pinned`: payloads stay in the transport's pinned receive slots and are
//!   uploaded from there on a copy stream as they arrive; the compute stream
//!   only waits for those copies before the reduce. Decode-sized waves (up to
//!   1 MiB) take the host path instead: their copies are cheaper than slot
//!   retention and a cross-stream wait.
//! - `host`: the original path: payloads are copied into a pinned staging
//!   buffer and uploaded on the compute stream after the wave.
//!
//! `auto` (the default) probes dma-buf landing once at startup (a loopback
//! QP pair timing sends into device vs pinned host memory, and pinned host to
//! device copies) and picks `gpu` when the NIC lands in device memory at
//! least [`GPU_SHARE_OF_PINNED`] of the pinned path's store-and-forward rate
//! (NIC into host memory, then the H2D copy), `pinned` otherwise. On raptor
//! the NIC writes GPU memory at ~19.6 GB/s (cross-root-complex peer writes)
//! against ~52 GB/s into host memory, yet landing wins end to end at every
//! wave size measured: device landing takes no host memory bandwidth, CPU,
//! copy engine or receive-slot retention, and prefill lanes overlap each
//! wave's wire time with the other lanes' GPU work, so a wave's standalone
//! landing time (18 ms for V4 Pro TP6's 352 MB) is not on the critical path.
//! Any `gpu` failure (no dma-buf, no registration, GPUDirect writes not
//! ordered for kernels) falls back to `pinned` and says why.
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_transport::expert::{LaneBuild, SparkExperts, SparkExpertLane, SparkExpertWave, WaveReceipt};
use cuteafd_transport::{DeviceLanding, GpuLandingProbe, VerbsHostProtocolV2ResponsePayload};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

/// Most ranks a compact reduction sums.
pub(crate) const MAX_INTAKE_RANKS: usize = 6;

/// Waves up to this many partial bytes (all ranks) are small: pinned mode
/// stages them, and uploads that remain go on the compute stream.
const SMALL_WAVE_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum IntakeMode {
    Host,
    Pinned,
    Gpu,
}

impl IntakeMode {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Pinned => "pinned",
            Self::Gpu => "gpu",
        }
    }
}

/// The process-wide intake setting, its probe, and why.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct IntakeChoice {
    /// `CUTEAFD_SPARK_INTAKE` (auto, gpu, pinned or host).
    pub(crate) setting: String,
    /// The mode before a transport's wave size is known: `auto` resolves
    /// `gpu` per transport ([`Self::mode_for`]).
    pub(crate) mode: IntakeMode,
    pub(crate) reason: String,
    pub(crate) probe: Option<GpuLandingProbe>,
    /// Pinned host to device copy rate, GB/s (the pinned path's second leg).
    pub(crate) h2d_gbps: Option<f64>,
}

/// `auto` lands in device memory when the probed landing rate is at least
/// this share of the pinned path's store-and-forward rate
/// ([`pinned_path_gbps`]). Measured end to end, landing won at shares 0.53
/// (loopback on GPU1, GLM 5.3 Flash TP4 8K prefill 566 -> 541 ms, per-wave
/// 6.4 vs 12.0 ms) and 0.71 (raptor GPU0 + Sparks, landing 19.6 GB/s, host
/// 52.4, H2D 57.6: GLM 5.3 Flash TP4 5329 -> 5477 tok/s, V4 Pro TP6 1502 ->
/// 1650 tok/s, C1 decode +2-3%), the win growing with the wave (134 -> 352
/// MB); below half nothing was measured.
pub(crate) const GPU_SHARE_OF_PINNED: f64 = 0.5;

/// The pinned path's rate for a wave: every byte lands in host memory and
/// then crosses to the GPU, and a rank's upload starts only once its rows
/// have landed, so the legs add up per byte.
pub(crate) fn pinned_path_gbps(host_gbps: f64, h2d_gbps: f64) -> f64 {
    1.0 / (1.0 / host_gbps + 1.0 / h2d_gbps)
}

/// Whether `auto` lands in device memory at the probed rates (landing
/// verified), and the rates it compared.
fn gpu_wins(probe: &GpuLandingProbe, h2d_gbps: Option<f64>) -> (bool, String) {
    match h2d_gbps {
        Some(h2d) if probe.gpu_gbps > 0.0 && probe.host_gbps > 0.0 && h2d > 0.0 => {
            let pinned = pinned_path_gbps(probe.host_gbps, h2d);
            let share = probe.gpu_gbps / pinned;
            (share >= GPU_SHARE_OF_PINNED, format!("landing {:.1} GB/s = {share:.2} of the pinned path's {pinned:.1} \
                (host {:.1}, H2D {h2d:.1}; gpu from {GPU_SHARE_OF_PINNED})", probe.gpu_gbps, probe.host_gbps))
        }
        _ => (probe.gpu_gbps > 0.0, format!("landing {:.1} GB/s (pinned path not probed)", probe.gpu_gbps)),
    }
}

impl IntakeChoice {
    /// The mode for a transport whose full wave carries `wave_bytes` (the
    /// same for every wave size; logged with the probed landing time).
    pub(crate) fn mode_for(&self, wave_bytes: usize) -> (IntakeMode, String) {
        match (&self.probe, self.mode) {
            (Some(probe), IntakeMode::Gpu) if probe.gpu_gbps > 0.0 => {
                let ms = wave_bytes as f64 / (probe.gpu_gbps * 1e6);
                (IntakeMode::Gpu, format!("{} ({:.0} MB waves land in {ms:.1} ms)", self.reason, wave_bytes as f64 / 1e6))
            }
            _ => (self.mode, self.reason.clone()),
        }
    }
}

/// Probes dma-buf GPU landing (64 MiB loopback sends) on the current device.
pub(crate) fn probe_gpu_landing(library: &NativeLibrary) -> Result<GpuLandingProbe> {
    ensure!(!std::env::var("CUTEAFD_GPU_LANDING_PROBE").is_ok_and(|v| v == "unavailable"),
        "GPU landing probe forced unavailable");
    cuteafd_transport::gpu_landing_probe(library, None, 64 << 20, 8)
}

fn device_choice_available(choice: &IntakeChoice) -> bool {
    choice.mode == IntakeMode::Gpu
}

fn v41_choice(mut choice: IntakeChoice) -> IntakeChoice {
    // V4.1's exchange loop lacks the overlap that justifies generic auto landing.
    if choice.setting == "auto" {
        choice.mode = IntakeMode::Pinned;
        choice.reason = format!("V4.1 auto retains pinned uploads; {}", choice.reason);
    }
    choice
}

/// Preserve V4.1's pinned default while retaining capability checks for explicit GPU intake.
pub(crate) fn choose_v41_mode(library: &NativeLibrary) -> Result<IntakeChoice> {
    choose_mode(library).map(v41_choice)
}

pub(crate) fn v41_device_exchange_available(library: &NativeLibrary) -> Result<bool> {
    let choice = choose_v41_mode(library)?;
    Ok(device_choice_available(&choice))
}

/// Optional device-driven exchange must not turn a failed landing probe into
/// a startup error. The existing pinned transport remains the fallback.
pub(crate) fn device_exchange_available(library: &NativeLibrary) -> Result<bool> {
    let choice = choose_mode(library)?;
    let available = device_choice_available(&choice);
    if !available {
        tracing::warn!(reason = %choice.reason, "GPU exchange unavailable; retaining pinned-host Spark intake");
    }
    Ok(available)
}

/// Pinned host to device copy rate on the current device, GB/s: eight
/// 64 MiB copies after one warm-up, on a stream of their own.
pub(crate) fn probe_h2d(library: &NativeLibrary) -> Result<f64> {
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("probe");
    const BYTES: usize = 64 << 20;
    let host = HostAllocation::new(library, BYTES)?;
    let device = DeviceAllocation::new(library, BYTES)?;
    let stream = library.cuda_stream_create()?;
    let timed = (|| -> Result<f64> {
        let mut started = std::time::Instant::now();
        for copy in 0..9 {
            // SAFETY: both buffers hold BYTES and outlive the stream sync below.
            unsafe { library.copy_host_buffer_h2d_async(device.buffer, host.buffer, BYTES, stream)? };
            if copy == 0 {
                // SAFETY: the stream was created above.
                unsafe { library.cuda_stream_synchronize(stream)? };
                started = std::time::Instant::now();
            }
        }
        // SAFETY: as above.
        unsafe { library.cuda_stream_synchronize(stream)? };
        Ok(8.0 * BYTES as f64 / started.elapsed().as_secs_f64() / 1e9)
    })();
    // SAFETY: the stream is idle (synchronized, or its copies failed to queue).
    unsafe {
        let _ = library.cuda_stream_synchronize(stream);
        library.cuda_stream_destroy(stream)?;
    }
    timed
}

fn resolve_landing_probe(setting: &str, probe: &Result<GpuLandingProbe>, h2d: Option<f64>) -> (IntakeMode, String) {
    match probe {
        Err(error) => (IntakeMode::Pinned, format!("GPU landing probe failed: {error:#}")),
        Ok(p) if !p.usable() => (IntakeMode::Pinned, format!("GPU landing unusable: {}{}",
            p.error.as_deref().unwrap_or(&p.status),
            if p.registered && p.writes_ordering < 100 { " (GPUDirect writes not ordered for kernels)" } else { "" })),
        Ok(p) if setting == "auto" => match gpu_wins(p, h2d) {
            (true, rates) => (IntakeMode::Gpu, format!("dma-buf landing verified, {rates}")),
            (false, rates) => (IntakeMode::Pinned, format!("dma-buf landing verified but slow, {rates}")),
        },
        Ok(p) => (IntakeMode::Gpu, format!("dma-buf landing verified, {:.1} GB/s into GPU vs {:.1} GB/s into host",
            p.gpu_gbps, p.host_gbps)),
    }
}

/// Resolves `CUTEAFD_SPARK_INTAKE` (and probes dma-buf landing for `auto` and
/// `gpu`) once per process.
pub(crate) fn choose_mode(library: &NativeLibrary) -> Result<IntakeChoice> {
    static CHOICE: OnceLock<IntakeChoice> = OnceLock::new();
    if let Some(choice) = CHOICE.get() {
        return Ok(choice.clone());
    }
    let setting = std::env::var("CUTEAFD_SPARK_INTAKE").unwrap_or_else(|_| "auto".into());
    let (mode, reason, probe, h2d_gbps) = match setting.as_str() {
        "host" => (IntakeMode::Host, "CUTEAFD_SPARK_INTAKE=host".to_string(), None, None),
        "pinned" => (IntakeMode::Pinned, "CUTEAFD_SPARK_INTAKE=pinned".to_string(), None, None),
        "gpu" | "auto" => {
            let probe = probe_gpu_landing(library);
            let h2d = match (&probe, setting.as_str()) {
                (Ok(p), "auto") if p.usable() => probe_h2d(library)
                    .map_err(|error| tracing::warn!("H2D copy probe failed: {error:#}")).ok(),
                _ => None,
            };
            let (mode, reason) = resolve_landing_probe(&setting, &probe, h2d);
            (mode, reason, probe.ok(), h2d)
        }
        other => anyhow::bail!("CUTEAFD_SPARK_INTAKE={other:?} is not one of auto, gpu, pinned, host"),
    };
    let choice = IntakeChoice { setting, mode, reason, probe, h2d_gbps };
    tracing::info!(setting = %choice.setting, mode = choice.mode.name(), reason = %choice.reason, "Spark partial intake");
    Ok(CHOICE.get_or_init(|| choice).clone())
}

/// The intake mode for a transport of `ranks` x `capacity` rows of
/// `row_bytes`, logged.
pub(crate) fn transport_mode(library: &NativeLibrary, ranks: usize, capacity: usize, row_bytes: usize)
    -> Result<IntakeMode> {
    let (mode, reason) = choose_mode(library)?.mode_for(ranks * capacity * row_bytes);
    tracing::info!(mode = mode.name(), ranks, capacity, %reason, "Spark transport intake");
    Ok(mode)
}

/// `cuteafd fabric`'s view of GPU landing on one CUDA device.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct FabricLanding {
    pub(crate) cuda_device: i32,
    pub(crate) probe: Option<GpuLandingProbe>,
    pub(crate) h2d_gbps: Option<f64>,
    pub(crate) error: Option<String>,
}

impl FabricLanding {
    pub(crate) fn summary(&self) -> String {
        match (&self.probe, &self.error) {
            (Some(p), _) if p.usable() => {
                let (gpu, rates) = gpu_wins(p, self.h2d_gbps);
                format!("GPU landing (cuda {}, {}): dma-buf ok, loopback {rates}; intake auto: {}",
                    self.cuda_device, p.rdma_device, if gpu { "gpu" } else { "pinned" })
            }
            (Some(p), _) => format!("GPU landing (cuda {}): unusable ({}); intake auto: pinned", self.cuda_device,
                p.error.as_deref().unwrap_or(&p.status)),
            (None, Some(error)) => format!("GPU landing (cuda {}): not probed ({error}); intake auto: pinned",
                self.cuda_device),
            (None, None) => "GPU landing: not probed".into(),
        }
    }
}

/// Runs the GPU landing probe for `cuteafd fabric` when a native library is
/// given (or `CUTEAFD_NATIVE_LIB` names one).
pub(crate) fn fabric_probe(native_lib: Option<&std::path::Path>, device: i32) -> FabricLanding {
    let path = native_lib.map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("CUTEAFD_NATIVE_LIB").map(std::path::PathBuf::from));
    let result = (|| -> Result<(GpuLandingProbe, Option<f64>)> {
        let path = path.context("no native library (pass --native-lib or set CUTEAFD_NATIVE_LIB)")?;
        // SAFETY: a trusted image library, loaded once for this command.
        let library = unsafe { NativeLibrary::load(&path) }?;
        library.cuda_set_device(device)?;
        let probe = probe_gpu_landing(&library)?;
        let h2d = probe_h2d(&library).ok();
        Ok((probe, h2d))
    })();
    match result {
        Ok((probe, h2d_gbps)) => FabricLanding { cuda_device: device, probe: Some(probe), h2d_gbps, error: None },
        Err(error) => FabricLanding { cuda_device: device, probe: None, h2d_gbps: None, error: Some(format!("{error:#}")) },
    }
}

/// One transport's rank planes and the machinery that fills them.
pub(crate) struct SparkIntake<'a> {
    library: &'a NativeLibrary,
    mode: IntakeMode,
    ranks: usize,
    rows: usize,
    row_bytes: usize,
    planes: Vec<DeviceAllocation<'a>>,
    /// Host mode: every rank's plane back to back.
    staging: Option<RefCell<HostAllocation<'a>>>,
    /// Pinned/GPU modes: uploads from receive slots, off the compute stream.
    copy_stream: *mut c_void,
    copied: *mut c_void,
    /// Recorded after the reduce that read the planes; a new wave may be
    /// dispatched (and so may overwrite the planes) only after it.
    consumed: *mut c_void,
    consumed_pending: Cell<bool>,
    copies_pending: Cell<bool>,
    /// Receive slots whose uploads are queued; released before the next wave.
    held: Arc<Mutex<Vec<(usize, u32, VerbsHostProtocolV2ResponsePayload)>>>,
}

impl<'a> SparkIntake<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, mode: IntakeMode, ranks: usize, rows: usize, row_bytes: usize)
        -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
        ensure!((1..=MAX_INTAKE_RANKS).contains(&ranks), "{ranks} Spark ranks exceed the {MAX_INTAKE_RANKS} intake planes");
        let plane_bytes = rows * row_bytes;
        let planes = (0..ranks).map(|_| DeviceAllocation::new(library, plane_bytes.max(256)))
            .collect::<Result<Vec<_>>>()?;
        // Host mode stages every wave; pinned mode only small ones (see `stages`).
        let staging = match mode {
            IntakeMode::Host => Some(RefCell::new(HostAllocation::new(library, ranks * plane_bytes)?)),
            IntakeMode::Pinned => Some(RefCell::new(HostAllocation::new(library,
                (ranks * plane_bytes).min(SMALL_WAVE_BYTES).max(256))?)),
            IntakeMode::Gpu => None,
        };
        let copy_stream = match mode {
            IntakeMode::Host => std::ptr::null_mut(),
            _ => library.cuda_stream_create()?,
        };
        Ok(Self {
            library, mode, ranks, rows, row_bytes, planes, staging, copy_stream,
            copied: library.cuda_event_create_ordering()?,
            consumed: library.cuda_event_create_ordering()?,
            consumed_pending: Cell::new(false),
            copies_pending: Cell::new(false),
            held: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// The device ranges a GPU-mode transport lands in.
    pub(crate) fn landing(&self) -> Option<Vec<DeviceLanding>> {
        (self.mode == IntakeMode::Gpu).then(|| self.planes.iter().map(|p| DeviceLanding::new(p.buffer)).collect())
    }

    /// Points `transport`'s receives at this intake's planes (GPU mode).
    ///
    /// # Safety
    /// The intake must outlive `transport` (or its landing must be cleared
    /// first), and every dispatch on it must follow [`Self::before_dispatch`].
    pub(crate) unsafe fn attach(&self, transport: &mut SparkExperts) -> Result<()> {
        ensure!(transport.world_size() == self.ranks, "intake ranks differ from the transport's");
        // SAFETY: forwarded from this function's contract: the planes stay
        // allocated while the transport lives and are read only between a
        // wave's receive and the next dispatch.
        unsafe { transport.set_gpu_landing(self.landing()) }
    }

    /// Spawns a lane transport that lands in this intake's planes (GPU mode).
    ///
    /// # Safety
    /// As [`Self::attach`], for the lane's thread and waves.
    pub(crate) unsafe fn spawn_lane(&self, peers: Vec<std::net::SocketAddr>, executors: Vec<u64>, capacity: u32,
        config: cuteafd_transport::TcpTransportConfig) -> Result<SparkExpertLane> {
        ensure!(peers.len() == self.ranks, "intake ranks differ from the lane's");
        // SAFETY: forwarded from this function's contract.
        unsafe { SparkExpertLane::spawn_with_landing(peers, executors, capacity, config, self.landing()) }
    }

    /// Before dispatching a wave on this intake's transport: the previous
    /// wave's reduce and uploads are done, and its receive slots are released.
    pub(crate) fn before_dispatch(&self) -> Result<()> {
        if self.consumed_pending.get() {
            // SAFETY: the event was recorded on the engine's stream by `consumed`.
            unsafe { self.library.cuda_event_synchronize(self.consumed)? };
            self.consumed_pending.set(false);
        }
        if self.copies_pending.get() {
            // SAFETY: recorded after the held slots' uploads; they must finish
            // before the slots go back to the transport.
            unsafe { self.library.cuda_event_synchronize(self.copied)? };
            self.copies_pending.set(false);
        }
        self.held.lock().map_err(|_| anyhow::anyhow!("intake slots poisoned"))?.clear();
        Ok(())
    }

    /// Whether a wave of `plane_bytes` per rank goes through the pinned
    /// staging: always in host mode, and for small (decode-sized) waves in
    /// pinned mode, whose few-KB copies cost less than retaining receive
    /// slots and ordering a separate upload.
    fn stages(&self, plane_bytes: usize) -> bool {
        match self.mode {
            IntakeMode::Host => true,
            IntakeMode::Pinned => self.ranks * plane_bytes <= SMALL_WAVE_BYTES,
            IntakeMode::Gpu => false,
        }
    }

    fn check(&self, t: usize, ranks: usize) -> Result<usize> {
        ensure!(ranks == self.ranks && t <= self.rows,
            "a wave of {t} rows from {ranks} ranks exceeds the intake ({} ranks x {} rows)", self.ranks, self.rows);
        Ok(t * self.row_bytes)
    }

    /// Receives `wave` into the planes and orders the compute `stream` after
    /// it; the reduce may follow on `stream` (then call [`Self::consumed`]).
    pub(crate) async fn receive(&self, transport: &mut SparkExperts, wave: SparkExpertWave, t: usize,
        stream: *mut c_void) -> Result<WaveReceipt> {
        let plane_bytes = self.check(t, transport.world_size())?;
        let receipt = match self.stages(plane_bytes) {
            true => {
                let mut staging = self.staging.as_ref().context("host intake staging")?.borrow_mut();
                let bytes = staging.bytes_mut();
                let row_bytes = self.row_bytes;
                let receipt = transport.receive_wave(wave, |rank, first, payload| {
                    let offset = first as usize * row_bytes;
                    ensure!(offset + payload.len() <= plane_bytes, "rank {rank} partial rows run past its plane");
                    let at = rank * plane_bytes + offset;
                    copy_parallel(&mut bytes[at..at + payload.len()], payload);
                    Ok(())
                }).await?;
                drop(staging);
                self.upload_staging(receipt.landed, plane_bytes, stream)?;
                receipt
            }
            false => {
                let target = self.upload_stream(plane_bytes, stream);
                let receipt = transport.receive_wave_owned(wave, |rank, first, payload| {
                    retain_before_enqueue(&self.held, (rank, first, payload), |(rank, first, payload)| {
                        self.queue_upload(*rank, *first, payload, plane_bytes, target)
                    })
                }).await?;
                self.join_uploads(target, stream, receipt.landed)?;
                receipt
            }
        };
        Ok(receipt)
    }

    /// Queues one lane wave: `build` makes its request on the lane thread
    /// and the rank rows are gathered there; finish with [`Self::after_lane`].
    pub(crate) fn submit_lane(&self, lane: &mut SparkExpertLane, t: usize, build: LaneBuild) -> Result<()> {
        let plane_bytes = self.check(t, lane.world_size())?;
        match self.stages(plane_bytes) {
            true => {
                let staging = self.staging.as_ref().context("host intake staging")?.borrow().buffer;
                // Crosses to the lane thread as an integer; see the SAFETY note below.
                let base = staging.ptr as usize;
                let row_bytes = self.row_bytes;
                lane.submit(build, Box::new(move |rank, first, payload: &[u8]| {
                    let offset = first as usize * row_bytes;
                    ensure!(offset + payload.len() <= plane_bytes, "rank {rank} partial rows run past its plane");
                    // SAFETY: rank planes are disjoint regions of the pinned
                    // staging; the inference thread reads it (for the uploads)
                    // only after waiting for this wave, and rewrites nothing.
                    let target = unsafe {
                        std::slice::from_raw_parts_mut((base + rank * plane_bytes + offset) as *mut u8, payload.len())
                    };
                    copy_parallel(target, payload);
                    Ok(())
                }))
            }
            false => {
                let held = Arc::clone(&self.held);
                lane.submit_owned(build, Box::new(move |rank, first, payload| {
                    held.lock().map_err(|_| anyhow::anyhow!("intake slots poisoned"))?.push((rank, first, payload));
                    Ok(())
                }))
            }
        }
    }

    /// After a lane wave's wait: uploads what did not land and orders `stream`.
    pub(crate) fn after_lane(&self, t: usize, landed: u8, stream: *mut c_void) -> Result<()> {
        let plane_bytes = self.check(t, self.ranks)?;
        match self.stages(plane_bytes) {
            true => self.upload_staging(landed, plane_bytes, stream),
            false => {
                let target = self.upload_stream(plane_bytes, stream);
                let held = self.held.lock().map_err(|_| anyhow::anyhow!("intake slots poisoned"))?;
                for (rank, first, payload) in held.iter() {
                    self.queue_upload(*rank, *first, payload, plane_bytes, target)?;
                }
                drop(held);
                self.join_uploads(target, stream, landed)
            }
        }
    }

    /// Benchmarks without Spark ranks: zero partials of `t` rows uploaded
    /// through the pinned staging, as a host-mode wave would (host mode).
    pub(crate) fn upload_zeros(&self, t: usize, stream: *mut c_void) -> Result<()> {
        ensure!(self.mode == IntakeMode::Host, "zero partials go through a host-mode intake");
        let plane_bytes = self.check(t, self.ranks)?;
        self.staging.as_ref().context("host intake staging")?.borrow_mut().bytes_mut()[..self.ranks * plane_bytes]
            .fill(0);
        self.upload_staging(0, plane_bytes, stream)
    }

    /// Plane pointers for the compact reducer.
    pub(crate) fn pointers(&self) -> [*const u16; MAX_INTAKE_RANKS] {
        let mut pointers = [std::ptr::null::<u16>(); MAX_INTAKE_RANKS];
        for (slot, plane) in pointers.iter_mut().zip(&self.planes) {
            *slot = plane.buffer.ptr.cast();
        }
        pointers
    }

    /// The planes' device buffers, in rank order.
    pub(crate) fn planes(&self) -> impl Iterator<Item = CuteafdDeviceBuffer> + '_ {
        self.planes.iter().map(|p| p.buffer)
    }

    /// Marks the planes read once `stream` reaches this point (after the reduce).
    pub(crate) fn consumed(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: both handles are live; the event only orders later dispatches.
        unsafe { self.library.cuda_event_record(self.consumed, stream)? };
        self.consumed_pending.set(true);
        Ok(())
    }

    /// Terminal ownership only: the transport must first quiesce its QPs and
    /// the engine must drain its compute stream. Drain the independent copy
    /// stream directly, including uploads whose event publication failed.
    /// This method deliberately keeps every held payload until all intakes
    /// and compute streams have drained successfully.
    pub(crate) fn terminal_drain_copies(&self) -> Result<()> {
        if self.copy_stream.is_null() { return Ok(()); }
        // SAFETY: the intake owns the stream; its planes and source owners
        // remain live throughout terminal quiescence and this synchronization.
        unsafe { self.library.cuda_stream_synchronize(self.copy_stream) }
    }

    pub(crate) fn terminal_release_payloads(&self) -> Result<()> {
        self.held.lock().map_err(|_| anyhow::anyhow!("intake slots poisoned"))?.clear();
        self.copies_pending.set(false);
        self.consumed_pending.set(false);
        Ok(())
    }

    /// Small waves upload on the compute stream itself (a cross-stream wait
    /// costs more than their copies); large ones on the copy stream, so they
    /// overlap the compute stream until the reduce needs them.
    fn upload_stream(&self, plane_bytes: usize, stream: *mut c_void) -> *mut c_void {
        if self.ranks * plane_bytes <= SMALL_WAVE_BYTES { stream } else { self.copy_stream }
    }

    /// Queues `payload`'s upload into `rank`'s plane on `target`; the payload
    /// must be held until that stream passes the copy.
    fn queue_upload(&self, rank: usize, first: u32, payload: &VerbsHostProtocolV2ResponsePayload, plane_bytes: usize,
        target: *mut c_void) -> Result<()> {
        let bytes = payload.as_ref().len();
        let offset = first as usize * self.row_bytes;
        ensure!(rank < self.ranks && offset + bytes <= plane_bytes, "rank {rank} partial rows run past its plane");
        let destination = CuteafdDeviceBuffer {
            // SAFETY: the range was checked against the plane above.
            ptr: unsafe { self.planes[rank].buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..self.planes[rank].buffer
        };
        match payload.pinned_host_buffer() {
            // SAFETY: the pinned receive slot stays alive (held) until the
            // copied event completes; the target range is inside the plane.
            Some(source) => unsafe {
                self.library.copy_host_buffer_h2d_async(destination, source, bytes, target)?
            },
            // SAFETY: a pageable source is staged by the driver before return.
            None => unsafe { self.library.copy_h2d_async(destination, payload.as_ref(), target)? },
        }
        Ok(())
    }

    /// Orders `stream` after uploads queued on the copy stream (none when
    /// every rank landed in device memory or the wave used `stream` itself).
    fn join_uploads(&self, target: *mut c_void, stream: *mut c_void, landed: u8) -> Result<()> {
        if landed.count_ones() as usize == self.ranks {
            return Ok(());
        }
        // SAFETY: both streams and the event are live handles on this device.
        unsafe {
            self.library.cuda_event_record(self.copied, target)?;
            if target != stream {
                self.library.cuda_stream_wait_event(stream, self.copied)?;
            }
        }
        self.copies_pending.set(true);
        Ok(())
    }

    fn upload_staging(&self, landed: u8, plane_bytes: usize, stream: *mut c_void) -> Result<()> {
        let staging = self.staging.as_ref().context("host intake staging")?.borrow();
        for rank in (0..self.ranks).filter(|rank| landed & (1 << rank) == 0) {
            let source = CuteafdHostBuffer {
                // SAFETY: rank planes are disjoint slices of the staging buffer.
                ptr: unsafe { staging.buffer.ptr.cast::<u8>().add(rank * plane_bytes) }.cast(),
                bytes: plane_bytes,
                ..staging.buffer
            };
            // SAFETY: pinned source and device plane both hold `plane_bytes`;
            // the next wave rewrites the staging only after `before_dispatch`.
            unsafe { self.library.copy_host_buffer_h2d_async(self.planes[rank].buffer, source, plane_bytes, stream)? };
        }
        Ok(())
    }
}

/// Acquiring the ownership store precedes every enqueue. If an enqueue
/// returns an error after accepting some asynchronous work, its source still
/// belongs to the intake and can only be released after terminal drainage.
fn retain_before_enqueue<T>(held: &std::sync::Mutex<Vec<T>>, item: T,
    enqueue: impl FnOnce(&T) -> Result<()>) -> Result<()> {
    let mut held = held.lock().map_err(|_| anyhow::anyhow!("intake slots poisoned"))?;
    held.push(item);
    enqueue(held.last().expect("item was retained before enqueue"))
}

/// The stream calls a [`WarmStream`] makes (the native library's; a fake in tests).
pub(crate) trait StreamLibrary {
    fn create_stream(&self) -> Result<*mut c_void>;
    /// # Safety
    /// `stream` is a live stream this library created.
    unsafe fn synchronize_stream(&self, stream: *mut c_void) -> Result<()>;
    /// # Safety
    /// `stream` is a live stream this library created; it is not used again.
    unsafe fn destroy_stream(&self, stream: *mut c_void) -> Result<()>;
}

impl StreamLibrary for NativeLibrary {
    fn create_stream(&self) -> Result<*mut c_void> {
        self.cuda_stream_create()
    }
    unsafe fn synchronize_stream(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: the caller vouches for the stream.
        unsafe { self.cuda_stream_synchronize(stream) }
    }
    unsafe fn destroy_stream(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: the caller vouches for the stream.
        unsafe { self.cuda_stream_destroy(stream) }
    }
}

/// The temporary stream a start-up Spark transport warm-up receives its waves on. [`Self::finish`]
/// drains and destroys it and returns either failure; any other way out (an early return or `?`)
/// drops it, which drains and destroys it too and logs failures. Declare it after the transports
/// it serves, so that it drops first and their queued uploads drain while the intakes are alive.
pub(crate) struct WarmStream<'a, L: StreamLibrary = NativeLibrary> {
    library: &'a L,
    raw: *mut c_void,
}

impl<'a, L: StreamLibrary> WarmStream<'a, L> {
    pub(crate) fn new(library: &'a L) -> Result<Self> {
        Ok(Self { library, raw: library.create_stream()? })
    }

    pub(crate) fn raw(&self) -> *mut c_void {
        self.raw
    }

    /// Drains the stream, then destroys it, even when draining fails (that error comes first).
    pub(crate) fn finish(self) -> Result<()> {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: the stream was created in `new`; `ManuallyDrop` keeps `drop` from destroying it again.
        let drained = unsafe { this.library.synchronize_stream(this.raw) };
        let destroyed = unsafe { this.library.destroy_stream(this.raw) };
        drained.and(destroyed)
    }
}

impl<L: StreamLibrary> Drop for WarmStream<'_, L> {
    fn drop(&mut self) {
        // SAFETY: the stream was created in `new`; `finish`, which destroys it, never drops the guard.
        if let Err(error) = unsafe { self.library.synchronize_stream(self.raw) } {
            tracing::error!(%error, "draining a Spark transport warm-up stream");
        }
        if let Err(error) = unsafe { self.library.destroy_stream(self.raw) } {
            tracing::error!(%error, "destroying a Spark transport warm-up stream");
        }
    }
}

#[cfg(test)]
mod warm_stream_tests {
    use super::*;

    /// Records every stream call; the calls named in `fail` return an error.
    #[derive(Default)]
    struct Fake { calls: RefCell<Vec<String>>, fail: Vec<&'static str> }
    impl Fake {
        fn call(&self, name: &'static str, stream: *mut c_void) -> Result<()> {
            self.calls.borrow_mut().push(format!("{name} {:#x}", stream as usize));
            ensure!(!self.fail.contains(&name), "injected {name} failure");
            Ok(())
        }
        fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }
    }
    impl StreamLibrary for Fake {
        fn create_stream(&self) -> Result<*mut c_void> {
            let stream = 0x5000 as *mut c_void;
            self.call("create", stream)?;
            Ok(stream)
        }
        unsafe fn synchronize_stream(&self, stream: *mut c_void) -> Result<()> {
            self.call("synchronize", stream)
        }
        unsafe fn destroy_stream(&self, stream: *mut c_void) -> Result<()> {
            self.call("destroy", stream)
        }
    }

    /// The warm-ups' shape: create, receive `waves` on the stream (the one at `fail_at` fails), finish.
    fn warm(library: &Fake, waves: usize, fail_at: Option<usize>) -> Result<()> {
        let stream = WarmStream::new(library)?;
        for wave in 0..waves {
            library.call("receive", stream.raw())?;
            ensure!(Some(wave) != fail_at, "wave {wave} failed");
        }
        stream.finish()
    }

    const ALL: [&str; 4] = ["create 0x5000", "receive 0x5000", "synchronize 0x5000", "destroy 0x5000"];

    #[test]
    fn an_early_return_drains_and_destroys_the_warm_up_stream() {
        let library = Fake::default();
        assert_eq!(warm(&library, 3, Some(1)).unwrap_err().to_string(), "wave 1 failed");
        assert_eq!(library.calls(), [ALL[0], ALL[1], ALL[1], ALL[2], ALL[3]]);
        // A failing call inside the loop (`?`) unwinds the same way.
        let library = Fake { fail: vec!["receive"], ..Fake::default() };
        assert_eq!(warm(&library, 3, None).unwrap_err().to_string(), "injected receive failure");
        assert_eq!(library.calls(), ALL);
    }

    #[test]
    fn finishing_drains_then_destroys_the_stream_once() {
        let library = Fake::default();
        warm(&library, 2, None).unwrap();
        assert_eq!(library.calls(), [ALL[0], ALL[1], ALL[1], ALL[2], ALL[3]]);
    }

    #[test]
    fn a_failed_drain_or_destroy_is_reported_and_the_stream_still_goes() {
        for fail in ["synchronize", "destroy"] {
            let library = Fake { fail: vec![fail], ..Fake::default() };
            assert_eq!(warm(&library, 1, None).unwrap_err().to_string(), format!("injected {fail} failure"));
            assert_eq!(library.calls(), ALL, "{fail}");
        }
        // Dropped on an early return, the guard logs a failed drain and still destroys the stream.
        let library = Fake { fail: vec!["synchronize"], ..Fake::default() };
        assert_eq!(warm(&library, 1, Some(0)).unwrap_err().to_string(), "wave 0 failed");
        assert_eq!(library.calls(), ALL);
    }

    #[test]
    fn a_stream_that_was_not_created_is_not_destroyed() {
        let library = Fake { fail: vec!["create"], ..Fake::default() };
        assert_eq!(warm(&library, 1, None).unwrap_err().to_string(), "injected create failure");
        assert_eq!(library.calls(), [ALL[0]]);
    }

    /// GLM 5.3 Flash's and MiMo's start-up warm-ups hold their temporary stream in the guard, so
    /// every way out of them destroys it, and neither handles a raw warm-up stream any more.
    #[test]
    fn the_spark_transport_warm_ups_hold_their_stream_in_the_guard() {
        // The needles are joined here, so that this file's own text does not match them.
        let created = ["let warm_stream = crate::shared::spark_intake::Warm", "Stream::new(&"].concat();
        let finished = ["warm_stream.", "finish()?;"].concat();
        for (family, source) in [("glm5_flash", include_str!("../families/glm5_flash/mod.rs")),
            ("mimo_v2", include_str!("../families/mimo_v2/mod.rs"))] {
            assert_eq!((source.matches(&created).count(), source.matches(&finished).count()), (1, 1), "{family}");
            let (start, end) = (source.find(&created).unwrap(), source.find(&finished).unwrap());
            assert!(start < end, "{family}");
            let warm_up = &source[start..end];
            assert!(warm_up.contains(&["warm_stream.", "raw()"].concat()), "{family}");
            for raw in ["_create(", "_synchronize(", "_destroy("] {
                assert!(!warm_up.contains(&["cuda_stream", raw].concat()), "{family}: {raw}");
            }
            assert!(!source.contains(&["(warm_", "stream)"].concat()), "{family}: a raw warm-up stream call");
        }
    }
}

#[cfg(test)]
mod terminal_ownership_tests {
    use super::*;
    use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner { fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); } }

    #[test]
    fn failed_enqueue_keeps_its_source_owner() {
        let drops = Arc::new(AtomicUsize::new(0));
        let held = Mutex::new(Vec::new());
        assert!(retain_before_enqueue(&held, Owner(drops.clone()), |_| anyhow::bail!("injected enqueue failure")).is_err());
        assert_eq!(held.lock().unwrap().len(), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        held.lock().unwrap().clear();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_ownership_insertion_never_enqueues() {
        let held = Arc::new(Mutex::new(Vec::<Owner>::new()));
        let poison = held.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poison.lock().unwrap();
            panic!("injected ownership lock failure");
        }).join();
        let calls = AtomicUsize::new(0);
        let drops = Arc::new(AtomicUsize::new(0));
        assert!(retain_before_enqueue(&held, Owner(drops.clone()), |_| {
            calls.fetch_add(1, Ordering::SeqCst); Ok(())
        }).is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

impl Drop for SparkIntake<'_> {
    fn drop(&mut self) {
        // SAFETY: the handles were created by this intake; queued copies drain first.
        unsafe {
            if !self.copy_stream.is_null() {
                let _ = self.library.cuda_stream_synchronize(self.copy_stream);
                let _ = self.library.cuda_stream_destroy(self.copy_stream);
            }
            let _ = self.library.cuda_event_destroy(self.copied);
            let _ = self.library.cuda_event_synchronize(self.consumed);
            let _ = self.library.cuda_event_destroy(self.consumed);
        }
    }
}

/// `dst.copy_from_slice(src)`, split over threads for large payloads (one
/// core copies pinned memory at about 20 GB/s).
pub(crate) fn copy_parallel(dst: &mut [u8], src: &[u8]) {
    const THREADS: usize = 8;
    if src.len() < 2 << 20 {
        dst.copy_from_slice(src);
        return;
    }
    let chunk = src.len().div_ceil(THREADS).next_multiple_of(4096);
    std::thread::scope(|scope| {
        for (d, s) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
            scope.spawn(move || d.copy_from_slice(s));
        }
    });
}

/// Waves whose expert input is at least this large go out through the
/// transport's zero-copy egress buffer ([`SparkLink::egress`]); smaller
/// (decode) requests keep their own copy.
const EGRESS_MIN_BYTES: usize = 1 << 20;

/// How prefill requests leave (`CUTEAFD_SPARK_EGRESS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Egress {
    /// Each rank's session copies the payload into its own send ring.
    Copied,
    /// Zero-copy from one registered buffer, every rank's send posted at once.
    Parallel,
    /// Zero-copy, each rank's send posted once the previous one left (default).
    Staggered,
}

fn egress_setting() -> Egress {
    static SETTING: OnceLock<Egress> = OnceLock::new();
    *SETTING.get_or_init(|| {
        let egress = match std::env::var("CUTEAFD_SPARK_EGRESS").as_deref() {
            Ok("off" | "0" | "copied") => Egress::Copied,
            Ok("parallel") => Egress::Parallel,
            Ok("on" | "staggered") | Err(_) => Egress::Staggered,
            Ok(other) => {
                tracing::warn!(setting = other, "CUTEAFD_SPARK_EGRESS is staggered, parallel or off; staggered");
                Egress::Staggered
            }
        };
        tracing::info!(?egress, min_bytes = EGRESS_MIN_BYTES, "Spark request egress");
        egress
    })
}

fn egress_enabled() -> bool {
    egress_setting() != Egress::Copied
}

/// Logs once how this coordinator moves expert traffic, from the fabric it
/// detected: requests leave zero-copy from one registered buffer
/// (`CUTEAFD_SPARK_EGRESS`), partials arrive by the probed intake
/// (`CUTEAFD_SPARK_INTAKE`), and every rank's traffic uses the coordinator's
/// first rail. Measured on raptor (one 400 Gb port, both GPUs a host bridge
/// away from the NIC) with six Sparks: the coordinator's intake (~19 GB/s
/// landed in GPU memory) bounds a wave long before the ranks' links do (six
/// rails of 100-200 Gb), so striping ranks over rails gains nothing there; a
/// split GPU/pinned intake was slower than GPU landing alone.
fn log_transfer_plan(mode: IntakeMode, ranks: usize) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| match cuteafd_transport::fabric::discover() {
        Ok(report) => {
            let ports = report.rails.rails.iter().map(|rail| format!("{}@{} {:.0}G", rail.device, rail.address,
                rail.effective_gbps)).collect::<Vec<_>>().join(", ");
            tracing::info!(ranks, intake = mode.name(), egress = ?egress_setting(),
                coordinator_rails = %ports, "Spark transfer plan: every rank on the first coordinator rail");
        }
        Err(error) => tracing::warn!("Spark transfer plan: fabric discovery failed: {error:#}"),
    });
}

/// A Spark transport and the intake its waves land in (dropped in that order).
pub(crate) struct SparkLink<'a> {
    pub(crate) transport: SparkExperts,
    pub(crate) intake: std::rc::Rc<SparkIntake<'a>>,
    /// Bytes of the transport's egress buffer (0: none).
    egress_bytes: usize,
    terminal_library: Option<Arc<NativeLibrary>>,
    terminal_released: bool,
}

impl<'a> SparkLink<'a> {
    /// Connects a [`SparkExperts::new_ranks`] transport whose `capacity`-row
    /// waves land in an intake of the process-wide [`choose_mode`].
    pub(crate) fn new(library: &'a NativeLibrary, peers: &[std::net::SocketAddr], executors: &[u64], capacity: u32,
        config: cuteafd_transport::TcpTransportConfig, row_bytes: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
        let mode = transport_mode(library, peers.len(), capacity as usize, row_bytes)?;
        let intake = SparkIntake::new(library, mode, peers.len(), capacity as usize, row_bytes)?;
        let mut transport = SparkExperts::new_ranks(peers, executors, capacity, config)?;
        // SAFETY: the link drops its transport before its intake, and every
        // dispatch goes through `dispatch`, which calls `before_dispatch`.
        unsafe { intake.attach(&mut transport)? };
        // Expert input rows are at most BF16, as wide as a partial row.
        log_transfer_plan(mode, peers.len());
        let egress_bytes = if egress_enabled() { capacity as usize * row_bytes } else { 0 };
        if egress_bytes > 0 {
            transport.enable_egress(egress_bytes, egress_setting() == Egress::Staggered)?;
        }
        Ok(Self { transport, intake: std::rc::Rc::new(intake), egress_bytes, terminal_library: None, terminal_released: false })
    }

    /// Where to write a wave's `bytes` of expert input so it goes out
    /// zero-copy (then build the request with [`Self::egress_payload`]), or
    /// `None` for a small wave or without egress. Waits for the previous
    /// wave's request sends; the previous request must have been dropped.
    pub(crate) fn egress(&mut self, bytes: usize) -> Result<Option<CuteafdHostBuffer>> {
        if bytes < EGRESS_MIN_BYTES || bytes > self.egress_bytes {
            return Ok(None);
        }
        self.transport.egress_target().map(Some)
    }

    /// The first `bytes` of the egress buffer as a request payload.
    pub(crate) fn egress_payload(&self, bytes: usize) -> Result<bytes::Bytes> {
        self.transport.egress_payload(bytes)
    }

    pub(crate) fn world_size(&self) -> usize {
        self.transport.world_size()
    }

    /// Must precede connection/bootstrap so a temporary failed endpoint can
    /// retain the same external landing and pinned-upload owners as submit.
    pub(crate) fn enable_terminal_ownership(&mut self, library: Arc<NativeLibrary>) -> Result<()> {
        ensure!(std::ptr::eq(library.as_ref(),self.intake.library),"terminal intake must retain its actual native library owner");
        self.intake.library.rdma_rc_endpoint_quiesce_available()?;
        self.transport.enable_terminal_ownership()?;
        self.terminal_library = Some(library);
        Ok(())
    }

    pub(crate) fn terminal_quiesce(&mut self) -> Result<()> { self.transport.terminal_quiesce() }
    pub(crate) fn terminal_drain_copies(&self) -> Result<()> { self.intake.terminal_drain_copies() }
    pub(crate) fn terminal_release(&mut self) -> Result<()> {
        // Validate successful QP quiescence before releasing any held source.
        // Those payloads retain their endpoint registrations until this final
        // clear, even after the transport drops its drained pending owners.
        self.transport.terminal_release()?;
        self.intake.terminal_release_payloads()?;
        self.terminal_released = true;
        Ok(())
    }

    /// Posts `request` to every rank once the previous wave's planes are free.
    pub(crate) fn dispatch(&mut self, request: &cuteafd_transport::ExpertProtocolV2Request) -> Result<SparkExpertWave> {
        self.intake.before_dispatch()?;
        self.transport.dispatch_wave(request)
    }

    /// Receives `wave` (`t` rows) into the planes, ordering `stream` after it.
    pub(crate) async fn receive(&mut self, wave: SparkExpertWave, t: usize, stream: *mut c_void)
        -> Result<WaveReceipt> {
        self.intake.receive(&mut self.transport, wave, t, stream).await
    }

    /// Queues `output = shared + sum of the rank planes` for `t` rows on
    /// `stream` (after [`Self::receive`]) and releases the planes after it.
    ///
    /// # Safety
    /// `shared` and `output` are live `[t, hidden]` BF16 device buffers,
    /// ordered on `stream` before this call.
    pub(crate) unsafe fn reduce(&self, shared: *const u16, output: *mut u16, t: usize, stream: *mut c_void)
        -> Result<()> {
        let library = self.intake.library;
        // SAFETY: the planes hold this wave's `t` rows once `stream` reaches
        // this point (see `receive`); the caller vouches for the rest.
        unsafe {
            library.v41_compact_reducer()?.reduce_planes(self.intake.pointers(), self.world_size() as u32, shared,
                output, t as u32, stream)?;
        }
        self.intake.consumed(stream)
    }
}

impl Drop for SparkLink<'_> {
    fn drop(&mut self) {
        if self.transport.terminal_owned() && !self.terminal_released {
            // Drop may run inside fallible connection or submission scopes,
            // before the engine regains control. QP quiescence alone cannot
            // prove the compute stream passed these planes/pinned sources.
            // Retain the whole intake until process teardown. The transport
            // separately retains endpoint/library owners if quiescence fails.
            std::mem::forget(self.intake.clone());
            if let Some(library)=self.terminal_library.take() { std::mem::forget(library); }
            tracing::error!("retaining terminal intake whose complete consumer drainage was not proven");
        }
    }
}

/// A prefill lane (transport on its own thread) and the intake its waves
/// land in (dropped in that order: the lane joins its thread first).
pub(crate) struct SparkLane<'a> {
    pub(crate) lane: SparkExpertLane,
    pub(crate) intake: SparkIntake<'a>,
}

impl<'a> SparkLane<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, peers: Vec<std::net::SocketAddr>, executors: Vec<u64>, capacity: u32,
        config: cuteafd_transport::TcpTransportConfig, row_bytes: usize) -> Result<Self> {
        let mode = transport_mode(library, peers.len(), capacity as usize, row_bytes)?;
        let intake = SparkIntake::new(library, mode, peers.len(), capacity as usize, row_bytes)?;
        // SAFETY: the lane drops (joining its thread) before its intake, and
        // every submit goes through `submit`, which calls `before_dispatch`.
        let lane = unsafe { intake.spawn_lane(peers, executors, capacity, config)? };
        Ok(Self { lane, intake })
    }

    pub(crate) fn world_size(&self) -> usize {
        self.lane.world_size()
    }

    /// Queues a `t`-row wave whose request `build` makes on the lane thread.
    pub(crate) fn submit(&mut self, t: usize, build: LaneBuild) -> Result<()> {
        self.intake.before_dispatch()?;
        self.intake.submit_lane(&mut self.lane, t, build)
    }

    /// Waits for the submitted wave and orders `stream` after its planes.
    pub(crate) fn wait(&mut self, t: usize, timeout: std::time::Duration, stream: *mut c_void)
        -> Result<cuteafd_transport::expert::LaneTimes> {
        let times = self.lane.wait(timeout)?;
        self.intake.after_lane(t, times.landed, stream)?;
        Ok(times)
    }
}

/// Whether engines that support it exchange decode/verify waves through the
/// device-driven [`SparkDeviceLink`] (`CUTEAFD_SPARK_DEVICE=1`; off by default).
pub(crate) fn device_exchange_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("CUTEAFD_SPARK_DEVICE").as_deref(), Ok("1" | "on" | "true")))
}

impl Drop for SparkDeviceLink<'_> {
    fn drop(&mut self) {
        let stats = self.lane.stats();
        tracing::info!(waves = stats.waves, build_post_us = stats.build_post_us, receive_us = stats.receive_us,
            wakes = stats.wakes, wake_us = stats.wake_us, "device-driven Spark exchange closed");
    }
}

/// The device-driven Spark exchange (PLAN.md): the engine's stream copies a
/// wave's routes and wire rows into a pinned mailbox and publishes it
/// ([`Self::dispatch`]); a proxy thread ([`SparkDeviceLane`]) posts it and
/// lands every rank's partials in this link's planes; the stream waits for
/// the proxy's completion ([`Self::collect`]) and reduces ([`Self::reduce`]).
/// No host wait, parse or launch sits between a layer's router and its
/// reduce, so a step's layers can be queued (and captured) back to back.
pub(crate) struct SparkDeviceLink<'a> {
    /// Dropped first: joins the proxy before its mailbox and planes go.
    lane: cuteafd_transport::expert::SparkDeviceLane,
    intake: SparkIntake<'a>,
    mailbox: HostAllocation<'a>,
    /// Write mode (`CUTEAFD_SPARK_WRITE=1`): the ranks RDMA-write their rows
    /// and completion flags here (planes, then one 64-byte flag line per rank);
    /// no receive and no host thread is involved in a wave's return.
    written: Option<DeviceAllocation<'a>>,
    library: &'a NativeLibrary,
    capacity: usize,
    topk: usize,
    wire_row_bytes: usize,
    row_bytes: usize,
    ranks: usize,
}

/// Write mode: one plane per rank, then one 64-byte line per rank flag.
fn written_layout(ranks: usize, capacity: usize, row_bytes: usize) -> (usize, usize) {
    let plane = (capacity * row_bytes).next_multiple_of(256);
    (plane, ranks * plane)
}

/// Whether device exchanges have the ranks write their responses
/// (`CUTEAFD_SPARK_WRITE=1`; needs workers that accept write targets).
pub(crate) fn written_responses() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("CUTEAFD_SPARK_WRITE").as_deref(), Ok("1" | "on")))
}

impl<'a> SparkDeviceLink<'a> {
    /// Connects the proxy's transport (`capacity` rows of `topk` routes and
    /// `wire_row_bytes` per wave; partial rows of `row_bytes`) on `device`.
    /// Needs GPU landing (the probed intake mode must be `gpu`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(library: &'a NativeLibrary, device: i32, peers: &[std::net::SocketAddr], executors: &[u64],
        capacity: usize, topk: usize, wire_row_bytes: usize, row_bytes: usize,
        config: cuteafd_transport::TcpTransportConfig, warm: Option<cuteafd_transport::ExpertProtocolV2Request>,
        build: cuteafd_transport::expert::DeviceBuild) -> Result<Self> {
        use cuteafd_transport::expert::device_mailbox;
        let choice = choose_mode(library)?;
        ensure!(device_choice_available(&choice),
            "the device Spark exchange needs GPU landing, but the intake is {} ({})", choice.mode.name(), choice.reason);
        let intake = SparkIntake::new(library, IntakeMode::Gpu, peers.len(), if written_responses() { 1 } else { capacity },
            row_bytes)?;
        let mailbox = HostAllocation::new(library, device_mailbox::bytes(capacity, topk, wire_row_bytes))?;
        library.peer_exchange_initialize()?;
        let landing = intake.landing().context("GPU intake planes")?;
        let ranks = peers.len();
        let (written, targets) = if written_responses() {
            let (plane, flags) = written_layout(ranks, capacity, row_bytes);
            // dma-buf export covers the whole allocation, which must be page-sized.
            let allocation = DeviceAllocation::new(library, (flags + ranks * 64).next_multiple_of(2 << 20))?;
            library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { ptr: unsafe { allocation.buffer.ptr.cast::<u8>()
                .add(flags) }.cast(), bytes: ranks * 64, ..allocation.buffer }, &vec![0u8; ranks * 64])?;
            let targets = (0..ranks).map(|rank| cuteafd_transport::DeviceWriteTarget {
                base: allocation.buffer.ptr as usize, bytes: allocation.buffer.bytes, plane_offset: rank * plane,
                plane_bytes: capacity * row_bytes, flag_offset: flags + rank * 64 }).collect();
            (Some(allocation), Some(targets))
        } else { (None, None) };
        // SAFETY: the mailbox and the intake planes are fields of this link,
        // dropped after the lane (which joins its thread first); the engine
        // publishes a wave only after it waited for the previous one and
        // reduced its planes (`dispatch` follows `collect` + `reduce` in
        // stream order).
        let lane = unsafe {
            cuteafd_transport::expert::SparkDeviceLane::spawn(peers.to_vec(), executors.to_vec(), capacity as u32,
                config, landing, device, mailbox.buffer.ptr as usize, mailbox.buffer.bytes, topk, wire_row_bytes,
                warm, targets, build)?
        };
        tracing::info!(ranks, capacity, written = written.is_some(),
            "device-driven Spark exchange ready (proxy thread posts; responses land in GPU memory)");
        Ok(Self { lane, intake, mailbox, written, library, capacity, topk, wire_row_bytes, row_bytes, ranks })
    }

    pub(crate) fn world_size(&self) -> usize {
        self.lane.world_size()
    }

    /// Plane pointers for the compact reducer.
    pub(crate) fn pointers(&self) -> [*const u16; MAX_INTAKE_RANKS] {
        match &self.written {
            Some(allocation) => {
                let (plane, _) = written_layout(self.ranks, self.capacity, self.row_bytes);
                let mut pointers = [std::ptr::null::<u16>(); MAX_INTAKE_RANKS];
                for (rank, slot) in pointers.iter_mut().enumerate().take(self.ranks) {
                    // SAFETY: rank planes lie inside the written allocation.
                    *slot = unsafe { allocation.buffer.ptr.cast::<u8>().add(rank * plane) }.cast();
                }
                pointers
            }
            None => self.intake.pointers(),
        }
    }

    /// Before queuing a step's waves; returns an earlier wave's error.
    pub(crate) fn arm(&self) -> Result<()> {
        self.lane.arm()
    }

    /// Announces `waves` waves the stream will publish (before it can): the
    /// proxy spins only while announced waves are outstanding.
    pub(crate) fn expect(&self, waves: u64) {
        self.lane.expect(waves)
    }

    /// After the step's stream drained: an error of any of its waves.
    pub(crate) fn check(&self) -> Result<()> {
        use cuteafd_transport::expert::device_mailbox as m;
        self.lane.check()?;
        if self.written.is_some() {
            // SAFETY: a u32 word of the live pinned mailbox, written by the GPU's waits.
            let word = unsafe { &*(self.mailbox_at(m::WRITE_ERROR).ptr as *const std::sync::atomic::AtomicU32) };
            let failed = word.swap(0, std::sync::atomic::Ordering::AcqRel);
            ensure!(failed == 0, "device Spark exchange: ranks {failed:#b} could not write their responses");
        }
        Ok(())
    }

    pub(crate) fn stats(&self) -> cuteafd_transport::expert::DeviceLaneStats {
        self.lane.stats()
    }

    /// The lane's and the mailbox's sequences, for a stuck-step report.
    pub(crate) fn state(&self) -> String {
        use cuteafd_transport::expert::device_mailbox as m;
        // SAFETY: u32 words of the live pinned mailbox.
        let word = |offset| unsafe { std::ptr::read_volatile(self.mailbox_at(offset).ptr as *const u32) };
        format!("{}; mailbox ready {} done {} send_state {} recv_state {}", self.lane.state(), word(m::READY),
            word(m::DONE), word(m::SEND_STATE), word(m::RECV_STATE))
    }

    fn mailbox_at(&self, offset: usize) -> CuteafdHostBuffer {
        CuteafdHostBuffer {
            // SAFETY: offsets come from `device_mailbox` and lie inside it.
            ptr: unsafe { self.mailbox.buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes: self.mailbox.buffer.bytes - offset,
            ..self.mailbox.buffer
        }
    }

    /// Loads the signal/wait kernels on another GPU whose streams will
    /// dispatch or collect on this link (the sequences live in the mailbox).
    pub(crate) fn initialize_on(&self, device: i32) -> Result<()> {
        let previous = self.library.cuda_get_device()?;
        self.library.cuda_set_device(device)?;
        let loaded = self.library.peer_exchange_initialize();
        self.library.cuda_set_device(previous)?;
        loaded
    }

    /// Queues on `stream` the copy of a wave's `rows` x `topk` route ids
    /// (u32) and gate weights (f32) and its wire rows into the mailbox, then
    /// publishes it (layer, rows and `kind` for the proxy's request builder).
    ///
    /// # Safety
    /// `ids`, `weights` and `wire` are live device buffers holding the wave,
    /// complete in `stream` order; every earlier wave on this link was
    /// collected and reduced earlier on `stream`.
    pub(crate) unsafe fn dispatch(&self, layer: usize, rows: usize, kind: u32, ids: CuteafdDeviceBuffer,
        weights: CuteafdDeviceBuffer, wire: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        use cuteafd_transport::expert::device_mailbox as m;
        ensure!(rows > 0 && rows <= self.capacity, "a device wave of {rows} rows exceeds the link's {}", self.capacity);
        let (route_bytes, wire_bytes) = (rows * self.topk * 4, rows * self.wire_row_bytes);
        // SAFETY: the mailbox ranges hold a full-capacity wave; the caller
        // vouches for the sources and their ordering.
        unsafe {
            self.library.copy_d2h_host_buffer_async(self.mailbox_at(m::ROUTES), ids, route_bytes, stream)?;
            self.library.copy_d2h_host_buffer_async(self.mailbox_at(m::weights(self.capacity, self.topk)), weights,
                route_bytes, stream)?;
            self.library.copy_d2h_host_buffer_async(self.mailbox_at(m::wire(self.capacity, self.topk)), wire,
                wire_bytes, stream)?;
            self.library.host_signal(self.mailbox_at(m::READY).ptr.cast(), self.mailbox_at(m::SEND_STATE).ptr.cast(),
                self.mailbox_at(m::DESCRIPTOR).ptr.cast(), [layer as u32, rows as u32, kind, self.topk as u32], stream)
        }
    }

    /// Queues on `stream` the wait for the proxy's completion of the oldest
    /// published, not yet collected wave (its partials are then in the planes).
    ///
    /// # Safety
    /// A [`Self::dispatch`] for this wait was queued earlier on `stream`.
    pub(crate) unsafe fn collect(&self, stream: *mut c_void) -> Result<()> {
        use cuteafd_transport::expert::device_mailbox as m;
        if let Some(allocation) = &self.written {
            let (_, flags) = written_layout(self.ranks, self.capacity, self.row_bytes);
            // SAFETY: the flags lie in the written allocation the ranks write;
            // the state and error words are pinned, device-mapped mailbox memory.
            return unsafe {
                self.library.spark_wait_written(allocation.buffer.ptr.cast::<u8>().add(flags).cast(), self.ranks as u32,
                    8, self.mailbox_at(m::RECV_STATE).ptr.cast(), self.mailbox_at(m::WRITE_ERROR).ptr.cast(), stream)
            };
        }
        // SAFETY: the DONE word is pinned, device-mapped mailbox memory; the
        // state word is pinned mailbox memory too.
        unsafe { self.library.peer_wait(self.mailbox_at(m::DONE).ptr.cast(), self.mailbox_at(m::RECV_STATE).ptr.cast(),
            stream) }
    }

    /// Queues `output = shared + sum of the rank planes` for `t` rows on
    /// `stream` (after [`Self::collect`]).
    ///
    /// # Safety
    /// As [`SparkLink::reduce`].
    pub(crate) unsafe fn reduce(&self, shared: *const u16, output: *mut u16, t: usize, stream: *mut c_void)
        -> Result<()> {
        // SAFETY: the planes hold the collected wave's `t` rows in `stream`
        // order; the caller vouches for the rest.
        unsafe {
            self.library.v41_compact_reducer()?.reduce_planes(self.pointers(), self.world_size() as u32, shared,
                output, t as u32, stream)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(gpu_gbps: f64) -> GpuLandingProbe {
        GpuLandingProbe {
            rdma_device: "mlx5_0".into(), cuda_device: 0, dma_buf: true, gpudirect_rdma: true,
            writes_ordering: 100, registered: true, gpu_gbps, host_gbps: 52.4, status: String::new(), error: None,
        }
    }

    #[test]
    fn auto_lands_on_the_gpu_from_half_the_pinned_path_rate() {
        // raptor GPU0: 19.6 GB/s landing vs 52.4 into host and 57.6 H2D ->
        // 27.4 pinned path, share 0.71: measured faster at 134 and 352 MB waves.
        let pinned = pinned_path_gbps(52.4, 57.6);
        assert!((pinned - 27.4).abs() < 0.1);
        assert!(gpu_wins(&probe(19.6), Some(57.6)).0);
        // Loopback GPU1 per-wave share 0.53 also won end to end.
        assert!(gpu_wins(&probe(0.53 * pinned), Some(57.6)).0);
        // A landing path at a third of the pinned path's rate stays pinned.
        assert!(!gpu_wins(&probe(pinned / 3.0), Some(57.6)).0);
        // Without the H2D probe, a verified landing path is used.
        assert!(gpu_wins(&probe(19.6), None).0);
    }

    #[test]
    fn v41_auto_keeps_pinned_at_parity_rates_without_changing_generic_auto() {
        let mut measured = probe(19.2);
        measured.host_gbps = 55.0;
        let h2d = Some(55.0);
        assert_eq!(pinned_path_gbps(measured.host_gbps, h2d.unwrap()), 27.5);
        let (mode, reason) = resolve_landing_probe("auto", &Ok(measured.clone()), h2d);
        assert_eq!(mode, IntakeMode::Gpu);
        let generic = IntakeChoice { setting: "auto".into(), mode, reason,
            probe: Some(measured), h2d_gbps: h2d };
        assert!(device_choice_available(&generic));
        let v41 = v41_choice(generic);
        assert_eq!(v41.mode, IntakeMode::Pinned);
        assert!(!device_choice_available(&v41));
        assert_eq!(v41.mode_for(4096 * 10240).0, IntakeMode::Pinned);
    }

    #[test]
    fn v41_explicit_gpu_preserves_the_capability_fallback() {
        let measured = probe(19.2);
        for supported in [true, false] {
            let result = if supported { Ok(measured.clone()) }
                else { Err(anyhow::anyhow!("forced unavailable")) };
            let (mode, reason) = resolve_landing_probe("gpu", &result, None);
            let choice = v41_choice(IntakeChoice { setting: "gpu".into(), mode, reason,
                probe: result.ok(), h2d_gbps: None });
            assert_eq!(device_choice_available(&choice), supported);
            assert_eq!(choice.mode, if supported { IntakeMode::Gpu } else { IntakeMode::Pinned });
        }
    }

    #[test]
    fn resolved_pinned_intake_cannot_be_upgraded_by_a_usable_probe() {
        let measured = probe(1.0);
        let (mode, reason) = resolve_landing_probe("auto", &Ok(measured.clone()), Some(57.6));
        assert_eq!(mode, IntakeMode::Pinned);
        let choice = IntakeChoice { setting: "auto".into(), mode, reason,
            probe: Some(measured), h2d_gbps: Some(57.6) };
        assert!(choice.probe.as_ref().unwrap().usable());
        assert!(!device_choice_available(&choice));
    }

    #[test]
    fn unavailable_landing_keeps_pinned_intake_and_disables_device_exchange() {
        for setting in ["auto", "gpu"] {
            let (mode, reason) = resolve_landing_probe(setting, &Err(anyhow::anyhow!("forced unavailable")), None);
            assert!(reason.contains("forced unavailable"));
            let choice = IntakeChoice { setting: setting.into(), mode,
                reason, probe: None, h2d_gbps: None };
            assert!(!device_choice_available(&choice));
            assert_eq!(choice.mode_for(4096 * 10240).0, IntakeMode::Pinned);
        }
    }

    #[test]
    fn mode_for_keeps_the_resolved_mode_for_every_wave_size() {
        let choice = |mode| IntakeChoice { setting: "auto".into(), mode, reason: "probe".into(),
            probe: Some(probe(19.6)), h2d_gbps: Some(57.6) };
        // GLM 5.3 Flash TP4 (134 MB) and V4 Pro TP6 (352 MB) both land on the GPU.
        assert_eq!(choice(IntakeMode::Gpu).mode_for(4 * 4096 * 4096 * 2).0, IntakeMode::Gpu);
        assert_eq!(choice(IntakeMode::Gpu).mode_for(6 * 4096 * 7168 * 2).0, IntakeMode::Gpu);
        assert_eq!(choice(IntakeMode::Pinned).mode_for(1).0, IntakeMode::Pinned);
    }
}
