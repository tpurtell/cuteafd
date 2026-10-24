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
pub(crate) const MAX_INTAKE_RANKS: usize = 8;

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
    let probe = cuteafd_transport::gpu_landing_probe(library, None, 64 << 20, 8)?;
    let link = landing_pcie_link(library);
    tracing::info!(cuda_device = probe.cuda_device, gpu_gbps = probe.gpu_gbps,
        host_gbps = probe.host_gbps, pcie_generation = ?link.as_ref().and_then(|link| link.generation),
        pcie_width = ?link.as_ref().and_then(|link| link.width), usable = probe.usable(), "GPU landing probe");
    Ok(probe)
}

fn device_choice_available(choice: &IntakeChoice) -> bool {
    choice.mode == IntakeMode::Gpu
}

/// Intake choice for executors whose receive uploads do not overlap compute.
/// Explicit settings still win; only the automatic default is constrained.
#[derive(Debug, Clone, Copy)]
pub(crate) enum IntakePreference { Probed, Pinned }

fn preferred_choice(mut choice: IntakeChoice, preference: IntakePreference) -> IntakeChoice {
    if choice.setting == "auto" && matches!(preference, IntakePreference::Pinned) {
        choice.mode = IntakeMode::Pinned;
        choice.reason = format!("executor retains pinned uploads; {}", choice.reason);
    }
    choice
}

pub(crate) fn choose_preferred_mode(library: &NativeLibrary, preference: IntakePreference) -> Result<IntakeChoice> {
    choose_mode(library).map(|choice| preferred_choice(choice, preference))
}

pub(crate) fn preferred_device_exchange_available(library: &NativeLibrary, preference: IntakePreference) -> Result<bool> {
    Ok(device_choice_available(&choose_preferred_mode(library, preference)?))
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

/// Exercise the copy engine and PCIe link before timing peer writes.
fn warm_landing_link(library: &NativeLibrary) -> Result<()> {
    probe_h2d(library).map(|_| ())
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

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct LandingPcieLink {
    generation: Option<u32>,
    width: Option<u32>,
}

fn landing_pcie_link(library: &NativeLibrary) -> Option<LandingPcieLink> {
    let bus = library.cuda_device_pci_bus_id(library.cuda_get_device().ok()?)?;
    let nvml = cuteafd_ffi::nvml::Nvml::open().ok()?;
    let devices = nvml.devices().ok()?;
    let device = devices.iter().find(|device| device.identity.pci_bus_id.trim_start_matches('0')
        .eq_ignore_ascii_case(bus.trim_start_matches('0')))?;
    let (generation, width) = device.pcie_link();
    Some(LandingPcieLink { generation, width })
}

/// `cuteafd fabric`'s view of GPU landing on one CUDA device.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct FabricLanding {
    pub(crate) cuda_device: i32,
    pub(crate) pcie_before: Option<LandingPcieLink>,
    pub(crate) pcie_link: Option<LandingPcieLink>,
    pub(crate) probe: Option<GpuLandingProbe>,
    pub(crate) h2d_gbps: Option<f64>,
    pub(crate) error: Option<String>,
}

impl FabricLanding {
    pub(crate) fn summary(&self) -> String {
        match (&self.probe, &self.error) {
            (Some(p), _) if p.usable() => {
                let (gpu, rates) = gpu_wins(p, self.h2d_gbps);
                let link = self.pcie_link.as_ref().map(|link| format!("PCIe gen {} x{}",
                    link.generation.map_or("?".into(), |value| value.to_string()),
                    link.width.map_or("?".into(), |value| value.to_string())))
                    .unwrap_or_else(|| "PCIe unknown".into());
                format!("GPU landing (cuda {}, {}, {link}): dma-buf ok, loopback {rates}; intake auto: {}",
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
pub(crate) fn fabric_probe(native_lib: Option<&std::path::Path>, device: i32, warmup: bool, repeats: u32)
    -> Vec<FabricLanding> {
    let library = (|| -> Result<NativeLibrary> {
        let path = native_lib.map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("CUTEAFD_NATIVE_LIB").map(std::path::PathBuf::from))
            .context("no native library (pass --native-lib or set CUTEAFD_NATIVE_LIB)")?;
        // SAFETY: a trusted image library, kept loaded across all repeats.
        let library = unsafe { NativeLibrary::load(&path) }?;
        library.cuda_set_device(device)?;
        Ok(library)
    })();
    (0..repeats.clamp(1, 5)).map(|_| {
        let pcie_before = library.as_ref().ok().and_then(landing_pcie_link);
        let result = (|| -> Result<(GpuLandingProbe, Option<f64>, Option<LandingPcieLink>)> {
            let library = library.as_ref().map_err(|error| anyhow::anyhow!("{error:#}"))?;
            if warmup { warm_landing_link(library)?; }
            let probe = probe_gpu_landing(library)?;
            let pcie_link = landing_pcie_link(library);
            let h2d = probe_h2d(library).ok();
            Ok((probe, h2d, pcie_link))
        })();
        match result {
            Ok((probe, h2d_gbps, pcie_link)) => FabricLanding { cuda_device: device, pcie_before, pcie_link, probe: Some(probe), h2d_gbps, error: None },
            Err(error) => FabricLanding { cuda_device: device, pcie_before, pcie_link: None, probe: None, h2d_gbps: None, error: Some(format!("{error:#}")) },
        }
    }).collect()
}

/// One transport's rank planes and the machinery that fills them.
///
/// With `waves > 1` the intake owns that many return slots, each a full set of
/// rank planes. Waves take slots in dispatch order (`sequence % waves`), the
/// same order the transport receives them; a slot is reused only after the
/// reduce that read its previous wave (its own consumed event).
pub(crate) struct SparkIntake<'a> {
    library: &'a NativeLibrary,
    mode: IntakeMode,
    ranks: usize,
    rows: usize,
    row_bytes: usize,
    /// One rank-contiguous allocation containing every slot for that rank.
    planes: Vec<DeviceAllocation<'a>>,
    waves: usize,
    state: IntakeState,
    /// Host mode: every rank's plane back to back.
    staging: Option<RefCell<HostAllocation<'a>>>,
    /// Pinned/GPU modes: uploads from receive slots, off the compute stream.
    copy_stream: *mut c_void,
    copied: *mut c_void,
    /// Per slot: recorded after the reduce that read the slot's planes; a new
    /// wave may be dispatched into that slot only after it.
    consumed: Vec<*mut c_void>,
    /// Receive slots whose uploads are queued; released before the next wave.
    held: Arc<Mutex<Vec<(usize, u32, VerbsHostProtocolV2ResponsePayload)>>>,
}

struct IntakeState {
    dispatch_slot: Cell<usize>,
    receive_slot: Cell<usize>,
    consumed_pending: Vec<Cell<bool>>,
    copies_pending: Cell<bool>,
}
impl IntakeState {
    fn new(waves: usize) -> Self {
        Self { dispatch_slot: Cell::new(0), receive_slot: Cell::new(0),
            consumed_pending: (0..waves).map(|_| Cell::new(false)).collect(),
            copies_pending: Cell::new(false) }
    }

    fn before_dispatch(&self, wait_consumed: impl FnOnce(usize) -> Result<()>,
        release_uploads: impl FnOnce() -> Result<()>) -> Result<()> {
        let slot = self.dispatch_slot.get();
        if self.consumed_pending[slot].get() {
            wait_consumed(slot)?;
            self.consumed_pending[slot].set(false);
        }
        release_uploads()?;
        self.dispatch_slot.set((slot + 1) % self.consumed_pending.len());
        Ok(())
    }

    fn release_uploads(&self, wait_copied: impl FnOnce() -> Result<()>,
        release: impl FnOnce() -> Result<()>) -> Result<()> {
        if self.copies_pending.get() {
            wait_copied()?;
            self.copies_pending.set(false);
        }
        release()
    }

    fn reset(&self, drain: impl FnOnce() -> Result<()>, reset_transport: impl FnOnce() -> Result<()>,
        release: impl FnOnce() -> Result<()>) -> Result<()> {
        drain()?;
        reset_transport()?;
        release()?;
        self.dispatch_slot.set(0);
        self.receive_slot.set(0);
        self.copies_pending.set(false);
        for pending in &self.consumed_pending { pending.set(false); }
        Ok(())
    }
}

impl<'a> SparkIntake<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, mode: IntakeMode, ranks: usize, rows: usize, row_bytes: usize)
        -> Result<Self> {
        Self::with_waves(library, mode, ranks, rows, row_bytes, 1)
    }

    /// `waves` return slots, registered as one contiguous range per rank.
    pub(crate) fn with_waves(library: &'a NativeLibrary, mode: IntakeMode, ranks: usize, rows: usize,
        row_bytes: usize, waves: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
        ensure!((1..=8).contains(&ranks), "{ranks} Spark ranks are outside the one-through-eight transport worlds");
        ensure!((1..=4).contains(&waves), "Spark intake admits 1..=4 return slots");
        let plane_bytes = rows * row_bytes;
        let planes = (0..ranks).map(|_| DeviceAllocation::new(library, (plane_bytes * waves).max(256)))
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
            library, mode, ranks, rows, row_bytes, planes, waves,
            state: IntakeState::new(waves), staging, copy_stream,
            copied: library.cuda_event_create_ordering()?,
            consumed: (0..waves).map(|_| library.cuda_event_create_ordering()).collect::<Result<_>>()?,
            held: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// The device ranges a GPU-mode transport lands in.
    pub(crate) fn landing(&self) -> Option<Vec<DeviceLanding>> {
        (self.mode == IntakeMode::Gpu).then(|| self.planes.iter().map(|p|
            DeviceLanding::with_slots(p.buffer, self.waves).expect("validated intake slot geometry")).collect())
    }

    pub(crate) fn waves(&self) -> usize { self.waves }

    /// The return slot the next received wave lands in.
    pub(crate) fn receive_slot(&self) -> usize { self.state.receive_slot.get() }

    fn slot_planes(&self, slot: usize) -> impl Iterator<Item = CuteafdDeviceBuffer> + '_ {
        assert!(slot < self.waves);
        let bytes = self.rows * self.row_bytes;
        self.planes.iter().map(move |plane| CuteafdDeviceBuffer {
            // SAFETY: every slot lies within this rank's contiguous allocation.
            ptr: unsafe { plane.buffer.ptr.cast::<u8>().add(slot * bytes) }.cast(),
            bytes,
            ..plane.buffer
        })
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
    pub(crate) unsafe fn spawn_lane(&self, world: cuteafd_transport::expert::SparkWorld,
        peers: Vec<std::net::SocketAddr>, capacity: u32,
        config: cuteafd_transport::TcpTransportConfig) -> Result<SparkExpertLane> {
        ensure!(peers.len() == self.ranks, "intake ranks differ from the lane's");
        // SAFETY: forwarded from this function's contract.
        unsafe { SparkExpertLane::spawn_world(world, peers, capacity, config, self.landing()) }
    }

    /// Before dispatch: the reused planes' reduce is done and completed
    /// uploads have released their receive slots, even with waves in flight.
    pub(crate) fn before_dispatch(&self) -> Result<()> {
        self.state.before_dispatch(|slot| {
            // SAFETY: the event was recorded on the engine's stream by `consumed`.
            unsafe { self.library.cuda_event_synchronize(self.consumed[slot]) }
        }, || self.release_held())
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
        let slot = self.state.receive_slot.get();
        if self.waves > 1 { self.release_held()?; }
        // A receive can fail or be cancelled after an upload queues but before
        // its completion event is recorded. Drain while every source is held.
        let drain = ReceiveDrain(Some(|| {
            // SAFETY: the caller's stream and the intake's copy stream
            // outlive this receive, as do all retained source owners.
            crate::shared::decode_graph::fatal_drain(unsafe {
                self.terminal_drain_copies().and_then(|()|
                    self.library.cuda_stream_synchronize(stream))
            }, "cancelled Spark intake receive");
        }));
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
                self.upload_staging_slot(slot, receipt.landed, plane_bytes, stream)?;
                if self.waves > 1 {
                    // The next receive rewrites the staging; it waits on this
                    // (`release_held`), not on a slot's consumed event.
                    // SAFETY: live event and stream on this device.
                    unsafe { self.library.cuda_event_record(self.copied, stream)? };
                    self.state.copies_pending.set(true);
                }
                receipt
            }
            false => {
                let target = self.upload_stream(plane_bytes, stream);
                let receipt = transport.receive_wave_owned(wave, |rank, first, payload| {
                    retain_before_enqueue(&self.held, (rank, first, payload), |(rank, first, payload)| {
                        self.queue_upload_slot(slot, *rank, *first, payload, plane_bytes, target)
                    })
                }).await?;
                self.join_uploads(target, stream, receipt.landed)?;
                receipt
            }
        };
        drain.complete();
        self.state.receive_slot.set((slot + 1) % self.waves);
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

    /// Plane pointers of the last received wave, for the compact reducer.
    pub(crate) fn pointers(&self) -> [*const u16; MAX_INTAKE_RANKS] {
        self.slot_pointers(self.last_received())
    }

    /// Plane pointers of return slot `slot`.
    pub(crate) fn slot_pointers(&self, slot: usize) -> [*const u16; MAX_INTAKE_RANKS] {
        let mut pointers = [std::ptr::null::<u16>(); MAX_INTAKE_RANKS];
        for (pointer, plane) in pointers.iter_mut().zip(self.slot_planes(slot)) {
            *pointer = plane.ptr.cast();
        }
        pointers
    }

    fn last_received(&self) -> usize { (self.state.receive_slot.get() + self.waves - 1) % self.waves }

    /// The last received wave's planes, in rank order.
    pub(crate) fn planes(&self) -> impl Iterator<Item = CuteafdDeviceBuffer> + '_ {
        self.slot_planes(self.last_received())
    }

    /// Marks the last received wave's planes read once `stream` reaches this
    /// point (after the reduce).
    pub(crate) fn consumed(&self, stream: *mut c_void) -> Result<()> {
        let slot = self.last_received();
        // SAFETY: both handles are live; the event only orders later dispatches.
        unsafe { self.library.cuda_event_record(self.consumed[slot], stream)? };
        self.state.consumed_pending[slot].set(true);
        Ok(())
    }

    /// Release uploaded sources before dispatch as well as receive: polling
    /// can retain a later wave's final frame before its receive starts, filling
    /// a depth-N ring unless the completed wave returns its slots first.
    fn release_held(&self) -> Result<()> {
        self.state.release_uploads(|| {
            // SAFETY: recorded after the held slots' uploads (`join_uploads`).
            unsafe { self.library.cuda_event_synchronize(self.copied) }
        }, || {
            self.held.lock().map_err(|_| anyhow::anyhow!("intake slots poisoned"))?.clear();
            Ok(())
        })
    }

    fn reset(&self, transport: &mut SparkExperts) -> Result<()> {
        transport.require_drained_reset()?;
        self.state.reset(|| {
            self.terminal_drain_copies()?;
            if self.state.copies_pending.get() {
                // SAFETY: also covers host-mode uploads recorded on the compute stream.
                unsafe { self.library.cuda_event_synchronize(self.copied)?; }
            }
            for (event, pending) in self.consumed.iter().zip(&self.state.consumed_pending) {
                if pending.get() {
                    // SAFETY: recorded after the completed receive's plane consumers.
                    unsafe { self.library.cuda_event_synchronize(*event)?; }
                }
            }
            Ok(())
        }, || transport.reset_drained(), || {
            self.held.lock().map_err(|_| anyhow::anyhow!("intake slots poisoned"))?.clear();
            Ok(())
        })
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
        self.state.copies_pending.set(false);
        for pending in &self.state.consumed_pending { pending.set(false); }
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
        self.queue_upload_slot(0, rank, first, payload, plane_bytes, target)
    }

    fn queue_upload_slot(&self, slot: usize, rank: usize, first: u32, payload: &VerbsHostProtocolV2ResponsePayload,
        plane_bytes: usize, target: *mut c_void) -> Result<()> {
        let bytes = payload.as_ref().len();
        let offset = upload_offset(self.ranks, rank, first, self.row_bytes, bytes, plane_bytes)?;
        let plane = self.slot_planes(slot).nth(rank).expect("validated intake rank");
        let destination = CuteafdDeviceBuffer {
            // SAFETY: the range was checked against the plane above.
            ptr: unsafe { plane.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..plane
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
        self.state.copies_pending.set(true);
        Ok(())
    }

    fn upload_staging(&self, landed: u8, plane_bytes: usize, stream: *mut c_void) -> Result<()> {
        self.upload_staging_slot(0, landed, plane_bytes, stream)
    }

    fn upload_staging_slot(&self, slot: usize, landed: u8, plane_bytes: usize, stream: *mut c_void) -> Result<()> {
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
            unsafe { self.library.copy_host_buffer_h2d_async(self.slot_planes(slot).nth(rank).expect("validated intake rank"), source, plane_bytes, stream)? };
        }
        Ok(())
    }
}

fn upload_offset(ranks: usize, rank: usize, first: u32, row_bytes: usize,
    bytes: usize, plane_bytes: usize) -> Result<usize> {
    ensure!(rank < ranks, "rank {rank} exceeds intake plane count");
    let offset = (first as usize).checked_mul(row_bytes).context("intake row offset overflow")?;
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= plane_bytes),
        "rank {rank} partial rows run past its plane");
    Ok(offset)
}

/// Error/cancellation must drain queued uploads even before event publication.
struct ReceiveDrain<F: FnOnce()>(Option<F>);
impl<F: FnOnce()> ReceiveDrain<F> {
    fn complete(mut self) { self.0.take(); }
}
impl<F: FnOnce()> Drop for ReceiveDrain<F> {
    fn drop(&mut self) {
        if let Some(drain) = self.0.take() { drain(); }
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
mod intake_reset_tests {
    use super::*;

    #[test]
    fn inflight_depth_two_and_three_release_upload_sources_before_replacement_dispatch() -> Result<()> {
        use std::{collections::VecDeque, rc::Rc};
        struct Frame { slot: usize, occupied: Rc<Vec<Cell<bool>>> }
        impl Frame {
            fn new(slot: usize, occupied: &Rc<Vec<Cell<bool>>>) -> Self {
                assert!(!occupied[slot].replace(true), "response slot reused while owned");
                Self { slot, occupied: occupied.clone() }
            }
        }
        impl Drop for Frame {
            fn drop(&mut self) { assert!(self.occupied[self.slot].replace(false)); }
        }

        for depth in [2, 3] {
            let state = IntakeState::new(depth);
            let occupied = Rc::new((0..depth).map(|_| Cell::new(false)).collect());
            for _ in 0..depth { state.before_dispatch(|_| unreachable!(), || Ok(()))?; }
            // One poll can retain both final frames. Only A has reached its upload;
            // B remains in its response channel when A's replacement is dispatched.
            let held = Mutex::new(vec![Frame::new(0, &occupied)]);
            let mut queued: VecDeque<_> = (1..depth).map(|slot| Frame::new(slot, &occupied)).collect();
            state.receive_slot.set(1);
            state.consumed_pending[0].set(true);
            state.copies_pending.set(true);
            let fences = Cell::new(0);
            for sequence in 0..4096 {
                let slot = sequence % depth;
                assert!(occupied.iter().all(Cell::get));
                assert_eq!(state.dispatch_slot.get(), slot);
                let copied = Cell::new(false);
                state.before_dispatch(|reused| {
                    assert_eq!(reused, slot);
                    assert_eq!(held.lock().unwrap()[0].slot, slot);
                    Ok(())
                }, || state.release_uploads(|| {
                    assert!(occupied.iter().all(Cell::get));
                    copied.set(true);
                    fences.set(fences.get() + 1);
                    Ok(())
                }, || {
                    assert!(copied.get(), "receive source released before upload fence");
                    held.lock().unwrap().clear();
                    Ok(())
                }))?;
                assert!(!occupied[slot].get(), "no receive slot for the replacement request");
                assert!(occupied.iter().enumerate().all(|(other, owned)| other == slot || owned.get()),
                    "queued wave lost its response owner");
                assert!(!state.copies_pending.get());
                assert!(!state.consumed_pending[slot].get());
                queued.push_back(Frame::new(slot, &occupied));
                let received = queued.pop_front().unwrap();
                let received_slot = state.receive_slot.get();
                assert_eq!(received.slot, received_slot);
                retain_before_enqueue(&held, received, |_| Ok(()))?;
                state.receive_slot.set((received_slot + 1) % depth);
                state.copies_pending.set(true);
                state.consumed_pending[received_slot].set(true);
            }
            assert_eq!(fences.get(), 4096);
            state.release_uploads(|| Ok(()), || { held.lock().unwrap().clear(); Ok(()) })?;
            drop(queued);
            assert!(occupied.iter().all(|slot| !slot.get()));
        }
        Ok(())
    }

    #[test]
    fn failed_dispatch_fence_keeps_sources_and_dispatch_phase() -> Result<()> {
        for depth in [2, 3] {
            for fail_consumed in [false, true] {
                let state = IntakeState::new(depth);
                state.consumed_pending[0].set(true);
                state.copies_pending.set(true);
                let held = RefCell::new(vec!["pinned source"]);
                let result = state.before_dispatch(|_| {
                    ensure!(!fail_consumed, "injected consumed fence failure");
                    Ok(())
                }, || state.release_uploads(|| anyhow::bail!("injected upload fence failure"), || {
                    held.borrow_mut().clear();
                    Ok(())
                }));
                assert!(result.is_err());
                assert_eq!(state.dispatch_slot.get(), 0);
                assert_eq!(held.borrow().as_slice(), ["pinned source"]);
                assert!(state.copies_pending.get());
                assert_eq!(state.consumed_pending[0].get(), fail_consumed);
                // Terminal cancellation must drain while the failed upload's
                // source is still retained, then reset the wire before release.
                let drained = Cell::new(false);
                state.reset(|| {
                    assert_eq!(held.borrow().len(), 1);
                    drained.set(true);
                    Ok(())
                }, || { assert!(drained.get()); Ok(()) }, || {
                    assert!(drained.get());
                    held.borrow_mut().clear();
                    Ok(())
                })?;
                assert!(held.borrow().is_empty());
                assert!(!state.copies_pending.get());
                state.before_dispatch(|_| unreachable!(), || Ok(()))?;
                assert_eq!(state.dispatch_slot.get(), 1);
            }
        }
        Ok(())
    }

    #[test]
    fn cancelled_unread_slots_reset_in_phase_after_consumers_drain() -> Result<()> {
        for waves in 2..=4 {
            for unread in 1..=waves {
                let state = IntakeState::new(waves);
                // A completed wave leaves retained pinned sources and a consumed event.
                state.dispatch_slot.set(1);
                state.receive_slot.set(1);
                state.consumed_pending[0].set(true);
                state.copies_pending.set(true);
                for _ in 0..unread {
                    state.dispatch_slot.set((state.dispatch_slot.get() + 1) % waves);
                }
                let calls = RefCell::new(Vec::new());
                let held = RefCell::new(vec!["pinned source"]);
                state.reset(|| {
                    assert_eq!(held.borrow().len(), 1);
                    assert!(state.copies_pending.get());
                    assert!(state.consumed_pending[0].get());
                    calls.borrow_mut().push("drain");
                    Ok(())
                }, || {
                    assert_eq!(calls.borrow().as_slice(), ["drain"]);
                    assert_eq!(held.borrow().len(), 1);
                    calls.borrow_mut().push("reset transport");
                    Ok(())
                }, || {
                    assert_eq!(calls.borrow().as_slice(), ["drain", "reset transport"]);
                    held.borrow_mut().clear();
                    calls.borrow_mut().push("release sources");
                    Ok(())
                })?;
                assert_eq!(state.dispatch_slot.get(), 0);
                assert_eq!(state.receive_slot.get(), 0);
                assert!(!state.copies_pending.get());
                assert!(state.consumed_pending.iter().all(|p| !p.get()));
                assert!(held.borrow().is_empty());
                // Reuse several full rotations: sends and receives select the same slots.
                for index in 0..3 * waves {
                    assert_eq!(state.dispatch_slot.get(), index % waves);
                    assert_eq!(state.dispatch_slot.get(), state.receive_slot.get());
                    state.dispatch_slot.set((state.dispatch_slot.get() + 1) % waves);
                    state.receive_slot.set((state.receive_slot.get() + 1) % waves);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn failed_reset_preserves_phase_and_copy_ownership() {
        for fail_at in 0..=2 {
            let state = IntakeState::new(2);
            state.dispatch_slot.set(1);
            state.consumed_pending[0].set(true);
            state.copies_pending.set(true);
            let calls = RefCell::new(Vec::new());
            let call = |at| {
                calls.borrow_mut().push(at);
                ensure!(at != fail_at, "injected reset failure");
                Ok(())
            };
            assert!(state.reset(|| call(0), || call(1), || call(2)).is_err());
            assert_eq!(*calls.borrow(), (0..=fail_at).collect::<Vec<_>>());
            assert_eq!(state.dispatch_slot.get(), 1);
            assert_eq!(state.receive_slot.get(), 0);
            assert!(state.copies_pending.get());
            assert!(state.consumed_pending[0].get());
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
        let finished = ["warm_stream.", "finish()?;"].concat();
        for (family, source, library) in [
            ("glm5_flash", include_str!("../families/glm5_flash/mod.rs"), "self.library.as_ref()"),
            ("mimo_v2", include_str!("../families/mimo_v2/mod.rs"), "&*self.library"),
        ] {
            let created = ["let warm_stream = crate::shared::spark_intake::Warm", "Stream::new(", library, ")?;"].concat();
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

    fn spark_transport_constructor(source: &str) -> Option<usize> {
        ["new", "new_waves"].iter()
            .filter_map(|constructor| source.find(&format!("SparkLink::{constructor}(")))
            .min()
    }

    #[test]
    fn transport_audit_recognizes_single_and_multi_wave_constructors() {
        for constructor in ["new", "new_waves"] {
            let source = format!("let transport = SparkLink::{constructor}(");
            assert_eq!(spark_transport_constructor(&source), Some(16));
        }
        assert_eq!(spark_transport_constructor("SparkLink::new_unchecked("), None);
    }

    /// Every family that builds a Spark transport bootstraps it with a request
    /// at install, so each rank's persistent session is open and its immutable
    /// installed expert grouping is published before anything plans against the
    /// workers. Qwen 4's draft planner reads the grouping that early, and a
    /// transport that had never opened a session answered "worker rank 0 did
    /// not report installed expert grouping" instead of launching.
    ///
    /// The request argument is load-bearing: the request that opens a rank's
    /// session sizes its registered rings, so the bootstrap must hand over the
    /// widest wave the family dispatches (its own start-up warm-up's request).
    #[test]
    fn every_spark_family_bootstraps_its_transport_with_a_wave_request() {
        // The needles are joined here, so this file's own text does not match them.
        let open = [".boot", "strap(&"].concat();
        for (family, source) in [
            ("deepseek_v4", include_str!("../families/deepseek_v4/mod.rs")),
            ("glm5", include_str!("../families/glm5/mod.rs")),
            ("glm5_flash", include_str!("../families/glm5_flash/mod.rs")),
            ("mimo_v2", include_str!("../families/mimo_v2/mod.rs")),
            ("qwen4", include_str!("../families/qwen4/mod.rs")),
            ("step5", include_str!("../families/step5/serve.rs")),
            ("step5 golden", include_str!("../families/step5/golden.rs")),
        ] {
            let transport = spark_transport_constructor(source)
                .unwrap_or_else(|| panic!("{family}: no Spark transport"));
            let bootstrap = source.find(&open)
                .unwrap_or_else(|| panic!("{family}: the Spark transport is never bootstrapped"));
            assert!(bootstrap > transport, "{family}: the bootstrap precedes the transport it opens");
        }
        // The argument-less form is the shape that leaves every session
        // unopened; it must not come back.
        let bare = [".boot", "strap()"].concat();
        for source in [
            include_str!("../families/deepseek_v4/mod.rs"),
            include_str!("../families/deepseek_v4/engine.rs"),
            include_str!("../families/glm5/mod.rs"),
            include_str!("../families/glm5/engine.rs"),
            include_str!("../families/glm5_flash/mod.rs"),
            include_str!("../families/mimo_v2/mod.rs"),
            include_str!("../families/qwen4/mod.rs"),
            include_str!("../families/qwen4/engine.rs"),
        ] {
            assert!(!source.contains(&bare), "a Spark transport is bootstrapped without a wave request");
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
    fn receive_error_and_cancellation_drain_before_source_release() {
        use std::{future::Future, task::Poll};
        for cancel in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let drains = AtomicUsize::new(0);
            let held = Mutex::new(Vec::new());
            let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
            runtime.block_on(async {
                let work = async {
                    let _drain = ReceiveDrain(Some(|| {
                        assert_eq!(drops.load(Ordering::SeqCst), 0);
                        assert_eq!(held.lock().unwrap().len(), 1);
                        drains.fetch_add(1, Ordering::SeqCst);
                    }));
                    retain_before_enqueue(&held, Owner(drops.clone()), |_| Ok(())).unwrap();
                    if cancel { std::future::pending::<()>().await; }
                    anyhow::bail!("injected receive failure before event publication");
                    #[allow(unreachable_code)]
                    Ok::<_, anyhow::Error>(())
                };
                if cancel {
                    let mut work = std::pin::pin!(work);
                    std::future::poll_fn(|cx| {
                        assert!(work.as_mut().poll(cx).is_pending());
                        Poll::Ready(())
                    }).await;
                } else {
                    assert!(work.await.is_err());
                }
            });
            assert_eq!(drains.load(Ordering::SeqCst), 1);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            held.lock().unwrap().clear();
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn successful_receive_defers_drain_to_normal_completion() {
        let drains = Cell::new(0);
        ReceiveDrain(Some(|| drains.set(drains.get() + 1))).complete();
        assert_eq!(drains.get(), 0);
    }

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
            for &consumed in &self.consumed {
                let _ = self.library.cuda_event_synchronize(consumed);
                let _ = self.library.cuda_event_destroy(consumed);
            }
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

fn registered_egress_bytes(request: &cuteafd_transport::ExpertProtocolV2Request) -> Result<usize> {
    let mut geometry = cuteafd_core::expert_geometry();
    geometry.hidden = request.header.hidden_dim;
    ensure!(request.header.row_count > 0 && request.header.route_count % request.header.row_count == 0,
        "egress bootstrap needs a uniform route geometry");
    geometry.topk = request.header.route_count / request.header.row_count;
    geometry.compact_wire_bytes(request.header.row_count,
        request.header.hidden_dtype == cuteafd_transport::ExpertV2Dtype::Bf16)
        .map(|wire| wire.0).context("registered egress geometry overflow")
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
    egress_enabled: bool,
    terminal_library: Option<Arc<NativeLibrary>>,
    terminal_released: bool,
}

impl<'a> SparkLink<'a> {
    /// Connects a [`SparkExperts::new_world`] transport whose `capacity`-row
    /// waves land in an intake of the process-wide [`choose_mode`].
    pub(crate) fn new(library: &'a NativeLibrary, world: cuteafd_transport::expert::SparkWorld, peers: &[std::net::SocketAddr], capacity: u32,
        config: cuteafd_transport::TcpTransportConfig, row_bytes: usize) -> Result<Self> {
        Self::new_waves(library, world, peers, capacity, config, row_bytes, 1, capacity)
    }

    /// A link admitting `waves` outstanding waves of up to `capacity` rows,
    /// each with its own return slot ([`SparkIntake::with_waves`]). Waves are
    /// received in dispatch order, with one GPU landing range per slot.
    /// `capacity` bounds every request on the transport; each return slot
    /// holds `wave_rows` (at most `capacity`), the widest wave in flight.
    pub(crate) fn new_waves(library: &'a NativeLibrary, world: cuteafd_transport::expert::SparkWorld,
        peers: &[std::net::SocketAddr], capacity: u32, config: cuteafd_transport::TcpTransportConfig,
        row_bytes: usize, waves: usize, wave_rows: u32) -> Result<Self> {
        ensure!((1..=capacity).contains(&wave_rows), "Spark return slot rows exceed the transport capacity");
        let mut transport = SparkExperts::new_world(world, peers, capacity, config)?;
        transport.set_max_waves(waves)?;
        let preference = IntakePreference::Probed;
        Self::with_egress_waves(library, transport, row_bytes, preference, true, waves, wave_rows)
    }

    pub(crate) fn waves(&self) -> usize { self.intake.waves() }

    /// Reuse after cancellation only once all wave tickets and unrecorded
    /// compute consumers have drained. Recorded plane/copy consumers are waited
    /// here; reset is refused while any detached wave ticket is still live.
    pub(crate) fn reset_connections(&mut self) -> Result<()> {
        self.intake.reset(&mut self.transport)
    }

    /// Adopts an existing topology-bound or canonical transport without changing
    /// its route contract, rank identities or connection configuration.
    pub(crate) fn from_transport(library: &'a NativeLibrary, transport: SparkExperts,
        row_bytes: usize, preference: IntakePreference) -> Result<Self> {
        Self::with_egress(library, transport, row_bytes, preference, false)
    }

    fn with_egress(library: &'a NativeLibrary, transport: SparkExperts,
        row_bytes: usize, preference: IntakePreference, enable_egress: bool) -> Result<Self> {
        let capacity = transport.capacity();
        Self::with_egress_waves(library, transport, row_bytes, preference, enable_egress, 1, capacity)
    }

    fn with_egress_waves(library: &'a NativeLibrary, mut transport: SparkExperts,
        row_bytes: usize, preference: IntakePreference, enable_egress: bool, waves: usize, wave_rows: u32)
        -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
        let ranks = transport.world_size();
        let capacity = transport.capacity();
        library.v41_compact_reducer()?.require_plane_reduction(ranks as u32)?;
        let (mode, reason) = choose_preferred_mode(library, preference)?.mode_for(ranks * capacity as usize * row_bytes);
        tracing::info!(mode = mode.name(), ranks, capacity, %reason, "Spark transport intake");
        let intake = SparkIntake::with_waves(library, mode, ranks, wave_rows as usize, row_bytes, waves)?;
        // SAFETY: the link drops its transport before its intake, and every
        // dispatch goes through `dispatch`, which calls `before_dispatch`.
        unsafe { intake.attach(&mut transport)? };
        log_transfer_plan(mode, ranks);
        Ok(Self { transport, intake: std::rc::Rc::new(intake), egress_bytes: 0,
            egress_enabled: enable_egress && egress_setting() != Egress::Copied, terminal_library: None, terminal_released: false })
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

    /// Completes every rank's persistent bootstrap without dispatching a wave,
    /// so the workers' immutable installed grouping is readable before the
    /// first request and the rings its widest wave needs are already
    /// registered. Every Spark family calls this once, right after
    /// [`Self::enable_terminal_ownership`] and before anything plans against
    /// the transport; dispatch then reuses the open sessions.
    ///
    /// `request` must be the widest wave the family dispatches (the one its
    /// start-up warm-up sends): it sizes every rank's registered rings, and a
    /// later wider request would have to retire and reconnect them.
    pub(crate) fn bootstrap(&mut self, request: &cuteafd_transport::ExpertProtocolV2Request) -> Result<()> {
        if self.egress_enabled && self.egress_bytes == 0 {
            let bytes = registered_egress_bytes(request)?;
            let _memory_scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
            self.transport.enable_egress(bytes, egress_setting() == Egress::Staggered)?;
            self.egress_bytes = bytes;
            tracing::info!(slot_bytes = bytes, slots = self.waves(),
                registered_bytes = bytes * self.waves(), egress = ?egress_setting(),
                "Spark registered request egress");
        }
        self.transport.bootstrap(request)
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
        let drain = ReceiveDrain(Some(|| {
            // SAFETY: the borrowed inputs outlive this call, including a partial
            // kernel launch or a failed consumption-event publication.
            crate::shared::decode_graph::fatal_drain(unsafe {
                self.intake.terminal_drain_copies().and_then(|()| library.cuda_stream_synchronize(stream))
            }, "failed Spark intake reduction");
        }));
        // SAFETY: the planes hold this wave's `t` rows once `stream` reaches
        // this point (see `receive`); the caller vouches for the rest.
        unsafe {
            library.v41_compact_reducer()?.reduce_planes(self.intake.pointers(), self.world_size() as u32, shared,
                output, t as u32, stream)?;
        }
        self.intake.consumed(stream)?;
        drain.complete();
        Ok(())
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

/// Retains external landing, pinned sources and their loaded library unless
/// the complete QP/compute/copy retirement handshake has succeeded.
struct LaneOwners<T>(Option<T>);
impl<T> LaneOwners<T> {
    fn release(&mut self) { self.0.take(); }
}
impl<T> Drop for LaneOwners<T> {
    fn drop(&mut self) {
        if let Some(owners) = self.0.take() {
            std::mem::forget(owners);
            tracing::error!("retaining terminal lane intake without complete quiescence and consumer drainage");
        }
    }
}

#[cfg(test)]
mod lane_owner_tests {
    use super::*;
    struct Owner(std::rc::Rc<Cell<usize>>);
    impl Drop for Owner { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn quiesce_failure_keeps_fake_dso_and_real_intake_owned() -> Result<()> {
        use crate::shared::experts::rtx::pair::{teardown, Teardown};
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = Arc::new(fixture.load()?);
        let intake = std::rc::Rc::new(SparkIntake::new(&library, IntakeMode::Host, 1, 1, 2)?);
        let intake_weak = std::rc::Rc::downgrade(&intake);
        let library_weak = Arc::downgrade(&library);
        let terminal = RefCell::new(LaneOwners(Some((intake.clone(), library.clone()))));
        let copies = Cell::new(false);
        let released = Cell::new(false);
        let result = teardown(&mut Teardown {
            abort: || Ok(()),
            ranks: || Ok(()),
            links: || anyhow::bail!("injected ordinary-lane quiesce failure"),
            copies: || { copies.set(true); intake.terminal_drain_copies() },
            graphs: || { released.set(true); terminal.borrow_mut().release(); Ok(()) },
            finish: |drained: bool| assert!(!drained),
        });
        assert!(result.unwrap_err().to_string().contains("link quiescence"));
        assert!(!copies.get() && !released.get());
        assert!(terminal.borrow().0.is_some());
        drop(intake);
        drop(terminal);
        // The same guard used by SparkLane retains the real allocations after
        // the external intake owner disappears, and keeps the DSO loaded.
        assert!(intake_weak.upgrade().is_some());
        drop(intake_weak);
        drop(library);
        assert!(library_weak.upgrade().is_some());
        assert!(fixture.resident()?);
        let events = fixture.events()?;
        assert!(!events.contains('d') && !events.contains('F') && !events.contains('U'));
        Ok(())
    }

    #[test]
    fn unproved_lane_release_retains_intake_and_loaded_library() {
        let drops = std::rc::Rc::new(Cell::new(0));
        drop(LaneOwners(Some((Owner(drops.clone()), Owner(drops.clone())))));
        assert_eq!(drops.get(), 0);
        let mut released = LaneOwners(Some((Owner(drops.clone()), Owner(drops.clone()))));
        released.release();
        drop(released);
        assert_eq!(drops.get(), 2);
    }
}

/// A prefill lane and its external intake. Keep this owner whole: joining a
/// thread alone is not proof that RDMA writes or queued copies have stopped.
pub(crate) struct SparkLane<'a> {
    pub(crate) lane: SparkExpertLane,
    pub(crate) intake: std::rc::Rc<SparkIntake<'a>>,
    terminal: LaneOwners<(std::rc::Rc<SparkIntake<'a>>, Arc<NativeLibrary>)>,
}

impl<'a> SparkLane<'a> {
    pub(crate) fn new(library: &'a Arc<NativeLibrary>, world: cuteafd_transport::expert::SparkWorld, peers: Vec<std::net::SocketAddr>, capacity: u32,
        config: cuteafd_transport::TcpTransportConfig, row_bytes: usize) -> Result<Self> {
        library.rdma_rc_endpoint_quiesce_available()?;
        let mode = transport_mode(library, peers.len(), capacity as usize, row_bytes)?;
        let intake = std::rc::Rc::new(SparkIntake::new(library, mode, peers.len(), capacity as usize, row_bytes)?);
        let terminal = LaneOwners(Some((intake.clone(), library.clone())));
        // SAFETY: the terminal guard retains the intake and its actual loaded
        // library until checked QP quiescence and all consumers are drained.
        let lane = unsafe { intake.spawn_lane(world, peers, capacity, config)? };
        Ok(Self { lane, intake, terminal })
    }

    pub(crate) fn terminal_quiesce(&mut self) -> Result<()> { self.lane.terminal_quiesce() }

    /// Called only after the submission's compute streams and intake copies
    /// drain successfully, never merely because the transport thread joined.
    pub(crate) fn terminal_release(&mut self) -> Result<()> {
        self.lane.terminal_release()?;
        self.intake.terminal_release_payloads()?;
        self.terminal.release();
        Ok(())
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
    pub(crate) fn terminal_drain_copies(&self) -> Result<()> {
        self.intake.terminal_drain_copies()
    }

    /// Connects the proxy's transport (`capacity` rows of `topk` routes and
    /// `wire_row_bytes` per wave; partial rows of `row_bytes`) on `device`.
    /// Needs GPU landing (the probed intake mode must be `gpu`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(library: &'a NativeLibrary, device: i32, world: cuteafd_transport::expert::SparkWorld, peers: &[std::net::SocketAddr],
        capacity: usize, topk: usize, wire_row_bytes: usize, row_bytes: usize,
        config: cuteafd_transport::TcpTransportConfig, warm: Option<cuteafd_transport::ExpertProtocolV2Request>,
        build: cuteafd_transport::expert::DeviceBuild) -> Result<Self> {
        use cuteafd_transport::expert::device_mailbox;
        let choice = choose_mode(library)?;
        ensure!(device_choice_available(&choice),
            "the device Spark exchange needs GPU landing, but the intake is {} ({})", choice.mode.name(), choice.reason);
        library.v41_compact_reducer()?.require_plane_reduction(peers.len() as u32)?;
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
            cuteafd_transport::expert::SparkDeviceLane::spawn(world, peers.to_vec(), capacity as u32,
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
    fn fabric_probe_reports_missing_library_for_each_bounded_repeat() {
        let reports = fabric_probe(Some(std::path::Path::new("/nonexistent-landing-probe-library.so")), 0, true, 3);
        assert_eq!(reports.len(), 3);
        for report in reports {
            assert!(report.probe.is_none());
            assert!(report.error.is_some());
            assert!(report.summary().contains("intake auto: pinned"));
        }
    }

    #[test]
    fn registered_egress_matches_widest_bootstrap_request_geometry() -> Result<()> {
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor,
            ExpertProtocolV2RouteEntry, ExpertV2SourceKind, ExpertV2Dtype};
        for rows in [64u32, 1024, 2048, 4096] {
            for dtype in [ExpertV2Dtype::Bf16, ExpertV2Dtype::Fp8E4m3Ue8m0K32] {
                let geometry = cuteafd_core::ExpertGeometry::MIMO_V26_PRO;
                let hidden_bytes = if dtype == ExpertV2Dtype::Bf16 { geometry.hidden * 2 }
                    else { geometry.hidden + geometry.hidden / 32 };
                let request = ExpertProtocolV2Request::new(1, 17, 0, geometry.hidden, dtype,
                    (0..rows).map(|row| ExpertProtocolV2RowDescriptor {
                        row_id: u64::from(row), source_kind: ExpertV2SourceKind::Prefill,
                        source_request_id: 1, token_position: u64::from(row),
                        route_offset: row * geometry.topk, route_count: geometry.topk,
                    }).collect(),
                    (0..rows).flat_map(|row| (0..geometry.topk).map(move |expert| ExpertProtocolV2RouteEntry {
                        row_index: row, expert_id: expert, gate_weight: 0.0,
                    })).collect(), vec![0; (rows * hidden_bytes) as usize])?;
                let expected = geometry.compact_wire_bytes(rows, dtype == ExpertV2Dtype::Bf16).unwrap().0;
                assert_eq!(registered_egress_bytes(&request)?, expected);
                assert_eq!(request.encode()?.len(), expected);
            }
        }
        Ok(())
    }

    #[test]
    fn upload_bounds_reject_missing_planes_and_overflow_before_pointer_arithmetic() {
        for rank in [0, 1, 2, 4, usize::MAX] {
            assert!(upload_offset(0, rank, 0, 10240, 10240, 10240).is_err());
        }
        assert!(upload_offset(2, 2, 0, 10240, 10240, 10240).is_err());
        assert!(upload_offset(2, 0, 1, 10240, 10240, 10240).is_err());
        assert!(upload_offset(2, 0, 1, usize::MAX, 1, usize::MAX).is_err());
        assert!(upload_offset(2, 0, 2, usize::MAX, 0, usize::MAX).is_err());
        assert_eq!(upload_offset(2, 1, 3, 10240, 20480, 51200).unwrap(), 30720);
    }

    #[test]
    fn pageable_uploads_match_rank_ordered_reduction_and_reuse_after_drain() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_PLANE_UPLOAD_LIBRARY") else {
            eprintln!("skip GPU intake upload test: CUTEAFD_PLANE_UPLOAD_LIBRARY unset");
            return Ok(());
        };
        // SAFETY: the opt-in path names the built native library for this GPU.
        let library = unsafe { NativeLibrary::load(path)? };
        for ranks in [2, 3, 4, 6] {
            let intake = SparkIntake::new(&library, IntakeMode::Pinned, ranks, 4096, 10240)?;
            let stream = WarmStream::new(&library)?;
            let shared = DeviceAllocation::new(&library, 4096 * 10240)?;
            let output = DeviceAllocation::new(&library, 4096 * 10240)?;
            let reducer = library.v41_compact_reducer()?;
            for rows in [1, 6, 80, 81, 256, 1024, 4096, 6] {
                intake.before_dispatch()?;
                let bytes = rows * 10240;
                library.copy_h2d(shared.buffer, &0x3f00u16.to_ne_bytes().repeat(bytes / 2))?;
                let payloads: Vec<Vec<u8>> = (0..ranks).map(|rank| (0..bytes / 2).flat_map(|i| {
                    let value = rank as f32 + 1.0 + (i % 31) as f32 / 32.0;
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                }).collect()).collect();
                for (plane, payload) in intake.planes.iter().zip(&payloads) {
                    library.copy_h2d(plane.buffer, payload)?;
                }
                // SAFETY: all initialized planes and output hold the live rows.
                unsafe {
                    match ranks {
                        2 => reducer.reduce_tp2(std::array::from_fn(|r| intake.planes[r].buffer.ptr.cast::<u16>().cast_const()),
                            shared.buffer.ptr.cast(), output.buffer.ptr.cast(), rows as u32, stream.raw())?,
                        4 => reducer.reduce(std::array::from_fn(|r| intake.planes[r].buffer.ptr.cast::<u16>().cast_const()),
                            shared.buffer.ptr.cast(), output.buffer.ptr.cast(), rows as u32, stream.raw())?,
                        _ => reducer.reduce_planes(intake.pointers(), ranks as u32,
                            shared.buffer.ptr.cast(), output.buffer.ptr.cast(), rows as u32, stream.raw())?,
                    }
                    library.cuda_stream_synchronize(stream.raw())?;
                }
                let mut expected = vec![0; bytes];
                library.copy_d2h(&mut expected, CuteafdDeviceBuffer { bytes, ..output.buffer })?;
                for plane in &intake.planes { library.copy_h2d(plane.buffer, &vec![0; bytes])?; }
                let drain = ReceiveDrain(Some(|| {
                    crate::shared::decode_graph::fatal_drain(intake.terminal_drain_copies()
                        .and_then(|()| unsafe { library.cuda_stream_synchronize(stream.raw()) }),
                        "intake upload test drain");
                }));
                let target = intake.upload_stream(bytes, stream.raw());
                for first in (0..rows).step_by(3) {
                    let end = (first + 3).min(rows);
                    for rank in (0..ranks).rev() {
                        let payload = VerbsHostProtocolV2ResponsePayload::from_owned(
                            payloads[rank][first * 10240..end * 10240].to_vec());
                        retain_before_enqueue(&intake.held, (rank, first as u32, payload), |(rank, first, payload)| {
                            intake.queue_upload(*rank, *first, payload, bytes, target)
                        })?;
                    }
                }
                intake.join_uploads(target, stream.raw(), 0)?;
                // SAFETY: upload completion precedes the reduction on this stream.
                unsafe {
                    reducer.reduce_planes(intake.pointers(), ranks as u32, shared.buffer.ptr.cast(),
                        output.buffer.ptr.cast(), rows as u32, stream.raw())?;
                }
                intake.consumed(stream.raw())?;
                drop(drain);
                let mut actual = vec![0; bytes];
                library.copy_d2h(&mut actual, CuteafdDeviceBuffer { bytes, ..output.buffer })?;
                assert_eq!(actual, expected, "ranks={ranks} rows={rows}");
                intake.before_dispatch()?;
                assert!(intake.held.lock().unwrap().is_empty());
            }
            stream.finish()?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires four idle live Spark workers and CUDA native library"]
    fn retained_roce_uploads_drain_before_recycling_live() -> Result<()> {
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor,
            ExpertProtocolV2RouteEntry, ExpertV2SourceKind, ExpertV2Dtype, TcpTransportConfig};
        // SAFETY: the opt-in path names the matching trusted native library.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_PLANE_UPLOAD_LIBRARY")?)? };
        let peers = std::env::var("CUTEAFD_LIVE_ROCE_PEERS")?.split(',')
            .map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        ensure!(peers.len() == 4, "four peers required");
        let mut transport = SparkExperts::new_world(cuteafd_transport::expert::SparkWorld::canonical(peers.len())?, &peers, 4096, TcpTransportConfig { timing: false, timeout: std::time::Duration::from_secs(30),
                max_frame_bytes: 64 << 20 })?;
        let intake = SparkIntake::new(&library, IntakeMode::Pinned, 4, 4096, 10240)?;
        // SAFETY: transport drops first and dispatch follows before_dispatch.
        unsafe { intake.attach(&mut transport)? };
        let stream = WarmStream::new(&library)?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        runtime.block_on(async {
            let mut request_id = 900_000;
            for rows in [1u32, 6, 80, 81, 256, 1024, 4096, 6] {
                let mut hidden = Vec::with_capacity(rows as usize * 5280);
                for row in 0..rows {
                    hidden.extend((0..5120).map(|i| 0x30 + ((i + row) % 8) as u8));
                    hidden.extend([120; 160]);
                }
                let mut request = ExpertProtocolV2Request::new(request_id, 17, 39, 5120,
                    ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                    (0..rows).map(|r| ExpertProtocolV2RowDescriptor {
                        row_id: r as u64, source_kind: ExpertV2SourceKind::Prefill,
                        source_request_id: request_id, token_position: r as u64,
                        route_offset: r * 6, route_count: 6,
                    }).collect(),
                    (0..rows).flat_map(|r| (0..6).map(move |j| ExpertProtocolV2RouteEntry {
                        row_index: r, expert_id: (r * 7 + j) % 384, gate_weight: 1.0 / 6.0,
                    })).collect(), hidden)?;
                request.header.flags = cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
                for fail in [false, true, false] {
                    request_id += 1;
                    request.header.request_id = request_id;
                    intake.before_dispatch()?;
                    let plane_bytes = rows as usize * 10240;
                    let target = intake.upload_stream(plane_bytes, stream.raw());
                    let mut expected = Vec::new();
                    {
                        let _drain = ReceiveDrain(Some(|| {
                            crate::shared::decode_graph::fatal_drain(intake.terminal_drain_copies()
                                .and_then(|()| unsafe { library.cuda_stream_synchronize(stream.raw()) }),
                                "live retained intake test drain");
                        }));
                        let wave = transport.dispatch_wave(&request)?;
                        let result = transport.receive_wave_owned(wave, |rank, first, payload| {
                            ensure!(payload.pinned_host_buffer().is_some() && payload.retains_receive_slot(),
                                "live response is not a retained pinned receive slot");
                            expected.push((rank, first, payload.as_ref().to_vec()));
                            retain_before_enqueue(&intake.held, (rank, first, payload), |(rank, first, payload)| {
                                intake.queue_upload(*rank, *first, payload, plane_bytes, target)
                            })?;
                            if fail { anyhow::bail!("injected failure after pinned upload enqueue"); }
                            Ok(())
                        }).await;
                        assert_eq!(result.is_err(), fail);
                        if fail { transport.reset_connections(); }
                    }
                    assert!(!expected.is_empty());
                    if !fail { assert_eq!(expected.iter().map(|(_, _, b)| b.len()).sum::<usize>(),
                        plane_bytes * 4); }
                    for (rank, first, bytes) in expected {
                        let offset = upload_offset(4, rank, first, 10240, bytes.len(), plane_bytes)?;
                        let plane = intake.planes[rank].buffer;
                        // SAFETY: the checked range belongs to this live plane, whose upload drained.
                        let source = CuteafdDeviceBuffer { ptr: unsafe { plane.ptr.cast::<u8>().add(offset).cast() },
                            bytes: bytes.len(), ..plane };
                        let mut actual = vec![0; bytes.len()];
                        library.copy_d2h(&mut actual, source)?;
                        assert_eq!(actual, bytes, "rows={rows} fail={fail} rank={rank}");
                    }
                    intake.before_dispatch()?;
                    assert!(intake.held.lock().unwrap().is_empty());
                }
            }
            Ok::<_, anyhow::Error>(())
        })?;
        stream.finish()
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
    fn pinned_preference_does_not_change_probed_auto() {
        let mut measured = probe(19.2);
        measured.host_gbps = 55.0;
        let h2d = Some(55.0);
        assert_eq!(pinned_path_gbps(measured.host_gbps, h2d.unwrap()), 27.5);
        let (mode, reason) = resolve_landing_probe("auto", &Ok(measured.clone()), h2d);
        assert_eq!(mode, IntakeMode::Gpu);
        let generic = IntakeChoice { setting: "auto".into(), mode, reason,
            probe: Some(measured), h2d_gbps: h2d };
        assert!(device_choice_available(&generic));
        let v41 = preferred_choice(generic, IntakePreference::Pinned);
        assert_eq!(v41.mode, IntakeMode::Pinned);
        assert!(!device_choice_available(&v41));
        assert_eq!(v41.mode_for(4096 * 10240).0, IntakeMode::Pinned);
    }

    #[test]
    fn pinned_preference_preserves_explicit_gpu_and_capability_fallback() {
        let measured = probe(19.2);
        for supported in [true, false] {
            let result = if supported { Ok(measured.clone()) }
                else { Err(anyhow::anyhow!("forced unavailable")) };
            let (mode, reason) = resolve_landing_probe("gpu", &result, None);
            let choice = preferred_choice(IntakeChoice { setting: "gpu".into(), mode, reason,
                probe: result.ok(), h2d_gbps: None }, IntakePreference::Pinned);
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
