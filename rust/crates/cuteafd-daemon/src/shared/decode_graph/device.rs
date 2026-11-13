//! Device-wide ARC policy. Native owners stay in GraphBank: this controller
//! marks victims; banks with persistent captured storage destroy them without
//! synchronization. Per-graph storage requires explicit drained retirement.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Cycle { Decode, Verify, Prefill, Wave, Other }

fn serialize_cycles<S: serde::Serializer, T: serde::Serialize>(cycles: &HashMap<Cycle, T>, serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(cycles.len()))?;
    for cycle in [Cycle::Decode, Cycle::Verify, Cycle::Prefill, Cycle::Wave, Cycle::Other] {
        if let Some(value) = cycles.get(&cycle) { map.serialize_entry(&cycle, value)?; }
    }
    map.end()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Preference { Adaptive, Capture, Eager }

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Basis {
    Bytes,
    Nodes { node_rate: u64, graph_rate: u64 },
}

impl Basis {
    fn pool_bytes(self, nodes: u64, execs: u64) -> u64 {
        match self {
            Self::Bytes => nodes,
            Self::Nodes { node_rate, graph_rate } => {
                let pack = |bytes: u64| bytes.div_ceil(2 << 20).saturating_mul(2 << 20);
                pack(nodes.saturating_mul(node_rate)).saturating_add(pack(execs.saturating_mul(graph_rate)))
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Site {
    pub name: &'static str,
    pub cycle: Cycle,
    pub preference: Preference,
    /// Family estimate used only until the device has a measured graph.
    pub bootstrap: u64,
    pub basis: Basis,
    /// Declared by the caller: kernel arguments can contain peer pointers that
    /// CUDA's generic graph inspection cannot safely identify.
    pub spans_devices: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Identity {
    pub bank: u64,
    pub site: &'static str,
    #[serde(rename = "key_id")]
    pub key: u64,
    #[serde(serialize_with = "serialize_shape")]
    pub shape: Arc<str>,
}

fn serialize_shape<S: serde::Serializer>(shape: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(shape)
}

// The bank interns typed keys with exact equality. Display text is diagnostic,
// never an identity: distinct keys may deliberately have identical Debug output.
impl PartialEq for Identity {
    fn eq(&self, other: &Self) -> bool {
        (self.bank, self.site, self.key) == (other.bank, other.site, other.key)
    }
}
impl Eq for Identity {}
impl std::hash::Hash for Identity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&(self.bank, self.site, self.key), state);
    }
}

#[derive(Default, Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Counters {
    pub hits: u64,
    pub misses: u64,
    pub captures: u64,
    pub eager_runs: u64,
    pub evictions: u64,
    pub physical_floor_skips: u64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub device: i32,
    pub reserve: u64,
    pub physical_floor: u64,
    pub reserve_policy: Option<cuteafd_loader::placement::graph_reserve::Resolved>,
    pub charged: u64,
    /// Current untracked level relative to the attach seed. Unlike `charged`
    /// (a monotone high-water that admission and eviction depend on), this can
    /// fall, so a transient peak that reverses releases its credit.
    pub level: Option<u64>,
    pub pinned: u64,
    pub pending_capture: u64,
    #[serde(serialize_with = "serialize_cycles")]
    pub cycles: HashMap<Cycle, Counters>,
    pub pools: Pools,
    pub basis: Basis,
    pub sampled_at_ms: u64,
    pub telemetry_bytes: usize,
    pub top_hits: Vec<TraceRow>,
    pub top_churn: Vec<TraceRow>,
}

/// Release observations are deliberately independent of cudaMemGetInfo. An
/// allocator experiment can supply reusable arena bytes instead of free bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReleaseMeasurement { pub released: u64 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum List { Recent, Frequent, RecentGhost, FrequentGhost }
impl List { fn index(self) -> usize { self as usize } }

#[derive(Default)]
struct Queue { first: Option<u64>, last: Option<u64>, bytes: u64, len: usize }

const OOM_RETRY_MISSES: u16 = 256;
const OOM_LEVEL_DROP: u64 = 2 << 20;

struct OomBackoff {
    remaining: u16,
    level: Option<u64>,
    recovery_generation: u64,
}

struct Record {
    identity: Identity,
    cycle: Cycle,
    bytes: u64,
    list: Option<List>,
    previous: Option<u64>,
    next: Option<u64>,
    pinned: bool,
    pending_destroy: bool,
    pending_capture: bool,
    frequent: bool,
    nodes: u64,
    node_rate: u64,
    graph_rate: u64,
    parallel_width: usize,
    spans_devices: bool,
    oom: Option<OomBackoff>,
    oom_warned: bool,
    permanently_eager: bool,
    stats: Box<KeyStats>,
}

const HIT_INTERVAL_MS: u64 = 10_000;
const HIT_INTERVALS: usize = 60;

/// Owned by a resident or bounded ARC ghost; forgetting the key frees its history.
#[derive(Clone)]
struct KeyStats {
    count: u64,
    hits: u64,
    misses: u64,
    captures: u64,
    evictions: u64,
    recaptures: u64,
    ghost_hits: u64,
    cycles: [u64; 5],
    first_seen_ms: u64,
    last_hit_ms: Option<u64>,
    hit_intervals: [u64; HIT_INTERVALS],
    interval: u64,
    capture_bytes: Option<u64>,
    native_nodes: Option<u64>,
    declared_capture_streams: usize,
    max_parallel_width: usize,
    spans_devices: bool,
    evicted_since_capture: bool,
}

impl KeyStats {
    fn new(now: u64) -> Box<Self> {
        Box::new(Self { count: 0, hits: 0, misses: 0, captures: 0, evictions: 0,
            recaptures: 0, ghost_hits: 0, cycles: [0; 5], first_seen_ms: now,
            last_hit_ms: None, hit_intervals: [0; HIT_INTERVALS], interval: now / HIT_INTERVAL_MS,
            capture_bytes: None, native_nodes: None, declared_capture_streams: 1, max_parallel_width: 0,
            spans_devices: false, evicted_since_capture: false })
    }

    fn event(&mut self, cycle: Cycle, count: u64) {
        self.count += count;
        self.cycles[cycle as usize] += count;
    }

    fn hit(&mut self, cycle: Cycle, count: u64, now: u64) {
        if count == 0 { return }
        self.event(cycle, count);
        self.hits += count;
        self.last_hit_ms = Some(now);
        // Startup Other hits must not look like sustained request demand.
        if !matches!(cycle, Cycle::Decode | Cycle::Verify) { return }
        let interval = now / HIT_INTERVAL_MS;
        if interval.saturating_sub(self.interval) >= HIT_INTERVALS as u64 {
            self.hit_intervals.fill(0);
        } else {
            for elapsed in self.interval + 1..=interval {
                self.hit_intervals[elapsed as usize % HIT_INTERVALS] = 0;
            }
        }
        self.interval = interval;
        self.hit_intervals[interval as usize % HIT_INTERVALS] += count;
    }
}

#[derive(Default, Clone)]
struct SizeClass { total: u64, samples: u64, largest: u64 }

/// Optional #1612 rates and node count. Estimates decide capture permission;
/// only measured arena growth contributes to the physical charge.
#[derive(Clone)]
pub(crate) struct Estimate {
    pub bytes: u64,
    pub nodes: Option<u64>,
    pub node_rate: u64,
    pub graph_rate: u64,
    pub parallel_width: usize,
}

#[derive(Default, Clone, Debug, serde::Serialize)]
pub(crate) struct Pools {
    pub live_nodes: u64,
    pub node_high_water: u64,
    pub live_execs: u64,
    pub exec_high_water: u64,
}

fn may_capture(preference: Preference, estimate: u64, headroom: u64, reserve: u64,
    arena: u64, pending_growth: u64, pending: u64, reuse: bool) -> bool {
    if preference == Preference::Eager || estimate.saturating_add(headroom) > reserve { return false }
    reserve.saturating_sub(arena.saturating_add(pending_growth)) >= estimate.saturating_add(headroom)
        || (reuse && pending == 0)
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct TraceRow {
    device: i32,
    #[serde(flatten)]
    identity: Identity,
    cycle: Cycle,
    count: u64,
    #[serde(serialize_with = "serialize_cycles")]
    cycles: HashMap<Cycle, u64>,
    first_seen_ms: u64,
    hits: u64,
    misses: u64,
    captures: u64,
    evictions: u64,
    recaptures: u64,
    ghost_hits: u64,
    last_hit_ms: Option<u64>,
    hit_histogram: HitHistogram,
    capture_bytes: Option<u64>,
    nodes: Option<u64>,
    executables: u64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    permanently_eager: bool,
    basis: Basis,
    permanent_bytes: u64,
    pinned: bool,
    declared_capture_streams: usize,
    max_parallel_width: usize,
    spans_devices: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
struct HitHistogram {
    rolling: bool,
    interval_ms: u64,
    start_ms: u64,
    counts: Vec<u64>,
}

pub(crate) enum Miss { Capture(u64), Eager }
pub(crate) type SharedCache = Arc<Mutex<DeviceCache>>;
// Banks and capture tickets own the controller, not the process registry.
// After teardown the next serving lifetime must sample a fresh baseline.
static DEVICES: OnceLock<Mutex<HashMap<i32, std::sync::Weak<Mutex<DeviceCache>>>>> = OnceLock::new();
static PHYSICAL_FLOOR: OnceLock<Option<String>> = OnceLock::new();

pub(crate) fn set_physical_floor(text: Option<String>) -> anyhow::Result<()> {
    cuteafd_loader::placement::graph_reserve::physical_floor(text.as_deref(), 0, 0)?;
    install_physical_floor(&PHYSICAL_FLOOR, text)
}

fn install_physical_floor(policy: &OnceLock<Option<String>>, text: Option<String>) -> anyhow::Result<()> {
    let installed = policy.get_or_init(|| text.clone());
    anyhow::ensure!(*installed == text, "graph physical floor was already installed with a different value");
    Ok(())
}

pub(crate) fn device_cache(library: &cuteafd_ffi::NativeLibrary, device: i32,
    reserve: u64, basis: Basis) -> anyhow::Result<SharedCache> {
    // Policy fixtures use synthetic ordinals, never valid CUDA device IDs.
    #[cfg(test)]
    if device < 0 { return device_cache_sampled(device, reserve, basis, || Ok((0, 0, 0, 0, 0))) }
    let cache = device_cache_sampled(device, reserve, basis, || memory_sample(library, device))?;
    let mut controller = cache.lock().unwrap_or_else(|p| p.into_inner());
    if controller.memory_query.is_none() { controller.memory_query = Some(library.physical_memory_query()?); }
    drop(controller);
    Ok(cache)
}

/// Tracked, rounding-tail and measured-unclaimed bytes for one physical device.
fn ledger_sample(device: i32) -> (u64, u64, u64) {
    let ledger = cuteafd_ffi::memory_ledger::snapshot();
    let space = cuteafd_ffi::memory_ledger::Space::Device;
    let tracked = ledger.total(space, device) as u64;
    let rounding = ledger.granularity.get(&(space, device))
        .map_or(0, |granularity| granularity.tail) as u64;
    let unclaimed = ledger.unclaimed(space, device) as u64;
    (tracked, rounding, unclaimed)
}

/// `(free, total, tracked, rounding, unclaimed)` for one physical device.
fn memory_sample(library: &cuteafd_ffi::NativeLibrary, device: i32) -> anyhow::Result<(u64, u64, u64, u64, u64)> {
    let (free, total) = crate::shared::memory::device::Device { library, id: device }
        .run(|| library.cuda_physical_memory_info())?;
    let (tracked, rounding, unclaimed) = ledger_sample(device);
    Ok((free as u64, total as u64, tracked, rounding, unclaimed))
}

pub(super) fn device_cache_sampled(device: i32, reserve: u64, basis: Basis,
    sample: impl FnOnce() -> anyhow::Result<(u64, u64, u64, u64, u64)>) -> anyhow::Result<SharedCache> {
    let mut devices = DEVICES.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner());
    if let Some(cache) = devices.get(&device).and_then(std::sync::Weak::upgrade) {
        anyhow::ensure!(cache.lock().unwrap_or_else(|p| p.into_inner()).reserve == reserve,
            "graph banks on device {device} disagree on reserve");
        anyhow::ensure!(cache.lock().unwrap_or_else(|p| p.into_inner()).basis == basis,
            "graph banks on device {device} disagree on estimator basis");
        return Ok(cache);
    }
    let mut controller = DeviceCache::new(device, reserve);
    // Synthetic ordinals in CPU policy fixtures have no physical GPU.
    if device >= 0 {
        let padding = cuteafd_core::memory_policy::MemoryPolicy::current().memory_padding
            .unwrap_or(cuteafd_core::memory_policy::DEFAULT_MEMORY_PADDING);
        controller.physical_floor = cuteafd_loader::placement::graph_reserve::physical_floor(
            PHYSICAL_FLOOR.get().and_then(|text| text.as_deref()), device as usize, padding)?;
    }
    controller.basis = basis;
    // Seed before the first attached bank can capture. Reattachment must not
    // consume growth from an earlier bank or overwrite an explicit baseline.
    let (free, total, tracked, rounding, unclaimed) = sample()?;
    controller.level = Some(controller.observer.observe(free, total, tracked, rounding, unclaimed).1);
    controller.sample_physical(free, tracked);
    let cache = Arc::new(Mutex::new(controller));
    devices.insert(device, Arc::downgrade(&cache));
    Ok(cache)
}

/// Attach the admitted charge's provenance after a family has bound its banks.
pub(crate) fn set_reserve_policy(device: i32, policy: cuteafd_loader::placement::graph_reserve::Resolved) -> anyhow::Result<()> {
    let devices = DEVICES.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner());
    let cache = devices.get(&device).and_then(std::sync::Weak::upgrade)
        .ok_or_else(|| anyhow::anyhow!("no graph cache on device {device}"))?;
    let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
    anyhow::ensure!(cache.reserve == policy.bytes, "graph reserve policy disagrees with admission on device {device}");
    cache.reserve_policy = Some(policy);
    Ok(())
}

pub(crate) fn snapshots() -> Vec<Snapshot> {
    let caches: Vec<_> = DEVICES.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner())
        .values().filter_map(std::sync::Weak::upgrade).collect();
    let mut rows: Vec<_> = caches.iter().map(|cache| cache.lock().unwrap_or_else(|p| p.into_inner()).snapshot()).collect();
    rows.sort_by_key(|row| row.device);
    rows
}

/// Incremental untracked observation. Tracked lazy workspace growth is removed
/// at every boundary, not merely when the observer is seeded. `high_water`
/// drives the admitted charge (`charged`) and must keep covering a transient
/// peak so the next capture's spike cannot exceed the reserve; `level` is the
/// current seed-relative residual used by reconciliation, so a peak that
/// reverses releases its credit.
#[derive(Default)]
struct GrowthObserver {
    baseline: Option<u64>,
    high_water: Option<u64>,
    level: Option<u64>,
}
impl GrowthObserver {
    /// Returns `(high-water growth, current level)`; the first sample seeds
    /// both from the attach residual and reports no growth.
    fn observe(&mut self, free: u64, total: u64, tracked: u64, rounding: u64, unclaimed: u64) -> (u64, u64) {
        let residual = untracked(free, total, tracked, rounding, unclaimed);
        let baseline = *self.baseline.get_or_insert(residual);
        let level = residual.saturating_sub(baseline);
        let growth = self.high_water.map_or(0, |before| residual.saturating_sub(before));
        self.high_water = Some(self.high_water.unwrap_or(residual).max(residual));
        self.level = Some(level);
        (growth, level)
    }

    /// The seed-relative residual at a settled sample without touching the
    /// high-water. Reconciliation uses this so its figure is the device now,
    /// not the last capture boundary, which can catch a transient peak.
    fn current(&self, free: u64, total: u64, tracked: u64, rounding: u64, unclaimed: u64) -> u64 {
        let residual = untracked(free, total, tracked, rounding, unclaimed);
        residual.saturating_sub(self.baseline.unwrap_or(residual))
    }
}

/// Physical bytes no ledger owner claims. `used - tracked` also contains the
/// 2 MiB granule padding of every tracked request, so remove the ledger's
/// known rounding tail: a page-aligned owner's padding is its own overhead, not
/// graph growth, and must never fund the graph reserve. Measured owners with no
/// allocation request behind them (mapped pinned host page tables) come out too;
/// the attach seed is itself such a residual, so an owner that predates the
/// controller cancels in it.
fn untracked(free: u64, total: u64, tracked: u64, rounding: u64, unclaimed: u64) -> u64 {
    total.saturating_sub(free).saturating_sub(tracked).saturating_sub(rounding).saturating_sub(unclaimed)
}

pub(crate) type GraphVisitor<'a> = dyn FnMut(i32, &'static str, bool,
    &mut dyn super::CacheAttached) -> anyhow::Result<()> + 'a;

fn existing_cache(device: i32) -> Option<SharedCache> {
    DEVICES.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner())
        .get(&device).and_then(std::sync::Weak::upgrade)
}

/// Reuse a periodic report's physical sample; older asynchronous reports may
/// not replace a newer boundary sample. No CUDA query or graph-level mutation.
pub(crate) fn monitor_physical(device: i32, sampled_at: Instant, free: u64, tracked: u64) {
    if let Some(cache) = existing_cache(device) {
        cache.lock().unwrap_or_else(|p| p.into_inner()).monitor_physical(sampled_at, free, tracked);
    }
}

/// Observe once per physical device, even when several banks or lanes completed.
/// Call only at a completed submission boundary, never within a peer segment.
pub(crate) fn observe_device(library: &cuteafd_ffi::NativeLibrary, device: i32) -> anyhow::Result<()> {
    observe_device_sampled(device, || memory_sample(library, device))
}

fn observe_device_sampled(device: i32,
    sample: impl FnOnce() -> anyhow::Result<(u64, u64, u64, u64, u64)>) -> anyhow::Result<()> {
    let Some(cache) = existing_cache(device) else { return Ok(()) };
    let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
    if cache.needs_boundary() {
        let (free, total, tracked, rounding, unclaimed) = sample()?;
        let (growth, level) = cache.observer.observe(free, total, tracked, rounding, unclaimed);
        cache.level = Some(level);
        cache.sample_physical(free, tracked);
        if cache.unmeasured { cache.measured_capture(growth); }
    }
    cache.boundary();
    Ok(())
}

/// Refresh the reported current level from a settled device sample and return
/// `(current level, admitted high-water)`. Reconciliation samples once per
/// stage after the device quiets, so a transient peak that reversed before that
/// sample releases its credit. The refreshed level is bounded by the admitted
/// high-water: any non-graph untracked allocation live at the sample (a
/// workspace, exchange staging, pending eager scratch) is attributed elsewhere,
/// never to the graph level, so this path can only release peak credit and
/// never raise the charge. `charged` and the observer's high-water stay
/// untouched, so admission still covers the peak. Stage samples only: never
/// call this per round or expose the result as a live value mid-request.
pub(crate) fn refreshed_arena(device: i32, free: u64, total: u64) -> Option<(u64, u64)> {
    let (tracked, rounding, unclaimed) = ledger_sample(device);
    refreshed_arena_sampled(device, || Ok((free, total, tracked, rounding, unclaimed)))
}

fn refreshed_arena_sampled(device: i32,
    sample: impl FnOnce() -> anyhow::Result<(u64, u64, u64, u64, u64)>) -> Option<(u64, u64)> {
    let cache = existing_cache(device)?;
    let (free, total, tracked, rounding, unclaimed) = sample().ok()?;
    let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
    let raw = cache.observer.current(free, total, tracked, rounding, unclaimed);
    let level = raw.min(cache.charged);
    cache.level = Some(level);
    cache.sample_physical(free, tracked);
    tracing::info!(target: "cuteafd::memory", device, free, total, tracked, unclaimed,
        raw, level, charged = cache.charged, "graph level refresh");
    Some((level, cache.charged))
}

/// Observe allocation growth and retire selected persistent-storage executables.
/// # Safety
/// Every visited bank's owner pins must borrow persistent engine storage (or be
/// empty); dropping an executable must not release allocations it captured.
pub(crate) unsafe fn observe_completed(library: &cuteafd_ffi::NativeLibrary,
    mut visit: impl FnMut(&mut GraphVisitor<'_>) -> anyhow::Result<()>) -> anyhow::Result<()> {
    let mut devices = std::collections::BTreeSet::new();
    visit(&mut |device, _, _, bank| { bank.tick(); devices.insert(device); Ok(()) })?;
    for device in devices { observe_device(library, device)?; }
    visit(&mut |_, _, _, bank| {
        // SAFETY: serving banks retain only borrowed handles of persistent engine
        // buffers (or ()); their containing pipelines order buffer reuse.
        unsafe { bank.evict_persistent(); }
        Ok(())
    })

}

pub(crate) struct DeviceCache {
    device: i32,
    reserve: u64,
    reserve_policy: Option<cuteafd_loader::placement::graph_reserve::Resolved>,
    charged: u64,
    level: Option<u64>,
    physical_floor: u64,
    physical_sample: Option<(Instant, u64, u64)>,
    physical_floor_warned: bool,
    floor_resample_pending: bool,
    floor_resample_used: bool,
    pending_bytes: u64,
    next_id: u64,
    next_bank: u64,
    identity_generation: u64,
    recovery_generation: u64,
    recent_target: u64,
    records: HashMap<u64, Record>,
    identities: HashMap<Identity, u64>,
    queues: [Queue; 4],
    classes: HashMap<&'static str, SizeClass>,
    pending_growth: u64,
    pools: Pools,
    basis: Basis,
    chain_only: bool,
    unmeasured: bool,
    pending_destroy: usize,
    observer: GrowthObserver,
    memory_query: Option<cuteafd_ffi::PhysicalMemoryQuery>,
    captured_since_observation: bool,
    counters: HashMap<Cycle, Counters>,
    started: Instant,
    clock_ms: u64,
    last_report: Instant,
    interval_evictions: u64,
    trace: Option<TraceWriter>,
    retired_trace: Option<TraceSnapshot>,
    last_trace: Instant,
}

impl DeviceCache {
    fn new(device: i32, reserve: u64) -> Self {
        Self { device, reserve, reserve_policy: None, charged: 0, level: None, pending_bytes: 0, next_id: 0, next_bank: 0, identity_generation: 0, recovery_generation: 0,
            physical_floor: 0, physical_sample: None, physical_floor_warned: false,
            floor_resample_pending: false, floor_resample_used: false,
            recent_target: 0, records: HashMap::new(), identities: HashMap::new(),
            queues: std::array::from_fn(|_| Queue::default()), classes: HashMap::new(),
            pending_growth: 0, pools: Pools::default(), basis: Basis::Bytes,
            chain_only: true, unmeasured: false, pending_destroy: 0,
            observer: GrowthObserver::default(), memory_query: None, captured_since_observation: false,
            counters: HashMap::new(), started: Instant::now(), clock_ms: 0, last_report: Instant::now(),
            interval_evictions: 0, trace: std::env::var_os("CUTEAFD_GRAPH_TRACE").and_then(|path| {
                TraceWriter::new(std::path::PathBuf::from(path).with_extension(format!("device-{device}.json")))
                    .map_err(|error| tracing::warn!(%error, "graph trace open failed")).ok()
            }), retired_trace: None,
            last_trace: Instant::now() - Duration::from_secs(1) }
    }

    pub fn bank(&mut self) -> u64 { self.next_bank += 1; self.next_bank }
    pub fn reserve(&self) -> u64 { self.reserve }
    pub fn identity_generation(&self) -> u64 { self.identity_generation }
    pub fn retains_identity(&self, identity: &Identity) -> bool { self.identities.contains_key(identity) }
    pub fn needs_boundary(&self) -> bool {
        self.unmeasured || self.pending_destroy > 0 || self.floor_resample_pending
    }
    pub fn has_pending_victims(&self) -> bool { self.pending_destroy > 0 }
    pub fn pending(&self, id: u64) -> bool { self.records.get(&id).is_some_and(|r| r.pending_destroy) }

    fn sample_physical(&mut self, free: u64, tracked: u64) {
        self.physical_sample = Some((Instant::now(), free, tracked));
        self.floor_resample_pending = false;
    }

    fn monitor_physical(&mut self, sampled_at: Instant, free: u64, tracked: u64) {
        if self.physical_sample.is_none_or(|(previous, _, _)| sampled_at > previous) {
            // An asynchronous report updates physical admission only, never the
            // graph observer, charge, pending tickets or retry state.
            self.physical_sample = Some((sampled_at, free, tracked));
        }
    }

    fn floor_skip(&mut self, old: Option<u64>, cycle: Cycle) -> Miss {
        if let Some(record) = old.and_then(|id| self.records.get_mut(&id)) {
            record.stats.event(cycle, 1);
            record.stats.misses += 1;
        }
        let counters = self.counters.entry(cycle).or_default();
        counters.eager_runs += 1;
        counters.physical_floor_skips += 1;
        Miss::Eager
    }

    fn physical_room(&mut self, estimate: u64, headroom: u64) -> bool {
        if self.physical_floor == 0 { return true }
        let Some((_, free, tracked)) = self.physical_sample else { return true };
        let current = cuteafd_ffi::memory_ledger::current_bytes(
            cuteafd_ffi::memory_ledger::Space::Device, self.device) as u64;
        let free = free.saturating_sub(current.saturating_sub(tracked))
            .saturating_sub(self.pending_growth);
        let required = estimate.saturating_add(headroom).saturating_add(self.physical_floor);
        if free >= required { return true }
        if !self.physical_floor_warned {
            tracing::warn!(device = self.device, free, estimate, headroom, floor = self.physical_floor,
                reserve = self.reserve, charged = self.charged,
                "physical graph headroom below floor while graph reserve has slack; memory accounting probably has a missing owner; running eagerly");
            self.physical_floor_warned = true;
        }
        if !self.floor_resample_used {
            self.floor_resample_pending = true;
            self.floor_resample_used = true;
        }
        false
    }

    fn unlink(&mut self, id: u64) {
        let record = &self.records[&id];
        let Some(list) = record.list else { return };
        let (previous, next, bytes) = (record.previous, record.next, record.bytes);
        if let Some(previous) = previous { self.records.get_mut(&previous).unwrap().next = next; }
        if let Some(next) = next { self.records.get_mut(&next).unwrap().previous = previous; }
        let queue = &mut self.queues[list.index()];
        if queue.first == Some(id) { queue.first = next; }
        if queue.last == Some(id) { queue.last = previous; }
        queue.bytes = queue.bytes.saturating_sub(bytes);
        queue.len -= 1;
        let record = self.records.get_mut(&id).unwrap();
        record.list = None; record.previous = None; record.next = None;
    }

    fn push(&mut self, id: u64, list: List) {
        self.unlink(id);
        let queue = &mut self.queues[list.index()];
        let last = queue.last;
        queue.last = Some(id);
        if queue.first.is_none() { queue.first = Some(id); }
        queue.bytes = queue.bytes.saturating_add(self.records[&id].bytes);
        queue.len += 1;
        if let Some(last) = last { self.records.get_mut(&last).unwrap().next = Some(id); }
        let record = self.records.get_mut(&id).unwrap();
        record.previous = last; record.next = None; record.list = Some(list);
    }

    fn forget(&mut self, id: u64) {
        self.unlink(id);
        if let Some(record) = self.records.remove(&id) { self.identities.remove(&record.identity); self.identity_generation += 1; }
    }

    fn trim_ghosts(&mut self) {
        for list in [List::RecentGhost, List::FrequentGhost] {
            // Both byte and count bounds matter: zero-byte allocator reuse must
            // not make the history an unbounded key archive.
            while self.queues[list.index()].bytes > self.reserve || self.queues[list.index()].len > 65_536 {
                self.forget(self.queues[list.index()].first.unwrap());
            }
        }
    }

    #[cfg(test)]
    pub fn hit(&mut self, id: u64) {
        let Some(record) = self.records.get(&id) else { return };
        self.hit_cycle(id, record.cycle);
    }

    /// One timestamp per cycle/boundary, not per executable replay.
    pub fn sample_clock(&mut self) { self.clock_ms = self.started.elapsed().as_millis() as u64; }

    pub fn hit_cycle(&mut self, id: u64, cycle: Cycle) {
        let now = self.clock_ms;
        let Some(record) = self.records.get_mut(&id) else { return };
        let (pinned, pending) = (record.pinned, record.pending_destroy);
        record.stats.hit(cycle, 1, now);
        self.counters.entry(cycle).or_default().hits += 1;
        if !pinned && !pending { self.push(id, List::Frequent); }
    }

    /// Pinned hits accumulate in the bank, without a lock or ARC mutation.
    pub fn pinned_hits(&mut self, id: u64, cycle: Cycle, count: u64) {
        if count == 0 { return }
        let now = self.clock_ms;
        if let Some(record) = self.records.get_mut(&id) { record.stats.hit(cycle, count, now); }
        self.counters.entry(cycle).or_default().hits += count;
    }

    #[cfg(test)]
    pub fn cycle_hits(&self, cycle: Cycle) -> u64 {
        self.counters.get(&cycle).map_or(0, |counters| counters.hits)
    }

    fn estimate(&self, site: Site, node_estimate: Option<u64>) -> (u64, u64) {
        let class = self.classes.get(site.name);
        let estimate = node_estimate.filter(|n| *n > 0).unwrap_or_else(|| class
            .filter(|c| c.samples > 0).map_or(site.bootstrap, |c| c.total.div_ceil(c.samples)));
        (estimate, class.map_or(estimate, |c| c.largest.max(estimate)))
    }

    fn mark_victim(&mut self, frequent_ghost: bool) -> Option<u64> {
        let recent = &self.queues[List::Recent.index()];
        let prefer_recent = recent.first.is_some() && (recent.bytes > self.recent_target
            || (frequent_ghost && recent.bytes == self.recent_target));
        let first = if prefer_recent { List::Recent } else { List::Frequent };
        let second = if prefer_recent { List::Frequent } else { List::Recent };
        let mut selected = None;
        for list in [first, second] {
            let mut next = self.queues[list.index()].first;
            while let Some(id) = next {
                let record = &self.records[&id];
                if !record.pending_capture {
                    selected = Some(id); break;
                }
                next = record.next;
            }
            if selected.is_some() { break }
        }
        let id = selected?;
        let frequent = self.records[&id].list == Some(List::Frequent);
        self.unlink(id);
        self.records.get_mut(&id).unwrap().frequent = frequent;
        self.records.get_mut(&id).unwrap().pending_destroy = true;
        self.pending_destroy += 1;
        Some(id)
    }

    fn capture_baseline_sampled(&mut self,
        sample: impl FnOnce() -> anyhow::Result<(u64, u64, u64, u64, u64)>) -> anyhow::Result<()> {
        if self.captured_since_observation { return Ok(()) }
        // Exclude eager first-touch since the last observation, but never swallow
        // growth from another capture in the same unobserved batch.
        let (free, total, tracked, rounding, unclaimed) = sample()?;
        let (_, level) = self.observer.observe(free, total, tracked, rounding, unclaimed);
        self.level = Some(level);
        self.sample_physical(free, tracked);
        self.captured_since_observation = true;
        Ok(())
    }

    pub fn miss(&mut self, identity: Identity, site: Site, node_estimate: Option<Estimate>, pinned: bool) -> Miss {
        let now = self.started.elapsed().as_millis() as u64;
        self.counters.entry(site.cycle).or_default().misses += 1;
        let compatible = site.basis == self.basis && match (self.basis, &node_estimate) {
            (Basis::Bytes, None | Some(Estimate { nodes: None, .. })) => true,
            (Basis::Nodes { node_rate, graph_rate }, Some(estimate)) =>
                estimate.nodes.is_some() && estimate.node_rate == node_rate && estimate.graph_rate == graph_rate,
            _ => false,
        };
        if !compatible {
            tracing::warn!(device = self.device, site = site.name, basis = ?self.basis,
                "graph estimate basis mismatch; running eagerly");
            self.counters.entry(site.cycle).or_default().eager_runs += 1;
            return Miss::Eager;
        }
        let old = self.identities.get(&identity).copied();
        if let Some(record) = old.and_then(|id| self.records.get_mut(&id)) {
            if record.permanently_eager {
                record.stats.event(site.cycle, 1);
                record.stats.misses += 1;
                self.counters.entry(site.cycle).or_default().eager_runs += 1;
                return Miss::Eager;
            }
            if let Some(backoff) = &mut record.oom {
                let level_recovered = backoff.level.zip(self.level)
                    .is_some_and(|(before, now)| before.saturating_sub(now) >= OOM_LEVEL_DROP);
                if backoff.remaining > 0 && backoff.recovery_generation == self.recovery_generation && !level_recovered {
                    backoff.remaining -= 1;
                    record.stats.event(site.cycle, 1);
                    record.stats.misses += 1;
                    self.counters.entry(site.cycle).or_default().eager_runs += 1;
                    return Miss::Eager;
                }
                record.oom = None;
            }
        }
        let (estimate, headroom) = self.estimate(site, node_estimate.as_ref().map(|e| e.bytes));
        let ghost = old.and_then(|id| self.records[&id].list)
            .filter(|list| matches!(list, List::RecentGhost | List::FrequentGhost));
        if let Some(list) = ghost {
            let b1 = self.queues[List::RecentGhost.index()].bytes;
            let b2 = self.queues[List::FrequentGhost.index()].bytes;
            if list == List::RecentGhost {
                self.recent_target = self.recent_target.saturating_add(estimate.max((b2 / b1.max(1)).saturating_mul(estimate))).min(self.reserve);
            } else {
                self.recent_target = self.recent_target.saturating_sub(estimate.max((b1 / b2.max(1)).saturating_mul(estimate)));
            }
        }
        let metadata = node_estimate.unwrap_or(Estimate { bytes: estimate, nodes: None,
            node_rate: 1, graph_rate: 0, parallel_width: 1 });
        // Without a node census use measured graph bytes as node-pool units;
        // the optional node model separates the two permanent driver pools.
        let nodes = metadata.nodes.unwrap_or(estimate);
        let next_nodes = self.pools.live_nodes.saturating_add(nodes);
        let next_execs = self.pools.live_execs.saturating_add(1);
        let chain = self.chain_only && metadata.parallel_width == 1 && !site.spans_devices;
        if !chain { self.chain_only = false; }
        let projected = if chain {
            self.basis.pool_bytes(next_nodes.max(self.pools.node_high_water),
                next_execs.max(self.pools.exec_high_water)).saturating_sub(
                    self.basis.pool_bytes(self.pools.node_high_water, self.pools.exec_high_water))
        } else {
            // Unqualified topology cannot reuse the calibrated chain pools.
            self.pending_growth.saturating_add(estimate)
        };
        let growth = projected.saturating_sub(self.pending_growth);
        let reuse = chain && projected == 0 && !self.unmeasured;
        let allowed = site.preference != Preference::Eager && estimate > 0;
        let room = may_capture(site.preference, growth, headroom, self.reserve,
            self.charged, self.pending_growth, self.pending_bytes, reuse);
        if allowed && room && old.is_none_or(|id| !self.records[&id].pending_destroy && !self.records[&id].pending_capture) {
            // No CUDA query or ticket for a planned physical-headroom miss.
            if !self.physical_room(estimate, headroom) {
                return self.floor_skip(old, site.cycle);
            }
            if !self.captured_since_observation {
                if let Some(query) = &self.memory_query {
                    let sample = query.sample(self.device).map(|(free, total)| {
                        let (tracked, rounding, unclaimed) = ledger_sample(self.device);
                        (free as u64, total as u64, tracked, rounding, unclaimed)
                    });
                    if let Err(error) = self.capture_baseline_sampled(|| sample) {
                        tracing::warn!(%error, device = self.device, "graph capture baseline unavailable; running eagerly");
                        self.counters.entry(site.cycle).or_default().eager_runs += 1;
                        return Miss::Eager;
                    }
                }
            }
            // The existing baseline query can reveal lower free memory than
            // the boundary precheck. Honor that sample without another query.
            if !self.physical_room(estimate, headroom) {
                if !self.unmeasured { self.captured_since_observation = false; }
                return self.floor_skip(old, site.cycle);
            }
            let id = if let Some(id) = old { self.unlink(id); id } else {
                self.next_id += 1;
                let id = self.next_id;
                self.identities.insert(identity.clone(), id);
                self.records.insert(id, Record { identity, cycle: site.cycle, bytes: 0, list: None,
                    previous: None, next: None, pinned, pending_destroy: false, pending_capture: false, frequent: false, nodes, node_rate: metadata.node_rate, graph_rate: metadata.graph_rate, parallel_width: metadata.parallel_width, spans_devices: site.spans_devices, oom: None, oom_warned: false, permanently_eager: false, stats: KeyStats::new(now) });
                id
            };
            let record = self.records.get_mut(&id).unwrap();
            record.bytes = estimate; record.pinned = pinned; record.pending_capture = true;
            record.nodes = nodes; record.node_rate = metadata.node_rate;
            record.graph_rate = metadata.graph_rate; record.parallel_width = metadata.parallel_width;
            record.spans_devices |= site.spans_devices;
            self.pending_growth = self.pending_growth.max(projected);
            self.unmeasured = true;
            self.pools.live_nodes = next_nodes;
            self.pools.live_execs = next_execs;
            let stats = &mut record.stats;
            stats.event(site.cycle, 1);
            stats.misses += 1;
            stats.ghost_hits += u64::from(ghost.is_some());
            stats.declared_capture_streams = metadata.parallel_width;
            stats.spans_devices |= site.spans_devices;
            if metadata.parallel_width > 1 {
                tracing::warn!(site = site.name, width = metadata.parallel_width,
                    "parallel graph capture; chain node model does not apply");
            }
            // A returning ghost enters frequency, a cold key enters recency.
            if !pinned { self.push(id, if ghost.is_some() { List::Frequent } else { List::Recent }); }
            self.pending_bytes = self.pending_bytes.saturating_add(estimate);
            self.floor_resample_used = false;

            return Miss::Capture(id);
        }
        if allowed && (ghost.is_some() || site.preference == Preference::Capture) {
            let selected_nodes: u64 = self.records.values().filter(|r| r.pending_destroy).map(|r| r.nodes).sum();
            let mut selected_nodes = selected_nodes;
            let mut selected_execs = self.pending_destroy as u64;
            while next_nodes.saturating_sub(selected_nodes) > self.pools.node_high_water
                || next_execs.saturating_sub(selected_execs) > self.pools.exec_high_water {
                let Some(id) = self.mark_victim(ghost == Some(List::FrequentGhost)) else { break };
                selected_nodes = selected_nodes.saturating_add(self.records[&id].nodes);
                selected_execs += 1;
            }
        }
        // A cold adaptive miss is remembered without displacing resident work.
        if old.is_none() && estimate > 0 {
            self.next_id += 1;
            let id = self.next_id;
            self.identities.insert(identity.clone(), id);
            self.records.insert(id, Record { identity: identity.clone(), cycle: site.cycle, bytes: estimate,
                list: None, previous: None, next: None, pinned: false, pending_destroy: false, pending_capture: false, frequent: false, nodes, node_rate: metadata.node_rate, graph_rate: metadata.graph_rate, parallel_width: metadata.parallel_width, spans_devices: site.spans_devices, oom: None, oom_warned: false, permanently_eager: false, stats: KeyStats::new(now) });
            self.push(id, List::RecentGhost);
            self.trim_ghosts();
        }
        if let Some(id) = self.identities.get(&identity) {
            let stats = &mut self.records.get_mut(id).unwrap().stats;
            stats.event(site.cycle, 1);
            stats.misses += 1;
            stats.ghost_hits += u64::from(ghost.is_some());
        }
        self.counters.entry(site.cycle).or_default().eager_runs += 1;
        Miss::Eager
    }

    /// Retain the failed identity in ARC's bounded ghost history. No executable
    /// survived, but driver growth still needs the ordinary completed sample.
    /// Returns true only for the key's first OOM warning while it is retained.
    pub fn capture_oom(&mut self, id: u64) -> bool {
        let Some(record) = self.records.get_mut(&id) else { return false };
        if !record.pending_capture { return false }
        self.pending_bytes = self.pending_bytes.saturating_sub(record.bytes);
        self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes);
        self.pools.live_execs = self.pools.live_execs.saturating_sub(1);
        record.pending_capture = false;
        record.pinned = false;
        record.oom = Some(OomBackoff { remaining: OOM_RETRY_MISSES, level: self.level,
            recovery_generation: self.recovery_generation });
        let warn = !record.oom_warned;
        record.oom_warned = true;
        let ghost = if record.frequent || record.list == Some(List::Frequent) {
            List::FrequentGhost
        } else { List::RecentGhost };
        self.push(id, ghost);
        self.trim_ghosts();
        warn
    }

    /// Retain exact rejected identities outside ARC's evictable ghost lists.
    /// Captured commands did not run and no executable was instantiated.
    pub fn capture_too_small(&mut self, id: u64, nodes: usize) {
        let Some(record) = self.records.get_mut(&id) else { return };
        if record.permanently_eager { return }
        if record.pending_capture {
            self.pending_bytes = self.pending_bytes.saturating_sub(record.bytes);
            self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes);
            self.pools.live_execs = self.pools.live_execs.saturating_sub(1);
        }
        record.pending_capture = false;
        record.pinned = false;
        record.permanently_eager = true;
        record.stats.native_nodes = Some(nodes as u64);
        tracing::info!(device = self.device, site = record.identity.site,
            key = %record.identity.shape, nodes, "small graph permanently eager");
        self.unlink(id);
        // Physical charge remains a monotone high-water, including failed captures.
    }

    pub fn cancel_capture(&mut self, id: u64) {
        if let Some(record) = self.records.get(&id) {
            if record.pending_capture {
                self.pending_bytes = self.pending_bytes.saturating_sub(record.bytes);
                self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes);
                self.pools.live_execs = self.pools.live_execs.saturating_sub(1);
                // Keep the in-flight growth reservation until the next sample:
                // a failed instantiate may still have expanded driver storage.
            }
        }
        self.forget(id);
    }

    pub fn captured_topology(&mut self, id: u64, nodes: u64, width: usize, spans_devices: bool) {
        let Some(record) = self.records.get_mut(&id) else { return };
        // Peer pointers in kernel arguments are invisible to native inspection.
        let spans_devices = spans_devices || record.spans_devices;
        record.spans_devices = spans_devices;
        if width != 1 || spans_devices {
            // Previously reserved chain discounts are no longer justified,
            // including pending captures elsewhere on this device.
            self.pending_growth = self.pending_growth.max(self.pending_bytes);
            self.chain_only = false;
            tracing::warn!(device = self.device, site = record.identity.site, width, spans_devices,
                "non-chain graph capture; disabling high-water reuse assumption");
        }
        // Unknown metadata cannot erase the caller's node estimate.
        if (nodes != 0 || width != 0) && matches!(self.basis, Basis::Nodes { .. }) {
            self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes).saturating_add(nodes);
            record.nodes = nodes;
        }
        record.parallel_width = width;
        if nodes != 0 || width != 0 { record.stats.native_nodes = Some(nodes); }
        record.stats.max_parallel_width = width;
        record.stats.spans_devices |= spans_devices;
    }

    /// Called once per device after the complete multi-rank step has drained.
    /// `bytes` is the measured net graph allocation, not a sum of estimates.
    pub fn measured_capture(&mut self, bytes: u64) {
        let pending: Vec<_> = self.records.iter().filter(|(_, r)| r.pending_capture).map(|(&id, r)| (id, r.bytes)).collect();
        if pending.is_empty() && !self.unmeasured && bytes == 0 { return }
        // Only successful owners surviving this completed boundary prove reuse.
        self.pools.node_high_water = self.pools.node_high_water.max(self.pools.live_nodes);
        self.pools.exec_high_water = self.pools.exec_high_water.max(self.pools.live_execs);
        let total = self.pending_bytes.max(1);
        let mut remaining = bytes;
        for (index, &(id, estimated)) in pending.iter().enumerate() {
            let share = if index + 1 == pending.len() { remaining }
                else { ((bytes as u128 * estimated as u128) / total as u128) as u64 };
            remaining = remaining.saturating_sub(share);
            let list = self.records[&id].list;
            self.unlink(id);
            let record = self.records.get_mut(&id).unwrap();
            record.pending_capture = false;
            record.stats.capture_bytes = Some(share);
            record.stats.captures += 1;
            if record.stats.evicted_since_capture { record.stats.recaptures += 1; }
            record.stats.evicted_since_capture = false;
            // Zero physical growth can mean allocator reuse, not a zero-size
            // graph. Keep its estimate for victim ordering, never for charge.
            record.bytes = if share > 0 { share } else { estimated };
            let class = self.classes.entry(record.identity.site).or_default();
            if share > 0 { class.total = class.total.saturating_add(share); class.samples += 1; class.largest = class.largest.max(share); }
            self.counters.entry(record.cycle).or_default().captures += 1;
            if let Some(list) = list { self.push(id, list); }
        }
        self.pending_bytes = 0;
        self.pending_growth = 0;
        self.unmeasured = false;
        self.captured_since_observation = false;
        self.charged = self.charged.saturating_add(bytes);
        if self.charged > self.reserve {
            tracing::warn!(device = self.device, charged = self.charged, reserve = self.reserve,
                "graph cache measured over reserve; scheduling eviction, never refusing request");
            let mut selected = 0u64;
            while selected < self.charged - self.reserve {
                let Some(id) = self.mark_victim(false) else { break };
                selected = selected.saturating_add(self.records[&id].bytes);
            }
        }
    }

    pub fn destroyed(&mut self, ids: &[u64], measurement: ReleaseMeasurement) {
        if ids.iter().any(|id| self.records.contains_key(id)) { self.floor_resample_used = false; }
        // Observed free deltas remain available to alternate accounting adapters;
        // the default driver-arena model never credits physical release.
        let _ = measurement.released;
        if ids.iter().any(|id| self.records.contains_key(id)) { self.recovery_generation += 1; }
        for &id in ids {
            let Some(record) = self.records.get_mut(&id) else { continue };
            let evicted = record.pending_destroy;
            record.pending_destroy = false;
            if evicted { self.pending_destroy -= 1; }
            self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes);
            self.pools.live_execs = self.pools.live_execs.saturating_sub(1);
            if evicted {
                record.stats.evictions += 1;
                record.stats.evicted_since_capture = true;
                self.counters.entry(record.cycle).or_default().evictions += 1;
                self.interval_evictions += 1;
            }
            // Keep the ghost's recency/frequency origin through retirement.
            let ghost = if record.frequent { List::FrequentGhost } else { List::RecentGhost };
            self.push(id, ghost);
        }
        self.trim_ghosts();
    }

    pub fn remove_bank(&mut self, bank: u64) {
        // The first retiring bank still sees the complete cache. Preserve that
        // final view until serving resumes, rather than accumulating retired rows.
        if self.trace.is_some() {
            if self.retired_trace.is_none() { self.retired_trace = Some(self.trace_snapshot()); }
            // Each retiring bank flushes its last pinned hits just before here.
            if let Some(snapshot) = &mut self.retired_trace {
                for row in snapshot.records.iter_mut().filter(|r| r.identity.bank == bank) {
                    if let Some(record) = self.identities.get(&row.identity).and_then(|id| self.records.get(id)) {
                        *row = RawRecord::copy(record);
                    }
                }
                snapshot.cache.cycles = self.counters.clone();
            }
        }
        let ids: Vec<_> = self.records.iter().filter(|(_, r)| r.identity.bank == bank).map(|(&id, _)| id).collect();
        self.remove_drained(&ids);
    }

    /// Forget destroyed owners, never crediting physical driver-arena release.
    pub fn remove_drained(&mut self, ids: &[u64]) {
        if ids.iter().any(|id| self.records.contains_key(id)) { self.floor_resample_used = false; }
        if ids.iter().any(|id| self.records.get(id).is_some_and(|record|
            record.pinned || record.pending_destroy || matches!(record.list, Some(List::Recent | List::Frequent)))) {
            self.recovery_generation += 1;
        }
        for &id in ids {
            let Some(record) = self.records.get(&id) else { continue };
            if record.pending_capture {
                self.pending_bytes = self.pending_bytes.saturating_sub(record.bytes);
                self.unmeasured = true;
            }
            if record.pinned || record.pending_destroy || matches!(record.list, Some(List::Recent | List::Frequent)) {
                self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes);
                self.pools.live_execs = self.pools.live_execs.saturating_sub(1);
                if record.pending_destroy { self.pending_destroy -= 1; }
            }
            self.forget(id);
        }
    }

    pub fn report_due(&mut self, elapsed: Duration) -> bool {
        if elapsed < Duration::from_secs(30) || self.interval_evictions == 0 { return false }
        tracing::info!(device = self.device, evictions = self.interval_evictions,
            reserve = self.reserve, charged = self.charged, cycles = ?self.counters, "graph cache interval");
        self.interval_evictions = 0;
        self.last_report = Instant::now();
        true
    }

    pub fn boundary(&mut self) {
        self.sample_clock();
        self.retired_trace = None;
        self.report_due(self.last_report.elapsed());
        if self.trace.is_some() && self.last_trace.elapsed() >= Duration::from_secs(1) {
            self.last_trace = Instant::now();
            self.publish_trace();
        }
    }

    fn trace_snapshot(&self) -> TraceSnapshot {
        let records = self.records.values().map(RawRecord::copy).collect();
        TraceSnapshot { cache: Snapshot { device: self.device, reserve: self.reserve, physical_floor: self.physical_floor,
            reserve_policy: self.reserve_policy, charged: self.charged, level: self.level,
            pinned: self.records.values().filter(|r| r.pinned).map(|r| r.bytes).sum(),
            pending_capture: self.pending_bytes, cycles: self.counters.clone(), pools: self.pools.clone(),
            basis: self.basis, sampled_at_ms: self.started.elapsed().as_millis() as u64,
            telemetry_bytes: self.records.len() * std::mem::size_of::<KeyStats>(),
            top_hits: Vec::new(), top_churn: Vec::new() },
            records, classes: self.classes.clone() }
    }

    fn publish_trace(&self) {
        if let Some(trace) = &self.trace { trace.submit(self.trace_snapshot()); }
    }

    fn finish_trace(&mut self) -> Option<TraceWriter> {
        if let Some(trace) = &self.trace {
            trace.submit(self.retired_trace.take().unwrap_or_else(|| self.trace_snapshot()));
        }
        self.trace.take()
    }

    fn snapshot(&self) -> Snapshot {
        let now = self.started.elapsed().as_millis() as u64;
        let ranked = |churn: bool| {
            let mut records: Vec<_> = self.records.values().filter(|r| if churn {
                r.stats.evictions + r.stats.recaptures > 0
            } else { r.stats.hits > 0 }).collect();
            records.sort_unstable_by(|a, b| {
                let score = |r: &Record| if churn { r.stats.evictions + r.stats.recaptures } else { r.stats.hits };
                score(b).cmp(&score(a)).then_with(|| (a.identity.site, a.identity.bank, &a.identity.shape)
                    .cmp(&(b.identity.site, b.identity.bank, &b.identity.shape)))
            });
            records.into_iter().take(16).map(|record| self.row(record, now)).collect()
        };
        Snapshot { device: self.device, reserve: self.reserve, physical_floor: self.physical_floor, reserve_policy: self.reserve_policy, charged: self.charged,
            level: self.level,
            pinned: self.records.values().filter(|r| r.pinned).map(|r| r.bytes).sum(),
            pending_capture: self.pending_bytes, cycles: self.counters.clone(), pools: self.pools.clone(), basis: self.basis,
            sampled_at_ms: now, telemetry_bytes: self.records.len() * std::mem::size_of::<KeyStats>(), top_hits: ranked(false), top_churn: ranked(true) }
    }

    fn row(&self, record: &Record, now: u64) -> TraceRow {
        raw_row(self.device, self.basis, &RawRecord::copy(record), now)
    }
}

fn raw_row(device: i32, basis: Basis, record: &RawRecord, now: u64) -> TraceRow {
        let stats = &record.stats;
        let end = now / HIT_INTERVAL_MS;
        let start = end.saturating_sub(HIT_INTERVALS as u64 - 1);
        let counts = (start..=end).map(|interval| {
            if interval <= stats.interval && stats.interval - interval < HIT_INTERVALS as u64 {
                stats.hit_intervals[interval as usize % HIT_INTERVALS]
            } else { 0 }
        }).collect();
        TraceRow { device: device, identity: record.identity.clone(), cycle: record.cycle,
            count: stats.count, cycles: [Cycle::Decode, Cycle::Verify, Cycle::Prefill, Cycle::Wave, Cycle::Other]
                .into_iter().map(|cycle| (cycle, stats.cycles[cycle as usize])).collect(),
            first_seen_ms: stats.first_seen_ms, hits: stats.hits, misses: stats.misses, captures: stats.captures,
            evictions: stats.evictions, recaptures: stats.recaptures, ghost_hits: stats.ghost_hits,
            last_hit_ms: stats.last_hit_ms, hit_histogram: HitHistogram {
                rolling: true, interval_ms: HIT_INTERVAL_MS, start_ms: start * HIT_INTERVAL_MS, counts },
            capture_bytes: stats.capture_bytes, nodes: stats.native_nodes.or_else(|| matches!(basis, Basis::Nodes { .. }).then_some(record.nodes)),
            executables: u64::from(!record.permanently_eager), permanently_eager: record.permanently_eager, basis,
            permanent_bytes: if record.permanently_eager { 0 } else { match basis {
                Basis::Bytes => record.bytes,
                Basis::Nodes { node_rate, graph_rate } => record.nodes.saturating_mul(node_rate).saturating_add(graph_rate),
            } }, pinned: record.pinned, declared_capture_streams: stats.declared_capture_streams,
            max_parallel_width: stats.max_parallel_width, spans_devices: stats.spans_devices }
 }

impl Drop for DeviceCache {
    fn drop(&mut self) { drop(self.finish_trace()); }
}

pub(crate) fn shutdown_traces() {
    let caches: Vec<_> = DEVICES.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner())
        .values().filter_map(std::sync::Weak::upgrade).collect();
    for cache in caches {
        let writer = cache.lock().unwrap_or_else(|p| p.into_inner()).finish_trace();
        // No cache lock is held while waiting for JSON or filesystem work.
        drop(writer);
    }
}

#[derive(Clone)]
struct RawRecord {
    identity: Identity,
    cycle: Cycle,
    bytes: u64,
    nodes: u64,
    pinned: bool,
    permanently_eager: bool,
    stats: KeyStats,
}
impl RawRecord {
    fn copy(record: &Record) -> Self {
        Self { identity: record.identity.clone(), cycle: record.cycle, bytes: record.bytes,
            nodes: record.nodes, pinned: record.pinned, permanently_eager: record.permanently_eager,
            stats: (*record.stats).clone() }
    }
}
struct TraceSnapshot {
    cache: Snapshot,
    records: Vec<RawRecord>,
    classes: HashMap<&'static str, SizeClass>,
}
impl TraceSnapshot {
    fn bytes(mut self) -> std::io::Result<Vec<u8>> {
        let mut rows: Vec<_> = self.records.iter().map(|r| raw_row(self.cache.device,
            self.cache.basis, r, self.cache.sampled_at_ms)).collect();
        rows.sort_by(|a, b| (a.identity.site, a.identity.bank, &a.identity.shape)
            .cmp(&(b.identity.site, b.identity.bank, &b.identity.shape)));
        let ranked = |churn: bool| {
            let score = |r: &TraceRow| if churn { r.evictions + r.recaptures } else { r.hits };
            let mut ranked: Vec<_> = rows.iter().filter(|r| score(r) > 0).collect();
            ranked.sort_unstable_by(|a, b| score(b).cmp(&score(a)).then_with(||
                (a.identity.site, a.identity.bank, &a.identity.shape)
                    .cmp(&(b.identity.site, b.identity.bank, &b.identity.shape))));
            ranked.into_iter().take(16).cloned().collect()
        };
        self.cache.top_hits = ranked(false);
        self.cache.top_churn = ranked(true);
        let document = serde_json::json!({"schema_version": 2, "cache": self.cache, "graphs": rows,
            "sites": self.classes.iter().map(|(name, class)| (*name, serde_json::json!({
                "samples": class.samples, "total_bytes": class.total, "largest_bytes": class.largest
            }))).collect::<std::collections::BTreeMap<_, _>>()});
        serde_json::to_vec(&document).map_err(std::io::Error::other)
    }
}

#[derive(Default)]
struct TraceSlot { pending: Option<TraceSnapshot>, closed: bool }
struct TraceWriter {
    slot: Arc<(Mutex<TraceSlot>, std::sync::Condvar)>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl TraceWriter {
    fn new(path: std::path::PathBuf) -> std::io::Result<Self> {
        Self::with_sink(move |snapshot| {
            let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
            let result = snapshot.bytes().and_then(|bytes| std::fs::write(&temporary, bytes))
                .and_then(|()| std::fs::rename(&temporary, &path));
            if let Err(error) = result {
                let _ = std::fs::remove_file(&temporary);
                tracing::warn!(%error, path = %path.display(), "graph trace write failed");
            }
        })
    }
    fn with_sink(mut sink: impl FnMut(TraceSnapshot) + Send + 'static) -> std::io::Result<Self> {
        let slot = Arc::new((Mutex::new(TraceSlot::default()), std::sync::Condvar::new()));
        let worker_slot = slot.clone();
        let thread = std::thread::Builder::new().name("graph-trace".into()).spawn(move || loop {
            let snapshot = {
                let (lock, wake) = &*worker_slot;
                let mut state = lock.lock().unwrap_or_else(|p| p.into_inner());
                while state.pending.is_none() && !state.closed {
                    state = wake.wait(state).unwrap_or_else(|p| p.into_inner());
                }
                match state.pending.take() { Some(snapshot) => snapshot, None => break }
            };
            sink(snapshot);
        })?;
        Ok(Self { slot, thread: Some(thread) })
    }
    fn submit(&self, snapshot: TraceSnapshot) {
        let (lock, wake) = &*self.slot;
        let old = lock.lock().unwrap_or_else(|p| p.into_inner()).pending.replace(snapshot);
        wake.notify_one();
        // Free superseded data outside the mailbox lock.
        drop(old);
    }
}
impl Drop for TraceWriter {
    fn drop(&mut self) {
        let (lock, wake) = &*self.slot;
        lock.lock().unwrap_or_else(|p| p.into_inner()).closed = true;
        wake.notify_one();
        if self.thread.take().is_some_and(|thread| thread.join().is_err()) {
            tracing::warn!("graph trace writer panicked");
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    fn site(preference: Preference) -> Site { Site { name: "test", cycle: Cycle::Decode, preference, bootstrap: 10, basis: Basis::Bytes, spans_devices: false } }
    fn key(n: u64) -> Identity { Identity { bank: 1, site: "test", key: n, shape: n.to_string().into() } }
    fn capture(cache: &mut DeviceCache, n: u64, pinned: bool) -> u64 {
        let Miss::Capture(id) = cache.miss(key(n), site(Preference::Capture), None, pinned) else { panic!("capture expected") };
        cache.measured_capture(10);
        id
    }
    #[test]
    fn rejected_small_graph_trace_has_no_executable_or_permanent_bytes() {
        for basis in [Basis::Bytes, Basis::Nodes { node_rate: 2560, graph_rate: 65536 }] {
            for nodes in 0..=2 {
                let mut cache = DeviceCache::new(-992, 64 << 20);
                cache.basis = basis;
                let mut capture_site = site(Preference::Capture);
                capture_site.basis = basis;
                let estimate = match basis {
                    Basis::Bytes => None,
                    Basis::Nodes { node_rate, graph_rate } => Some(Estimate {
                        bytes: node_rate * 3 + graph_rate, nodes: Some(3),
                        node_rate, graph_rate, parallel_width: 1 }),
                };
                let Miss::Capture(rejected) = cache.miss(key(1), capture_site, estimate.clone(), true) else { panic!() };
                cache.measured_capture(10);
                cache.capture_too_small(rejected, nodes);
                assert!(matches!(cache.miss(key(2), capture_site, estimate, false), Miss::Capture(_)));
                cache.measured_capture(10);
                let charged = cache.charged;
                let value: serde_json::Value = serde_json::from_slice(&cache.trace_snapshot().bytes().unwrap()).unwrap();
                let graphs = value["graphs"].as_array().unwrap();
                let eager = graphs.iter().find(|row| row["permanently_eager"] == true).unwrap();
                assert_eq!(eager["executables"], 0);
                assert_eq!(eager["permanent_bytes"], 0);
                assert_eq!(eager["nodes"], nodes);
                assert_eq!(eager["captures"], 1, "capture history survives rejection");
                assert_eq!(eager["pinned"], false);
                let retained = graphs.iter().find(|row| row["permanently_eager"].is_null()).unwrap();
                assert_eq!(retained["executables"], 1);
                assert!(retained["permanent_bytes"].as_u64().unwrap() > 0);
                assert_eq!(value["cache"]["charged"], charged);
            }
        }
    }
    #[test]
    fn oom_backoff_skips_256_misses_preserves_accounting_and_warns_once() {
        let mut cache = DeviceCache::new(-992, 1000);
        cache.level = Some(100);
        let Miss::Capture(id) = cache.miss(key(1), site(Preference::Capture), None, true) else { panic!() };
        assert!(cache.capture_oom(id));
        assert_eq!(cache.records[&id].list, Some(List::RecentGhost));
        assert!(!cache.records[&id].pinned);
        assert_eq!((cache.pending_bytes, cache.pools.live_nodes, cache.pools.live_execs), (0, 0, 0));
        assert!(cache.needs_boundary(), "failed instantiation can still grow driver storage");
        cache.measured_capture(10);
        assert_eq!(cache.charged, 10);
        assert!(!cache.needs_boundary());
        for _ in 0..OOM_RETRY_MISSES {
            assert!(matches!(cache.miss(key(1), site(Preference::Capture), None, false), Miss::Eager));
            assert!(!cache.needs_boundary(), "suppressed misses must not request a device sample");
        }
        assert_eq!(cache.counters[&Cycle::Decode].eager_runs, u64::from(OOM_RETRY_MISSES));
        let Miss::Capture(retry) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
        assert_eq!(retry, id, "the retained exact key is reused");
        assert!(!cache.capture_oom(retry), "repeated failures must not log again");
        cache.measured_capture(0);
        for _ in 0..OOM_RETRY_MISSES {
            assert!(matches!(cache.miss(key(1), site(Preference::Capture), None, false), Miss::Eager));
        }
        let Miss::Capture(retry) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
        cache.measured_capture(0);
        assert!(cache.records[&retry].oom.is_none());
        assert_eq!(cache.records[&retry].stats.captures, 1);
        assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Capture(_)));
    }

    #[test]
    fn oom_backoff_retries_after_real_eviction_not_victim_selection_or_ghost_removal() {
        for remove in [false, true] {
            let mut cache = DeviceCache::new(-993, 1000);
            let owner = capture(&mut cache, 1, false);
            let Miss::Capture(id) = cache.miss(key(2), site(Preference::Capture), None, false) else { panic!() };
            assert!(cache.capture_oom(id));
            cache.measured_capture(0);
            let Miss::Capture(ghost) = cache.miss(key(3), site(Preference::Capture), None, false) else { panic!() };
            cache.capture_oom(ghost);
            cache.measured_capture(0);
            cache.remove_drained(&[ghost]);
            assert_eq!(cache.mark_victim(false), Some(owner));
            assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
            if remove { cache.remove_drained(&[owner]); }
            else { cache.destroyed(&[owner], ReleaseMeasurement { released: 0 }); }
            assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Capture(retry) if retry == id));
        }
    }

    #[test]
    fn oom_backoff_retries_only_after_meaningful_observed_level_drop() {
        let mut cache = DeviceCache::new(-994, 16 << 20);
        cache.level = Some(4 << 20);
        let Miss::Capture(id) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
        cache.capture_oom(id);
        cache.measured_capture(0);
        cache.level = Some((4 << 20) - OOM_LEVEL_DROP + 1);
        assert!(matches!(cache.miss(key(1), site(Preference::Capture), None, false), Miss::Eager));
        cache.level = Some((4 << 20) - OOM_LEVEL_DROP);
        assert!(matches!(cache.miss(key(1), site(Preference::Capture), None, false), Miss::Capture(retry) if retry == id));
        assert!(!cache.capture_oom(id));
        cache.measured_capture(0);
        assert!(matches!(cache.miss(key(1), site(Preference::Capture), None, false), Miss::Eager),
            "a second OOM resets the level baseline and cooldown");
    }

    #[test]
    fn oom_ghosts_obey_arc_byte_and_count_bounds_and_forget_warning_history() {
        for (reserve, bootstrap, failures, retained) in [(100, 10, 11, 10), (1_000_000, 1, 65_537, 65_536)] {
            let mut cache = DeviceCache::new(-995, reserve);
            let site = Site { bootstrap, ..site(Preference::Capture) };
            for n in 0..failures {
                let Miss::Capture(id) = cache.miss(key(n), site, None, false) else { panic!() };
                assert!(cache.capture_oom(id));
            }
            let ghosts = &cache.queues[List::RecentGhost.index()];
            assert_eq!(ghosts.len, retained);
            assert!(ghosts.bytes <= reserve);
            assert_eq!(cache.records.len(), retained);
            assert_eq!(cache.identities.len(), retained);
            assert!(!cache.identities.contains_key(&key(0)));
            let Miss::Capture(id) = cache.miss(key(0), site, None, false) else { panic!() };
            assert!(cache.capture_oom(id), "forgotten bounded history may warn again");
            assert_eq!(cache.records.len(), retained);
        }
    }

    #[test]
    fn stats_publish_cost_with_five_thousand_graph_records() {
        use std::hint::black_box;
        let mut cache = DeviceCache::new(-991, 100_000);
        for n in 0..5_000 {
            let id = capture(&mut cache, n, false);
            cache.hit(id);
        }
        let publish = || serde_json::json!({"requests": 1, "generated_tokens": 128,
            "active": 16, "prefilling": 0, "prefix_cache": {}, "verify": {},
            "draft_policy": null, "kv_reserve": {}, "media": {}});
        let rounds = 256;
        let before = Instant::now();
        for _ in 0..rounds {
            let mut stats = publish();
            stats["graph_cache"] = serde_json::json!([cache.snapshot()]);
            black_box(stats);
        }
        let before_us = before.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
        let after = Instant::now();
        for _ in 0..rounds { black_box(publish()); }
        let after_us = after.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
        eprintln!("stats publish, 5000 hit records, {rounds} rounds, graph inventory inline -> lazy: {before_us:.2} -> {after_us:.2} us/round");
        assert_eq!(cache.snapshot().top_hits.len(), 16);
    }

    #[test]
    fn sigterm_http_shutdown_writes_complete_nonempty_final_traces() {
        const CHILD: &str = "CUTEAFD_TRACE_SIGTERM_TEST_CHILD";
        if let Some(directory) = std::env::var_os(CHILD) {
            let directory = std::path::PathBuf::from(directory);
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async {
                let mut cache = DeviceCache::new(-710, 100);
                cache.trace = Some(TraceWriter::new(directory.join("graph.json")).unwrap());
                let a = capture(&mut cache, 1, false);
                cache.hit(a);
                let trace = crate::shared::speculation_trace::Trace::open().unwrap().unwrap();
                trace.write(crate::shared::speculation_trace::record! { final_record: true }).unwrap();
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let mut shutdown = Box::pin(cuteafd_bench::shutdown_signal());
                // Poll once so both signal subscriptions precede the ready line.
                tokio::select! { biased; _ = &mut shutdown => panic!("premature signal"), _ = std::future::ready(()) => () }
                println!("TRACE_SERVER_READY {}", std::process::id());
                use std::io::Write;
                std::io::stdout().flush().unwrap();
                axum::serve(listener, axum::Router::new()).with_graceful_shutdown(shutdown).await.unwrap();
                drop(cache);
                drop(trace);
                crate::shared::speculation_trace::shutdown();
            });
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let mut child = std::process::Command::new("/usr/bin/timeout")
            .args(["--signal=KILL", "30"])
            .arg(std::env::current_exe().unwrap())
            .args(["--exact", "shared::decode_graph::device::tests::sigterm_http_shutdown_writes_complete_nonempty_final_traces", "--nocapture"])
            .env(CHILD, directory.path()).env("CUTEAFD_SPECULATION_TRACE", directory.path().join("speculation.jsonl"))
            .env_remove("CUTEAFD_GRAPH_TRACE")
            .stdout(std::process::Stdio::piped()).spawn().unwrap();
        use std::io::BufRead;
        let reader = std::io::BufReader::new(child.stdout.take().unwrap());
        let mut signalled = false;
        for line in reader.lines() {
            let line = line.unwrap();
            if let Some((_, pid)) = line.split_once("TRACE_SERVER_READY ") {
                assert!(std::process::Command::new("/bin/kill").args(["-TERM", pid.trim()]).status().unwrap().success());
                signalled = true;
            }
        }
        assert!(child.wait().unwrap().success());
        assert!(signalled);
        let graph: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.path().join("graph.json")).unwrap()).unwrap();
        assert_eq!(graph["graphs"].as_array().unwrap().len(), 1);
        assert_eq!(graph["graphs"][0]["hits"], 1);
        assert_eq!(std::fs::read(directory.path().join("speculation.jsonl")).unwrap(), b"{\"final_record\":true}\n");
    }

    #[test]
    fn trace_coalesces_busy_writer_and_drop_flushes_latest() {
        let (entered, waiting) = std::sync::mpsc::channel();
        let (release, gate) = std::sync::mpsc::channel();
        let written = Arc::new(Mutex::new(Vec::new()));
        let seen = written.clone();
        let mut first = true;
        let writer = TraceWriter::with_sink(move |snapshot| {
            if first { first = false; entered.send(()).unwrap(); gate.recv().unwrap(); }
            seen.lock().unwrap().push(snapshot.cache.charged);
        }).unwrap();
        let mut cache = DeviceCache::new(-700, 100);
        writer.submit(cache.trace_snapshot());
        waiting.recv_timeout(Duration::from_secs(5)).unwrap();
        for charged in 1..=10 { cache.charged = charged; writer.submit(cache.trace_snapshot()); }
        release.send(()).unwrap();
        drop(writer);
        assert_eq!(*written.lock().unwrap(), [0, 10]);
    }

    #[test]
    fn controller_drop_keeps_all_retired_banks_and_atomic_final_trace() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.json");
        let mut cache = DeviceCache::new(-701, 100);
        cache.trace = Some(TraceWriter::new(path.clone()).unwrap());
        let a = capture(&mut cache, 1, false);
        cache.hit(a);
        let mut identity = key(2); identity.bank = 2;
        assert!(matches!(cache.miss(identity, site(Preference::Capture), None, false), Miss::Capture(_)));
        cache.measured_capture(10);
        cache.remove_bank(1);
        cache.remove_bank(2);
        let owner = Arc::new(Mutex::new(cache));
        let weak = Arc::downgrade(&owner);
        DEVICES.get_or_init(Mutex::default).lock().unwrap().insert(-701, weak.clone());
        drop(owner);
        assert!(weak.upgrade().is_none());
        DEVICES.get().unwrap().lock().unwrap().remove(&-701);
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["graphs"].as_array().unwrap().len(), 2);
        assert_eq!(value["graphs"][0]["hits"], 1);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn explicit_shutdown_upgrades_live_weak_controllers_and_joins_final_trace() {
        const CHILD: &str = "CUTEAFD_TRACE_WEAK_SHUTDOWN_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            assert!(std::process::Command::new("/usr/bin/timeout")
                .args(["--signal=KILL", "30"])
                .arg(std::env::current_exe().unwrap())
                .args(["--exact", "shared::decode_graph::device::tests::explicit_shutdown_upgrades_live_weak_controllers_and_joins_final_trace"])
                .env(CHILD, "1").env_remove("CUTEAFD_GRAPH_TRACE").status().unwrap().success());
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.json");
        let mut cache = DeviceCache::new(-703, 100);
        cache.trace = Some(TraceWriter::new(path.clone()).unwrap());
        let id = capture(&mut cache, 1, false);
        cache.hit(id);
        cache.remove_bank(1);
        let owner = Arc::new(Mutex::new(cache));
        let weak = Arc::downgrade(&owner);
        {
            let mut registry = DEVICES.get_or_init(Mutex::default).lock().unwrap();
            registry.insert(-703, weak.clone());
            registry.insert(-704, std::sync::Weak::new());
        }
        shutdown_traces();
        assert!(owner.lock().unwrap().trace.is_none());
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["graphs"].as_array().unwrap().len(), 1);
        assert_eq!(value["graphs"][0]["hits"], 1);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        shutdown_traces();
        drop(owner);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn trace_schema_bytes_match_legacy_fixture() {
        let mut cache = DeviceCache::new(-702, 100);
        let a = capture(&mut cache, 1, false);
        cache.records.get_mut(&a).unwrap().stats.first_seen_ms = 0;
        cache.clock_ms = 20_000;
        cache.hit(a);
        let mut snapshot = cache.trace_snapshot();
        snapshot.cache.sampled_at_ms = 30_000;
        snapshot.cache.telemetry_bytes = 0;
        let bytes = snapshot.bytes().unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), include_str!("trace-schema.json").trim());
    }

    #[test]
    #[ignore = "CPU trace boundary microbenchmark; run explicitly with --nocapture"]
    fn trace_snapshot_microseconds() {
        use std::hint::black_box;
        let mut cache = DeviceCache::new(-703, 100_000);
        for n in 0..5_000 { let id = capture(&mut cache, n, false); cache.hit(id); }
        let repeats = 50;
        let start = Instant::now();
        for _ in 0..repeats { black_box(cache.trace_snapshot()); }
        let copy = start.elapsed().as_secs_f64() * 1e6 / repeats as f64;
        let start = Instant::now();
        for _ in 0..repeats {
            let mut rows: Vec<_> = cache.records.values().map(|r| cache.row(r, 600_000)).collect();
            rows.sort_by(|a, b| (a.identity.site, a.identity.bank, &a.identity.shape)
                .cmp(&(b.identity.site, b.identity.bank, &b.identity.shape)));
            let document = serde_json::json!({"schema_version": 2, "cache": cache.snapshot(), "graphs": rows,
                "sites": cache.classes.iter().map(|(name, class)| (*name, serde_json::json!({
                    "samples": class.samples, "total_bytes": class.total, "largest_bytes": class.largest
                }))).collect::<std::collections::BTreeMap<_, _>>()});
            black_box(serde_json::to_vec(&document).unwrap());
        }
        println!("graph trace 5000 records: legacy {:.1} us -> raw copy {:.1} us (no filesystem, {} iterations)",
            start.elapsed().as_secs_f64() * 1e6 / repeats as f64, copy, repeats);
    }

    #[test]
    fn reserve_policy_snapshot_distinguishes_admission_from_live_arena() {
        let policy = cuteafd_loader::placement::graph_reserve::resolve_with_extra(
            None, Some("50%"), Some("20MB"), 0, false, 100_000_000, 0).unwrap();
        let mut cache = DeviceCache::new(-901, policy.bytes);
        cache.reserve_policy = Some(policy);
        let snapshot = serde_json::to_value(cache.snapshot()).unwrap();
        assert_eq!(snapshot["reserve"], 120_000_000);
        assert_eq!(snapshot["charged"], 0); // live graph arena, not the admission charge
        let policy = &snapshot["reserve_policy"];
        assert_eq!(policy["base_bytes"], 100_000_000);
        assert_eq!(policy["extra_bytes"], 50_000_000);
        assert_eq!(policy["cap_bytes"], 20_000_000);
        assert_eq!(policy["charged_bytes"], 120_000_000);
        assert_eq!(snapshot["reserve"], policy["charged_bytes"]);
        assert_eq!(policy["charged_source"], "family default with extra");
        assert_eq!(policy["extra_source"], "user override");
    }
    #[test]
    fn per_key_churn_cost_and_forget_lifetime() {
        let mut cache = DeviceCache::new(-213, 100);
        let a = capture(&mut cache, 1, false);
        cache.captured_topology(a, 0, 1, false);
        cache.hit_cycle(a, Cycle::Verify);
        assert_eq!(cache.mark_victim(false), Some(a));
        cache.destroyed(&[a], ReleaseMeasurement { released: 0 });
        assert!(matches!(cache.miss(key(1), site(Preference::Capture), None, false), Miss::Capture(id) if id == a));
        cache.measured_capture(0);
        let row = cache.row(&cache.records[&a], 0);
        assert_eq!((row.hits, row.misses, row.captures), (1, 2, 2));
        assert_eq!((row.evictions, row.recaptures, row.ghost_hits), (1, 1, 1));
        assert_eq!(row.capture_bytes, Some(0));
        assert_eq!(row.permanent_bytes, 10, "reuse must not imply free permanent graph");
        assert_eq!(row.executables, 1);
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.top_churn.len(), 1);
        assert_eq!(snapshot.top_hits.len(), 1);
        assert_eq!(snapshot.telemetry_bytes, std::mem::size_of::<KeyStats>());
        cache.forget(a);
        assert_eq!(cache.snapshot().telemetry_bytes, 0);
        assert!(cache.snapshot().top_churn.is_empty());
    }

    #[test]
    fn boundary_cost_shares_sum_to_physical_growth_with_node_basis() {
        let mut cache = DeviceCache::new(-217, 64 << 20);
        cache.basis = Basis::Nodes { node_rate: 2560, graph_rate: 65536 };
        let mut node_site = site(Preference::Capture);
        node_site.basis = cache.basis;
        let mut ids = Vec::new();
        for n in [1, 2] {
            let estimate = Some(Estimate { bytes: n * 2560 + 65536, nodes: Some(n),
                node_rate: 2560, graph_rate: 65536, parallel_width: 1 });
            let Miss::Capture(id) = cache.miss(key(n), node_site, estimate, false) else { panic!() };
            ids.push(id);
        }
        cache.measured_capture(4 << 20);
        assert_eq!(ids.iter().map(|id| cache.records[id].stats.capture_bytes.unwrap()).sum::<u64>(), 4 << 20);
        let row = cache.row(&cache.records[&ids[1]], 0);
        assert_eq!(row.nodes, Some(2));
        assert_eq!(row.permanent_bytes, 2 * 2560 + 65536);
        assert_eq!(row.basis, cache.basis);
    }

    #[test]
    fn canceled_capture_and_cold_ghost_are_not_recaptures() {
        let mut cache = DeviceCache::new(-214, 20);
        capture(&mut cache, 1, true);
        assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
        assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
        let id = cache.identities[&key(2)];
        assert_eq!(cache.records[&id].stats.ghost_hits, 1);
        assert_eq!(cache.records[&id].stats.recaptures, 0);
        cache.remove_bank(1);
        let Miss::Capture(id) = cache.miss(key(3), site(Preference::Capture), None, false) else { panic!() };
        cache.cancel_capture(id);
        assert!(cache.snapshot().top_churn.is_empty());
        assert_eq!(cache.snapshot().telemetry_bytes, 0);
    }

    #[test]
    fn histogram_wraps_ages_idle_and_preserves_lifetime_hits() {
        let mut cache = DeviceCache::new(-215, 100);
        let id = capture(&mut cache, 1, false);
        let stats = &mut cache.records.get_mut(&id).unwrap().stats;
        stats.hit(Cycle::Decode, 3, 0);
        stats.hit(Cycle::Verify, 5, 20_000);
        let row = cache.row(&cache.records[&id], 30_000);
        assert_eq!(row.hit_histogram.counts, vec![3, 0, 5, 0]);
        let row = cache.row(&cache.records[&id], 620_000);
        assert_eq!(row.hit_histogram.counts, vec![0; HIT_INTERVALS]);
        assert!(row.hit_histogram.rolling);
        let stats = &mut cache.records.get_mut(&id).unwrap().stats;
        stats.hit(Cycle::Decode, 7, 630_000);
        let row = cache.row(&cache.records[&id], 630_000);
        assert_eq!(row.hit_histogram.counts.iter().sum::<u64>(), 7);
        assert_eq!(row.hits, 15);
        assert_eq!(row.last_hit_ms, Some(630_000));
        assert_eq!(row.hit_histogram.start_ms, 40_000);
    }

    #[test]
    fn arc_hits_use_cycle_clock_without_per_replay_sampling() {
        let mut cache = DeviceCache::new(-218, 100);
        let id = capture(&mut cache, 1, false);
        cache.clock_ms = 25_000;
        cache.hit_cycle(id, Cycle::Verify);
        cache.hit_cycle(id, Cycle::Verify);
        let row = cache.row(&cache.records[&id], 30_000);
        assert_eq!(row.last_hit_ms, Some(25_000));
        assert_eq!(row.hit_histogram.counts, vec![0, 0, 2, 0]);
        assert_eq!(row.cycles[&Cycle::Verify], 2);
    }

    #[test]
    #[ignore = "CPU telemetry attribution microbenchmark; run explicitly with --nocapture"]
    fn telemetry_nanoseconds() {
        use std::hint::black_box;
        let iterations = 1_000_000;
        let origin = Instant::now();
        let started = Instant::now();
        for _ in 0..iterations { black_box(origin.elapsed().as_millis() as u64); }
        println!("timestamp: {:.1} ns/update", started.elapsed().as_nanos() as f64 / iterations as f64);
        let mut stats = KeyStats::new(0);
        let started = Instant::now();
        for _ in 0..iterations { black_box(&mut stats).event(Cycle::Verify, black_box(1)); }
        println!("boxed_counters: {:.1} ns/update", started.elapsed().as_nanos() as f64 / iterations as f64);
        let started = Instant::now();
        for _ in 0..iterations { black_box(&mut stats).hit(Cycle::Verify, black_box(1), black_box(25_000)); }
        println!("boxed_counters_and_ring: {:.1} ns/update", started.elapsed().as_nanos() as f64 / iterations as f64);
    }

    #[test]
    fn stats_top_lists_are_bounded_sorted_and_available_without_trace() {
        let mut cache = DeviceCache::new(-216, 1000);
        assert!(cache.trace.is_none());
        for n in 0..30 {
            let id = capture(&mut cache, n, false);
            cache.pinned_hits(id, Cycle::Decode, n + 1);
        }
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.top_hits.len(), 16);
        assert_eq!(snapshot.top_hits[0].hits, 30);
        assert_eq!(snapshot.top_hits[15].hits, 15);
        while let Some(id) = cache.mark_victim(false) {
            cache.destroyed(&[id], ReleaseMeasurement { released: 0 });
        }
        assert_eq!(cache.snapshot().top_churn.len(), 16);
        cache.remove_bank(1);
        assert_eq!(cache.snapshot().telemetry_bytes, 0);
    }

    #[test]
    fn declared_peer_span_survives_single_device_native_inspection() {
        let mut cache = DeviceCache::new(-8803, 100);
        let mut peer_site = site(Preference::Capture);
        peer_site.spans_devices = true;
        let Miss::Capture(id) = cache.miss(key(1), peer_site, None, false) else { panic!("capture expected") };
        assert!(!cache.chain_only); // Admission already excludes peer reuse.
        // Isolate topology's independent protection against hidden peer pointers.
        cache.chain_only = true;
        cache.captured_topology(id, 1, 1, false);
        assert!(!cache.chain_only);
        assert!(cache.records[&id].spans_devices);
        assert!(cache.pending_growth >= cache.pending_bytes);
        cache.measured_capture(10);
        cache.captured_topology(id, 1, 1, false);
        assert!(!cache.chain_only);
    }

    #[test]
    fn cold_step_is_capped_and_unknown_site_bootstraps() {
        let mut cache = DeviceCache::new(-200, 100);
        let mut captures = 0;
        for n in 0..100 {
            if matches!(cache.miss(key(n), site(Preference::Adaptive), None, false), Miss::Capture(_)) { captures += 1; }
        }
        assert_eq!(captures, 9);
        cache.measured_capture(90);
        assert!(cache.charged <= cache.reserve + 10);
        assert_eq!(cache.snapshot().pending_capture, 0);
    }
    #[test]
    fn arc_frequency_and_ghost_adaptation() {
        let mut cache = DeviceCache::new(-201, 100);
        let a = capture(&mut cache, 1, false);
        let b = capture(&mut cache, 2, false);
        cache.hit(a);
        assert_eq!(cache.records[&a].list, Some(List::Frequent));
        assert_eq!(cache.mark_victim(false), Some(b));
        cache.destroyed(&[b], ReleaseMeasurement { released: 0 });
        assert_eq!(cache.records[&b].list, Some(List::RecentGhost));
        let before = cache.recent_target;
        assert!(matches!(cache.miss(key(2), site(Preference::Adaptive), None, false), Miss::Capture(_)));
        assert!(cache.recent_target > before);
        assert_eq!(cache.records[&b].list, Some(List::Frequent));
    }
    #[test]
    fn pinned_never_enters_arc_and_full_miss_never_refuses() {
        let mut cache = DeviceCache::new(-202, 20);
        let a = capture(&mut cache, 1, true);
        assert!(cache.records[&a].list.is_none());
        assert!(cache.mark_victim(false).is_none());
        assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
        assert_eq!(cache.snapshot().pinned, 10);
    }
    #[test]
    fn destruction_never_reduces_arena_and_both_pools_allow_reuse() {
        let mut cache = DeviceCache::new(-203, 20);
        let a = capture(&mut cache, 1, false);
        cache.mark_victim(false);
        cache.destroyed(&[a], ReleaseMeasurement { released: 999 });
        assert_eq!(cache.charged, 10);
        assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Capture(_)));
        cache.measured_capture(0);
        assert_eq!(cache.charged, 10);
        assert_eq!(cache.pools.exec_high_water, 1);
    }
    #[test]
    fn devices_only_charge_their_own_measured_growth() {
        let mut a = DeviceCache::new(-206, 100);
        let mut b = DeviceCache::new(-207, 100);
        assert!(matches!(a.miss(key(1), site(Preference::Capture), None, false), Miss::Capture(_)));
        a.measured_capture(10);
        b.measured_capture(0);
        assert_eq!((a.charged, b.charged), (10, 0));
        // A peer allocation observed on B is charged to B, not inferred from A.
        b.measured_capture(5);
        assert_eq!((a.charged, b.charged), (10, 5));
    }

    #[test]
    fn node_and_exec_pools_have_independent_peaks() {
        let mut cache = DeviceCache::new(-205, 64 << 20);
        cache.basis = Basis::Nodes { node_rate: 2560, graph_rate: 65536 };
        let mut node_site = site(Preference::Capture);
        node_site.basis = cache.basis;
        let estimate = |nodes| Some(Estimate { bytes: nodes * 2560 + 65536,
            nodes: Some(nodes), node_rate: 2560, graph_rate: 65536, parallel_width: 1 });
        for n in 0..8 {
            assert!(matches!(cache.miss(key(n), node_site, estimate(1), false), Miss::Capture(_)));
        }
        cache.measured_capture(4 << 20);
        let mut victims = Vec::new();
        while let Some(id) = cache.mark_victim(false) { victims.push(id); }
        cache.destroyed(&victims, ReleaseMeasurement { released: 0 });
        assert_eq!(cache.pools.exec_high_water, 8);
        assert_eq!(cache.pools.live_execs, 0);
        assert!(matches!(cache.miss(key(99), node_site, estimate(1024), false), Miss::Capture(_)));
        assert_eq!(cache.pools.node_high_water, 8);
        cache.measured_capture(2 << 20);
        assert_eq!(cache.pools.node_high_water, 1024);
        assert_eq!(cache.pools.exec_high_water, 8);
        assert_eq!(cache.charged, 6 << 20);
    }

    #[test]
    fn canceled_tickets_never_prove_pool_reuse() {
        let mut cache = DeviceCache::new(-208, 20);
        let Miss::Capture(id) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
        assert_eq!(cache.pools.node_high_water, 0);
        cache.cancel_capture(id);
        assert!(cache.needs_boundary());
        cache.measured_capture(10); // Failed instantiate may retain driver pages.
        assert_eq!((cache.pools.node_high_water, cache.pools.exec_high_water), (0, 0));
        assert_eq!(cache.pools.live_nodes, 0);
        assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
    }

    #[test]
    fn device_basis_and_coefficients_cannot_mix() {
        let mut cache = DeviceCache::new(-209, 64 << 20);
        let estimate = Some(Estimate { bytes: 1000, nodes: Some(10), node_rate: 2560,
            graph_rate: 65536, parallel_width: 1 });
        assert!(matches!(cache.miss(key(1), site(Preference::Capture), estimate.clone(), false), Miss::Eager));
        cache.basis = Basis::Nodes { node_rate: 2560, graph_rate: 65536 };
        let mut nodes = site(Preference::Capture);
        nodes.basis = cache.basis;
        assert!(matches!(cache.miss(key(2), nodes, None, false), Miss::Eager));
        assert!(matches!(cache.miss(key(3), nodes, estimate, false), Miss::Capture(_)));
        assert_eq!(cache.pools.node_high_water, 0);
        cache.measured_capture(4 << 20);
        assert_eq!(cache.pools.node_high_water, 10);
    }

    #[test]
    fn pending_node_pool_growth_is_reserved_once_per_chunk() {
        let mut cache = DeviceCache::new(-210, 5 << 20);
        cache.basis = Basis::Nodes { node_rate: 2560, graph_rate: 65536 };
        let mut nodes = site(Preference::Capture);
        nodes.basis = cache.basis;
        for n in 0..10 {
            let estimate = Some(Estimate { bytes: 68096, nodes: Some(1), node_rate: 2560,
                graph_rate: 65536, parallel_width: 1 });
            assert!(matches!(cache.miss(key(n), nodes, estimate, false), Miss::Capture(_)));
        }
        assert_eq!(cache.pending_growth, 4 << 20);
        cache.measured_capture(4 << 20);
        assert_eq!((cache.pools.node_high_water, cache.pools.exec_high_water), (10, 10));
    }

    #[test]
    fn unknown_topology_restores_full_pending_reservation_and_forbids_reuse() {
        let mut cache = DeviceCache::new(-212, 100);
        cache.pools.node_high_water = 1000;
        cache.pools.exec_high_water = 100;
        cache.charged = 60;
        let Miss::Capture(id) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
        assert_eq!(cache.pending_growth, 0);
        cache.captured_topology(id, 0, 0, false);
        assert!(!cache.chain_only);
        assert_eq!(cache.pending_growth, 10);
        assert_eq!(cache.records[&id].nodes, 10);
        for n in 2..4 {
            assert!(matches!(cache.miss(key(n), site(Preference::Capture), None, false), Miss::Capture(_)));
        }
        assert_eq!(cache.pending_growth, 30);
        assert!(matches!(cache.miss(key(4), site(Preference::Capture), None, false), Miss::Eager));
    }

    #[test]
    fn non_chain_and_declared_peer_sites_reserve_each_capture_below_old_peaks() {
        for peer in [false, true] {
            let mut cache = DeviceCache::new(-211, 100);
            cache.pools.node_high_water = 1000;
            cache.pools.exec_high_water = 100;
            cache.charged = 60;
            cache.chain_only = peer;
            let mut declared = site(Preference::Capture);
            declared.spans_devices = peer;
            for n in 0..3 {
                assert!(matches!(cache.miss(key(n), declared, None, false), Miss::Capture(_)));
            }
            assert_eq!(cache.pending_growth, 30);
            assert!(matches!(cache.miss(key(4), declared, None, false), Miss::Eager));
        }
    }

    #[test]
    fn stats_report_only_when_evicted_and_trace_schema() {
        let mut cache = DeviceCache::new(-204, 100);
        assert!(!cache.report_due(Duration::from_secs(90)));
        let a = capture(&mut cache, 1, false);
        cache.mark_victim(false);
        cache.destroyed(&[a], ReleaseMeasurement { released: 0 });
        assert!(!cache.report_due(Duration::from_secs(1)));
        assert!(cache.report_due(Duration::from_secs(30)));
        assert!(!cache.report_due(Duration::from_secs(30)));
        cache.hit_cycle(a, Cycle::Decode);
        cache.hit_cycle(a, Cycle::Verify);
        let row = serde_json::to_value(cache.row(&cache.records[&a], 0)).unwrap();
        assert_eq!(row["site"], "test"); assert_eq!(row["count"], 3);
        assert_eq!(row["cycles"]["decode"], 2);
        assert_eq!(row["cycles"]["verify"], 1);
        assert!(row["first_seen_ms"].is_u64());
    }
}

#[cfg(test)]
mod observer_tests {
    use super::*;
    fn attached(device: i32) -> SharedCache {
        device_cache_sampled(device, 1000, Basis::Bytes, || Ok((900, 1000, 0, 0, 0))).unwrap()
    }

    fn pending(cache: &SharedCache) {
        let mut cache = cache.lock().unwrap();
        let site = Site { name: "observer-fixture", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 10, basis: Basis::Bytes,
            spans_devices: false };
        let key = Identity { bank: 1, site: site.name, key: 1, shape: "1".into() };
        assert!(matches!(cache.miss(key, site, None, false), Miss::Capture(_)));
    }

    #[test]
    fn attachment_measures_capture_before_explicit_observe() {
        let cache = attached(-99101);
        pending(&cache);
        observe_device_sampled(-99101, || Ok((875, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
    }

    #[test]
    fn unchanged_boundary_does_not_sample() {
        let cache = attached(-99111);
        observe_device_sampled(-99111, || panic!("unchanged cache must not sample")).unwrap();
        pending(&cache);
        observe_device_sampled(-99111, || Ok((875, 1000, 0, 0, 0))).unwrap();
        observe_device_sampled(-99111, || panic!("measured capture must not resample")).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
    }

    #[test]
    fn pending_destroy_still_samples_without_a_capture() {
        let cache = attached(-99108);
        cache.lock().unwrap().pending_destroy = 1;
        let sampled = std::cell::Cell::new(false);
        observe_device_sampled(-99108, || { sampled.set(true); Ok((900, 1000, 0, 0, 0)) }).unwrap();
        assert!(sampled.get());
        assert!(cache.lock().unwrap().has_pending_victims());
        cache.lock().unwrap().pending_destroy = 0;
        observe_device_sampled(-99108, || panic!("retired victims must not resample")).unwrap();
    }

    #[test]
    fn capture_baseline_refreshes_once_per_observed_batch() {
        let cache = attached(-99109);
        cache.lock().unwrap().capture_baseline_sampled(|| Ok((800, 1000, 0, 0, 0))).unwrap();
        pending(&cache);
        cache.lock().unwrap().capture_baseline_sampled(|| panic!("must retain earlier capture growth")).unwrap();
        observe_device_sampled(-99109, || Ok((775, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
        cache.lock().unwrap().capture_baseline_sampled(|| Ok((700, 1000, 0, 0, 0))).unwrap();
        pending(&cache);
        observe_device_sampled(-99109, || Ok((675, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 50);
    }

    #[test]
    fn physical_floor_policy_reinstall_is_idempotent() {
        let policy = OnceLock::new();
        install_physical_floor(&policy, Some("50MiB".into())).unwrap();
        install_physical_floor(&policy, Some("50MiB".into())).unwrap();
        assert!(install_physical_floor(&policy, Some("0".into())).is_err());
    }

    #[test]
    fn monitor_sample_recovers_stuck_floor_and_ignores_older_reports() {
        let device = -99125;
        let cache = device_cache_sampled(device, 1000, Basis::Bytes,
            || Ok((5, 1000, 0, 0, 0))).unwrap();
        cache.lock().unwrap().physical_floor = 50;
        let site = Site { name: "floor-monitor-fixture", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 10, basis: Basis::Bytes, spans_devices: false };
        let key = Identity { bank: 1, site: site.name, key: 1, shape: "1".into() };
        assert!(matches!(cache.lock().unwrap().miss(key.clone(), site, None, false), Miss::Eager));
        let stale = Instant::now();
        observe_device_sampled(device, || Ok((5, 1000, 0, 0, 0))).unwrap();
        monitor_physical(device, stale, 1000, 0);
        assert!(matches!(cache.lock().unwrap().miss(key.clone(), site, None, false), Miss::Eager));
        assert!(!cache.lock().unwrap().needs_boundary());
        let (level, charged) = { let cache = cache.lock().unwrap(); (cache.level, cache.charged) };
        monitor_physical(device, Instant::now(), 1000, 0);
        let mut cache = cache.lock().unwrap();
        assert_eq!((cache.level, cache.charged), (level, charged));
        assert!(cache.floor_resample_used, "monitor must not reset bounded refresh or OOM state");
        assert!(matches!(cache.miss(key, site, None, false), Miss::Capture(_)));
    }

    #[test]
    fn refreshed_capture_baseline_must_also_honor_the_floor() -> anyhow::Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory_usage(&library, 6, 1000, 995)?;
        let mut cache = DeviceCache::new(6, 1000);
        cache.physical_floor = 50;
        cache.sample_physical(1000, 0);
        cache.memory_query = Some(library.physical_memory_query()?);
        let site = Site { name: "floor-refresh-fixture", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 10, basis: Basis::Bytes, spans_devices: false };
        let key = Identity { bank: 1, site: site.name, key: 1, shape: "1".into() };
        let before = fixture.memory_queries(&library)?;
        assert!(matches!(cache.miss(key, site, None, false), Miss::Eager));
        assert_eq!(fixture.memory_queries(&library)?, before + 1);
        assert_eq!((cache.next_id, cache.pending_bytes, cache.pending_growth), (0, 0, 0));
        assert!(!cache.unmeasured);
        assert!(!cache.captured_since_observation);
        Ok(())
    }

    #[test]
    fn physical_floor_skips_without_ticket_query_or_cooldown() -> anyhow::Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory_usage(&library, 6, 1000, 995)?;
        let mut cache = DeviceCache::new(6, 1000);
        cache.physical_floor = 50;
        cache.sample_physical(5, 0);
        cache.memory_query = Some(library.physical_memory_query()?);
        let site = Site { name: "floor-fixture", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 10, basis: Basis::Bytes, spans_devices: true };
        let key = Identity { bank: 1, site: site.name, key: 1, shape: "1".into() };
        let before = fixture.memory_queries(&library)?;
        for _ in 0..4 {
            assert!(matches!(cache.miss(key.clone(), site, None, false), Miss::Eager));
        }
        assert_eq!(fixture.memory_queries(&library)?, before);
        assert_eq!((cache.next_id, cache.pending_bytes, cache.pending_growth), (0, 0, 0));
        assert!(cache.records.is_empty());
        assert!(!cache.unmeasured);
        assert!(cache.floor_resample_pending);
        let stats = serde_json::to_value(cache.snapshot())?;
        assert_eq!(stats["cycles"]["decode"]["physical_floor_skips"], 4);
        assert_eq!(stats["cycles"]["decode"]["eager_runs"], 4);
        // Adequate headroom permits the same key immediately: no OOM cooldown.
        fixture.configure_memory_usage(&library, 6, 1000, 100)?;
        cache.sample_physical(900, 0);
        assert!(matches!(cache.miss(key, site, None, false), Miss::Capture(_)));
        assert_eq!(fixture.memory_queries(&library)?, before + 1, "only the existing baseline query");
        Ok(())
    }

    #[test]
    fn physical_floor_honors_estimate_headroom_pending_growth_and_disable() {
        let site = Site { name: "floor-size-fixture", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 10, basis: Basis::Bytes, spans_devices: true };
        let key = |key| Identity { bank: 1, site: site.name, key, shape: "floor".into() };
        let mut cache = DeviceCache::new(-99122, 10_000);
        cache.physical_floor = 50;
        cache.sample_physical(250, 0);
        let large = Estimate { bytes: 101, nodes: None, node_rate: 1, graph_rate: 0, parallel_width: 1 };
        assert!(matches!(cache.miss(key(1), site, Some(large), false), Miss::Eager));
        assert!(matches!(cache.miss(key(1), site, None, false), Miss::Capture(_)));
        assert_eq!(cache.pending_growth, 10);
        cache.sample_physical(79, 0);
        assert!(matches!(cache.miss(key(2), site, None, false), Miss::Eager));
        cache.physical_floor = 0;
        cache.sample_physical(0, 0);
        assert!(matches!(cache.miss(key(2), site, None, false), Miss::Capture(_)));
    }

    #[test]
    fn physical_floor_subtracts_ledger_growth_without_crediting_frees() {
        use cuteafd_ffi::memory_ledger::{self, MemoryKind, Space};
        let device = -99123;
        let mut cache = DeviceCache::new(device, 1000);
        cache.physical_floor = 50;
        cache.sample_physical(100, 0);
        let tag = memory_ledger::tag(MemoryKind::Workspace, "physical-floor-test");
        let ptr = 0x173100;
        memory_ledger::record_tagged(Space::Device, device, ptr, 40, tag);
        assert!(!cache.physical_room(10, 10));
        memory_ledger::record_released(ptr);
        assert!(cache.physical_room(10, 10));
        cache.sample_physical(60, 40);
        assert!(!cache.physical_room(10, 10), "unobserved frees cannot inflate physical free");
    }

    #[test]
    fn physical_floor_requests_only_one_boundary_sample_until_state_changes() {
        let device = -99124;
        let cache = device_cache_sampled(device, 1000, Basis::Bytes,
            || Ok((5, 1000, 0, 0, 0))).unwrap();
        cache.lock().unwrap().physical_floor = 50;
        let site = Site { name: "floor-boundary-fixture", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 10, basis: Basis::Bytes, spans_devices: false };
        let key = |key| Identity { bank: 1, site: site.name, key, shape: "floor".into() };
        assert!(matches!(cache.lock().unwrap().miss(key(1), site, None, false), Miss::Eager));
        observe_device_sampled(device, || Ok((5, 1000, 0, 0, 0))).unwrap();
        for _ in 0..4 {
            assert!(matches!(cache.lock().unwrap().miss(key(1), site, None, false), Miss::Eager));
            observe_device_sampled(device, || panic!("stuck floor must not sample each step")).unwrap();
        }
        cache.lock().unwrap().sample_physical(1000, 0);
        let Miss::Capture(id) = cache.lock().unwrap().miss(key(1), site, None, false) else { panic!() };
        observe_device_sampled(device, || Ok((5, 1000, 0, 0, 0))).unwrap();
        assert!(matches!(cache.lock().unwrap().miss(key(2), site, None, false), Miss::Eager));
        assert!(cache.lock().unwrap().needs_boundary(), "new capture resets the resample allowance");
        observe_device_sampled(device, || Ok((5, 1000, 0, 0, 0))).unwrap();
        cache.lock().unwrap().destroyed(&[id], ReleaseMeasurement { released: 0 });
        assert!(matches!(cache.lock().unwrap().miss(key(2), site, None, false), Miss::Eager));
        assert!(cache.lock().unwrap().needs_boundary(), "retirement resets the allowance");
    }

    #[test]
    fn admitted_miss_refreshes_baseline_and_failed_query_runs_eagerly() -> anyhow::Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory_usage(&library, 6, 1000, 100)?;
        let mut cache = DeviceCache::new(6, 1000);
        cache.observer.observe(900, 1000, 0, 0, 0);
        cache.memory_query = Some(library.physical_memory_query()?);
        let site = Site { name: "observer-miss-fixture", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 10, basis: Basis::Bytes,
            spans_devices: false };
        let identity = |key| Identity { bank: 1, site: site.name, key, shape: key.to_string().into() };
        fixture.configure_memory_usage(&library, 6, 1000, 200)?;
        assert!(matches!(cache.miss(identity(1), site, None, false), Miss::Capture(_)));
        // An invalid second sample proves another capture retains the first baseline.
        fixture.configure_memory(&library, 6, 0)?;
        assert!(matches!(cache.miss(identity(2), site, None, false), Miss::Capture(_)));
        let growth = cache.observer.observe(750, 1000, 0, 0, 0);
        cache.measured_capture(growth.0);
        assert_eq!(cache.charged, 50);
        assert!(!cache.captured_since_observation);
        assert!(matches!(cache.miss(identity(3), site, None, false), Miss::Eager));
        assert!(!cache.captured_since_observation);
        assert!(!cache.unmeasured);
        fixture.configure_memory_usage(&library, 6, 1000, 300)?;
        assert!(matches!(cache.miss(identity(3), site, None, false), Miss::Capture(_)));
        let growth = cache.observer.observe(675, 1000, 0, 0, 0);
        cache.measured_capture(growth.0);
        assert_eq!(cache.charged, 75);
        assert_eq!(cache.counters[&Cycle::Decode].eager_runs, 1);
        Ok(())
    }

    #[test]
    fn oom_backoff_uses_settled_level_refresh_without_resampling_suppressed_misses() {
        let device = -99118;
        let total = 64 << 20;
        let cache = device_cache_sampled(device, 64 << 20, Basis::Bytes,
            || Ok((total, total, 0, 0, 0))).unwrap();
        pending(&cache);
        observe_device_sampled(device, || Ok((total - (8 << 20), total, 0, 0, 0))).unwrap();
        let site = Site { name: "observer-fixture", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 10, basis: Basis::Bytes, spans_devices: false };
        let key = Identity { bank: 1, site: site.name, key: 2, shape: "2".into() };
        {
            let mut cache = cache.lock().unwrap();
            let Miss::Capture(id) = cache.miss(key.clone(), site, None, false) else { panic!() };
            cache.capture_oom(id);
        }
        observe_device_sampled(device, || Ok((total - (8 << 20), total, 0, 0, 0))).unwrap();
        for _ in 0..3 {
            assert!(matches!(cache.lock().unwrap().miss(key.clone(), site, None, false), Miss::Eager));
            observe_device_sampled(device, || panic!("OOM cooldown must not add a device query")).unwrap();
        }
        let peak = 8 << 20;
        let below_threshold = peak - OOM_LEVEL_DROP + 1;
        assert_eq!(refreshed_arena_sampled(device, || Ok((total - below_threshold, total, 0, 0, 0))), Some((below_threshold, peak)));
        assert!(matches!(cache.lock().unwrap().miss(key.clone(), site, None, false), Miss::Eager));
        let recovered = peak - OOM_LEVEL_DROP;
        assert_eq!(refreshed_arena_sampled(device, || Ok((total - recovered, total, 0, 0, 0))), Some((recovered, peak)));
        assert!(matches!(cache.lock().unwrap().miss(key, site, None, false), Miss::Capture(_)));
    }

    #[test]
    fn failed_capture_baseline_can_retry_without_advancing_observer() {
        let cache = attached(-99110);
        assert!(cache.lock().unwrap().capture_baseline_sampled(|| anyhow::bail!("query failed")).is_err());
        assert!(!cache.lock().unwrap().captured_since_observation);
        cache.lock().unwrap().capture_baseline_sampled(|| Ok((800, 1000, 0, 0, 0))).unwrap();
        pending(&cache);
        observe_device_sampled(-99110, || Ok((775, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
    }

    #[test]
    fn explicit_observe_excludes_eager_first_touch() {
        let cache = attached(-99102);
        observe_device_sampled(-99102, || panic!("eager-only boundary must not sample")).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 0);
        cache.lock().unwrap().capture_baseline_sampled(|| Ok((800, 1000, 0, 0, 0))).unwrap();
        pending(&cache);
        observe_device_sampled(-99102, || Ok((775, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
    }

    #[test]
    fn explicit_observe_does_not_double_seed_or_charge() {
        let cache = attached(-99103);
        observe_device_sampled(-99103, || Ok((900, 1000, 0, 0, 0))).unwrap();
        pending(&cache);
        observe_device_sampled(-99103, || Ok((875, 1000, 0, 0, 0))).unwrap();
        observe_device_sampled(-99103, || Ok((875, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
    }

    #[test]
    fn repeated_attachment_neither_samples_nor_advances_baseline() {
        let cache = attached(-99104);
        pending(&cache);
        let again = device_cache_sampled(-99104, 1000, Basis::Bytes,
            || panic!("reattachment must not consume graph growth")).unwrap();
        assert!(Arc::ptr_eq(&cache, &again));
        observe_device_sampled(-99104, || Ok((875, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
        device_cache_sampled(-99104, 1000, Basis::Bytes,
            || panic!("reattachment must not reset the explicit baseline")).unwrap();
        observe_device_sampled(-99104, || Ok((875, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
    }

    #[test]
    fn failed_attachment_sample_does_not_publish_cache() {
        let result = device_cache_sampled(-99105, 1000, Basis::Bytes,
            || anyhow::bail!("sample unavailable"));
        assert!(result.is_err());
        assert!(existing_cache(-99105).is_none());
        let cache = attached(-99105);
        pending(&cache);
        observe_device_sampled(-99105, || Ok((875, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
    }

    #[test]
    fn reattachment_after_last_owner_drops_seeds_a_fresh_controller() {
        let cache = attached(-99106);
        let old = Arc::downgrade(&cache);
        pending(&cache);
        observe_device_sampled(-99106, || Ok((500, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 400);
        drop(cache);
        assert!(old.upgrade().is_none());
        assert!(existing_cache(-99106).is_none());
        assert!(!snapshots().iter().any(|row| row.device == -99106));

        // A new lifetime may have a different reserve and lower residual.
        let cache = device_cache_sampled(-99106, 2000, Basis::Bytes,
            || Ok((950, 1000, 0, 0, 0))).unwrap();
        pending(&cache);
        observe_device_sampled(-99106, || Ok((925, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
    }

    #[test]
    fn temporary_owner_drop_preserves_another_live_owners_baseline() {
        let owner = attached(-99107);
        let temporary = device_cache_sampled(-99107, 1000, Basis::Bytes,
            || panic!("a live controller must not be reseeded")).unwrap();
        pending(&owner);
        drop(temporary);
        let rebuilt = device_cache_sampled(-99107, 1000, Basis::Bytes,
            || panic!("lane rebuild must retain the other owner's baseline")).unwrap();
        assert!(Arc::ptr_eq(&owner, &rebuilt));
        observe_device_sampled(-99107, || Ok((875, 1000, 0, 0, 0))).unwrap();
        assert_eq!(rebuilt.lock().unwrap().snapshot().charged, 25);
        drop(owner);
        let again = device_cache_sampled(-99107, 1000, Basis::Bytes,
            || panic!("the rebuilt owner still pins the controller")).unwrap();
        assert!(Arc::ptr_eq(&rebuilt, &again));
    }

    #[test]
    fn kimi_startup_owners_survive_lane_attachment_and_keep_physical_high_water() {
        use crate::shared::decode_graph::GraphBank;
        let devices = [-99113, -99114];
        let controllers = devices.map(attached);
        let weak = controllers.each_ref().map(Arc::downgrade);
        let mut lanes = Vec::new();
        for (rank, &device) in devices.iter().enumerate() {
            let mut lane = GraphBank::<u64, ()>::new();
            let site = Site { name: "kimi_k3.step", cycle: Cycle::Other,
                preference: Preference::Adaptive, bootstrap: 10, basis: Basis::Bytes,
                spans_devices: true };
            lane.attach_device_for_test(device, 1000, site).unwrap();
            assert!(matches!(lane.cache_miss(&1, true, None), Miss::Capture(_)));
            observe_device_sampled(device, || Ok((875, 1000, 0, 0, 0))).unwrap();
            assert_eq!(controllers[rank].lock().unwrap().snapshot().charged, 25);
            lanes.push(lane);
        }
        drop(controllers);
        for (rank, &device) in devices.iter().enumerate() {
            let cache = device_cache_sampled(device, 1000, Basis::Bytes,
                || panic!("Kimi lane attachment must retain the startup seed")).unwrap();
            assert!(Arc::ptr_eq(&cache, &weak[rank].upgrade().unwrap()));
            assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
            observe_device_sampled(device, || Ok((875, 1000, 0, 0, 0))).unwrap();
            assert_eq!(cache.lock().unwrap().snapshot().charged, 25);
            pending(&cache);
            observe_device_sampled(device, || Ok((865, 1000, 0, 0, 0))).unwrap();
            assert_eq!(cache.lock().unwrap().snapshot().charged, 35);
        }
        drop(lanes);
        assert!(weak.iter().all(|cache| cache.upgrade().is_none()));
    }

    #[test]
    fn seeded_startup_arena_is_nonzero_and_equals_capture_shares() {
        for device in [-8810, -8811] {
            let mut cache = DeviceCache::new(device, 64 << 20);
            assert_eq!(cache.observer.observe(900 << 20, 1000 << 20, 100 << 20, 0, 0).0, 0);
            let site = Site { name: "kimi_k3.step", cycle: Cycle::Other,
                preference: Preference::Adaptive, bootstrap: 128 << 10,
                basis: Basis::Bytes, spans_devices: true };
            for key in 1..=3 {
                let identity = Identity { bank: 1, site: site.name, key,
                    shape: format!("startup-{key}").into() };
                assert!(matches!(cache.miss(identity, site, None, true), Miss::Capture(_)));
            }
            let growth = cache.observer.observe(896 << 20, 1000 << 20, 100 << 20, 0, 0).0;
            cache.measured_capture(growth);
            assert_eq!(cache.charged, 4 << 20);
            assert!(cache.charged > 0);
            assert_eq!(cache.records.values().map(|record|
                record.stats.capture_bytes.expect("startup capture measured")).sum::<u64>(), cache.charged);
            assert_eq!(cache.snapshot().cycles[&Cycle::Other].captures, 3);
        }
    }

    #[test]
    fn lazy_tracked_workspaces_do_not_become_graph_growth() {
        let mut observer = GrowthObserver::default();
        assert_eq!(observer.observe(900, 1000, 100, 0, 0), (0, 0));
        assert_eq!(observer.observe(590, 1000, 400, 0, 0), (10, 10));
        assert_eq!(observer.observe(590, 1000, 400, 0, 0), (0, 10), "second lane cannot charge same capture twice");
        assert_eq!(observer.observe(290, 1000, 700, 0, 0), (0, 10), "lazy workspace is tracked at each sample");
        assert_eq!(observer.observe(280, 1000, 700, 0, 0), (10, 20));
        assert_eq!(observer.observe(590, 1000, 400, 0, 0), (0, 10));
        assert_eq!(observer.observe(580, 1000, 400, 0, 0), (0, 20), "returning physical high-water is not new growth");
        assert_eq!(observer.observe(900, 1000, 100, 0, 0), (0, 0), "the level falls back with the residual");
    }

    #[test]
    fn an_owners_granule_padding_is_not_graph_growth() {
        let mut observer = GrowthObserver::default();
        // 100 B of untracked bytes seed the attach baseline.
        assert_eq!(observer.observe(900, 1000, 0, 0, 0), (0, 0));
        // A tracked owner requests 100 B and receives a full granule: physical
        // use rises by the request plus its padding, both reported by the
        // ledger. Without removing the tail the padding would read as growth.
        assert_eq!(observer.observe(739, 1000, 100, 61, 0), (0, 0), "rounding belongs to its owner");
        // A real capture still counts once the ledger's rounding is removed.
        assert_eq!(observer.observe(689, 1000, 100, 61, 0), (50, 50), "untracked growth above the seed is graph growth");
    }

    #[test]
    fn granule_padding_at_a_refresh_does_not_charge_the_graph_level() {
        let cache = attached(-99115);
        pending(&cache);
        observe_device_sampled(-99115, || Ok((739, 1000, 100, 61, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().charged, 0);
        let arena = refreshed_arena_sampled(-99115, || Ok((739, 1000, 100, 61, 0))).unwrap();
        assert_eq!(arena, (0, 0), "padding at a settled sample is not level");
    }

    #[test]
    fn transient_peak_leaves_the_level_at_baseline_while_charge_keeps_the_peak() {
        let cache = attached(-99116);
        pending(&cache);
        observe_device_sampled(-99116, || Ok((800, 1000, 0, 0, 0))).unwrap();
        let peak = cache.lock().unwrap().snapshot();
        assert_eq!(peak.charged, 100);
        assert_eq!(peak.level, Some(100));
        // The graph-instantiate/upload transient reverses within the request. No
        // boundary work is pending, so the settle refresh is what re-reads it:
        // an unchanged boundary never samples.
        let arena = refreshed_arena_sampled(-99116, || Ok((900, 1000, 0, 0, 0))).unwrap();
        assert_eq!(arena, (0, 100));
        let settled = cache.lock().unwrap().snapshot();
        assert_eq!(settled.level, Some(0), "the current level returns to the attach baseline");
        assert_eq!(settled.charged, 100, "the admitted charge keeps covering the transient peak");
        let row = serde_json::to_value(settled).unwrap();
        assert_eq!(row["level"], 0);
        assert_eq!(row["charged"], 100);
    }

    #[test]
    fn reconcile_refresh_reads_the_settled_level_without_releasing_the_charge() {
        let cache = attached(-99117);
        pending(&cache);
        // A capture boundary catches the instantiate/upload peak.
        observe_device_sampled(-99117, || Ok((800, 1000, 0, 0, 0))).unwrap();
        assert_eq!(cache.lock().unwrap().snapshot().level, Some(100));
        // Reconciliation samples again once the stage has settled, with no
        // boundary observation in between.
        let arena = refreshed_arena_sampled(-99117, || Ok((900, 1000, 0, 0, 0))).unwrap();
        assert_eq!(arena, (0, 100), "the level is current; the charge keeps the peak");
        let snapshot = cache.lock().unwrap().snapshot();
        assert_eq!(snapshot.level, Some(0));
        assert_eq!(snapshot.charged, 100);
        let row = serde_json::to_value(snapshot).unwrap();
        assert_eq!(row["level"], 0);
        assert_eq!(row["charged"], 100);
        // A live non-graph untracked allocation at the sample cannot inflate the
        // graph level past the admitted high-water.
        let bounded = refreshed_arena_sampled(-99117, || Ok((600, 1000, 0, 0, 0))).unwrap();
        assert_eq!(bounded, (100, 100), "the level never exceeds the charge");
        let snapshot = cache.lock().unwrap().snapshot();
        assert_eq!(snapshot.level, Some(100));
        assert_eq!(snapshot.charged, 100);
    }

    #[test]
    fn a_measured_owner_created_after_the_seed_is_not_graph_growth() {
        let mut observer = GrowthObserver::default();
        // No measured owners at attach: 100 KiB of untracked bytes seed the baseline.
        assert_eq!(observer.observe(900_000, 1_000_000, 0, 0, 0), (0, 0));
        // A mapped pinned host pool is measured at construction: 64 KiB of
        // device page tables that no allocation row claims. Physical use rises
        // by exactly that, and the ledger names it.
        assert_eq!(observer.observe(834_464, 1_000_000, 0, 0, 65_536), (0, 0),
            "a measured owner's page tables belong to that owner");
        // A capture above the seed still counts once the measured owner is removed.
        assert_eq!(observer.observe(834_444, 1_000_000, 0, 0, 65_536), (20, 20));
    }

    #[test]
    fn a_measured_owner_present_at_the_seed_cancels_and_never_reaches_the_level() {
        let mut observer = GrowthObserver::default();
        // The pool predates the graph controller: it is part of the attach
        // residual, so it must not be removed twice.
        assert_eq!(observer.observe(900, 1000, 0, 0, 64), (0, 0));
        let arena = observer.current(900, 1000, 0, 0, 64);
        assert_eq!(arena, 0);
        // Its retirement leaves the residual without it and never lowers the level below zero.
        assert_eq!(observer.observe(964, 1000, 0, 0, 0).1, 0);
    }

    #[test]
    fn a_measured_owner_at_a_refresh_is_not_level() {
        let cache = attached(-99112);
        pending(&cache);
        observe_device_sampled(-99112, || Ok((900, 1000, 0, 0, 64))).unwrap();
        let arena = refreshed_arena_sampled(-99112, || Ok((900, 1000, 0, 0, 64))).unwrap();
        assert_eq!(arena, (0, 0), "a measured owner at a settled sample is not level");
    }
}
