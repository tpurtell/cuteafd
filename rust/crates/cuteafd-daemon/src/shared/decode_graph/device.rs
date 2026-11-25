//! Device-wide ARC policy. Native owners stay in GraphBank: this controller
//! marks victims; banks with persistent captured storage destroy them without
//! synchronization. Per-graph storage requires explicit drained retirement.
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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

fn is_zero_u64(value: &u64) -> bool { *value == 0 }

#[derive(Default, Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Counters {
    pub hits: u64,
    pub misses: u64,
    pub captures: u64,
    pub segments_launched: u64,
    pub eager_runs: u64,
    pub evictions: u64,
    pub physical_floor_skips: u64,
    pub eager_vampire: u64,
    pub hot_guard_skips: u64,
    pub updates_soft: u64,
    pub updates_hard: u64,
    /// Cumulative wall-clock nanoseconds spent in successful cold-donor updates
    /// (from update ticket acquisition through host source capture,
    /// `cudaGraphExecUpdate` and donor rekey).
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub update_ns: u64,
    /// Cumulative wall-clock nanoseconds spent inside the driver
    /// `cudaGraphExecUpdate` call of successful donor updates.
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub update_exec_ns: u64,
    pub donor_rekeys: u64,
    pub update_failures: u64,
    pub no_cold_donor_soft: u64,
    pub no_cold_donor_hard: u64,
    pub eager_projected_hard_stop: u64,
    pub eager_soft_room: u64,
    pub eager_hard_stop: u64,
    pub eager_sealed: u64,
    pub eager_preference: u64,
    pub eager_other: u64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub device: i32,
    pub reserve: u64,
    pub physical_floor: u64,
    pub reserve_policy: Option<cuteafd_loader::placement::graph_reserve::Resolved>,
    pub charged: u64,
    /// Measured growth shares held by live executables, excluding estimates and
    /// ghosts. A lower bound after allocator reuse, not physical arena residency.
    pub live_bytes: u64,
    /// Sum of ARC ordering estimates, NOT physical residency. Bootstrap slab
    /// estimates can far exceed the arena when many owners share its backing.
    pub resident_estimated_bytes: u64,
    pub hard_stopped: bool,
    pub native_pools: NativePools,
    pub soft: SoftState,
    /// Current untracked level relative to the attach seed. Unlike `charged`
    /// (the physical arena high-water retained for admission), this can
    /// fall, so a transient peak that reverses releases its credit.
    pub level: Option<u64>,
    pub pinned: u64,
    /// Unevictable spanning graphs: measured growth shares, not a physical split.
    pub spanning_bytes: u64,
    pub spanning_nodes: u64,
    pub spanning_execs: u64,
    pub pending_capture: u64,
    #[serde(serialize_with = "serialize_cycles")]
    pub cycles: HashMap<Cycle, Counters>,
    pub pools: Pools,
    pub basis: Basis,
    pub sampled_at_ms: u64,
    pub telemetry_bytes: usize,
    pub top_hits: Vec<TraceRow>,
    pub top_churn: Vec<TraceRow>,
    pub rankings_age_ms: u64,
    pub epochs: EpochStats,
    pub per_cycle: std::collections::BTreeMap<&'static str, HashMap<Cycle, PerCycle>>,
    pub update_sites: std::collections::BTreeMap<&'static str, HashMap<Cycle, Counters>>,
}

/// Rates use the site's registered owner-lane completed cycles, not GPU polls
/// or wall time. Totals allow exact deltas between trace snapshots.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct PerCycle {
    owner_cycles: u64,
    hits: f64,
    misses: f64,
    segments_launched: f64,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub(crate) struct EpochStats {
    pub cycles: [u64; 2],
    pub completed: [u64; 2],
    pub selections: [u64; 2],
    pub pending: [usize; 2],
    pub lagged_chunks: usize,
    pub uncredited_nodes: u64,
    pub reuse_growth_bytes: u64,
    pub event_pending_cycles: u64,
    pub boundary_drains: u64,
    pub boundary_wait_us: u64,
}

#[derive(Default)]
struct LaneEpoch {
    index: u64,
    completed: u64,
    active: bool,
    vampires: HashSet<Identity>,
}

#[derive(Default)]
pub(super) struct EpochClock {
    pub(super) index: AtomicU64,
    completed: AtomicU64,
    active: AtomicBool,
    pending: AtomicUsize,
}

struct LaggedCredit {
    until: [u64; 2],
    nodes: u64,
    pool_nodes: u64,
    execs: u64,
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

#[derive(Default)]
struct DonorQueue { first: Option<u64>, last: Option<u64> }

struct NativeGraph {
    bank: u64,
    launch_cycle: Cycle,
    exec: usize,
    counted_launches: u64,
    signature: Vec<u64>,
    indexed: Option<usize>,
    previous: Option<u64>,
    next: Option<u64>,
}
struct UpdateRequest {
    identity: Identity,
    site: Site,
    estimate: Option<Estimate>,
    hard: bool,
    pinned: bool,
    started: Instant,
}

const OOM_RETRY_MISSES: u16 = 256;
const OOM_LEVEL_DROP: u64 = 2 << 20;

#[derive(Clone)]
struct OomBackoff {
    remaining: u16,
    level: Option<u64>,
    recovery_generation: u64,
}

/// One resident generation. Banks retain this allocation, never the controller
/// lock, across lookups. A retired generation is never made valid again.
#[derive(Default)]
pub(super) struct HitToken {
    valid: AtomicBool,
    last_used: AtomicU64,
}
impl HitToken {
    pub(super) fn valid(&self) -> bool { self.valid.load(Ordering::Acquire) }
    pub(super) fn hit(&self, epoch: u64) {
        if self.last_used.load(Ordering::Relaxed) != epoch {
            self.last_used.store(epoch, Ordering::Release);
        }
    }
    fn invalidate(&self) { self.valid.store(false, Ordering::Release); }
}

#[derive(Clone)]
struct Record {
    token: Option<Arc<HitToken>>,
    last_used: Option<u64>,
    identity: Identity,
    cycle: Cycle,
    bytes: u64,
    estimate: u64,
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
    signature: Option<Vec<u64>>,
    stats: Box<KeyStats>,
    accounted: Inventory,
}

#[derive(Default, Clone, Copy)]
struct Inventory { nodes: u64, execs: u64, unknown: u64, resident: u64, pinned: u64, spanning_bytes: u64, spanning_nodes: u64, spanning_execs: u64 }

impl Record {
    fn used(&self) -> Option<u64> {
        let observed = self.token.as_ref().map(|token| token.last_used.load(Ordering::Acquire));
        self.last_used.into_iter().chain(observed).max()
    }
    fn invalidate(&mut self) {
        if let Some(token) = self.token.take() { token.invalidate(); }
    }
    fn ordering_bytes(&self) -> u64 {
        if self.bytes > 0 { self.bytes } else { self.estimate }
    }
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

/// Optional legacy estimator metadata. Byte estimates order ARC records and
/// physical-floor projections; measured memory controls hard and soft stops.
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

#[derive(Default, Clone, Debug, serde::Serialize)]
pub(crate) struct NativePools {
    pub live_nodes: u64,
    pub live_execs: u64,
    pub unknown_execs: u64,
}

use cuteafd_loader::placement::graph_reserve::{GRAPH_CHUNK, capture_growth_headroom};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct SoftState {
    pub limit: u64,
    pub margin: u64,
    pub measured_bytes: Option<u64>,
    pub measured_nodes: Option<u64>,
    #[serde(skip)]
    granule: u64,
}

impl SoftState {
    fn new(hard: u64) -> Self {
        let margin = GRAPH_CHUNK.min(hard / 2);
        Self { limit: hard.saturating_sub(margin), margin,
            measured_bytes: None, measured_nodes: None, granule: GRAPH_CHUNK }
    }
    fn fits(&self, nodes: u64) -> bool {
        match (self.measured_bytes, self.measured_nodes) {
            (Some(bytes), Some(denominator)) =>
                u128::from(nodes) * u128::from(bytes) <= u128::from(self.limit) * u128::from(denominator),
            _ => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SoftRoom {
    /// Incoming nodes fit in live soft capacity right now; capture immediately.
    Available,
    /// Live soft capacity is full, but cold victims covering this cycle's
    /// accumulated unfulfilled capture demand (within the per-cycle clamp) are
    /// queued for retirement at the next cycle boundary.
    Queued,
    /// Cannot make room this cycle (reclaimable cold capacity is insufficient
    /// or the per-cycle eviction batch clamp is reached).
    Full,
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
#[cfg(test)]
thread_local! {
    static TEST_CONTROLLERS: std::cell::RefCell<Vec<std::sync::Weak<Mutex<DeviceCache>>>> = const { std::cell::RefCell::new(Vec::new()) };
}

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
    if controller.launch_query.is_none() { controller.launch_query = Some(library.graph_launch_query()?); }
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
    #[cfg(test)]
    if device < 0 && reserve < GRAPH_CHUNK {
        controller.soft.granule = 1;
        controller.soft.margin = 1;
        controller.soft.limit = reserve.saturating_sub(1);
    }
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
    #[cfg(test)]
    TEST_CONTROLLERS.with(|owned| owned.borrow_mut().push(Arc::downgrade(&cache)));
    Ok(cache)
}

/// Attach the admitted charge's provenance after a family has bound its banks.
pub(crate) fn set_reserve_policy(device: i32, policy: cuteafd_loader::placement::graph_reserve::Resolved) -> anyhow::Result<()> {
    let devices = DEVICES.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner());
    let cache = devices.get(&device).and_then(std::sync::Weak::upgrade)
        .ok_or_else(|| anyhow::anyhow!("no graph cache on device {device}"))?;
    let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
    anyhow::ensure!(cache.reserve == policy.bytes, "graph reserve policy disagrees with admission on device {device}");
    cache.apply_reserve_policy(policy);
    Ok(())
}

pub(crate) fn snapshots() -> Vec<Snapshot> {
    let caches: Vec<_> = DEVICES.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner())
        .values().filter(|cache| {
            #[cfg(test)]
            { return TEST_CONTROLLERS.with(|owned| owned.borrow().iter().any(|weak| std::sync::Weak::ptr_eq(weak, cache))); }
            #[cfg(not(test))]
            { let _ = cache; true }
        }).filter_map(std::sync::Weak::upgrade).collect();
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
    if cache.needs_boundary() && (!cache.event_completion || cache.lanes[0].completed >= cache.capture_frontier) {
        let (free, total, tracked, rounding, unclaimed) = match sample() {
            Ok(sample) => sample,
            Err(error) => {
                tracing::warn!(%error, device, "boundary graph measurement failed; capture disabled, serving eagerly");
                cache.measurement_failed = true;
                cache.floor_resample_pending = false;
                // Live event-backed owners cannot be invalidated by a failed sample.
                if cache.event_completion { return Ok(()) }
                let pending: Vec<_> = cache.records.iter().filter(|(_, r)| r.pending_capture).map(|(&id, _)| id).collect();
                for id in pending { cache.cancel_capture(id); }
                cache.unmeasured = false;
                cache.pending_bytes = 0;
                cache.pending_growth = 0;
                cache.captured_since_observation = false;
                cache.boundary();
                return Ok(());
            }
        };
        let (growth, level) = cache.observer.observe(free, total, tracked, rounding, unclaimed);
        cache.level = Some(level);
        cache.sample_physical(free, tracked);
        if cache.unmeasured || cache.failed_measurement_credit > 0 { cache.measured_capture(growth); }
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
    for &device in &devices {
        observe_device(library, device)?;
        if let Some(cache) = existing_cache(device) { cache.lock().unwrap_or_else(|p| p.into_inner()).serial_boundary(); }
    }
    visit(&mut |_, _, _, bank| {
        // SAFETY: serving banks retain only borrowed handles of persistent engine
        // buffers (or ()); their containing pipelines order buffer reuse.
        unsafe { bank.evict_persistent(); }
        Ok(())
    })?;
    Ok(())
}

/// Fixed-capacity event slots. If all slots are occupied, re-record the newest
/// marker: the later FIFO marker covers the previous one. Keep older markers
/// available for incremental progress, without allocating or waiting.
const COMPLETION_SLOTS: usize = 8;

#[derive(Default)]
struct CompletionSlots {
    pending: [Option<u64>; COMPLETION_SLOTS],
}
impl CompletionSlots {
    fn slot(&self) -> usize {
        self.pending.iter().position(Option::is_none).unwrap_or_else(||
            self.pending.iter().enumerate().max_by_key(|(_, cycle)| *cycle).unwrap().0)
    }
    fn record(&mut self, slot: usize, cycle: u64, record: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<()> {
        // A partial multi-rank record error cannot leave a valid old marker.
        self.pending[slot] = None;
        record()?;
        self.pending[slot] = Some(cycle);
        Ok(())
    }
    fn query(&mut self, mut query: impl FnMut(usize) -> anyhow::Result<bool>) -> anyhow::Result<Option<u64>> {
        let mut newest = None;
        for (slot, cycle) in self.pending.iter().enumerate() {
            if let Some(cycle) = cycle {
                if query(slot)? { newest = Some(newest.map_or(*cycle, |old: u64| old.max(*cycle))); }
            }
        }
        // A later marker on the same FIFO streams proves earlier work even
        // if an older marker wasn't ready when we queried it.
        if let Some(done) = newest {
            for cycle in &mut self.pending {
                if cycle.is_some_and(|cycle| cycle <= done) { *cycle = None; }
            }
        }
        Ok(newest)
    }
}

#[cfg(test)]
mod completion_tests {
    use super::*;

    #[test]
    fn completion_waits_for_all_owners_and_handles_out_of_order_queries() -> anyhow::Result<()> {
        let mut slots = CompletionSlots::default();
        slots.record(0, 1, || Ok(()))?;
        slots.record(1, 2, || Ok(()))?;
        let mut ready = [[true, false], [false, true]];
        assert_eq!(slots.query(|slot| Ok(ready[slot].iter().all(|x| *x)))?, None);
        ready[1][0] = true;
        assert_eq!(slots.query(|slot| Ok(ready[slot].iter().all(|x| *x)))?, Some(2));
        assert!(slots.pending.iter().all(Option::is_none));
        Ok(())
    }

    #[test]
    fn completion_pool_is_bounded_and_errors_never_prove_completion() -> anyhow::Result<()> {
        let mut slots = CompletionSlots::default();
        for cycle in 1..=100 {
            let slot = slots.slot();
            slots.record(slot, cycle, || Ok(()))?;
        }
        assert_eq!(slots.pending.iter().flatten().count(), COMPLETION_SLOTS);
        assert!(slots.pending.contains(&Some(100)));
        let slot = slots.slot();
        assert!(slots.record(slot, 101, || anyhow::bail!("record failed")).is_err());
        assert!(!slots.pending.contains(&Some(100)));
        let before = slots.pending;
        assert!(slots.query(|slot| if slot == 1 { anyhow::bail!("query failed") } else { Ok(true) }).is_err());
        assert_eq!(slots.pending, before);
        assert_eq!(slots.query(|_| Ok(true))?, Some((COMPLETION_SLOTS - 1) as u64));
        Ok(())
    }
}

/// One serial cycle shared by all owner streams (including peer ranks).
/// Allocate all timing-disabled events before the first step. An error never
/// publishes completion; teardown owns the existing family drain contract.
pub(crate) struct EventCycles<'a> {
    owners: Vec<(crate::shared::memory::device::Device<'a>, *mut std::ffi::c_void,
        [crate::shared::memory::device::Event<'a>; COMPLETION_SLOTS])>,
    devices: Vec<(crate::shared::memory::device::Device<'a>, SharedCache)>,
    slots: CompletionSlots,
    index: u64,
}
impl<'a> EventCycles<'a> {
    pub fn new(library: &'a cuteafd_ffi::NativeLibrary,
        streams: &[(i32, *mut std::ffi::c_void)]) -> anyhow::Result<Self> {
        use crate::shared::memory::device::{Device, Event};
        let mut owners: Vec<(Device<'a>, *mut std::ffi::c_void, [Event<'a>; COMPLETION_SLOTS])> = Vec::with_capacity(streams.len());
        let mut devices: Vec<(Device<'a>, SharedCache)> = Vec::new();
        for &(id, stream) in streams {
            if owners.iter().any(|(device, raw, _)| device.id == id && *raw == stream) { continue; }
            let device = Device { library, id };
            if !devices.iter().any(|(device, _)| device.id == id) {
                let cache = existing_cache(id).ok_or_else(|| anyhow::anyhow!("no graph cache on device {id}"))?;
                devices.push((device, cache));
            }
            let events: [Event<'a>; COMPLETION_SLOTS] = (0..COMPLETION_SLOTS)
                .map(|_| Event::new(device)).collect::<anyhow::Result<Vec<_>>>()?
                .try_into().ok().expect("fixed event pool size");
            owners.push((device, stream, events));
        }
        for (_, cache) in &devices {
            cache.lock().unwrap_or_else(|p| p.into_inner()).event_completion = true;
        }
        let index = devices.iter().map(|(_, cache)| cache.lock().unwrap_or_else(|p| p.into_inner())
            .clocks[0].index.load(Ordering::Acquire)).max().unwrap_or(0);
        Ok(Self { owners, devices, slots: CompletionSlots::default(), index })
    }

    pub fn record_drain(&self, elapsed: Duration) {
        for (_, cache) in &self.devices {
            let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
            cache.epochs.boundary_drains += 1;
            cache.epochs.boundary_wait_us += elapsed.as_micros().min(u64::MAX as u128) as u64;
        }
    }

    pub fn poll(&mut self) -> anyhow::Result<()> {
        let owners = &self.owners;
        let completed = self.slots.query(|slot| {
            let mut ready = true;
            for (device, _, events) in owners {
                // SAFETY: the pool owns this live event on this device.
                ready &= device.run(|| unsafe { device.library.cuda_event_query(events[slot].raw) })?;
            }
            Ok(ready)
        })?;
        for (device, cache) in &self.devices {
            {
                let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(done) = completed {
                    cache.clocks[0].completed.fetch_max(done, Ordering::Release);
                    cache.refresh_cycles();
                }
                cache.epochs.event_pending_cycles = self.index.saturating_sub(cache.lanes[0].completed);
            }
            if cache.lock().unwrap_or_else(|p| p.into_inner()).needs_boundary() {
                observe_device(device.library, device.id)?;
            }
        }
        Ok(())
    }

    /// Poll prior submissions and retire only proved-idle persistent owners.
    /// # Safety
    /// Visitor banks capture only persistent engine storage.
    pub unsafe fn begin_persistent(&mut self,
        mut visit: impl FnMut(&mut GraphVisitor<'_>) -> anyhow::Result<()>) -> anyhow::Result<()> {
        self.poll()?;
        visit(&mut |_, _, _, bank| {
            // SAFETY: forwarded persistent-storage contract.
            unsafe { bank.evict_persistent(); }
            Ok(())
        })?;
        self.begin();
        Ok(())
    }

    pub fn abandon(&mut self) {
        for (_, cache) in &self.devices {
            cache.lock().unwrap_or_else(|p| p.into_inner()).clocks[0].active.store(false, Ordering::Release);
        }
    }

    pub fn begin(&mut self) {
        self.index += 1;
        for (_, cache) in &self.devices {
            let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
            cache.clocks[0].index.store(self.index, Ordering::Release);
            cache.clocks[0].active.store(true, Ordering::Release);
            cache.refresh_cycles();
        }
    }

    /// # Safety
    /// Every owner stream is live and all this cycle's launches have been queued.
    pub unsafe fn complete(&mut self) -> anyhow::Result<()> {
        {
            let slot = self.slots.slot();
            let owners = &self.owners;
            self.slots.record(slot, self.index, || {
                for (device, stream, events) in owners {
                    // SAFETY: caller guarantees live streams; no stream waits on this query-only event.
                    device.run(|| unsafe { device.library.cuda_event_record(events[slot].raw, *stream) })?;
                }
                Ok(())
            })?;
        }
        for (_, cache) in &self.devices {
            let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
            cache.clocks[0].active.store(false, Ordering::Release);
            cache.epochs.event_pending_cycles = self.index.saturating_sub(cache.lanes[0].completed);
        }
        Ok(())
    }
}

/// Start/finish one independently scheduled cycle, visiting only its own banks.
/// # Safety
/// The visitor contains only this lane's persistent-storage banks. At start,
/// every launch from its previous cycle and commit has completed.
#[cfg(test)]
pub(crate) unsafe fn lane_cycle(lane: usize, begin: bool,
    mut visit: impl FnMut(&mut GraphVisitor<'_>) -> anyhow::Result<()>) -> anyhow::Result<()> {
    let mut devices = std::collections::BTreeSet::new();
    visit(&mut |device, _, _, bank| { bank.lane_owner(lane); devices.insert(device); Ok(()) })?;
    for device in devices {
        if let Some(cache) = existing_cache(device) {
            let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
            if begin { cache.begin_lane(lane); } else { cache.complete_lane(lane); }
        }
    }
    if begin {
        visit(&mut |_, _, _, bank| {
            // SAFETY: forwarded lane-local persistent-storage contract.
            unsafe { bank.evict_persistent(); }
            Ok(())
        })?;
    }
    Ok(())
}

// The lane executor and graph hits run on the same thread. Registration
// allocates this mirror once; hit stamps and round begin never allocate or
// acquire an extra lock/atomic beyond the existing cache hit path.
thread_local! {
    static HIT_EPOCHS: std::cell::RefCell<HashMap<i32, [u64; 2]>> = std::cell::RefCell::new(HashMap::new());
}

fn hit_epoch(device: i32, lane: usize, fallback: u64) -> u64 {
    HIT_EPOCHS.with(|epochs| epochs.borrow().get(&device).map_or(fallback, |epochs| epochs[lane].max(fallback)))
}

pub(crate) struct LaneCycles {
    lane: usize,
    devices: Vec<i32>,
    clocks: Vec<Arc<[EpochClock; 2]>>,
}

impl LaneCycles {
    pub fn register(lane: usize, concurrent_cycles: usize,
        mut visit: impl FnMut(&mut GraphVisitor<'_>) -> anyhow::Result<()>) -> anyhow::Result<Self> {
        // K bounds simultaneously in-flight decode/prefill operations (V4.1's
        // dual lane = 2). In-flight cycles protect every graph they may launch:
        // retire only after completion, and never update a live lane's operands.
        anyhow::ensure!((1..=2).contains(&concurrent_cycles) && lane < concurrent_cycles,
            "invalid graph lane/concurrent cycle count");
        let mut devices = std::collections::BTreeSet::new();
        visit(&mut |device, _, _, bank| { bank.lane_owner(lane); devices.insert(device); Ok(()) })?;
        let devices: Vec<_> = devices.into_iter().collect();
        let clocks = devices.iter().filter_map(|&device| existing_cache(device))
            .map(|cache| {
                let mut cache = cache.lock().unwrap_or_else(|p| p.into_inner());
                cache.lane_max_cycles[lane] = concurrent_cycles as u64;
                HIT_EPOCHS.with(|epochs| { epochs.borrow_mut().insert(cache.device,
                    cache.clocks.as_ref().each_ref().map(|clock| clock.index.load(Ordering::Acquire))); });
                cache.clocks.clone()
            }).collect();
        Ok(Self { lane, devices, clocks })
    }

    /// # Safety
    /// Every previous-round consumer has completed. The visitor contains only
    /// this registered lane's persistent-storage banks.
    pub unsafe fn begin(&mut self,
        mut visit: impl FnMut(&mut GraphVisitor<'_>) -> anyhow::Result<()>) -> anyhow::Result<()> {
        let mut pending = false;
        for (&device, clocks) in self.devices.iter().zip(&self.clocks) {
            let clock = &clocks[self.lane];
            // The current-thread lane executor never overlaps begin/serial
            // boundaries. A cancelled future can leave active set: quarantine
            // its index rather than panic or pretend it completed.
            if clock.active.swap(true, Ordering::AcqRel) {
                tracing::warn!(lane = self.lane, "resetting cancelled graph cycle without completion credit");
            }
            let index = clock.index.fetch_add(1, Ordering::AcqRel) + 1;
            HIT_EPOCHS.with(|epochs| {
                if let Some(epochs) = epochs.borrow_mut().get_mut(&device) { epochs[self.lane] = index; }
            });
            pending |= clock.pending.load(Ordering::Acquire) > 0;
        }
        if pending {
            visit(&mut |_, _, _, bank| {
                // SAFETY: forwarded lane-local persistent-storage contract.
                unsafe { bank.evict_persistent(); }
                Ok(())
            })?;
        }
        Ok(())
    }

    pub fn complete(&mut self) {
        for clocks in &self.clocks {
            let clock = &clocks[self.lane];
            clock.completed.store(clock.index.load(Ordering::Acquire), Ordering::Release);
            clock.active.store(false, Ordering::Release);
        }
    }

    pub fn abandon(&mut self) {
        // An error is not a completed-cycle proof. A later successful cycle,
        // or the scheduler's fail_round + reset_connections drained boundary,
        // can release credit; clearing active alone cannot.
        for clocks in &self.clocks { clocks[self.lane].active.store(false, Ordering::Release); }
    }
}

pub(crate) struct DeviceCache {
    device: i32,
    reserve: u64,
    reserve_policy: Option<cuteafd_loader::placement::graph_reserve::Resolved>,
    charged: u64,
    live_bytes: u64,
    inventory: Inventory,
    ranked_hits: Vec<TraceRow>,
    ranked_churn: Vec<TraceRow>,
    ranked_at: Instant,
    level: Option<u64>,
    physical_floor: u64,
    physical_sample: Option<(Instant, u64, u64)>,
    physical_floor_warned: bool,
    floor_resample_pending: bool,
    floor_resample_used: bool,
    soft: SoftState,
    margin_clamp_warned: bool,
    measurement_failed: bool,
    failed_measurement_credit: u64,
    pending_bytes: u64,
    event_completion: bool,
    capture_frontier: u64,
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
    cycle_demand_nodes: u64,
    bank_lanes: HashMap<u64, usize>,
    lanes: [LaneEpoch; 2],
    clocks: Arc<[EpochClock; 2]>,
    lagged: Vec<LaggedCredit>,
    epochs: EpochStats,
    reuse_captures: HashSet<u64>,
    selected_epochs: HashMap<u64, [u64; 2]>,
    pending_lanes: [usize; 2],
    donor_always: bool,
    donor_cold_lag: u64,
    lane_max_cycles: [u64; 2],
    site_lanes: HashMap<&'static str, u8>,
    update_sites: std::collections::BTreeMap<&'static str, HashMap<Cycle, Counters>>,
    native_graphs: HashMap<u64, NativeGraph>,
    bank_native: HashMap<u64, HashSet<u64>>,
    donor_candidates: HashMap<Vec<u64>, [DonorQueue; 2]>,
    update_requests: HashMap<u64, UpdateRequest>,
    observer: GrowthObserver,
    memory_query: Option<cuteafd_ffi::PhysicalMemoryQuery>,
    launch_query: Option<cuteafd_ffi::GraphLaunchQuery>,
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

/// Set once the serving engine is ready (`memory_report::log("ready")`), after every
/// family's startup prewarm: from then on, graph sites spanning both GPUs run eagerly
/// instead of capturing mid-request (#1878).
static CROSS_DEVICE_SEALED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn seal_cross_device_capture() {
    if !CROSS_DEVICE_SEALED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        tracing::info!("serving: cross-device graph sites run eagerly on a miss (startup prewarm stays captured)");
    }
}

fn cross_device_capture_sealed() -> bool {
    // Tests run in parallel threads, and any of them may reach a readiness log:
    // only a thread's own explicit seal applies there.
    #[cfg(test)]
    return TEST_SEAL.with(|seal| seal.get()).unwrap_or(false);
    #[cfg(not(test))]
    CROSS_DEVICE_SEALED.load(std::sync::atomic::Ordering::Acquire)
}

// Tests run in parallel threads: each seals only its own thread.
#[cfg(test)]
thread_local! { static TEST_SEAL: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) }; }

#[cfg(test)]
pub(crate) fn seal_cross_device_capture_on_this_thread(sealed: bool) {
    TEST_SEAL.with(|seal| seal.set(Some(sealed)));
}

impl DeviceCache {
    fn new(device: i32, reserve: u64) -> Self {
        Self { device, reserve, reserve_policy: None, charged: 0, live_bytes: 0, inventory: Inventory::default(), ranked_hits: Vec::new(), ranked_churn: Vec::new(), ranked_at: Instant::now() - Duration::from_secs(30), level: None, pending_bytes: 0, event_completion: false, capture_frontier: 0, next_id: 0, next_bank: 0, identity_generation: 0, recovery_generation: 0,
            soft: SoftState::new(reserve), margin_clamp_warned: false, measurement_failed: false, failed_measurement_credit: 0, physical_floor: 0, physical_sample: None, physical_floor_warned: false,
            floor_resample_pending: false, floor_resample_used: false,
            recent_target: 0, records: HashMap::new(), identities: HashMap::new(),
            queues: std::array::from_fn(|_| Queue::default()), classes: HashMap::new(),
            pending_growth: 0, pools: Pools::default(), basis: Basis::Bytes,
            chain_only: true, unmeasured: false, pending_destroy: 0, cycle_demand_nodes: 0,
            bank_lanes: HashMap::new(), lanes: std::array::from_fn(|_| LaneEpoch::default()), clocks: Arc::new(std::array::from_fn(|_| EpochClock::default())),
            lagged: Vec::new(), epochs: EpochStats::default(), reuse_captures: HashSet::new(), selected_epochs: HashMap::new(), pending_lanes: [0; 2], donor_always: super::donor_always(), donor_cold_lag: super::donor_cold_lag(), lane_max_cycles: [1; 2], site_lanes: HashMap::new(), update_sites: Default::default(), native_graphs: HashMap::new(), bank_native: HashMap::new(), donor_candidates: HashMap::new(), update_requests: HashMap::new(),
            observer: GrowthObserver::default(), memory_query: None, launch_query: None, captured_since_observation: false,
            counters: HashMap::new(), started: Instant::now(), clock_ms: 0, last_report: Instant::now(),
            interval_evictions: 0, trace: std::env::var_os("CUTEAFD_GRAPH_TRACE").and_then(|path| {
                TraceWriter::new(std::path::PathBuf::from(path).with_extension(format!("device-{device}.json")))
                    .map_err(|error| tracing::warn!(%error, "graph trace open failed")).ok()
            }), retired_trace: None,
            last_trace: Instant::now() - Duration::from_secs(1) }
    }

    fn site_counters(&mut self, site: &'static str, cycle: Cycle, bank: u64) -> &mut Counters {
        let lane = self.bank_lane(bank);
        *self.site_lanes.entry(site).or_default() |= 1 << lane;
        self.update_sites.entry(site).or_default().entry(cycle).or_default()
    }

    pub(super) fn retire_native_counts(&mut self, ids: &[u64]) {
        for &id in ids { self.remove_donor_candidate(id); self.take_native(id); }
    }

    fn take_native(&mut self, id: u64) -> Option<NativeGraph> {
        let native = self.native_graphs.remove(&id)?;
        if let Some(ids) = self.bank_native.get_mut(&native.bank) { ids.remove(&id); }
        if let (Some(query), Some(record)) = (&self.launch_query, self.records.get(&id)) {
            if native.exec != 0 {
                let count = query.sample(native.exec).saturating_sub(native.counted_launches);
                self.counters.entry(native.launch_cycle).or_default().segments_launched += count;
                self.update_sites.entry(record.identity.site).or_default().entry(native.launch_cycle).or_default().segments_launched += count;
            }
        }
        Some(native)
    }

    fn launch_counts(&self) -> (HashMap<Cycle, Counters>, std::collections::BTreeMap<&'static str, HashMap<Cycle, Counters>>) {
        let (mut cycles, mut sites) = (self.counters.clone(), self.update_sites.clone());
        if let Some(query) = &self.launch_query {
            for (&id, native) in &self.native_graphs {
                if native.exec == 0 { continue }
                let Some(record) = self.records.get(&id) else { continue };
                let count = query.sample(native.exec).saturating_sub(native.counted_launches);
                cycles.entry(native.launch_cycle).or_default().segments_launched += count;
                sites.entry(record.identity.site).or_default().entry(native.launch_cycle).or_default().segments_launched += count;
            }
        }
        (cycles, sites)
    }

    fn per_cycle(&self, sites: &std::collections::BTreeMap<&'static str, HashMap<Cycle, Counters>>) -> std::collections::BTreeMap<&'static str, HashMap<Cycle, PerCycle>> {
        let completed = self.clocks.as_ref().each_ref().map(|clock| clock.completed.load(Ordering::Acquire));
        sites.iter().map(|(&site, cycles)| {
            let lanes = self.site_lanes.get(site).copied().unwrap_or(0);
            let owner_cycles: u64 = completed.iter().enumerate().filter(|(lane, _)| lanes & (1 << lane) != 0)
                .map(|(_, count)| count).sum();
            let denominator = owner_cycles.max(1) as f64;
            (site, cycles.iter().map(|(&cycle, count)| (cycle, PerCycle { owner_cycles,
                hits: count.hits as f64 / denominator, misses: count.misses as f64 / denominator,
                segments_launched: count.segments_launched as f64 / denominator })).collect())
        }).collect()
    }

    pub(super) fn bank_cycle(&mut self, bank: u64, cycle: Cycle) {
        let Some(ids) = self.bank_native.get(&bank) else { return };
        for id in ids {
            let native = self.native_graphs.get_mut(id).expect("bank native membership");
            if native.launch_cycle == cycle { continue }
            if let (Some(query), Some(record)) = (&self.launch_query, self.records.get(id)) {
                if native.exec != 0 {
                    let total = query.sample(native.exec);
                    let delta = total.saturating_sub(native.counted_launches);
                    self.counters.entry(native.launch_cycle).or_default().segments_launched += delta;
                    self.update_sites.entry(record.identity.site).or_default().entry(native.launch_cycle).or_default().segments_launched += delta;
                    native.counted_launches = total;
                }
            }
            native.launch_cycle = cycle;
        }
    }

    pub fn bank(&mut self) -> u64 { self.next_bank += 1; self.next_bank }
    pub fn assign_bank_lane(&mut self, bank: u64, lane: usize) {
        assert!(lane < self.lanes.len());
        if self.bank_lanes.get(&bank) == Some(&lane) { return }
        let old = self.bank_lane(bank);
        if old != lane {
            let pending = self.records.values().filter(|r| r.identity.bank == bank && r.pending_destroy).count();
            self.pending_lanes[old] = self.pending_lanes[old].checked_sub(pending)
                .expect("graph pending lane ownership mismatch");
            self.pending_lanes[lane] += pending;
            let previous = self.clocks[old].pending.fetch_sub(pending, Ordering::AcqRel);
            debug_assert!(previous >= pending, "graph pending lane atomic mismatch");
            self.clocks[lane].pending.fetch_add(pending, Ordering::Release);
        }
        let owners: Vec<_> = self.records.iter().filter(|(_, record)| record.identity.bank == bank)
            .map(|(&id, _)| id).collect();
        for &id in &owners { self.remove_donor_candidate(id); }
        self.bank_lanes.insert(bank, lane);
        for id in owners { self.insert_donor_candidate(id); }
    }

    pub(super) fn bank_lane(&self, bank: u64) -> usize { self.bank_lanes.get(&bank).copied().unwrap_or(0) }

    #[cfg(test)]
    fn begin_lane(&mut self, lane: usize) {
        let clock = &self.clocks[lane];
        assert!(!clock.active.swap(true, Ordering::AcqRel), "graph lane cycle already active");
        clock.index.fetch_add(1, Ordering::AcqRel);
        self.refresh_cycles();
    }

    #[cfg(test)]
    fn complete_lane(&mut self, lane: usize) {
        let clock = &self.clocks[lane];
        clock.completed.store(clock.index.load(Ordering::Acquire), Ordering::Release);
        clock.active.store(false, Ordering::Release);
        self.refresh_cycles();
    }

    pub(super) fn refresh_cycles(&mut self) {
        let mut index_advanced = false;
        for (lane, clock) in self.lanes.iter_mut().zip(self.clocks.iter()) {
            let index = clock.index.load(Ordering::Acquire);
            let completed = clock.completed.load(Ordering::Acquire);
            if lane.index != index { index_advanced = true; }
            if lane.index != index || lane.completed != completed { lane.vampires.clear(); }
            lane.index = index;
            lane.completed = completed;
            lane.active = clock.active.load(Ordering::Acquire);
        }
        if index_advanced { self.cycle_demand_nodes = 0; }
        self.epochs.cycles = self.lanes.each_ref().map(|lane| lane.index);
        self.epochs.completed = self.lanes.each_ref().map(|lane| lane.completed);
        self.credit_completed();

    }

    fn credit_completed(&mut self) {
        self.lagged.retain(|credit| {
            let ready = self.lanes.iter().zip(credit.until).all(|(lane, until)| lane.completed >= until);
            if ready {
                self.pools.live_nodes = self.pools.live_nodes.saturating_sub(credit.pool_nodes);
                self.pools.live_execs = self.pools.live_execs.saturating_sub(credit.execs);
            }
            !ready
        });
        for credit in &mut self.lagged {
            for (until, lane) in credit.until.iter_mut().zip(&self.lanes) {
                if lane.completed >= *until { *until = 0; }
            }
        }
        let mut i = 0;
        while i < self.lagged.len() {
            let mut j = i + 1;
            while j < self.lagged.len() {
                if self.lagged[i].until == self.lagged[j].until {
                    let credit = self.lagged.swap_remove(j);
                    self.lagged[i].nodes += credit.nodes;
                    self.lagged[i].pool_nodes += credit.pool_nodes;
                    self.lagged[i].execs += credit.execs;
                } else { j += 1; }
            }
            i += 1;
        }
        self.epochs.uncredited_nodes = self.lagged.iter().map(|credit| credit.nodes).sum();
    }

    // Called only by the serial executor after drainage (V4.1 failures pass
    // fail_round + reset_connections first). The current-thread LaneSet polls
    // begin/complete and these boundaries serially; not a multi-thread barrier.
    pub(super) fn serial_boundary(&mut self) {
        self.refresh_cycles();
        if self.event_completion || self.lanes.iter().any(|lane| lane.active) { return }
        for clock in self.clocks.iter() {
            let next = clock.index.fetch_add(1, Ordering::AcqRel) + 1;
            clock.completed.store(next, Ordering::Release);
        }
        self.refresh_cycles();
        HIT_EPOCHS.with(|epochs| {
            if let Some(epochs) = epochs.borrow_mut().get_mut(&self.device) {
                *epochs = self.lanes.each_ref().map(|lane| lane.index);
            }
        });
    }
    pub fn reserve(&self) -> u64 { self.reserve }
    pub fn identity_generation(&self) -> u64 { self.identity_generation }
    pub fn retains_identity(&self, identity: &Identity) -> bool {
        self.identities.contains_key(identity) || self.lanes.iter().any(|lane| lane.vampires.contains(identity))
    }
    pub fn needs_boundary(&self) -> bool {
        self.unmeasured || self.failed_measurement_credit > 0 || self.pending_destroy > 0 || self.floor_resample_pending
    }
    pub fn has_pending_victims(&self) -> bool { self.pending_destroy > 0 }
    pub fn pending(&self, id: u64) -> bool { self.records.get(&id).is_some_and(|r| r.pending_destroy) }
    /// Call refresh_cycles once before inspecting a batch of victims.
    pub fn persistent_victim_ready(&self, id: u64) -> bool {
        self.records.get(&id).is_some_and(|record| {
            let lane = self.bank_lane(record.identity.bank);
            let epoch = &self.lanes[lane];
            if self.event_completion {
                return record.pending_destroy && record.used().is_some_and(|used| used <= epoch.completed)
                    && self.selected_epochs.get(&id).is_none_or(|until| epoch.completed >= until[lane]);
            }
            let closed = if epoch.active { epoch.completed >= epoch.index.saturating_sub(1) }
                else { epoch.completed >= epoch.index };
            record.pending_destroy && closed && self.selected_epochs.get(&id).is_none_or(|until|
                epoch.completed >= until[lane])
        })
    }

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
                "physical graph headroom below floor; running eagerly");
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
        let (previous, next, bytes) = (record.previous, record.next, record.ordering_bytes());
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
        queue.bytes = queue.bytes.saturating_add(self.records[&id].ordering_bytes());
        queue.len += 1;
        if let Some(last) = last { self.records.get_mut(&last).unwrap().next = Some(id); }
        let record = self.records.get_mut(&id).unwrap();
        record.previous = last; record.next = None; record.list = Some(list);
    }

    fn forget(&mut self, id: u64) {
        if !self.records.contains_key(&id) { return }
        self.remove_donor_candidate(id);
        self.take_native(id);
        self.reuse_captures.remove(&id);
        self.unlink(id);
        if let Some(mut record) = self.records.remove(&id) {
            record.invalidate();
            let old = record.accounted;
            self.inventory.nodes = self.inventory.nodes.saturating_sub(old.nodes);
            self.inventory.execs = self.inventory.execs.saturating_sub(old.execs);
            self.inventory.unknown = self.inventory.unknown.saturating_sub(old.unknown);
            self.inventory.resident = self.inventory.resident.saturating_sub(old.resident);
            self.inventory.pinned = self.inventory.pinned.saturating_sub(old.pinned);
            self.inventory.spanning_bytes = self.inventory.spanning_bytes.saturating_sub(old.spanning_bytes);
            self.inventory.spanning_nodes = self.inventory.spanning_nodes.saturating_sub(old.spanning_nodes);
            self.inventory.spanning_execs = self.inventory.spanning_execs.saturating_sub(old.spanning_execs);
            self.live_bytes = self.live_bytes.saturating_sub(record.bytes); self.identities.remove(&record.identity); self.identity_generation += 1; }
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
        let Some(bank) = self.records.get(&id).map(|record| record.identity.bank) else { return };
        let lane = self.bank_lane(bank);
        let epoch = hit_epoch(self.device, lane, self.lanes[lane].index);
        let record = self.records.get_mut(&id).unwrap();
        let changed = record.last_used != Some(epoch);
        record.last_used = Some(epoch);
        let promote = !record.pinned && !record.pending_destroy && matches!(record.list, Some(List::Recent | List::Frequent));
        record.stats.hit(cycle, 1, now);
        self.counters.entry(cycle).or_default().hits += 1;
        self.site_counters(self.records[&id].identity.site, cycle, bank).hits += 1;
        if changed { self.remove_donor_candidate(id); self.insert_donor_candidate(id); }
        if promote {
            self.push(id, List::Frequent);
        }
    }

    /// Fold bank-local counters once per cycle. The shared token already made
    /// each hit visible to pressure and donor selection before this fold.
    pub(super) fn batched_hits(&mut self, id: u64, cycle: Cycle, count: u64) {
        if count == 0 { return }
        let Some(record) = self.records.get_mut(&id) else { return };
        if !(record.pinned || record.pending_capture || record.pending_destroy ||
            matches!(record.list, Some(List::Recent | List::Frequent))) { return }
        let promote = !record.pinned && !record.pending_destroy && record.list == Some(List::Recent);
        record.stats.hit(cycle, count, self.clock_ms);
        let site = record.identity.site;
        self.counters.entry(cycle).or_default().hits += count;
        self.update_sites.entry(site).or_default().entry(cycle).or_default().hits += count;
        // Frequent residents are a FIFO. Last-use tokens protect hot owners;
        // replay need not remove/reinsert an already frequent node each cycle.
        if promote { self.push(id, List::Frequent); }
    }

    /// Pinned hits accumulate in the bank, without a lock or ARC mutation.
    pub fn pinned_hits(&mut self, id: u64, cycle: Cycle, count: u64) {
        if count == 0 { return }
        let now = self.clock_ms;
        if let Some(record) = self.records.get_mut(&id) { record.stats.hit(cycle, count, now); }
        self.counters.entry(cycle).or_default().hits += count;
        if let Some(record) = self.records.get(&id) {
            let (site, bank) = (record.identity.site, record.identity.bank);
            let counters = self.site_counters(site, cycle, bank);
            counters.hits += count;
        }
    }

    #[cfg(test)]
    pub fn cycle_hits(&self, cycle: Cycle) -> u64 {
        self.counters.get(&cycle).map_or(0, |counters| counters.hits)
    }

    fn sync_inventory(&mut self, id: u64) {
        let Some(record) = self.records.get_mut(&id) else { return };
        let live = !record.permanently_eager && (record.pending_capture || record.pinned
            || record.pending_destroy || matches!(record.list, Some(List::Recent | List::Frequent)));
        let next = if live { Inventory { nodes: record.stats.native_nodes.unwrap_or(0), execs: 1,
            unknown: u64::from(record.stats.native_nodes.is_none()),
            resident: if record.pending_capture { 0 } else { record.ordering_bytes() },
            pinned: if record.pinned { record.bytes } else { 0 },
            spanning_bytes: if record.spans_devices { record.bytes } else { 0 },
            spanning_nodes: if record.spans_devices { record.stats.native_nodes.unwrap_or(0) } else { 0 },
            spanning_execs: u64::from(record.spans_devices) } } else { Inventory::default() };
        let old = std::mem::replace(&mut record.accounted, next);
        self.inventory.nodes = self.inventory.nodes.saturating_sub(old.nodes).saturating_add(next.nodes);
        self.inventory.execs = self.inventory.execs.saturating_sub(old.execs).saturating_add(next.execs);
        self.inventory.unknown = self.inventory.unknown.saturating_sub(old.unknown).saturating_add(next.unknown);
        self.inventory.resident = self.inventory.resident.saturating_sub(old.resident).saturating_add(next.resident);
        self.inventory.pinned = self.inventory.pinned.saturating_sub(old.pinned).saturating_add(next.pinned);
        self.inventory.spanning_bytes = self.inventory.spanning_bytes.saturating_sub(old.spanning_bytes).saturating_add(next.spanning_bytes);
        self.inventory.spanning_nodes = self.inventory.spanning_nodes.saturating_sub(old.spanning_nodes).saturating_add(next.spanning_nodes);
        self.inventory.spanning_execs = self.inventory.spanning_execs.saturating_sub(old.spanning_execs).saturating_add(next.spanning_execs);
    }

    fn native_pools(&self) -> NativePools {
        NativePools { live_nodes: self.inventory.nodes, live_execs: self.inventory.execs,
            unknown_execs: self.inventory.unknown }
    }

    fn resident_bytes(&self) -> u64 { self.inventory.resident }

    fn reported_live_bytes(&self) -> u64 {
        match (self.soft.measured_bytes, self.soft.measured_nodes) {
            (Some(_), Some(measured_nodes)) if measured_nodes > 0 => {
                let live_nodes = self.native_pools().live_nodes;
                ((u128::from(self.charged) * u128::from(live_nodes)) / u128::from(measured_nodes))
                    .min(u128::from(self.charged)) as u64
            }
            _ => self.live_bytes,
        }
    }

    fn estimate(&self, site: Site, node_estimate: Option<u64>) -> (u64, u64) {
        let class = self.classes.get(site.name);
        let estimate = node_estimate.filter(|n| *n > 0).unwrap_or_else(|| class
            .filter(|c| c.samples > 0 && c.total > 0)
            .map_or(site.bootstrap, |c| c.total.div_ceil(c.samples)));
        (estimate, class.map_or(estimate, |c| c.largest.max(estimate)))
    }

    #[cfg(test)]
    pub(super) fn mark_victim(&mut self, _frequent_ghost: bool) -> Option<u64> { self.select_victim(None) }

    /// Shared eviction and donor safety floor. Event-enabled caches also
    /// require proof that the last use completed; others use cycle age alone.
    fn cold_age(&self, record: &Record) -> Option<u64> {
        let used = record.used()?;
        let lane = self.bank_lane(record.identity.bank);
        if self.event_completion && used > self.lanes[lane].completed { return None }
        let age = self.lanes[lane].index.saturating_sub(used);
        (age > self.lane_max_cycles[lane]).then_some(age)
    }

    fn hot(&self, record: &Record) -> bool {
        self.cold_age(record).is_none()
    }

    fn select_victim(&mut self, excluded: Option<u64>) -> Option<u64> {
        self.refresh_cycles();
        let recent = &self.queues[List::Recent.index()];
        let prefer_recent = recent.first.is_some() && (recent.bytes > self.recent_target);
        let first = if prefer_recent { List::Recent } else { List::Frequent };
        let second = if prefer_recent { List::Frequent } else { List::Recent };
        let mut selected = None;
        for list in [first, second] {
            let mut next = self.queues[list.index()].first;
            while let Some(id) = next {
                let record = &self.records[&id];
                if !record.pending_capture && !record.spans_devices && Some(id) != excluded {
                    if self.hot(record) {
                        self.counters.entry(record.cycle).or_default().hot_guard_skips += 1;
                        self.update_sites.entry(record.identity.site).or_default().entry(record.cycle).or_default().hot_guard_skips += 1;
                    } else { selected = Some(id); break; }
                }
                next = record.next;
            }
            if selected.is_some() { break }
        }
        let id = selected?;
        self.remove_donor_candidate(id);
        let frequent = self.records[&id].list == Some(List::Frequent);
        self.unlink(id);
        self.records.get_mut(&id).unwrap().frequent = frequent;
        self.records.get_mut(&id).unwrap().pending_destroy = true;
        self.pending_destroy += 1;
        // A selected executable can still be replayed by its owner before
        // retirement. Include every subsequently started cycle when destroying.
        self.selected_epochs.insert(id, self.lanes.each_ref().map(|lane| lane.index));
        let lane = self.bank_lane(self.records[&id].identity.bank);
        self.epochs.selections[lane] += 1;
        self.pending_lanes[lane] += 1;
        self.clocks[lane].pending.fetch_add(1, Ordering::Release);
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

    pub(super) fn source_capture_failed(&mut self, id: u64, small: Option<usize>) {
        if let Some(request) = self.update_requests.remove(&id) {
            // End/begin failures precede any update. Settle through the ordinary
            // miss accounting, without retaining a source-only reservation.
            self.take_native(id);
            let counters = self.counters.entry(request.site.cycle).or_default();
            counters.misses += 1;
            counters.eager_runs += 1;
            counters.eager_other += 1;
            let counters = self.site_counters(request.site.name, request.site.cycle, request.identity.bank);
            counters.misses += 1; counters.eager_runs += 1; counters.eager_other += 1;
            if small.is_some() {
                if let Some(old) = self.identities.get(&request.identity).copied() { self.forget(old); }
                self.identities.insert(request.identity.clone(), id);
                self.records.insert(id, Record { token: None, last_used: None, identity: request.identity, cycle: request.site.cycle,
                    bytes: 0, estimate: 0, list: None, previous: None, next: None, pinned: false,
                    pending_destroy: false, pending_capture: false, frequent: false, nodes: 0,
                    node_rate: 1, graph_rate: 0, parallel_width: 0, spans_devices: request.site.spans_devices,
                    oom: None, oom_warned: false, permanently_eager: false, signature: None,
                    stats: KeyStats::new(self.clock_ms), accounted: Inventory::default() });
            }
        }
    }

    pub(super) fn hit_token(&mut self, id: u64) -> Arc<HitToken> {
        let record = self.records.get_mut(&id).expect("resident graph identity");
        record.token.get_or_insert_with(|| Arc::new(HitToken {
            valid: AtomicBool::new(true),
            last_used: AtomicU64::new(record.last_used.unwrap_or(0)),
        })).clone()
    }

    pub(super) fn bank_clock(&self, bank: u64) -> (Arc<[EpochClock; 2]>, usize) {
        (self.clocks.clone(), self.bank_lane(bank))
    }

    pub(super) fn owns(&self, id: u64, bank: u64) -> bool {
        self.records.get(&id).is_some_and(|r| r.identity.bank == bank &&
            (r.pinned || r.pending_capture || r.pending_destroy || matches!(r.list, Some(List::Recent | List::Frequent))))
    }

    /// A source-only capture defers eviction and hard-stop admission until its
    /// exact native topology is known. It reserves no driver arena bytes.
    /// Spanning prewarm records must never donate: after readiness their missing
    /// shapes cannot be recaptured safely, so startup reuse would destroy coverage.
    pub(super) fn update_ticket(&mut self, identity: Identity, site: Site,
        estimate: Option<Estimate>, pinned: bool) -> Option<u64> {
        self.refresh_cycles();
        // Only one live source may settle this exact key. Otherwise re-keying
        // could forget a resident executable without retiring its pool charge.
        if self.update_requests.values().any(|request| request.identity == identity) { return None }
        self.site_counters(site.name, site.cycle, identity.bank);
        let hard = self.charged >= self.reserve;
        if self.reserve == 0 || self.measurement_failed || self.soft.limit == 0 ||
            site.preference == Preference::Eager || site.basis != self.basis ||
            site.spans_devices ||
            self.identities.get(&identity).is_some_and(|id| {
                let r = &self.records[id];
                r.pinned || r.pending_capture || r.pending_destroy || r.permanently_eager || r.oom.is_some() ||
                    matches!(r.list, Some(List::Recent | List::Frequent))
            }) || (!self.donor_always && !hard && self.charged < self.soft.limit) { return None }
        self.donor_candidates.retain(|_, lanes| lanes.iter().any(|lane| lane.first.is_some()));
        let signature = self.identities.get(&identity).and_then(|id| self.records[id].signature.clone());
        // Known empty or mismatched topologies never justify a hard-cap source
        // capture. An unseen shape (`None`) may still do one host-only source
        // capture (no device arena allocation) when a cold donor exists, learn
        // its signature, and transfer the oldest matching unpinned donor; if
        // nothing matches, `fallback_update` retains the learned signature on
        // the ghost and returns `Miss::Eager`.
        let any_cold = match signature {
            Some(signature) => self.cold_donor(&signature).is_some(),
            None => {
                let signatures: Vec<_> = self.donor_candidates.keys().cloned().collect();
                signatures.iter().any(|signature| self.cold_donor(signature).is_some())
            },
        };
        if !any_cold {
            let counters = self.counters.entry(site.cycle).or_default();
            if hard { counters.no_cold_donor_hard += 1 } else { counters.no_cold_donor_soft += 1 }
            let counters = self.update_sites.entry(site.name).or_default().entry(site.cycle).or_default();
            if hard { counters.no_cold_donor_hard += 1 } else { counters.no_cold_donor_soft += 1 }
            return None;
        }
        self.next_id += 1;
        let id = self.next_id;
        self.update_requests.insert(id, UpdateRequest { identity, site, estimate, hard, pinned, started: Instant::now() });
        Some(id)
    }

    fn remove_donor_candidate(&mut self, id: u64) {
        let Some(native) = self.native_graphs.get_mut(&id) else { return };
        let Some(lane) = native.indexed.take() else { return };
        let previous = native.previous.take();
        let next = native.next.take();
        let candidates = self.donor_candidates.get_mut(native.signature.as_slice()).expect("indexed topology");
        if let Some(previous) = previous { self.native_graphs.get_mut(&previous).unwrap().next = next; }
        else { candidates[lane].first = next; }
        if let Some(next) = next { self.native_graphs.get_mut(&next).unwrap().previous = previous; }
        else { candidates[lane].last = previous; }
    }

    fn donor_age(&self, id: u64) -> Option<u64> {
        let record = self.records.get(&id)?;
        if record.pinned || record.pending_capture || record.pending_destroy ||
            record.spans_devices ||
            !matches!(record.list, Some(List::Recent | List::Frequent)) ||
            !self.native_graphs.contains_key(&id) { return None }
        let age = self.cold_age(record)?;
        let lane = self.bank_lane(record.identity.bank);
        // Pressure keeps the safety floor K; optional extra lag only protects
        // reusable shapes while donor-always is operating below the soft cap.
        let lag = if self.donor_always && self.charged < self.soft.limit && self.charged < self.reserve {
            self.donor_cold_lag
        } else { 0 };
        (age > self.lane_max_cycles[lane].saturating_add(lag)).then_some(age)
    }

    fn insert_donor_candidate(&mut self, id: u64) {
        let Some(native) = self.native_graphs.get(&id) else { return };
        let Some(record) = self.records.get(&id) else { return };
        if native.indexed.is_some() || native.signature.is_empty() || record.pinned ||
            record.pending_capture || record.pending_destroy ||
            !matches!(record.list, Some(List::Recent | List::Frequent)) { return }
        let Some(last) = record.last_used else { return };
        let lane = self.bank_lane(record.identity.bank);
        if !self.donor_candidates.contains_key(native.signature.as_slice()) {
            self.donor_candidates.insert(native.signature.clone(), Default::default());
        }
        let queue = &self.donor_candidates[native.signature.as_slice()][lane];
        let mut previous = queue.last;
        // Captures and hits append in O(1). Rare bank reassignment can insert an
        // older stamp; preserve last-use order without an index repair on misses.
        while let Some(candidate) = previous {
            if self.records[&candidate].last_used <= Some(last) { break }
            previous = self.native_graphs[&candidate].previous;
        }
        let next = previous.map_or(queue.first, |id| self.native_graphs[&id].next);
        let native = self.native_graphs.get_mut(&id).unwrap();
        native.indexed = Some(lane); native.previous = previous; native.next = next;
        let queue = &mut self.donor_candidates.get_mut(native.signature.as_slice()).unwrap()[lane];
        if let Some(previous) = previous { self.native_graphs.get_mut(&previous).unwrap().next = Some(id); }
        else { queue.first = Some(id); }
        if let Some(next) = next { self.native_graphs.get_mut(&next).unwrap().previous = Some(id); }
        else { queue.last = Some(id); }
    }

    fn cold_donor(&mut self, signature: &[u64]) -> Option<u64> {
        self.refresh_cycles();
        if signature.is_empty() { return None }
        // Hits only stamp their token. Repair touched queue heads on a miss,
        // never relink FIFO nodes (or take the cache mutex) on a replay.
        for lane in 0..self.lanes.len() {
            loop {
                let Some(id) = self.donor_candidates.get(signature)?.get(lane)?.first else { break };
                let record = &self.records[&id];
                if record.spans_devices {
                    self.remove_donor_candidate(id);
                    continue;
                }
                let used = record.used();
                if used == record.last_used { break }
                self.remove_donor_candidate(id);
                self.records.get_mut(&id).unwrap().last_used = used;
                self.insert_donor_candidate(id);
            }
        }
        self.donor_candidates.get(signature)?.iter().filter_map(|lane| {
            let id = lane.first?;
            self.donor_age(id).map(|age| (id, age))
        }).max_by_key(|&(id, age)| (age, std::cmp::Reverse(id))).map(|(id, _)| id)
    }

    pub(super) fn register_native(&mut self, id: u64, exec: usize, signature: Vec<u64>) {
        self.remove_donor_candidate(id);
        if let Some(record) = self.records.get_mut(&id) { record.signature = Some(signature.clone()); }
        let (bank, launch_cycle) = self.records.get(&id).map(|r| (r.identity.bank, r.cycle))
            .or_else(|| self.update_requests.get(&id).map(|r| (r.identity.bank, r.site.cycle)))
            .expect("native graph requires a capture ticket");
        self.bank_native.entry(bank).or_default().insert(id);
        self.native_graphs.insert(id, NativeGraph { bank, launch_cycle, exec, counted_launches: 0, signature, indexed: None, previous: None, next: None });
        self.insert_donor_candidate(id);
    }

    /// Re-keying and the driver update share the device lock. Every old bank
    /// lookup validates this record, so neither stale keys nor stale owners can
    /// launch the transferred executable again.
    pub(super) fn finish_update(&mut self, id: u64,
        update: impl FnOnce(usize) -> anyhow::Result<bool>,
        retain: impl FnOnce(usize) -> anyhow::Result<usize>, invalidate: impl FnOnce(usize)) -> Option<usize> {
        let request = self.update_requests.remove(&id)?;
        assert!(!self.update_requests.values().any(|other| other.identity == request.identity),
            "duplicate graph update ticket");
        if let Some(old) = self.identities.get(&request.identity) {
            let old = &self.records[old];
            assert!(!old.pinned && !old.pending_capture && !old.pending_destroy &&
                !matches!(old.list, Some(List::Recent | List::Frequent)), "update would forget a resident graph");
        }
        let donor = self.cold_donor_for_request(id, &request);
        let Some(donor) = donor else {
            let counters = self.counters.entry(request.site.cycle).or_default();
            if request.hard { counters.no_cold_donor_hard += 1 } else { counters.no_cold_donor_soft += 1 }
            let counters = self.update_sites.entry(request.site.name).or_default().entry(request.site.cycle).or_default();
            if request.hard { counters.no_cold_donor_hard += 1 } else { counters.no_cold_donor_soft += 1 }
            self.update_requests.insert(id, request);
            return None;
        };
        let exec = self.native_graphs[&donor].exec;
        let exec_started = Instant::now();
        let accepted = update(exec).unwrap_or_else(|error| { tracing::warn!(%error, "cold graph update failed"); false });
        let exec_ns = exec_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        if !accepted {
            invalidate(exec);
            self.remove_drained(&[donor]);
            self.counters.entry(request.site.cycle).or_default().update_failures += 1;
            self.update_sites.entry(request.site.name).or_default().entry(request.site.cycle).or_default().update_failures += 1;
            self.update_requests.insert(id, request);
            return None;
        }
        let transferred = match retain(exec) {
            Ok(exec) => exec,
            Err(error) => {
                tracing::error!(%error, "transferring graph executable");
                invalidate(exec);
                self.remove_drained(&[donor]);
                self.counters.entry(request.site.cycle).or_default().update_failures += 1;
                self.update_sites.entry(request.site.name).or_default().entry(request.site.cycle).or_default().update_failures += 1;
                self.update_requests.insert(id, request);
                return None;
            }
        };
        if let Some(old) = self.identities.get(&request.identity).copied() {
            self.forget(old);
        }
        self.remove_donor_candidate(donor);
        self.unlink(donor);
        let mut native = self.take_native(donor).unwrap();
        let mut record = self.records.remove(&donor).unwrap();
        record.invalidate();
        let mut ghost = record.clone();
        ghost.bytes = 0;
        ghost.accounted = Inventory::default();
        ghost.list = None; ghost.previous = None; ghost.next = None;
        self.records.insert(donor, ghost);
        self.push(donor, List::RecentGhost);
        native.bank = request.identity.bank;
        native.launch_cycle = request.site.cycle;
        native.exec = transferred;
        native.counted_launches = 0;
        native.indexed = None;
        record.identity = request.identity.clone();
        record.cycle = request.site.cycle;
        record.pinned = request.pinned;
        record.spans_devices = request.site.spans_devices;
        record.last_used = Some(self.lanes[self.bank_lane(request.identity.bank)].index);
        record.previous = None; record.next = None; record.list = None;
        let native_nodes = record.stats.native_nodes;
        let declared_capture_streams = record.stats.declared_capture_streams;
        let max_parallel_width = record.stats.max_parallel_width;
        record.stats = KeyStats::new(self.clock_ms);
        record.stats.declared_capture_streams = declared_capture_streams;
        record.stats.max_parallel_width = max_parallel_width;
        record.stats.spans_devices = record.spans_devices;
        record.stats.event(request.site.cycle, 1);
        record.stats.misses = 1;
        record.stats.native_nodes = native_nodes;
        record.stats.capture_bytes = Some(record.bytes);
        self.site_counters(request.site.name, request.site.cycle, request.identity.bank).misses += 1;
        self.identities.insert(request.identity, id);
        self.records.insert(id, record);
        self.bank_native.entry(native.bank).or_default().insert(id);
        self.native_graphs.insert(id, native);
        if !request.pinned { self.push(id, List::Recent); }
        self.insert_donor_candidate(id);
        self.sync_inventory(id);
        self.identity_generation += 1;
        self.trim_ghosts();
        let total_ns = request.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let counters = self.counters.entry(request.site.cycle).or_default();
        counters.misses += 1;
        counters.donor_rekeys += 1;
        if request.hard { counters.updates_hard += 1 } else { counters.updates_soft += 1 }
        counters.update_ns = counters.update_ns.saturating_add(total_ns);
        counters.update_exec_ns = counters.update_exec_ns.saturating_add(exec_ns);
        let counters = self.update_sites.entry(request.site.name).or_default().entry(request.site.cycle).or_default();
        counters.donor_rekeys += 1;
        if request.hard { counters.updates_hard += 1 } else { counters.updates_soft += 1 }
        counters.update_ns = counters.update_ns.saturating_add(total_ns);
        counters.update_exec_ns = counters.update_exec_ns.saturating_add(exec_ns);
        Some(transferred)
    }

    fn cold_donor_for_request(&mut self, id: u64, _request: &UpdateRequest) -> Option<u64> {
        let signature = self.native_graphs.get(&id)?.signature.clone();
        self.cold_donor(&signature)
    }

    pub(super) fn fallback_update(&mut self, id: u64) -> Miss {
        let request = self.update_requests.remove(&id).expect("active update request");
        let signature = self.take_native(id).map(|native| native.signature);
        let identity = request.identity.clone();
        let cycle = request.site.cycle;
        let spans_devices = request.site.spans_devices;
        let result = self.miss(request.identity, request.site, request.estimate, request.pinned);
        if let Some(record) = self.identities.get(&identity).and_then(|id| self.records.get_mut(id)) {
            record.signature = signature;
        } else if let Some(signature) = signature {
            // Hard-stop fallback returns `Miss::Eager` before `miss()` creates a ghost.
            // Retain a zero-byte ghost with the learned signature so future misses at
            // hard stop avoid repeating host-only source capture unless a matching donor exists.
            let now = self.started.elapsed().as_millis() as u64;
            self.next_id += 1;
            let ghost_id = self.next_id;
            self.identities.insert(identity.clone(), ghost_id);
            self.records.insert(ghost_id, Record { token: None, last_used: None, identity, cycle, bytes: 0,
                estimate: 0, list: None, previous: None, next: None, pinned: false, pending_destroy: false,
                pending_capture: false, frequent: false, nodes: 0, node_rate: 1, graph_rate: 0, parallel_width: 1,
                spans_devices, oom: None, oom_warned: false, permanently_eager: false, signature: Some(signature),
                stats: KeyStats::new(now), accounted: Inventory::default() });
            self.push(ghost_id, List::RecentGhost);
            self.trim_ghosts();
        }
        result
    }

    pub(super) fn updating(&self, id: u64) -> bool { self.update_requests.contains_key(&id) }

    #[cfg(test)]
    pub(super) fn set_soft_limit_for_fixture(&mut self, limit: u64) { self.soft.limit = limit; }

    #[cfg(test)]
    pub(super) fn advance_for_update_fixture(&mut self, lane: usize, cycles: usize) {
        for _ in 0..cycles { self.begin_lane(lane); self.complete_lane(lane); }
    }

    pub fn miss(&mut self, identity: Identity, site: Site, node_estimate: Option<Estimate>, pinned: bool) -> Miss {
        self.refresh_cycles();
        let now = self.started.elapsed().as_millis() as u64;
        self.counters.entry(site.cycle).or_default().misses += 1;
        self.site_counters(site.name, site.cycle, identity.bank).misses += 1;
        if self.update_requests.values().any(|request| request.identity == identity) {
            self.counters.entry(site.cycle).or_default().eager_runs += 1;
            self.site_counters(site.name, site.cycle, identity.bank).eager_runs += 1;
            return Miss::Eager;
        }
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
            self.site_counters(site.name, site.cycle, identity.bank).eager_runs += 1;
            return Miss::Eager;
        }
        let lane = self.bank_lane(identity.bank);
        let vampire = self.lanes[lane].vampires.contains(&identity);
        let old = self.identities.get(&identity).copied();
        // A bank must retire its old executable before recapturing this identity.
        // Re-entering would replace its ticket while the native owner stays live.
        if old.is_some_and(|id| {
            let r = &self.records[&id];
            r.pinned || r.pending_destroy || r.pending_capture || matches!(r.list, Some(List::Recent | List::Frequent))
        }) {
            let record = self.records.get_mut(&old.unwrap()).unwrap();
            record.stats.event(site.cycle, 1);
            record.stats.misses += 1;
            self.counters.entry(site.cycle).or_default().eager_runs += 1;
            self.site_counters(site.name, site.cycle, identity.bank).eager_runs += 1;
            return Miss::Eager;
        }
        if let Some(record) = old.and_then(|id| self.records.get_mut(&id)) {
            if record.permanently_eager {
                record.stats.event(site.cycle, 1);
                record.stats.misses += 1;
                self.counters.entry(site.cycle).or_default().eager_runs += 1;
                self.site_counters(site.name, site.cycle, identity.bank).eager_runs += 1;
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
                    self.site_counters(site.name, site.cycle, identity.bank).eager_runs += 1;
                    return Miss::Eager;
                }
                record.oom = None;
            }
        }
        // The measured hard stop admits no new tickets, even nominal reuse.
        // Destruction cannot return driver arena bytes or clear this stop.
        if self.charged >= self.reserve || self.measurement_failed || self.soft.limit == 0 {
            if let Some(record) = old.and_then(|id| self.records.get_mut(&id)) {
                record.stats.event(site.cycle, 1);
                record.stats.misses += 1;
            }
            let counters = self.site_counters(site.name, site.cycle, identity.bank);
            counters.eager_runs += 1; counters.eager_hard_stop += 1;
            let counters = self.counters.entry(site.cycle).or_default();
            counters.eager_runs += 1;
            counters.eager_hard_stop += 1;
            return Miss::Eager;
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
        // #1878: once serving, a graph spanning both GPUs is never captured. Its first
        // instantiation runs while the other GPU spins on a peer wait whose push this
        // thread queues only after the instantiation returns (cudaGraphInstantiate then
        // waits for that GPU forever). The miss runs eagerly and stays a remembered
        // key, so a GRAPH_TRACE run still lists it for the layout's prewarm set.
        let sealed = site.spans_devices && cross_device_capture_sealed();
        // A physical sample covers the entire outstanding capture batch. Do
        // not mix a newer capture cycle into that batch before its proof.
        let allowed = site.preference != Preference::Eager && estimate > 0 && !sealed
            && (!self.event_completion || !self.unmeasured || self.capture_frontier == self.lanes[lane].index);
        let mut projected_hard_stop = false;
        if allowed && !vampire {
            // No CUDA query or ticket for a planned physical-headroom miss.
            if !self.physical_room(estimate, headroom) {
                let counters = self.site_counters(site.name, site.cycle, identity.bank);
                counters.eager_runs += 1; counters.physical_floor_skips += 1;
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
                        self.site_counters(site.name, site.cycle, identity.bank).eager_runs += 1;
                        return Miss::Eager;
                    }
                }
            }
            // The existing baseline query can reveal lower free memory than
            // the boundary precheck. Honor that sample without another query.
            if !self.physical_room(estimate, headroom) {
                if !self.unmeasured { self.captured_since_observation = false; }
                let counters = self.site_counters(site.name, site.cycle, identity.bank);
                counters.eager_runs += 1; counters.physical_floor_skips += 1;
                return self.floor_skip(old, site.cycle);
            }
            let new_nodes = metadata.nodes.filter(|nodes| *nodes > 0)
                .or_else(|| old.and_then(|id| self.records[&id].stats.native_nodes).filter(|n| *n > 0))
                .or_else(|| {
                    let (sum, count) = self.records.values()
                        .filter(|r| r.identity.site == site.name)
                        .filter_map(|r| r.stats.native_nodes.filter(|n| *n > 0))
                        .fold((0u64, 0u64), |(s, c), n| (s.saturating_add(n), c + 1));
                    (count > 0).then(|| sum.div_ceil(count))
                });
            // Historical physical shares are slab-sized, not the cost of a
            // replacement graph. Known native nodes that fit the frozen soft
            // capacity after retirement reuse the arena; pending/uncredited
            // nodes remain charged until their frontier completes.
            let native_reuse = chain && self.soft.measured_bytes.is_some() && new_nodes.is_some_and(|incoming| {
                let reclaimable = self.records.values().filter(|r| !r.pending_capture && !r.pinned
                    && (r.pending_destroy || (!self.hot(r) && matches!(r.list, Some(List::Recent | List::Frequent)))))
                    .map(|r| r.stats.native_nodes.unwrap_or(0)).sum::<u64>();
                self.soft.fits(self.native_pools().live_nodes.saturating_add(self.epochs.uncredited_nodes)
                    .saturating_sub(reclaimable).saturating_add(incoming))
            });
            let hard_room = (!self.event_completion && self.soft.measured_bytes.is_none()) || native_reuse
                || self.charged.saturating_add(projected) <= self.reserve;
            projected_hard_stop = !hard_room;
            let room = if !hard_room { SoftRoom::Full }
                else if self.soft.measured_bytes.is_none() { SoftRoom::Available }
                else if let Some(nodes) = new_nodes { self.make_soft_room(nodes, None) }
                else { SoftRoom::Available };
            if room != SoftRoom::Available { self.lanes[lane].vampires.insert(identity.clone()); }
            // Selected owners and lagged chunks still consume capacity. Only
            // their owner's next cycle can destroy them; no device-wide gate.
            if room == SoftRoom::Available && hard_room {
            let id = if let Some(id) = old { self.unlink(id); id } else {
                self.next_id += 1;
                let id = self.next_id;
                self.identities.insert(identity.clone(), id);
                self.records.insert(id, Record { token: None, last_used: None, identity, cycle: site.cycle, bytes: 0, estimate, list: None,
                    previous: None, next: None, pinned, pending_destroy: false, pending_capture: false, frequent: false, nodes, node_rate: metadata.node_rate, graph_rate: metadata.graph_rate, parallel_width: metadata.parallel_width, spans_devices: site.spans_devices, oom: None, oom_warned: false, permanently_eager: false, signature: None, stats: KeyStats::new(now), accounted: Inventory::default() });
                id
            };
            let record = self.records.get_mut(&id).unwrap();
            record.last_used = Some(self.lanes[lane].index);
            record.estimate = estimate; record.pinned = pinned; record.pending_capture = true;
            record.nodes = nodes; record.node_rate = metadata.node_rate;
            record.graph_rate = metadata.graph_rate; record.parallel_width = metadata.parallel_width;
            record.spans_devices |= site.spans_devices;
            if projected == 0 || native_reuse { self.reuse_captures.insert(id); }
            self.pending_growth = self.pending_growth.max(projected);
            self.unmeasured = true;
            if self.event_completion { self.capture_frontier = self.lanes[lane].index; }
            self.pools.live_nodes = next_nodes;
            self.pools.live_execs = next_execs;
            let stats = &mut record.stats;
            stats.event(site.cycle, 1);
            stats.misses += 1;
            stats.ghost_hits += u64::from(ghost.is_some());
            stats.native_nodes = None;
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

            self.sync_inventory(id);
            return Miss::Capture(id);
            }
        }
        // A cold adaptive miss is remembered without displacing resident work.
        if old.is_none() && estimate > 0 {
            self.next_id += 1;
            let id = self.next_id;
            self.identities.insert(identity.clone(), id);
            self.records.insert(id, Record { token: None, last_used: None, identity: identity.clone(), cycle: site.cycle, bytes: 0, estimate,
                list: None, previous: None, next: None, pinned: false, pending_destroy: false, pending_capture: false, frequent: false, nodes, node_rate: metadata.node_rate, graph_rate: metadata.graph_rate, parallel_width: metadata.parallel_width, spans_devices: site.spans_devices, oom: None, oom_warned: false, permanently_eager: false, signature: None, stats: KeyStats::new(now), accounted: Inventory::default() });
            self.push(id, List::RecentGhost);
            self.trim_ghosts();
        }
        if let Some(id) = self.identities.get(&identity) {
            let stats = &mut self.records.get_mut(id).unwrap().stats;
            stats.event(site.cycle, 1);
            stats.misses += 1;
            stats.ghost_hits += u64::from(ghost.is_some());
        }
        let counters = self.counters.entry(site.cycle).or_default();
        counters.eager_runs += 1;
        if sealed { counters.eager_sealed += 1; }
        else if site.preference == Preference::Eager { counters.eager_preference += 1; }
        else if vampire { counters.eager_vampire += 1; }
        else if projected_hard_stop { counters.eager_projected_hard_stop += 1; }
        else if self.soft.measured_bytes.is_some() { counters.eager_soft_room += 1; }
        else { counters.eager_other += 1; }
        let counters = self.update_sites.entry(site.name).or_default().entry(site.cycle).or_default();
        counters.eager_runs += 1;
        if sealed { counters.eager_sealed += 1; }
        else if site.preference == Preference::Eager { counters.eager_preference += 1; }
        else if vampire { counters.eager_vampire += 1; }
        else if projected_hard_stop { counters.eager_projected_hard_stop += 1; }
        else if self.soft.measured_bytes.is_some() { counters.eager_soft_room += 1; }
        else { counters.eager_other += 1; }
        Miss::Eager
    }

    /// Retain the failed identity in ARC's bounded ghost history. No executable
    /// survived, but driver growth still needs the ordinary completed sample.
    /// Returns true only for the key's first OOM warning while it is retained.
    pub fn capture_oom(&mut self, id: u64) -> bool {
        self.reuse_captures.remove(&id);
        let Some(record) = self.records.get_mut(&id) else { return false };
        if !record.pending_capture { return false }
        self.pending_bytes = self.pending_bytes.saturating_sub(record.estimate);
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
        self.sync_inventory(id);
        self.trim_ghosts();
        if self.measurement_failed { self.unmeasured = self.records.values().any(|r| r.pending_capture); }
        warn
    }

    /// Retain exact rejected identities outside ARC's evictable ghost lists.
    /// Captured commands did not run and no executable was instantiated.
    pub fn capture_too_small(&mut self, id: u64, nodes: usize) {
        self.reuse_captures.remove(&id);
        let Some(record) = self.records.get_mut(&id) else { return };
        if record.permanently_eager { return }
        if record.pending_capture {
            self.pending_bytes = self.pending_bytes.saturating_sub(record.estimate);
            self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes);
            self.pools.live_execs = self.pools.live_execs.saturating_sub(1);
        }
        record.invalidate();
        record.pending_capture = false;
        record.pinned = false;
        record.permanently_eager = true;
        record.stats.native_nodes = Some(nodes as u64);
        tracing::info!(device = self.device, site = record.identity.site,
            key = %record.identity.shape, nodes, "small graph permanently eager");
        self.unlink(id);
        let record = self.records.get_mut(&id).unwrap();
        self.live_bytes = self.live_bytes.saturating_sub(record.bytes);
        record.bytes = 0;
        self.sync_inventory(id);
        if self.measurement_failed { self.unmeasured = self.records.values().any(|r| r.pending_capture); }
        // Physical charge remains a monotone high-water, including failed captures.
    }

    pub fn cancel_capture(&mut self, id: u64) {
        self.update_requests.remove(&id);
        self.remove_donor_candidate(id);
        self.take_native(id);
        if let Some(record) = self.records.get(&id) {
            if record.pending_capture {
                self.pending_bytes = self.pending_bytes.saturating_sub(record.estimate);
                self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes);
                self.pools.live_execs = self.pools.live_execs.saturating_sub(1);
                // Keep the in-flight growth reservation until the next sample:
                // a failed instantiate may still have expanded driver storage.
            }
        }
        self.forget(id);
        if self.measurement_failed { self.unmeasured = self.records.values().any(|r| r.pending_capture); }
    }

    pub fn captured_topology(&mut self, id: u64, nodes: u64, width: usize, spans_devices: bool) {
        let Some(record) = self.records.get_mut(&id) else { return };
        // Peer pointers in kernel arguments are invisible to native inspection.
        let spans_devices = spans_devices || record.spans_devices;
        record.spans_devices = spans_devices;
        if width != 1 || spans_devices {
            self.reuse_captures.remove(&id);
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
        self.sync_inventory(id);
    }

    pub fn completed_capture(&mut self, id: u64) -> bool {
        if self.event_completion { return true }
        let Some(query) = &self.memory_query else { return true };
        let sample = query.sample(self.device);
        match sample {
            Ok((free, total)) => {
                let (tracked, rounding, unclaimed) = ledger_sample(self.device);
                let (growth, level) = self.observer.observe(free as u64, total as u64, tracked, rounding, unclaimed);
                self.level = Some(level);
                self.sample_physical(free as u64, tracked);
                self.settle_captures(growth, Some(id));
                true
            }
            Err(error) => {
                // Without a physical observation another capture cannot be safe.
                tracing::warn!(%error, device = self.device, "capture measurement failed; freezing capture");
                self.measurement_failed = true;
                let fallback = self.records.get(&id).map_or(0, |record| record.estimate);
                self.charged = self.charged.saturating_add(fallback);
                self.failed_measurement_credit = self.failed_measurement_credit.saturating_add(fallback);
                self.cancel_capture(id);
                self.unmeasured = self.records.values().any(|r| r.pending_capture);
                if !self.unmeasured { self.pending_growth = 0; self.captured_since_observation = false; }
                false
            }
        }
    }

    fn make_soft_room(&mut self, incoming_nodes: u64, excluded: Option<u64>) -> SoftRoom {
        let native = self.native_pools();
        let uncredited = self.epochs.uncredited_nodes;
        let live_total = native.live_nodes.saturating_add(uncredited);
        if self.soft.fits(live_total.saturating_add(incoming_nodes)) {
            return SoftRoom::Available;
        }
        let mut selected = self.records.values().filter(|r| r.pending_destroy)
            .map(|r| r.stats.native_nodes.unwrap_or(0)).sum::<u64>();
        let reclaimable = self.records.iter().filter(|(&id, r)| Some(id) != excluded && !r.pending_capture && !r.spans_devices
            && !self.hot(r) && matches!(r.list, Some(List::Recent | List::Frequent)))
            .map(|(_, r)| r.stats.native_nodes.unwrap_or(0)).sum::<u64>();
        let mut nodes = live_total.saturating_sub(selected);
        if incoming_nodes == 0 {
            if !self.soft.fits(nodes.saturating_sub(reclaimable)) { return SoftRoom::Full; }
            while !self.soft.fits(nodes) {
                let Some(id) = self.select_victim(excluded) else { return SoftRoom::Full; };
                nodes = nodes.saturating_sub(self.records[&id].stats.native_nodes.unwrap_or(0));
            }
            return SoftRoom::Queued;
        }
        // Size the victim batch to this cycle's accumulated unfulfilled capture
        // demand, clamped to <= min(total_reclaimable, soft.measured_nodes / 4)
        // (with a floor of incoming_nodes so tiny fixtures can still admit one graph).
        let total_reclaimable = selected.saturating_add(reclaimable);
        let quarter = self.soft.measured_nodes.map_or(u64::MAX, |m| m / 4);
        let batch_cap = total_reclaimable.min(quarter.max(incoming_nodes));
        let target_demand = self.cycle_demand_nodes.saturating_add(incoming_nodes);
        if !self.soft.fits(live_total.saturating_sub(batch_cap).saturating_add(target_demand)) {
            return SoftRoom::Full;
        }
        while !self.soft.fits(nodes.saturating_add(target_demand)) {
            if selected >= batch_cap { return SoftRoom::Full; }
            let Some(id) = self.select_victim(excluded) else { return SoftRoom::Full; };
            let vnodes = self.records[&id].stats.native_nodes.unwrap_or(0);
            selected = selected.saturating_add(vnodes);
            nodes = nodes.saturating_sub(vnodes);
        }
        self.cycle_demand_nodes = target_demand;
        SoftRoom::Queued
    }

    fn apply_reserve_policy(&mut self, policy: cuteafd_loader::placement::graph_reserve::Resolved) {
        if let Some(headroom) = policy.growth_headroom {
            self.soft.margin = self.soft.margin.max(headroom);
            self.soft.limit = self.reserve - self.soft.margin;
        }
        self.reserve_policy = Some(policy);
    }

    fn apply_soft_pressure(&mut self, growth: u64, completed: Option<u64>) {
        if self.charged >= self.reserve { return }
        let native = self.native_pools();
        if self.soft.measured_bytes.is_none() {
            let requested = self.soft.margin.max(capture_growth_headroom(growth, self.soft.granule));
            let maximum = self.reserve / 2;
            if requested > maximum && !self.margin_clamp_warned {
                tracing::warn!(device = self.device, hard = self.reserve, requested_margin = requested,
                    clamped_margin = maximum, "graph margin clamped to preserve half the reserve as soft capacity");
                self.margin_clamp_warned = true;
            }
            let margin = requested.min(maximum);
            if let Some(policy) = self.reserve_policy.filter(|p| p.growth_headroom.is_some()) {
                if margin > self.soft.margin {
                    tracing::warn!(device = self.device, growth, census_soft = policy.base_bytes,
                        previous_soft = self.soft.limit, soft = self.reserve - margin,
                        "observed graph growth moved soft cap below the census target");
                }
            }
            self.soft.margin = margin;
            self.soft.limit = self.reserve.saturating_sub(self.soft.margin);
            if self.charged < self.soft.limit || native.live_nodes == 0 || self.unmeasured { return }
            let pinned_nodes = self.records.values().filter(|r| r.pinned || r.spans_devices)
                .map(|r| r.stats.native_nodes.unwrap_or(0)).sum::<u64>();
            // A pinned-heavy startup sample cannot establish a usable ARC budget.
            let typical = self.records.values().filter(|r| !r.pinned && !r.spans_devices && !r.pending_capture
                && matches!(r.list, Some(List::Recent | List::Frequent)))
                .filter_map(|r| r.stats.native_nodes).filter(|nodes| *nodes > 0).max().unwrap_or(1);
            if u128::from(pinned_nodes.saturating_add(typical)) * u128::from(self.charged)
                > u128::from(self.soft.limit) * u128::from(native.live_nodes) { return }
            self.soft.measured_bytes = Some(self.charged);
            self.soft.measured_nodes = Some(native.live_nodes);
        }
        if self.charged < self.soft.limit { return }
        let before = self.pending_destroy;
        self.make_soft_room(0, completed);
        if self.pending_destroy > before {
            if let Some(identity) = completed.and_then(|id| self.records.get(&id)).map(|r| r.identity.clone()) {
                let lane = self.bank_lane(identity.bank);
                self.lanes[lane].vampires.insert(identity);
            }
        }
    }

    /// Called once per device after the complete multi-rank step has drained.
    /// `bytes` is the measured net graph allocation, not a sum of estimates.
    pub fn measured_capture(&mut self, bytes: u64) { self.settle_captures(bytes, None); }

    fn settle_captures(&mut self, bytes: u64, completed: Option<u64>) {
        let pending: Vec<_> = self.records.iter().filter(|(&id, r)| r.pending_capture && completed.is_none_or(|done| id == done))
            .map(|(&id, r)| (id, r.estimate)).collect();
        if pending.is_empty() && !self.unmeasured && bytes == 0 { return }
        // Only successful owners surviving this completed boundary prove reuse.
        let outstanding: Vec<_> = self.records.iter().filter(|(&id, r)| r.pending_capture && completed.is_some_and(|done| id != done)).collect();
        let outstanding_nodes = outstanding.iter().map(|(_, r)| r.nodes).sum::<u64>();
        self.pools.node_high_water = self.pools.node_high_water.max(self.pools.live_nodes.saturating_sub(outstanding_nodes));
        self.pools.exec_high_water = self.pools.exec_high_water.max(self.pools.live_execs.saturating_sub(outstanding.len() as u64));
        let settled_bytes = pending.iter().map(|(_, bytes)| *bytes).sum::<u64>();
        let total = settled_bytes.max(1);
        let mut remaining = bytes;
        for (index, &(id, estimated)) in pending.iter().enumerate() {
            let share = if index + 1 == pending.len() { remaining }
                else { ((bytes as u128 * estimated as u128) / total as u128) as u64 };
            remaining = remaining.saturating_sub(share);
            let list = self.records[&id].list;
            self.unlink(id);
            let record = self.records.get_mut(&id).unwrap();
            record.pending_capture = false;
            if self.reuse_captures.remove(&id) && share > 0 {
                self.epochs.reuse_growth_bytes = self.epochs.reuse_growth_bytes.saturating_add(share);
                tracing::warn!(device = self.device, id, share, "graph capture grew driver arena despite credited native capacity");
            }
            record.stats.capture_bytes = Some(share);
            record.stats.captures += 1;
            if record.stats.evicted_since_capture { record.stats.recaptures += 1; }
            record.stats.evicted_since_capture = false;
            // Zero physical growth can mean allocator reuse, not a zero-size
            // graph. Keep its estimate for victim ordering, never for charge.
            self.live_bytes = self.live_bytes.saturating_sub(record.bytes).saturating_add(share);
            record.bytes = share;
            let class = self.classes.entry(record.identity.site).or_default();
            class.samples += 1;
            if share > 0 { class.total = class.total.saturating_add(share); class.largest = class.largest.max(share); }
            self.counters.entry(record.cycle).or_default().captures += 1;
            self.update_sites.entry(record.identity.site).or_default().entry(record.cycle).or_default().captures += 1;
            if let Some(list) = list { self.push(id, list); }
            self.sync_inventory(id);
            self.insert_donor_candidate(id);
        }
        self.pending_bytes = self.pending_bytes.saturating_sub(settled_bytes);
        self.unmeasured = self.records.values().any(|r| r.pending_capture);
        if !self.unmeasured { self.pending_growth = 0; self.captured_since_observation = false; }
        let credited = bytes.min(self.failed_measurement_credit);
        self.failed_measurement_credit -= credited;
        self.charged = self.charged.saturating_add(bytes - credited);
        self.apply_soft_pressure(bytes, completed);

    }

    pub fn destroyed(&mut self, ids: &[u64], measurement: ReleaseMeasurement) {
        self.refresh_cycles();
        if ids.iter().any(|id| self.records.contains_key(id)) { self.floor_resample_used = false; }
        // Observed free deltas remain available to alternate accounting adapters;
        // the default driver-arena model never credits physical release.
        let _ = measurement.released;
        if ids.iter().any(|id| self.records.contains_key(id)) { self.recovery_generation += 1; }
        self.retire_native_counts(ids);
        for &id in ids {
            self.unlink(id);
            let owner = self.records.get(&id).map(|record| self.bank_lane(record.identity.bank));
            let until = self.selected_epochs.remove(&id).map(|mut until| {
                for (index, (epoch, lane)) in until.iter_mut().zip(&self.lanes).enumerate() {
                    // The owner's new cycle has not launched yet: retirement
                    // runs first. Peers may already be replaying retained keys.
                    if Some(index) != owner && lane.active { *epoch = (*epoch).max(lane.index); }
                }
                until
            });
            if let Some(record) = self.records.get(&id).filter(|r| r.pending_destroy) {
                let lane = self.bank_lane(record.identity.bank);
                self.pending_lanes[lane] = self.pending_lanes[lane].checked_sub(1)
                    .expect("graph pending lane retirement mismatch");
                self.clocks[lane].pending.fetch_sub(1, Ordering::AcqRel);
            }
            let Some(record) = self.records.get_mut(&id) else { continue };
            let evicted = record.pending_destroy;
            record.invalidate();
            self.live_bytes = self.live_bytes.saturating_sub(record.bytes);
            record.estimate = record.ordering_bytes();
            record.pending_destroy = false;
            record.pinned = false;
            if evicted { self.pending_destroy -= 1; }
            if let Some(mut until) = until.filter(|until| self.lanes.iter().zip(*until).any(|(lane, epoch)| lane.completed < epoch)) {
                for (epoch, lane) in until.iter_mut().zip(&self.lanes) {
                    if lane.completed >= *epoch { *epoch = 0; }
                }
                // Coalesce by in-flight frontier: at most K independently
                // advancing cycles retain capacity, rather than one chunk/key.
                if let Some(chunk) = self.lagged.iter_mut().find(|chunk| chunk.until == until) {
                    chunk.nodes += record.stats.native_nodes.unwrap_or(0);
                    chunk.pool_nodes += record.nodes;
                    chunk.execs += 1;
                } else {
                    self.lagged.push(LaggedCredit { until, nodes: record.stats.native_nodes.unwrap_or(0),
                        pool_nodes: record.nodes, execs: 1 });
                }
            } else {
                self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes);
                self.pools.live_execs = self.pools.live_execs.saturating_sub(1);
            }
            if evicted {
                record.stats.evictions += 1;
                record.stats.evicted_since_capture = true;
                self.counters.entry(record.cycle).or_default().evictions += 1;
                self.update_sites.entry(record.identity.site).or_default().entry(record.cycle).or_default().evictions += 1;
                self.interval_evictions += 1;
            }
            // Keep the ghost's recency/frequency origin through retirement.
            let ghost = if record.frequent { List::FrequentGhost } else { List::RecentGhost };
            self.push(id, ghost);
            self.records.get_mut(&id).unwrap().bytes = 0;
            self.sync_inventory(id);
        }
        if self.pending_destroy == 0 { self.cycle_demand_nodes = 0; }
        self.credit_completed();
        if !self.lanes.iter().any(|lane| lane.active) { self.serial_boundary(); }
        self.trim_ghosts();
    }

    pub fn remove_bank(&mut self, bank: u64) {
        // The first retiring bank still sees the complete cache. Preserve that
        // final view until serving resumes, rather than accumulating retired rows.
        if self.trace.is_some() {
            if self.retired_trace.is_none() { self.retired_trace = Some(self.trace_snapshot()); }
            // Each retiring bank flushes its last pinned hits just before here.
            let (cycles, update_sites) = self.launch_counts();
            let per_cycle = self.per_cycle(&update_sites);
            if let Some(snapshot) = &mut self.retired_trace {
                for row in snapshot.records.iter_mut().filter(|r| r.identity.bank == bank) {
                    if let Some(record) = self.identities.get(&row.identity).and_then(|id| self.records.get(id)) {
                        *row = RawRecord::copy(record);
                    }
                }
                snapshot.cache.cycles = cycles;
                snapshot.cache.update_sites = update_sites;
                snapshot.cache.per_cycle = per_cycle;
            }
        }
        let ids: Vec<_> = self.records.iter().filter(|(_, r)| r.identity.bank == bank).map(|(&id, _)| id).collect();
        self.remove_drained(&ids);
        self.bank_lanes.remove(&bank);
    }

    /// Forget destroyed owners, never crediting physical driver-arena release.
    pub fn remove_drained(&mut self, ids: &[u64]) {
        if ids.iter().any(|id| self.records.contains_key(id)) { self.floor_resample_used = false; }
        if ids.iter().any(|id| self.records.get(id).is_some_and(|record|
            record.pinned || record.pending_destroy || matches!(record.list, Some(List::Recent | List::Frequent)))) {
            self.recovery_generation += 1;
        }
        for &id in ids {
            self.selected_epochs.remove(&id);
            self.reuse_captures.remove(&id);
            let Some(record) = self.records.get(&id) else { continue };
            if record.pending_destroy {
                let lane = self.bank_lane(record.identity.bank);
                self.pending_lanes[lane] = self.pending_lanes[lane].checked_sub(1)
                    .expect("graph pending lane retirement mismatch");
                self.clocks[lane].pending.fetch_sub(1, Ordering::AcqRel);
            }
            if record.pending_capture {
                self.pending_bytes = self.pending_bytes.saturating_sub(record.estimate);
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
            reserve = self.reserve, charged = self.charged, live_bytes = self.reported_live_bytes(), resident_bytes = self.resident_bytes(), cycles = ?self.counters, "graph cache interval");
        self.interval_evictions = 0;
        self.last_report = Instant::now();
        true
    }

    pub fn boundary(&mut self) {
        self.sample_clock();
        self.retired_trace = None;
        if self.ranked_at.elapsed() >= Duration::from_secs(30) { self.refresh_rankings(); }
        self.report_due(self.last_report.elapsed());
        if self.trace.is_some() && self.last_trace.elapsed() >= Duration::from_secs(1) {
            self.last_trace = Instant::now();
            self.publish_trace();
        }
    }

    fn trace_snapshot(&self) -> TraceSnapshot {
        let (cycles, update_sites) = self.launch_counts();
        let per_cycle = self.per_cycle(&update_sites);
        let records = self.records.values().map(RawRecord::copy).collect();
        TraceSnapshot { cache: Snapshot { device: self.device, reserve: self.reserve, physical_floor: self.physical_floor,
            reserve_policy: self.reserve_policy, charged: self.charged, live_bytes: self.reported_live_bytes(), resident_estimated_bytes: self.resident_bytes(), hard_stopped: self.charged >= self.reserve || self.measurement_failed || self.soft.limit == 0, native_pools: self.native_pools(), soft: self.soft, level: self.level,
            pinned: self.inventory.pinned, spanning_bytes: self.inventory.spanning_bytes, spanning_nodes: self.inventory.spanning_nodes, spanning_execs: self.inventory.spanning_execs,
            pending_capture: self.pending_bytes, cycles, pools: self.pools.clone(),
            basis: self.basis, sampled_at_ms: self.started.elapsed().as_millis() as u64,
            telemetry_bytes: self.records.len() * std::mem::size_of::<KeyStats>(),
            top_hits: Vec::new(), top_churn: Vec::new(), rankings_age_ms: self.ranked_at.elapsed().as_millis() as u64, epochs: self.epoch_snapshot(), per_cycle, update_sites },
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

    fn refresh_rankings(&mut self) {
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
        let hits = ranked(false);
        let churn = ranked(true);
        self.ranked_hits = hits;
        self.ranked_churn = churn;
        self.ranked_at = Instant::now();
    }

    fn epoch_snapshot(&self) -> EpochStats {
        let mut epochs = self.epochs.clone();
        epochs.pending = self.pending_lanes;
        epochs.lagged_chunks = self.lagged.len();
        epochs.cycles = self.clocks.as_ref().each_ref().map(|clock| clock.index.load(Ordering::Acquire));
        epochs.completed = self.clocks.as_ref().each_ref().map(|clock| clock.completed.load(Ordering::Acquire));
        epochs
    }

    fn snapshot(&self) -> Snapshot {
        let (cycles, update_sites) = self.launch_counts();
        let per_cycle = self.per_cycle(&update_sites);
        let now = self.started.elapsed().as_millis() as u64;
        Snapshot { device: self.device, reserve: self.reserve, physical_floor: self.physical_floor, reserve_policy: self.reserve_policy, charged: self.charged, live_bytes: self.reported_live_bytes(), resident_estimated_bytes: self.resident_bytes(), hard_stopped: self.charged >= self.reserve || self.measurement_failed || self.soft.limit == 0, native_pools: self.native_pools(), soft: self.soft,
            level: self.level,
            pinned: self.inventory.pinned, spanning_bytes: self.inventory.spanning_bytes, spanning_nodes: self.inventory.spanning_nodes, spanning_execs: self.inventory.spanning_execs,
            pending_capture: self.pending_bytes, cycles, pools: self.pools.clone(), basis: self.basis,
            sampled_at_ms: now, telemetry_bytes: self.records.len() * std::mem::size_of::<KeyStats>(), top_hits: self.ranked_hits.clone(), top_churn: self.ranked_churn.clone(), rankings_age_ms: self.ranked_at.elapsed().as_millis() as u64, epochs: self.epoch_snapshot(), per_cycle, update_sites }
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
            executables: u64::from(record.resident && !record.permanently_eager), permanently_eager: record.permanently_eager, basis,
            permanent_bytes: if record.permanently_eager || !record.resident { 0 } else { match basis {
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
        .values().filter(|cache| {
            #[cfg(test)]
            { return TEST_CONTROLLERS.with(|owned| owned.borrow().iter().any(|weak| std::sync::Weak::ptr_eq(weak, cache))); }
            #[cfg(not(test))]
            { let _ = cache; true }
        }).filter_map(std::sync::Weak::upgrade).collect();
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
    resident: bool,
    stats: KeyStats,
}
impl RawRecord {
    fn copy(record: &Record) -> Self {
        Self { identity: record.identity.clone(), cycle: record.cycle, bytes: record.ordering_bytes(),
            nodes: record.nodes, pinned: record.pinned, permanently_eager: record.permanently_eager,
            resident: record.pinned || record.pending_capture || record.pending_destroy || matches!(record.list, Some(List::Recent | List::Frequent)),
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
fn test_cache(device: i32, reserve: u64) -> DeviceCache {
    let mut cache = DeviceCache::new(device, reserve);
    if reserve < GRAPH_CHUNK {
        cache.soft.granule = 1;
        cache.soft.margin = 1;
        cache.soft.limit = reserve.saturating_sub(1);
    }
    cache
}

#[cfg(test)]
mod tests {
    use super::*;
    fn site(preference: Preference) -> Site { Site { name: "test", cycle: Cycle::Decode, preference, bootstrap: 10, basis: Basis::Bytes, spans_devices: false } }
    fn key(n: u64) -> Identity { Identity { bank: 1, site: "test", key: n, shape: n.to_string().into() } }
    fn capture(cache: &mut DeviceCache, n: u64, pinned: bool) -> u64 {
        let Miss::Capture(id) = cache.miss(key(n), site(Preference::Capture), None, pinned) else { panic!("capture expected") };
        cache.measured_capture(10);
        cache.serial_boundary();
        cache.serial_boundary();
        id
    }
    fn epoch_fixture() -> DeviceCache {
        let mut cache = test_cache(-1020, 1000);
        for n in 0..4 {
            let id = capture(&mut cache, n, false);
            cache.captured_topology(id, 10, 1, false);
        }
        cache.soft.measured_bytes = Some(100);
        cache.soft.measured_nodes = Some(40);
        cache.soft.limit = 100;
        cache.charged = 100;
        cache
    }

    fn aged_update_fixture(age: u64, hard: bool) -> (DeviceCache, u64) {
        let mut cache = test_cache(-1030, 1000);
        let donor = capture(&mut cache, 1, false);
        let last = cache.lanes[0].index;
        cache.records.get_mut(&donor).unwrap().last_used = Some(last);
        cache.register_native(donor, 101, vec![3, 7, 11]);
        while cache.lanes[0].index < last + age {
            cache.begin_lane(0); cache.complete_lane(0);
        }
        cache.soft.limit = 10;
        cache.charged = if hard { 1000 } else { 10 };
        (cache, donor)
    }
    #[test]
    fn event_completion_gates_donors_and_retirement_independent_of_age() {
        let (mut cache, donor) = aged_update_fixture(5, false);
        cache.event_completion = true;
        let used = cache.records[&donor].used().unwrap();
        cache.clocks[0].completed.store(used - 1, Ordering::Release);
        cache.refresh_cycles();
        assert!(cache.donor_age(donor).is_none());
        cache.serial_boundary();
        assert_eq!(cache.lanes[0].completed, used - 1);
        cache.clocks[0].completed.store(used, Ordering::Release);
        cache.refresh_cycles();
        assert!(cache.donor_age(donor).is_some());
        assert_eq!(cache.mark_victim(false), Some(donor));
        assert!(!cache.persistent_victim_ready(donor));
        cache.clocks[0].completed.store(cache.lanes[0].index, Ordering::Release);
        cache.refresh_cycles();
        assert!(cache.persistent_victim_ready(donor));
    }

    #[test]
    fn event_capture_measurement_is_deferred_until_cycle_proof() -> anyhow::Result<()> {
        let device = -28105;
        let controller = device_cache_sampled(device, 1000, Basis::Bytes, || Ok((1000, 1000, 0, 0, 0)))?;
        {
            let mut cache = controller.lock().unwrap();
            cache.event_completion = true;
            cache.clocks[0].index.store(1, Ordering::Release);
            let Miss::Capture(id) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!("capture expected") };
            assert!(cache.completed_capture(id));
            assert!(cache.records[&id].pending_capture);
            assert_eq!(cache.charged, 0);
        }
        observe_device_sampled(device, || panic!("premature measurement"))?;
        {
            let mut cache = controller.lock().unwrap();
            cache.clocks[0].completed.store(1, Ordering::Release);
            cache.refresh_cycles();
        }
        observe_device_sampled(device, || Ok((990, 1000, 0, 0, 0)))?;
        let cache = controller.lock().unwrap();
        assert_eq!(cache.charged, 10);
        assert_eq!(cache.pending_bytes, 0);
        Ok(())
    }

    #[test]
    fn spanning_residents_are_unevictable_before_and_after_ready() {
        let (mut cache, spanning) = aged_update_fixture(5, false);
        cache.records.get_mut(&spanning).unwrap().spans_devices = true;
        cache.captured_topology(spanning, 8, 1, true);
        cache.soft.limit = 900;
        let local = capture(&mut cache, 2, false);
        cache.captured_topology(local, 8, 1, false);
        cache.advance_for_update_fixture(0, 5);
        assert_eq!(cache.mark_victim(false), Some(local));
        assert!(cache.mark_victim(false).is_none(), "startup cannot select a future sealed victim");
        seal_cross_device_capture_on_this_thread(true);
        let victim = cache.mark_victim(false);
        seal_cross_device_capture_on_this_thread(false);
        assert!(victim.is_none());
        assert!(cache.owns(spanning, 1));
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.spanning_execs, 1);
        assert_eq!(snapshot.spanning_nodes, 8);
        assert_eq!(snapshot.spanning_bytes, 10);
        cache.remove_drained(&[spanning]);
        assert_eq!(cache.snapshot().spanning_execs, 0, "explicit drained teardown still releases ownership");
    }

    #[test]
    fn donor_always_preserves_spanning_prewarm_set_at_ready() {
        let (mut cache, first) = aged_update_fixture(5, false);
        cache.records.get_mut(&first).unwrap().spans_devices = true;
        cache.donor_always = true;
        cache.soft.limit = 900;
        let mut spanning = site(Preference::Capture);
        spanning.spans_devices = true;
        seal_cross_device_capture_on_this_thread(false);
        assert!(cache.update_ticket(key(2), spanning, None, false).is_none());
        assert!(cache.update_ticket(key(3), site(Preference::Capture), None, false).is_none());
        let Miss::Capture(second) = cache.miss(key(2), spanning, None, false) else { panic!("prewarm capture") };
        cache.measured_capture(10);
        cache.register_native(second, 202, vec![3, 7, 11]);
        seal_cross_device_capture_on_this_thread(true);
        let intact = cache.owns(first, 1) && cache.owns(second, 1);
        seal_cross_device_capture_on_this_thread(false);
        assert!(intact, "readiness retains both spanning geometries instead of re-keying the first");
        assert_eq!(cache.counters[&Cycle::Decode].donor_rekeys, 0);
    }

    #[test]
    fn sealed_spanning_donor_cannot_transfer_to_a_local_requester() {
        let (mut cache, donor) = aged_update_fixture(5, false);
        cache.records.get_mut(&donor).unwrap().spans_devices = true;
        cache.donor_always = true;
        seal_cross_device_capture_on_this_thread(true);
        let result = cache.update_ticket(key(2), site(Preference::Capture), None, false);
        seal_cross_device_capture_on_this_thread(false);
        assert!(result.is_none());
        assert!(cache.owns(donor, 1));
    }

    #[test]
    fn donor_always_reuses_below_soft_without_changing_default() {
        let (mut cache, donor) = aged_update_fixture(5, false);
        cache.soft.limit = 500;
        assert!(cache.update_ticket(key(2), site(Preference::Capture), None, false).is_none());
        cache.donor_always = true;
        let ticket = cache.update_ticket(key(2), site(Preference::Capture), None, false).unwrap();
        cache.register_native(ticket, 0, vec![3, 7, 11]);
        assert_eq!(cache.finish_update(ticket, |_| Ok(true), |_| Ok(202), |_| panic!()), Some(202));
        assert!(!cache.owns(donor, 1));
        assert!(cache.owns(ticket, 1));
        assert_eq!(cache.charged, 10);
    }

    #[test]
    fn donor_cold_lag_boundary_applies_only_below_soft_cap() {
        for k in [1, 2] {
            for lag in [0, 16] {
                for pressure in [false, true] {
                    for age in 0..=k + lag + 1 {
                        let (mut cache, donor) = aged_update_fixture(age, false);
                        cache.donor_always = true;
                        cache.donor_cold_lag = lag;
                        cache.lane_max_cycles[0] = k;
                        cache.soft.limit = if pressure { cache.charged } else { 500 };
                        let eligible = age > k + if pressure { 0 } else { lag };
                        assert_eq!(cache.hot(&cache.records[&donor]), age <= k);
                        assert_eq!(cache.cold_donor(&[3, 7, 11]), eligible.then_some(donor));
                        let ticket = cache.update_ticket(key(2), site(Preference::Capture), None, false);
                        assert_eq!(ticket.is_some(), eligible, "K={k}, lag={lag}, age={age}, pressure={pressure}");
                        if let Some(ticket) = ticket {
                            cache.register_native(ticket, 0, vec![3, 7, 11]);
                            assert_eq!(cache.finish_update(ticket, |_| Ok(true), |_| Ok(202), |_| panic!()), Some(202));
                            assert!(!cache.owns(donor, 1));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn donor_cold_lag_does_not_change_hard_pressure_or_overflow() {
        for k in [1, 2] {
            for age in 0..=k + 1 {
                let (mut cache, donor) = aged_update_fixture(age, true);
                cache.donor_always = true;
                cache.donor_cold_lag = u64::MAX;
                cache.lane_max_cycles[0] = k;
                assert_eq!(cache.cold_donor(&[3, 7, 11]), (age > k).then_some(donor));
            }
        }
        let (mut cache, _) = aged_update_fixture(40, false);
        cache.donor_always = true;
        cache.donor_cold_lag = u64::MAX;
        cache.soft.limit = 500;
        assert!(cache.cold_donor(&[3, 7, 11]).is_none(), "K + lag must not wrap below the safety floor");
    }

    #[test]
    fn donor_cold_lag_is_rechecked_before_transfer() {
        let (mut cache, donor) = aged_update_fixture(5, false);
        cache.donor_always = true;
        cache.donor_cold_lag = 16;
        let ticket = cache.update_ticket(key(2), site(Preference::Capture), None, false).unwrap();
        cache.register_native(ticket, 0, vec![3, 7, 11]);
        cache.soft.limit = 500;
        assert!(cache.finish_update(ticket, |_| panic!("donor is no longer cold"), |_| panic!(), |_| panic!()).is_none());
        assert!(cache.owns(donor, 1));
    }

    #[test]
    fn lazy_hit_at_queue_head_does_not_hide_a_cold_donor() {
        let (mut cache, older) = aged_update_fixture(5, false);
        cache.soft.limit = 999;
        let younger = capture(&mut cache, 2, false);
        cache.register_native(younger, 303, vec![3, 7, 11]);
        cache.advance_for_update_fixture(0, 5);
        let token = cache.hit_token(older);
        token.hit(cache.lanes[0].index);
        cache.soft.limit = cache.charged;
        let ticket = cache.update_ticket(key(3), site(Preference::Capture), None, false)
            .expect("unknown requester repairs touched FIFO head before precheck");
        cache.register_native(ticket, 0, vec![3, 7, 11]);
        assert_eq!(cache.finish_update(ticket, |exec| { assert_eq!(exec, 303); Ok(true) }, |_| Ok(404), |_| panic!()), Some(404));
        assert!(cache.owns(older, 1));
    }

    #[test]
    fn launch_cycle_switch_preserves_capture_cycle_and_other_banks() {
        let (mut cache, id) = aged_update_fixture(5, false);
        cache.bank_cycle(2, Cycle::Verify);
        assert_eq!(cache.native_graphs[&id].launch_cycle, Cycle::Decode);
        cache.bank_cycle(1, Cycle::Verify);
        assert_eq!(cache.records[&id].cycle, Cycle::Decode);
        assert_eq!(cache.native_graphs[&id].launch_cycle, Cycle::Verify);
    }

    fn update_request(cache: &mut DeviceCache, key: Identity) -> u64 {
        cache.next_id += 1;
        let id = cache.next_id;
        cache.update_requests.insert(id, UpdateRequest { identity: key, site: site(Preference::Capture),
            estimate: None, hard: cache.charged >= cache.reserve, pinned: false, started: Instant::now() });
        cache.register_native(id, 0, vec![3, 7, 11]);
        id
    }

    #[test]
    fn measured_ordering_share_replaces_estimate_even_when_smaller() {
        let (mut cache, donor) = aged_update_fixture(2, false);
        cache.unlink(donor);
        let record = cache.records.get_mut(&donor).unwrap();
        record.estimate = 100;
        record.bytes = 10;
        assert_eq!(record.ordering_bytes(), 10);
        cache.push(donor, List::Recent);
        assert_eq!(cache.queues[List::Recent.index()].bytes, 10);
        cache.unlink(donor);
        cache.records.get_mut(&donor).unwrap().bytes = 0;
        assert_eq!(cache.records[&donor].ordering_bytes(), 100);
    }

    #[test]
    fn small_source_only_capture_is_permanently_eager_above_soft_cap() {
        let (mut cache, _) = aged_update_fixture(2, false);
        let identity = key(2);
        let id = cache.update_ticket(identity.clone(), site(Preference::Capture), None, false).unwrap();
        cache.source_capture_failed(id, Some(2));
        cache.capture_too_small(id, 2);
        assert!(cache.records[&id].permanently_eager);
        for _ in 0..10 {
            assert!(cache.update_ticket(identity.clone(), site(Preference::Capture), None, false).is_none());
            assert!(matches!(cache.miss(identity.clone(), site(Preference::Capture), None, false), Miss::Eager));
        }
    }

    #[test]
    fn hard_cap_empty_and_mismatched_signatures_never_capture() {
        for signature in [Some(vec![]), Some(vec![42])] {
            let (mut cache, donor) = aged_update_fixture(2, false);
            cache.captured_topology(donor, 10, 1, false);
            // Remember a ghost while soft-limited, as an earlier failed donor
            // search would do, then reach the measured hard stop.
            cache.soft.measured_bytes = Some(10);
            cache.soft.measured_nodes = Some(10);
            assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
            let id = cache.identities[&key(2)];
            assert_eq!(cache.records[&id].list, Some(List::RecentGhost));
            cache.records.get_mut(&id).unwrap().signature = signature;
            cache.charged = cache.reserve;
            for _ in 0..10 {
                assert!(cache.update_ticket(key(2), site(Preference::Capture), None, false).is_none());
                assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
            }
            assert!(cache.update_requests.is_empty());
        }
    }

    #[test]
    fn hard_cap_unseen_shape_learns_signature_and_reuses_oldest_cold_donor() {
        let mut cache = test_cache(-1088, 1000);
        cache.soft.limit = 500;
        let pinned = capture(&mut cache, 1, true);
        cache.register_native(pinned, 100, vec![7, 1]);
        let oldest = capture(&mut cache, 2, false);
        cache.register_native(oldest, 101, vec![7, 1]);
        cache.advance_for_update_fixture(0, 2);
        let newer = capture(&mut cache, 3, false);
        cache.register_native(newer, 102, vec![7, 1]);
        cache.advance_for_update_fixture(0, 2);
        cache.charged = cache.reserve;
        // Brand-new key(4) has never been seen: host-only source capture learns its
        // signature at hard stop and transfers the oldest unpinned matching donor (exec 101).
        let id = cache.update_ticket(key(4), site(Preference::Capture), None, false)
            .expect("unseen shape at hard stop should probe cold donor");
        cache.register_native(id, 0, vec![7, 1]);
        let mut updated_exec = None;
        let transferred = cache.finish_update(id, |exec| { updated_exec = Some(exec); Ok(true) },
            |exec| Ok(exec + 1000), |_| panic!());
        assert_eq!(updated_exec, Some(101), "must select oldest unpinned donor, never pinned (100) or newer (102)");
        assert_eq!(transferred, Some(1101));
        assert_eq!(cache.charged, cache.reserve, "in-place donor transfer allocates zero arena bytes");
        assert!(cache.owns(pinned, 1) && cache.owns(newer, 1) && !cache.owns(oldest, 1));
        assert_eq!(cache.counters[&Cycle::Decode].updates_hard, 1);

        // Brand-new key(5) with a mismatched topology fails update once, caches its
        // signature on a zero-byte ghost, runs eager, and never re-captures while no donor matches.
        let id2 = cache.update_ticket(key(5), site(Preference::Capture), None, false)
            .expect("first miss of unseen shape learns signature via host-only capture");
        cache.register_native(id2, 0, vec![99]);
        assert!(cache.finish_update(id2, |_| panic!(), |_| panic!(), |_| panic!()).is_none());
        assert!(matches!(cache.fallback_update(id2), Miss::Eager));
        assert_eq!(cache.charged, cache.reserve);
        let ghost_id = cache.identities[&key(5)];
        assert_eq!(cache.records[&ghost_id].signature.as_deref(), Some(&[99][..]));
        for _ in 0..5 {
            assert!(cache.update_ticket(key(5), site(Preference::Capture), None, false).is_none());
        }
    }

    #[test]
    fn d2d_memcpy_signature_is_indexed_and_donor_updated_not_eagered() {
        let mut cache = test_cache(-1087, 1000);
        let d2d_sig = |device: u64| vec![3, 0, 0x101, 1, 3, device, 0, 0x202, 2, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0];
        let donor = capture(&mut cache, 1, false);
        cache.records.get_mut(&donor).unwrap().last_used = Some(cache.lanes[0].index);
        cache.register_native(donor, 501, d2d_sig(0));
        cache.soft.limit = 10;
        cache.charged = 10;
        for _ in 0..2 { cache.begin_lane(0); cache.complete_lane(0); }
        // Matching D2D memcpy signature (including across different 1D row byte counts): indexed and donor-updated, not eagered.
        let id = cache.update_ticket(key(2), site(Preference::Adaptive), None, false)
            .expect("graph with D2D memcpy signature must get a donor update ticket");
        cache.register_native(id, 0, d2d_sig(0));
        let mut updated_exec = None;
        let transferred = cache.finish_update(id, |exec| { updated_exec = Some(exec); Ok(true) },
            |exec| Ok(exec + 1000), |_| panic!());
        assert_eq!(updated_exec, Some(501));
        assert_eq!(transferred, Some(1501));
        assert!(cache.owns(id, 1) && !cache.owns(donor, 1));
        assert_eq!(cache.counters[&Cycle::Decode].updates_soft, 1);
        assert_eq!(cache.counters[&Cycle::Decode].eager_soft_room, 0);
        assert_eq!(cache.counters[&Cycle::Decode].no_cold_donor_soft, 0);
        // Different device signature must not steal a mismatched-device donor.
        cache.records.get_mut(&id).unwrap().last_used = Some(cache.lanes[0].index);
        for _ in 0..2 { cache.begin_lane(0); cache.complete_lane(0); }
        cache.charged = cache.reserve;
        let id_diff = cache.update_ticket(key(3), site(Preference::Adaptive), None, false)
            .expect("unseen shape probes signature via host-only capture");
        cache.register_native(id_diff, 0, d2d_sig(1));
        assert!(cache.finish_update(id_diff, |_| panic!(), |_| panic!(), |_| panic!()).is_none());
        assert!(matches!(cache.fallback_update(id_diff), Miss::Eager));
        assert!(cache.owns(id, 1), "mismatched device must leave existing donor intact");
    }

    #[test]
    fn working_set_reserve_seeds_adaptive_headroom() {
        use cuteafd_loader::placement::graph_reserve::{WorkingSet, resolve_working_set};
        let policy = resolve_working_set(None, None, None, 0, WorkingSet {
            w1: 94 << 20, step_growth: 10 << 20, capture_growth: 4 << 20,
        }).unwrap();
        let mut cache = DeviceCache::new(-1089, policy.bytes);
        cache.apply_reserve_policy(policy);
        assert_eq!(cache.soft.margin, 8 << 20);
        assert_eq!(cache.soft.limit, 190 << 20);
        cache.apply_soft_pressure(2 << 20, None);
        assert_eq!(cache.soft.limit, 190 << 20, "smaller growth must not undo the census seed");
        cache.apply_soft_pressure((4 << 20) + 1, None);
        assert_eq!(cache.soft.margin, 10 << 20, "larger growth still adjusts adaptive headroom");
        assert_eq!(cache.soft.limit, 188 << 20);
    }

    #[test]
    fn failed_donor_search_remembers_signature_and_duplicate_ticket_is_eager() {
        let (mut cache, donor) = aged_update_fixture(2, false);
        cache.captured_topology(donor, 10, 1, false);
        let id = cache.update_ticket(key(2), site(Preference::Capture), None, false).unwrap();
        assert!(cache.update_ticket(key(2), site(Preference::Capture), None, false).is_none());
        assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
        cache.register_native(id, 0, vec![99]);
        assert!(cache.finish_update(id, |_| panic!(), |_| panic!(), |_| panic!()).is_none());
        // Force ordinary fallback to retain an ARC ghost, not a new executable.
        cache.soft.measured_bytes = Some(10);
        cache.soft.measured_nodes = Some(10);
        assert!(matches!(cache.fallback_update(id), Miss::Eager));
        assert_eq!(cache.records[&cache.identities[&key(2)]].signature.as_deref(), Some(&[99][..]));
        cache.charged = cache.reserve;
        for _ in 0..10 { assert!(cache.update_ticket(key(2), site(Preference::Capture), None, false).is_none()); }
    }

    #[test]
    fn soft_cold_donor_updates_without_eviction_and_keeps_old_ghost() {
        let (mut cache, donor) = aged_update_fixture(2, false);
        let charged = cache.charged;
        let live = cache.live_bytes;
        let id = update_request(&mut cache, key(2));
        assert_eq!(cache.finish_update(id, |_| Ok(true), |_| Ok(202), |_| panic!()), Some(202));
        assert_eq!(cache.charged, charged);
        assert_eq!(cache.live_bytes, live);
        assert_eq!(cache.records[&id].bytes, 10);
        assert_eq!(cache.records[&id].last_used, Some(cache.lanes[0].index));
        assert!(!cache.owns(donor, 1));
        assert_eq!(cache.records[&donor].list, Some(List::RecentGhost));
        assert!(cache.records[&donor].last_used.is_some());
        assert_eq!(cache.counters[&Cycle::Decode].updates_soft, 1);
        assert!(cache.counters[&Cycle::Decode].update_ns > 0);
        assert!(cache.counters[&Cycle::Decode].update_ns >= cache.counters[&Cycle::Decode].update_exec_ns);
        assert_eq!(cache.update_sites["test"][&Cycle::Decode].update_ns, cache.counters[&Cycle::Decode].update_ns);
        assert_eq!(cache.pending_destroy, 0);
        assert_eq!(cache.counters[&Cycle::Decode].evictions, 0);
        let _ = cache.miss(key(1), site(Preference::Capture), None, false);
        assert!(cache.counters[&Cycle::Decode].misses >= 2);
    }

    #[test]
    fn soft_hot_donor_runs_eager_without_evicting_protected_graph() {
        let (mut cache, donor) = aged_update_fixture(1, false);
        cache.soft.measured_bytes = Some(10);
        cache.soft.measured_nodes = Some(10);
        cache.soft.limit = 10;
        cache.captured_topology(donor, 10, 1, false);
        let id = update_request(&mut cache, key(2));
        assert_eq!(cache.finish_update(id, |_| panic!(), |_| panic!(), |_| panic!()), None);
        assert!(matches!(cache.fallback_update(id), Miss::Eager));
        assert!(!cache.records[&donor].pending_destroy, "age-K hot donor must not be evicted on fallback");
        assert_eq!(cache.pending_destroy, 0);
        assert_eq!(cache.counters[&Cycle::Decode].updates_soft, 0);
        assert_eq!(cache.counters[&Cycle::Decode].no_cold_donor_soft, 1);
    }

    #[test]
    fn hard_cold_donor_updates_but_hot_donor_runs_eager() {
        for age in [1, 2] {
            let (mut cache, _) = aged_update_fixture(age, true);
            let id = update_request(&mut cache, key(2));
            let updated = cache.finish_update(id, |_| Ok(true), |_| Ok(202), |_| panic!());
            assert_eq!(updated.is_some(), age > 1);
            if updated.is_none() {
                assert!(matches!(cache.fallback_update(id), Miss::Eager));
                assert_eq!(cache.counters[&Cycle::Decode].no_cold_donor_hard, 1);
                assert_eq!(cache.counters[&Cycle::Decode].eager_hard_stop, 1);
            } else { assert_eq!(cache.counters[&Cycle::Decode].updates_hard, 1); }
            assert_eq!(cache.charged, 1000);
        }
    }

    #[test]
    fn cross_lane_donor_uses_owners_clock_and_moves_charge() {
        let (mut cache, donor) = aged_update_fixture(0, false);
        cache.assign_bank_lane(1, 1);
        cache.lane_max_cycles = [1, 2];
        let owner_start = cache.lanes[1].index;
        cache.remove_donor_candidate(donor);
        cache.records.get_mut(&donor).unwrap().last_used = Some(owner_start);
        cache.insert_donor_candidate(donor);
        for _ in 0..100 { cache.begin_lane(0); cache.complete_lane(0); }
        assert!(cache.cold_donor(&[3, 7, 11]).is_none());
        for _ in 0..2 { cache.begin_lane(1); cache.complete_lane(1); }
        assert!(cache.cold_donor(&[3, 7, 11]).is_none());
        cache.begin_lane(1); cache.complete_lane(1);
        let identity = Identity { bank: 2, ..key(2) };
        cache.assign_bank_lane(2, 0);
        let id = update_request(&mut cache, identity);
        assert_eq!(cache.finish_update(id, |_| Ok(true), |_| Ok(202), |_| panic!()), Some(202));
        assert_eq!(cache.records[&id].identity.bank, 2);
        assert_eq!(cache.records[&id].bytes, 10);
        assert_eq!(cache.native_pools().live_execs, 1);
        assert_eq!(cache.charged, 10);
        assert!(!cache.owns(donor, 1));
    }

    #[test]
    fn failed_update_invalidates_donor_and_falls_back() {
        let (mut cache, donor) = aged_update_fixture(2, false);
        let id = update_request(&mut cache, key(2));
        let invalidated = std::cell::Cell::new(false);
        assert_eq!(cache.finish_update(id, |_| Ok(false), |_| panic!(), |_| invalidated.set(true)), None);
        assert!(invalidated.get());
        assert!(!cache.records.contains_key(&donor));
        assert!(matches!(cache.fallback_update(id), Miss::Capture(_)));
        assert_eq!(cache.counters[&Cycle::Decode].update_failures, 1);
    }

    #[test]
    fn pinned_pending_and_selected_entries_never_donate_and_coldest_wins() {
        let (mut cache, donor) = aged_update_fixture(40, false);
        for flag in 0..3 {
            { let record = cache.records.get_mut(&donor).unwrap();
                record.pinned = flag == 0; record.pending_capture = flag == 1; record.pending_destroy = flag == 2; }
            assert!(cache.cold_donor(&[3, 7, 11]).is_none());
        }
        { let record = cache.records.get_mut(&donor).unwrap();
            record.pinned = false; record.pending_capture = false; record.pending_destroy = false; }
        cache.insert_donor_candidate(donor);
        assert_eq!(cache.cold_donor(&[3, 7, 11]), Some(donor));
        assert!(cache.cold_donor(&[3, 7, 12]).is_none());
        assert!(cache.cold_donor(&[]).is_none());
    }

    #[test]
    fn donor_index_selects_coldest_and_revalidates_hits_and_victims() {
        let (mut cache, older) = aged_update_fixture(40, false);
        let mut second = cache.records[&older].clone();
        cache.next_id += 1;
        let younger = cache.next_id;
        second.identity = key(2);
        second.last_used = Some(cache.lanes[0].index - 20);
        second.list = None; second.previous = None; second.next = None;
        second.accounted = Inventory::default();
        cache.identities.insert(second.identity.clone(), younger);
        cache.records.insert(younger, second);
        cache.push(younger, List::Recent);
        cache.register_native(younger, 303, vec![3, 7, 11]);
        assert_eq!(cache.cold_donor(&[3, 7, 11]), Some(older));
        cache.hit(older);
        assert_eq!(cache.cold_donor(&[3, 7, 11]), Some(younger));
        assert_eq!(cache.select_victim(None), Some(younger));
        assert!(cache.cold_donor(&[3, 7, 11]).is_none());
        assert!(cache.donor_candidates.values().all(|lanes| lanes.iter().all(|lane|
            lane.first.is_none_or(|id| cache.donor_age(id).is_none()))));
    }

    #[test]
    fn k_boundary_is_identical_at_soft_and_hard_caps() {
        for hard in [false, true] {
            for k in [1, 2] {
                for age in 0..=k + 1 {
                    let (mut cache, donor) = aged_update_fixture(age, hard);
                    cache.lane_max_cycles[0] = k;
                    assert_eq!(cache.cold_donor(&[3, 7, 11]), (age > k).then_some(donor));
                }
            }
        }
    }

    #[test]
    fn fifo_tracks_same_epoch_hits_reassignment_and_retirement() {
        let (mut cache, old) = aged_update_fixture(5, false);
        cache.soft.limit = 999;
        let young = capture(&mut cache, 2, false);
        cache.register_native(young, 303, vec![3, 7, 11]);
        cache.hit(old);
        let queue = &cache.donor_candidates[&vec![3, 7, 11]][0];
        assert_eq!((queue.first, queue.last), (Some(young), Some(old)));
        cache.hit(young);
        cache.hit(old);
        // Equal stamps may appear in either order; both are hot.
        assert!(cache.cold_donor(&[3, 7, 11]).is_none());
        cache.assign_bank_lane(1, 1);
        assert!(cache.donor_candidates[&vec![3, 7, 11]][0].first.is_none());
        let ids = [old, young];
        cache.remove_drained(&ids);
        assert!(cache.cold_donor(&[3, 7, 11]).is_none());
    }

    #[test]
    fn per_site_activity_uses_only_registered_owner_cycles() {
        let mut cache = test_cache(-1033, 1000);
        cache.assign_bank_lane(1, 1);
        let id = capture(&mut cache, 1, false);
        cache.hit(id);
        cache.pinned_hits(id, Cycle::Decode, 3);
        cache.begin_lane(0); cache.complete_lane(0);
        let snapshot = cache.snapshot();
        let counters = snapshot.update_sites["test"][&Cycle::Decode];
        assert_eq!((counters.hits, counters.misses, counters.captures), (4, 1, 1));
        let rates = &snapshot.per_cycle["test"][&Cycle::Decode];
        assert_eq!(rates.owner_cycles, cache.clocks[1].completed.load(Ordering::Acquire));
        assert_eq!(rates.hits, 4.0 / rates.owner_cycles as f64);
        assert_eq!(rates.segments_launched, 0.0, "lookups and captures are not launches");
    }

    #[test]
    #[ignore = "CPU donor-search microbenchmark; run explicitly with --nocapture"]
    fn cold_donor_search_microseconds() {
        use std::hint::black_box;
        let mut cache = test_cache(-1032, 1_000_000);
        let initial = capture(&mut cache, 0, false);
        for n in 1..22_000 {
            let mut record = cache.records[&initial].clone();
            record.identity = key(n);
            record.list = None; record.previous = None; record.next = None;
            record.accounted = Inventory::default();
            cache.next_id += 1;
            let id = cache.next_id;
            cache.identities.insert(record.identity.clone(), id);
            cache.records.insert(id, record);
            cache.push(id, List::Recent);
            cache.sync_inventory(id);
            cache.register_native(id, n as usize + 1, vec![3, n % 41]);
        }
        for _ in 0..40 { cache.begin_lane(0); cache.complete_lane(0); }
        cache.soft.limit = cache.charged;
        let iterations = 10_000;
        let start = Instant::now();
        for n in 0..iterations { black_box(cache.cold_donor(black_box(&[3, n % 41]))); }
        println!("cold_donor_selection: {:.3} us/miss; 22000 records, 41 topologies, {iterations} misses",
            start.elapsed().as_secs_f64() * 1e6 / iterations as f64);
        let start = Instant::now();
        for n in 0..iterations {
            if n % 100 == 0 { cache.begin_lane(0); cache.complete_lane(0); }
            let ticket = cache.update_ticket(key(30_000 + n), site(Preference::Capture), None, false).unwrap();
            black_box(cache.cold_donor(black_box(&[3, n % 41])));
            cache.cancel_capture(ticket);
        }
        println!("cold_donor_selection_with_cycle: {:.3} us/miss; 22000 records, 41 topologies, {iterations} misses; precheck+selection+one cycle/100 misses; no epoch rebuild",
            start.elapsed().as_secs_f64() * 1e6 / iterations as f64);
        let hot: Vec<_> = cache.native_graphs.iter().filter(|(_, native)| native.signature == [3, 1])
            .map(|(&id, _)| id).collect();
        cache.begin_lane(0); cache.complete_lane(0);
        for &id in &hot { cache.hit(id); }
        let start = Instant::now();
        assert!(cache.cold_donor(&[3, 1]).is_none());
        println!("cold_donor_after_all_hot: {:.3} us/miss; {} matching candidates hit since last miss; no repair",
            start.elapsed().as_secs_f64() * 1e6, hot.len());
    }

    #[test]
    #[ignore = "CPU lane boundary microbenchmark; run explicitly with --nocapture"]
    fn cycle_hit_only_boundary_microseconds() -> anyhow::Result<()> {
        let mut cycles = LaneCycles { lane: 0, devices: vec![-1020],
            clocks: vec![Arc::new(std::array::from_fn(|_| EpochClock::default()))] };
        let iterations = 1_000_000;
        let started = Instant::now();
        for _ in 0..iterations {
            // SAFETY: this clock-only fixture has no captured storage or native work.
            unsafe { cycles.begin(|_| unreachable!("no pending banks")) }?;
            cycles.complete();
        }
        let ns = started.elapsed().as_nanos() as f64 / f64::from(iterations);
        println!("cycle_hit_only_boundary: {ns:.1} ns/round, 1 device, {iterations} rounds");
        Ok(())
    }

    #[test]
    fn hot_stamp_without_thread_mirror_uses_locked_owner_epoch() {
        std::thread::spawn(|| {
            let mut cache = epoch_fixture();
            cache.lane_max_cycles[0] = 2;
            cache.begin_lane(0);
            let id = *cache.records.keys().next().unwrap();
            HIT_EPOCHS.with(|epochs| { epochs.borrow_mut().remove(&cache.device); });
            cache.hit(id);
            assert_eq!(cache.records[&id].last_used, Some(cache.lanes[0].index));
            assert!(cache.hot(&cache.records[&id]));
            // A stale mirror is equally unable to move the stamp backwards.
            HIT_EPOCHS.with(|epochs| { epochs.borrow_mut().insert(cache.device, [0; 2]); });
            cache.hit(id);
            assert_eq!(cache.records[&id].last_used, Some(cache.lanes[0].index));
            assert!(cache.hot(&cache.records[&id]));
            cache.complete_lane(0);
            cache.begin_lane(0);
            assert!(cache.hot(&cache.records[&id]), "hot through the K2 lag window");
        }).join().unwrap();
    }

    #[test]
    fn hot_entries_are_not_selected_inside_k1_or_k2_lag_window() {
        for k in [1, 2] {
            let mut cache = epoch_fixture();
            cache.lane_max_cycles[0] = k;
            cache.begin_lane(0);
            let ids: Vec<_> = cache.records.keys().copied().collect();
            for &id in &ids { cache.hit(id); }
            assert!(cache.select_victim(None).is_none());
            cache.complete_lane(0);
            for _ in 0..k {
                cache.begin_lane(0);
                assert!(cache.select_victim(None).is_none(), "age <= K must stay protected");
                cache.complete_lane(0);
            }
            cache.begin_lane(0);
            assert!(cache.select_victim(None).is_some(), "K+1 elapsed cycles closes the guard");
        }
    }

    #[test]
    fn hot_and_donor_share_lane_max_cycles_threshold() {
        for event_completion in [false, true] {
            for k in [1, 2] {
                let (mut cache, donor) = aged_update_fixture(k, false);
                cache.event_completion = event_completion;
                cache.lane_max_cycles[0] = k;
                assert!(cache.hot(&cache.records[&donor]));
                assert!(cache.select_victim(None).is_none());
                assert!(cache.donor_age(donor).is_none());
                assert!(cache.cold_donor(&[3, 7, 11]).is_none());
                cache.begin_lane(0);
                cache.complete_lane(0);
                assert!(!cache.hot(&cache.records[&donor]));
                assert_eq!(cache.donor_age(donor), Some(k + 1));
                assert_eq!(cache.cold_donor(&[3, 7, 11]), Some(donor));
                assert_eq!(cache.select_victim(None), Some(donor));
            }
        }
    }

    #[test]
    fn uncompleted_old_graph_is_protected_only_with_event_completion() {
        for event_completion in [false, true] {
            let (mut cache, donor) = aged_update_fixture(2, false);
            cache.event_completion = event_completion;
            let used = cache.records[&donor].used().unwrap();
            cache.clocks[0].completed.store(used - 1, Ordering::Release);
            cache.refresh_cycles();
            assert_eq!(cache.hot(&cache.records[&donor]), event_completion);
            assert_eq!(cache.donor_age(donor), (!event_completion).then_some(2));
            assert_eq!(cache.cold_donor(&[3, 7, 11]), (!event_completion).then_some(donor));
            if event_completion {
                assert!(cache.select_victim(None).is_none());
                cache.clocks[0].completed.store(used, Ordering::Release);
                cache.refresh_cycles();
                assert!(!cache.hot(&cache.records[&donor]));
                assert_eq!(cache.donor_age(donor), Some(2));
                assert_eq!(cache.cold_donor(&[3, 7, 11]), Some(donor));
            }
            assert_eq!(cache.select_victim(None), Some(donor));
        }
    }

    #[test]
    fn two_lane_hot_and_donor_use_each_lanes_own_max_cycles() {
        let mut cache = test_cache(-1035, 1000);
        cache.lane_max_cycles = [1, 2];
        cache.assign_bank_lane(1, 0);
        cache.assign_bank_lane(2, 1);
        let id0 = capture(&mut cache, 1, false);
        let key_lane1 = Identity { bank: 2, site: "test", key: 2, shape: "2".into() };
        let Miss::Capture(id1) = cache.miss(key_lane1, site(Preference::Capture), None, false) else { panic!() };
        cache.measured_capture(10);
        cache.soft.limit = 20;
        cache.charged = 20;
        cache.event_completion = true;
        for (id, lane, native, signature) in [(id0, 0, 101, vec![3, 7, 11]), (id1, 1, 102, vec![3, 7, 12])] {
            cache.records.get_mut(&id).unwrap().last_used = Some(cache.lanes[lane].index);
            cache.register_native(id, native, signature);
            for _ in 0..2 { cache.begin_lane(lane); cache.complete_lane(lane); }
        }
        assert!(!cache.hot(&cache.records[&id0]) && cache.donor_age(id0) == Some(2));
        assert!(cache.hot(&cache.records[&id1]) && cache.donor_age(id1).is_none());
        assert_eq!(cache.cold_donor(&[3, 7, 11]), Some(id0));
        assert!(cache.cold_donor(&[3, 7, 12]).is_none());
        assert_eq!(cache.select_victim(Some(id0)), None);
        assert_eq!(cache.select_victim(None), Some(id0));
        cache.begin_lane(1); cache.complete_lane(1);
        assert!(!cache.hot(&cache.records[&id1]) && cache.donor_age(id1) == Some(3));
        assert_eq!(cache.cold_donor(&[3, 7, 12]), Some(id1));
        assert_eq!(cache.select_victim(None), Some(id1));
    }

    #[test]
    fn soft_room_full_without_cold_candidates_runs_eager() {
        for (age, event_completion) in [(1, false), (1, true), (2, true)] {
            let (mut cache, donor) = aged_update_fixture(age, false);
            cache.event_completion = event_completion;
            if event_completion && age == 2 {
                let used = cache.records[&donor].used().unwrap();
                cache.clocks[0].completed.store(used - 1, Ordering::Release);
                cache.refresh_cycles();
            }
            cache.captured_topology(donor, 10, 1, false);
            cache.soft.measured_bytes = Some(10);
            cache.soft.measured_nodes = Some(10);
            assert_eq!(cache.make_soft_room(10, None), SoftRoom::Full);
            assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
            assert_eq!(cache.pending_destroy, 0);
            assert!(!cache.records[&donor].pending_destroy);
        }
    }

    #[test]
    fn hot_per_layer_workload_remains_cached_without_evictions() {
        let mut cache = epoch_fixture();
        cache.lane_max_cycles[0] = 2;
        let ids: Vec<_> = cache.records.keys().copied().collect();
        for _ in 0..20 {
            cache.begin_lane(0);
            for &id in &ids { cache.hit(id); }
            cache.apply_soft_pressure(0, None);
            assert_eq!(cache.pending_destroy, 0);
            cache.complete_lane(0);
        }
        assert_eq!(cache.counters[&Cycle::Decode].evictions, 0);
        assert_eq!(cache.native_pools().live_execs, 4);
    }

    #[test]
    fn hot_over_cap_workload_runs_vampires_instead_of_evicting_layers() {
        let mut cache = epoch_fixture();
        cache.lane_max_cycles[0] = 2;
        cache.begin_lane(0);
        let ids: Vec<_> = cache.records.keys().copied().collect();
        for &id in &ids { cache.hit(id); }
        for _ in 0..4 {
            assert!(matches!(cache.miss(key(99), site(Preference::Capture), None, false), Miss::Eager));
        }
        assert_eq!(cache.pending_destroy, 0);
        assert!(cache.lanes[0].vampires.contains(&key(99)));
        assert_eq!(cache.counters[&Cycle::Decode].eager_vampire, 3);
        assert_eq!(cache.counters[&Cycle::Decode].evictions, 0);
    }

    #[test]
    fn cycle_vampire_selects_once_and_unrelated_fitting_miss_can_capture() {
        let mut cache = epoch_fixture();
        cache.begin_lane(0);
        assert!(matches!(cache.miss(key(4), site(Preference::Capture), None, false), Miss::Eager));
        let selected = cache.pending_destroy;
        assert_eq!(selected, 1);
        for _ in 0..10 {
            assert!(matches!(cache.miss(key(4), site(Preference::Capture), None, false), Miss::Eager));
        }
        assert_eq!(cache.pending_destroy, selected, "a vampire cannot evict twice in its cycle");
        assert_eq!(cache.counters[&Cycle::Decode].eager_vampire, 10);
        // Unrelated misses are not blocked merely by pending owners.
        cache.soft.limit = 200;
        assert!(matches!(cache.miss(key(5), site(Preference::Capture), None, false), Miss::Capture(_)));
        cache.complete_lane(0);
        cache.begin_lane(0);
        assert!(!cache.lanes[0].vampires.contains(&key(4)));
    }

    #[test]
    fn cycle_k1_and_k2_retirement_credit_waits_for_inflight_frontier() {
        for k in [1, 2] {
            let mut cache = epoch_fixture();
            cache.begin_lane(0);
            if k == 2 { cache.begin_lane(1); }
            let id = cache.mark_victim(false).unwrap();
            assert!(!cache.persistent_victim_ready(id), "own in-flight launch must survive");
            let pool_nodes = cache.pools.live_nodes;
            cache.complete_lane(0);
            cache.begin_lane(0);
            assert!(cache.persistent_victim_ready(id));
            cache.destroyed(&[id], ReleaseMeasurement { released: 0 });
            if k == 2 {
                assert_eq!(cache.pools.live_nodes, pool_nodes, "peer's in-flight cycle retains credit");
                assert_eq!(cache.epochs.uncredited_nodes, 10);
                assert_ne!(cache.make_soft_room(10, None), SoftRoom::Available, "uncredited nodes do not fit incoming work");
                cache.complete_lane(1);
            }
            assert_eq!(cache.pools.live_nodes, pool_nodes - 10);
            assert_eq!(cache.epochs.uncredited_nodes, 0);
            assert_eq!(cache.charged, 100, "physical high-water never falls on destruction");
        }
    }

    #[test]
    fn cycle_abandon_is_not_completion_and_cannot_release_peer_credit() -> anyhow::Result<()> {
        let mut cache = epoch_fixture();
        let mut cycles = LaneCycles { lane: 0, devices: vec![-1020], clocks: vec![cache.clocks.clone()] };
        // SAFETY: no native work in this policy fixture.
        unsafe { cycles.begin(|_| Ok(())) }?;
        cycles.abandon();
        let victim = cache.mark_victim(false).unwrap();
        assert!(!cache.persistent_victim_ready(victim));
        // An explicit completed serial boundary provides the missing proof.
        cache.serial_boundary();
        assert!(cache.persistent_victim_ready(victim));
        cache.destroyed(&[victim], ReleaseMeasurement { released: 0 });
        assert!(cache.lagged.is_empty());
        Ok(())
    }

    #[test]
    fn cycle_peer_selected_after_abandon_waits_until_serial_drain() -> anyhow::Result<()> {
        let mut cache = epoch_fixture();
        cache.assign_bank_lane(1, 1);
        let mut abandoned = LaneCycles { lane: 0, devices: vec![-1020], clocks: vec![cache.clocks.clone()] };
        // SAFETY: synthetic policy clocks contain no native work.
        unsafe { abandoned.begin(|_| Ok(())) }?;
        abandoned.abandon();
        cache.begin_lane(1);
        let victim = cache.mark_victim(false).unwrap();
        cache.complete_lane(1);
        cache.begin_lane(1);
        cache.destroyed(&[victim], ReleaseMeasurement { released: 0 });
        assert_eq!(cache.epochs.uncredited_nodes, 10, "abandoned A frontier still blocks B's credit");
        cache.complete_lane(1);
        // Repeated failures cannot strand capacity beyond the next drained boundary.
        for _ in 0..3 {
            unsafe { abandoned.begin(|_| Ok(())) }?;
            abandoned.abandon();
        }
        cache.serial_boundary();
        assert_eq!(cache.epochs.uncredited_nodes, 0);
        assert!(cache.lagged.is_empty());
        Ok(())
    }

    #[test]
    fn cycle_fast_owner_keeps_one_lagged_chunk_behind_slow_peer() {
        let mut cache = epoch_fixture();
        cache.soft.limit = 1000;
        cache.begin_lane(0);
        cache.begin_lane(1);
        for _ in 0..4 {
            let id = cache.mark_victim(false).unwrap();
            cache.complete_lane(0);
            cache.begin_lane(0);
            cache.destroyed(&[id], ReleaseMeasurement { released: 0 });
            assert_eq!(cache.lagged.len(), 1, "all fast-lane retirements wait on the same peer cycle");
        }
        assert_eq!(cache.epochs.uncredited_nodes, 40);
        cache.complete_lane(1);
        assert!(cache.lagged.is_empty());
        assert_eq!(cache.pools.live_nodes, 0);
    }

    #[test]
    fn cycle_credited_reuse_preserves_physical_high_water_and_reports_driver_growth() {
        let mut cache = epoch_fixture();
        cache.begin_lane(0);
        let id = cache.mark_victim(false).unwrap();
        cache.complete_lane(0);
        cache.begin_lane(0);
        cache.destroyed(&[id], ReleaseMeasurement { released: 0 });
        // Bootstrap is the actual 10-node graph in this synthetic byte model.
        cache.classes.clear();
        let Miss::Capture(reused) = cache.miss(key(4), site(Preference::Capture), None, false) else { panic!() };
        cache.captured_topology(reused, 10, 1, false);
        cache.measured_capture(0);
        assert_eq!(cache.charged, 100);
        assert_eq!(cache.epochs.reuse_growth_bytes, 0);
        cache.soft.limit = 200;
        cache.pools.node_high_water += 10;
        cache.pools.exec_high_water += 1;
        let Miss::Capture(grew) = cache.miss(key(5), site(Preference::Capture), None, false) else { panic!() };
        cache.captured_topology(grew, 10, 1, false);
        cache.measured_capture(3);
        assert_eq!(cache.charged, 103);
        assert_eq!(cache.epochs.reuse_growth_bytes, 3);
    }

    #[test]
    fn cycle_uncredited_chunks_cannot_finance_hard_cap_growth() {
        let mut cache = epoch_fixture();
        cache.reserve = 105;
        cache.begin_lane(0);
        cache.begin_lane(1);
        let id = cache.mark_victim(false).unwrap();
        cache.complete_lane(0);
        cache.begin_lane(0);
        cache.destroyed(&[id], ReleaseMeasurement { released: 0 });
        cache.soft.limit = 1000;
        cache.classes.clear();
        let too_large = Site { name: "new-large-site", bootstrap: 1000, ..site(Preference::Capture) };
        assert!(matches!(cache.miss(key(4), too_large, None, false), Miss::Eager));
        assert_eq!(cache.charged, 100);
        assert!(!cache.unmeasured);
    }

    #[test]
    fn completed_capture_measures_once_and_failure_cancels_without_fake_charge() -> anyhow::Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory_usage(&library, 6, 64 << 20, 0)?;
        let mut cache = test_cache(6, 32 << 20);
        cache.observer.observe(64 << 20, 64 << 20, 0, 0, 0);
        cache.memory_query = Some(library.physical_memory_query()?);
        let Miss::Capture(a) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
        cache.captured_topology(a, 7, 1, false);
        fixture.configure_memory_usage(&library, 6, 64 << 20, 2 << 20)?;
        assert!(cache.completed_capture(a));
        assert_eq!(cache.charged, 2 << 20);
        assert_eq!(cache.level, Some(2 << 20));
        assert!(!cache.needs_boundary());
        let (growth, level) = cache.observer.observe(62 << 20, 64 << 20, 0, 0, 0);
        cache.level = Some(level);
        cache.measured_capture(growth);
        assert_eq!(cache.charged, 2 << 20, "GLM-shaped whole-step observer must not charge commit twice");
        let Miss::Capture(b) = cache.miss(key(2), site(Preference::Capture), None, false) else { panic!() };
        cache.captured_topology(b, 11, 1, false);
        fixture.configure_memory(&library, 6, 0)?;
        assert!(!cache.completed_capture(b));
        assert_eq!(cache.charged, 4 << 20, "failed sample conservatively charges the retained exec estimate");
        assert_eq!(cache.failed_measurement_credit, 2 << 20);
        assert!(!cache.records.contains_key(&b));
        assert!(cache.needs_boundary(), "fallback credit requests a later physical reconciliation");
        assert!(cache.measurement_failed);
        fixture.configure_memory_usage(&library, 6, 64 << 20, 4 << 20)?;
        let shared = Arc::new(Mutex::new(cache));
        DEVICES.get_or_init(Mutex::default).lock().unwrap().insert(6, Arc::downgrade(&shared));
        observe_device_sampled(6, || Ok((60 << 20, 64 << 20, 0, 0, 0))).unwrap();
        let mut cache = shared.lock().unwrap();
        assert_eq!(cache.charged, 4 << 20, "successful later sample cannot double-charge fallback credit");
        assert_eq!(cache.failed_measurement_credit, 0);
        assert!(matches!(cache.miss(key(3), site(Preference::Capture), None, false), Miss::Eager));
        Ok(())
    }

    #[test]
    fn soft_failure_remembers_requested_key_and_retries_after_one_retirement() {
        let mut cache = test_cache(-1014, 1000);
        for n in 0..4 {
            let id = capture(&mut cache, n, false);
            cache.captured_topology(id, 10, 1, false);
        }
        cache.soft.measured_bytes = Some(100);
        cache.soft.measured_nodes = Some(40);
        cache.soft.limit = 100;
        cache.charged = 100;
        assert!(matches!(cache.miss(key(4), site(Preference::Capture), None, false), Miss::Eager));
        let ghost = cache.identities[&key(4)];
        assert_eq!(cache.records[&ghost].stats.misses, 1);
        assert_eq!(cache.records[&ghost].list, Some(List::RecentGhost));
        assert_eq!(cache.pending_destroy, 1);
        let victims: Vec<_> = cache.records.iter().filter(|(_, r)| r.pending_destroy).map(|(&id, _)| id).collect();
        cache.destroyed(&victims, ReleaseMeasurement { released: 0 });
        assert!(matches!(cache.miss(key(4), site(Preference::Capture), None, false), Miss::Capture(id) if id == ghost));
    }

    #[test]
    fn near_pinned_soft_budget_waits_for_a_typical_unpinned_shape() {
        let mut cache = test_cache(-1015, 1000);
        let pinned = capture(&mut cache, 1, true);
        cache.captured_topology(pinned, 90, 1, false);
        let free = capture(&mut cache, 2, false);
        cache.captured_topology(free, 10, 1, false);
        cache.soft.margin = 990;
        cache.soft.limit = 10;
        cache.charged = 11;
        cache.apply_soft_pressure(0, Some(free));
        assert!(cache.soft.measured_bytes.is_none(), "less than one unpinned graph must not freeze the ratio");
        assert_eq!(cache.pending_destroy, 0);
        cache.hit(free);
        assert_eq!(cache.native_pools().live_nodes, 100);
        cache.mark_victim(false);
        cache.destroyed(&[free], ReleaseMeasurement { released: 0 });
        cache.hit(free);
        assert_eq!(cache.native_pools().live_nodes, 90, "ghost hit must not resurrect inventory");
        cache.destroyed(&[pinned], ReleaseMeasurement { released: 0 });
        assert_eq!(cache.native_pools().live_execs, 0, "destroy clears a pinned owner's residency");
    }

    #[test]
    fn noncapturable_misses_never_select_soft_victims() {
        let mut cache = test_cache(-1011, 1000);
        for n in 0..4 {
            let id = capture(&mut cache, n, false);
            cache.captured_topology(id, 10, 1, false);
        }
        cache.soft.measured_bytes = Some(100);
        cache.soft.measured_nodes = Some(40);
        cache.soft.limit = 100;
        cache.charged = 100;
        assert!(matches!(cache.miss(key(10), site(Preference::Eager), None, false), Miss::Eager));
        assert_eq!(cache.pending_destroy, 0);
        cache.physical_floor = 100;
        cache.sample_physical(1, 0);
        assert!(matches!(cache.miss(key(11), site(Preference::Capture), None, false), Miss::Eager));
        assert_eq!(cache.pending_destroy, 0);
        cache.physical_floor = 0;
        seal_cross_device_capture_on_this_thread(true);
        let peer = Site { spans_devices: true, ..site(Preference::Capture) };
        assert!(matches!(cache.miss(key(12), peer, None, false), Miss::Eager));
        seal_cross_device_capture_on_this_thread(false);
        assert_eq!(cache.pending_destroy, 0);
    }

    #[test]
    fn pinned_equality_defers_soft_freeze_and_failed_boundary_is_not_serving_error() {
        let mut cache = test_cache(-1012, 1000);
        let id = capture(&mut cache, 1, true);
        cache.captured_topology(id, 10, 1, false);
        cache.soft.limit = 100;
        cache.soft.margin = 900;
        cache.charged = 100;
        cache.apply_soft_pressure(0, Some(id));
        assert!(cache.soft.measured_bytes.is_none());
        let device = -1013;
        let shared = device_cache_sampled(device, 1000, Basis::Bytes, || Ok((1000, 1000, 0, 0, 0))).unwrap();
        {
            let mut cache = shared.lock().unwrap();
            let Miss::Capture(id) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
            cache.cancel_capture(id);
        }
        assert!(observe_device_sampled(device, || anyhow::bail!("fixture physical query failure")).is_ok());
        assert!(shared.lock().unwrap().measurement_failed);
        assert!(!shared.lock().unwrap().needs_boundary());
    }

    #[test]
    fn zero_soft_disables_capture_and_unknown_nodes_do_not_disable_known_pressure() {
        let mut empty = DeviceCache::new(-1008, GRAPH_CHUNK);
        assert!(matches!(empty.miss(key(1), site(Preference::Capture), None, false), Miss::Capture(_)));
        assert_eq!(empty.soft.limit, GRAPH_CHUNK / 2);
        let mut cache = test_cache(-1009, 16 * GRAPH_CHUNK);
        let capture_site = Site { bootstrap: 1, ..site(Preference::Capture) };
        let Miss::Capture(unknown) = cache.miss(key(1), capture_site, None, false) else { panic!() };
        cache.captured_topology(unknown, 0, 0, false);
        cache.measured_capture(GRAPH_CHUNK);
        for n in 2..15 {
            let Miss::Capture(id) = cache.miss(key(n), capture_site, None, false) else { panic!() };
            cache.captured_topology(id, 10, 1, false);
            cache.measured_capture(GRAPH_CHUNK);
            cache.serial_boundary();
            cache.serial_boundary();
        }
        assert!(cache.soft.measured_bytes.is_some());
        assert!(matches!(cache.miss(key(15), capture_site, None, false), Miss::Eager));
        assert!(cache.pending(unknown));
        let victims: Vec<_> = cache.records.iter().filter(|(_, r)| r.pending_destroy).map(|(&id, _)| id).collect();
        assert_eq!(victims.len(), 2, "unknown owner plus one known owner release a graph's capacity");
        cache.destroyed(&victims, ReleaseMeasurement { released: 0 });
        assert_eq!(cache.records[&unknown].list, Some(List::RecentGhost));
    }

    #[test]
    fn single_owner_measurement_preserves_another_outstanding_ticket() {
        let mut cache = test_cache(-1006, 64 * GRAPH_CHUNK);
        let Miss::Capture(a) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
        let Miss::Capture(b) = cache.miss(key(2), site(Preference::Capture), None, false) else { panic!() };
        cache.captured_topology(a, 7, 1, false);
        cache.settle_captures(GRAPH_CHUNK, Some(a));
        assert!(!cache.records[&a].pending_capture);
        assert!(cache.records[&b].pending_capture);
        assert!(cache.unmeasured);
        assert_eq!(cache.pending_bytes, 10);
        assert_eq!(cache.pools.exec_high_water, 1);
        cache.captured_topology(b, 11, 1, false);
        cache.settle_captures(0, Some(b));
        assert!(!cache.unmeasured);
        assert_eq!(cache.pending_bytes, 0);
        assert_eq!(cache.charged, GRAPH_CHUNK);
        assert_eq!(cache.native_pools().live_nodes, 18);
        assert_eq!(cache.counters[&Cycle::Decode].captures, 2);
    }

    #[test]
    fn doubled_measured_growth_margin_rounds_once_and_preserves_small_reserve_capture() {
        let mut cache = test_cache(-1018, 32 * GRAPH_CHUNK);
        let id = capture(&mut cache, 1, false);
        cache.captured_topology(id, 7, 1, false);
        cache.apply_soft_pressure(GRAPH_CHUNK + 1, Some(id));
        assert_eq!(cache.soft.margin, 3 * GRAPH_CHUNK, "round 2x raw growth, not twice rounded growth");
        assert_eq!(cache.soft.limit, 29 * GRAPH_CHUNK);
        assert!(cache.soft.measured_bytes.is_none());

        let mut small = DeviceCache::new(-1019, 4 * GRAPH_CHUNK);
        let capture_site = Site { bootstrap: 1, ..site(Preference::Capture) };
        let Miss::Capture(a) = small.miss(key(1), capture_site, None, false) else { panic!() };
        small.captured_topology(a, 10, 1, false);
        small.measured_capture(3 * GRAPH_CHUNK / 2);
        assert_eq!(small.soft.margin, 2 * GRAPH_CHUNK);
        assert_eq!(small.soft.limit, 2 * GRAPH_CHUNK);
        assert!(small.margin_clamp_warned);
        assert!(!small.snapshot().hard_stopped);
        let Miss::Capture(b) = small.miss(key(2), capture_site, None, false) else { panic!("large slab must not lock out capture below hard") };
        small.captured_topology(b, 10, 1, false);
        small.measured_capture(0);
        assert_eq!(small.charged, 3 * GRAPH_CHUNK / 2);
        assert_eq!(small.native_pools().live_execs, 2);
        assert!(!small.snapshot().hard_stopped);
        let snapshot = serde_json::to_value(small.snapshot()).unwrap();
        assert_eq!(snapshot["soft"]["margin"], 2 * GRAPH_CHUNK);
        assert_eq!(snapshot["soft"]["limit"], 2 * GRAPH_CHUNK);
    }

    #[test]
    fn soft_ratio_and_margin_freeze_from_measured_crossing_and_ignore_reuse_shares() {
        let mut cache = test_cache(-1004, 22 * GRAPH_CHUNK);
        let capture_site = Site { bootstrap: 1, ..site(Preference::Capture) };
        let mut owners = Vec::new();
        for n in 0..5 {
            let Miss::Capture(id) = cache.miss(key(n), capture_site, None, false) else { panic!() };
            cache.captured_topology(id, 10, 1, false);
            cache.measured_capture(3 * GRAPH_CHUNK);
            cache.serial_boundary();
            cache.serial_boundary();
            owners.push(id);
        }
        assert!(cache.soft.measured_bytes.is_none());
        let Miss::Capture(id) = cache.miss(key(5), capture_site, None, false) else { panic!() };
        cache.captured_topology(id, 10, 1, false);
        cache.measured_capture(3 * GRAPH_CHUNK);
        assert_eq!(cache.soft.margin, 6 * GRAPH_CHUNK);
        assert_eq!(cache.soft.limit, 16 * GRAPH_CHUNK);
        assert_eq!(cache.soft.measured_bytes, Some(18 * GRAPH_CHUNK));
        assert_eq!(cache.soft.measured_nodes, Some(60));
        assert_eq!(cache.pending_destroy, 1);
        let victims: Vec<_> = cache.records.iter().filter(|(_, r)| r.pending_destroy).map(|(&id, _)| id).collect();
        cache.destroyed(&victims, ReleaseMeasurement { released: 0 });
        let frozen = (cache.soft.margin, cache.soft.measured_bytes, cache.soft.measured_nodes);
        for _ in 0..8 {
            for &owner in owners.iter().chain(std::iter::once(&id)) {
                if !victims.contains(&owner) { cache.hit(owner); }
            }
            cache.unmeasured = true;
            cache.measured_capture(0);
            assert_eq!(cache.pending_destroy, 0, "fitting hot set must settle above soft");
            assert_eq!((cache.soft.margin, cache.soft.measured_bytes, cache.soft.measured_nodes), frozen);
        }
        cache.serial_boundary();
        cache.serial_boundary();
        assert!(matches!(cache.miss(key(6), capture_site, None, false), Miss::Eager));
        let victims: Vec<_> = cache.records.iter().filter(|(_, r)| r.pending_destroy).map(|(&id, _)| id).collect();
        assert_eq!(victims.len(), 1, "incoming node reservation makes room before capture");
        cache.destroyed(&victims, ReleaseMeasurement { released: 0 });
        let Miss::Capture(reuse) = cache.miss(key(6), capture_site, None, false) else { panic!() };
        cache.captured_topology(reuse, 10, 1, false);
        cache.measured_capture(0);
        assert_eq!(cache.records[&reuse].bytes, 0);
        assert_eq!(cache.pending_destroy, 0, "newly captured graph must not evict itself");
        assert_eq!((cache.soft.margin, cache.soft.measured_bytes, cache.soft.measured_nodes), frozen);
    }

    #[test]
    fn soft_rotating_shapes_reuse_a_fixed_arena_until_terminal_hard_stop() {
        let unit = GRAPH_CHUNK;
        let mut cache = test_cache(-1005, 16 * unit);
        let capture_site = Site { bootstrap: 1, ..site(Preference::Capture) };
        let mut allocator_nodes = 0u64;
        for n in 0..200 {
            let mut ticket = cache.miss(key(n), capture_site, None, false);
            if matches!(ticket, Miss::Eager) {
                let victims: Vec<_> = cache.records.iter().filter(|(_, r)| r.pending_destroy).map(|(&id, _)| id).collect();
                assert_eq!(victims.len(), 1, "one same-size owner must make room for key {n}");
                cache.destroyed(&victims, ReleaseMeasurement { released: 0 });
                ticket = cache.miss(key(n), capture_site, None, false);
            }
            let Miss::Capture(id) = ticket else { panic!("key {n} stays eager") };
            cache.captured_topology(id, 10, 1, false);
            let nodes = cache.native_pools().live_nodes;
            let growth = if n == 0 { 3 * unit } else { nodes.saturating_sub(allocator_nodes) / 10 * unit };
            allocator_nodes = allocator_nodes.max(nodes);
            cache.measured_capture(growth);
            let victims: Vec<_> = cache.records.iter().filter(|(_, r)| r.pending_destroy).map(|(&id, _)| id).collect();
            cache.destroyed(&victims, ReleaseMeasurement { released: 0 });
            assert!(cache.charged < cache.reserve);
            if cache.soft.measured_bytes.is_some() { assert!(cache.soft.fits(cache.native_pools().live_nodes)); }
        }
        assert_eq!(cache.counters[&Cycle::Decode].captures, 200);
        assert_eq!(cache.charged, 10 * unit, "double-growth margin and equality reservation makes room before another slab is needed");
        cache.charged = cache.reserve;
        assert!(matches!(cache.miss(key(200), capture_site, None, false), Miss::Eager));
    }

    #[test]
    fn high_water_overshoot_does_not_churn_a_bounded_hot_set() {
        let mut cache = test_cache(-996, 1000);
        let mut hot = Vec::new();
        for n in 0..36 { hot.push(capture(&mut cache, n, false)); }
        // Retired/transient growth left the permanent arena over reserve.
        cache.charged = 1010;
        for round in 0..8 {
            // A zero-growth completion still occurred in another bank this round.
            cache.unmeasured = true;
            for n in 0..36 {
                let id = hot[n as usize];
                if matches!(cache.records[&id].list, Some(List::Recent | List::Frequent))
                    && !cache.pending(id) { cache.hit(id); }
                else { let _ = cache.miss(key(n), site(Preference::Capture), None, false); }
            }
            cache.measured_capture(0);
            let victims: Vec<_> = cache.records.iter().filter(|(_, r)| r.pending_destroy)
                .map(|(&id, _)| id).collect();
            cache.destroyed(&victims, ReleaseMeasurement { released: 0 });
            assert_eq!(cache.counters[&Cycle::Decode].evictions, 0, "hot round {round}");
            assert_eq!(cache.charged, 1010, "destroy must not credit physical release");
        }
        assert!(matches!(cache.miss(key(36), site(Preference::Capture), None, false), Miss::Eager));
        assert_eq!(cache.pending_destroy, 0, "hard stop never churns existing owners");

    }

    #[test]
    fn measured_hard_stop_freezes_new_tickets_including_reuse_and_survives_destroy() {
        for measured in [100, 110] {
            let mut cache = test_cache(-1002, 100);
            let Miss::Capture(id) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
            cache.captured_topology(id, 7, 1, false);
            cache.measured_capture(measured);
            for _ in 0..16 {
                cache.hit(id);
                assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
                assert_eq!(cache.pending_destroy, 0);
                assert_eq!(cache.charged, measured);
            }
            cache.mark_victim(false);
            cache.destroyed(&[id], ReleaseMeasurement { released: measured });
            assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
            assert_eq!(cache.charged, measured);
            assert!(cache.snapshot().hard_stopped);
        }
    }

    #[test]
    fn native_node_counts_are_retained_independently_of_byte_estimator_units() {
        let mut cache = test_cache(-1003, 64 << 20);
        let Miss::Capture(a) = cache.miss(key(1), site(Preference::Capture), None, true) else { panic!() };
        cache.captured_topology(a, 7, 1, false);
        cache.measured_capture(10);
        let Miss::Capture(b) = cache.miss(key(2), site(Preference::Capture), None, false) else { panic!() };
        cache.captured_topology(b, 13, 1, false);
        cache.measured_capture(0);
        assert_eq!((cache.native_pools().live_nodes, cache.native_pools().live_execs), (20, 2));
        assert_eq!(cache.native_pools().unknown_execs, 0);
        assert_eq!(cache.pools.live_nodes, 20, "byte-based prediction remains separate");
        cache.mark_victim(false);
        assert_eq!(cache.native_pools().live_nodes, 20, "selected victims still own native storage");
        cache.destroyed(&[b], ReleaseMeasurement { released: 0 });
        assert_eq!((cache.native_pools().live_nodes, cache.native_pools().live_execs), (7, 1));
        cache.remove_bank(1);
        assert_eq!(cache.native_pools().live_execs, 0);
    }

    #[test]
    fn thirty_six_percent_measured_live_set_does_not_evict() {
        let mut cache = test_cache(-997, 1000);
        for n in 0..35 { capture(&mut cache, n, false); }
        assert!(matches!(cache.miss(key(35), site(Preference::Capture), None, false), Miss::Capture(_)));
        cache.charged = 1000;
        cache.measured_capture(10);
        assert_eq!(cache.pending_destroy, 0, "36% live is not over reserve");
        assert_eq!(cache.live_bytes, 360);
    }

    #[test]
    fn measured_hard_stop_bounds_rotating_unknown_node_shapes() {
        let mut cache = test_cache(-999, 100);
        for n in 0..10 {
            assert!(matches!(cache.miss(key(n), site(Preference::Capture), None, false), Miss::Capture(_)));
            cache.measured_capture(10);
        }
        for n in 10..200 {
            assert!(matches!(cache.miss(key(n), site(Preference::Capture), None, false), Miss::Eager));
            assert_eq!(cache.charged, 100);
        }
        assert_eq!(cache.counters[&Cycle::Decode].captures, 10);
    }

    #[test]
    fn measured_overshoot_keeps_a_fitting_hot_set_without_boundary_eviction() {
        let mut cache = test_cache(-1000, 100);
        let mut ids = Vec::new();
        for n in 0..3 {
            let Miss::Capture(id) = cache.miss(key(n), site(Preference::Capture), None, false) else { panic!() };
            ids.push(id);
        }
        cache.measured_capture(110);
        for _ in 0..16 {
            for &id in &ids { cache.hit(id); }
            cache.unmeasured = true;
            cache.measured_capture(0);
            assert_eq!(cache.pending_destroy, 0);
            assert_eq!(cache.counters[&Cycle::Decode].evictions, 0);
            assert_eq!(cache.charged, 110);
            assert_eq!(cache.pools.live_execs, 3);
        }
        let snapshot = serde_json::to_value(cache.snapshot()).unwrap();
        let trace: serde_json::Value = serde_json::from_slice(&cache.trace_snapshot().bytes().unwrap()).unwrap();
        assert_eq!(snapshot["live_bytes"], 110);
        assert_eq!(trace["cache"]["live_bytes"], snapshot["live_bytes"]);
        cache.remove_bank(1);
        assert_eq!(cache.live_bytes, 0);
        assert_eq!(cache.charged, 110);
    }

    #[test]
    fn below_soft_estimates_do_not_evict_and_pinned_soft_excess_terminates() {
        let mut cache = test_cache(-1001, 32 * GRAPH_CHUNK);
        let capture_site = Site { bootstrap: 1, ..site(Preference::Capture) };
        let Miss::Capture(id) = cache.miss(key(1), capture_site, None, true) else { panic!() };
        cache.captured_topology(id, 10, 1, false);
        cache.measured_capture(9 * GRAPH_CHUNK);
        assert_eq!(cache.pending_destroy, 0, "pinned excess cannot cause an eviction loop");
        assert!(cache.soft.measured_bytes.is_none(), "pinned-only ratio must wait for evictable owners");
        assert!(matches!(cache.miss(key(2), capture_site, None, false), Miss::Capture(_)));
    }

    #[test]
    fn estimate_only_records_do_not_count_as_measured_live_bytes() {
        let mut cache = test_cache(-998, 1000);
        let Miss::Capture(id) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
        cache.measured_capture(0);
        assert_eq!(cache.records[&id].bytes, 0);
        assert_eq!(cache.live_bytes, 0);
        assert_eq!(cache.records[&id].estimate, 10);
        assert_eq!(cache.queues[List::Recent.index()].bytes, 10);
        assert_eq!(cache.charged, 0);
        cache.remove_drained(&[id]);
        assert_eq!(cache.live_bytes, 0);
    }

    #[test]
    fn rejected_small_graph_trace_has_no_executable_or_permanent_bytes() {
        for basis in [Basis::Bytes, Basis::Nodes { node_rate: 2560, graph_rate: 65536 }] {
            for nodes in 0..=2 {
                let mut cache = test_cache(-992, 64 << 20);
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
        let mut cache = test_cache(-992, 1000);
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
            let mut cache = test_cache(-993, 1000);
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
        let mut cache = test_cache(-994, 16 << 20);
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
            let mut cache = test_cache(-995, reserve);
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
        let mut cache = test_cache(-991, 100_000);
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
        cache.refresh_rankings();
        assert_eq!(cache.snapshot().top_hits.len(), 16);
    }

    #[test]
    fn sigterm_http_shutdown_writes_complete_nonempty_final_traces() {
        const CHILD: &str = "CUTEAFD_TRACE_SIGTERM_TEST_CHILD";
        if let Some(directory) = std::env::var_os(CHILD) {
            let directory = std::path::PathBuf::from(directory);
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async {
                let mut cache = test_cache(-710, 100);
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
        let mut cache = test_cache(-700, 100);
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
        let mut cache = test_cache(-701, 100);
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
        let mut cache = test_cache(-703, 100);
        cache.trace = Some(TraceWriter::new(path.clone()).unwrap());
        let id = capture(&mut cache, 1, false);
        cache.hit(id);
        cache.remove_bank(1);
        let owner = Arc::new(Mutex::new(cache));
        let weak = Arc::downgrade(&owner);
        {
            let mut registry = DEVICES.get_or_init(Mutex::default).lock().unwrap();
            registry.insert(-703, weak.clone());
            TEST_CONTROLLERS.with(|owned| owned.borrow_mut().push(weak.clone()));
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
        let mut cache = test_cache(-702, 100);
        let a = capture(&mut cache, 1, false);
        cache.records.get_mut(&a).unwrap().stats.first_seen_ms = 0;
        cache.clock_ms = 20_000;
        cache.hit(a);
        let mut snapshot = cache.trace_snapshot();
        snapshot.cache.sampled_at_ms = 30_000;
        snapshot.cache.telemetry_bytes = 0;
        snapshot.cache.rankings_age_ms = 0;
        let bytes = snapshot.bytes().unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), include_str!("trace-schema.json").trim());
    }

    #[test]
    #[ignore = "CPU trace boundary microbenchmark; run explicitly with --nocapture"]
    fn trace_snapshot_microseconds() {
        use std::hint::black_box;
        let mut cache = test_cache(-703, 100_000);
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
        let mut cache = test_cache(-901, policy.bytes);
        cache.reserve_policy = Some(policy);
        let snapshot = serde_json::to_value(cache.snapshot()).unwrap();
        assert_eq!(snapshot["reserve"], 120_000_000);
        assert_eq!(snapshot["charged"], 0); // physical high-water, not the admitted reserve
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
        let mut cache = test_cache(-213, 100);
        let a = capture(&mut cache, 1, false);
        cache.captured_topology(a, 0, 1, false);
        cache.hit_cycle(a, Cycle::Verify);
        cache.serial_boundary();
        cache.serial_boundary();
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
        cache.refresh_rankings();
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.top_churn.len(), 1);
        assert_eq!(snapshot.top_hits.len(), 1);
        assert_eq!(snapshot.telemetry_bytes, std::mem::size_of::<KeyStats>());
        cache.forget(a);
        cache.refresh_rankings();
        assert_eq!(cache.snapshot().telemetry_bytes, 0);
        assert!(cache.snapshot().top_churn.is_empty());
    }

    #[test]
    fn boundary_cost_shares_sum_to_physical_growth_with_node_basis() {
        let mut cache = test_cache(-217, 64 << 20);
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
        let mut cache = test_cache(-214, 100);
        capture(&mut cache, 1, true);
        assert!(matches!(cache.miss(key(2), site(Preference::Eager), None, false), Miss::Eager));
        assert!(matches!(cache.miss(key(2), site(Preference::Eager), None, false), Miss::Eager));
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
        let mut cache = test_cache(-215, 100);
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
        let mut cache = test_cache(-218, 100);
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
    fn inventory_counters_match_owner_lifetimes_and_rankings_are_cached() {
        let mut cache = test_cache(-1010, 64 * GRAPH_CHUNK);
        let verify = |cache: &DeviceCache| {
            let expected = cache.records.values().fold(Inventory::default(), |mut total, r| {
                total.nodes += r.accounted.nodes; total.execs += r.accounted.execs;
                total.unknown += r.accounted.unknown; total.resident += r.accounted.resident;
                total.pinned += r.accounted.pinned; total
            });
            assert_eq!((cache.inventory.nodes, cache.inventory.execs, cache.inventory.unknown,
                cache.inventory.resident, cache.inventory.pinned),
                (expected.nodes, expected.execs, expected.unknown, expected.resident, expected.pinned));
        };
        let a = capture(&mut cache, 1, true);
        cache.captured_topology(a, 11, 1, false);
        let b = capture(&mut cache, 2, false);
        cache.captured_topology(b, 17, 1, false);
        verify(&cache);
        assert_eq!(cache.native_pools().live_nodes, 28);
        cache.hit(b);
        cache.boundary();
        let ranking = cache.snapshot().top_hits[0].hits;
        cache.hit(b);
        assert_eq!(cache.snapshot().top_hits[0].hits, ranking, "stats reads reuse cached rankings");
        assert_eq!(cache.snapshot().native_pools.live_nodes, 28);
        cache.mark_victim(false);
        verify(&cache);
        cache.destroyed(&[b], ReleaseMeasurement { released: 0 });
        verify(&cache);
        assert_eq!(cache.native_pools().live_nodes, 11);
        cache.remove_bank(1);
        verify(&cache);
        assert_eq!(cache.native_pools().live_execs, 0);
    }

    #[test]
    fn stats_top_lists_are_bounded_sorted_and_available_without_trace() {
        let mut cache = test_cache(-216, 1000);
        assert!(cache.trace.is_none());
        for n in 0..30 {
            let id = capture(&mut cache, n, false);
            cache.pinned_hits(id, Cycle::Decode, n + 1);
        }
        cache.refresh_rankings();
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.top_hits.len(), 16);
        assert_eq!(snapshot.top_hits[0].hits, 30);
        assert_eq!(snapshot.top_hits[15].hits, 15);
        while let Some(id) = cache.mark_victim(false) {
            cache.destroyed(&[id], ReleaseMeasurement { released: 0 });
        }
        cache.refresh_rankings();
        assert_eq!(cache.snapshot().top_churn.len(), 16);
        cache.remove_bank(1);
        assert_eq!(cache.snapshot().telemetry_bytes, 0);
    }

    #[test]
    fn declared_peer_span_survives_single_device_native_inspection() {
        let mut cache = test_cache(-8803, 100);
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
        let mut cache = test_cache(-200, 100);
        let mut captures = 0;
        for n in 0..100 {
            if matches!(cache.miss(key(n), site(Preference::Adaptive), None, false), Miss::Capture(_)) { captures += 1; }
        }
        assert_eq!(captures, 100, "before measurement no estimate-only pressure is applied");
        cache.measured_capture(90);
        assert!(cache.charged <= cache.reserve + 10);
        assert_eq!(cache.snapshot().pending_capture, 0);
    }
    #[test]
    fn arc_frequency_and_ghost_adaptation() {
        let mut cache = test_cache(-201, 100);
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
        let mut cache = test_cache(-202, 20);
        let a = capture(&mut cache, 1, true);
        assert!(cache.records[&a].list.is_none());
        assert!(cache.mark_victim(false).is_none());
        cache.charged = cache.reserve;
        assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Eager));
        assert_eq!(cache.snapshot().pinned, 10);
    }
    #[test]
    fn destruction_never_reduces_arena_and_both_pools_allow_reuse() {
        let mut cache = test_cache(-203, 100);
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
        let mut a = test_cache(-206, 100);
        let mut b = test_cache(-207, 100);
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
        let mut cache = test_cache(-205, 64 << 20);
        cache.basis = Basis::Nodes { node_rate: 2560, graph_rate: 65536 };
        let mut node_site = site(Preference::Capture);
        node_site.basis = cache.basis;
        let estimate = |nodes| Some(Estimate { bytes: nodes * 2560 + 65536,
            nodes: Some(nodes), node_rate: 2560, graph_rate: 65536, parallel_width: 1 });
        for n in 0..8 {
            assert!(matches!(cache.miss(key(n), node_site, estimate(1), false), Miss::Capture(_)));
        }
        cache.measured_capture(4 << 20);
        cache.serial_boundary();
        cache.serial_boundary();
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
        let mut cache = test_cache(-208, 100);
        let Miss::Capture(id) = cache.miss(key(1), site(Preference::Capture), None, false) else { panic!() };
        assert_eq!(cache.pools.node_high_water, 0);
        cache.cancel_capture(id);
        assert!(cache.needs_boundary());
        cache.measured_capture(10); // Failed instantiate may retain driver pages.
        assert_eq!((cache.pools.node_high_water, cache.pools.exec_high_water), (0, 0));
        assert_eq!(cache.pools.live_nodes, 0);
        assert!(matches!(cache.miss(key(2), site(Preference::Capture), None, false), Miss::Capture(_)),
            "below measured hard limit an estimate cannot permanently refuse the shape");
        assert_eq!(cache.pending_destroy, 0);
    }

    #[test]
    fn device_basis_and_coefficients_cannot_mix() {
        let mut cache = test_cache(-209, 64 << 20);
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
        let mut cache = test_cache(-210, 5 << 20);
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
        let mut cache = test_cache(-212, 100);
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
        assert!(matches!(cache.miss(key(4), site(Preference::Capture), None, false), Miss::Capture(_)));
        assert_eq!(cache.pending_growth, 40);
    }

    #[test]
    fn non_chain_and_declared_peer_sites_reserve_each_capture_below_old_peaks() {
        for peer in [false, true] {
            let mut cache = test_cache(-211, 100);
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
            assert!(matches!(cache.miss(key(4), declared, None, false), Miss::Capture(_)));
            assert_eq!(cache.pending_growth, 40);
        }
    }

    #[test]
    fn stats_report_only_when_evicted_and_trace_schema() {
        let mut cache = test_cache(-204, 100);
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

    fn large_soft_fixture(count: u64, nodes_each: u64) -> DeviceCache {
        let mut cache = test_cache(-99120, 10_000);
        for n in 0..count {
            let id = capture(&mut cache, n, false);
            cache.captured_topology(id, nodes_each, 1, false);
        }
        let total_nodes = count * nodes_each;
        cache.soft.measured_bytes = Some(1000);
        cache.soft.measured_nodes = Some(total_nodes);
        cache.soft.limit = 1000;
        cache.charged = 1000;
        cache
    }

    #[test]
    fn soft_room_batches_unfulfilled_cycle_demand_and_captures_after_retirement() {
        // 40 cold graphs x 10 nodes = 400-node soft cap (quarter clamp = 100 nodes).
        let mut cache = large_soft_fixture(40, 10);
        cache.begin_lane(0);
        // 6 distinct misses of 10 nodes in one cycle must queue 6 victims (60 nodes),
        // not stop at 1 victim while running misses #2..#6 without queuing room.
        for n in 100..106 {
            assert!(matches!(cache.miss(key(n), site(Preference::Capture), None, false), Miss::Eager));
        }
        assert_eq!(cache.pending_destroy, 6, "each unfulfilled miss must grow the cycle's victim batch");
        assert_eq!(cache.cycle_demand_nodes, 60);
        assert_eq!(cache.charged, 1000, "charged must remain strictly monotone");
        // Complete cycle, retire the whole batch on the next cycle boundary.
        cache.complete_lane(0);
        cache.begin_lane(0);
        let ready: Vec<u64> = cache.records.keys().copied().filter(|&id| cache.persistent_victim_ready(id)).collect();
        assert_eq!(ready.len(), 6);
        cache.destroyed(&ready, ReleaseMeasurement { released: 0 });
        assert_eq!(cache.charged, 1000, "destruction never credits charged");
        assert_eq!(cache.native_pools().live_nodes, 340);
        assert_eq!(cache.cycle_demand_nodes, 0);
        // All 6 previously eager shapes now fit and capture immediately in one cycle.
        for n in 100..106 {
            let Miss::Capture(id) = cache.miss(key(n), site(Preference::Capture), None, false) else {
                panic!("key {n} must capture immediately after batched retirement");
            };
            cache.captured_topology(id, 10, 1, false);
            cache.measured_capture(0);
        }
        assert_eq!(cache.native_pools().live_nodes, 400);
        assert_eq!(cache.charged, 1000);
    }

    #[test]
    fn soft_room_return_value_matches_queued_state() {
        let mut cache = large_soft_fixture(40, 10);
        cache.begin_lane(0);
        // At full soft cap, queuing victims returns Queued (never Full when room was queued).
        assert_eq!(cache.make_soft_room(10, None), SoftRoom::Queued);
        assert_eq!(cache.pending_destroy, 1);
        // A second distinct miss in the same cycle queues a second victim and also returns Queued.
        assert_eq!(cache.make_soft_room(10, None), SoftRoom::Queued);
        assert_eq!(cache.pending_destroy, 2);
        // Retiring those 2 victims makes room Available immediately without further evictions.
        cache.complete_lane(0);
        cache.begin_lane(0);
        let ready: Vec<u64> = cache.records.keys().copied().filter(|&id| cache.persistent_victim_ready(id)).collect();
        cache.destroyed(&ready, ReleaseMeasurement { released: 0 });
        assert_eq!(cache.make_soft_room(10, None), SoftRoom::Available);
        assert_eq!(cache.pending_destroy, 0);
        // When all remaining entries are hot, cannot make room -> Full with 0 selections.
        cache.soft.measured_nodes = Some(cache.native_pools().live_nodes);
        let ids: Vec<u64> = cache.records.iter()
            .filter(|(_, r)| matches!(r.list, Some(List::Recent | List::Frequent)))
            .map(|(&id, _)| id).collect();
        for id in ids { cache.hit(id); }
        assert_eq!(cache.make_soft_room(10, None), SoftRoom::Full);
        assert_eq!(cache.pending_destroy, 0);
    }

    #[test]
    fn soft_room_clamps_cycle_victim_batch_to_quarter_soft_cap() {
        // 40 cold graphs x 10 nodes = 400-node soft cap -> quarter clamp = 100 nodes (10 victims).
        let mut cache = large_soft_fixture(40, 10);
        cache.begin_lane(0);
        let mut queued = 0usize;
        let mut full = 0usize;
        for _ in 0..25 {
            match cache.make_soft_room(10, None) {
                SoftRoom::Queued => queued += 1,
                SoftRoom::Full => full += 1,
                SoftRoom::Available => panic!("soft cap is full before retirement"),
            }
        }
        assert_eq!(queued, 10, "at most soft.measured_nodes / 4 nodes queued in one cycle");
        assert_eq!(full, 15, "misses past the clamp must return Full without over-evicting");
        assert_eq!(cache.pending_destroy, 10);
        assert_eq!(cache.cycle_demand_nodes, 100);

        // Also check min(reclaimable, quarter): if only 40 nodes (4 graphs) are cold, clamp to 40.
        let mut cache2 = large_soft_fixture(40, 10);
        cache2.begin_lane(0);
        let all_ids: Vec<u64> = cache2.records.keys().copied().collect();
        for &id in &all_ids[..36] { cache2.hit(id); }
        let mut q2 = 0usize;
        for _ in 0..10 {
            if cache2.make_soft_room(10, None) == SoftRoom::Queued { q2 += 1; }
        }
        assert_eq!(q2, 4, "clamp must also respect reclaimable_nodes when < soft.measured_nodes / 4");
        assert_eq!(cache2.pending_destroy, 4);
    }

    #[test]
    fn soft_room_never_evicts_hot_graphs_when_demand_exceeds_cold_set_and_clamp() {
        // 40 graphs x 10 nodes = 400-node soft cap (quarter clamp = 100 nodes).
        // Put all 40 graphs in Frequent, keep 36 hot via HitToken at the LRU head
        // ahead of 4 cold graphs (40 reclaimable nodes < 100 quarter clamp).
        let mut cache = large_soft_fixture(40, 10);
        let mut ids: Vec<u64> = (0..40).map(|n| cache.identities[&key(n)]).collect();
        for &id in &ids { cache.hit(id); }
        cache.serial_boundary();
        cache.serial_boundary();
        cache.begin_lane(0);
        let epoch = cache.lanes[0].index;
        let hot_ids: Vec<u64> = ids.drain(..36).collect();
        let cold_ids = ids;
        for &id in &hot_ids {
            let token = cache.hit_token(id);
            token.hit(epoch);
        }
        // Demand 150 nodes (15 distinct misses x 10) > quarter clamp (100) AND > cold set (40).
        for n in 200..215 {
            assert!(matches!(cache.miss(key(n), site(Preference::Capture), None, false), Miss::Eager));
        }
        assert_eq!(cache.make_soft_room(10, None), SoftRoom::Full,
            "demand past the available cold set must return Full");
        assert_eq!(cache.pending_destroy, 4,
            "pending_destroy must never increase beyond the 4 cold victims");
        for &id in &cold_ids {
            assert!(cache.records[&id].pending_destroy, "cold graph {id} should be selected");
        }
        for &id in &hot_ids {
            assert!(!cache.records[&id].pending_destroy, "hot graph {id} must never be selected");
        }
        assert!(cache.counters[&Cycle::Decode].hot_guard_skips >= 36,
            "select_victim must walk past and skip hot graphs (saw {})",
            cache.counters[&Cycle::Decode].hot_guard_skips);
    }

    #[test]
    fn calibrated_live_bytes_tracks_node_share_not_slab_carrier_tags() {
        let mut cache = test_cache(-99121, 100 << 20);
        // 15 zero-growth captures + 1 slab-crossing 2 MiB capture, all 10 nodes each.
        let mut ids = Vec::new();
        for n in 0..16 {
            let Miss::Capture(id) = cache.miss(key(n), site(Preference::Capture), None, false) else { panic!() };
            cache.captured_topology(id, 10, 1, false);
            cache.measured_capture(if n == 0 { 2 << 20 } else { 0 });
            ids.push(id);
        }
        cache.serial_boundary();
        cache.serial_boundary();
        cache.soft.limit = 2 << 20;
        cache.soft.measured_bytes = Some(2 << 20);
        cache.soft.measured_nodes = Some(160);
        assert_eq!(cache.snapshot().live_bytes, 2 << 20);
        // Evict the slab-crossing carrier (id 0) and replace with a 0-byte reuse capture.
        let victim = ids[0];
        cache.records.get_mut(&victim).unwrap().pending_destroy = true;
        cache.pending_destroy += 1;
        cache.pending_lanes[0] += 1;
        cache.clocks[0].pending.fetch_add(1, Ordering::Release);
        cache.destroyed(&[victim], ReleaseMeasurement { released: 0 });
        assert_eq!(cache.snapshot().live_bytes, (2 << 20) * 150 / 160);
        let Miss::Capture(repl) = cache.miss(key(99), site(Preference::Capture), None, false) else { panic!() };
        cache.captured_topology(repl, 10, 1, false);
        cache.measured_capture(0);
        assert_eq!(cache.snapshot().live_bytes, 2 << 20, "reuse replacement keeps live_bytes proportional to live_nodes");
        assert_eq!(cache.charged, 2 << 20);
    }

    #[test]
    fn class_samples_counts_all_settled_captures_and_prevents_estimate_inflation() {
        let mut cache = test_cache(-99122, 100 << 20);
        // Initial zero-growth capture increments samples but still falls back to bootstrap (> 0).
        let Miss::Capture(_) = cache.miss(key(0), site(Preference::Capture), None, false) else { panic!() };
        cache.measured_capture(0);
        assert_eq!(cache.classes["test"].samples, 1);
        assert_eq!(cache.estimate(site(Preference::Capture), None).0, 10);
        // 14 more zero-growth captures + 1 two-MiB slab capture -> 16 samples total, 128 KiB mean.
        for n in 1..16 {
            let Miss::Capture(_) = cache.miss(key(n), site(Preference::Capture), None, false) else { panic!() };
            cache.measured_capture(if n == 15 { 2 << 20 } else { 0 });
        }
        assert_eq!(cache.classes["test"].samples, 16);
        let (est, headroom) = cache.estimate(site(Preference::Capture), None);
        assert_eq!(est, (2 << 20) / 16, "estimate amortizes slab bytes across all settled captures");
        assert_eq!(headroom, 2 << 20, "headroom retains worst single slab jump");
    }

    #[test]
    fn unseen_key_uses_site_mean_nodes_instead_of_max() {
        let mut cache = test_cache(-99123, 10_000);
        for n in 0..9 {
            let id = capture(&mut cache, n, false);
            cache.captured_topology(id, 6, 1, false);
        }
        let outlier = capture(&mut cache, 9, false);
        cache.captured_topology(outlier, 27, 1, false);
        // Total live_nodes = 9*6 + 27 = 81; site mean = ceil(81/10) = 9, max = 27.
        // Set soft cap to 91 (10 free nodes): fits site mean (9) without evicting, would fail max (27).
        cache.soft.measured_bytes = Some(100);
        cache.soft.measured_nodes = Some(91);
        cache.soft.limit = 100;
        cache.charged = 100;
        cache.begin_lane(0);
        assert!(matches!(cache.miss(key(100), site(Preference::Capture), None, false), Miss::Capture(_)),
            "unseen key must admit against site mean (9 <= 10), not max (27)");
        assert_eq!(cache.pending_destroy, 0);
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
        let next = cache.next_id + 1;
        let key = Identity { bank: 1, site: site.name, key: next, shape: next.to_string().into() };
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
        let mut cache = test_cache(6, 1000);
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
        let mut cache = test_cache(6, 1000);
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
        let mut cache = test_cache(-99122, 10_000);
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
        let mut cache = test_cache(device, 1000);
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
        let mut cache = test_cache(6, 1000);
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
            let mut cache = test_cache(device, 64 << 20);
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
