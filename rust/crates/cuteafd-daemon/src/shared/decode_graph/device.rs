//! Device-wide ARC policy. Native owners stay in GraphBank: this controller
//! marks victims, and banks destroy them only at a drained cycle boundary.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Cycle { Decode, Verify, Prefill, Wave, Other }

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

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub(crate) struct Identity {
    pub bank: u64,
    pub site: &'static str,
    pub shape: String,
}

#[derive(Default, Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Counters {
    pub hits: u64,
    pub misses: u64,
    pub captures: u64,
    pub eager_runs: u64,
    pub evictions: u64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Snapshot {
    pub device: i32,
    pub reserve: u64,
    pub charged: u64,
    pub pinned: u64,
    pub pending_capture: u64,
    pub cycles: HashMap<Cycle, Counters>,
    pub pools: Pools,
    pub basis: Basis,
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
}

#[derive(Default)]
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
struct TraceRow {
    device: i32,
    #[serde(flatten)]
    identity: Identity,
    cycle: Cycle,
    count: u64,
    cycles: HashMap<Cycle, u64>,
    first_seen_ms: u64,
    declared_capture_streams: usize,
    max_parallel_width: usize,
    spans_devices: bool,
}

pub(crate) enum Miss { Capture(u64), Eager }
pub(crate) type SharedCache = Arc<Mutex<DeviceCache>>;
static DEVICES: OnceLock<Mutex<HashMap<i32, SharedCache>>> = OnceLock::new();

pub(crate) fn device_cache(device: i32, reserve: u64, basis: Basis) -> anyhow::Result<SharedCache> {
    let mut devices = DEVICES.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner());
    if let Some(cache) = devices.get(&device).cloned() {
        anyhow::ensure!(cache.lock().unwrap_or_else(|p| p.into_inner()).reserve == reserve,
            "graph banks on device {device} disagree on reserve");
        anyhow::ensure!(cache.lock().unwrap_or_else(|p| p.into_inner()).basis == basis,
            "graph banks on device {device} disagree on estimator basis");
        return Ok(cache);
    }
    let mut controller = DeviceCache::new(device, reserve);
    controller.basis = basis;
    let cache = Arc::new(Mutex::new(controller));
    devices.insert(device, cache.clone());
    Ok(cache)
}

pub(crate) fn snapshots() -> Vec<Snapshot> {
    let caches: Vec<_> = DEVICES.get_or_init(Mutex::default).lock().unwrap_or_else(|p| p.into_inner())
        .values().cloned().collect();
    let mut rows: Vec<_> = caches.iter().map(|cache| cache.lock().unwrap_or_else(|p| p.into_inner()).snapshot()).collect();
    rows.sort_by_key(|row| row.device);
    rows
}

pub(crate) struct DeviceCache {
    device: i32,
    reserve: u64,
    charged: u64,
    pending_bytes: u64,
    next_id: u64,
    next_bank: u64,
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
    counters: HashMap<Cycle, Counters>,
    started: Instant,
    last_report: Instant,
    interval_evictions: u64,
    trace: Option<std::path::PathBuf>,
    last_trace: Instant,
    trace_rows: HashMap<Identity, TraceRow>,
}

impl DeviceCache {
    fn new(device: i32, reserve: u64) -> Self {
        Self { device, reserve, charged: 0, pending_bytes: 0, next_id: 0, next_bank: 0,
            recent_target: 0, records: HashMap::new(), identities: HashMap::new(),
            queues: std::array::from_fn(|_| Queue::default()), classes: HashMap::new(),
            pending_growth: 0, pools: Pools::default(), basis: Basis::Bytes,
            chain_only: true, unmeasured: false, pending_destroy: 0,
            counters: HashMap::new(), started: Instant::now(), last_report: Instant::now(),
            interval_evictions: 0, trace: std::env::var_os("CUTEAFD_GRAPH_TRACE").map(Into::into),
            last_trace: Instant::now() - Duration::from_secs(1), trace_rows: HashMap::new() }
    }

    pub fn bank(&mut self) -> u64 { self.next_bank += 1; self.next_bank }
    pub fn reserve(&self) -> u64 { self.reserve }
    pub fn needs_boundary(&self) -> bool {
        self.unmeasured || self.pending_destroy > 0
    }
    pub fn pending(&self, id: u64) -> bool { self.records.get(&id).is_some_and(|r| r.pending_destroy) }
    pub fn tracing(&self) -> bool { self.trace.is_some() }

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
        if let Some(record) = self.records.remove(&id) { self.identities.remove(&record.identity); }
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

    fn trace(&mut self, identity: &Identity, cycle: Cycle) {
        if self.trace.is_none() { return }
        let row = self.trace_rows.entry(identity.clone()).or_insert_with(|| TraceRow {
            device: self.device, identity: identity.clone(), cycle, count: 0, cycles: HashMap::new(),
            first_seen_ms: self.started.elapsed().as_millis() as u64, declared_capture_streams: 1,
            max_parallel_width: 0, spans_devices: false });
        row.count += 1;
        *row.cycles.entry(cycle).or_default() += 1;
    }

    #[cfg(test)]
    pub fn hit(&mut self, id: u64) {
        let Some(record) = self.records.get(&id) else { return };
        self.hit_cycle(id, record.cycle);
    }

    pub fn hit_cycle(&mut self, id: u64, cycle: Cycle) {
        let Some(record) = self.records.get(&id) else { return };
        let (pinned, pending) = (record.pinned, record.pending_destroy);
        if self.trace.is_some() { let identity = record.identity.clone(); self.trace(&identity, cycle); }
        self.counters.entry(cycle).or_default().hits += 1;
        if !pinned && !pending { self.push(id, List::Frequent); }
    }

    /// Pinned hits accumulate in the bank, without a lock or ARC mutation.
    pub fn pinned_hits(&mut self, cycle: Cycle, count: u64) {
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

    pub fn miss(&mut self, identity: Identity, site: Site, node_estimate: Option<Estimate>, pinned: bool) -> Miss {
        self.trace(&identity, site.cycle);
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
        let (estimate, headroom) = self.estimate(site, node_estimate.as_ref().map(|e| e.bytes));
        let old = self.identities.get(&identity).copied();
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
            let id = if let Some(id) = old { self.unlink(id); id } else {
                self.next_id += 1;
                let id = self.next_id;
                self.identities.insert(identity.clone(), id);
                self.records.insert(id, Record { identity, cycle: site.cycle, bytes: 0, list: None,
                    previous: None, next: None, pinned, pending_destroy: false, pending_capture: false, frequent: false, nodes, node_rate: metadata.node_rate, graph_rate: metadata.graph_rate, parallel_width: metadata.parallel_width, spans_devices: site.spans_devices });
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
            if let Some(row) = self.trace_rows.get_mut(&self.records[&id].identity) {
                row.declared_capture_streams = metadata.parallel_width;
                row.spans_devices |= site.spans_devices;
            }
            if metadata.parallel_width > 1 {
                tracing::warn!(site = site.name, width = metadata.parallel_width,
                    "parallel graph capture; chain node model does not apply");
            }
            // A returning ghost enters frequency, a cold key enters recency.
            if !pinned { self.push(id, if ghost.is_some() { List::Frequent } else { List::Recent }); }
            self.pending_bytes = self.pending_bytes.saturating_add(estimate);

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
            self.records.insert(id, Record { identity, cycle: site.cycle, bytes: estimate,
                list: None, previous: None, next: None, pinned: false, pending_destroy: false, pending_capture: false, frequent: false, nodes, node_rate: metadata.node_rate, graph_rate: metadata.graph_rate, parallel_width: metadata.parallel_width, spans_devices: site.spans_devices });
            self.push(id, List::RecentGhost);
            self.trim_ghosts();
        }
        self.counters.entry(site.cycle).or_default().eager_runs += 1;
        Miss::Eager
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
        if let Some(row) = self.trace_rows.get_mut(&record.identity) {
            row.max_parallel_width = width;
            row.spans_devices |= spans_devices;
        }
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
        // Observed free deltas remain available to alternate accounting adapters;
        // the default driver-arena model never credits physical release.
        let _ = measurement.released;
        for &id in ids {
            let Some(record) = self.records.get_mut(&id) else { continue };
            let evicted = record.pending_destroy;
            record.pending_destroy = false;
            if evicted { self.pending_destroy -= 1; }
            self.pools.live_nodes = self.pools.live_nodes.saturating_sub(record.nodes);
            self.pools.live_execs = self.pools.live_execs.saturating_sub(1);
            if evicted {
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
        let ids: Vec<_> = self.records.iter().filter(|(_, r)| r.identity.bank == bank).map(|(&id, _)| id).collect();
        for id in ids {
            let record = &self.records[&id];
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
        self.report_due(self.last_report.elapsed());
        if let Some(path) = &self.trace {
            // Trace mode counts every hit, but serializing thousands of keys
            // on every token would perturb the concurrency being observed.
            if self.last_trace.elapsed() < Duration::from_secs(1) { return; }
            self.last_trace = Instant::now();
            // Each device has its own snapshot file, avoiding concurrent writers.
            let path = path.with_extension(format!("device-{}.json", self.device));
            let mut rows: Vec<_> = self.trace_rows.values().collect();
            rows.sort_by(|a, b| (a.identity.site, a.identity.bank, &a.identity.shape)
                .cmp(&(b.identity.site, b.identity.bank, &b.identity.shape)));
            let document = serde_json::json!({"schema_version": 1, "cache": self.snapshot(), "graphs": rows,
                "sites": self.classes.iter().map(|(name, class)| (*name, serde_json::json!({
                    "samples": class.samples, "total_bytes": class.total, "largest_bytes": class.largest
                }))).collect::<HashMap<_, _>>()});
            let result = serde_json::to_vec(&document).map_err(std::io::Error::other)
                .and_then(|bytes| std::fs::write(&path, bytes));
            if let Err(error) = result { tracing::warn!(%error, path = %path.display(), "graph trace write failed"); }
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot { device: self.device, reserve: self.reserve, charged: self.charged,
            pinned: self.records.values().filter(|r| r.pinned).map(|r| r.bytes).sum(),
            pending_capture: self.pending_bytes, cycles: self.counters.clone(), pools: self.pools.clone(), basis: self.basis }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn site(preference: Preference) -> Site { Site { name: "test", cycle: Cycle::Decode, preference, bootstrap: 10, basis: Basis::Bytes, spans_devices: false } }
    fn key(n: u64) -> Identity { Identity { bank: 1, site: "test", shape: n.to_string() } }
    fn capture(cache: &mut DeviceCache, n: u64, pinned: bool) -> u64 {
        let Miss::Capture(id) = cache.miss(key(n), site(Preference::Capture), None, pinned) else { panic!("capture expected") };
        cache.measured_capture(10);
        id
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
        cache.trace = Some("unused".into());
        cache.trace(&key(1), Cycle::Decode);
        cache.trace(&key(1), Cycle::Verify);
        let row = serde_json::to_value(&cache.trace_rows[&key(1)]).unwrap();
        assert_eq!(row["site"], "test"); assert_eq!(row["count"], 2);
        assert_eq!(row["cycles"]["decode"], 1);
        assert_eq!(row["cycles"]["verify"], 1);
        assert!(row["first_seen_ms"].is_u64());
    }
}
