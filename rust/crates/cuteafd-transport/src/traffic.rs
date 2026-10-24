//! Off-path fabric observations. NIC data counters are four-octet words;
//! engine counters are protocol wire bytes (not payloads or receive capacities).
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::sync::{atomic::{AtomicBool, AtomicU64, Ordering}, Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PortSample {
    pub device: String,
    pub port: u32,
    pub state: Option<String>,
    pub link_gbps: Option<f64>,
    #[serde(default)]
    pub mtu: Option<u32>,
    /// Advances on an observed counter reset, disappearance or link change.
    #[serde(default)]
    pub counter_epoch: u64,
    #[serde(default)]
    pub health_since_start: Option<crate::fabric_health::Health>,
    /// Missing or unreadable counters are absent, not zero.
    pub counters: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_xmit_wait_ticks_per_s: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineSample {
    pub device: String,
    pub port: u32,
    pub peer: String,
    #[serde(default)]
    pub local_gid: String,
    /// Subset of sent_bytes posted as one-sided writes, including their flags.
    #[serde(default)]
    pub written_sent_bytes: u64,
    pub receives_writes: bool,
    pub sent_bytes: u64,
    pub received_bytes: u64,
    #[serde(default)]
    pub path_mtu: Option<u32>,
    #[serde(default)]
    pub ops_sent: Option<u64>,
    #[serde(default)]
    pub ops_received: Option<u64>,
    #[serde(default)]
    pub packets_sent: Option<u64>,
    #[serde(default)]
    pub packets_received: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HostSample {
    /// Process incarnation; a restart must not be interpreted as a counter wrap.
    pub incarnation: String,
    /// The host's own name (kernel hostname), for display; fabric keys stay addresses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    /// This host's latest GPU readings (`thermal::current()["gpus"]`), attached
    /// only to a worker's reply to the coordinator. Absent from older workers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<serde_json::Value>,
    /// Seconds on this host's monotonic clock, never compared across hosts.
    pub sampled_s: f64,
    /// Start of the continuous demand-driven observation window, host-local.
    #[serde(default)]
    pub observation_started_s: Option<f64>,
    #[serde(default)]
    pub health_since_start: Option<crate::fabric_health::Verdict>,
    pub ports: Vec<PortSample>,
    pub engine: Vec<EngineSample>,
    /// Future producers (e.g. storage) use the same cumulative byte shape.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sources: BTreeMap<String, SourceSample>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceSample {
    pub incarnation: String,
    pub counters: Vec<EngineSample>,
    /// Empty for historical sources; unavailable samples never imply zero.
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub mono_ns: Option<u64>,
}

/// A third producer supplies cumulative bytes keyed by device/port/peer. Sampling
/// is called only by the observer, never by an inference or verbs post path.
pub trait TrafficSource: Send + Sync {
    fn name(&self) -> &'static str;
    fn sample(&self) -> SourceSample;
    fn suspend(&self) {}
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Snapshot {
    pub hosts: BTreeMap<String, HostSample>,
}

#[derive(Debug, thiserror::Error)]
#[error("served fabric traffic probe")]
pub struct TrafficProbeOnly;

#[derive(Debug, thiserror::Error)]
pub enum TrafficError {
    #[error("fabric control I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("fabric control JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid fabric control reply")]
    InvalidReply,
}

/// Isolate each QP's hot counters from metadata and other QPs.
#[derive(Debug, Default)]
#[repr(align(64))]
struct WireBytes {
    written: AtomicU64,
    sent: AtomicU64,
    received: AtomicU64,
}

/// One handle per QP, registered before serving. Relaxed atomic adds on the
/// exchange path, no lookup, allocation, lock, logging or sysfs access.
#[derive(Debug)]
pub(crate) struct WireCounter {
    device: String,
    port: u32,
    path_mtu: Option<u32>,
    peer: OnceLock<String>,
    local_gid: String,
    receives_writes: AtomicBool,
    bytes: WireBytes,
}
impl WireCounter {
    pub(crate) fn sent(&self, bytes: usize) { self.bytes.sent.fetch_add(bytes as u64, Ordering::Relaxed); }
    pub(crate) fn written(&self, bytes: usize) {
        self.sent(bytes);
        self.bytes.written.fetch_add(bytes as u64, Ordering::Relaxed);
    }
    pub(crate) fn receives_writes(&self) { self.receives_writes.store(true, Ordering::Relaxed); }
    pub(crate) fn received(&self, bytes: usize) { self.bytes.received.fetch_add(bytes as u64, Ordering::Relaxed); }
    pub(crate) fn peer(&self, peer: &str) { let _ = self.peer.set(peer.into()); }
    fn sample(&self) -> EngineSample {
        EngineSample { device: self.device.clone(), port: self.port, peer: self.peer.get().cloned().unwrap_or_default(),
            receives_writes: self.receives_writes.load(Ordering::Relaxed),
            local_gid: self.local_gid.clone(), written_sent_bytes: self.bytes.written.load(Ordering::Relaxed),
            sent_bytes: self.bytes.sent.load(Ordering::Relaxed), received_bytes: self.bytes.received.load(Ordering::Relaxed), path_mtu: self.path_mtu, ..Default::default() }
    }
}

#[derive(Default)]
struct Registry {
    // Handles survive QP retirement: cumulative observations must never decrease.
    counters: Vec<Arc<WireCounter>>,
    peers: BTreeSet<SocketAddr>,
    sources: Vec<Arc<dyn TrafficSource>>,
}
fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

pub(crate) fn register(device: &str, port: u32, peer: &str, local_gid: String, mtu: u32) -> Arc<WireCounter> {
    let counter = Arc::new(WireCounter { device: device.into(), port, path_mtu: (1..=5).contains(&mtu).then(|| 128u32 << mtu), peer: { let p = OnceLock::new(); if !peer.is_empty() { let _ = p.set(peer.into()); } p },
        local_gid, receives_writes: AtomicBool::new(false), bytes: WireBytes::default() });
    registry().lock().unwrap().counters.push(counter.clone());
    counter
}

/// Total one-sided bytes written by every QP in this process (loopback tests).
#[cfg(test)]
pub(crate) fn written_total() -> u64 {
    registry().lock().unwrap().counters.iter().map(|c| c.bytes.written.load(Ordering::Relaxed)).sum()
}

/// Called during transport setup, not while dispatching requests.
pub fn observe_peers(peers: &[SocketAddr]) {
    registry().lock().unwrap().peers.extend(peers.iter().copied());
}
pub fn add_source(source: Arc<dyn TrafficSource>) {
    registry().lock().unwrap().sources.push(source);
}

fn latest() -> &'static Mutex<Snapshot> {
    static LATEST: OnceLock<Mutex<Snapshot>> = OnceLock::new();
    LATEST.get_or_init(|| Mutex::new(Snapshot::default()))
}

/// Reading the cache does not subscribe or wake the observer.
pub fn snapshot() -> Snapshot {
    latest().lock().map(|s| s.clone()).unwrap_or_default()
}
pub fn local() -> HostSample {
    renew();
    let mut sample = snapshot().hosts.remove("coordinator").unwrap_or_default();
    // Only on a peer's request (the coordinator's lease), never on a serving path.
    sample.gpus = cuteafd_core::thermal::current().get("gpus").cloned().filter(|g| g.as_array().is_some_and(|g| !g.is_empty()));
    sample
}

/// The Spark workers' GPU rows from the cached observation, named by host:
/// `[{host, temperature_c, power_w, sm_clock_mhz, throttling, sampled_s}]`.
/// Reads the cache only; the fabric lease (a console or `/v1/fabric` reader) keeps it fresh.
pub fn peer_gpus(subscribe: bool) -> Vec<serde_json::Value> {
    let snapshot = if subscribe { subscribed_snapshot() } else { snapshot() };
    let mut rows: Vec<serde_json::Value> = snapshot.hosts.iter().filter(|(key, _)| key.as_str() != "coordinator")
        .flat_map(|(key, host)| {
            let name = host.hostname.clone().unwrap_or_else(|| key.rsplit_once(':').map_or(key.as_str(), |(h, _)| h).to_string());
            host.gpus.as_ref().and_then(|g| g.as_array()).into_iter().flatten().map(move |gpu| serde_json::json!({
                "host": name, "temperature_c": gpu["temperature_c"], "power_w": gpu["power_w"],
                "sm_clock_mhz": gpu["sm_clock_mhz"], "throttling": gpu["throttling"], "sampled_s": gpu["sampled_s"],
            }))
        }).collect();
    rows.sort_by(|a, b| a["host"].as_str().cmp(&b["host"].as_str()));
    rows
}

/// Read every port and every numeric counter; unsupported files simply disappear.
pub fn read_ports(root: &Path) -> Vec<PortSample> {
    read_ports_with_net(root, Path::new("/sys/class/net"))
}
fn read_ports_with_net(root: &Path, net_root: &Path) -> Vec<PortSample> {
    let mut result = Vec::new();
    let Ok(devices) = std::fs::read_dir(root) else { return result };
    for device in devices.flatten() {
        let Ok(ports) = std::fs::read_dir(device.path().join("ports")) else { continue };
        for entry in ports.flatten() {
            let Some(port) = entry.file_name().to_str().and_then(|p| p.parse().ok()) else { continue };
            let path = entry.path();
            let mut counters = BTreeMap::new();
            for directory in ["counters", "hw_counters"] {
                let Ok(files) = std::fs::read_dir(path.join(directory)) else { continue };
                for file in files.flatten() {
                    let name = file.file_name().to_string_lossy().into_owned();
                    let Some(mut value) = std::fs::read_to_string(file.path()).ok()
                        .and_then(|s| s.trim().parse::<u64>().ok()) else { continue };
                    if matches!(name.as_str(), "port_xmit_data" | "port_rcv_data") { value = value.saturating_mul(4); }
                    // Hardware identity must not change when a standard read fails.
                    let key = if directory == "hw_counters" { format!("hw_counters/{name}") } else { name };
                    counters.insert(key, value);
                }
            }
            result.push(PortSample { device: device.file_name().to_string_lossy().into_owned(), port,
                state: std::fs::read_to_string(path.join("state")).ok().map(|s| s.trim().into()),
                link_gbps: std::fs::read_to_string(path.join("rate")).ok()
                    .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok()).filter(|v| v.is_finite() && *v > 0.0),
                mtu: std::fs::read_dir(path.join("gid_attrs/ndevs")).ok().and_then(|entries| {
                    let mtus: BTreeSet<u32> = entries.flatten().filter_map(|entry| {
                        let name = std::fs::read_to_string(entry.path()).ok()?;
                        let name = name.trim();
                        if name.is_empty() || name.contains('/') || name == "." || name == ".." { return None; }
                        std::fs::read_to_string(net_root.join(name).join("mtu")).ok()?.trim().parse().ok()
                    }).filter(|n| *n > 0).collect();
                    (mtus.len() == 1).then(|| *mtus.first().unwrap())
                }),
                counters, ..Default::default() });
        }
    }
    result.sort_by(|a, b| (&a.device, a.port).cmp(&(&b.device, b.port)));
    result
}

/// Completed phase evidence travels over the existing bounded control channel.
/// It is emitted off-path, once per host/phase, never per inference request.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhaseLog {
    pub name: String,
    pub timed: bool,
    pub ports: Vec<serde_json::Value>,
}
pub fn log_phase(phase: &PhaseLog) {
    if phase.name.len() > 128 || phase.ports.len() > 256 { return; }
    let summary = phase.ports.iter().filter_map(|p| {
        let pair = |tx: &str, rx: &str| p[tx].as_u64().zip(p[rx].as_u64()).map(|(a,b)| a.saturating_add(b));
        Some(serde_json::json!({"device": p["device"].as_str()?.chars().take(64).collect::<String>(),
            "port": p["port"].as_u64()?, "seconds": p["seconds"].as_f64(),
            "nic_bytes": pair("sent_bytes","received_bytes"), "engine_bytes": pair("engine_sent_bytes","engine_received_bytes"),
            "sparknest_bytes": pair("sparknest_sent_bytes","sparknest_received_bytes"),
            "headers_est_bytes": pair("estimated_header_sent_bytes","estimated_header_received_bytes"),
            "residual_bytes": pair("foreign_sent_bytes","foreign_received_bytes")}))
    }).collect::<Vec<_>>();
    tracing::info!(target: "cuteafd::fabric", phase = %phase.name, timed = phase.timed,
        attribution = %serde_json::json!(summary), "fabric phase bytes (tx+rx per port; estimated headers, residual includes retries)");
}
pub fn control_phase(value: &serde_json::Value) {
    if let Some(phase) = value.get("phase_log").and_then(|v| serde_json::from_value::<PhaseLog>(v.clone()).ok()) {
        log_phase(&phase);
    }
}
pub fn publish_phase(phase: PhaseLog) {
    if phase.name.len() > 128 || phase.ports.len() > 256 { return; }
    let local = PhaseLog { ports: phase.ports.iter().filter(|p| p["host"] == "coordinator").cloned().collect(), ..phase.clone() };
    log_phase(&local);
    let peers = registry().lock().unwrap().peers.clone();
    std::thread::scope(|scope| {
        for peer in peers {
            let selected = PhaseLog { ports: phase.ports.iter().filter(|p| p["host"].as_str() == Some(&peer.to_string())).cloned().collect(), ..phase.clone() };
            if selected.ports.is_empty() { continue; }
            scope.spawn(move || {
                use std::io::{BufRead, BufReader, Write};
                let timeout = Duration::from_millis(150);
                let send = || -> std::io::Result<()> {
                    let mut stream = TcpStream::connect_timeout(&peer, timeout)?;
                    stream.set_write_timeout(Some(timeout))?;
                    stream.set_read_timeout(Some(timeout))?;
                    let bytes = serde_json::to_vec(&serde_json::json!({"message":"fabric_traffic", "phase_log": selected}))?;
                    if bytes.len() > 64 * 1024 { return Ok(()); }
                    stream.write_all(&bytes)?; stream.write_all(b"\n")?;
                    // Consume the reply before closing: a broken write on an
                    // expert's control path must not be mistaken for wire failure.
                    let mut reply = String::new();
                    BufReader::new(std::io::Read::take(stream, 1 << 20)).read_line(&mut reply)?;
                    Ok(())
                };
                if let Err(error) = send() { tracing::debug!(target:"cuteafd::fabric", %peer, %error, "phase log delivery unavailable"); }
            });
        }
    });
}

fn read_peer(peer: SocketAddr) -> Result<HostSample, TrafficError> {
    use std::io::{BufRead, BufReader, Write};
    let timeout = Duration::from_millis(150);
    let mut stream = TcpStream::connect_timeout(&peer, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.write_all(b"{\"message\":\"fabric_traffic\"}\n")?;
    let mut line = String::new();
    BufReader::new(std::io::Read::take(stream, 1 << 20)).read_line(&mut line)?;
    let value: serde_json::Value = serde_json::from_str(&line)?;
    if value["message"] != "fabric_traffic" { return Err(TrafficError::InvalidReply); }
    let sample: HostSample = serde_json::from_value(value["sample"].clone())?;
    if sample.incarnation.is_empty() || !sample.sampled_s.is_finite() || sample.sampled_s <= 0.0 {
        return Err(TrafficError::InvalidReply);
    }
    Ok(sample)
}

/// Renewable leases expire after disconnected clients/coordinators disappear.
/// No lease means no sysfs reads, peer probes or timer wakeups.
#[derive(Default)]
struct Demand {
    until: Mutex<Option<Instant>>,
    wake: Condvar,
}
impl Demand {
    fn renew(&self, now: Instant) {
        *self.until.lock().unwrap() = Some(now + Duration::from_secs(5));
        self.wake.notify_one();
    }
    fn active(until: Option<Instant>, now: Instant) -> bool {
        until.is_some_and(|end| end > now)
    }
}
fn demand() -> &'static Demand {
    static DEMAND: OnceLock<Demand> = OnceLock::new();
    DEMAND.get_or_init(Demand::default)
}
fn sparknest() -> &'static Arc<crate::sparknest_traffic::SparkNest> {
    static SOURCE: OnceLock<Arc<crate::sparknest_traffic::SparkNest>> = OnceLock::new();
    SOURCE.get_or_init(|| {
        let source = Arc::new(crate::sparknest_traffic::SparkNest::new("/srv/sparknest/run/rdma-counters.sock"));
        add_source(source.clone());
        source
    })
}
/// Startup status checks only the local socket path, without subscribing.
pub fn storage_status() -> &'static str { if sparknest().present() { "socket-present" } else { "absent" } }
pub fn renew() {
    let _ = sparknest();
    demand().renew(Instant::now());
    start();
}
pub fn subscribed_snapshot() -> Snapshot {
    renew();
    snapshot()
}

/// Sysfs and network reads run on a niced OS thread at 1 Hz while leased.
/// API handlers only renew demand and clone cached observations.
pub fn background_priority() { cuteafd_ffi::background_thread_priority(); }

fn start() {
    static START: OnceLock<()> = OnceLock::new();
    START.get_or_init(|| {
        let _ = std::thread::Builder::new().name("fabric-observer".into()).spawn(|| {
            cuteafd_ffi::background_thread_priority();
            let clock = Instant::now();
            let incarnation = format!("{}-{}", std::process::id(), std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
            let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname").ok()
                .map(|h| h.trim().to_string()).filter(|h| !h.is_empty());
            let mut previous: Option<HostSample> = None;
            let mut idle = BTreeMap::new();
            let mut health_window = crate::fabric_health::Window::default();
            loop {
                {
                    let demand = demand();
                    let mut until = demand.until.lock().unwrap();
                    while !Demand::active(*until, Instant::now()) {
                        latest().lock().unwrap().hosts.clear();
                        for source in &registry().lock().unwrap().sources { source.suspend(); }
                        previous = None;
                        idle.clear();
                        health_window = crate::fabric_health::Window::default();
                        until = demand.wake.wait(until).unwrap();
                    }
                }
                let at = Instant::now();
                let (counters, peers, sources) = {
                    let state = registry().lock().unwrap();
                    (state.counters.clone(), state.peers.clone(), state.sources.clone())
                };
                let mut local = HostSample { incarnation: incarnation.clone(), sampled_s: 0.0, hostname: hostname.clone(),
                    ports: read_ports(Path::new("/sys/class/infiniband")),
                    engine: counters.iter().map(|counter| counter.sample()).collect(),
                    sources: sources.iter().map(|s| (s.name().into(), s.sample())).collect(), ..Default::default() };
                local.sampled_s = clock.elapsed().as_secs_f64();
                if let Some(before) = previous.as_ref() {
                    // Freeze the first quiet interval per port as its idle reference.
                    // If serving starts immediately, wait until engine bytes stop moving.
                    let total = |host: &HostSample| host.engine.iter()
                        .fold(0u64, |n, s| n.saturating_add(s.sent_bytes).saturating_add(s.received_bytes));
                    let quiet = total(&local) == total(before);
                    let dt = local.sampled_s - before.sampled_s;
                    for port in &mut local.ports {
                        let key = (port.device.clone(), port.port);
                        if quiet && dt > 0.0 {
                            if let Some(old) = before.ports.iter().find(|p| p.device == port.device && p.port == port.port) {
                                if let Some(delta) = port.counters.get("port_xmit_wait").zip(old.counters.get("port_xmit_wait"))
                                    .and_then(|(a,b)| a.checked_sub(*b)) {
                                    idle.entry(key.clone()).or_insert(delta as f64 / dt);
                                }
                            }
                        }
                        port.idle_xmit_wait_ticks_per_s = idle.get(&key).copied();
                    }
                }
                local.observation_started_s = Some(health_window.update(local.sampled_s, &mut local.ports));
                local.health_since_start = Some(crate::fabric_health::verdict(local.ports.iter()
                    .filter_map(|p| p.health_since_start.as_ref())));
                previous = Some(local.clone());
                // Publish local before remote reads: a peer probe must never wait on another probe.
                latest().lock().unwrap().hosts.insert("coordinator".into(), local);
                let remote: BTreeMap<_, _> = std::thread::scope(|scope| {
                    let jobs: Vec<_> = peers.into_iter().map(|peer| (peer, scope.spawn(move || read_peer(peer)))).collect();
                    jobs.into_iter().filter_map(|(peer, job)| job.join().ok()?.ok().map(|sample| (peer.to_string(), sample))).collect()
                });
                {
                    let mut state = latest().lock().unwrap();
                    state.hosts.retain(|name, _| name == "coordinator");
                    state.hosts.extend(remote);
                }
                std::thread::sleep(Duration::from_secs(1).saturating_sub(at.elapsed()));
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_worker_samples_without_hostname_or_gpus_still_parse() {
        let old = r#"{"incarnation":"1-2","sampled_s":3.5,"ports":[],"engine":[]}"#;
        let sample: HostSample = serde_json::from_str(old).unwrap();
        assert!(sample.hostname.is_none() && sample.gpus.is_none());
        // And a new sample round-trips its optional fields.
        let new = HostSample { hostname: Some("ostrich".into()), gpus: Some(serde_json::json!([{"temperature_c": 51}])), ..sample };
        let back: HostSample = serde_json::from_str(&serde_json::to_string(&new).unwrap()).unwrap();
        assert_eq!(back.hostname.as_deref(), Some("ostrich"));
        assert_eq!(back.gpus.unwrap()[0]["temperature_c"], 51);
    }
    #[test]
    fn sysfs_words_missing_and_bad_counters() {
        let root = std::env::temp_dir().join(format!("traffic-test-{}", std::process::id()));
        let port = root.join("nic/ports/1");
        std::fs::create_dir_all(port.join("counters")).unwrap();
        std::fs::write(port.join("counters/port_xmit_data"), "123\n").unwrap();
        std::fs::write(port.join("counters/port_rcv_data"), "not supported").unwrap();
        std::fs::create_dir_all(port.join("hw_counters")).unwrap();
        std::fs::write(port.join("hw_counters/port_xmit_data"), "456").unwrap();
        std::fs::write(port.join("rate"), "200 Gb/sec (4X HDR)").unwrap();
        std::fs::create_dir_all(port.join("gid_attrs/ndevs")).unwrap();
        std::fs::write(port.join("gid_attrs/ndevs/0"), "net0\n").unwrap();
        std::fs::create_dir_all(root.join("net/net0")).unwrap();
        std::fs::write(root.join("net/net0/mtu"), "9000\n").unwrap();
        let ports = read_ports_with_net(&root, &root.join("net"));
        assert_eq!(ports[0].mtu, Some(9000));
        assert_eq!(ports[0].counters["port_xmit_data"], 492);
        assert_eq!(ports[0].counters["hw_counters/port_xmit_data"], 1824);
        assert!(!ports[0].counters.contains_key("port_rcv_data"));
        assert_eq!(ports[0].link_gbps, Some(200.0));
        std::fs::write(port.join("counters/port_xmit_data"), "unavailable").unwrap();
        std::fs::write(port.join("hw_counters/port_xmit_data"), "500").unwrap();
        let next = read_ports(&root);
        assert!(!next[0].counters.contains_key("port_xmit_data"));
        assert_eq!(next[0].counters["hw_counters/port_xmit_data"], 2000);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn wire_counter_is_cumulative_and_directional() {
        let counter = WireCounter { device: "nic".into(), port: 1, path_mtu: Some(4096), peer: OnceLock::from("peer".to_string()),
            local_gid: "local".into(), receives_writes: AtomicBool::new(false), bytes: WireBytes::default() };
        counter.sent(128); counter.sent(4096); counter.written(72); counter.received(64);
        assert_eq!(counter.sample().sent_bytes, 4296);
        assert_eq!(counter.sample().written_sent_bytes, 72);
        assert_eq!(counter.sample().received_bytes, 64);
        assert_eq!(std::mem::align_of::<WireBytes>(), 64);
        assert_eq!((&counter.bytes as *const WireBytes as usize) % 64, 0);
    }
    #[test]
    fn peer_sample_must_be_initialized() {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut line = String::new();
            BufReader::new(stream.try_clone().unwrap()).read_line(&mut line).unwrap();
            writeln!(stream, "{}", serde_json::json!({"message": "fabric_traffic", "sample": HostSample::default()})).unwrap();
        });
        assert!(matches!(read_peer(address), Err(TrafficError::InvalidReply)));
        worker.join().unwrap();
    }

    #[test]
    fn sampling_demand_expires_and_renews() {
        let demand = Demand::default();
        let now = Instant::now();
        assert!(!Demand::active(*demand.until.lock().unwrap(), now));
        demand.renew(now);
        assert!(Demand::active(*demand.until.lock().unwrap(), now + Duration::from_secs(4)));
        assert!(!Demand::active(*demand.until.lock().unwrap(), now + Duration::from_secs(5)));
        demand.renew(now + Duration::from_secs(4));
        assert!(Demand::active(*demand.until.lock().unwrap(), now + Duration::from_secs(8)));
    }
}
