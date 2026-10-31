//! Fabric deltas share the NIC observation clock on each host. A missing,
//! reset, restarted or stale counter is unknown, never invented as zero.
//! Bytes no engine counter or registered source claims stay residual. The
//! transport meters every byte it posts, so a residual is same-port traffic
//! those counters cannot see: framing, retries, and same-host bulk whose
//! volume tracks the exchange. A host's own allowance is therefore bounded by
//! the exchange its engine reports, and residual beyond that bound is
//! reported as foreign traffic.
use cuteafd_transport::traffic::{EngineSample, HostSample, Snapshot};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Share of the engine-reported exchange a host may hold as residual before
/// the rest counts as foreign. Engine counters reconcile with the wire to a
/// few percent on steady phases, but short prefill phases carry same-port
/// bulk they cannot see; the worst own-share measured on a prefill phase was
/// 0.63 of the metered bytes. A flow that is not the card's own adds bytes
/// without adding engine bytes, so it clears this bound exactly.
const OWN_EXCHANGE_SLACK: f64 = 0.65;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Phase {
    pub name: String,
    pub timed: bool,
    pub ports: Vec<Port>,
    pub assessments: Vec<String>,
    #[serde(default)]
    pub health: BTreeMap<String, cuteafd_transport::fabric_health::Verdict>,
    /// Contention on the coordinator host during this phase; absent when the
    /// host did not report pressure. Recorded even when quiet, so a card can
    /// show the run was taken on an idle host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_load: Option<HostLoad>,
}

/// Coordinator-host contention over one phase. Pressure is the peak 10 s
/// "some" average seen at the window boundaries; scanning is pages/s across
/// the window. Values are kept low when the host was quiet.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HostLoad {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_some_avg10: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_some_avg10: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pgscan_pages_per_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load1: Option<f64>,
    /// Host-local observation window the values describe.
    #[serde(default)]
    pub seconds: f64,
}

/// CPU pressure, memory pressure and reclaim rates above which a phase's
/// timing cannot be trusted: concurrent cargo gates and kernel reclaim made
/// the V4.1 Engram gather 2-3x slower on the coordinator (#895).
const HOST_CPU_PRESSURE_WARN: f64 = 10.0;
const HOST_MEMORY_PRESSURE_WARN: f64 = 5.0;
const HOST_SCAN_PAGES_PER_S_WARN: f64 = 1_000.0;


#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Port {
    pub host: String,
    /// The host's own name (kernel hostname) when its sampler reports one; `host` stays the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    pub device: String,
    pub port: u32,
    pub state: Option<String>,
    pub link_gbps: Option<f64>,
    #[serde(default)]
    pub mtu: Option<u32>,
    #[serde(default)]
    pub health: Option<cuteafd_transport::fabric_health::Health>,
    #[serde(default)]
    pub health_since_start: Option<cuteafd_transport::fabric_health::Health>,
    pub seconds: f64,
    /// Actual host-local observation window; phase edges are sampled at 1 Hz.
    pub sampled_start_s: f64,
    pub sampled_end_s: f64,
    pub sent_bytes: Option<u64>,
    pub received_bytes: Option<u64>,
    pub engine_sent_bytes: Option<u64>,
    pub engine_received_bytes: Option<u64>,
    #[serde(default)]
    pub sparknest_sent_bytes: Option<u64>,
    #[serde(default)]
    pub sparknest_received_bytes: Option<u64>,
    #[serde(default)]
    pub sparknest_status: String,
    /// RoCEv2 62 B/estimated packet. Engine packets are an MTU lower bound;
    /// storage packets are per-operation estimates. ACK/RETH/retries stay residual.
    #[serde(default)]
    pub estimated_header_sent_bytes: Option<u64>,
    #[serde(default)]
    pub estimated_header_received_bytes: Option<u64>,
    #[serde(default)]
    pub engine_header_sent_bytes: Option<u64>,
    #[serde(default)]
    pub engine_header_received_bytes: Option<u64>,
    #[serde(default)]
    pub sparknest_header_sent_bytes: Option<u64>,
    #[serde(default)]
    pub sparknest_header_received_bytes: Option<u64>,
    #[serde(default)]
    pub residual_kind: String,
    pub foreign_sent_bytes: Option<u64>,
    pub foreign_received_bytes: Option<u64>,
    pub xmit_gbps: Option<f64>,
    pub rcv_gbps: Option<f64>,
    /// Full duplex: max(tx, rx) / link rate, not (tx + rx) / link rate.
    pub utilization: Option<f64>,
    pub counters: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xmit_wait_ticks_per_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_xmit_wait_ticks_per_s: Option<f64>,
    pub peers: Vec<EngineSample>,
}

fn delta(before: Option<&u64>, after: Option<&u64>) -> Option<u64> {
    after?.checked_sub(*before?)
}
fn engine(host: &HostSample, device: &str, port: u32) -> BTreeMap<String, (u64, u64)> {
    let mut peers = BTreeMap::new();
    for sample in host.engine.iter().filter(|s| s.device == device && s.port == port) {
        let pair = peers.entry(sample.peer.clone()).or_insert((0u64, 0u64));
        pair.0 = pair.0.saturating_add(sample.sent_bytes);
        pair.1 = pair.1.saturating_add(sample.received_bytes);
    }
    peers
}
fn source_delta(before: &HostSample, after: &HostSample, device: &str, port: u32) -> Option<(u64, u64)> {
    if before.sources.keys().ne(after.sources.keys()) { return None; }
    let mut total = (0u64, 0u64);
    for (name, a) in &after.sources {
        let b = before.sources.get(name)?;
        if a.incarnation != b.incarnation || a.incarnation.is_empty()
            || (!a.status.is_empty() && a.status != "available")
            || (!b.status.is_empty() && b.status != "available")
            || a.mono_ns.zip(b.mono_ns).is_some_and(|(a,b)| a <= b) { return None; }
        let totals = |source: &cuteafd_transport::traffic::SourceSample| {
            let mut peers = BTreeMap::new();
            for sample in source.counters.iter().filter(|s| s.device == device && s.port == port) {
                let pair = peers.entry(sample.peer.clone()).or_insert((0u64, 0u64));
                pair.0 = pair.0.saturating_add(sample.sent_bytes);
                pair.1 = pair.1.saturating_add(sample.received_bytes);
            }
            peers
        };
        let ea = totals(a); let eb = totals(b);
        if eb.keys().any(|key| !ea.contains_key(key)) { return None; }
        for (peer, (at, ar)) in ea {
            let (bt, br) = eb.get(&peer).copied().unwrap_or_default();
            total.0 = total.0.saturating_add(at.checked_sub(bt)?);
            total.1 = total.1.saturating_add(ar.checked_sub(br)?);
        }
    }
    Some(total)
}

fn storage_delta(before: &HostSample, after: &HostSample, device: &str, port: u32) -> Option<(u64,u64,u64,u64)> {
    let a = after.sources.get("sparknest")?; let b = before.sources.get("sparknest")?;
    if a.incarnation.is_empty() || a.incarnation != b.incarnation || a.status != "available" || b.status != "available"
        || a.mono_ns? <= b.mono_ns? { return None; }
    let find = |source: &cuteafd_transport::traffic::SourceSample| source.counters.iter()
        .find(|p| p.device == device && p.port == port).cloned();
    match (find(b), find(a)) {
        (Some(b),Some(a)) => Some((a.sent_bytes.checked_sub(b.sent_bytes)?, a.received_bytes.checked_sub(b.received_bytes)?,
            a.packets_sent?.checked_sub(b.packets_sent?)?.checked_mul(62)?,
            a.packets_received?.checked_sub(b.packets_received?)?.checked_mul(62)?)),
        // An available, unchanged port table explicitly has no storage QP here.
        (None,None) => Some((0,0,0,0)),
        _ => None,
    }
}
fn engine_headers(host: &HostSample, device: &str, port: u32, bytes: Option<u64>) -> Option<u64> {
    let n = bytes?;
    if n == 0 { return Some(0); }
    let mtu = host.engine.iter().filter(|s| s.device == device && s.port == port)
        .map(|s| s.path_mtu).collect::<Option<Vec<_>>>()?.into_iter().filter(|n| *n > 0).max()? as u64;
    n.div_ceil(mtu).checked_mul(62)
}
/// Peer-side counters cover older receivers that did not meter prefill ingress.
/// Match the exact coordinator peer GID to the Spark address, not all-host totals.
fn peer_exchange(phase: &Phase, port: &Port) -> (Option<u64>, Option<u64>) {
    if port.peers.is_empty() { return (None, None); }
    // Old seals omit local GIDs. Only use the host address if it identifies
    // one metered port; otherwise a second rail could borrow the same bytes.
    let local_gids: BTreeSet<_> = port.peers.iter().filter(|p| !p.local_gid.is_empty())
        .map(|p| p.local_gid.replace(':', "").to_lowercase()).collect();
    if local_gids.is_empty() && phase.ports.iter().filter(|p| p.host == port.host && !p.peers.is_empty()).count() != 1 {
        return (None, None);
    }
    let address = port.host.rsplit_once(':').map(|(a, _)| a).unwrap_or(&port.host);
    let gid = address.parse::<std::net::Ipv4Addr>().ok().map(|ip|
        ip.to_ipv6_mapped().octets().iter().map(|b| format!("{b:02x}")).collect::<String>());
    let peers: Vec<_> = phase.ports.iter().filter(|p| p.host == "coordinator" && p.seconds > 0.0)
        .flat_map(|p| &p.peers).filter(|p| {
            let peer_gid = p.peer.replace(':', "").to_lowercase();
            if local_gids.is_empty() { gid.as_ref() == Some(&peer_gid) } else { local_gids.contains(&peer_gid) }
        }).collect();
    if peers.is_empty() { return (None, None); }
    (Some(peers.iter().map(|p| p.received_bytes).sum()), Some(peers.iter().map(|p| p.sent_bytes).sum()))
}

/// Directional engine bytes reconciled with the matching peer's opposite meter.
pub fn exchange_bytes(phase: &Phase, port: &Port) -> (Option<u64>, Option<u64>) {
    let (tx, rx) = peer_exchange(phase, port);
    (port.engine_sent_bytes.map(|local| local.max(tx.unwrap_or(0))),
     port.engine_received_bytes.map(|local| local.max(rx.unwrap_or(0))))
}

/// Only measured background traffic, host contention and a down engine port
/// call for another A/B. Reassess raw evidence so reports from older seals do
/// not repeat stale labels.
pub fn rerun_reasons(phase: &Phase) -> Vec<String> {
    if !phase.timed { return Vec::new(); }
    let mut reasons = Vec::new();
    if let Some(load) = &phase.host_load {
        if load.cpu_some_avg10.is_some_and(|cpu| cpu > HOST_CPU_PRESSURE_WARN) {
            reasons.push(format!("host load: coordinator CPU pressure {:.1}% (>{}%)",
                load.cpu_some_avg10.unwrap_or_default(), HOST_CPU_PRESSURE_WARN));
        }
        if load.memory_some_avg10.is_some_and(|memory| memory > HOST_MEMORY_PRESSURE_WARN) {
            reasons.push(format!("host load: coordinator memory pressure {:.1}% (>{}%)",
                load.memory_some_avg10.unwrap_or_default(), HOST_MEMORY_PRESSURE_WARN));
        }
        if load.pgscan_pages_per_s.is_some_and(|scan| scan > HOST_SCAN_PAGES_PER_S_WARN) {
            reasons.push(format!("host load: coordinator page reclaim {:.0} pages/s (>{} pages/s)",
                load.pgscan_pages_per_s.unwrap_or_default(), HOST_SCAN_PAGES_PER_S_WARN));
        }
    }
    for p in &phase.ports {
        let label = format!("{} {}/{}", p.host, p.device, p.port);
        if (p.engine_sent_bytes.is_some_and(|n| n > 0) || p.engine_received_bytes.is_some_and(|n| n > 0))
            && p.state.as_deref().is_some_and(|s| s.contains("DOWN")) {
            reasons.push(format!("port down while used: {label}"));
        }
        if p.seconds <= 0.0 { continue; }
        let nic = p.sent_bytes.unwrap_or(0).saturating_add(p.received_bytes.unwrap_or(0));
        let storage = p.sparknest_sent_bytes.unwrap_or(0).saturating_add(p.sparknest_received_bytes.unwrap_or(0));
        if storage as f64 * 8.0 / p.seconds / 1e9 > 1.0 && storage as f64 > nic as f64 * 0.05 {
            reasons.push(format!("SparkNest traffic: {label} {:.2} Gb/s (>5% and >1 Gb/s)", storage as f64 * 8.0 / p.seconds / 1e9));
        }
        let (peer_tx, peer_rx) = peer_exchange(phase, p);
        for (physical, local, peer, storage, headers, side) in [
            (p.sent_bytes, p.engine_sent_bytes, peer_tx, p.sparknest_sent_bytes, p.estimated_header_sent_bytes, "sent"),
            (p.received_bytes, p.engine_received_bytes, peer_rx, p.sparknest_received_bytes, p.estimated_header_received_bytes, "received"),
        ] {
            let Some((physical, local)) = physical.zip(local) else { continue };
            // A registered but unavailable source cannot be identified as foreign.
            if !p.sparknest_status.is_empty() && p.sparknest_status != "unregistered" && storage.is_none() { continue; }
            let own = local.max(peer.unwrap_or(0));
            let bytes = physical.saturating_sub(own).saturating_sub((own as f64 * OWN_EXCHANGE_SLACK) as u64)
                .saturating_sub(headers.unwrap_or(0)).saturating_sub(storage.unwrap_or(0));
            if bytes as f64 * 8.0 / p.seconds / 1e9 > 1.0 && bytes as f64 > nic as f64 * 0.05 {
                reasons.push(format!("foreign traffic: {label} {:.2} Gb/s {side} beyond this host's own exchange (>5% and >1 Gb/s)", bytes as f64 * 8.0 / p.seconds / 1e9));
            }
        }
    }
    reasons
}

/// Coordinator-host contention across a phase window. The peak boundary
/// pressure is used because the 10 s PSI average at either end covers the
/// phase; scanning is a cumulative-counter delta over the same window.
fn host_load(before: &Snapshot, after: &Snapshot) -> Option<HostLoad> {
    let (bc, ac) = (before.hosts.get("coordinator")?, after.hosts.get("coordinator")?);
    // A restarted coordinator resets the cumulative scan counters.
    if bc.incarnation.is_empty() || ac.incarnation != bc.incarnation { return None; }
    let b = bc.host_load.as_ref()?;
    let a = ac.host_load.as_ref()?;
    let seconds = ac.sampled_s - bc.sampled_s;
    if !(seconds > 0.0) || !seconds.is_finite() { return None; }
    let peak = |old: Option<f64>, new: Option<f64>| match (old, new) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    };
    let scanned = |load: &cuteafd_transport::traffic::HostLoad| load.pgscan_kswapd.zip(load.pgscan_direct)
        .map(|(kswapd, direct)| kswapd.saturating_add(direct));
    let pages_per_s = scanned(a).zip(scanned(b)).and_then(|(now, then)| now.checked_sub(then))
        .map(|pages| pages as f64 / seconds);
    let load = HostLoad {
        cpu_some_avg10: peak(b.cpu_some_avg10, a.cpu_some_avg10),
        memory_some_avg10: peak(b.memory_some_avg10, a.memory_some_avg10),
        pgscan_pages_per_s: pages_per_s,
        load1: peak(b.load1, a.load1),
        seconds,
    };
    // An older coordinator reports no counters; keep the phase silent then.
    (load.cpu_some_avg10.is_some() || load.memory_some_avg10.is_some() || load.pgscan_pages_per_s.is_some())
        .then_some(load)
}

/// Fabric recovery and missing telemetry are informational, never invalid results.
pub fn caveats(phases: &[Phase]) -> Vec<String> {
    let mut lines = Vec::new();
    for phase in phases.iter().filter(|p| p.timed) {
        lines.extend(rerun_reasons(phase).into_iter().map(|s| format!("{}: {s}; compare in a quiet window", phase.name)));
        for p in &phase.ports {
            if let Some(h) = &p.health {
                if h.retransmit_judged && h.retransmit_pct.is_some_and(|n| n > h.retransmit_warn_pct.max(0.5)) {
                    lines.push(format!("{}: retransmits {}; results remain valid, throughput may be lower", phase.name,
                        cuteafd_transport::fabric_health::pct(h.retransmit_pct)));
                }
            }
        }
        lines.extend(phase.assessments.iter().filter(|a| a.starts_with("storage attribution unavailable:"))
            .map(|a| format!("{}: {a}; informational", phase.name)));
    }
    lines
}

pub fn interval(name: &str, timed: bool, before: &Snapshot, after: &Snapshot) -> Phase {
    let mut phase = Phase { name: name.into(), timed, ..Default::default() };
    // Keep disappeared/restarted/nonadvancing hosts and ports as explicit unknown
    // health evidence instead of silently dropping them from host verdicts.
    for host in before.hosts.keys().chain(after.hosts.keys()).collect::<BTreeSet<_>>() {
        let b = before.hosts.get(host); let a = after.hosts.get(host);
        let ports: BTreeSet<_> = b.into_iter().chain(a).flat_map(|h| h.ports.iter())
            .map(|p| (p.device.as_str(), p.port)).collect();
        for (device, number) in ports {
            let old = b.and_then(|h| h.ports.iter().find(|p| p.device == device && p.port == number));
            let new = a.and_then(|h| h.ports.iter().find(|p| p.device == device && p.port == number));
            let continuous = b.zip(a).is_some_and(|(b,a)| a.incarnation == b.incarnation && a.observation_started_s == b.observation_started_s
                && a.sampled_s > b.sampled_s && (a.sampled_s - b.sampled_s).is_finite());
            if continuous && old.is_some() && new.is_some() { continue; }
            let sample = new.or(old).unwrap();
            let mut health = cuteafd_transport::fabric_health::evaluate(sample, sample, 0.0);
            health.reasons = vec!["missing/restarted host or port, or nonadvancing observation".into()];
            phase.ports.push(Port { host: host.clone(), device: device.into(), port: number,
                hostname: a.or(b).and_then(|h| h.hostname.clone()),
                state: new.and_then(|p| p.state.clone()), link_gbps: new.and_then(|p| p.link_gbps),
                mtu: new.and_then(|p| p.mtu), health: Some(health), ..Default::default() });
        }
    }
    for (host, a) in &after.hosts {
        let Some(b) = before.hosts.get(host) else { continue };
        let seconds = a.sampled_s - b.sampled_s;
        if a.incarnation != b.incarnation || a.observation_started_s != b.observation_started_s || seconds <= 0.0 || !seconds.is_finite() { continue; }
        for port in &a.ports {
            let Some(old) = b.ports.iter().find(|p| p.device == port.device && p.port == port.port) else { continue };
            let counters: BTreeMap<_, _> = port.counters.iter().filter_map(|(key, value)|
                delta(old.counters.get(key), Some(value)).map(|d| (key.clone(), d))).collect();
            // A NIC counter reset invalidates byte attribution for this interval.
            let sent_bytes = counters.get("port_xmit_data").copied();
            let received_bytes = counters.get("port_rcv_data").copied();
            let eb = engine(b, &port.device, port.port);
            let ea = engine(a, &port.device, port.port);
            let mut peers = Vec::new();
            let ea_len = ea.len();
            for (peer, (tx, rx)) in ea {
                let (old_tx, old_rx) = eb.get(&peer).copied().unwrap_or_default();
                if let (Some(sent_bytes), Some(received_bytes)) = (tx.checked_sub(old_tx), rx.checked_sub(old_rx)) {
                    let local_gid = a.engine.iter().find(|p| p.peer == peer && p.device == port.device && p.port == port.port)
                        .map(|p| p.local_gid.clone()).unwrap_or_default();
                    peers.push(EngineSample { peer, local_gid, device: port.device.clone(), port: port.port, sent_bytes, received_bytes, ..Default::default() });
                }
            }
            let engine_sent_bytes = (peers.len() == ea_len && eb.keys().all(|peer| peers.iter().any(|p| &p.peer == peer)))
                .then(|| peers.iter().fold(0u64, |n, p| n.saturating_add(p.sent_bytes)));
            let engine_received_bytes = engine_sent_bytes
                .map(|_| peers.iter().fold(0u64, |n, p| n.saturating_add(p.received_bytes)));
            let sources = source_delta(b, a, &port.device, port.port);
            let storage = storage_delta(b, a, &port.device, port.port);
            let sparknest_sent_bytes = storage.map(|s| s.0);
            let sparknest_received_bytes = storage.map(|s| s.1);
            let sparknest_header_sent_bytes = storage.map(|s| s.2);
            let sparknest_header_received_bytes = storage.map(|s| s.3);
            let engine_header_sent_bytes = engine_headers(a, &port.device, port.port, engine_sent_bytes);
            let engine_header_received_bytes = engine_headers(a, &port.device, port.port, engine_received_bytes);
            let estimated_header_sent_bytes = engine_header_sent_bytes.zip(sparknest_header_sent_bytes).map(|(a,b)| a.saturating_add(b));
            let estimated_header_received_bytes = engine_header_received_bytes.zip(sparknest_header_received_bytes).map(|(a,b)| a.saturating_add(b));
            let sparknest_status = if storage.is_some() { "available".into() } else {
                a.sources.get("sparknest").map(|s| if s.status == "available" { "edge/reset mismatch" } else { &s.status }).unwrap_or("unregistered").into()
            };
            let rate = |bytes: u64| bytes as f64 * 8.0 / seconds / 1e9;
            let xmit_gbps = sent_bytes.map(rate);
            let rcv_gbps = received_bytes.map(rate);
            let utilization = port.link_gbps.filter(|r| *r > 0.0).and_then(|link|
                Some(xmit_gbps?.max(rcv_gbps?) / link));
            let foreign_sent_bytes = sent_bytes.zip(sources).zip(engine_sent_bytes).map(|((n, (tx, _)), engine)| n.saturating_sub(engine).saturating_sub(tx).saturating_sub(estimated_header_sent_bytes.unwrap_or(0)));
            let foreign_received_bytes = received_bytes.zip(sources).zip(engine_received_bytes).map(|((n, (_, rx)), engine)| n.saturating_sub(engine).saturating_sub(rx).saturating_sub(estimated_header_received_bytes.unwrap_or(0)));
            let label = format!("{host} {}/{}", port.device, port.port);
            let health = cuteafd_transport::fabric_health::evaluate(old, port, seconds);
            if timed && health.verdict == cuteafd_transport::fabric_health::Verdict::Warn {
                for reason in &health.reasons { phase.assessments.push(format!("fabric health: {label}: {reason}")); }
            }
            // Protocol bytes exclude Ethernet/IP/RDMA overhead, and prefill
            // carries same-port bulk the engine counters cannot see; both stay
            // in the residual. That residual is the host's own traffic up to
            // its engine-reported exchange times the slack, whatever port the
            // card configured; only the excess over that bound is foreign.
            let nic = sent_bytes.unwrap_or(0).saturating_add(received_bytes.unwrap_or(0));
            if timed && nic > 0 && a.sources.contains_key("sparknest") && storage.is_none() {
                phase.assessments.push(format!("storage attribution unavailable: {label} ({sparknest_status})"));
            }
            let storage_bytes = sparknest_sent_bytes.unwrap_or(0).saturating_add(sparknest_received_bytes.unwrap_or(0));
            if timed && rate(storage_bytes) > 1.0 && storage_bytes as f64 > nic as f64 * 0.05 {
                phase.assessments.push(format!("SparkNest traffic: {label} {:.2} Gb/s (>5% and >1 Gb/s)", rate(storage_bytes)));
            }
            let signal = |key: &str| counters.iter().any(|(name, n)| *n > 0 && (name == key || name.strip_prefix("hw_counters/") == Some(key)));
            let wait_rate = counters.get("port_xmit_wait").map(|ticks| *ticks as f64 / seconds);
            let rising_wait = wait_rate.zip(port.idle_xmit_wait_ticks_per_s)
                .is_some_and(|(rate, idle)| rate > idle + (idle * 0.1).max(1.0));
            if timed && rising_wait {
                phase.assessments.push(format!("xmit_wait rising above idle: {label}"));
            }
            if utilization.is_some_and(|u| u >= 0.7)
                && (rising_wait || ["np_cnp_sent", "rp_cnp_handled", "np_ecn_marked_roce_packets"].iter().any(|key| signal(key))) {
                phase.assessments.push(format!("congestion at high utilization: {label}"));
            }
            if counters.iter().any(|(key, n)| *n > 0 && (key.contains("retrans")
                || matches!(key.strip_prefix("hw_counters/").unwrap_or(key), "out_of_sequence" | "packet_seq_err" | "local_ack_timeout_err"))) {
                phase.assessments.push(format!("retransmit/sequence signal: {label}"));
            }
            let residual_kind = if counters.iter().any(|(key,n)| *n > 0 && (key.contains("retrans") || key.contains("sequence") || key.contains("packet_seq_err"))) {
                "unattributed/retry overhead"
            } else { "unattributed (headers/ACK/skew or foreign)" }.into();
            phase.ports.push(Port { host: host.clone(), device: port.device.clone(), port: port.port,
                hostname: a.hostname.clone(),
                mtu: port.mtu, health: Some(health), health_since_start: port.health_since_start.clone(),
                sampled_start_s: b.sampled_s, sampled_end_s: a.sampled_s, state: port.state.clone(), link_gbps: port.link_gbps, seconds, sent_bytes, received_bytes,
                engine_sent_bytes, engine_received_bytes, foreign_sent_bytes, foreign_received_bytes,
                sparknest_sent_bytes, sparknest_received_bytes, sparknest_status, residual_kind,
                estimated_header_sent_bytes, estimated_header_received_bytes,
                engine_header_sent_bytes, engine_header_received_bytes, sparknest_header_sent_bytes, sparknest_header_received_bytes,
                xmit_gbps, rcv_gbps, utilization, counters, xmit_wait_ticks_per_s: wait_rate,
                idle_xmit_wait_ticks_per_s: port.idle_xmit_wait_ticks_per_s, peers });
        }
    }
    phase.host_load = host_load(before, after);
    phase.assessments.extend(rerun_reasons(&phase));
    for host in before.hosts.keys().chain(after.hosts.keys()) {
        phase.health.insert(host.clone(), cuteafd_transport::fabric_health::verdict(phase.ports.iter()
            .filter(|p| &p.host == host).filter_map(|p| p.health.as_ref())));
    }
    phase
}

pub fn health_header(phases: &[Phase]) -> Option<String> {
    use cuteafd_transport::fabric_health::{pct, RETRANSMIT_WARN_PCT};
    let peak = phases.iter().filter(|p| p.timed).flat_map(|p| &p.ports)
        .filter_map(|p| p.health.as_ref()).filter(|h| h.retransmit_judged)
        .filter_map(|h| h.retransmit_pct).filter(|&rate| rate > RETRANSMIT_WARN_PCT)
        .reduce(f64::max)?;
    Some(format!("Fabric: peak retransmits {} of packets (above {})",
        pct(Some(peak)), pct(Some(RETRANSMIT_WARN_PCT))))
}

pub fn header(phases: &[Phase]) -> Option<String> {
    let ports: Vec<_> = phases.iter().flat_map(|phase| &phase.ports).collect();
    let used = |p: &Port| p.engine_sent_bytes.is_some_and(|n| n > 0) || p.engine_received_bytes.is_some_and(|n| n > 0);
    let spark = ports.iter().filter(|p| used(p)).find(|p| p.host != "coordinator").map(|p| p.host.as_str())
        .or_else(|| ports.iter().filter(|p| used(p)).next().map(|p| p.host.as_str()))?;
    let rails: BTreeMap<_, _> = ports.iter().filter(|p| p.host == spark
        && p.state.as_deref().is_some_and(|s| s.contains("ACTIVE"))
        && (p.engine_sent_bytes.is_some_and(|n| n > 0) || p.engine_received_bytes.is_some_and(|n| n > 0)))
        .filter_map(|p| Some(((p.device.as_str(), p.port), p.link_gbps?))).collect();
    if rails.is_empty() { return None; }
    let rates: Vec<_> = rails.values().map(|rate| format!("{rate:.0}")).collect();
    let links = if rates.windows(2).all(|w| w[0] == w[1]) { format!("{} x {} Gb/s", rates.len(), rates[0]) }
        else { format!("{} rails [{}] Gb/s", rates.len(), rates.join(" + ")) };
    let peak = ports.iter().filter_map(|p| p.utilization).reduce(f64::max);
    Some(match peak {
        Some(peak) => format!("RoCE {links} · peak {:.1}%", peak * 100.0),
        None => format!("RoCE {links}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_transport::traffic::PortSample;
    fn snapshot(time: f64, nic: u64, engine: u64) -> Snapshot {
        Snapshot { hosts: BTreeMap::from([("spark".into(), HostSample {
            incarnation: "one".into(), sampled_s: time,
            ports: vec![PortSample { device: "nic".into(), port: 1, link_gbps: Some(100.0), state: Some("4: ACTIVE".into()),
                counters: BTreeMap::from([("port_xmit_data".into(), nic), ("port_rcv_data".into(), nic),
                    ("rp_cnp_handled".into(), nic)]), idle_xmit_wait_ticks_per_s: None, ..Default::default() }],
            engine: vec![EngineSample { device: "nic".into(), port: 1, peer: "peer".into(),
                sent_bytes: engine, received_bytes: engine, ..Default::default() }], ..Default::default()
        })]) }
    }
    #[test]
    fn directional_utilization_and_congestion() {
        let phase = interval("decode", true, &snapshot(1.0, 0, 0), &snapshot(2.0, 10_000_000_000, 9_000_000_000));
        let port = &phase.ports[0];
        assert_eq!(port.utilization, Some(0.8));
        assert_eq!(port.foreign_sent_bytes, Some(1_000_000_000));
        // A residual inside the host's own-exchange allowance is not foreign.
        let found = caveats(std::slice::from_ref(&phase));
        assert!(found.iter().all(|c| !c.contains("foreign traffic")));
        assert_eq!(phase.assessments.len(), 1);
        assert_eq!(header(&[phase]).unwrap(), "RoCE 1 x 100 Gb/s · peak 80.0%");
    }
    fn exchange_host(key: &str, time: f64, nic: (u64, u64), engine: (u64, u64)) -> Snapshot {
        Snapshot { hosts: BTreeMap::from([(key.into(), HostSample {
            incarnation: "one".into(), sampled_s: time,
            ports: vec![PortSample { device: "nic".into(), port: 1, link_gbps: Some(200.0), state: Some("4: ACTIVE".into()),
                counters: BTreeMap::from([("port_xmit_data".into(), nic.0), ("port_rcv_data".into(), nic.1)]),
                ..Default::default() }],
            engine: vec![EngineSample { device: "nic".into(), port: 1, peer: "peer".into(),
                sent_bytes: engine.0, received_bytes: engine.1, ..Default::default() }], ..Default::default()
        })]) }
    }
    #[test]
    fn old_step_receiver_ingress_is_claimed_by_exact_coordinator_peer() {
        let mut phase = Phase { timed: true, ports: vec![
            Port { host: "10.55.0.1:19843".into(), seconds: 5.019005078,
                sent_bytes: Some(5_806_438_220), received_bytes: Some(5_908_870_656),
                engine_sent_bytes: Some(5_745_843_392), engine_received_bytes: Some(109_576_192),
                peers: vec![EngineSample { peer: "00000000000000000000ffff0a370016".into(), ..Default::default() }],
                ..Default::default() },
            Port { host: "coordinator".into(), seconds: 5.000768098,
                peers: vec![EngineSample { peer: "00000000000000000000ffff0a370001".into(),
                    sent_bytes: 6_018_321_408, received_bytes: 5_920_040_192, ..Default::default() }],
                ..Default::default() },
        ], ..Default::default() };
        assert!(rerun_reasons(&phase).is_empty());
        phase.ports[0].received_bytes = Some(11_908_870_656);
        assert_eq!(rerun_reasons(&phase).len(), 1);
        phase.timed = false;
        assert!(rerun_reasons(&phase).is_empty());
    }
    #[test]
    fn own_exchange_on_a_non_default_expert_port_is_not_foreign() {
        // A card whose expert port is 19935: the residual is its own native
        // exchange, inside this host's allowance, not a foreign flow on that
        // port; the configured port number is irrelevant to attribution.
        let key = "10.55.0.1:19935";
        let before = exchange_host(key, 1.0, (0, 0), (0, 0));
        let after = exchange_host(key, 2.0, (11_565_000_000, 7_304_000_000), (8_462_000_000, 4_439_000_000));
        let phase = interval("prefill", true, &before, &after);
        assert_eq!(phase.ports[0].foreign_sent_bytes, Some(3_103_000_000));
        assert_eq!(phase.ports[0].foreign_received_bytes, Some(2_865_000_000));
        assert!(caveats(std::slice::from_ref(&phase)).is_empty(), "{:?}", phase.assessments);

        // The same port receiving bytes no host in the card sent: foreign.
        let after = exchange_host(key, 2.0, (11_565_000_000, 9_500_000_000), (8_462_000_000, 4_439_000_000));
        let phase = interval("prefill", true, &before, &after);
        let found = caveats(&[phase]);
        assert!(found.iter().any(|c| c.contains("foreign traffic") && c.contains(key) && c.contains("received beyond")),
            "{found:?}");

        // A second symmetric flow on top of the host's own exchange is not the
        // host's own traffic just because both directions rose: it exceeds the
        // engine-reported exchange bound in both directions.
        let after = exchange_host(key, 2.0, (15_693_000_000, 9_658_500_000), (8_462_000_000, 4_439_000_000));
        let phase = interval("prefill", true, &before, &after);
        let found = caveats(&[phase]);
        assert_eq!(found.iter().filter(|c| c.contains("foreign traffic")).count(), 2, "{found:?}");
        assert!(found.iter().any(|c| c.contains("sent beyond")), "{found:?}");
        assert!(found.iter().any(|c| c.contains("received beyond")), "{found:?}");
    }
    #[test]
    fn source_restart_and_disappearance_leave_foreign_unknown() {
        let mut before = snapshot(1.0, 0, 0);
        let mut after = snapshot(2.0, 1000, 100);
        let source = cuteafd_transport::traffic::SourceSample { incarnation: "a".into(),
            counters: vec![EngineSample { device: "nic".into(), port: 1, sent_bytes: 100,
                received_bytes: 100, ..Default::default() }], ..Default::default() };
        before.hosts.get_mut("spark").unwrap().sources.insert("storage".into(), source.clone());
        assert_eq!(interval("x", true, &before, &after).ports[0].foreign_sent_bytes, None);
        let mut next = source.clone(); next.incarnation = "b".into();
        after.hosts.get_mut("spark").unwrap().sources.insert("storage".into(), next);
        assert_eq!(interval("x", true, &before, &after).ports[0].foreign_sent_bytes, None);
        after.hosts.get_mut("spark").unwrap().sources.insert("storage".into(), source);
        assert_eq!(interval("x", true, &before, &after).ports[0].foreign_sent_bytes, Some(900));
    }
    #[test]
    fn mixed_mtu_engine_headers_remain_a_lower_bound() {
        let mut host = HostSample::default();
        host.engine = [1024,4096].into_iter().map(|mtu| EngineSample {
            device:"nic".into(), port:1, path_mtu:Some(mtu), ..Default::default()
        }).collect();
        assert_eq!(engine_headers(&host,"nic",1,Some(8192)),Some(124));
        host.engine[0].path_mtu=None;
        assert_eq!(engine_headers(&host,"nic",1,Some(8192)),None);
    }
    #[test]
    fn storage_headers_residual_and_contamination() {
        let mut before = snapshot(1.0,0,0); let mut after = snapshot(2.0,1_000_000_000,100_000_000);
        for (snapshot, n) in [(&mut before,0u64),(&mut after,900_000_000u64)] {
            let host = snapshot.hosts.get_mut("spark").unwrap(); host.engine[0].path_mtu = Some(4096);
            host.sources.insert("sparknest".into(), cuteafd_transport::traffic::SourceSample {
                incarnation:"storage:1".into(), status:"available".into(), mono_ns:Some(n+1),
                counters:vec![EngineSample { device:"nic".into(),port:1,sent_bytes:n,received_bytes:n,
                    packets_sent:Some(n/4096),packets_received:Some(n/4096), ..Default::default() }],
            });
        }
        let phase = interval("decode",true,&before,&after); let p=&phase.ports[0];
        assert_eq!(p.sparknest_sent_bytes,Some(900_000_000));
        assert_eq!(p.sparknest_header_sent_bytes,Some((900_000_000/4096)*62));
        assert_eq!(p.engine_header_sent_bytes,Some(100_000_000u64.div_ceil(4096)*62));
        assert_eq!(p.foreign_sent_bytes,Some(0));
        assert!(caveats(&[phase]).iter().any(|c| c.contains("SparkNest traffic:") && c.contains("quiet window")));
        let host=after.hosts.get_mut("spark").unwrap(); host.sources.get_mut("sparknest").unwrap().status="stale".into();
        let phase=interval("decode",true,&before,&after);
        assert_eq!(phase.ports[0].sparknest_sent_bytes,None); assert_eq!(phase.ports[0].foreign_sent_bytes,None);
        assert!(caveats(&[phase]).iter().any(|c| c.contains("unavailable")));
    }
    #[test]
    fn storage_reset_missing_and_nonstorage_ports() {
        let mut before=snapshot(1.0,0,0); let mut after=snapshot(2.0,1000,100);
        for (snapshot, stamp) in [(&mut before,1u64),(&mut after,2u64)] {
            snapshot.hosts.get_mut("spark").unwrap().sources.insert("sparknest".into(),cuteafd_transport::traffic::SourceSample {
                incarnation:"one:1".into(),status:"available".into(),mono_ns:Some(stamp),..Default::default() });
        }
        assert_eq!(interval("x",true,&before,&after).ports[0].sparknest_sent_bytes,Some(0));
        after.hosts.get_mut("spark").unwrap().sources.get_mut("sparknest").unwrap().incarnation="one:2".into();
        assert_eq!(interval("x",true,&before,&after).ports[0].sparknest_sent_bytes,None);
        after.hosts.get_mut("spark").unwrap().sources.clear();
        assert_eq!(interval("x",true,&before,&after).ports[0].foreign_sent_bytes,None);
    }
    #[test]
    fn stale_reset_restart_and_missing_are_not_traffic() {
        let before = snapshot(1.0, 500, 500);
        let stale = interval("x", true, &before, &before);
        assert_eq!(stale.ports[0].health.as_ref().unwrap().verdict, cuteafd_transport::fabric_health::Verdict::Unknown);
        assert_eq!(stale.ports[0].sent_bytes, None);
        let mut after = snapshot(2.0, 100, 600);
        let phase = interval("x", true, &before, &after);
        assert_eq!(phase.ports[0].sent_bytes, None);
        after.hosts.get_mut("spark").unwrap().incarnation = "two".into();
        assert_eq!(interval("x", true, &before, &after).ports[0].sent_bytes, None);
    }
    #[test]
    fn health_evidence_survives_reports_and_only_timed_warnings_gate() {
        let mut before = snapshot(1.0, 0, 0);
        let mut after = snapshot(2.0, 5_000_000_000, 0);
        for (snapshot, marks, gaps) in [(&mut before, 0, 0), (&mut after, 300_000, 600)] {
            let p = &mut snapshot.hosts.get_mut("spark").unwrap().ports[0];
            for key in ["out_of_sequence", "packet_seq_err", "roce_adp_retrans", "local_ack_timeout_err", "rnr_nak_retry_err"] {
                p.counters.insert(format!("hw_counters/{key}"), if key == "out_of_sequence" { gaps } else { 0 });
            }
            p.counters.insert("hw_counters/np_ecn_marked_roce_packets".into(), marks);
        }
        let phase = interval("decode", true, &before, &after);
        assert_eq!(phase.health["spark"], cuteafd_transport::fabric_health::Verdict::Ok);
        let json = serde_json::to_value(&phase).unwrap();
        // 600 gaps over 10 GB at a 4 KB RoCE MTU: 0.025% of packets.
        assert_eq!(json["ports"][0]["health"]["retransmit_events"], 600);
        assert_eq!(health_header(&[phase]), None);
        after.hosts.get_mut("spark").unwrap().ports[0].counters.insert("hw_counters/np_ecn_marked_roce_packets".into(), 0);
        let phase = interval("decode", true, &before, &after);
        assert_eq!(phase.health["spark"], cuteafd_transport::fabric_health::Verdict::Ok);
        assert!(!caveats(&[phase]).iter().any(|s| s.contains("retransmits")));
        assert!(caveats(&[interval("warmup", false, &before, &after)]).is_empty());
    }
    #[test]
    fn health_header_only_shows_judged_timed_retransmit_warnings() {
        use cuteafd_transport::fabric_health::{Health, Verdict, RETRANSMIT_WARN_PCT};
        let mut phase = Phase { timed: true, ports: vec![Port { health: Some(Health {
            verdict: Verdict::Warn, retransmit_pct: Some(1.2), retransmit_judged: true,
            ..Default::default()
        }), ..Default::default() }], ..Default::default() };
        assert_eq!(health_header(&[phase.clone()]).as_deref(),
            Some("Fabric: peak retransmits 1.2% of packets (above 0.5%)"));
        phase.timed = false;
        assert_eq!(health_header(&[phase.clone()]), None);
        phase.timed = true;
        phase.ports[0].health.as_mut().unwrap().retransmit_judged = false;
        assert_eq!(health_header(&[phase.clone()]), None);
        let health = phase.ports[0].health.as_mut().unwrap();
        health.retransmit_judged = true;
        health.retransmit_pct = Some(RETRANSMIT_WARN_PCT);
        health.verdict = Verdict::Ok;
        assert_eq!(health_header(&[phase]), None);
        assert_eq!(health_header(&[]), None);
    }

    #[test]
    fn coordinator_host_load_flags_pressure_and_reclaim() {
        use cuteafd_transport::traffic::HostLoad as Sample;
        fn snapshot(time: f64, load: Sample) -> Snapshot {
            Snapshot { hosts: BTreeMap::from([("coordinator".into(), HostSample {
                incarnation: "one".into(), sampled_s: time, host_load: Some(load), ..Default::default() })]) }
        }
        let sample = |cpu, memory, kswapd, direct, load1| Sample {
            cpu_some_avg10: Some(cpu), memory_some_avg10: Some(memory),
            pgscan_kswapd: Some(kswapd), pgscan_direct: Some(direct), load1: Some(load1) };
        let caveat = |before: Sample, after: Sample, timed| {
            caveats(&[interval("decode", timed, &snapshot(10.0, before), &snapshot(20.0, after))])
        };

        // A quiet host records its values and produces no caveat.
        let quiet = caveat(sample(1.0, 0.5, 1_000, 0, 0.4), sample(2.0, 1.0, 1_000, 0, 0.5), true);
        assert!(quiet.is_empty(), "{quiet:?}");
        let phase = interval("decode", true, &snapshot(10.0, sample(1.0, 0.5, 1_000, 0, 0.4)),
            &snapshot(20.0, sample(2.0, 1.0, 1_000, 0, 0.5)));
        let load = phase.host_load.expect("quiet values are recorded");
        assert_eq!(load.cpu_some_avg10, Some(2.0));
        assert_eq!(load.memory_some_avg10, Some(1.0));
        assert_eq!(load.pgscan_pages_per_s, Some(0.0));
        assert_eq!(load.seconds, 10.0);

        let cpu = caveat(sample(2.0, 0.5, 0, 0, 0.4), sample(25.0, 1.0, 0, 0, 0.5), true);
        assert_eq!(cpu.len(), 1, "{cpu:?}");
        assert!(cpu[0].contains("host load") && cpu[0].contains("CPU pressure") && cpu[0].contains("25.0%"),
            "{cpu:?}");
        assert!(cpu[0].contains("quiet window"), "{cpu:?}");

        let memory = caveat(sample(2.0, 0.5, 0, 0, 0.4), sample(2.0, 9.0, 0, 0, 0.5), true);
        assert_eq!(memory.len(), 1, "{memory:?}");
        assert!(memory[0].contains("memory pressure") && memory[0].contains("9.0%"), "{memory:?}");

        // 15k + 5k pages over 10 s is 2000 pages/s of kswapd + direct reclaim.
        let reclaim = caveat(sample(2.0, 0.5, 0, 0, 0.4), sample(2.0, 1.0, 15_000, 5_000, 0.5), true);
        assert_eq!(reclaim.len(), 1, "{reclaim:?}");
        assert!(reclaim[0].contains("page reclaim") && reclaim[0].contains("2000 pages/s"), "{reclaim:?}");

        // Warm-up and other untimed phases never gate.
        let untimed = caveat(sample(2.0, 0.5, 0, 0, 0.4), sample(90.0, 90.0, 900_000, 900_000, 9.0), false);
        assert!(untimed.is_empty(), "{untimed:?}");
    }

    #[test]
    fn disappeared_or_restarted_evidence_is_unknown() {
        let before = snapshot(1.0, 0, 0);
        let mut after = snapshot(2.0, 100, 0);
        after.hosts.get_mut("spark").unwrap().incarnation = "restart".into();
        for phase in [interval("x", true, &before, &after), interval("x", true, &before, &Snapshot::default())] {
            assert_eq!(phase.health["spark"], cuteafd_transport::fabric_health::Verdict::Unknown);
            assert_eq!(phase.ports[0].sent_bytes, None);
            assert_eq!(health_header(&[phase]), None);
        }
    }

}
