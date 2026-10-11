//! Flow labels for the coordinator's expert QPs (`CUTEAFD_RDMA_BOND_BALANCE`,
//! see [`crate::bond`]): one process-wide [`Placement`], the probe exchange
//! that measures which bond member a label's traffic arrives on, and the
//! per-transport [`FlowSlots`] that hold each rank's label for as long as the
//! transport lives, so a reconnect keeps its member. Each transport (a GLM
//! Flash lane) is split across the members on its own: its ranks answer one
//! wave together, so four flows on one member overrun it however the other
//! lane sits. Every placement logs its transport's member counts, and each
//! transport logs one line saying whether its flows split evenly.
//!
//! A probe is its own small QP pair on the worker's ordinary listener: the
//! worker connects its side with the label, exposes 4 MiB for remote reads,
//! and the coordinator reads them while sampling its bond members' received
//! bytes. Workers without probe support close the connection, and that rank
//! keeps label 0 (the kernel's QP-number label, as before).
use super::*;
use crate::bond::{self, Assignment, BondBalance, BondPorts, EthtoolCounters, Placement, PortCounters, ProbeResult};
use std::collections::BTreeSet;
use std::sync::atomic::AtomicU64;
use std::sync::OnceLock;

const FLOW_PROBE_START: &str = "rdma_flow_probe_start";
const FLOW_PROBE_READY: &str = "rdma_flow_probe_ready";
const FLOW_PROBE_DONE: &str = "rdma_flow_probe_done";
/// The probe QPs' unused direction.
const PROBE_CONTROL_BYTES: usize = 4096;
const PROBE_CONTROL_TIMEOUT: Duration = Duration::from_secs(10);
const PROBE_READ_TIMEOUT_MS: u32 = 2_000;
/// Inconclusive measurements of one label before moving to the next.
const PROBE_ATTEMPTS: u32 = 3;
/// Probing time one flow may spend before it settles for what was measured.
const PLACEMENT_BUDGET: Duration = Duration::from_secs(10);
/// Bytes each bond member received on the wire (`ethtool -S`), RDMA included.
const RX_COUNTER: &str = "rx_bytes_phy";

/// Sample physical member counters away from the request polling hot path.
/// Physical counters include competing traffic, which is named in the log.
pub(super) fn monitor_bond(selection: &crate::fabric::RdmaSelection, address: IpAddr) {
    static MONITORS: OnceLock<Mutex<BTreeSet<String>>> = OnceLock::new();
    let IpAddr::V4(address) = address else { return };
    let gid = format!("00000000000000000000ffff{:08x}", u32::from(address));
    let Ok(Some(bond)) = bond::discover_bond(std::path::Path::new("/sys"),
        &selection.device, selection.port, &gid) else { return };
    let Ok(mut monitored) = MONITORS.get_or_init(|| Mutex::new(BTreeSet::new())).lock() else { return };
    if monitored.contains(&bond.bond) { return; }
    let mut counters = match EthtoolCounters::open(&bond.slaves, RX_COUNTER) {
        Ok(counters) => counters,
        Err(error) => { tracing::warn!(%error, bond = %bond.bond, "cannot monitor RoCE bond traffic"); return; }
    };
    monitored.insert(bond.bond.clone());
    thread::spawn(move || {
        let Ok(mut before) = counters.read() else { return };
        loop {
            thread::sleep(Duration::from_secs(5));
            let Ok(after) = counters.read() else { return };
            let deltas: Vec<_> = after.iter().zip(&before).map(|(a, b)| a.saturating_sub(*b)).collect();
            before = after;
            let total: u64 = deltas.iter().sum();
            if total < 64 * 1024 * 1024 { continue; }
            for (member, bytes) in bond.slaves.iter().zip(deltas) {
                let share = bytes as f64 / total as f64;
                tracing::info!(bond = %bond.bond, %member, bytes, share_percent = share * 100.0,
                    "expert fabric bond traffic share (physical counters include competing traffic)");
                if bond.slaves.len() == 2 && !(0.42..=0.58).contains(&share) {
                    tracing::warn!(bond = %bond.bond, %member, share_percent = share * 100.0,
                        "RoCE bond traffic outside 42-58 percent balance");
                }
            }
        }
    });
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct VerbsHostFlowProbeStart {
    pub(super) message: String,
    pub(super) flow_label: u32,
    pub(super) probe_bytes: usize,
    pub(super) client_native_endpoint: VerbsHostNativeEndpointDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct VerbsHostFlowProbeReady {
    pub(super) message: String,
    pub(super) flow_label: u32,
    pub(super) server_native_endpoint: VerbsHostNativeEndpointDescriptor,
    pub(super) read_addr: u64,
    pub(super) read_rkey: u32,
    pub(super) read_bytes: usize,
}

/// A worker's control connection carried only flow probes: there is no
/// expert session to admit.
#[derive(Debug, thiserror::Error)]
#[error("control connection carried RDMA flow probes only")]
pub struct FlowProbesOnly;

pub(super) fn is_flow_probe_start(value: &serde_json::Value) -> bool {
    control_message(value).is_ok_and(|message| message == FLOW_PROBE_START)
}

/// Worker side: serves the probe in `first` and any that follow on `stream`.
/// Returns the first other message, or `None` when the coordinator closed the
/// connection.
pub(super) fn serve_flow_probes(
    stream: &mut TcpStream,
    reader: &mut BufReader<TcpStream>,
    library: &Arc<NativeLibrary>,
    first: serde_json::Value,
) -> Result<Option<serde_json::Value>> {
    let mut value = first;
    loop {
        let start: VerbsHostFlowProbeStart =
            serde_json::from_value(value).context("decoding an RDMA flow probe start")?;
        serve_flow_probe(stream, reader, library, start)?;
        value = match read_control_value(reader) {
            Ok(next) => next,
            Err(error) if error.to_string().contains("control plane closed") => return Ok(None),
            Err(error) => return Err(error),
        };
        if !is_flow_probe_start(&value) {
            return Ok(Some(value));
        }
    }
}

fn serve_flow_probe(
    stream: &mut TcpStream,
    reader: &mut BufReader<TcpStream>,
    library: &Arc<NativeLibrary>,
    start: VerbsHostFlowProbeStart,
) -> Result<()> {
    anyhow::ensure!(
        (1..=bond::MAX_FLOW_LABEL).contains(&start.flow_label),
        "RDMA flow probe label {} is outside 1..=0xfffff",
        start.flow_label
    );
    anyhow::ensure!(
        (bond::MIN_PROBE_BYTES..=bond::MAX_PROBE_BYTES).contains(&start.probe_bytes),
        "RDMA flow probe of {} bytes is outside {}..={}",
        start.probe_bytes,
        bond::MIN_PROBE_BYTES,
        bond::MAX_PROBE_BYTES
    );
    let rdma_device = verbs_host_rdma_device_for_stream(stream)?;
    // Server role: the probe bytes are its send buffer, read remotely.
    let endpoint = NativeRdmaEndpoint::create_from_wire_bytes_on_device(
        Arc::clone(library),
        "server",
        PROBE_CONTROL_BYTES,
        start.probe_bytes,
        PROBE_CONTROL_BYTES,
        start.probe_bytes,
        next_local_psn("server"),
        rdma_device.as_ref(),
    )?;
    endpoint.connect_with_flow_label(&start.client_native_endpoint, start.flow_label)?;
    let (read_addr, read_rkey) =
        library.rdma_rc_endpoint_expose_send_read(endpoint.info.handle, start.probe_bytes)?;
    write_control(
        stream,
        &VerbsHostFlowProbeReady {
            message: FLOW_PROBE_READY.to_owned(),
            flow_label: start.flow_label,
            server_native_endpoint: endpoint.native_descriptor(),
            read_addr,
            read_rkey,
            read_bytes: start.probe_bytes,
        },
    )?;
    // The coordinator reads, then says done (or closes); only then does the
    // exposed buffer go away.
    match read_control_value(reader) {
        Ok(done) => anyhow::ensure!(
            control_message(&done)? == FLOW_PROBE_DONE,
            "RDMA flow probe expected {FLOW_PROBE_DONE}"
        ),
        Err(error) if error.to_string().contains("control plane closed") => {}
        Err(error) => return Err(error),
    }
    drop(endpoint);
    Ok(())
}

/// One measurement of where `label`'s traffic from `peer` arrives.
fn probe_once(
    library: &Arc<NativeLibrary>,
    peer: SocketAddr,
    label: u32,
    probe_bytes: usize,
    counters: &mut dyn PortCounters,
) -> Result<ProbeResult> {
    let mut stream = connect_control_stream(&peer.to_string(), PROBE_CONTROL_TIMEOUT)?;
    configure_control_stream(&stream, PROBE_CONTROL_TIMEOUT)?;
    let selection = verbs_host_rdma_device_for_stream(&stream)?;
    let endpoint = NativeRdmaEndpoint::create_from_wire_bytes_on_device(
        Arc::clone(library), "client", PROBE_CONTROL_BYTES, probe_bytes,
        PROBE_CONTROL_BYTES, probe_bytes, next_local_psn("client"), selection.as_ref(),
    )?;
    let mut reader = BufReader::new(stream.try_clone()?);
    write_control(
        &mut stream,
        &VerbsHostFlowProbeStart {
            message: FLOW_PROBE_START.to_owned(),
            flow_label: label,
            probe_bytes,
            client_native_endpoint: endpoint.native_descriptor(),
        },
    )?;
    let ready: VerbsHostFlowProbeReady = read_control(&mut reader)
        .map_err(|error| anyhow::Error::new(ProbeUnsupported).context(format!("{peer}: {error:#}")))?;
    anyhow::ensure!(
        ready.message == FLOW_PROBE_READY && ready.flow_label == label && ready.read_bytes >= probe_bytes,
        "{peer} answered an RDMA flow probe with {} label {} for {} bytes",
        ready.message,
        ready.flow_label,
        ready.read_bytes
    );
    endpoint.connect_with_flow_label(&ready.server_native_endpoint, label)?;
    wait_quiet(counters, probe_bytes)?;
    let before = counters.read()?;
    library.rdma_rc_endpoint_read_wait(
        endpoint.info.handle,
        0,
        probe_bytes,
        ready.read_addr,
        ready.read_rkey,
        PROBE_READ_TIMEOUT_MS,
    )?;
    let after = counters.read()?;
    let _ = write_control(&mut stream, &serde_json::json!({ "message": FLOW_PROBE_DONE }));
    Ok(bond::classify(&before, &after, probe_bytes as u64))
}

/// The worker does not serve flow probes (an older build): its flows keep label 0.
#[derive(Debug, thiserror::Error)]
#[error("worker does not serve RDMA flow probes")]
struct ProbeUnsupported;

/// Waits (up to ~0.6 s) until the members carry less than a sixteenth of a
/// probe per 2 ms, so the probe is measured alone; [`bond::classify`] still
/// rejects a probe that something else overlapped.
fn wait_quiet(counters: &mut dyn PortCounters, probe_bytes: usize) -> Result<()> {
    for _ in 0..25 {
        let before = counters.read()?;
        thread::sleep(Duration::from_millis(2));
        let after = counters.read()?;
        if bond::quiet(&before, &after, probe_bytes as u64) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}

struct Measure {
    bond: BondPorts,
    counters: Box<dyn PortCounters>,
    library: Arc<NativeLibrary>,
}

struct Balancer {
    mode: BondBalance,
    max_probes: u32,
    probe_bytes: usize,
    placement: Placement,
    /// Probe mode, once its bond and counters are known (`None`: fixed labels).
    measure: Option<Measure>,
    initialized: bool,
    /// Ranks whose worker refused a probe: label 0.
    unsupported: BTreeSet<IpAddr>,
}

fn env_number<T: std::str::FromStr>(name: &str, default: T) -> Result<T> {
    match env::var(name) {
        Ok(raw) => raw.trim().parse().map_err(|_| anyhow::anyhow!("{name}={raw:?} is not a number")),
        Err(_) => Ok(default),
    }
}

impl Balancer {
    fn from_env() -> Result<Self> {
        let mode = BondBalance::from_env()?;
        let max_probes = env_number(bond::BOND_PROBES_ENV, bond::DEFAULT_PROBES)?.clamp(1, 256);
        let probe_bytes = env_number(bond::BOND_PROBE_BYTES_ENV, bond::DEFAULT_PROBE_BYTES)?
            .clamp(bond::MIN_PROBE_BYTES, bond::MAX_PROBE_BYTES);
        Ok(Self {
            mode,
            max_probes,
            probe_bytes,
            placement: Placement::new(2),
            measure: None,
            initialized: false,
            unsupported: BTreeSet::new(),
        })
    }

    /// Probe mode's first use: find the bond behind the RDMA device this
    /// process's QPs open (as a probe endpoint reports it) and its counters.
    fn initialize(&mut self, peer: SocketAddr) {
        self.initialized = true;
        if self.mode != BondBalance::Probe {
            return;
        }
        match Self::find_measure(peer, self.probe_bytes) {
            Ok(Some(measure)) => {
                tracing::info!(bond = %measure.bond.bond, members = ?measure.bond.slaves, counter = RX_COUNTER,
                    max_probes = self.max_probes, probe_bytes = self.probe_bytes,
                    "RDMA bond balance: flows are placed by measured member");
                self.placement = Placement::new(measure.bond.slaves.len());
                self.measure = Some(measure);
            }
            Ok(None) => tracing::info!("RDMA bond balance: the RDMA port is not a bond; flow labels are fixed"),
            Err(error) => tracing::warn!(error = %format!("{error:#}"),
                "RDMA bond balance: cannot measure the bond; flow labels are fixed"),
        }
    }

    fn find_measure(peer: SocketAddr, probe_bytes: usize) -> Result<Option<Measure>> {
        let library = load_verbs_host_native_library()?;
        let stream = connect_control_stream(&peer.to_string(), PROBE_CONTROL_TIMEOUT)?;
        let selection = verbs_host_rdma_device_for_stream(&stream)?;
        let endpoint = NativeRdmaEndpoint::create_from_wire_bytes_on_device(
            Arc::clone(&library), "client", PROBE_CONTROL_BYTES, probe_bytes,
            PROBE_CONTROL_BYTES, probe_bytes, next_local_psn("client"), selection.as_ref(),
        )?;
        let device = c_char_array_to_string(&endpoint.info.device_name);
        let gid = c_char_array_to_string(&endpoint.info.gid_hex);
        let port = endpoint.info.port_num;
        drop(endpoint);
        let Some(bond) = bond::discover_bond(std::path::Path::new("/sys"), &device, port, &gid)? else {
            return Ok(None);
        };
        let counters = EthtoolCounters::open(&bond.slaves, RX_COUNTER)?;
        Ok(Some(Measure { bond, counters: Box::new(counters), library }))
    }

    fn acquire(&mut self, peer: SocketAddr, transport: u64) -> Result<Assignment> {
        if !self.initialized {
            self.initialize(peer);
        }
        let ip = peer.ip();
        let Some(measure) = self.measure.as_mut().filter(|_| !self.unsupported.contains(&ip)) else {
            if self.unsupported.contains(&ip) {
                return Ok(Assignment { peer: ip, transport, label: 0, port: None, probes: 0 });
            }
            return Ok(self.placement.fixed(ip, transport));
        };
        let (library, probe_bytes) = (Arc::clone(&measure.library), self.probe_bytes);
        let counters = measure.counters.as_mut();
        let deadline = Instant::now() + PLACEMENT_BUDGET;
        let result = self.placement.acquire(ip, transport, self.max_probes, |label| {
            for _ in 0..PROBE_ATTEMPTS {
                if Instant::now() >= deadline {
                    // Out of time (a fabric that never goes quiet): settle
                    // for what has been measured.
                    return Ok(ProbeResult::Inconclusive);
                }
                match probe_once(&library, peer, label, probe_bytes, counters)? {
                    ProbeResult::Inconclusive => continue,
                    placed => return Ok(placed),
                }
            }
            Ok(ProbeResult::Inconclusive)
        });
        match result {
            Ok(assignment) => Ok(assignment),
            Err(error) if error.downcast_ref::<ProbeUnsupported>().is_some() => {
                tracing::warn!(peer = %peer, error = %format!("{error:#}"),
                    "RDMA bond balance: worker does not serve flow probes; its flows keep the kernel's label");
                self.unsupported.insert(ip);
                Ok(Assignment { peer: ip, transport, label: 0, port: None, probes: 0 })
            }
            Err(error) => Err(error),
        }
    }

    fn member(&self, port: Option<usize>) -> String {
        match (port, &self.measure) {
            (Some(port), Some(measure)) => measure.bond.slaves.get(port).cloned().unwrap_or_else(|| port.to_string()),
            (Some(port), None) => port.to_string(),
            (None, _) => "unmeasured".to_owned(),
        }
    }

    /// "member:flows,member:flows" for one transport's measured flows.
    fn members(&self, load: &[u32]) -> String {
        load.iter()
            .enumerate()
            .map(|(port, flows)| format!("{}:{flows}", self.member(Some(port))))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Identifies each transport (one per GLM Flash lane, numbered in the order
/// they are built) so its flows are split among themselves.
static NEXT_TRANSPORT: AtomicU64 = AtomicU64::new(0);

fn balancer() -> Result<&'static Mutex<Balancer>> {
    static BALANCER: OnceLock<std::result::Result<Mutex<Balancer>, String>> = OnceLock::new();
    BALANCER
        .get_or_init(|| Balancer::from_env().map(Mutex::new).map_err(|error| format!("{error:#}")))
        .as_ref()
        .map_err(|error| anyhow::anyhow!("{error}"))
}

/// One rank's label for one transport; returns its share of the placement
/// when dropped.
pub(crate) struct FlowSlot {
    assignment: Assignment,
}

impl FlowSlot {
    fn acquire(peer: SocketAddr, transport: u64) -> Result<Self> {
        let mut balancer = balancer()?.lock().map_err(|_| anyhow::anyhow!("RDMA flow placement lock poisoned"))?;
        let assignment = balancer.acquire(peer, transport)?;
        let sport = bond::flow_label_to_udp_sport(assignment.label);
        let member = balancer.member(assignment.port);
        if balancer.measure.is_some() && assignment.label != 0 && assignment.port.is_none() {
            tracing::warn!(peer = %peer, transport, label = assignment.label, sport = %format_args!("{sport:#06x}"),
                probes = assignment.probes, "RDMA flow label placed unmeasured: every probe was inconclusive");
        } else {
            let lane = balancer.members(&balancer.placement.transport_load(transport));
            tracing::info!(peer = %peer, transport, label = assignment.label, sport = %format_args!("{sport:#06x}"),
                member = %member, probes = assignment.probes, mode = balancer.mode.name(),
                transport_members = %lane, "RDMA flow label");
        }
        Ok(Self { assignment })
    }

    pub(crate) fn label(&self) -> u32 {
        self.assignment.label
    }
}

impl Drop for FlowSlot {
    fn drop(&mut self) {
        if let Ok(balancer) = balancer() {
            if let Ok(mut balancer) = balancer.lock() {
                balancer.placement.release(&self.assignment);
            }
        }
    }
}

/// The flow labels of one transport's ranks: all 0 unless
/// `CUTEAFD_RDMA_BOND_BALANCE` is set.
pub(crate) struct FlowSlots {
    enabled: bool,
    transport: u64,
    slots: Vec<Option<FlowSlot>>,
}

impl FlowSlots {
    /// Places every rank's flow now, while nothing else is in flight; a rank
    /// that cannot be placed yet (its worker not listening) is placed at its
    /// first connection instead.
    pub(crate) fn prepare(peers: &[SocketAddr]) -> Self {
        let enabled = match BondBalance::from_env() {
            Ok(mode) => mode != BondBalance::Off,
            Err(error) => {
                tracing::error!(error = %format!("{error:#}"), "RDMA bond balance setting is invalid");
                true
            }
        };
        let transport = NEXT_TRANSPORT.fetch_add(1, Ordering::Relaxed);
        let mut slots = Self { enabled, transport, slots: peers.iter().map(|_| None).collect() };
        if enabled {
            for (rank, peer) in peers.iter().enumerate() {
                match FlowSlot::acquire(*peer, transport) {
                    Ok(slot) => slots.slots[rank] = Some(slot),
                    Err(error) => tracing::warn!(peer = %peer, transport, error = %format!("{error:#}"),
                        "RDMA flow label deferred to the first connection"),
                }
            }
            slots.log_placement();
        }
        slots
    }

    /// One line per transport: how its flows sit on the bond's members.
    fn log_placement(&self) {
        let Ok(balancer) = balancer() else { return };
        let Ok(balancer) = balancer.lock() else { return };
        if balancer.measure.is_none() {
            return;
        }
        let load = balancer.placement.transport_load(self.transport);
        let placed = self.slots.iter().filter(|slot| slot.as_ref().is_some_and(|s| s.assignment.port.is_some())).count();
        let even = load.iter().max().zip(load.iter().min()).is_some_and(|(max, min)| max - min <= 1);
        if placed == self.slots.len() && even {
            tracing::info!(transport = self.transport, members = %balancer.members(&load), "RDMA transport flows split");
        } else {
            tracing::warn!(transport = self.transport, members = %balancer.members(&load), placed,
                flows = self.slots.len(), "RDMA transport flows not evenly split");
        }
    }

    /// The label rank `rank`'s next session connects with (0 when off). An
    /// invalid setting is an error; a rank that cannot be placed now connects
    /// with the kernel's label (with a warning) and is placed again at its
    /// next connection.
    pub(crate) fn label(&mut self, rank: usize, peer: SocketAddr) -> Result<u32> {
        if !self.enabled {
            return Ok(0);
        }
        let slot = self.slots.get_mut(rank).context("RDMA flow slot rank out of range")?;
        if slot.is_none() {
            balancer()?;
            match FlowSlot::acquire(peer, self.transport) {
                Ok(placed) => *slot = Some(placed),
                Err(error) => {
                    tracing::warn!(peer = %peer, error = %format!("{error:#}"),
                        "RDMA flow label unavailable; this connection keeps the kernel's label");
                    return Ok(0);
                }
            }
        }
        Ok(slot.as_ref().map_or(0, FlowSlot::label))
    }

    /// Places every rank not placed yet, before any request goes out.
    pub(crate) fn ensure_all(&mut self, peers: &[SocketAddr]) -> Result<()> {
        if self.enabled {
            for (rank, peer) in peers.iter().enumerate() {
                self.label(rank, *peer)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint() -> VerbsHostNativeEndpointDescriptor {
        VerbsHostNativeEndpointDescriptor {
            port_num: 1,
            qp_num: 0x1234,
            psn: 7,
            lid: 0,
            active_mtu: 5,
            gid_hex: "00000000000000000000ffffc0000203".into(),
            send_frame_bytes: 4096,
            recv_frame_bytes: 4 << 20,
            send_registered_span_bytes: 4096,
            recv_registered_span_bytes: 4 << 20,
            max_send_wr: 8,
            max_recv_wr: 8,
            max_sge: 1,
            device_name: "mlx5_0".into(),
            status: "ok".into(),
        }
    }

    #[test]
    fn probe_messages_round_trip_and_are_recognised() {
        let start = VerbsHostFlowProbeStart {
            message: FLOW_PROBE_START.into(),
            flow_label: 3,
            probe_bytes: 4 << 20,
            client_native_endpoint: endpoint(),
        };
        let value = serde_json::to_value(&start).unwrap();
        assert!(is_flow_probe_start(&value));
        let back: VerbsHostFlowProbeStart = serde_json::from_value(value).unwrap();
        assert_eq!((back.flow_label, back.probe_bytes), (3, 4 << 20));
        assert!(!is_flow_probe_start(&serde_json::json!({ "message": "protocol_v2_persistent_start" })));
        assert!(!is_flow_probe_start(&serde_json::json!({ "flow_label": 3 })));
        let ready = VerbsHostFlowProbeReady {
            message: FLOW_PROBE_READY.into(),
            flow_label: 3,
            server_native_endpoint: endpoint(),
            read_addr: 0x7f00_0000_1000,
            read_rkey: 0x55,
            read_bytes: 4 << 20,
        };
        let back: VerbsHostFlowProbeReady = serde_json::from_str(&serde_json::to_string(&ready).unwrap()).unwrap();
        assert_eq!((back.read_addr, back.read_rkey), (0x7f00_0000_1000, 0x55));
    }

    #[test]
    fn a_probe_start_is_not_a_session_start() {
        // An older worker reads a probe as a persistent start and must refuse
        // it (closing the connection), which tells the coordinator to keep label 0.
        let value = serde_json::to_value(VerbsHostFlowProbeStart {
            message: FLOW_PROBE_START.into(),
            flow_label: 1,
            probe_bytes: 4 << 20,
            client_native_endpoint: endpoint(),
        })
        .unwrap();
        assert!(serde_json::from_value::<VerbsHostProtocolV2PersistentStart>(value).is_err());
    }

    fn session_start(flow_label: u32) -> VerbsHostProtocolV2PersistentStart {
        let native = endpoint();
        VerbsHostProtocolV2PersistentStart {
            message: "protocol_v2_persistent_start".into(),
            execution_lane: 0,
            request_capacity_wire_bytes: 8 << 20,
            response_capacity_wire_bytes: 8 << 20,
            request_registered_span_bytes: 64 << 20,
            response_registered_span_bytes: 64 << 20,
            ring_depth: 8,
            request_slot_stride_bytes: 8 << 20,
            response_slot_stride_bytes: 8 << 20,
            client_endpoint: VerbsHostRcEndpointDescriptor {
                role: "client".into(),
                host: "coordinator".into(),
                port_num: native.port_num,
                qp_num: native.qp_num,
                psn: native.psn,
                gid_hex: native.gid_hex.clone(),
                send_frame_bytes: 8 << 20,
                recv_frame_bytes: 8 << 20,
                send_registered_span_bytes: 64 << 20,
                recv_registered_span_bytes: 64 << 20,
                max_send_wr: 8,
                max_recv_wr: 8,
                max_sge: 2,
            },
            client_native_endpoint: native,
            write_target: None,
            flow_label,
            worker_selection_digest: None,
        }
    }

    #[test]
    fn label_zero_keeps_the_session_handshake_byte_identical() {
        // Off: the start and ready messages carry no new field, so either end
        // may be an older build.
        let start = serde_json::to_value(session_start(0)).unwrap();
        assert!(start.get("flow_label").is_none());
        let back: VerbsHostProtocolV2PersistentStart = serde_json::from_value(start).unwrap();
        assert_eq!(back.flow_label, 0);
        // On: the label travels, and an older worker ignores it.
        let start = serde_json::to_value(session_start(7)).unwrap();
        assert_eq!(start["flow_label"], 7);
        assert_eq!(serde_json::from_value::<VerbsHostProtocolV2PersistentStart>(start).unwrap().flow_label, 7);
        let ready = |flow_label| VerbsHostProtocolV2PersistentReady {
            message: "protocol_v2_persistent_ready".into(),
            server_endpoint: session_start(0).client_endpoint,
            server_native_endpoint: endpoint(),
            flow_label,
        };
        let old = serde_json::to_value(ready(0)).unwrap();
        assert!(old.get("flow_label").is_none());
        assert_eq!(serde_json::from_value::<VerbsHostProtocolV2PersistentReady>(old).unwrap().flow_label, 0);
        let echoed = serde_json::to_value(ready(7)).unwrap();
        assert_eq!(serde_json::from_value::<VerbsHostProtocolV2PersistentReady>(echoed).unwrap().flow_label, 7);
    }

    #[test]
    fn a_transport_reports_its_members_compactly() {
        let balancer = Balancer {
            mode: BondBalance::Probe,
            max_probes: bond::DEFAULT_PROBES,
            probe_bytes: bond::DEFAULT_PROBE_BYTES,
            placement: Placement::new(2),
            measure: None,
            initialized: true,
            unsupported: BTreeSet::new(),
        };
        // Without a measured bond the members are numbered; tools that read the
        // log grep "members=<member>:<flows>,...".
        assert_eq!(balancer.members(&[2, 2]), "0:2,1:2");
        assert_eq!(balancer.member(None), "unmeasured");
    }

    #[test]
    fn off_slots_never_touch_the_placement() {
        // The process environment does not set the switch in tests.
        if env::var(bond::BOND_BALANCE_ENV).is_ok() {
            return;
        }
        let peers: Vec<SocketAddr> = vec!["192.0.2.1:9100".parse().unwrap(), "192.0.2.2:9100".parse().unwrap()];
        let mut slots = FlowSlots::prepare(&peers);
        assert!(!slots.enabled);
        assert_eq!(slots.label(1, peers[1]).unwrap(), 0);
        slots.ensure_all(&peers).unwrap();
        assert!(slots.slots.iter().all(Option::is_none));
    }
}
