//! Boot-time RoCE fabric discovery.
//!
//! Reads the RDMA ports from sysfs (state, link rate, RoCE v2 GIDs and the
//! netdev behind each), the PCIe link of every NIC function, and the IPv4
//! subnets from iproute2. The report is logged at startup and drives the rail
//! recommendation: a Spark reaches its ConnectX through PCIe Gen5 x4 links, so
//! one rail tops out near the PCIe rate, and a second rail pays off only on its
//! own subnet and when the switch links are not slower than PCIe (at 100G the
//! second rail head-of-line blocks behind 200+ Gb/s of PCIe ingress).
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::net::Ipv4Addr;
use std::path::Path;

const SYSFS_INFINIBAND: &str = "/sys/class/infiniband";

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FabricReport {
    pub ports: Vec<RdmaPort>,
    pub rails: RailPlan,
    /// GRH traffic byte applied to all native RC QPs by the shared startup key.
    pub rdma_traffic_class: u8,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RdmaPort {
    pub device: String,
    pub port: u32,
    pub active: bool,
    pub link_layer: String,
    pub link_gbps: f64,
    pub netdev: Option<String>,
    pub mtu: Option<u32>,
    /// RoCE v2 IPv4 GIDs: (index, address).
    pub roce_v2: Vec<(u32, Ipv4Addr)>,
    /// IPv4 networks on the netdev as (network, prefix).
    pub subnets: Vec<(Ipv4Addr, u8)>,
    pub pci: Option<PciLink>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PciLink {
    pub address: String,
    pub gts: f64,
    pub width: u32,
    pub numa_node: Option<i32>,
}

impl PciLink {
    /// Usable PCIe data rate after 128b/130b encoding (Gen3 and newer).
    pub fn gbps(&self) -> f64 {
        self.gts * f64::from(self.width) * 128.0 / 130.0
    }
}

impl RdmaPort {
    /// The narrower of the switch link and the NIC's PCIe link.
    pub fn effective_gbps(&self) -> f64 {
        self.pci
            .as_ref()
            .map_or(self.link_gbps, |pci| self.link_gbps.min(pci.gbps()))
    }
    fn usable(&self) -> bool {
        self.active && self.link_layer == "Ethernet" && !self.roce_v2.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RailPlan {
    /// Usable RoCE v2 rails, one per distinct subnet, fastest first.
    pub rails: Vec<Rail>,
    /// Recommended rail count for expert traffic.
    pub use_rails: usize,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Rail {
    pub device: String,
    pub netdev: Option<String>,
    pub address: Ipv4Addr,
    pub subnet: (Ipv4Addr, u8),
    pub link_gbps: f64,
    pub effective_gbps: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdmaSelection {
    pub device: String,
    pub port: u32,
    pub gid_index: u32,
}

#[derive(Debug, thiserror::Error)]
#[error("no active RoCE v2 IPv4 GID matches control address {address}{device_hint}")]
pub struct RdmaAddressError {
    pub address: std::net::IpAddr,
    device_hint: String,
}

/// Select an exact source GID, not the first device or the first IPv4 rail.
pub fn select_rdma_address(ports: &[RdmaPort], address: std::net::IpAddr,
                           device_override: Option<&str>) -> std::result::Result<RdmaSelection, RdmaAddressError> {
    for port in ports.iter().filter(|p| p.usable()) {
        if device_override.is_some_and(|device| device != port.device) { continue; }
        for &(gid_index, ipv4) in &port.roce_v2 {
            if address == std::net::IpAddr::V4(ipv4) {
                return Ok(RdmaSelection { device: port.device.clone(), port: port.port, gid_index });
            }
        }
    }
    Err(RdmaAddressError { address,
        device_hint: device_override.map(|d| format!(" on overridden device {d}")).unwrap_or_default() })
}

/// The device, port and source GID a QP on `stream`'s link binds: the RoCE v2
/// GID of the control connection's own local address.
pub fn select_rdma_for_stream(ports: &[RdmaPort], stream: &std::net::TcpStream) -> Result<RdmaSelection> {
    let local = stream.local_addr().context("reading control connection local address")?;
    Ok(select_rdma_address(ports, local.ip(), None)?)
}

/// The local address the kernel routes `peer` from: what a control connection
/// to `peer` would bind, read from a connected UDP socket (nothing is sent).
pub fn route_source(peer: std::net::SocketAddr) -> std::io::Result<std::net::IpAddr> {
    let unspecified: std::net::SocketAddr = match peer {
        std::net::SocketAddr::V4(_) => (std::net::Ipv4Addr::UNSPECIFIED, 0).into(),
        std::net::SocketAddr::V6(_) => (std::net::Ipv6Addr::UNSPECIFIED, 0).into(),
    };
    let socket = std::net::UdpSocket::bind(unspecified)?;
    socket.connect(peer)?;
    Ok(socket.local_addr()?.ip())
}

/// The device, port and source GID that reach `peer`: the RoCE v2 GID of the
/// address the kernel routes `peer` from. For QPs without a control
/// connection of their own (the GPU landing loopback, bench probes).
pub fn select_rdma_route(ports: &[RdmaPort], peer: std::net::SocketAddr) -> Result<RdmaSelection> {
    let address = route_source(peer).with_context(|| format!("routing to RDMA peer {peer}"))?;
    Ok(select_rdma_address(ports, address, None).with_context(|| format!("RDMA source for peer {peer}"))?)
}

/// Discovers this host's RDMA ports. A host without RDMA returns an empty report.
pub fn discover() -> Result<FabricReport> {
    let root = Path::new(SYSFS_INFINIBAND);
    let mut ports = Vec::new();
    if root.is_dir() {
        let addresses = ipv4_addresses().unwrap_or_default();
        let mut devices: Vec<_> = fs::read_dir(root)?.filter_map(|entry| entry.ok()).collect();
        devices.sort_by_key(|entry| entry.file_name());
        for device in devices {
            ports.extend(read_device(&device.path(), &addresses)?);
        }
    }
    let rails = plan_rails(&ports);
    Ok(FabricReport { ports, rails, rdma_traffic_class: RDMA_TRAFFIC_CLASS })
}

/// GRH traffic class the native layer sets on every RC QP (`rdma_traffic_class.h`): DSCP 26.
pub const RDMA_TRAFFIC_CLASS: u8 = 26 << 2;

fn read_device(path: &Path, addresses: &[(String, Ipv4Addr, u8)]) -> Result<Vec<RdmaPort>> {
    let device = path.file_name().context("RDMA device name")?.to_string_lossy().into_owned();
    let pci = read_pci(&path.join("device"));
    let mut ports = Vec::new();
    let Ok(entries) = fs::read_dir(path.join("ports")) else {
        return Ok(ports);
    };
    for entry in entries.filter_map(|entry| entry.ok()) {
        let port_path = entry.path();
        let Ok(port) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        let read = |name: &str| fs::read_to_string(port_path.join(name)).map(|s| s.trim().to_string());
        let state = read("state").unwrap_or_default();
        let rate = read("rate").unwrap_or_default();
        let mut roce_v2 = Vec::new();
        let mut netdev = None;
        if let Ok(types) = fs::read_dir(port_path.join("gid_attrs/types")) {
            for gid_type in types.filter_map(|entry| entry.ok()) {
                let Ok(index) = gid_type.file_name().to_string_lossy().parse::<u32>() else {
                    continue;
                };
                // Unpopulated GID slots fail to read.
                let Ok(kind) = fs::read_to_string(gid_type.path()) else { continue };
                if kind.trim() != "RoCE v2" {
                    continue;
                }
                let Some(address) = read(&format!("gids/{index}")).ok().and_then(|gid| gid_ipv4(&gid)) else {
                    continue;
                };
                netdev = netdev.or_else(|| read(&format!("gid_attrs/ndevs/{index}")).ok());
                roce_v2.push((index, address));
            }
        }
        roce_v2.sort();
        let mtu = netdev
            .as_ref()
            .and_then(|name| fs::read_to_string(format!("/sys/class/net/{name}/mtu")).ok())
            .and_then(|mtu| mtu.trim().parse().ok());
        let subnets = netdev.as_ref().map_or_else(Vec::new, |name| {
            let mut subnets: Vec<_> = addresses
                .iter()
                .filter(|(dev, _, _)| dev == name)
                .map(|(_, address, prefix)| (network(*address, *prefix), *prefix))
                .collect();
            subnets.sort();
            subnets.dedup();
            subnets
        });
        ports.push(RdmaPort {
            device: device.clone(),
            port,
            active: state.ends_with("ACTIVE"),
            link_layer: read("link_layer").unwrap_or_default(),
            link_gbps: parse_rate(&rate).unwrap_or(0.0),
            netdev,
            mtu,
            roce_v2,
            subnets,
            pci: pci.clone(),
        });
    }
    ports.sort_by_key(|port| port.port);
    Ok(ports)
}

fn read_pci(path: &Path) -> Option<PciLink> {
    let read = |name: &str| fs::read_to_string(path.join(name)).ok().map(|s| s.trim().to_string());
    let address = fs::canonicalize(path).ok()?.file_name()?.to_string_lossy().into_owned();
    let gts = read("current_link_speed")?.split_whitespace().next()?.parse().ok()?;
    let width = read("current_link_width")?.parse().ok()?;
    let numa_node = read("numa_node").and_then(|node| node.parse().ok()).filter(|node| *node >= 0);
    Some(PciLink { address, gts, width, numa_node })
}

/// "200 Gb/sec (4X HDR)" -> 200.0
fn parse_rate(rate: &str) -> Option<f64> {
    rate.split_whitespace().next()?.parse().ok()
}

/// IPv4-mapped GID "0000:...:ffff:0a37:0001" -> 10.55.0.1
fn gid_ipv4(gid: &str) -> Option<Ipv4Addr> {
    let groups: Vec<u16> = gid.split(':').map(|g| u16::from_str_radix(g, 16)).collect::<Result<_, _>>().ok()?;
    if groups.len() != 8 || groups[..5].iter().any(|g| *g != 0) || groups[5] != 0xffff {
        return None;
    }
    let [a, b] = groups[6].to_be_bytes();
    let [c, d] = groups[7].to_be_bytes();
    Some(Ipv4Addr::new(a, b, c, d))
}

fn network(address: Ipv4Addr, prefix: u8) -> Ipv4Addr {
    let mask = u32::MAX.checked_shl(32 - u32::from(prefix.min(32))).unwrap_or(0);
    Ipv4Addr::from(u32::from(address) & mask)
}

/// (netdev, address, prefix) for every IPv4 address, from `ip -j -4 addr`.
fn ipv4_addresses() -> Result<Vec<(String, Ipv4Addr, u8)>> {
    let output = std::process::Command::new("ip").args(["-j", "-4", "addr"]).output()?;
    anyhow::ensure!(output.status.success(), "ip -j -4 addr failed");
    let links: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout)?;
    let mut addresses = Vec::new();
    for link in links {
        let Some(name) = link["ifname"].as_str() else { continue };
        for info in link["addr_info"].as_array().into_iter().flatten() {
            let (Some(local), Some(prefix)) = (info["local"].as_str(), info["prefixlen"].as_u64()) else {
                continue;
            };
            if let (Ok(address), Ok(prefix)) = (local.parse(), u8::try_from(prefix)) {
                addresses.push((name.to_string(), address, prefix));
            }
        }
    }
    Ok(addresses)
}

/// Picks the rails for expert traffic from the discovered ports.
pub fn plan_rails(ports: &[RdmaPort]) -> RailPlan {
    let mut rails = Vec::new();
    let mut seen = BTreeSet::new();
    for port in ports.iter().filter(|port| port.usable()) {
        for &(_, address) in &port.roce_v2 {
            let Some(&subnet) = port.subnets.iter().find(|(net, prefix)| network(address, *prefix) == *net) else {
                continue;
            };
            if seen.insert(subnet) {
                rails.push(Rail {
                    device: port.device.clone(),
                    netdev: port.netdev.clone(),
                    address,
                    subnet,
                    link_gbps: port.link_gbps,
                    effective_gbps: port.effective_gbps(),
                });
            }
        }
    }
    rails.sort_by(|a, b| b.effective_gbps.total_cmp(&a.effective_gbps).then(a.subnet.cmp(&b.subnet)));
    let devices: BTreeSet<_> = rails.iter().map(|rail| rail.device.as_str()).collect();
    let pcie_bound = rails.iter().all(|rail| rail.effective_gbps < rail.link_gbps);
    let (use_rails, reason) = match rails.len() {
        0 => (0, "no active RoCE v2 port with an IPv4 subnet".to_string()),
        1 => (1, "one RoCE v2 subnet".to_string()),
        _ if devices.len() == 1 => (
            rails.len(),
            format!("{} subnets share one {:.0} Gb/s port; each rail is a subnet of the same link", rails.len(), rails[0].link_gbps),
        ),
        _ if pcie_bound => (
            rails.len(),
            format!(
                "{} rails on separate subnets, each PCIe-bound at {:.0} Gb/s below a {:.0} Gb/s link",
                rails.len(),
                rails[0].effective_gbps,
                rails[0].link_gbps
            ),
        ),
        _ => (
            1,
            format!(
                "switch links ({:.0} Gb/s) are slower than PCIe; a second rail head-of-line blocks, so use one",
                rails[0].link_gbps
            ),
        ),
    };
    RailPlan { rails, use_rails, reason }
}

impl FabricReport {
    /// One line per rail for startup logs.
    pub fn summary(&self) -> String {
        if self.rails.rails.is_empty() {
            return format!("fabric: rdma_tc={} dscp={} {}", self.rdma_traffic_class, self.rdma_traffic_class >> 2, self.rails.reason);
        }
        let rails: Vec<_> = self
            .rails
            .rails
            .iter()
            .map(|rail| {
                format!(
                    "{}@{}/{} {:.0}G(eff {:.0}G)",
                    rail.device, rail.address, rail.subnet.1, rail.link_gbps, rail.effective_gbps
                )
            })
            .collect();
        format!("fabric: rdma_tc={} dscp={} use {} rail(s) of [{}]: {}",
            self.rdma_traffic_class, self.rdma_traffic_class >> 2, self.rails.use_rails, rails.join(", "), self.rails.reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdma_marking_is_dscp_26_and_logged() {
        assert_eq!(RDMA_TRAFFIC_CLASS, 104);
        let report = FabricReport { ports: vec![], rails: plan_rails(&[]), rdma_traffic_class: RDMA_TRAFFIC_CLASS };
        assert!(report.summary().contains("rdma_tc=104 dscp=26"));
    }


    fn port(device: &str, gbps: f64, address: [u8; 4], pcie_width: u32) -> RdmaPort {
        let address = Ipv4Addr::from(address);
        RdmaPort {
            device: device.into(),
            port: 1,
            active: true,
            link_layer: "Ethernet".into(),
            link_gbps: gbps,
            netdev: Some(format!("{device}-net")),
            mtu: Some(9000),
            roce_v2: vec![(3, address)],
            subnets: vec![(network(address, 24), 24)],
            pci: Some(PciLink { address: "0000:01:00.0".into(), gts: 32.0, width: pcie_width, numa_node: None }),
        }
    }

    #[test]
    fn selects_exact_address_device_port_and_gid_from_sysfs() {
        let root = std::env::temp_dir().join(format!("cuteafd-fabric-{}", std::process::id()));
        let device = root.join("mlx5_fixture");
        let port_path = device.join("ports/2");
        for directory in ["gid_attrs/types", "gid_attrs/ndevs", "gids"] {
            fs::create_dir_all(port_path.join(directory)).unwrap();
        }
        for (name, value) in [("state", "4: ACTIVE"), ("rate", "200 Gb/sec"),
            ("link_layer", "Ethernet"), ("gid_attrs/types/3", "RoCE v2"),
            ("gid_attrs/types/7", "RoCE v2"), ("gid_attrs/types/9", "RoCE v1"),
            ("gid_attrs/ndevs/3", "bond0"), ("gid_attrs/ndevs/7", "bond0"),
            ("gids/3", "0000:0000:0000:0000:0000:ffff:0a37:0016"),
            ("gids/7", "0000:0000:0000:0000:0000:ffff:0a37:0116"),
            ("gids/9", "0000:0000:0000:0000:0000:ffff:0a37:0216")] {
            fs::write(port_path.join(name), value).unwrap();
        }
        let ports = read_device(&device, &[]).unwrap();
        let address = "10.55.1.22".parse().unwrap();
        assert_eq!(select_rdma_address(&ports, address, None).unwrap(),
            RdmaSelection { device: "mlx5_fixture".into(), port: 2, gid_index: 7 });
        assert!(select_rdma_address(&ports, address, Some("mlx5_other")).is_err());
        assert!(select_rdma_address(&ports, "10.55.2.22".parse().unwrap(), None).is_err());
        let mut inactive = ports;
        inactive[0].active = false;
        assert!(select_rdma_address(&inactive, address, None).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    /// A coordinator whose RoCE rail shares its port with a DHCP management
    /// LAN (#898): the management GID registered first and holds the lower
    /// index. Selection follows the address that reaches the peer, never
    /// table order, whichever way the two GIDs are ordered.
    #[test]
    fn management_gid_at_lower_index_is_never_chosen_for_the_rail() {
        let root = std::env::temp_dir().join(format!("cuteafd-fabric-mgmt-{}", std::process::id()));
        let device = root.join("mlx5_0");
        let port_path = device.join("ports/1");
        for directory in ["gid_attrs/types", "gid_attrs/ndevs", "gids"] {
            fs::create_dir_all(port_path.join(directory)).unwrap();
        }
        // Index 0/1: link-local RoCE v1/v2; 2: management 192.168.1.50 (v1);
        // 3: management (v2); 6/7: rail 10.55.0.22 (v1/v2).
        for (name, value) in [("state", "4: ACTIVE"), ("rate", "400 Gb/sec"), ("link_layer", "Ethernet"),
            ("gid_attrs/types/0", "IB/RoCE v1"), ("gids/0", "fe80:0000:0000:0000:5a95:69ff:fe0c:2c60"),
            ("gid_attrs/types/1", "RoCE v2"), ("gids/1", "fe80:0000:0000:0000:5a95:69ff:fe0c:2c60"),
            ("gid_attrs/types/2", "IB/RoCE v1"), ("gids/2", "0000:0000:0000:0000:0000:ffff:c0a8:0132"),
            ("gid_attrs/types/3", "RoCE v2"), ("gids/3", "0000:0000:0000:0000:0000:ffff:c0a8:0132"),
            ("gid_attrs/types/6", "IB/RoCE v1"), ("gids/6", "0000:0000:0000:0000:0000:ffff:0a37:0016"),
            ("gid_attrs/types/7", "RoCE v2"), ("gids/7", "0000:0000:0000:0000:0000:ffff:0a37:0016"),
            ("gid_attrs/ndevs/3", "eno1"), ("gid_attrs/ndevs/7", "eno1")] {
            fs::write(port_path.join(name), value).unwrap();
        }
        let ports = read_device(&device, &[]).unwrap();
        assert_eq!(ports[0].roce_v2, vec![(3, Ipv4Addr::new(192, 168, 1, 50)), (7, Ipv4Addr::new(10, 55, 0, 22))]);
        let rail = RdmaSelection { device: "mlx5_0".into(), port: 1, gid_index: 7 };
        assert_eq!(select_rdma_address(&ports, "10.55.0.22".parse().unwrap(), None).unwrap(), rail);
        assert_eq!(select_rdma_address(&ports, "192.168.1.50".parse().unwrap(), None).unwrap().gid_index, 3);
        // An address the port does not carry is refused, not mapped to index 3.
        assert!(select_rdma_address(&ports, "10.55.1.22".parse().unwrap(), None).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn route_source_is_the_address_a_connection_to_the_peer_binds() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let peer = listener.local_addr().unwrap();
        let stream = std::net::TcpStream::connect(peer).unwrap();
        assert_eq!(route_source(peer).unwrap(), stream.local_addr().unwrap().ip());
        // Loopback carries no RoCE GID: refused rather than any table entry.
        let mut fixture = port("mlx5_0", 400.0, [192, 168, 1, 50], 16);
        fixture.roce_v2.push((7, Ipv4Addr::new(10, 55, 0, 22)));
        assert!(select_rdma_route(&[fixture], peer).is_err());
    }

    #[test]
    fn parses_sysfs_values() {
        assert_eq!(parse_rate("200 Gb/sec (4X HDR)"), Some(200.0));
        assert_eq!(
            gid_ipv4("0000:0000:0000:0000:0000:ffff:0a37:0101"),
            Some(Ipv4Addr::new(10, 55, 1, 1))
        );
        assert_eq!(gid_ipv4("fe80:0000:0000:0000:5a95:69ff:fe0c:2c60"), None);
        assert_eq!(network(Ipv4Addr::new(10, 55, 1, 22), 24), Ipv4Addr::new(10, 55, 1, 0));
    }

    #[test]
    fn spark_200g_dual_rail_is_pcie_bound_and_uses_both() {
        let plan = plan_rails(&[port("rocep1", 200.0, [10, 55, 0, 1], 4), port("roceP2", 200.0, [10, 55, 1, 1], 4)]);
        assert_eq!((plan.rails.len(), plan.use_rails), (2, 2), "{}", plan.reason);
        assert!(plan.rails[0].effective_gbps < 130.0);
    }

    #[test]
    fn spark_100g_dual_rail_falls_back_to_one() {
        let plan = plan_rails(&[port("rocep1", 100.0, [10, 55, 0, 1], 4), port("roceP2", 100.0, [10, 55, 1, 1], 4)]);
        assert_eq!(plan.use_rails, 1, "{}", plan.reason);
    }

    #[test]
    fn one_port_with_two_subnets_counts_both_rails() {
        let mut raptor = port("mlx5_0", 400.0, [10, 55, 0, 22], 16);
        raptor.roce_v2.push((5, Ipv4Addr::new(10, 55, 1, 22)));
        raptor.subnets.push((Ipv4Addr::new(10, 55, 1, 0), 24));
        let plan = plan_rails(&[raptor]);
        assert_eq!((plan.rails.len(), plan.use_rails), (2, 2), "{}", plan.reason);
    }

    #[test]
    fn same_subnet_twice_is_one_rail() {
        let plan = plan_rails(&[port("a", 200.0, [10, 55, 0, 1], 4), port("b", 200.0, [10, 55, 0, 2], 4)]);
        assert_eq!((plan.rails.len(), plan.use_rails), (1, 1));
    }
}
