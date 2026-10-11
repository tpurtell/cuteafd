//! Startup-only, coordinator-authoritative Spark weight selection.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::{BufRead, BufReader, Read, Write}, net::{SocketAddr, TcpStream},
    path::Path, sync::{Mutex, OnceLock}, time::Duration};

const CONTROL_LIMIT: usize = 64 * 1024;
const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerIdentity {
    checkpoint: String,
    geometry: [u32; 5],
    world: u32,
    topology: Option<[u8; 2]>,
}

impl WorkerIdentity {
    /// Paths may differ between hosts; the checkpoint's config and tensor index
    /// bytes, not a local pathname or a separately computed plan, identify it.
    pub fn from_snapshot(snapshot: &Path, geometry: cuteafd_core::ExpertGeometry,
        world: usize, topology: Option<crate::expert::SparkTopology>) -> Result<Self> {
        ensure!(matches!(world, 1 | 2 | 3 | 4 | 6), "unsupported worker world");
        if let Some(topology) = topology {
            ensure!(topology.world_size() == world, "worker topology/world mismatch");
        }
        let mut hash = Sha256::new();
        for name in ["config.json", "model.safetensors.index.json"] {
            let mut file = std::fs::File::open(snapshot.join(name))
                .with_context(|| format!("worker checkpoint identity {name}"))?;
            let length = file.metadata()?.len();
            ensure!(length <= 64 * 1024 * 1024, "worker identity file {name} is too large");
            hash.update(length.to_le_bytes());
            let mut buffer = [0u8; 64 * 1024];
            let mut read_bytes = 0u64;
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 { break; }
                read_bytes += count as u64;
                ensure!(read_bytes <= length, "worker identity file {name} changed while reading");
                hash.update(&buffer[..count]);
            }
            ensure!(read_bytes == length, "worker identity file {name} changed while reading");
        }
        Ok(Self { checkpoint: format!("{:x}", hash.finalize()),
            geometry: [geometry.hidden, geometry.experts, geometry.topk, geometry.intermediate, geometry.layers],
            world: u32::try_from(world)?, topology: topology.map(|topology| [topology.tp(), topology.ep()]) })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerRange {
    pub first: u32,
    /// Exclusive absolute layer bound.
    pub end: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSelection {
    version: u32,
    identity: WorkerIdentity,
    ranges: Vec<LayerRange>,
}

impl WorkerSelection {
    pub fn new(identity: WorkerIdentity, layers: &[usize]) -> Result<Self> {
        ensure!(layers.windows(2).all(|pair| pair[0] < pair[1]), "worker layers must be sorted and unique");
        let mut ranges = Vec::<LayerRange>::new();
        for &layer in layers {
            let layer = u32::try_from(layer)?;
            if let Some(range) = ranges.last_mut().filter(|range| range.end == layer) { range.end = layer.checked_add(1).context("worker layer overflow")?; }
            else { ranges.push(LayerRange { first: layer, end: layer.checked_add(1).context("worker layer overflow")? }); }
        }
        let selection = Self { version: VERSION, identity, ranges };
        selection.validate()?;
        Ok(selection)
    }

    pub fn ranges(&self) -> &[LayerRange] { &self.ranges }
    pub fn identity(&self) -> &WorkerIdentity { &self.identity }
    pub fn digest(&self) -> Result<String> { Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?))) }

    fn validate(&self) -> Result<()> {
        ensure!(self.version == VERSION, "unsupported worker selection version");
        ensure!(self.identity.checkpoint.len() == 64 && self.identity.checkpoint.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid worker checkpoint digest");
        ensure!(matches!(self.identity.world, 1 | 2 | 3 | 4 | 6), "unsupported worker world");
        ensure!(self.identity.geometry.iter().all(|&value| value > 0), "invalid worker geometry");
        if let Some([tp, ep]) = self.identity.topology {
            ensure!(crate::expert::SparkTopology::new(tp, ep)?.world_size() == self.identity.world as usize,
                "worker topology/world mismatch");
        }
        ensure!(self.ranges.iter().all(|range| range.first < range.end && range.end <= self.identity.geometry[4]),
            "worker layer range is outside checkpoint");
        ensure!(self.ranges.windows(2).all(|pair| pair[0].end < pair[1].first),
            "worker ranges must be sorted, disjoint and coalesced");
        Ok(())
    }

    pub fn validate_for(&self, identity: &WorkerIdentity) -> Result<()> {
        self.validate()?;
        ensure!(&self.identity == identity, "worker checkpoint/geometry/topology identity mismatch");
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Start { message: String, rank: usize, selection: WorkerSelection, digest: String }
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready { message: String, rank: usize, selection: WorkerSelection, digest: String }

fn write(stream: &mut TcpStream, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *stream, value)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

/// Read one bounded control message without consuming the following bootstrap.
fn read<T: serde::de::DeserializeOwned>(stream: &TcpStream) -> Result<T> {
    let mut reader = BufReader::with_capacity(1, stream.try_clone()?);
    let mut line = Vec::new();
    let bytes = reader.by_ref().take((CONTROL_LIMIT + 1) as u64).read_until(b'\n', &mut line)?;
    ensure!(bytes > 0 && bytes <= CONTROL_LIMIT && line.last() == Some(&b'\n'), "invalid worker selection control frame");
    let value: serde_json::Value = serde_json::from_slice(&line)?;
    if value["message"] == "protocol_v2_bootstrap_error" {
        anyhow::bail!("worker selection rejected: {}", value["error"]);
    }
    Ok(serde_json::from_value(value)?)
}

pub fn receive_worker_selection(stream: &TcpStream, rank: usize, timeout: Duration) -> Result<WorkerSelection> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let start: Start = read(stream)?;
    ensure!(start.message == "worker_selection_start", "worker requires coordinator placement handshake");
    ensure!(start.rank == rank && rank < start.selection.identity.world as usize, "worker selection rank mismatch");
    start.selection.validate()?;
    ensure!(start.digest == start.selection.digest()?, "worker selection digest mismatch");
    Ok(start.selection)
}

/// Call only after the exact selected set has loaded and passed admission.
pub fn acknowledge_worker_selection(stream: &mut TcpStream, rank: usize, selection: &WorkerSelection) -> Result<()> {
    write(stream, &Ready { message: "worker_selection_ready".into(), rank,
        selection: selection.clone(), digest: selection.digest()? })
}

pub fn reject_worker_selection(stream: &mut TcpStream, error: &anyhow::Error) -> Result<()> {
    write(stream, &serde_json::json!({"message": "protocol_v2_bootstrap_error", "error": format!("{error:#}")}))
}

// Immutable peer seals are consulted only while opening a connection, never on
// a request's hot path. This lets every later lane carry the same solved set.
fn seals() -> &'static Mutex<BTreeMap<SocketAddr, String>> {
    static SEALS: OnceLock<Mutex<BTreeMap<SocketAddr, String>>> = OnceLock::new();
    SEALS.get_or_init(Mutex::default)
}
pub(crate) fn peer_selection_digest(peer: SocketAddr) -> Result<Option<String>> {
    Ok(seals().lock().map_err(|_| anyhow::anyhow!("worker selection seals poisoned"))?.get(&peer).cloned())
}

/// Solve first, then select all workers concurrently before opening Spark links.
/// An acknowledgment names the exact loaded set, rank and identity; all later
/// connections to these peers are sealed to that selection for this process.
pub fn select_worker_layers(peers: &[SocketAddr], selection: &WorkerSelection, timeout: Duration) -> Result<()> {
    selection.validate()?;
    ensure!(peers.len() == selection.identity.world as usize, "worker peer world mismatch");
    ensure!(peers.iter().collect::<std::collections::BTreeSet<_>>().len() == peers.len(), "duplicate worker peer");
    let digest = selection.digest()?;
    for &peer in peers {
        if let Some(previous) = peer_selection_digest(peer)? { ensure!(previous == digest, "worker peer already sealed to another placement"); }
    }
    std::thread::scope(|scope| -> Result<()> {
        let jobs = peers.iter().enumerate().map(|(rank, &peer)| scope.spawn(move || -> Result<()> {
            let mut stream = TcpStream::connect_timeout(&peer, timeout)?;
            stream.set_read_timeout(Some(timeout))?;
            stream.set_write_timeout(Some(timeout))?;
            write(&mut stream, &Start { message: "worker_selection_start".into(), rank, selection: selection.clone(), digest: selection.digest()? })?;
            let ready: Ready = read(&stream)?;
            ensure!(ready.message == "worker_selection_ready" && ready.rank == rank && &ready.selection == selection
                && ready.digest == selection.digest()?, "worker loaded-set acknowledgment mismatch for rank {rank}");
            Ok(())
        })).collect::<Vec<_>>();
        let mut failure = None;
        for job in jobs {
            let result = job.join().map_err(|_| anyhow::anyhow!("worker selection thread panicked")).and_then(|result| result);
            if let Err(error) = result { if failure.is_none() { failure = Some(error); } }
        }
        if let Some(error) = failure { return Err(error); }
        Ok(())
    })?;
    let mut seals = seals().lock().map_err(|_| anyhow::anyhow!("worker selection seals poisoned"))?;
    for &peer in peers {
        if let Some(previous) = seals.get(&peer) { ensure!(previous == &digest, "worker peer placement changed concurrently"); }
    }
    for &peer in peers { seals.insert(peer, digest.clone()); }
    Ok(())
}

/// A startup/reconnect control-only request; no QP or GPU allocation occurred.
#[derive(Debug)]
pub struct WorkerSelectionOnly;
impl std::fmt::Display for WorkerSelectionOnly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("worker selection acknowledged") }
}
impl std::error::Error for WorkerSelectionOnly {}

pub(crate) fn acknowledge_repeat(stream: &mut TcpStream, value: serde_json::Value, rank: usize,
    expected: &WorkerSelection) -> Result<()> {
    let start: Start = serde_json::from_value(value)?;
    ensure!(start.message == "worker_selection_start" && start.rank == rank && &start.selection == expected && start.digest == expected.digest()?,
        "worker selection is immutable until restart");
    acknowledge_worker_selection(stream, rank, expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity() -> WorkerIdentity { WorkerIdentity { checkpoint: "0".repeat(64), geometry: [6144, 384, 8, 2048, 70], world: 2, topology: None } }
    #[test]
    fn worker_selection_validates_absolute_ranges_and_identity() {
        let selection = WorkerSelection::new(identity(), &[13, 14, 15, 20]).unwrap();
        assert_eq!(selection.ranges(), &[LayerRange { first: 13, end: 16 }, LayerRange { first: 20, end: 21 }]);
        for invalid in [vec![70], vec![14, 13], vec![13, 13]] { assert!(WorkerSelection::new(identity(), &invalid).is_err()); }
        let mut other = identity(); other.checkpoint = "1".repeat(64);
        assert!(selection.validate_for(&other).is_err());
        assert_ne!(selection.digest().unwrap(), WorkerSelection::new(identity(), &[14, 15, 20]).unwrap().digest().unwrap());
        assert!(WorkerSelection::new(identity(), &[]).unwrap().ranges().is_empty());
        let mut invalid = identity(); invalid.world = 5;
        assert!(WorkerSelection::new(invalid, &[]).is_err());
        let mut invalid = identity(); invalid.topology = Some([2, 2]);
        assert!(WorkerSelection::new(invalid, &[]).is_err());
        assert!(WorkerSelection::new(identity(), &[u32::MAX as usize - 1, u32::MAX as usize]).is_err());
    }
    #[test]
    fn worker_selection_control_is_bounded_and_checks_rank_digest() {
        let selection = WorkerSelection::new(identity(), &[13, 14]).unwrap();
        for (rank, digest) in [(1, selection.digest().unwrap()), (0, "1".repeat(64))] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let (server, _) = listener.accept().unwrap();
            write(&mut client, &Start { message: "worker_selection_start".into(), rank,
                selection: selection.clone(), digest }).unwrap();
            assert!(receive_worker_selection(&server, 0, Duration::from_secs(3)).is_err());
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        client.write_all(&vec![b' '; CONTROL_LIMIT + 1]).unwrap();
        assert!(receive_worker_selection(&server, 0, Duration::from_secs(3)).is_err());
    }
    #[test]
    fn worker_selection_control_checks_exact_ack_and_reconnect() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let selection = WorkerSelection::new(WorkerIdentity { world: 1, ..identity() }, &[13, 14]).unwrap();
        let peer = listener.local_addr().unwrap();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let (mut stream, _) = listener.accept().unwrap();
                let received = receive_worker_selection(&stream, 0, Duration::from_secs(3)).unwrap();
                assert_eq!(received, selection);
                acknowledge_worker_selection(&mut stream, 0, &received).unwrap();
            });
            select_worker_layers(&[peer], &selection, Duration::from_secs(3)).unwrap();
        });
        assert_eq!(peer_selection_digest(peer).unwrap(), Some(selection.digest().unwrap()));
        let changed = WorkerSelection::new(WorkerIdentity { world: 1, ..identity() }, &[14]).unwrap();
        assert!(select_worker_layers(&[peer], &changed, Duration::from_secs(3)).is_err());
    }
}
