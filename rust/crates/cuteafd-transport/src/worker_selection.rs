//! Startup-only, coordinator-authoritative Spark weight selection.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::{BufRead, BufReader, Read, Write}, net::{SocketAddr, TcpStream},
    path::Path, sync::{Mutex, OnceLock}, time::Duration};

const CONTROL_LIMIT: usize = 64 * 1024;
const VERSION: u32 = 2;

fn check_version(actual: u64, expected: u32) -> Result<()> {
    ensure!(actual == u64::from(expected),
        "unsupported worker selection version {actual}; expected {expected}; rebuild coordinator and workers together");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerIdentity {
    checkpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    manifest: Option<cuteafd_loader::plan::checkpoint::identity::ManifestIdentity>,
    geometry: [u32; 5],
    world: u32,
    topology: [u8; 2],
    ownership_wire: bool,
}

impl WorkerIdentity {
    /// Paths may differ between hosts; the checkpoint's config and tensor index
    /// bytes, not a local pathname or a separately computed plan, identify it.
    pub fn from_snapshot(snapshot: &Path, geometry: cuteafd_core::ExpertGeometry,
        selected: crate::expert::SparkWorld) -> Result<Self> {
        Self::from_snapshot_manifest(snapshot, geometry, selected, None)
    }

    pub fn from_snapshot_manifest(snapshot: &Path, geometry: cuteafd_core::ExpertGeometry,
        selected: crate::expert::SparkWorld, manifest: Option<&str>) -> Result<Self> {
        let world = selected.topology().world_size();
        let identity = cuteafd_loader::plan::checkpoint::identity::CheckpointIdentity::from_snapshot(snapshot, manifest)?;
        Ok(Self { checkpoint: identity.fingerprint, manifest: identity.manifest,
            geometry: [geometry.hidden, geometry.experts, geometry.topk, geometry.intermediate, geometry.layers],
            world: u32::try_from(world)?, topology: [selected.topology().tp(), selected.topology().ep()],
            ownership_wire: selected.ownership_topology().is_some() })
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
    model: Option<ModelAdmission>,
}

/// The same immutable placement seal also fixes every persistent endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAdmission {
    /// Actual row storage and endpoint admission extent.
    pub capacity: u32,
    /// Main native export extent; scratch/program state may exceed live rows.
    pub compiled_capacity: u32,
    pub endpoints: Vec<cuteafd_core::SparkEndpoint>,
    pub rings: cuteafd_core::SparkRings,
}

impl ModelAdmission {
    /// Persistent clients retain endpoint leases between waves, even when their
    /// requests never overlap. Admit one endpoint per client, not per live wave.
    pub fn for_clients(geometry: cuteafd_core::ExpertGeometry,
        endpoint: cuteafd_core::SparkEndpoint, compiled_capacity: u32,
        rings: cuteafd_core::SparkRings, clients: usize) -> Result<Self> {
        ensure!((1..=8).contains(&clients), "ModelAdmission requires 1..=8 persistent clients");
        let model = Self { capacity: endpoint.capacity, compiled_capacity,
            endpoints: vec![endpoint; clients], rings };
        model.validate(geometry)?;
        model.require_clients(endpoint, clients)?;
        Ok(model)
    }

    /// Check the immutable worker contract before allocating or opening clients.
    pub fn require_clients(&self, endpoint: cuteafd_core::SparkEndpoint, clients: usize) -> Result<()> {
        ensure!((1..=8).contains(&clients), "ModelAdmission requires 1..=8 persistent clients");
        let admitted = self.endpoints.iter().filter(|spec|
            spec.capacity >= endpoint.capacity && spec.ingress == endpoint.ingress).count();
        ensure!(admitted >= clients,
            "ModelAdmission admits {admitted} matching endpoints for {clients} persistent clients ({endpoint:?})");
        Ok(())
    }

    pub fn validate(&self, geometry: cuteafd_core::ExpertGeometry) -> Result<()> {
        ensure!((1..=4096).contains(&self.capacity), "invalid ModelAdmission capacity");
        ensure!((self.capacity..=4096).contains(&self.compiled_capacity),
            "ModelAdmission compiled capacity {} must cover live capacity {} and be at most 4096",
            self.compiled_capacity, self.capacity);
        ensure!(!self.endpoints.is_empty() && self.endpoints.len() <= 8,
            "ModelAdmission requires 1..=8 endpoints");
        let endpoint_capacity = self.endpoints.iter().map(|endpoint| endpoint.capacity).max().unwrap();
        ensure!(self.capacity == endpoint_capacity,
            "ModelAdmission capacity {} differs from maximum endpoint capacity {endpoint_capacity}", self.capacity);
        for (endpoint, spec) in self.endpoints.iter().enumerate() {
            ensure!(spec.capacity <= self.capacity && spec.ring_bytes(geometry, self.rings).is_some(),
                "invalid ModelAdmission endpoint {endpoint} capacity/ingress/ring geometry");
        }
        self.ring_bytes(geometry)?;
        Ok(())
    }

    pub fn ring_bytes(&self, geometry: cuteafd_core::ExpertGeometry) -> Result<usize> {
        self.endpoints.iter().try_fold(0usize, |total, endpoint| {
            total.checked_add(endpoint.ring_bytes(geometry, self.rings)
                .context("invalid ModelAdmission endpoint")?).context("ModelAdmission ring overflow")
        })
    }
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
        let selection = Self { version: VERSION, identity, ranges, model: None };
        selection.validate()?;
        Ok(selection)
    }

    pub fn ranges(&self) -> &[LayerRange] { &self.ranges }
    /// Exact ordered absolute backbone IDs; ranges are only the compact wire encoding.
    pub fn layers(&self) -> Vec<usize> {
        self.ranges.iter().flat_map(|range| range.first as usize..range.end as usize).collect()
    }
    pub fn identity(&self) -> &WorkerIdentity { &self.identity }
    pub fn model(&self) -> Option<&ModelAdmission> { self.model.as_ref() }
    pub fn with_model(mut self, model: ModelAdmission) -> Result<Self> {
        self.model = Some(model);
        self.validate()?;
        Ok(self)
    }
    pub fn geometry(&self) -> cuteafd_core::ExpertGeometry {
        let [hidden, experts, topk, intermediate, layers] = self.identity.geometry;
        cuteafd_core::ExpertGeometry { hidden, experts, topk, intermediate, layers }
    }
    pub fn digest(&self) -> Result<String> { Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?))) }

    fn validate(&self) -> Result<()> {
        check_version(u64::from(self.version), VERSION)?;
        ensure!(self.identity.checkpoint.len() == 64 && self.identity.checkpoint.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid worker checkpoint digest");
        if let Some(manifest) = &self.identity.manifest {
            ensure!(!manifest.name.is_empty() && manifest.name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
                && manifest.sha256.len() == 64 && manifest.sha256.bytes().all(|c| c.is_ascii_hexdigit()),
                "invalid worker manifest identity");
        }
        ensure!((1..=6).contains(&self.identity.world), "unsupported worker world");
        ensure!(self.identity.geometry.iter().all(|&value| value > 0), "invalid worker geometry");
        let [tp, ep] = self.identity.topology;
        let topology = crate::expert::SparkTopology::new(tp, ep)?;
        let wire = if self.identity.ownership_wire { crate::expert::SparkWire::NativeOwnership }
            else { crate::expert::SparkWire::Canonical };
        let world = crate::expert::SparkWorld::new(topology, wire)?;
        ensure!(world.topology().world_size() == self.identity.world as usize,
            "worker topology/world mismatch");
        ensure!(self.ranges.iter().all(|range| range.first < range.end && range.end <= self.identity.geometry[4]),
            "worker layer range is outside checkpoint");
        ensure!(self.ranges.windows(2).all(|pair| pair[0].end < pair[1].first),
            "worker ranges must be sorted, disjoint and coalesced");
        if let Some(model) = &self.model { model.validate(self.geometry())?; }
        Ok(())
    }

    pub fn validate_for(&self, identity: &WorkerIdentity) -> Result<()> {
        self.validate()?;
        ensure!(&self.identity == identity, "worker checkpoint/geometry/topology identity mismatch");
        Ok(())
    }
}

/// One actually installed capacity specialization, including a hybrid's compact prefix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstalledExpertVariant {
    capacity: usize,
    group_rows: usize,
    compact_live_rows: usize,
    segments: Vec<ExpertGroupingSegment>,
    contract: ExpertTrafficContract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertTrafficContract { ExactGrouping, ConservativeRouteUpperBound }

impl ExpertTrafficContract {
    pub fn label(self) -> &'static str {
        match self { Self::ExactGrouping => "exact_grouping",
            Self::ConservativeRouteUpperBound => "conservative_route_upper_bound" }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertGroupingSegment { end_rows: usize, group_rows: usize }

impl InstalledExpertVariant {
    /// Read the loaded execution once at startup, never from a coordinator-side manifest.
    pub fn from_queries(capacity: usize, mut query: impl FnMut(usize) -> Result<usize>) -> Result<Self> {
        ensure!((1..=4096).contains(&capacity), "invalid installed expert variant capacity");
        let group_rows = query(0)?;
        ensure!(matches!(group_rows, 1 | 8 | 16 | 32 | 64 | 128), "invalid native installed expert group");
        let mut compact_live_rows = 0;
        let mut grouped = false;
        for rows in 1..=capacity {
            let group = query(rows)?;
            if group == group_rows { grouped = true; }
            else {
                ensure!(group == 1 && !grouped,
                    "installed expert grouping is not a compact prefix at rows {rows}");
                compact_live_rows = rows;
            }
        }
        let variant = Self { capacity, group_rows, compact_live_rows,
            segments: Vec::new(), contract: ExpertTrafficContract::ExactGrouping };
        variant.validate()?;
        Ok(variant)
    }

    /// Piecewise groups from the actual loaded FP8-family capacity/form.
    pub fn from_piecewise_queries(capacity: usize, contract: ExpertTrafficContract,
        mut query: impl FnMut(usize) -> Result<usize>) -> Result<Self> {
        ensure!((1..=4096).contains(&capacity), "invalid installed expert variant capacity");
        let group_rows = query(0)?;
        let mut segments = Vec::<ExpertGroupingSegment>::new();
        for rows in 1..=capacity {
            let group = query(rows)?;
            if let Some(last) = segments.last_mut().filter(|last| last.group_rows == group) {
                last.end_rows = rows;
            } else { segments.push(ExpertGroupingSegment { end_rows: rows, group_rows: group }); }
        }
        let variant = Self { capacity, group_rows, compact_live_rows: 0, segments, contract };
        variant.validate()?;
        Ok(variant)
    }

    fn validate(&self) -> Result<()> {
        ensure!((1..=4096).contains(&self.capacity)
            && (1..=4096).contains(&self.group_rows)
            && self.compact_live_rows <= self.capacity,
            "invalid installed expert capacity/group/compact cutoff");
        ensure!(self.group_rows != 1 || self.compact_live_rows == 0,
            "group-one expert variant cannot have a separate compact prefix");
        if !self.segments.is_empty() {
            ensure!(self.compact_live_rows == 0 && self.segments.len() <= 16
                && self.segments.iter().all(|segment| segment.end_rows > 0
                    && segment.end_rows <= self.capacity && (1..=4096).contains(&segment.group_rows))
                && self.segments.windows(2).all(|pair| pair[0].end_rows < pair[1].end_rows
                    && pair[0].group_rows != pair[1].group_rows)
                && self.segments.last().is_some_and(|last| last.end_rows == self.capacity
                    && last.group_rows == self.group_rows), "invalid piecewise installed expert grouping");
        }
        if self.contract == ExpertTrafficContract::ConservativeRouteUpperBound {
            ensure!(self.group_rows == 1 && self.compact_live_rows == 0
                && self.segments.iter().all(|segment| segment.group_rows == 1),
                "conservative expert traffic bound must charge every routed occurrence");
        }
        Ok(())
    }

    fn group_rows(&self, rows: usize) -> usize {
        if rows > 0 && !self.segments.is_empty() {
            return self.segments.iter().find(|segment| rows <= segment.end_rows)
                .expect("validated segment covers installed capacity").group_rows;
        }
        if rows > 0 && rows <= self.compact_live_rows { 1 } else { self.group_rows }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertGroupingSelector { Native, Exl3Nearest, Exl3GlmFlashK64, Fp8Nearest }

/// The exact loaded worker, its resident layers and its immutable launch selector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstalledExpertGrouping {
    loaded: WorkerSelection,
    rank: usize,
    capacity: usize,
    selector: ExpertGroupingSelector,
    variants: Vec<InstalledExpertVariant>,
    bf16_variants: Option<Vec<InstalledExpertVariant>>,
    /// Checkpoint weight and scale read extent per expert for this loaded rank.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    weight_read_bytes: Option<usize>,
}

impl InstalledExpertGrouping {
    pub fn new(loaded: WorkerSelection, rank: usize, capacity: usize,
        selector: ExpertGroupingSelector, mut variants: Vec<InstalledExpertVariant>) -> Result<Self> {
        variants.sort_by_key(|variant| variant.capacity);
        let report = Self { loaded, rank, capacity, selector, variants, bf16_variants: None, weight_read_bytes: None };
        report.validate()?;
        Ok(report)
    }

    pub fn with_bf16_variants(mut self, mut variants: Vec<InstalledExpertVariant>) -> Result<Self> {
        ensure!(self.selector == ExpertGroupingSelector::Fp8Nearest, "BF16 sibling requires FP8 selector");
        variants.sort_by_key(|variant| variant.capacity);
        self.bf16_variants = Some(variants);
        self.validate()?;
        Ok(self)
    }

    pub fn with_weight_read_bytes(mut self, bytes: usize) -> Result<Self> {
        self.weight_read_bytes = Some(bytes);
        self.validate()?;
        Ok(self)
    }

    pub fn weight_read_bytes(&self) -> Result<usize> {
        self.weight_read_bytes.context("installed expert read-byte descriptor missing; rebuild coordinator and workers together")
    }

    pub fn layers(&self) -> &[LayerRange] { self.loaded.ranges() }

    pub fn traffic_contract(&self) -> ExpertTrafficContract {
        if self.variants.iter().chain(self.bf16_variants.iter().flatten())
            .any(|variant| variant.contract == ExpertTrafficContract::ConservativeRouteUpperBound) {
            ExpertTrafficContract::ConservativeRouteUpperBound
        } else { ExpertTrafficContract::ExactGrouping }
    }

    fn validate(&self) -> Result<()> {
        self.loaded.validate()?;
        ensure!(self.weight_read_bytes.is_none_or(|bytes| bytes > 0)
            && (self.selector == ExpertGroupingSelector::Fp8Nearest || self.weight_read_bytes.is_none()),
            "invalid installed FP8 expert read-byte descriptor");
        ensure!(self.rank < self.loaded.identity.world as usize && !self.loaded.ranges.is_empty(),
            "invalid installed expert rank/resident layers");
        ensure!((1..=4096).contains(&self.capacity) && (1..=6).contains(&self.variants.len()),
            "invalid installed expert report capacity/variant count");
        if let Some(model) = self.loaded.model() {
            ensure!(self.capacity == model.capacity as usize,
                "installed expert live capacity differs from ModelAdmission");
        }
        for variant in &self.variants { variant.validate()?; }
        if self.selector != ExpertGroupingSelector::Fp8Nearest {
            ensure!(self.variants.iter().all(|variant| variant.segments.is_empty()
                && variant.contract == ExpertTrafficContract::ExactGrouping
                && matches!(variant.group_rows, 1 | 8 | 16 | 32 | 64 | 128)),
                "non-FP8 installed grouping must retain its native group contract");
        }
        ensure!(self.variants.windows(2).all(|pair| pair[0].capacity < pair[1].capacity),
            "installed expert capacities must be distinct and ascending");
        let capacities: Vec<_> = self.variants.iter().map(|variant| variant.capacity).collect();
        let expected = match self.selector {
            ExpertGroupingSelector::Native => {
                let main = *capacities.last().context("native main capacity missing")?;
                ensure!(main >= self.capacity
                    && capacities.iter().all(|&c| c == main || c == 1 || (c == 80 && main > 80)),
                    "native report must name its actual main and optional m1/m80 executions");
                if let Some(model) = self.loaded.model() {
                    ensure!(main == model.compiled_capacity as usize,
                        "installed native main capacity differs from ModelAdmission compiled capacity");
                }
                capacities.clone()
            }
            ExpertGroupingSelector::Fp8Nearest => {
                ensure!(capacities.last().is_some_and(|&last| last >= self.capacity),
                    "FP8 report must cover its admitted capacity with installed programs");
                capacities.clone()
            }
            ExpertGroupingSelector::Exl3Nearest | ExpertGroupingSelector::Exl3GlmFlashK64 => {
                let policy = if self.selector == ExpertGroupingSelector::Exl3Nearest {
                    cuteafd_core::Exl3RowPolicy::Nearest
                } else { cuteafd_core::Exl3RowPolicy::GlmFlashK64 };
                policy.capacities(self.capacity)?.into_iter().map(|c| c as usize).collect()
            }
        };
        ensure!(capacities == expected, "installed expert capacities disagree with launch selector");
        if let Some(variants) = &self.bf16_variants {
            ensure!(self.selector == ExpertGroupingSelector::Fp8Nearest && (1..=6).contains(&variants.len())
                && variants.windows(2).all(|pair| pair[0].capacity < pair[1].capacity)
                && variants.last().is_some_and(|last| last.capacity >= self.capacity),
                "invalid installed BF16 sibling capacities");
            for variant in variants { variant.validate()?; }
        }
        Ok(())
    }

    /// Zero is the main compiled tile; positive rows follow the installed selector.
    pub fn group_rows(&self, rows: usize) -> Result<usize> {
        self.group_rows_for_input(rows, false)
    }

    /// Select the same loaded BF16 sibling as the worker request adapter.
    pub fn group_rows_for_input(&self, rows: usize, bf16: bool) -> Result<usize> {
        let variants = if bf16 { self.bf16_variants.as_ref().unwrap_or(&self.variants) }
            else { &self.variants };
        ensure!(rows <= self.capacity, "expert rows {rows} exceed installed capacity {}", self.capacity);
        let required = if self.selector == ExpertGroupingSelector::Exl3GlmFlashK64 {
            cuteafd_core::Exl3RowPolicy::GlmFlashK64.required_capacity(rows)
        } else { rows };
        let variant = if rows == 0 { variants.last() } else {
            variants.iter().find(|variant| variant.capacity >= required)
        }.context("missing installed expert grouping capacity")?;
        Ok(variant.group_rows(rows))
    }

    pub fn validate_for(&self, identity: &WorkerIdentity, rank: usize, rows: usize,
        mut layers: std::ops::Range<usize>) -> Result<()> {
        self.validate()?;
        self.loaded.validate_for(identity)?;
        ensure!(self.rank == rank, "installed expert grouping rank mismatch");
        ensure!(rows <= self.capacity, "installed expert grouping does not cover verifier rows {rows}");
        ensure!(layers.all(|layer| self.loaded.ranges.iter()
            .any(|range| (range.first as usize..range.end as usize).contains(&layer))),
            "installed expert grouping does not cover admitted remote layers");
        Ok(())
    }

    pub(crate) fn validate_connection(&self, rank: usize, world: usize) -> Result<()> {
        self.validate()?;
        let geometry = cuteafd_core::expert_geometry();
        ensure!(self.rank == rank && self.loaded.identity.world as usize == world
            && self.loaded.identity.geometry == [geometry.hidden, geometry.experts, geometry.topk,
                geometry.intermediate, geometry.layers],
            "installed expert grouping connection rank/world/geometry mismatch");
        Ok(())
    }

    /// Mixed packages for any shared layer fail before the coordinator becomes ready.
    pub fn validate_ranks(reports: &[Self]) -> Result<()> {
        ensure!(!reports.is_empty(), "no installed expert grouping reports");
        for (rank, report) in reports.iter().enumerate() {
            report.validate()?;
            ensure!(report.rank == rank && report.loaded.identity.world as usize == reports.len()
                && report.loaded.identity == reports[0].loaded.identity,
                "installed expert grouping identity/rank/world mismatch");
            for previous in &reports[..rank] {
                let overlap = report.layers().iter().any(|a| previous.layers().iter()
                    .any(|b| a.first < b.end && b.first < a.end));
                ensure!(!overlap || (report.capacity == previous.capacity
                    && report.selector == previous.selector && report.variants == previous.variants
                    && report.bf16_variants == previous.bf16_variants
                    && report.weight_read_bytes.is_some() == previous.weight_read_bytes.is_some()),
                    "installed expert grouping differs across ranks {} and {rank} for the same layer (mixed packages)",
                    previous.rank);
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Start { message: String, rank: usize, selection: WorkerSelection, digest: String }
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    message: String,
    rank: usize,
    selection: WorkerSelection,
    digest: String,
    admission: Option<cuteafd_core::ModelAdmissionResponse>,
}

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
    if matches!(value["message"].as_str(), Some("worker_selection_start" | "worker_selection_ready")) {
        check_version(value["selection"]["version"].as_u64().context("missing worker selection version")?, VERSION)?;
    }
    Ok(serde_json::from_value(value)?)
}

pub fn receive_worker_selection(stream: &TcpStream, rank: usize, timeout: Duration) -> Result<WorkerSelection> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let value: serde_json::Value = match read(stream) {
        Ok(value) => value,
        Err(error) => {
            let _ = reject_worker_selection(&mut stream.try_clone()?, &error);
            return Err(error);
        }
    };
    if value["message"] == "fabric_traffic" {
            crate::traffic::control_phase(&value);
        write(&mut stream.try_clone()?, &serde_json::json!({"message": "fabric_traffic", "sample": crate::traffic::local()}))?;
        return Err(crate::traffic::TrafficProbeOnly.into());
    }
    let start: Start = serde_json::from_value(value)?;
    ensure!(start.message == "worker_selection_start", "worker requires coordinator placement handshake");
    ensure!(start.rank == rank && rank < start.selection.identity.world as usize, "worker selection rank mismatch");
    start.selection.validate()?;
    ensure!(start.digest == start.selection.digest()?, "worker selection digest mismatch");
    Ok(start.selection)
}

pub fn acknowledge_model_admission(stream: &mut TcpStream, rank: usize, selection: &WorkerSelection,
    admission: Option<&cuteafd_core::ModelAdmissionResponse>) -> Result<()> {
    validate_admission_response(selection, rank, admission)?;
    write(stream, &Ready { message: "worker_selection_ready".into(), rank,
        selection: selection.clone(), digest: selection.digest()?, admission: admission.cloned() })
}

fn validate_admission_response(selection: &WorkerSelection, rank: usize,
    report: Option<&cuteafd_core::ModelAdmissionResponse>) -> Result<()> {
    match (selection.model(), report) {
        (Some(model), Some(report)) => {
            ensure!(report.rank as usize == rank, "ModelAdmissionResponse rank mismatch for rank {rank}");
            ensure!(report.mapped_ring_bytes == if selection.ranges().is_empty() { 0 }
                else { model.ring_bytes(selection.geometry())? as u64 },
                "ModelAdmissionResponse ring bytes differ for rank {rank}");
            let layers = selection.ranges().iter().flat_map(|range| range.first..range.end).collect::<Vec<_>>();
            ensure!(report.slices.iter().map(|slice| slice.layer).collect::<Vec<_>>() == layers,
                "ModelAdmissionResponse slice layers differ for rank {rank}");
            ensure!(report.slices.iter().all(|slice| slice.width > 0 && slice.stored_width >= slice.width
                && slice.first.checked_add(slice.width).is_some_and(|end| end <= selection.geometry().intermediate)
                && slice.tiers.len() == slice.projection_counts.len()),
                "ModelAdmissionResponse malformed slice geometry for rank {rank}");
            Ok(())
        }
        (None, None) => Ok(()),
        _ => anyhow::bail!("ModelAdmissionResponse presence differs for rank {rank}"),
    }
}

pub fn reject_worker_selection(stream: &mut TcpStream, error: &anyhow::Error) -> Result<()> {
    write(stream, &serde_json::json!({"message": "protocol_v2_bootstrap_error", "error": format!("{error:#}")}))
}

// Immutable peer seals are consulted only while opening a connection, never on
// a request's hot path. This lets every later lane carry the same solved set.
#[derive(Clone)]
struct PeerSeal {
    digest: String,
    admission: Option<cuteafd_core::ModelAdmissionResponse>,
    endpoints: std::sync::Arc<EndpointSet>,
}
impl PartialEq for PeerSeal {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest && self.admission == other.admission
    }
}
impl Eq for PeerSeal {}
fn seals() -> &'static Mutex<BTreeMap<SocketAddr, PeerSeal>> {
    static SEALS: OnceLock<Mutex<BTreeMap<SocketAddr, PeerSeal>>> = OnceLock::new();
    SEALS.get_or_init(Mutex::default)
}
pub(crate) fn peer_selection_digest(peer: SocketAddr) -> Result<Option<String>> {
    Ok(seals().lock().map_err(|_| anyhow::anyhow!("worker selection seals poisoned"))?
        .get(&peer).map(|seal| seal.digest.clone()))
}

/// Refuse an undersized worker contract before constructing persistent clients.
pub fn require_peer_clients(peers: &[SocketAddr], endpoint: cuteafd_core::SparkEndpoint,
    clients: usize) -> Result<()> {
    let seals = seals().lock().map_err(|_| anyhow::anyhow!("worker selection seals poisoned"))?;
    for peer in peers {
        seals.get(peer).and_then(|seal| seal.endpoints.selection.model())
            .context("persistent clients require sealed ModelAdmission")?
            .require_clients(endpoint, clients)
            .with_context(|| format!("worker {peer} persistent-client admission"))?;
    }
    Ok(())
}

/// Claim one coordinator endpoint before allocating its native buffers. The
/// lease lives on that endpoint, including any retained response payloads.
pub(crate) fn reserve_peer_endpoint(peer: SocketAddr, header: &crate::ExpertProtocolV2RequestHeader,
    capacity: Option<u32>) -> Result<Option<EndpointReservation>> {
    let endpoints = seals().lock().map_err(|_| anyhow::anyhow!("worker selection seals poisoned"))?
        .get(&peer).map(|seal| seal.endpoints.clone());
    let Some(endpoints) = endpoints else { return Ok(None); };
    let Some(model) = endpoints.selection.model() else { return Ok(None); };
    let installed = InstalledModelAdmission { rank: 0, endpoints: endpoints.clone(), report: None };
    let mut candidates = model.endpoints.iter().enumerate().collect::<Vec<_>>();
    candidates.sort_by_key(|(_, spec)| spec.capacity);
    for (index, spec) in candidates {
        if capacity.is_some_and(|rows| rows > spec.capacity) { continue; }
        let valid = validate_endpoint_request(&endpoints.selection, index, header).is_ok();
        if valid {
            if let Ok(lease) = installed.reserve_endpoint(Some(index as u32)) { return Ok(lease); }
        }
    }
    anyhow::bail!("worker {peer} has no free ModelAdmission endpoint for capacity {:?} / {:?}", capacity, header.hidden_dtype)
}

// Prefill lanes and device proxies open separate clients to the same worker.
// Seal their loaded reports together, including reconnects, before posting work.
pub(crate) fn seal_installed_grouping(peer: SocketAddr, current: Option<&InstalledExpertGrouping>) -> Result<()> {
    static GROUPING: OnceLock<Mutex<BTreeMap<SocketAddr, InstalledExpertGrouping>>> = OnceLock::new();
    if let Some(report) = current {
        report.validate()?;
        if let Some(digest) = peer_selection_digest(peer)? {
            ensure!(report.loaded.digest()? == digest, "worker {peer} grouping differs from sealed placement");
        }
    }
    let mut seals = GROUPING.get_or_init(Mutex::default).lock()
        .map_err(|_| anyhow::anyhow!("worker grouping seals poisoned"))?;
    if let Some(previous) = seals.get(&peer) {
        ensure!(current == Some(previous), "worker {peer} changed installed grouping across lanes or reconnect");
    } else if let Some(report) = current {
        seals.insert(peer, report.clone());
    }
    Ok(())
}

/// Sent selections own only control sockets; dropping closes them without leaving
/// startup threads behind. No peer seal is published until every reply validates.
pub struct PendingModelAdmission {
    peers: Vec<SocketAddr>,
    selection: WorkerSelection,
    streams: Vec<TcpStream>,
}

/// Send the solved selection concurrently, without waiting for worker loading.
pub fn start_model_admission(peers: &[SocketAddr], selection: &WorkerSelection,
    timeout: Duration) -> Result<PendingModelAdmission> {
    selection.validate()?;
    ensure!(selection.model().is_some(), "Spark selection requires typed ModelAdmission");
    ensure!(peers.len() == selection.identity.world as usize, "worker peer world mismatch");
    ensure!(peers.iter().collect::<std::collections::BTreeSet<_>>().len() == peers.len(), "duplicate worker peer");
    let digest = selection.digest()?;
    for &peer in peers {
        if let Some(previous) = peer_selection_digest(peer)? { ensure!(previous == digest, "worker peer already sealed to another placement"); }
    }
    let streams = std::thread::scope(|scope| -> Result<Vec<TcpStream>> {
        let jobs = peers.iter().enumerate().map(|(rank, &peer)| scope.spawn(move || -> Result<TcpStream> {
            let mut stream = TcpStream::connect_timeout(&peer, timeout)?;
            stream.set_read_timeout(Some(timeout))?;
            stream.set_write_timeout(Some(timeout))?;
            write(&mut stream, &Start { message: "worker_selection_start".into(), rank,
                selection: selection.clone(), digest: selection.digest()? })?;
            Ok(stream)
        })).collect::<Vec<_>>();
        let mut streams = Vec::new();
        let mut failure = None;
        for job in jobs {
            match job.join().map_err(|_| anyhow::anyhow!("worker selection thread panicked")).and_then(|result| result) {
                Ok(stream) => streams.push(stream),
                Err(error) => { if failure.is_none() { failure = Some(error); } }
            }
        }
        if let Some(error) = failure { return Err(error); }
        Ok(streams)
    })?;
    Ok(PendingModelAdmission { peers: peers.to_vec(), selection: selection.clone(), streams })
}

impl PendingModelAdmission {
    /// Await rank-ordered measurements and seal validated replies before links open.
    pub fn wait(self) -> Result<Vec<cuteafd_core::ModelAdmissionResponse>> {
        let Self { peers, selection, streams } = self;
        let digest = selection.digest()?;
        let selection = &selection;
        let reports = std::thread::scope(|scope| -> Result<Vec<cuteafd_core::ModelAdmissionResponse>> {
            let jobs = streams.into_iter().enumerate().map(|(rank, stream)| scope.spawn(move || -> Result<Option<cuteafd_core::ModelAdmissionResponse>> {
                let ready: Ready = read(&stream)?;
                ensure!(ready.message == "worker_selection_ready" && ready.rank == rank && &ready.selection == selection
                    && ready.digest == selection.digest()?, "worker loaded-set acknowledgment mismatch for rank {rank}");
                validate_admission_response(selection, rank, ready.admission.as_ref())?;
                Ok(ready.admission)
            })).collect::<Vec<_>>();
            let mut failure = None;
            let mut reports = Vec::new();
            for job in jobs {
                match job.join().map_err(|_| anyhow::anyhow!("worker selection thread panicked")).and_then(|result| result) {
                    Ok(Some(report)) => reports.push(report),
                    Ok(None) => {},
                    Err(error) => { if failure.is_none() { failure = Some(error); } }
                }
            }
            if let Some(error) = failure { return Err(error); }
            Ok(reports)
        })?;
        let mut seals = seals().lock().map_err(|_| anyhow::anyhow!("worker selection seals poisoned"))?;
        let candidates = peers.iter().enumerate().map(|(rank, &peer)| (peer, PeerSeal {
            digest: digest.clone(), admission: reports.get(rank).cloned(),
            endpoints: std::sync::Arc::new(EndpointSet { selection: (*selection).clone(), used: std::sync::atomic::AtomicUsize::new(0) }),
        })).collect::<Vec<_>>();
        for (peer, seal) in &candidates {
            if let Some(previous) = seals.get(peer) {
                ensure!(previous == seal, "worker {peer} changed placement or admission across reconnect");
            }
        }
        for (peer, seal) in candidates { seals.entry(peer).or_insert(seal); }
        crate::traffic::observe_peers(&peers);
        Ok(reports)
    }
}

/// Send and await immediately when no independent coordinator load can overlap.
pub fn select_model_admission(peers: &[SocketAddr], selection: &WorkerSelection,
    timeout: Duration) -> Result<Vec<cuteafd_core::ModelAdmissionResponse>> {
    start_model_admission(peers, selection, timeout)?.wait()
}

/// A startup/reconnect control-only request; no QP or GPU allocation occurred.
#[derive(Debug)]
pub struct WorkerSelectionOnly;
impl std::fmt::Display for WorkerSelectionOnly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("worker selection acknowledged") }
}
impl std::error::Error for WorkerSelectionOnly {}

/// The worker's loaded, immutable selection and measured response. Every later
/// control replay and endpoint admission uses this same owner.
pub struct InstalledModelAdmission {
    rank: usize,
    endpoints: std::sync::Arc<EndpointSet>,
    report: Option<cuteafd_core::ModelAdmissionResponse>,
}

struct EndpointSet {
    selection: WorkerSelection,
    used: std::sync::atomic::AtomicUsize,
}

impl InstalledModelAdmission {
    pub fn new(rank: usize, selection: WorkerSelection,
        report: Option<cuteafd_core::ModelAdmissionResponse>) -> Result<Self> {
        selection.validate()?;
        ensure!(rank < selection.identity.world as usize, "installed worker rank outside world");
        validate_admission_response(&selection, rank, report.as_ref())?;
        Ok(Self { rank, endpoints: std::sync::Arc::new(EndpointSet {
            selection, used: std::sync::atomic::AtomicUsize::new(0),
        }), report })
    }

    pub fn selection(&self) -> &WorkerSelection { &self.endpoints.selection }
    pub fn acknowledge(&self, stream: &mut TcpStream) -> Result<()> {
        acknowledge_model_admission(stream, self.rank, self.selection(), self.report.as_ref())
    }
    pub(crate) fn acknowledge_repeat(&self, stream: &mut TcpStream, value: serde_json::Value) -> Result<()> {
        let start: Start = serde_json::from_value(value)?;
        ensure!(start.message == "worker_selection_start" && start.rank == self.rank
            && &start.selection == self.selection() && start.digest == self.selection().digest()?,
            "worker selection is immutable until restart");
        self.acknowledge(stream)
    }

    pub(crate) fn endpoint_spec(&self, index: Option<u32>) -> Result<Option<cuteafd_core::SparkEndpoint>> {
        match self.selection().model() {
            None => { ensure!(index.is_none(), "untyped worker cannot admit a typed endpoint"); Ok(None) }
            Some(model) => {
                ensure!(!self.selection().ranges().is_empty(), "empty worker cannot admit expert endpoints");
                let index = index.context("ModelAdmission endpoint identity is required")? as usize;
                Ok(Some(*model.endpoints.get(index).context("ModelAdmission endpoint index outside admitted set")?))
            }
        }
    }

    pub(crate) fn reserve_endpoint(&self, index: Option<u32>) -> Result<Option<EndpointReservation>> {
        use std::sync::atomic::Ordering;
        if self.endpoint_spec(index)?.is_none() { return Ok(None); }
        let index = index.expect("typed endpoint validated") as usize;
        let bit = 1usize << index;
        self.endpoints.used.try_update(Ordering::AcqRel, Ordering::Acquire, |used|
            (used & bit == 0).then_some(used | bit))
            .map_err(|_| anyhow::anyhow!("ModelAdmission endpoint {index} is already live or pending"))?;
        Ok(Some(EndpointReservation { endpoints: self.endpoints.clone(), index }))
    }
}

pub(crate) struct EndpointReservation {
    endpoints: std::sync::Arc<EndpointSet>,
    index: usize,
}

impl EndpointReservation {
    pub(crate) fn validate_request(&self, header: &crate::ExpertProtocolV2RequestHeader) -> Result<()> {
        validate_endpoint_request(&self.endpoints.selection, self.index, header)
    }
    pub(crate) fn index(&self) -> u32 { self.index as u32 }
    pub(crate) fn contract(&self) -> (cuteafd_core::SparkEndpoint, cuteafd_core::SparkRings) {
        let model = self.endpoints.selection.model().expect("typed endpoint");
        (model.endpoints[self.index], model.rings)
    }
}

fn validate_endpoint_request(selection: &WorkerSelection, index: usize,
    header: &crate::ExpertProtocolV2RequestHeader) -> Result<()> {
    use cuteafd_core::SparkIngress;
    use crate::ExpertV2Dtype;
    let spec = selection.model().expect("typed endpoint").endpoints[index];
    ensure!(header.row_count <= spec.capacity,
        "endpoint {} rows {} exceed admitted capacity {}", index, header.row_count, spec.capacity);
    let rotated = header.flags & crate::EXPERT_PROTOCOL_V2_FLAG_EXLR8_ROTATED != 0;
    ensure!(rotated == (spec.ingress == SparkIngress::Exlr8Rotated),
        "endpoint {} rejects a different activation domain", index);
    let ingress = match spec.ingress {
        SparkIngress::Exlr8Rotated => header.hidden_dtype == ExpertV2Dtype::F16
            && header.hidden_dim.checked_mul(2) == Some(header.hidden_row_stride_bytes)
            && header.hidden_dim % 128 == 0,
        SparkIngress::Fp8K32 => header.hidden_dtype == ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        SparkIngress::Bf16 => header.hidden_dtype == ExpertV2Dtype::Bf16,
        SparkIngress::Bf16Decode { rows } => header.hidden_dtype == ExpertV2Dtype::Fp8E4m3Ue8m0K32
            || (header.hidden_dtype == ExpertV2Dtype::Bf16 && header.row_count <= rows),
    };
    ensure!(ingress, "endpoint {} rejects {:?} ingress at {} rows", index, header.hidden_dtype, header.row_count);
    let geometry = selection.geometry();
    ensure!(header.hidden_dim == geometry.hidden && header.route_count as u64 <= header.row_count as u64 * geometry.topk as u64,
        "endpoint {} request exceeds admitted hidden/route geometry", index);
    ensure!(selection.ranges().iter().any(|range| (range.first..range.end).contains(&header.layer_id)),
        "endpoint {} request layer {} outside loaded selection", index, header.layer_id);
    Ok(())

}


impl Drop for EndpointReservation {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        self.endpoints.used.fetch_and(!(1usize << self.index), Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity() -> WorkerIdentity { WorkerIdentity { checkpoint: "0".repeat(64), manifest: None, geometry: [6144, 384, 8, 2048, 70], world: 2, topology: [2, 1], ownership_wire: false } }
    #[test]
    fn exlr8_bootstrap_refuses_different_manifest_name_or_content() {
        use cuteafd_loader::plan::checkpoint::identity::ManifestIdentity;
        let mut coordinator = identity();
        coordinator.manifest = Some(ManifestIdentity { name: "up1".into(), sha256: "a".repeat(64) });
        let selection = WorkerSelection::new(coordinator.clone(), &[13]).unwrap();
        // Exercise the serialized bootstrap seal, not just the digest helper.
        let selection: WorkerSelection = serde_json::from_slice(&serde_json::to_vec(&selection).unwrap()).unwrap();
        selection.validate_for(&coordinator).unwrap();
        let mut worker = coordinator.clone();
        worker.manifest.as_mut().unwrap().name = "bit325".into();
        assert!(selection.validate_for(&worker).is_err());
        worker = coordinator.clone();
        worker.manifest.as_mut().unwrap().sha256 = "b".repeat(64);
        assert!(selection.validate_for(&worker).is_err());
        worker = coordinator.clone();
        worker.checkpoint = "c".repeat(64);
        assert!(selection.validate_for(&worker).is_err());
        worker = coordinator;
        worker.manifest = None;
        assert!(selection.validate_for(&worker).is_err());
    }

    #[test]
    fn exlr8_manifest_mismatch_rejects_control_handshake_without_publishing_seal() -> Result<()> {
        use cuteafd_loader::plan::checkpoint::identity::ManifestIdentity;
        let mut coordinator = WorkerIdentity { world: 1, topology: [1, 1], ..identity() };
        coordinator.manifest = Some(ManifestIdentity { name: "up1".into(), sha256: "a".repeat(64) });
        for content_mismatch in [false, true] {
            let mut worker = coordinator.clone();
            if content_mismatch { worker.manifest.as_mut().unwrap().sha256 = "b".repeat(64); }
            else { worker.manifest.as_mut().unwrap().name = "bit325".into(); }
            let selection = WorkerSelection::new(coordinator.clone(), &[13])?
                .with_model(admission_selection().model().unwrap().clone())?;
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let peer = listener.local_addr()?;
            let pending = start_model_admission(&[peer], &selection, Duration::from_secs(3))?;
            let (mut stream, _) = listener.accept()?;
            let received = receive_worker_selection(&stream, 0, Duration::from_secs(3))?;
            let error = received.validate_for(&worker).unwrap_err();
            reject_worker_selection(&mut stream, &error)?;
            assert!(pending.wait().is_err());
            assert!(peer_selection_digest(peer)?.is_none());
        }
        Ok(())
    }

    #[test]
    fn worker_selection_versions_refuse_before_decoding_identity() -> Result<()> {
        for (actual, expected) in [(1, 2), (2, 1)] {
            let error = check_version(actual, expected).unwrap_err().to_string();
            assert!(error.contains(&format!("version {actual}; expected {expected}")));
        }
        for message in ["worker_selection_start", "worker_selection_ready"] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let mut client = TcpStream::connect(listener.local_addr()?)?;
            let (server, _) = listener.accept()?;
            server.set_read_timeout(Some(Duration::from_secs(3)))?;
            // Deliberately undecodable identity: version must win over schema errors.
            write(&mut client, &serde_json::json!({"message":message,
                "selection":{"version":1,"identity":{"legacy":true}}}))?;
            let error = read::<serde_json::Value>(&server).unwrap_err().to_string();
            assert!(error.contains("version 1; expected 2"), "{error}");
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let mut client = TcpStream::connect(listener.local_addr()?)?;
        let (server, _) = listener.accept()?;
        write(&mut client, &serde_json::json!({"message":"worker_selection_start",
            "selection":{"version":1}}))?;
        let error = receive_worker_selection(&server, 0, Duration::from_secs(3)).unwrap_err();
        assert!(error.to_string().contains("version 1; expected 2"));
        let rejection = read::<serde_json::Value>(&client).unwrap_err();
        assert!(rejection.to_string().contains("version 1; expected 2"));
        // A v1 decoder accepts the unchanged HF envelope shape far enough to
        // issue its own clear version refusal when presented with v2.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct V1Identity {
            checkpoint: String, geometry: [u32; 5], world: u32,
            topology: [u8; 2], ownership_wire: bool,
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct V1Selection {
            version: u32, identity: V1Identity, ranges: Vec<LayerRange>, model: Option<ModelAdmission>,
        }
        let envelope = serde_json::to_value(admission_selection())?;
        assert!(envelope["identity"].get("manifest").is_none());
        let legacy: V1Selection = serde_json::from_value(envelope)?;
        assert_eq!(legacy.identity.checkpoint.len(), 64);
        assert_eq!((legacy.identity.world, legacy.identity.topology, legacy.identity.ownership_wire), (1, [1, 1], false));
        assert_eq!(legacy.identity.geometry[0], 6144);
        assert!(!legacy.ranges.is_empty() && legacy.model.is_some());
        assert!(check_version(u64::from(legacy.version), 1).unwrap_err()
            .to_string().contains("version 2; expected 1"));
        Ok(())
    }

    #[test]
    fn hf_indexed_glm_worker_bootstrap_identity_is_stable() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_snapshot(root.path(),
            &cuteafd_loader::plan::testing::glm5_config(),
            &[cuteafd_loader::plan::testing::t("model.norm.weight", "BF16", &[6144])], None);
        let geometry = cuteafd_core::ExpertGeometry::GLM5;
        let world = crate::expert::SparkWorld::canonical(1)?;
        let coordinator = WorkerIdentity::from_snapshot(root.path(), geometry, world)?;
        let worker = WorkerIdentity::from_snapshot(root.path(), geometry, world)?;
        assert_eq!(coordinator, worker);
        let selection = WorkerSelection::new(coordinator, &[1])?;
        selection.validate_for(&worker)?;
        assert_eq!(selection.digest()?, WorkerSelection::new(worker.clone(), &[1])?.digest()?);
        let selection = selection.with_model(admission_selection().model().unwrap().clone())?;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let peer = listener.local_addr()?;
        let pending = start_model_admission(&[peer], &selection, Duration::from_secs(3))?;
        let (mut stream, _) = listener.accept()?;
        let received = receive_worker_selection(&stream, 0, Duration::from_secs(3))?;
        received.validate_for(&worker)?;
        let mut report = admission_report(&selection);
        report.slices[0].layer = 1;
        acknowledge_model_admission(&mut stream, 0, &received, Some(&report))?;
        assert_eq!(pending.wait()?, [report]);
        assert_eq!(peer_selection_digest(peer)?, Some(selection.digest()?));
        Ok(())
    }

    fn grouping(rank: usize, compact: usize) -> InstalledExpertGrouping {
        let loaded = WorkerSelection::new(identity(), &[13, 14, 15]).unwrap();
        let variants = [1, 80, 256].into_iter().map(|capacity|
            InstalledExpertVariant::from_queries(capacity, |rows|
                Ok(if rows > 0 && rows <= compact.min(capacity) { 1 } else { 16 })).unwrap()).collect();
        InstalledExpertGrouping::new(loaded, rank, 256, ExpertGroupingSelector::Native, variants).unwrap()
    }

    fn admission_selection() -> WorkerSelection {
        use cuteafd_core::{SparkEndpoint, SparkIngress, SparkRings};
        WorkerSelection::new(WorkerIdentity { world: 1, topology: [1, 1], ..identity() }, &[13]).unwrap()
            .with_model(ModelAdmission { capacity: 4096, compiled_capacity: 4096,
                endpoints: vec![SparkEndpoint { capacity: 4096, ingress: SparkIngress::Bf16Decode { rows: 80 } }],
                rings: SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 } }).unwrap()
    }

    fn admission_report(selection: &WorkerSelection) -> cuteafd_core::ModelAdmissionResponse {
        use cuteafd_core::{ModelAdmissionResponse, RuntimeSample, SparkRuntimeInventory, SparkSlice};
        ModelAdmissionResponse { rank: 0, resident_bytes: 1 << 30,
            slices: vec![SparkSlice { layer: 13, first: 0, width: 2048, stored_width: 2048,
                tiers: vec![], projection_counts: vec![] }],
            runtime: SparkRuntimeInventory { detail: Default::default(), sample: RuntimeSample { device: 0, arch: "sm_121".into(),
                sms: 48, driver: "test".into(), total_bytes: 121 << 30, free_bytes: 100 << 30,
                context_bytes: 1 << 30, module_bytes: 1 << 20 },
                workspace_bytes: 1 << 28, pinned_workspace_bytes: 0 },
            mapped_ring_bytes: selection.model().unwrap().ring_bytes(selection.geometry()).unwrap() as u64,
            host_exchange_bytes: 4096 * (6144 * 2 + 8 * 8), row_indices_bytes: 4096 * 4 }
    }

    #[test]
    fn endpoint_identity_capacity_ingress_and_lifetime_are_sealed() -> Result<()> {
        let selection = admission_selection();
        let installed = InstalledModelAdmission::new(0, selection.clone(), Some(admission_report(&selection)))?;
        assert!(installed.reserve_endpoint(None).is_err());
        assert!(installed.reserve_endpoint(Some(1)).is_err());
        let lease = installed.reserve_endpoint(Some(0))?.unwrap();
        assert!(installed.reserve_endpoint(Some(0)).is_err());
        let mut header = crate::ExpertProtocolV2RequestHeader {
            request_id: 1, placement_version: 1, layer_id: 13, row_count: 80,
            hidden_dim: 6144, hidden_dtype: crate::ExpertV2Dtype::Bf16,
            hidden_row_stride_bytes: 12288, route_count: 640, row_descriptor_bytes: 40,
            route_bytes: 12, hidden_payload_bytes: 80 * 12288, flags: 0,
        };
        lease.validate_request(&header)?;
        header.row_count = 81;
        assert!(lease.validate_request(&header).is_err());
        header.hidden_dtype = crate::ExpertV2Dtype::Fp8E4m3Ue8m0K32;
        lease.validate_request(&header)?;
        header.row_count = 4097;
        assert!(lease.validate_request(&header).is_err());
        header.row_count = 80;
        header.layer_id = 14;
        assert!(lease.validate_request(&header).is_err());
        header.layer_id = 13;
        header.hidden_dim = 5120;
        assert!(lease.validate_request(&header).is_err());
        header.hidden_dim = 6144;
        header.route_count = 641;
        assert!(lease.validate_request(&header).is_err());
        drop(lease);
        drop(installed.reserve_endpoint(Some(0))?);
        Ok(())
    }

    #[test]
    fn exlr8_rotated_endpoint_binds_domain_dtype_capacity_and_ring_extent() -> Result<()> {
        let mut selection = admission_selection();
        let model = selection.model.as_mut().unwrap();
        model.capacity = 64;
        model.compiled_capacity = 64;
        model.endpoints[0] = cuteafd_core::SparkEndpoint {
            capacity: 64, ingress: cuteafd_core::SparkIngress::Exlr8Rotated,
        };
        let mut report = admission_report(&selection);
        report.host_exchange_bytes = 64 * (6144 * 4 + 8 * 8);
        report.row_indices_bytes = 64 * 4;
        let installed = InstalledModelAdmission::new(0, selection, Some(report))?;
        let lease = installed.reserve_endpoint(Some(0))?.unwrap();
        let mut header = crate::ExpertProtocolV2RequestHeader {
            request_id: 1, placement_version: 1, layer_id: 13, row_count: 64,
            hidden_dim: 6144, hidden_dtype: crate::ExpertV2Dtype::F16,
            hidden_row_stride_bytes: 12288, route_count: 512, row_descriptor_bytes: 64 * 40,
            route_bytes: 512 * 12, hidden_payload_bytes: 64 * 12288,
            flags: crate::EXPERT_PROTOCOL_V2_FLAG_EXLR8_ROTATED,
        };
        lease.validate_request(&header)?;
        header.flags = 0;
        assert!(lease.validate_request(&header).is_err());
        header.flags = crate::EXPERT_PROTOCOL_V2_FLAG_EXLR8_ROTATED;
        header.hidden_dtype = crate::ExpertV2Dtype::Bf16;
        assert!(lease.validate_request(&header).is_err());
        header.hidden_dtype = crate::ExpertV2Dtype::F16;
        header.hidden_row_stride_bytes += 2;
        assert!(lease.validate_request(&header).is_err());
        header.hidden_row_stride_bytes -= 2;
        header.row_count = 65;
        assert!(lease.validate_request(&header).is_err());
        Ok(())
    }

    #[test]
    fn sparse_installed_layers_are_exact_and_gaps_reject_requests() -> Result<()> {
        let mut selection = admission_selection();
        selection.ranges = WorkerSelection::new(selection.identity.clone(), &[13, 15, 20])?.ranges;
        let mut report = admission_report(&selection);
        let slice = report.slices[0].clone();
        report.slices = selection.layers().into_iter().map(|layer| cuteafd_core::SparkSlice {
            layer: layer as u32, ..slice.clone()
        }).collect();
        let installed = InstalledModelAdmission::new(0, selection.clone(), Some(report.clone()))?;
        assert_eq!(installed.selection().layers(), vec![13, 15, 20]);
        let lease = installed.reserve_endpoint(Some(0))?.unwrap();
        let mut header = crate::ExpertProtocolV2RequestHeader {
            request_id: 1, placement_version: 1, layer_id: 13, row_count: 1,
            hidden_dim: 6144, hidden_dtype: crate::ExpertV2Dtype::Bf16,
            hidden_row_stride_bytes: 12288, route_count: 8, row_descriptor_bytes: 40,
            route_bytes: 12, hidden_payload_bytes: 12288, flags: 0,
        };
        for layer in [13, 15, 20] {
            header.layer_id = layer;
            lease.validate_request(&header)?;
        }
        for layer in [12, 14, 16, 19, 21] {
            header.layer_id = layer;
            assert!(lease.validate_request(&header).is_err());
        }
        report.slices[1].layer = 14;
        assert!(InstalledModelAdmission::new(0, selection, Some(report)).is_err());
        Ok(())
    }

    #[test]
    fn coordinator_endpoint_credit_survives_selection_replay() -> Result<()> {
        let selection = admission_selection();
        let report = admission_report(&selection);
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let peer = listener.local_addr()?;
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(|| -> Result<()> {
                let installed = InstalledModelAdmission::new(0, selection.clone(), Some(report))?;
                for _ in 0..2 {
                    let (mut stream, _) = listener.accept()?;
                    let received = receive_worker_selection(&stream, 0, Duration::from_secs(3))?;
                    ensure!(received == selection);
                    installed.acknowledge(&mut stream)?;
                }
                Ok(())
            });
            select_model_admission(&[peer], &selection, Duration::from_secs(3))?;
            let header = crate::ExpertProtocolV2RequestHeader {
                request_id: 1, placement_version: 1, layer_id: 13, row_count: 80,
                hidden_dim: 6144, hidden_dtype: crate::ExpertV2Dtype::Bf16,
                hidden_row_stride_bytes: 12288, route_count: 640, row_descriptor_bytes: 40,
                route_bytes: 12, hidden_payload_bytes: 80 * 12288, flags: 0,
            };
            let lease = reserve_peer_endpoint(peer, &header, Some(4096))?.unwrap();
            assert_eq!(lease.index(), 0);
            assert!(reserve_peer_endpoint(peer, &header, Some(4096)).is_err());
            select_model_admission(&[peer], &selection, Duration::from_secs(3))?;
            assert!(reserve_peer_endpoint(peer, &header, Some(4096)).is_err());
            drop(lease);
            assert!(reserve_peer_endpoint(peer, &header, Some(4096))?.is_some());
            assert!(reserve_peer_endpoint(peer, &header, Some(4097)).is_err());
            Ok(())
        })
    }

    #[test]
    fn step5_two_rank_clients_replay_admission_and_retained_endpoint_credit() -> Result<()> {
        use cuteafd_core::{SparkEndpoint, SparkIngress, ExpertGeometry};
        use std::sync::atomic::Ordering;
        let geometry = ExpertGeometry::STEP5;
        let endpoint = SparkEndpoint { capacity: 640, ingress: SparkIngress::Bf16 };
        let rings = admission_selection().model().unwrap().rings;
        for clients in [0, 9, usize::MAX] {
            assert!(ModelAdmission::for_clients(geometry, endpoint, 4096, rings, clients).is_err());
        }
        for clients in [1, 2] {
            let model = ModelAdmission::for_clients(geometry, endpoint, 4096, rings, clients)?;
            assert_eq!(model.ring_bytes(geometry)?, clients * 134217728);
            let selection = WorkerSelection::new(WorkerIdentity {
                checkpoint: "0".repeat(64), manifest: None,
                geometry: [4096, 352, 8, 1536, 91], world: 6,
                topology: [6, 1], ownership_wire: false,
            }, &(3..91).collect::<Vec<_>>())?.with_model(model)?;
            let mut reports = Vec::new();
            for rank in 0..6 {
                let mut report = admission_report(&selection);
                report.rank = rank;
                report.resident_bytes = 100965469184;
                report.runtime.workspace_bytes = 199584768;
                report.slices = (3..91).map(|layer| cuteafd_core::SparkSlice {
                    layer, first: rank * 256, width: 256, stored_width: 256,
                    tiers: vec![], projection_counts: vec![],
                }).collect();
                report.host_exchange_bytes = 640 * (4096 * 2 + 8 * 8);
                report.row_indices_bytes = 640 * 4;
                reports.push(report);
            }
            let listeners = (0..6).map(|_| std::net::TcpListener::bind("127.0.0.1:0"))
                .collect::<std::io::Result<Vec<_>>>()?;
            let peers = listeners.iter().map(|l| l.local_addr()).collect::<std::io::Result<Vec<_>>>()?;
            std::thread::scope(|scope| -> Result<()> {
                let mut servers = Vec::new();
                for (rank, (listener, report)) in listeners.into_iter().zip(reports).enumerate() {
                    let selection = selection.clone();
                    servers.push(scope.spawn(move || -> Result<()> {
                        let installed = InstalledModelAdmission::new(rank, selection.clone(), Some(report))?;
                        listener.set_nonblocking(true)?;
                        for _ in 0..2 {
                            let deadline = std::time::Instant::now() + Duration::from_secs(10);
                            let mut stream = loop {
                                match listener.accept() {
                                    Ok((stream, _)) => break stream,
                                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                                        ensure!(std::time::Instant::now() < deadline, "test admission accept timed out");
                                        std::thread::sleep(Duration::from_millis(10));
                                    }
                                    Err(error) => return Err(error.into()),
                                }
                            };
                            assert_eq!(receive_worker_selection(&stream, rank, Duration::from_secs(5))?, selection);
                            installed.acknowledge(&mut stream)?;
                        }
                        Ok(())
                    }));
                }
                select_model_admission(&peers, &selection, Duration::from_secs(5))?;
                assert_eq!(require_peer_clients(&peers, endpoint, 2).is_ok(), clients == 2);
                for peer in &peers {
                    assert_eq!(seals().lock().unwrap()[peer].endpoints.used.load(Ordering::Acquire), 0);
                }
                let mut header = crate::ExpertProtocolV2RequestHeader {
                    request_id: 4, placement_version: 17, layer_id: 3, row_count: 640,
                    hidden_dim: 4096, hidden_dtype: crate::ExpertV2Dtype::Bf16,
                    hidden_row_stride_bytes: 8192, route_count: 5120,
                    row_descriptor_bytes: 40, route_bytes: 12,
                    hidden_payload_bytes: 640 * 8192,
                    flags: crate::protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
                };
                let first = peers.iter().map(|&peer| reserve_peer_endpoint(peer, &header, None)
                    .map(|lease| lease.unwrap())).collect::<Result<Vec<_>>>()?;
                // Rank0's layer3 wave has completed, but its persistent QPs live.
                header.layer_id = 7;
                header.request_id = 8;
                select_model_admission(&peers, &selection, Duration::from_secs(5))?;
                for &peer in &peers {
                    let seals = seals().lock().unwrap();
                    let endpoints = &seals[&peer].endpoints;
                    assert_eq!(endpoints.used.load(Ordering::Acquire).count_ones(), 1);
                    validate_endpoint_request(&selection, 0, &header)?;
                }
                let mut second = Vec::new();
                for &peer in &peers {
                    match reserve_peer_endpoint(peer, &header, None) {
                        Ok(Some(lease)) => { assert_eq!(clients, 2); assert_eq!(lease.index(), 1); second.push(lease); }
                        Err(error) => {
                            assert_eq!(clients, 1);
                            assert!(error.to_string().contains("has no free ModelAdmission endpoint for capacity None / Bf16"));
                        }
                        _ => panic!("typed endpoint lease missing"),
                    }
                }
                for server in servers { server.join().unwrap()?; }
                drop(second);
                drop(first);
                for peer in peers { seals().lock().unwrap().remove(&peer); }
                Ok(())
            })?;
        }
        Ok(())
    }

    #[test]
    fn installed_report_replay_preserves_measured_bytes() -> Result<()> {
        let selection = admission_selection();
        let report = admission_report(&selection);
        let installed = InstalledModelAdmission::new(0, selection.clone(), Some(report.clone()))?;
        for _ in 0..2 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let mut client = TcpStream::connect(listener.local_addr()?)?;
            let (mut server, _) = listener.accept()?;
            installed.acknowledge_repeat(&mut server, serde_json::to_value(Start {
                message: "worker_selection_start".into(), rank: 0,
                selection: selection.clone(), digest: selection.digest()?,
            })?)?;
            let ready: Ready = read(&mut client)?;
            assert_eq!(ready.admission, Some(report.clone()));
        }
        let mut changed = selection.clone();
        changed.model.as_mut().unwrap().endpoints[0].ingress = cuteafd_core::SparkIngress::Fp8K32;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let _client = TcpStream::connect(listener.local_addr()?)?;
        let (mut server, _) = listener.accept()?;
        assert!(installed.acknowledge_repeat(&mut server, serde_json::to_value(Start {
            message: "worker_selection_start".into(), rank: 0,
            digest: changed.digest()?, selection: changed,
        })?).is_err());
        Ok(())
    }

    #[test]
    fn native_compiled_capacity_covers_live_but_does_not_admit_extra_rows() -> Result<()> {
        let mut selection = admission_selection();
        let model = selection.model.as_mut().unwrap();
        model.capacity = 2048;
        model.endpoints[0].capacity = 2048;
        selection.validate()?;
        let variants = [1, 80, 4096].into_iter().map(|capacity|
            InstalledExpertVariant::from_queries(capacity, |_| Ok(16))).collect::<Result<Vec<_>>>()?;
        let report = InstalledExpertGrouping::new(selection.clone(), 0, 2048,
            ExpertGroupingSelector::Native, variants.clone())?;
        assert!(report.group_rows(2048).is_ok());
        assert!(report.group_rows(2049).is_err());
        assert!(InstalledExpertGrouping::new(selection.clone(), 0, 4096,
            ExpertGroupingSelector::Native, variants.clone()).is_err());
        selection.model.as_mut().unwrap().compiled_capacity = 1024;
        assert!(selection.validate().is_err());
        selection.model.as_mut().unwrap().compiled_capacity = 2048;
        assert!(InstalledExpertGrouping::new(selection, 0, 2048,
            ExpertGroupingSelector::Native, variants).is_err());
        Ok(())
    }

    #[test]
    fn model_admission_capacity_equals_the_largest_endpoint() {
        let selection = admission_selection();
        let mut admission = selection.model().unwrap().clone();
        admission.validate(selection.geometry()).unwrap();
        admission.endpoints.push(cuteafd_core::SparkEndpoint {
            capacity: 80, ingress: cuteafd_core::SparkIngress::Bf16,
        });
        admission.validate(selection.geometry()).unwrap();
        admission.endpoints[0].capacity = 128;
        let error = admission.validate(selection.geometry()).unwrap_err().to_string();
        assert!(error.contains("capacity 4096"), "{error}");
        assert!(error.contains("endpoint capacity 128"), "{error}");
        admission.capacity = 128;
        admission.validate(selection.geometry()).unwrap();
        admission.capacity = 80;
        let error = admission.validate(selection.geometry()).unwrap_err().to_string();
        assert!(error.contains("capacity 80"), "{error}");
        assert!(error.contains("endpoint capacity 128"), "{error}");
    }

    #[test]
    fn model_admission_is_sealed_and_requires_an_exact_response() {
        let selection = admission_selection();
        let report = admission_report(&selection);
        validate_admission_response(&selection, 0, Some(&report)).unwrap();
        assert!(validate_admission_response(&selection, 0, None).is_err());
        assert!(validate_admission_response(&selection, 1, Some(&report)).is_err());
        let bare = WorkerSelection::new(WorkerIdentity { world: 1, topology: [1, 1], ..identity() }, &[13]).unwrap();
        assert_ne!(selection.digest().unwrap(), bare.digest().unwrap());
        assert!(validate_admission_response(&bare, 0, Some(&report)).is_err());
        for mutate in [0, 1, 2, 3] {
            let mut changed = report.clone();
            match mutate {
                0 => changed.mapped_ring_bytes += 1,
                1 => changed.slices[0].layer = 14,
                2 => changed.slices[0].stored_width = 128,
                _ => changed.slices[0].first = 1,
            }
            assert!(validate_admission_response(&selection, 0, Some(&changed)).is_err());
        }
    }

    #[test]
    fn selection_send_overlaps_load_and_wait_seals_only_valid_replies() -> Result<()> {
        let selection = admission_selection();
        let report = admission_report(&selection);
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let peer = listener.local_addr()?;
        let (loaded_tx, loaded_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| -> Result<()> {
            let worker_selection = &selection;
            let worker_report = &report;
            let worker = scope.spawn(move || -> Result<()> {
                let (mut stream, _) = listener.accept()?;
                let received = receive_worker_selection(&stream, 0, Duration::from_secs(3))?;
                // The coordinator can finish an independent load before Ready.
                ensure!(&received == worker_selection);
                loaded_rx.recv_timeout(Duration::from_secs(3))?;
                acknowledge_model_admission(&mut stream, 0, &received, Some(worker_report))
            });
            let pending = start_model_admission(&[peer], &selection, Duration::from_secs(3))?;
            assert!(peer_selection_digest(peer)?.is_none());
            loaded_tx.send(())?;
            assert_eq!(pending.wait()?, [report.clone()]);
            worker.join().unwrap()?;
            Ok(())
        })?;
        assert_eq!(peer_selection_digest(peer)?, Some(selection.digest()?));
        Ok(())
    }

    #[test]
    fn abandoned_selection_closes_control_without_waiting_for_ready() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let peer = listener.local_addr()?;
        let pending = start_model_admission(&[peer], &admission_selection(), Duration::from_secs(3))?;
        let (mut stream, _) = listener.accept()?;
        receive_worker_selection(&stream, 0, Duration::from_secs(3))?;
        drop(pending);
        let mut byte = [0];
        assert_eq!(stream.read(&mut byte)?, 0);
        assert!(peer_selection_digest(peer)?.is_none());
        Ok(())
    }

    #[test]
    fn malformed_ready_never_publishes_peer_seal() -> Result<()> {
        let selection = admission_selection();
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let peer = listener.local_addr()?;
        let pending = start_model_admission(&[peer], &selection, Duration::from_secs(3))?;
        let (mut stream, _) = listener.accept()?;
        receive_worker_selection(&stream, 0, Duration::from_secs(3))?;
        write(&mut stream, &Ready { message: "worker_selection_ready".into(), rank: 1,
            selection: selection.clone(), digest: selection.digest()?,
            admission: Some(admission_report(&selection)) })?;
        assert!(pending.wait().is_err());
        assert!(peer_selection_digest(peer)?.is_none());
        Ok(())
    }

    #[test]
    fn reduced_worker_endpoints_or_ring_report_fail_before_peer_seal() -> Result<()> {
        use cuteafd_core::{SparkEndpoint, SparkIngress};
        let mut selection = admission_selection();
        let geometry = selection.geometry();
        selection.model = Some(ModelAdmission::for_clients(geometry,
            SparkEndpoint { capacity: 640, ingress: SparkIngress::Bf16 }, 4096,
            selection.model().unwrap().rings, 2)?);
        for shrink_selection in [true, false] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let peer = listener.local_addr()?;
            let pending = start_model_admission(&[peer], &selection, Duration::from_secs(3))?;
            let (mut stream, _) = listener.accept()?;
            receive_worker_selection(&stream, 0, Duration::from_secs(3))?;
            let mut echoed = selection.clone();
            let mut report = admission_report(&selection);
            if shrink_selection {
                echoed.model.as_mut().unwrap().endpoints.pop();
                report.mapped_ring_bytes = echoed.model().unwrap().ring_bytes(geometry)? as u64;
            } else { report.mapped_ring_bytes /= 2; }
            write(&mut stream, &Ready { message: "worker_selection_ready".into(), rank: 0,
                digest: echoed.digest()?, selection: echoed, admission: Some(report) })?;
            assert!(pending.wait().is_err());
            assert!(peer_selection_digest(peer)?.is_none());
        }
        Ok(())
    }

    #[test]
    fn model_admission_returns_rank_reports_and_rejects_changed_reconnect() {
        let selection = admission_selection();
        let report = admission_report(&selection);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let peer = listener.local_addr().unwrap();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for changed in [false, false, true] {
                    let (mut stream, _) = listener.accept().unwrap();
                    let received = receive_worker_selection(&stream, 0, Duration::from_secs(3)).unwrap();
                    let mut response = report.clone();
                    if changed { response.runtime.workspace_bytes += 1; }
                    acknowledge_model_admission(&mut stream, 0, &received, Some(&response)).unwrap();
                }
            });
            for _ in 0..2 {
                let reports = select_model_admission(&[peer], &selection, Duration::from_secs(3)).unwrap();
                assert_eq!(reports, [report.clone()]);
            }
            assert!(select_model_admission(&[peer], &selection, Duration::from_secs(3)).is_err());
        });
    }

    #[test]
    fn installed_fp8_rank_read_bytes_allow_unequal_exact_slices_but_not_mixed_descriptors() {
        let report = |rank| {
            let loaded = WorkerSelection::new(identity(), &[13, 14, 15]).unwrap();
            let variant = InstalledExpertVariant::from_piecewise_queries(80,
                ExpertTrafficContract::ExactGrouping, |_| Ok(48)).unwrap();
            InstalledExpertGrouping::new(loaded, rank, 80, ExpertGroupingSelector::Fp8Nearest,
                vec![variant]).unwrap()
        };
        let left = report(0).with_weight_read_bytes(9_216_024).unwrap();
        let right = report(1).with_weight_read_bytes(6_144_024).unwrap();
        InstalledExpertGrouping::validate_ranks(&[left.clone(), right]).unwrap();
        let restored: InstalledExpertGrouping = serde_json::from_slice(&serde_json::to_vec(&left).unwrap()).unwrap();
        assert_eq!(restored.weight_read_bytes().unwrap(), 9_216_024);
        assert!(report(0).weight_read_bytes().is_err());
        assert!(report(0).with_weight_read_bytes(0).is_err());
        assert!(InstalledExpertGrouping::validate_ranks(&[left, report(1)]).is_err());
        assert!(grouping(0, 8).with_weight_read_bytes(10).is_err());
    }

    #[test]
    fn loaded_grouping_roundtrips_and_checks_identity_rank_range_and_capacity() {
        let report = grouping(0, 8);
        let restored: InstalledExpertGrouping = serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
        assert_eq!(report, restored);
        restored.validate_for(&identity(), 0, 128, 13..16).unwrap();
        for (rows, group) in [(0, 16), (1, 1), (8, 1), (9, 16), (80, 16), (81, 16), (256, 16)] {
            assert_eq!(restored.group_rows(rows).unwrap(), group);
        }
        assert!(restored.group_rows(257).is_err());
        assert!(restored.validate_for(&identity(), 1, 128, 13..16).is_err());
        assert!(restored.validate_for(&identity(), 0, 257, 13..16).is_err());
        assert!(restored.validate_for(&identity(), 0, 128, 12..16).is_err());
        let mut other = identity(); other.checkpoint = "1".repeat(64);
        assert!(restored.validate_for(&other, 0, 128, 13..16).is_err());
    }

    #[test]
    fn mixed_loaded_grouping_is_rejected_for_shared_layers() {
        InstalledExpertGrouping::validate_ranks(&[grouping(0, 8), grouping(1, 8)]).unwrap();
        let error = InstalledExpertGrouping::validate_ranks(&[grouping(0, 8), grouping(1, 4)]).unwrap_err();
        assert!(error.to_string().contains("mixed packages"));
        assert!(InstalledExpertGrouping::validate_ranks(&[grouping(0, 8), grouping(0, 8)]).is_err());
    }

    #[test]
    fn loaded_grouping_seal_rejects_changed_or_missing_reconnect_report() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let peer = listener.local_addr().unwrap();
        let original = grouping(0, 8);
        seal_installed_grouping(peer, Some(&original)).unwrap();
        seal_installed_grouping(peer, Some(&original)).unwrap();
        assert!(seal_installed_grouping(peer, Some(&grouping(0, 4))).is_err());
        assert!(seal_installed_grouping(peer, None).is_err());
    }

    #[test]
    fn loaded_grouping_checks_selector_and_compact_prefix() {
        assert!(InstalledExpertVariant::from_queries(80, |rows|
            Ok(if rows == 7 { 1 } else { 16 })).is_err());
        assert!(InstalledExpertVariant::from_queries(0, |_| Ok(16)).is_err());
        assert!(InstalledExpertVariant::from_queries(1, |_| Ok(3)).is_err());
        let loaded = WorkerSelection::new(identity(), &[13]).unwrap();
        let variants = [1, 16, 80].into_iter().map(|capacity|
            InstalledExpertVariant::from_queries(capacity, |_| Ok(capacity.min(64))).unwrap()).collect::<Vec<_>>();
        let report = InstalledExpertGrouping::new(loaded.clone(), 0, 16,
            ExpertGroupingSelector::Exl3GlmFlashK64, variants.iter().filter(|v| v.capacity != 16).cloned().collect()).unwrap();
        assert_eq!(report.group_rows(1).unwrap(), 1);
        assert_eq!(report.group_rows(2).unwrap(), 64);
        assert_eq!(report.group_rows(16).unwrap(), 64);
        let nearest = InstalledExpertGrouping::new(loaded.clone(), 0, 16,
            ExpertGroupingSelector::Exl3Nearest, variants.clone()).unwrap();
        assert_eq!(nearest.group_rows(0).unwrap(), 64);
        assert_eq!(nearest.group_rows(2).unwrap(), 16);
        assert_eq!(nearest.group_rows(16).unwrap(), 16);
        assert!(InstalledExpertGrouping::new(loaded, 0, 16,
            ExpertGroupingSelector::Exl3Nearest, variants[..2].to_vec()).is_err());
    }

    #[test]
    fn fp8_loaded_capacities_and_input_forms_follow_piecewise_groups() {
        let loaded = WorkerSelection::new(identity(), &[13]).unwrap();
        let variants = [1, 16, 80, 4096].into_iter().map(|capacity|
            InstalledExpertVariant::from_piecewise_queries(capacity, ExpertTrafficContract::ExactGrouping,
                |rows| Ok(if capacity == 1 || capacity == 16 { 16 }
                    else if capacity == 80 || (1..=80).contains(&rows) { 64 } else { 128 })).unwrap()).collect();
        let bf16 = [16, 4096].into_iter().map(|capacity|
            InstalledExpertVariant::from_piecewise_queries(capacity, ExpertTrafficContract::ExactGrouping,
                |_| Ok(if capacity == 16 { 16 } else { 64 })).unwrap()).collect();
        let report = InstalledExpertGrouping::new(loaded, 0, 128, ExpertGroupingSelector::Fp8Nearest, variants)
            .unwrap().with_bf16_variants(bf16).unwrap();
        let restored: InstalledExpertGrouping = serde_json::from_slice(&serde_json::to_vec(&report).unwrap()).unwrap();
        for (rows, expected) in [(1, 16), (16, 16), (17, 64), (80, 64), (81, 128), (128, 128)] {
            assert_eq!(restored.group_rows(rows).unwrap(), expected);
        }
        assert_eq!(restored.group_rows_for_input(81, true).unwrap(), 64);
        assert_eq!(restored.traffic_contract(), ExpertTrafficContract::ExactGrouping);
    }

    #[test]
    fn fp8_grouping_rejects_malformed_segments_and_companions_at_admission() {
        let loaded = WorkerSelection::new(identity(), &[13]).unwrap();
        let variant = InstalledExpertVariant::from_piecewise_queries(80,
            ExpertTrafficContract::ExactGrouping, |_| Ok(48)).unwrap();
        let report = InstalledExpertGrouping::new(loaded, 0, 80,
            ExpertGroupingSelector::Fp8Nearest, vec![variant]).unwrap();
        assert_eq!(report.group_rows(17).unwrap(), 48);
        for segments in [
            serde_json::json!([{"end_rows":79,"group_rows":48}]),
            serde_json::json!([{"end_rows":80,"group_rows":0}]),
            serde_json::json!([{"end_rows":80,"group_rows":16},{"end_rows":80,"group_rows":48}]),
        ] {
            let mut value = serde_json::to_value(&report).unwrap();
            value["variants"][0]["segments"] = segments;
            let malformed: InstalledExpertGrouping = serde_json::from_value(value).unwrap();
            assert!(malformed.validate_for(&identity(), 0, 80, 13..14).is_err());
        }
        let mut value = serde_json::to_value(&report).unwrap();
        value["variants"][0]["contract"] = "conservative_route_upper_bound".into();
        let malformed: InstalledExpertGrouping = serde_json::from_value(value).unwrap();
        assert!(malformed.validate_for(&identity(), 0, 80, 13..14).is_err());
        let companion = InstalledExpertVariant::from_piecewise_queries(16,
            ExpertTrafficContract::ExactGrouping, |_| Ok(16)).unwrap();
        assert!(report.with_bf16_variants(vec![companion]).is_err());
    }

    /// An NVFP4 package's workers install one FP8 family of tiles and report no
    /// BF16 siblings, while the checkpoint's Spark wire is BF16 (the workers
    /// quantize the activations themselves). The report must answer that
    /// package's own wire form from the variants it did install, so a planner
    /// pricing a BF16 request is not left without a grouping.
    #[test]
    fn nvfp4_report_without_bf16_siblings_answers_the_bf16_wire() {
        let loaded = WorkerSelection::new(identity(), &[13]).unwrap();
        let variants = [1, 16, 80, 4096].into_iter().map(|capacity|
            InstalledExpertVariant::from_piecewise_queries(capacity, ExpertTrafficContract::ExactGrouping,
                |rows| Ok(if capacity <= 16 { 16 }
                    else if capacity == 80 || (1..=80).contains(&rows) { 64 } else { 128 })).unwrap()).collect();
        let report = InstalledExpertGrouping::new(loaded, 0, 128, ExpertGroupingSelector::Fp8Nearest, variants).unwrap();
        assert!(report.bf16_variants.is_none() && report.validate().is_ok());
        for (rows, expected) in [(1, 16), (16, 16), (17, 64), (80, 64), (81, 128), (128, 128)] {
            assert_eq!(report.group_rows(rows).unwrap(), expected, "fp8 rows {rows}");
            assert_eq!(report.group_rows_for_input(rows, true).unwrap(), expected, "bf16 rows {rows}");
        }
    }

    #[test]
    fn nvfp4_installed_bound_admits_without_claiming_an_exact_tile() {
        let loaded = WorkerSelection::new(identity(), &[13]).unwrap();
        let variant = InstalledExpertVariant::from_piecewise_queries(4096,
            ExpertTrafficContract::ConservativeRouteUpperBound, |_| Ok(1)).unwrap();
        let report = InstalledExpertGrouping::new(loaded, 0, 128, ExpertGroupingSelector::Fp8Nearest,
            vec![variant]).unwrap();
        report.validate_for(&identity(), 0, 128, 13..14).unwrap();
        assert_eq!(report.group_rows(128).unwrap(), 1);
        assert_eq!(report.traffic_contract().label(), "conservative_route_upper_bound");
        assert!(InstalledExpertVariant::from_piecewise_queries(16,
            ExpertTrafficContract::ConservativeRouteUpperBound, |_| Ok(16)).is_err());
    }

    #[test]
    fn worker_selection_validates_absolute_ranges_and_identity() {
        let selection = WorkerSelection::new(identity(), &[13, 14, 15, 20]).unwrap();
        assert_eq!(selection.ranges(), &[LayerRange { first: 13, end: 16 }, LayerRange { first: 20, end: 21 }]);
        assert_eq!(selection.layers(), vec![13, 14, 15, 20]);
        for invalid in [vec![70], vec![14, 13], vec![13, 13]] { assert!(WorkerSelection::new(identity(), &invalid).is_err()); }
        let mut other = identity(); other.checkpoint = "1".repeat(64);
        assert!(selection.validate_for(&other).is_err());
        assert_ne!(selection.digest().unwrap(), WorkerSelection::new(identity(), &[14, 15, 20]).unwrap().digest().unwrap());
        assert!(WorkerSelection::new(identity(), &[]).unwrap().ranges().is_empty());
        let mut invalid = identity(); invalid.world = 9;
        assert!(WorkerSelection::new(invalid, &[]).is_err());
        let mut invalid = identity(); invalid.topology = [2, 2];
        assert!(WorkerSelection::new(invalid, &[]).is_err());
        assert!(WorkerSelection::new(identity(), &[u32::MAX as usize - 1, u32::MAX as usize]).is_err());
    }
    #[test]
    fn worker_identity_requires_topology_and_wire_without_compatibility_defaults() {
        let encoded = serde_json::to_value(identity()).unwrap();
        for field in ["topology", "ownership_wire"] {
            let mut missing = encoded.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<WorkerIdentity>(missing).is_err());
        }
        let mut null = encoded;
        null["topology"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<WorkerIdentity>(null).is_err());
    }

    #[test]
    fn worker_identity_seals_wire_and_physical_topology_independently() {
        let canonical = WorkerSelection::new(identity(), &[13]).unwrap();
        let mut owned = identity();
        owned.ownership_wire = true;
        assert!(canonical.validate_for(&owned).is_err());
        assert_ne!(canonical.digest().unwrap(), WorkerSelection::new(owned, &[13]).unwrap().digest().unwrap());
        let canonical_four = WorkerIdentity { world: 4, topology: [4, 1], ..identity() };
        let canonical = WorkerSelection::new(canonical_four.clone(), &[13]).unwrap();
        let replicated = WorkerIdentity { topology: [2, 2], ownership_wire: true, ..canonical_four };
        assert!(canonical.validate_for(&replicated).is_err());
        assert!(WorkerSelection::new(replicated, &[13]).is_ok());
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
    fn fabric_control_is_served_without_worker_or_native_admission() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        write(&mut client, &serde_json::json!({"message": "fabric_traffic"})).unwrap();
        let error = receive_worker_selection(&server, 0, Duration::from_secs(3)).unwrap_err();
        assert!(error.is::<crate::traffic::TrafficProbeOnly>());
        let reply: serde_json::Value = read(&client).unwrap();
        assert_eq!(reply["message"], "fabric_traffic");
        assert!(reply["sample"].is_object());
    }
}
