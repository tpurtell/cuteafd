//! Spark owners: manifests describe execution, checkpoint selection describes weights.
use crate::placement::SparkAssignment;
use cuteafd_core::memory_layout::{Basis, Category, Item};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("V4.1 Spark inventory: {0}")]
pub struct SparkInventoryError(String);
type Result<T> = std::result::Result<T, SparkInventoryError>;
fn invalid(message: impl Into<String>) -> SparkInventoryError { SparkInventoryError(message.into()) }
fn add(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0u64, |a, b| a.checked_add(b).ok_or_else(|| invalid("byte sum overflow")))
}
fn mul(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |a, b| a.checked_mul(*b).ok_or_else(|| invalid("byte product overflow")))
}
fn number(value: &Value, key: &str) -> Result<u64> {
    value[key].as_u64().ok_or_else(|| invalid(format!("missing unsigned {key}")))
}
fn item(category: Category, name: &str, bytes: u64, basis: Basis) -> Item {
    Item::new(category, name, "", bytes, basis)
}

/// Every manifest is fingerprinted, including its object hashes and capacity state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SparkManifestProvenance {
    sha256: String,
    revision: String,
    capacity_rows: Vec<u32>,
    physical_sms: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SparkMemoryDomains {
    device: Vec<Item>,
    pinned: Vec<Item>,
    host: Vec<Item>,
}
impl SparkMemoryDomains {
    pub fn device(&self) -> &[Item] { &self.device }
    pub fn pinned(&self) -> &[Item] { &self.pinned }
    pub fn host(&self) -> &[Item] { &self.host }
    pub fn total(&self) -> Result<u64> {
        add(self.device.iter().chain(&self.pinned).chain(&self.host).map(|i| i.bytes))
    }
}

/// Actual loader high-water owners, not a multiple of resident weights. Maxima
/// are across sequential layer loads; simultaneous banks/read lanes are summed
/// by the checkpoint loader before constructing this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SparkLoadInventory {
    device_staging_bytes: u64,
    pinned_staging_bytes: u64,
    host_read_scratch_bytes: u64,
    source: String,
}

impl SparkLoadInventory {
    pub fn device_staging_bytes(&self) -> u64 { self.device_staging_bytes }
    pub fn pinned_staging_bytes(&self) -> u64 { self.pinned_staging_bytes }
    pub fn host_read_scratch_bytes(&self) -> u64 { self.host_read_scratch_bytes }
    pub fn source(&self) -> &str { &self.source }
    pub fn native(catalog: &crate::OfficialV41Catalog, selections: &[crate::V41ExpertSelection]) -> Result<Self> {
        if selections.is_empty() { return Err(invalid("no selected native layer load plans")); }
        let mut staging = 0u64;
        let mut scratch = 0u64;
        for &selection in selections {
            let (bytes, read) = if catalog.nvfp4().is_some() {
                let plan = catalog.nvfp4_expert_staging(selection).map_err(|e| invalid(e.to_string()))?;
                (plan.staging_bytes(), plan.minimum_read_scratch_bytes())
            } else {
                let plan = catalog.expert_staging(selection).map_err(|e| invalid(e.to_string()))?;
                (plan.staging_bytes(), plan.minimum_read_scratch_bytes())
            };
            staging = staging.max(bytes as u64);
            scratch = scratch.max(read as u64);
        }
        Ok(Self { device_staging_bytes: staging, pinned_staging_bytes: mul(&[staging, 16])?,
            host_read_scratch_bytes: mul(&[scratch, 64, 16])?,
            source: "checkpoint staging plans; 16 reader lanes, 64 source rows per pread; sequential layers".into() })
    }

    pub fn exl3(catalog: &crate::OfficialV41Catalog, plans: &[crate::V41Exl3Residency]) -> Result<Self> {
        if plans.is_empty() { return Err(invalid("no selected EXL3 layer load plans")); }
        let mut pinned = 0u64;
        let mut read = 0u64;
        for plan in plans {
            if plan.loads.len() != plan.experts.checked_mul(9).ok_or_else(|| invalid("EXL3 load count overflow"))? {
                return Err(invalid("EXL3 staging inventory mismatch"));
            }
            let staging = plan.loads.chunks(9).map(|jobs| add(jobs.iter().map(|j| j.bytes as u64)))
                .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
            let init = plan.buffers.iter().map(|b| mul(&[b.initial_words.len() as u64, 4]))
                .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
            pinned = pinned.max(mul(&[staging.max(init), 16, 2])?);
            read = read.max(mul(&[plan.scratch_bytes(catalog).map_err(|e| invalid(e.to_string()))? as u64, 64, 16, 2])?);
        }
        Ok(Self { device_staging_bytes: 0, pinned_staging_bytes: pinned, host_read_scratch_bytes: read,
            source: "checkpoint EXL3 residency jobs; 9 jobs per expert, 16 reader lanes, 2 banks, 64 rows per pread; sequential layers".into() })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
/// Validated installed-package proof. Not deserializable or publicly writable.
///
/// ```compile_fail
/// use cuteafd_loader::placement::inventory::SparkExecutionInventory;
/// let _: SparkExecutionInventory = serde_json::from_str("{}").unwrap();
/// ```
pub struct SparkExecutionInventory {
    compiled_capacity: u32,
    live_capacity: u32,
    retained_capacities: Vec<u32>,
    physical_sms: u32,
    logical_intermediate: u32,
    stored_intermediate: u32,
    manifests: Vec<SparkManifestProvenance>,
    retained: SparkMemoryDomains,
    load: SparkLoadInventory,
}

fn capacities(compiled: u32, live: u32, retained: &[u32]) -> Result<()> {
    if live == 0 || compiled < live || compiled > 4096 || retained.is_empty()
        || retained.iter().any(|&r| r == 0 || r > compiled)
        || retained.iter().copied().collect::<BTreeSet<_>>().len() != retained.len()
        || !retained.contains(&compiled) {
        return Err(invalid("unsupported compiled/live/retained capacity state"));
    }
    Ok(())
}
fn provenance(manifest: &Value, rows: Vec<u32>, sms: u32) -> Result<SparkManifestProvenance> {
    let revision = manifest["sparkinfer_revision"].as_str().filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("missing package revision"))?;
    Ok(SparkManifestProvenance { sha256: format!("{:x}", Sha256::digest(
        serde_json::to_vec(manifest).map_err(|e| invalid(e.to_string()))?)),
        revision: revision.into(), capacity_rows: rows, physical_sms: sms })
}

impl SparkExecutionInventory {
    pub fn compiled_capacity(&self) -> u32 { self.compiled_capacity }
    pub fn live_capacity(&self) -> u32 { self.live_capacity }
    pub fn retained_capacities(&self) -> &[u32] { &self.retained_capacities }
    pub fn physical_sms(&self) -> u32 { self.physical_sms }
    pub fn logical_intermediate(&self) -> u32 { self.logical_intermediate }
    pub fn stored_intermediate(&self) -> u32 { self.stored_intermediate }
    pub fn manifests(&self) -> &[SparkManifestProvenance] { &self.manifests }
    pub fn retained(&self) -> &SparkMemoryDomains { &self.retained }
    pub fn load(&self) -> &SparkLoadInventory { &self.load }

    /// Native Spark owns separate main, serial and grouped-decode arenas. The
    /// compact reducer owns code/handles, not another device scratch arena.
    pub fn native(manifest: &Value, physical_sms: u32, compiled: u32, live: u32,
        retained: &[u32], load: SparkLoadInventory) -> Result<Self> {
        capacities(compiled, live, retained)?;
        if manifest["schema"] != 1 || manifest["capability"] != serde_json::json!([12, 1])
            || physical_sms == 0 || number(manifest, "physical_sms")? != physical_sms as u64 {
            return Err(invalid("native package schema/architecture/physical SM mismatch"));
        }
        let tp = match manifest["role"].as_str() {
            Some("spark") => 4, Some("spark_tp2") => 2,
            Some("spark_tp3") => 3, Some("spark_tp6") => 6,
            _ => return Err(invalid("not a supported native Spark role")),
        };
        if number(manifest, "spark_tp_degree")? != tp {
            return Err(invalid("native role/TP degree mismatch"));
        }
        let geometry = &manifest["geometry"];
        let logical = number(geometry, "intermediate")?;
        let stored = number(geometry, "kernel_intermediate")?;
        if number(geometry, "hidden")? != 5120 || number(geometry, "experts")? != 384
            || number(geometry, "topk")? != 6 || logical != 2304 / tp
            || stored < logical || stored > 2304 || stored % 32 != 0 {
            return Err(invalid("native package is not the selected V4.1 Spark geometry"));
        }
        let mut expected = vec![compiled];
        if compiled > 1 { expected.push(1); }
        if compiled > 80 { expected.push(80); }
        if retained != expected { return Err(invalid("native retained state must be main, m1, m80 in owner order")); }
        let input = match manifest["input_format"].as_str() {
            Some("bf16") => 5120 * 2, Some("fp8_k32") => 5120 + 5120 / 32,
            _ => return Err(invalid("unsupported native Spark ingress")),
        };
        let variants = manifest["variants"].as_array().ok_or_else(|| invalid("missing native variants"))?;
        let mut seen = BTreeSet::new();
        for variant in variants {
            if !seen.insert(number(variant, "capacity_rows")?) { return Err(invalid("duplicate native capacity")); }
        }
        let mut device = Vec::new();
        for &rows in retained {
            let variant = variants.iter().find(|v| v["capacity_rows"] == rows)
                .ok_or_else(|| invalid(format!("missing native capacity {rows}")))?;
            device.push(item(Category::Workspace, &format!("expert_scratch_m{rows}"),
                number(variant, "core_scratch_nbytes")?, Basis::Exact));
        }
        device.push(item(Category::Workspace, "expert_hidden", mul(&[live as u64, input])?, Basis::Formula));
        device.push(item(Category::Workspace, "expert_routes", mul(&[live as u64, 6, 8])?, Basis::Formula));
        device.push(item(Category::Workspace, "expert_compact_output", mul(&[live as u64, 5120, 2])?, Basis::Formula));
        // Reducer function handles carry no separately allocated device payload.
        device.push(item(Category::Workspace, "compact_reducer_scratch", 0, Basis::Exact));
        Ok(Self { compiled_capacity: compiled, live_capacity: live, retained_capacities: retained.into(),
            physical_sms, logical_intermediate: logical as u32, stored_intermediate: stored as u32,
            manifests: vec![provenance(manifest, retained.into(), physical_sms)?],
            retained: SparkMemoryDomains { device, pinned: vec![item(Category::Workspace,
                "expert_route_staging", mul(&[live as u64, 6, 8])?, Basis::Formula)], host: vec![] }, load })
    }

    /// The selected schedule's actual capacity manifests, not a coordinator
    /// export. The existing helper preserves alias/shared/private LUT semantics.
    pub fn exl3(manifests: &[Value], physical_sms: u32, compiled: u32, live: u32,
        retained: &[u32], load: SparkLoadInventory) -> Result<Self> {
        capacities(compiled, live, retained)?;
        let expected = cuteafd_core::Exl3RowPolicy::Nearest.capacities(live as usize)
            .map_err(|e| invalid(e.to_string()))?;
        if retained != expected || expected.last() != Some(&compiled) {
            return Err(invalid("EXL3 retained states differ from V4.1 row policy"));
        }
        if physical_sms == 0 || manifests.len() != retained.len() {
            return Err(invalid("EXL3 manifest/retained state count mismatch"));
        }
        let mut sources = Vec::new();
        let mut width = None;
        for (manifest, &rows) in manifests.iter().zip(retained) {
            let intermediate = number(manifest, "intermediate")?;
            if manifest["schema"] != "cuteafd.v41-exl3-aot.v1" || manifest["output_dtype"] != "bf16"
                || number(manifest, "hidden")? != 5120 || number(manifest, "experts")? != 384
                || number(manifest, "top_k")? != 6 || number(manifest, "sms")? != physical_sms as u64
                || number(manifest, "capacity")? != rows as u64 || intermediate == 0
                || intermediate > 2304 || intermediate % 128 != 0
                || width.is_some_and(|w| w != intermediate) {
                return Err(invalid("EXL3 package geometry/capacity/physical SM mismatch"));
            }
            width = Some(intermediate);
            sources.push(provenance(manifest, vec![rows], physical_sms)?);
        }
        let arena = crate::serving_capacity::exl3_workspace_bytes(manifests, true)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self { compiled_capacity: compiled, live_capacity: live, retained_capacities: retained.into(),
            physical_sms, logical_intermediate: width.unwrap() as u32, stored_intermediate: width.unwrap() as u32,
            manifests: sources, retained: SparkMemoryDomains {
                device: vec![item(Category::Workspace, "exl3_retained_capacity_arenas", arena, Basis::Exact),
                    item(Category::Workspace, "expert_hidden", mul(&[live as u64, 5280])?, Basis::Formula),
                    item(Category::Workspace, "expert_routes", mul(&[live as u64, 6, 8])?, Basis::Formula)],
                pinned: vec![item(Category::Workspace, "expert_route_staging", mul(&[live as u64, 6, 8])?, Basis::Formula)],
                host: vec![] }, load })
    }
}

/// Calibration uses isolated phases and the explicit device-kind witness policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SparkRuntimeProvenance {
    source: String,
    method: SparkRuntimeMethod,
    phase: String,
    pid: u32,
    quiet_host: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SparkRuntimeMethod {
    ProcessAttributedPhaseDelta,
    IntegratedSystemPhaseDelta,
    MeasuredResidual,
}
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SparkRuntimeTerm {
    item: Item,
    provenance: SparkRuntimeProvenance,
}
#[derive(Debug, Clone, PartialEq, Serialize)]
/// Runtime proof can only be constructed through validation, not deserialization.
///
/// ```compile_fail
/// use cuteafd_loader::placement::inventory::QualifiedSparkRuntime;
/// let _: QualifiedSparkRuntime = serde_json::from_str("{}").unwrap();
/// ```
pub struct QualifiedSparkRuntime {
    arch: String,
    driver: String,
    physical_sms: u32,
    /// Fingerprints of the selected capacity packages, not just family name.
    manifest_sha256: Vec<String>,
    /// Admission identity and lazy host owners bound to this sample, not subtracted
    /// from its resident growth: zero-filled Vec storage is not prefaulted.
    rank: u32,
    live_capacity: u32,
    host_owners: SparkHostOwners,
    device: Vec<SparkRuntimeTerm>,
    pinned: Vec<SparkRuntimeTerm>,
    host: Vec<SparkRuntimeTerm>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct SparkHostOwners {
    exchange: u64,
    row_indices: u64,
    retained: Vec<Item>,
}
impl SparkHostOwners {
    fn new(assignment: &SparkAssignment, execution: &SparkExecutionInventory) -> Self {
        Self { exchange: assignment.io.host_exchange, row_indices: assignment.io.row_indices,
            retained: execution.retained.host.clone() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SparkRankInventory {
    rank: u32,
    /// Serving owners; domains are disjoint even on GB10's unified pool.
    device: Vec<Item>,
    pinned: Vec<Item>,
    host: Vec<Item>,
    /// Additional load-only owners. They do not coexist with execution/rings.
    load_transient: SparkMemoryDomains,
    /// Full independent unified-memory peaks, including resident and runtime.
    load_peak: u64,
    serve_peak: u64,
    manifests: Vec<SparkManifestProvenance>,
    runtime_provenance: Vec<SparkRuntimeProvenance>,
    load_provenance: String,
}

impl SparkManifestProvenance {
    pub fn sha256(&self) -> &str { &self.sha256 }
    pub fn revision(&self) -> &str { &self.revision }
    pub fn capacity_rows(&self) -> &[u32] { &self.capacity_rows }
    pub fn physical_sms(&self) -> u32 { self.physical_sms }
}

impl SparkRuntimeProvenance {
    fn new(source: String, method: SparkRuntimeMethod, phase: String, pid: u32,
        quiet_host: bool) -> Result<Self> {
        if source.trim().is_empty() || phase.trim().is_empty() || pid == 0 || !quiet_host {
            return Err(invalid("runtime provenance requires a quiet same-process phase source"));
        }
        Ok(Self { source, method, phase, pid, quiet_host })
    }
    pub fn source(&self) -> &str { &self.source }
    pub fn method(&self) -> SparkRuntimeMethod { self.method }
    pub fn phase(&self) -> &str { &self.phase }
    pub fn pid(&self) -> u32 { self.pid }
    pub fn quiet_host(&self) -> bool { self.quiet_host }
}

impl SparkRuntimeTerm {
    fn new(item: Item, provenance: SparkRuntimeProvenance) -> Result<Self> {
        if item.basis != Basis::Calibrated || item.category != Category::Runtime
            || item.group.trim().is_empty()
            || (provenance.method == SparkRuntimeMethod::MeasuredResidual
                && !item.group.contains("measured_residual")) {
            return Err(invalid("runtime terms require calibrated runtime owners; residuals must be named measured_residual"));
        }
        Ok(Self { item, provenance })
    }
    pub fn item(&self) -> &Item { &self.item }
    pub fn provenance(&self) -> &SparkRuntimeProvenance { &self.provenance }
}

fn census_residual(before: &cuteafd_core::expert_geometry::SparkProcessSample,
    ready: &cuteafd_core::expert_geometry::SparkProcessSample, context: u64, modules: u64) -> Result<u64> {
    let total_growth = before.system_available_bytes.checked_sub(ready.system_available_bytes)
        .ok_or_else(|| invalid("system memory grew across census"))?;
    let tracked_device = ready.tracked_device_bytes.checked_sub(before.tracked_device_bytes)
        .ok_or_else(|| invalid("device payload ledger decreased"))?;
    let tracked_pinned = ready.tracked_pinned_bytes.checked_sub(before.tracked_pinned_bytes)
        .ok_or_else(|| invalid("pinned payload ledger decreased"))?;
    // Lazy zero-filled host owners are charged separately at serving peak.
    // They are not known resident at this sample, so cannot reduce its residual.
    total_growth.checked_sub(add([context, modules, tracked_device, tracked_pinned])?)
        .ok_or_else(|| invalid("system witness below attributed owners; cannot form disjoint residual"))
}

impl QualifiedSparkRuntime {
    /// Build only from isolated same-process phase evidence and externally
    /// established quiet-host conditions and device-kind-specific witnesses.
    pub fn from_census(assignment: &SparkAssignment, execution: &SparkExecutionInventory,
        sample: &cuteafd_core::RuntimeSample,
        manifest_sha256: Vec<String>, samples: &[cuteafd_core::expert_geometry::SparkProcessSample],
        source: String, quiet_host: bool) -> Result<Self> {
        use cuteafd_core::expert_geometry::{spark_census_deltas, CensusDeviceWitness};
        if !quiet_host || source.trim().is_empty() { return Err(invalid("quiet-host census source required")); }
        let deltas = spark_census_deltas(samples).map_err(|error| invalid(error.to_string()))?;
        let context_before = samples.iter().find(|s| s.phase == "before_context").unwrap();
        let ready = samples.iter().find(|s| s.phase == "steady_after_first_expert_request").unwrap();
        let context = deltas.context;
        let modules = deltas.modules;
        let (method, witness) = match deltas.witness {
            CensusDeviceWitness::SystemAvailable => (SparkRuntimeMethod::IntegratedSystemPhaseDelta,
                "integrated device: isolated MemAvailable (cgroup is an admission limit); retained ledger/system control within 5%; host-only maps/system within 16 MiB"),
            CensusDeviceWitness::CudaMappingPss => (SparkRuntimeMethod::ProcessAttributedPhaseDelta,
                "discrete device: CUDA mapping PSS + system phase within 16 MiB; retained ledger/maps control within 5%"),
        };
        let residual = census_residual(context_before, ready, context, modules)?;
        let term = |name: &str, bytes, method, phase: &str| -> Result<SparkRuntimeTerm> {
            SparkRuntimeTerm::new(item(Category::Runtime, name, bytes, Basis::Calibrated),
                SparkRuntimeProvenance::new(source.clone(), method, phase.into(), context_before.pid, quiet_host)?)
        };
        Self::new(assignment, execution, sample, manifest_sha256,
            vec![term("cuda_context", context, method,
                    &format!("before_context/after_context_before_media: {witness}"))?,
                term("expert_modules", modules, method,
                    &format!("before_expert_modules/after_expert_modules_before_weights: {witness}"))?],
            vec![], vec![term("host_measured_residual", residual, SparkRuntimeMethod::MeasuredResidual,
                "before_context/steady_after_first_expert_request: system growth less tracked payload/context/modules; lazy host owners not subtracted")?])
    }

    fn new(assignment: &SparkAssignment, execution: &SparkExecutionInventory,
        sample: &cuteafd_core::RuntimeSample, manifest_sha256: Vec<String>,
        device: Vec<SparkRuntimeTerm>, pinned: Vec<SparkRuntimeTerm>, host: Vec<SparkRuntimeTerm>) -> Result<Self> {
        let runtime = Self { arch: sample.arch.clone(), driver: sample.driver.clone(), physical_sms: sample.sms,
            rank: assignment.rank, live_capacity: execution.live_capacity,
            host_owners: SparkHostOwners::new(assignment, execution), manifest_sha256, device, pinned, host };
        validate_runtime(execution, &runtime)?;
        Ok(runtime)
    }
    pub fn arch(&self) -> &str { &self.arch }
    pub fn driver(&self) -> &str { &self.driver }
    pub fn physical_sms(&self) -> u32 { self.physical_sms }
    pub fn manifest_sha256(&self) -> &[String] { &self.manifest_sha256 }
    pub fn rank(&self) -> u32 { self.rank }
    pub fn live_capacity(&self) -> u32 { self.live_capacity }
    pub fn device(&self) -> &[SparkRuntimeTerm] { &self.device }
    pub fn pinned(&self) -> &[SparkRuntimeTerm] { &self.pinned }
    pub fn host(&self) -> &[SparkRuntimeTerm] { &self.host }
}

impl SparkRankInventory {
    pub fn rank(&self) -> u32 { self.rank }
    pub fn device(&self) -> &[Item] { &self.device }
    pub fn pinned(&self) -> &[Item] { &self.pinned }
    pub fn host(&self) -> &[Item] { &self.host }
    pub fn load_transient(&self) -> &SparkMemoryDomains { &self.load_transient }
    pub fn load_peak(&self) -> u64 { self.load_peak }
    pub fn serve_peak(&self) -> u64 { self.serve_peak }
    pub fn manifests(&self) -> &[SparkManifestProvenance] { &self.manifests }
    pub fn runtime_provenance(&self) -> &[SparkRuntimeProvenance] { &self.runtime_provenance }
    pub fn load_provenance(&self) -> &str { &self.load_provenance }
}

fn validate_runtime(execution: &SparkExecutionInventory, runtime: &QualifiedSparkRuntime) -> Result<()> {
    if runtime.arch != "sm_121" || runtime.driver.trim().is_empty()
        || runtime.physical_sms != execution.physical_sms {
        return Err(invalid("runtime architecture/driver/physical SM mismatch"));
    }
    let fingerprints: Vec<_> = execution.manifests.iter().map(|m| m.sha256.clone()).collect();
    if fingerprints != runtime.manifest_sha256 { return Err(invalid("runtime calibrated for different package manifests")); }
    let all: Vec<_> = runtime.device.iter().chain(&runtime.pinned).chain(&runtime.host).collect();
    for term in &all {
        let p = &term.provenance;
        if p.source.is_empty() || p.phase.is_empty() || p.pid == 0 || !p.quiet_host
            || term.item.basis != Basis::Calibrated || term.item.category != Category::Runtime
            || (p.method == SparkRuntimeMethod::MeasuredResidual && !term.item.group.contains("measured_residual")) {
            return Err(invalid("unqualified runtime term; residuals must be named measured_residual"));
        }
    }
    for domain in [&runtime.device, &runtime.pinned, &runtime.host] {
        let mut groups = BTreeSet::new();
        if domain.iter().any(|term| !groups.insert(&term.item.group)) {
            return Err(invalid("duplicate runtime owner in one memory domain"));
        }
    }
    if !runtime.device.iter().any(|t| t.item.group == "cuda_context")
        || !runtime.device.iter().any(|t| t.item.group == "expert_modules")
        || !runtime.host.iter().any(|t| t.item.group.contains("measured_residual")
            && t.provenance.method == SparkRuntimeMethod::MeasuredResidual) {
        return Err(invalid("qualified context, modules and physical host residual are required"));
    }
    if runtime.device.iter().any(|t| matches!(t.item.group.as_str(), "cuda_context" | "expert_modules")
        && !matches!(t.provenance.method, SparkRuntimeMethod::ProcessAttributedPhaseDelta | SparkRuntimeMethod::IntegratedSystemPhaseDelta)) {
        return Err(invalid("context/modules require isolated device-kind-qualified phase deltas"));
    }
    if runtime.device.iter().filter(|t| matches!(t.item.group.as_str(), "cuda_context" | "expert_modules"))
        .map(|t| t.provenance.method).collect::<Vec<_>>().windows(2).any(|p| p[0] != p[1]) {
        return Err(invalid("context/modules use different device witnesses"));
    }
    add(all.iter().map(|t| t.item.bytes))?;
    Ok(())
}

pub fn v41_spark_inventory(assignment: &SparkAssignment, execution: &SparkExecutionInventory,
    runtime: &QualifiedSparkRuntime, sample: &cuteafd_core::RuntimeSample) -> Result<SparkRankInventory> {
    capacities(execution.compiled_capacity, execution.live_capacity, &execution.retained_capacities)?;
    if runtime.arch != "sm_121" || runtime.driver.is_empty() || runtime.physical_sms != execution.physical_sms
        || execution.load.source.is_empty() || assignment.slices.is_empty()
        || assignment.slices.iter().any(|s| s.width != execution.logical_intermediate
            || s.stored_width != execution.stored_intermediate) {
        return Err(invalid("assignment/execution/runtime qualification mismatch (idle rank needs no execution)"));
    }
    validate_runtime(execution, runtime)?;
    if runtime.host_owners != SparkHostOwners::new(assignment, execution)
        || runtime.rank != assignment.rank || runtime.live_capacity != execution.live_capacity
        || runtime.driver != sample.driver || runtime.arch != sample.arch || runtime.physical_sms != sample.sms {
        return Err(invalid("runtime calibrated for different host owners, rank, live capacity or device identity"));
    }
    let all: Vec<_> = runtime.device.iter().chain(&runtime.pinned).chain(&runtime.host).collect();
    let mut retained = execution.retained.clone();
    retained.device.insert(0, Item::new(Category::Experts, "expert_weights", "checkpoint_native",
        assignment.resident_bytes, Basis::Exact));
    // Intake is already the input/routes/output owners above. EXL3 output is
    // inside its manifest arena. Never add assignment.io.intake a second time.
    retained.host.push(item(Category::Transport, "expert_exchange", assignment.io.host_exchange, Basis::Formula));
    retained.host.push(item(Category::Transport, "expert_row_indices", assignment.io.row_indices, Basis::Formula));
    retained.pinned.push(item(Category::Transport, "mapped_endpoint_rings", assignment.io.mapped_rings, Basis::Formula));
    retained.device.extend(runtime.device.iter().map(|t| t.item.clone()));
    retained.pinned.extend(runtime.pinned.iter().map(|t| t.item.clone()));
    retained.host.extend(runtime.host.iter().map(|t| t.item.clone()));
    let load_transient = SparkMemoryDomains {
        device: vec![item(Category::Staging, "expert_load_device", execution.load.device_staging_bytes, Basis::Exact)],
        pinned: vec![item(Category::Staging, "expert_load_pinned", execution.load.pinned_staging_bytes, Basis::Exact)],
        host: vec![item(Category::Staging, "expert_read_scratch", execution.load.host_read_scratch_bytes, Basis::Exact)],
    };
    let load_peak = add([assignment.resident_bytes, add(all.iter().map(|t| t.item.bytes))?, load_transient.total()?])?;
    Ok(SparkRankInventory { rank: assignment.rank, load_peak, serve_peak: retained.total()?,
        device: retained.device, pinned: retained.pinned, host: retained.host, load_transient,
        load_provenance: execution.load.source.clone(),
        manifests: execution.manifests.clone(), runtime_provenance: all.iter().map(|t| t.provenance.clone()).collect() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn load() -> SparkLoadInventory {
        SparkLoadInventory { device_staging_bytes: 100, pinned_staging_bytes: 200,
            host_read_scratch_bytes: 300, source: "test checkpoint staging plans".into() }
    }
    fn native() -> Value {
        json!({"schema":1,"role":"spark_tp3","spark_tp_degree":3,"capability":[12,1],
            "physical_sms":48,"sparkinfer_revision":"test","input_format":"fp8_k32",
            "geometry":{"experts":384,"hidden":5120,"intermediate":768,"kernel_intermediate":768,"topk":6},
            "variants":[{"capacity_rows":1,"core_scratch_nbytes":11},
                {"capacity_rows":80,"core_scratch_nbytes":80},
                {"capacity_rows":4096,"core_scratch_nbytes":4096}]})
    }
    fn execution() -> SparkExecutionInventory {
        SparkExecutionInventory::native(&native(), 48, 4096, 1024, &[4096, 1, 80], load()).unwrap()
    }
    fn runtime(e: &SparkExecutionInventory) -> QualifiedSparkRuntime {
        let term = |name, bytes, method| SparkRuntimeTerm {
            item: item(Category::Runtime, name, bytes, Basis::Calibrated),
            provenance: SparkRuntimeProvenance { source: "test quiet-host census".into(),
                method, phase: "before/after isolated same-process phase".into(), pid: 42, quiet_host: true } };
        QualifiedSparkRuntime { arch: "sm_121".into(), driver: "test".into(), physical_sms: 48,
            manifest_sha256: e.manifests.iter().map(|m| m.sha256.clone()).collect(),
            host_owners: SparkHostOwners::new(&assignment(), e), rank: assignment().rank, live_capacity: e.live_capacity,
            device: vec![term("cuda_context", 10, SparkRuntimeMethod::ProcessAttributedPhaseDelta),
                term("expert_modules", 20, SparkRuntimeMethod::ProcessAttributedPhaseDelta)],
            pinned: vec![], host: vec![term("host_measured_residual", 30, SparkRuntimeMethod::MeasuredResidual)] }
    }
    fn runtime_sample() -> cuteafd_core::RuntimeSample {
        cuteafd_core::RuntimeSample { device: 0, total_bytes: 16384 << 20,
            free_bytes: 8192 << 20, context_bytes: 0, module_bytes: 0, sms: 48,
            arch: "sm_121".into(), driver: "test".into() }
    }
    fn assignment() -> SparkAssignment {
        SparkAssignment { rank: 1, resident_bytes: 1000, slices: vec![cuteafd_core::SparkSlice {
            layer: 5, first: 768, width: 768, stored_width: 768, tiers: vec![], projection_counts: vec![] }],
            io: crate::placement::SparkIoBytes { intake: u64::MAX, host_exchange: 111, row_indices: 222, mapped_rings: 333 } }
    }
    #[test]
    fn constructors_validate_before_publishing_read_only_proofs() {
        let e = execution();
        let r = runtime(&e);
        let qualified = QualifiedSparkRuntime::new(&assignment(), &e, &runtime_sample(),
            r.manifest_sha256.clone(), r.device.clone(), r.pinned.clone(), r.host.clone()).unwrap();
        assert_eq!(qualified.arch(), "sm_121");
        assert_eq!(e.compiled_capacity(), 4096);
        assert_eq!(e.retained_capacities(), &[4096, 1, 80]);
        let mut wrong = runtime_sample(); wrong.arch = "sm_120".into();
        assert!(QualifiedSparkRuntime::new(&assignment(), &e, &wrong,
            r.manifest_sha256.clone(), r.device.clone(), r.pinned.clone(), r.host.clone()).is_err());
        assert!(SparkRuntimeProvenance::new(" ".into(), SparkRuntimeMethod::ProcessAttributedPhaseDelta,
            "phase".into(), 42, true).is_err());
        assert!(SparkRuntimeProvenance::new("source".into(), SparkRuntimeMethod::ProcessAttributedPhaseDelta,
            "phase".into(), 42, false).is_err());
        let provenance = SparkRuntimeProvenance::new("isolated same-process maps/system pair".into(),
            SparkRuntimeMethod::MeasuredResidual, "phase".into(), 42, true).unwrap();
        assert!(SparkRuntimeTerm::new(item(Category::Runtime, "cuda_context", 10, Basis::Calibrated),
            provenance.clone()).is_err());
        assert!(SparkRuntimeTerm::new(item(Category::Workspace, "host_measured_residual", 10, Basis::Calibrated),
            provenance.clone()).is_err());
        let term = SparkRuntimeTerm::new(item(Category::Runtime, "host_measured_residual", 10, Basis::Calibrated),
            provenance).unwrap();
        assert_eq!(term.item().bytes, 10);
    }

    fn census_samples() -> Vec<cuteafd_core::expert_geometry::SparkProcessSample> {
        let phases = ["before_context", "after_context_before_media", "before_expert_modules",
            "after_expert_modules_before_weights", "after_retained_arenas", "serve_owners_before_endpoint_rings",
            "steady_after_first_expert_request"];
        phases.into_iter().enumerate().map(|(i, phase)| {
            let mib = 1 << 20;
            let cuda = [0, 32, 32, 96, 1120, 1120, 1120][i] * mib;
            cuteafd_core::expert_geometry::SparkProcessSample {
                phase: phase.into(), pid: 42, monotonic_ns: i as u64 + 1,
                rss_bytes: cuda, peak_rss_bytes: Some(cuda), cgroup_memory: vec![], pss_bytes: cuda, private_bytes: 0,
                tracked_device_bytes: if i >= 4 { 1024 * mib } else { 0 }, tracked_pinned_bytes: 0,
                device_kind: Some(cuteafd_core::expert_geometry::CensusDeviceKind::Discrete),
                device_witness: Some(cuteafd_core::expert_geometry::CensusDeviceWitness::CudaMappingPss),
                cuda_mapping_pss_bytes: cuda, anonymous_pss_bytes: 0, file_pss_bytes: 0,
                system_available_bytes: 8192 * mib - cuda - if i >= 5 { 8 * mib + 333 } else { 0 },
                system_headroom_bytes: 8192 * mib - cuda - if i >= 5 { 8 * mib + 333 } else { 0 },
                system_total_bytes: 16384 * mib, allocatable_bytes: None,
                raw_cuda_free_bytes: None, method: "synthetic unit-test maps/system".into() }
        }).collect()
    }
    #[test]
    fn public_constructor_requires_dual_witness_evidence_and_package_identity() {
        let e = execution();
        let runtime = cuteafd_core::RuntimeSample { device: 0, total_bytes: 16384 << 20,
            free_bytes: 8192 << 20, context_bytes: 0, module_bytes: 0, sms: 48,
            arch: "sm_121".into(), driver: "test".into() };
        let hashes: Vec<_> = e.manifests().iter().map(|m| m.sha256().to_owned()).collect();
        let construct = |samples: &[_], hashes, quiet| QualifiedSparkRuntime::from_census(
            &assignment(), &e, &runtime, hashes, samples,
            "synthetic unit test; not hardware qualification".into(), quiet);
        let qualified = construct(&census_samples(), hashes.clone(), true).unwrap();
        assert_eq!(qualified.device()[0].item().bytes, 32 << 20);
        assert_eq!(qualified.device()[1].item().bytes, 64 << 20);
        assert_eq!(qualified.host()[0].item().bytes, (8 << 20) + 333);
        assert_eq!(qualified.host_owners.exchange, 111);
        assert_eq!(qualified.host_owners.row_indices, 222);
        let mut integrated = census_samples();
        for sample in &mut integrated {
            sample.device_kind = Some(cuteafd_core::expert_geometry::CensusDeviceKind::Integrated);
            sample.device_witness = Some(cuteafd_core::expert_geometry::CensusDeviceWitness::SystemAvailable);
            sample.cuda_mapping_pss_bytes = 0;
        }
        let uma = construct(&integrated, hashes.clone(), true).unwrap();
        assert_eq!(uma.device()[0].item().bytes, 32 << 20);
        assert_eq!(uma.device()[0].provenance().method(), SparkRuntimeMethod::IntegratedSystemPhaseDelta);
        assert!(uma.device()[0].provenance().phase().contains("cgroup"));
        let mut different = assignment();
        different.io.host_exchange += 1;
        assert!(v41_spark_inventory(&different, &e, &qualified, &runtime).is_err());
        assert_eq!(qualified.host()[0].provenance().method(), SparkRuntimeMethod::MeasuredResidual);
        assert!(construct(&census_samples(), hashes.clone(), false).is_err());
        assert!(construct(&census_samples(), vec!["wrong-package".into()], true).is_err());
        for bad in 0..12 {
            let mut samples = census_samples();
            match bad {
                0 => samples[1].pid = 43,
                1 => samples[2].monotonic_ns = samples[1].monotonic_ns,
                2 => samples[1].tracked_device_bytes = 1,
                3 => samples[3].tracked_pinned_bytes = 1,
                4 => samples[6].cuda_mapping_pss_bytes = samples[3].cuda_mapping_pss_bytes,
                5 => samples[1].cuda_mapping_pss_bytes = 0,
                6 => samples[3].system_available_bytes -= 32 << 20,
                7 => samples[1].system_total_bytes = 0,
                8 => samples.push(samples[0].clone()),
                9 => samples[4].tracked_device_bytes = 0,
                10 => samples[5].system_available_bytes += 9 << 20,
                _ => { samples.pop(); },
            }
            assert!(construct(&samples, hashes.clone(), true).is_err(), "case {bad}");
        }
    }

    #[test]
    fn binding_checks_host_tuple_rank_capacity_and_driver() {
        let e = execution();
        let a = assignment();
        let r = runtime(&e);
        let sample = runtime_sample();
        let mut changed = a.clone();
        changed.io.host_exchange += 1;
        changed.io.row_indices -= 1;
        assert!(v41_spark_inventory(&changed, &e, &r, &sample).is_err());
        changed = a.clone();
        changed.rank += 1;
        assert!(v41_spark_inventory(&changed, &e, &r, &sample).is_err());
        let mut changed_execution = e.clone();
        changed_execution.live_capacity -= 1;
        assert!(v41_spark_inventory(&a, &changed_execution, &r, &sample).is_err());
        let mut changed_sample = sample.clone();
        changed_sample.driver = "different-driver".into();
        assert!(v41_spark_inventory(&a, &e, &r, &changed_sample).is_err());
        changed_execution = e.clone();
        changed_execution.retained.host.push(item(Category::Workspace, "host_scratch", 1, Basis::Formula));
        assert!(v41_spark_inventory(&a, &changed_execution, &r, &sample).is_err());
    }

    #[test]
    fn lazy_host_capacity_does_not_reduce_resident_residual() {
        let e = execution();
        let mut a = assignment();
        a.io.host_exchange = 64 << 20;
        let proof = QualifiedSparkRuntime::from_census(&a, &e, &runtime_sample(),
            e.manifests.iter().map(|m| m.sha256.clone()).collect(), &census_samples(),
            "synthetic lazy zero-page census".into(), true).unwrap();
        assert_eq!(proof.host()[0].item().bytes, (8 << 20) + 333);
        let inventory = v41_spark_inventory(&a, &e, &proof, &runtime_sample()).unwrap();
        assert_eq!(inventory.host().iter().find(|i| i.group == "expert_exchange").unwrap().bytes, 64 << 20);
    }

    #[test]
    #[ignore = "requires CUTEAFD_SPARK_CENSUS_REPLAY pointing to recorded rank samples"]
    fn replay_recorded_rank_residuals() {
        use cuteafd_core::expert_geometry::{spark_census_deltas, SparkProcessSample};
        let path = std::env::var("CUTEAFD_SPARK_CENSUS_REPLAY").unwrap();
        let ranks: std::collections::BTreeMap<String, Vec<SparkProcessSample>> =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(!ranks.is_empty());
        for (rank, samples) in ranks {
            let deltas = spark_census_deltas(&samples).unwrap();
            let before = samples.iter().find(|s| s.phase == "before_context").unwrap();
            let ready = samples.iter().find(|s| s.phase == "steady_after_first_expert_request").unwrap();
            let residual = census_residual(before, ready, deltas.context, deltas.modules).unwrap();
            println!("{rank}: host_measured_residual_bytes={residual}");
        }
    }

    #[test]
    fn native_sums_retained_arenas_instead_of_coordinator_maximum() {
        let e = execution();
        assert_eq!(e.retained.device.iter().filter(|i| i.group.starts_with("expert_scratch"))
            .map(|i| i.bytes).sum::<u64>(), 4096 + 11 + 80);
        assert_eq!(e.retained.pinned[0].bytes, 1024 * 6 * 8);
        let r = runtime(&e);
        let result = v41_spark_inventory(&assignment(), &e, &r, &runtime_sample()).unwrap();
        assert_eq!(result.load_peak, 1000 + 60 + 100 + 200 + 300);
        assert_eq!(result.serve_peak, 1000 + 60 + e.retained.total().unwrap() + 111 + 222 + 333);
        assert!(!result.device.iter().any(|i| i.group == "intake"));
        assert_eq!(result.device.iter().chain(&result.pinned).chain(&result.host).map(|i| i.bytes).sum::<u64>(), result.serve_peak);
    }
    #[test]
    fn native_rejects_unsupported_states_and_package_mismatches() {
        for (sms, compiled, live, states) in [(47,4096,1024,vec![4096,1,80]),
            (48,4096,4097,vec![4096,1,80]), (48,4096,1024,vec![4096]),
            (48,4096,1024,vec![4096,80,1]), (48,80,80,vec![80,1,80])] {
            assert!(SparkExecutionInventory::native(&native(), sms, compiled, live, &states, load()).is_err());
        }
        for key in ["role", "physical_sms", "variants", "geometry", "sparkinfer_revision"] {
            let mut m = native(); m[key] = Value::Null;
            assert!(SparkExecutionInventory::native(&m,48,4096,1024,&[4096,1,80],load()).is_err(), "{key}");
        }
        let mut duplicate = native(); let v = duplicate["variants"][0].clone();
        duplicate["variants"].as_array_mut().unwrap().push(v);
        assert!(SparkExecutionInventory::native(&duplicate,48,4096,1024,&[4096,1,80],load()).is_err());
    }
    #[test]
    fn exl3_preserves_shared_max_and_private_capacity_state() {
        let manifests: Vec<_> = [1,16,80].into_iter().map(|rows| json!({
            "schema":"cuteafd.v41-exl3-aot.v1", "output_dtype":"bf16", "hidden":5120,
            "intermediate":768,"experts":384,"top_k":6,"sms":48,"capacity":rows,
            "sparkinfer_revision":"test","input_format":"e4m3_k32",
            "trellis_lut":{"bytes":32},"buffers":{
                "scratch":{"bytes":rows*100,"allocation":"scratch","dtype":"bf16","zero_on_create":false},
                "alias":{"bytes":99999,"allocation":"scratch","dtype":"bf16","zero_on_create":false},
                "state":{"bytes":0,"allocation":"state","dtype":"i32","zero_on_create":true}}})).collect();
        let e = SparkExecutionInventory::exl3(&manifests,48,80,16,&[1,16,80],load()).unwrap();
        assert_eq!(e.retained.device[0].bytes,8000 + 3*(32+16));
        assert_eq!(e.retained.device[1].bytes,16*5280);
        assert!(SparkExecutionInventory::exl3(&manifests,47,80,16,&[1,16,80],load()).is_err());
        assert!(SparkExecutionInventory::exl3(&manifests[2..],48,80,16,&[80],load()).is_err());
    }
    #[test]
    fn runtime_requires_isolated_evidence_and_matching_package() {
        let e=execution(); let a=assignment();
        for bad in 0..9 {
            let mut r=runtime(&e);
            match bad {
                0 => r.device[0].provenance.quiet_host=false,
                1 => r.device[0].provenance.pid=0,
                2 => r.device[0].item.basis=Basis::Exact,
                3 => r.manifest_sha256[0]="different".into(),
                4 => r.device[0].provenance.method=SparkRuntimeMethod::MeasuredResidual,
                5 => r.host[0].item.group="context".into(),
                6 => r.host.clear(),
                7 => r.device.push(r.device[0].clone()),
                _ => r.host[0].provenance.method=SparkRuntimeMethod::ProcessAttributedPhaseDelta,
            }
            assert!(v41_spark_inventory(&a,&e,&r,&runtime_sample()).is_err(),"{bad}");
        }
        let mut overflow=e.clone(); overflow.load.device_staging_bytes=u64::MAX;
        assert!(v41_spark_inventory(&a,&overflow,&runtime(&overflow),&runtime_sample()).is_err());
    }
}

#[cfg(test)]
mod checkpoint_load_tests {
    use super::*;
    #[test]
    fn native_load_inventory_matches_checkpoint_staging_not_resident_size() {
        for nvfp4 in [false,true] {
            let dir=tempfile::tempdir().unwrap();
            if nvfp4 { crate::plan::testing::write_v41_nvfp4_snapshot(dir.path()); }
            else { crate::plan::testing::write_v41_snapshot(dir.path()); }
            let catalog=crate::read_expert_catalog(dir.path()).unwrap();
            let selection=crate::V41ExpertSelection::BackboneTp { layer:3,expert:0,rank:1,world:3 };
            let inventory=SparkLoadInventory::native(&catalog,&[selection]).unwrap();
            let (staging,scratch)=if nvfp4 {
                let plan=catalog.nvfp4_expert_staging(selection).unwrap();
                (plan.staging_bytes() as u64,plan.minimum_read_scratch_bytes() as u64)
            } else {
                let plan=catalog.expert_staging(selection).unwrap();
                (plan.staging_bytes() as u64,plan.minimum_read_scratch_bytes() as u64)
            };
            assert_eq!(inventory.device_staging_bytes,staging);
            assert_eq!(inventory.pinned_staging_bytes,staging*16);
            assert_eq!(inventory.host_read_scratch_bytes,scratch*64*16);
            assert!(SparkLoadInventory::native(&catalog,&[]).is_err());
        }
    }
    #[test]
    fn exl3_load_inventory_matches_simultaneous_banks_not_all_layers() {
        let dir=tempfile::tempdir().unwrap();
        crate::plan::testing::write_v41_exl3_snapshot(dir.path(),3);
        let catalog=crate::read_expert_catalog(dir.path()).unwrap();
        let manifest=catalog.exl3().unwrap();
        let plans=[3,4].into_iter().map(|layer|manifest.residency(crate::V41Exl3Layer::Backbone(layer),3,1).unwrap()).collect::<Vec<_>>();
        let result=SparkLoadInventory::exl3(&catalog,&plans).unwrap();
        let staging=plans.iter().map(|p|p.loads.chunks(9).map(|jobs|jobs.iter().map(|j|j.bytes as u64).sum::<u64>()).max().unwrap()).max().unwrap();
        let init=plans.iter().flat_map(|p|&p.buffers).map(|b|b.initial_words.len() as u64*4).max().unwrap();
        let scratch=plans.iter().map(|p|p.scratch_bytes(&catalog).unwrap() as u64).max().unwrap();
        assert_eq!(result.device_staging_bytes,0);
        assert_eq!(result.pinned_staging_bytes,staging.max(init)*16*2);
        assert_eq!(result.host_read_scratch_bytes,scratch*64*16*2);
    }
}
