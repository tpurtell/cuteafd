//! Exact inventory items (placement PR 2): what a coordinator GPU holds before
//! any family demand, the programs a family launches, the decode graph set it
//! captures and the expert package scratch it allocates. The planner and the
//! runtime admission share every definition here; they differ only in where
//! the GPU's baseline comes from:
//! - **runtime**: [`RuntimeSample`] per GPU, measured after the CUDA context,
//!   cuBLAS and the family's program modules (`RuntimeInventory::measure` in
//!   the daemon), before any weight;
//! - **planner**: [`ArchContext::for_device`], the same bytes measured once per
//!   architecture and driver, plus the family's [`ProgramSet`] module bytes.
//!
//! [`GraphSet`] is the one definition of the executables a family captures:
//! the startup set is charged at ready ([`Lifetime::Startup`]), lazily captured
//! graphs are growth ([`Lifetime::Growth`]): reserved, but outside the ready
//! ledger. S3's `GraphBank` consumes the same set as its warm list.
use cuteafd_core::memory_layout::{Basis, Category};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Only inputs that affect a maintained figure participate in compatibility.
/// The producing engine commit belongs to evidence, not this identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FigureIdentity<'a> {
    pub declared_graphs: &'a str,
    /// The manifest hash includes its kernel object hashes.
    pub programs_sha256: &'a str,
    pub native_sha256: &'a str,
    pub toolchain: &'a str,
    pub gpu_class: &'a str,
    pub driver: &'a str,
    pub layout: &'a str,
}

impl FigureIdentity<'_> {
    pub fn stale_parts(&self, current: FigureIdentity<'_>) -> Vec<&'static str> {
        [
            ("declared graph set", self.declared_graphs == current.declared_graphs),
            ("kernel/program hashes", self.programs_sha256 == current.programs_sha256),
            ("native library", self.native_sha256 == current.native_sha256),
            ("toolchain", self.toolchain == current.toolchain),
            ("GPU class", self.gpu_class == current.gpu_class),
            ("driver", self.driver == current.driver),
            ("layout", self.layout == current.layout),
        ].into_iter().filter_map(|(part, equal)| (!equal).then_some(part)).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FigureAttribution {
    Measured,
    Estimated { derivation: &'static str },
}

/// A family-code parameter, never a receipt or an independently proved bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryFigure {
    pub bytes: u64,
    pub attribution: FigureAttribution,
    pub evidence: &'static str,
    pub producing_commit: &'static str,
    pub identity: FigureIdentity<'static>,
}

impl MemoryFigure {
    /// Derive an explicitly estimated layout from a whole measured figure.
    /// Rounding is integral-byte ceiling, not an automatic safety margin.
    pub fn scaled(self, numerator: u64, denominator: u64, derivation: &'static str)
        -> Result<Self, &'static str> {
        if denominator == 0 || derivation.trim().is_empty() {
            return Err("invalid maintained memory figure derivation");
        }
        let bytes = u128::from(self.bytes).checked_mul(u128::from(numerator))
            .ok_or("maintained memory figure overflow")?.div_ceil(u128::from(denominator));
        Ok(Self { bytes: bytes.try_into().map_err(|_| "maintained memory figure overflow")?,
            attribution: FigureAttribution::Estimated { derivation }, ..self })
    }
}

/// Call only for figures consumed by the resolved plan. Unselected alternatives
/// must not produce recommendations, and compatibility never changes the bytes.
#[derive(Debug, Clone, Copy)]
pub struct FigureUse<'a> {
    pub name: &'a str,
    pub figure: MemoryFigure,
    pub current: FigureIdentity<'a>,
}

pub fn census_recommendation<'a>(figures: impl IntoIterator<Item = FigureUse<'a>>) -> Option<String> {
    let mut findings = std::collections::BTreeSet::new();
    for used in figures {
        let mut reasons = Vec::new();
        if let FigureAttribution::Estimated { derivation } = used.figure.attribution {
            reasons.push(format!("estimated, derived from {}: {derivation}", used.figure.evidence));
        }
        let stale = used.figure.identity.stale_parts(used.current);
        if !stale.is_empty() { reasons.push(format!("stale {}", stale.join(", "))); }
        if !reasons.is_empty() {
            findings.insert(format!("{} ({}; evidence {})", used.name, reasons.join("; "), used.figure.evidence));
        }
    }
    (!findings.is_empty()).then(|| format!(
        "census recommended for maintained runtime memory figures: {}; using maintained bytes unchanged",
        findings.into_iter().collect::<Vec<_>>().join("; ")))
}

#[path = "spark_inventory.rs"]
mod spark_inventory;
pub use spark_inventory::{v41_spark_inventory, QualifiedSparkRuntime, SparkExecutionInventory, SparkInventoryError, SparkLoadInventory, SparkManifestProvenance, SparkMemoryDomains, SparkRankInventory, SparkRuntimeMethod, SparkRuntimeProvenance, SparkRuntimeTerm};

const MIB: u64 = 1 << 20;

/// A coordinator GPU class the planner can lay out without a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ArchContext {
    /// `sm_120`, `sm_121`.
    pub arch: &'static str,
    /// Card class: physical memory at most this many bytes (CUDA total).
    pub max_total_bytes: u64,
    /// The CUDA-reported total of the class's reference card (`cudaMemGetInfo`),
    /// what a planner without a device lays out (0: unknown, take the budget).
    pub total_bytes: u64,
    /// Streaming multiprocessors (native scratch and grid sizes key on it).
    pub sms: u32,
    /// CUDA context and runtime bookkeeping, measured on an otherwise empty
    /// device after the first allocation (no program modules, no cuBLAS).
    pub context_bytes: u64,
    /// A cuBLAS handle and its first GEMM's workspace. The native handle is
    /// thread-local and created on the serving thread's first GEMM, after the
    /// runtime's inventory sample, so both sides charge it as a known future
    /// item (planner: inside "context+modules"; runtime: beside the sample).
    pub cublas_bytes: u64,
    /// Additional peer mapping/context residency. TODO: measure independently of
    /// graph executables on SM120 PRO / driver 595.91.07; rc3 split ledgers mix both.
    pub peer_context_bytes: u64,
    /// Device bytes one captured decode-graph executable holds (measured).
    pub graph_executable_bytes: u64,
    /// The driver these numbers were measured on.
    pub driver: &'static str,
    /// Measurement provenance.
    pub source: &'static str,
}

/// Per-arch context table (placement design section 1, "Estimates replaced").
/// `context_bytes` is the CUDA context plus a cuBLAS handle with its first GEMM
/// workspace, on an otherwise empty device (`scripts/bench/cuda-context-probe.cu`,
/// 2026-10-10): SM120 PRO 586,416,128 B context + 71,303,168 B cuBLAS =
/// 657,719,296 B (rc3 MiMo `non_engine` samples agree: 586,416,128 /
/// 596,901,888 B); GB10 by MemAvailable deltas (unified memory) 189-257 MiB
/// context + 150-160 MiB cuBLAS, recorded at the larger sample.
pub const ARCH_CONTEXTS: &[ArchContext] = &[
    ArchContext { arch: "sm_120", max_total_bytes: 34 << 30, total_bytes: 0, sms: 170, context_bytes: 586_416_128,
        cublas_bytes: 71_303_168,
        peer_context_bytes: 0, graph_executable_bytes: 149_712, driver: "595.91.07",
        source: "RTX 5090 class: SM120 PRO probe on the same driver/CUDA 13.2; SM count from the 5090 spec" },
    ArchContext { arch: "sm_120", max_total_bytes: u64::MAX, total_bytes: 101_973_491_712, sms: 188,
        context_bytes: 586_416_128, cublas_bytes: 71_303_168,
        peer_context_bytes: 0, graph_executable_bytes: 149_712, driver: "595.91.07",
        source: "RTX PRO 6000 Blackwell probe (CUDA 13.2); Qwen 12,397 graphs = 1,855,979,520 B" },
    ArchContext { arch: "sm_121", max_total_bytes: u64::MAX, total_bytes: 130_594_156_544, sms: 48,
        context_bytes: 257 << 20, cublas_bytes: 160 << 20,
        peer_context_bytes: 0, graph_executable_bytes: 149_712, driver: "580.178.04",
        source: "GB10 probe (CUDA 13.0, MemAvailable deltas): context 189-257 MiB + cuBLAS 150-160 MiB; graph bytes from SM120" },
];

/// The coordinator GPU budget `cuteafd plan` lays out without a device or a
/// `--gpu-memory-cap`: the RTX PRO 6000's CUDA-reported total
/// (94.97 GiB; the card's nominal 95.5 GiB is not what CUDA hands out).
pub const PRO_TOTAL_BYTES: u64 = 101_973_491_712;

impl ArchContext {
    /// The class of a device with `total_bytes` of memory on `arch`, or the
    /// PRO entry when nothing matches.
    pub fn for_device(arch: &str, total_bytes: u64) -> &'static ArchContext {
        ARCH_CONTEXTS.iter().find(|c| c.arch == arch && total_bytes <= c.max_total_bytes)
            .unwrap_or(&ARCH_CONTEXTS[1])
    }

    /// The planner's coordinator class: SM120, sized by its card budget. A
    /// simulated small card (`--gpu-memory-cap` on a PRO) keeps the
    /// PRO's SM count: the simulation caps memory only.
    pub fn coordinator(total_bytes: u64, simulated_on: Option<&'static ArchContext>) -> ArchContext {
        let class = *Self::for_device("sm_120", total_bytes);
        match simulated_on { Some(physical) => ArchContext { sms: physical.sms, ..class }, None => class }
    }
}

/// Canonical family of a program prefix (`dsv4f`, `mimop2`, ...).
pub fn loaded_code_family(program_family: &str) -> &str {
    let base = program_family.trim_end_matches('2');
    if base.starts_with("mimo") { "mimo" } else if base.starts_with("dsv4") { "dsv4" } else { base }
}

/// Ready code beyond the CUDA context, shared by offline and live admission.
#[derive(Debug, Clone, Copy)]
pub struct CodeAllowance {
    pub bytes: u64,
    pub basis: Basis,
    /// Formula paths cannot attribute a large module sample to future BLAS handles.
    future_library_bytes: u64,
}

impl CodeAllowance {
    pub fn pending(self, untracked_at_sample: u64, context_bytes: u64) -> u64 {
        let resident_code = untracked_at_sample.saturating_sub(context_bytes);
        self.future_library_bytes + self.bytes.saturating_sub(self.future_library_bytes)
            .saturating_sub(resident_code)
    }
}

/// Selected-program code, separate from named BLAS owners in migrated families.
/// Every family here charges its modules by name and its library storage in
/// disjoint typed owners; none keeps a mixed residency row.
pub fn loaded_code_allowance(program_family: &str, _experts: &str, split: bool, _rank: u8,
    arch: &ArchContext, programs: Option<&ProgramSet>) -> CodeAllowance {
    // Migrated families keep BLAS and driver storage in disjoint typed owners.
    if matches!(loaded_code_family(program_family), "glmf" | "glmf-k6" | "kimik3" | "qwen4" | "mimo" | "muse" | "step5" | "dsv4") {
        return CodeAllowance { bytes: if loaded_code_family(program_family) == "step5" {
            programs.map_or(32 * MIB, ProgramSet::module_bytes)
        } else {
            programs.map_or_else(
                || ProgramSet::from_names(&[program_family], std::iter::empty()).module_bytes(),
                ProgramSet::module_bytes)
        },
            basis: Basis::Estimated, future_library_bytes: 0 };
    }
    CodeAllowance { bytes: arch.cublas_bytes
        + programs.map_or(32 * MIB, ProgramSet::module_bytes)
        + if split { arch.peer_context_bytes } else { 0 }, basis: Basis::Formula,
        future_library_bytes: arch.cublas_bytes }
}

pub use cuteafd_core::RuntimeSample;

/// Typed native projection entries, parsed once at an artifact boundary. Other
/// exporter programs remain opaque so their independent schemas can evolve.
#[derive(Debug, Clone, Default)]
pub struct ProgramsManifest {
    entries: std::collections::BTreeMap<String, ProgramEntry>,
}

#[derive(Debug, Clone)]
pub struct ProgramEntry {
    pub spec: cuteafd_ffi::programs::ProgramSpec,
}

impl ProgramsManifest {
    pub fn from_manifest(manifest: &Value) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        let programs = manifest["programs"].as_array().context("PROGRAMS.json has no program list")?;
        let mut entries = std::collections::BTreeMap::new();
        for entry in programs {
            let Some(name) = entry["name"].as_str().filter(|name| Self::is_native(name)) else { continue; };
            let parsed = ProgramEntry::from_manifest(entry).with_context(|| format!("native program {name}"))?;
            ensure!(entries.insert(name.to_owned(), parsed).is_none(), "duplicate native program {name}");
        }
        Ok(Self { entries })
    }

    /// Reuse an already parsed runtime program table without rereading its manifest.
    pub fn from_specs<'a>(specs: impl Iterator<Item = &'a cuteafd_ffi::programs::ProgramSpec>)
        -> anyhow::Result<Self> {
        let mut entries = std::collections::BTreeMap::new();
        for spec in specs.filter(|spec| Self::is_native(&spec.name)) {
            anyhow::ensure!(entries.insert(spec.name.clone(), ProgramEntry { spec: spec.clone() }).is_none(),
                "duplicate native program {}", spec.name);
        }
        Ok(Self { entries })
    }

    fn is_native(name: &str) -> bool {
        name.starts_with("draft_bf16_k") || name.starts_with("native_attention_n")
    }

    pub fn get(&self, name: &str) -> Option<&ProgramEntry> { self.entries.get(name) }
}

impl ProgramEntry {
    fn from_manifest(entry: &Value) -> anyhow::Result<Self> {
        use anyhow::Context;
        #[derive(Deserialize)]
        struct Pointer { name: String }
        #[derive(Deserialize)]
        struct Scalar { #[serde(rename = "type")] kind: String }
        #[derive(Deserialize)]
        struct Entry {
            name: String,
            pointers: Vec<Pointer>,
            scalars: Vec<Scalar>,
            scratch_bytes_at_capacity: std::collections::HashMap<String, u64>,
            capacity_rows: u32,
            geometry: Option<Value>,
        }
        let entry: Entry = serde_json::from_value(entry.clone()).context("native projection schema")?;
        let geometry = entry.geometry.context("native projection geometry missing")?;
        let scalar_kinds = entry.scalars.into_iter().map(|scalar| match scalar.kind.as_str() {
            "int32" => Ok(b'i'), "int64" => Ok(b'l'), "float32" => Ok(b'f'),
            kind => Err(anyhow::anyhow!("unsupported native scalar {kind}")),
        }).collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { spec: cuteafd_ffi::programs::ProgramSpec { name: entry.name,
            pointers: entry.pointers.into_iter().map(|pointer| pointer.name).collect(), scalar_kinds,
            scratch: entry.scratch_bytes_at_capacity, capacity_rows: entry.capacity_rows, geometry } })
    }
}

/// The programs a family launches: a prefix filter over the image's program
/// manifest (`<family>_...` and its head-split `<family>2_...`), not every
/// program in the image. Runtime module loads and shared scratch use the same
/// set (`cuteafd_core::coordinator_programs::CoordinatorPrograms`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramSet {
    pub families: Vec<String>,
    pub names: Vec<String>,
    /// Largest per-program scratch at capacity among `names`, by program.
    pub scratch: Vec<(String, u64)>,
}

impl ProgramSet {
    /// Selected runtime program names use the same prefix filter as PROGRAMS.json.
    pub fn from_names<'a>(families: &[&str], names: impl IntoIterator<Item = &'a str>) -> Self {
        Self { families: families.iter().map(|f| f.to_string()).collect(),
            names: names.into_iter().filter(|name| cuteafd_core::coordinator_programs::program_namespace(name)
                .is_some_and(|prefix| families.contains(&prefix))).map(str::to_string).collect(),
            scratch: Vec::new() }
    }

    /// Programs of `families` in `manifest` (PROGRAMS.json).
    pub fn from_manifest(manifest: &Value, families: &[&str]) -> Self {
        let keep = |name: &str| cuteafd_core::coordinator_programs::program_namespace(name)
            .is_some_and(|prefix| families.contains(&prefix));
        let mut names = Vec::new();
        let mut scratch = Vec::new();
        for program in manifest["programs"].as_array().into_iter().flatten() {
            let Some(name) = program["name"].as_str().filter(|n| keep(n)) else { continue };
            names.push(name.to_string());
            let bytes = program["scratch_bytes_at_capacity"].as_object().into_iter().flatten()
                .filter_map(|(_, v)| v.as_u64()).max().unwrap_or(0);
            if bytes > 0 { scratch.push((name.to_string(), bytes)); }
        }
        Self { families: families.iter().map(|f| f.to_string()).collect(), names, scratch }
    }

    /// Native V4.1 projection plans retain a separate aligned arena for each
    /// small-row specialization and the first compiled capacity holding `rows`.
    /// These exports live in V41_FP8_AOT.json, not PROGRAMS.json. Module residency
    /// is charged separately; `scratch` here describes allocation owners.
    pub fn v41_fp8(manifest: &Value, matrices: &[(u64, u64)], rows: u64, sms: u32)
        -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        ensure!((1..=4096).contains(&rows) && sms > 0, "invalid V4.1 FP8 plan geometry");
        let covers = if rows == 40 || rows == cuteafd_core::V41_DRAFT_LEGACY_ROWS as u64 { [1, 16, cuteafd_core::V41_DRAFT_LEGACY_ROWS as u64, cuteafd_core::V41_DRAFT_WIDE_ROWS as u64, 1024, 4096] } else { cuteafd_core::V41_TARGET_CAPACITIES };
        let compiled = covers.into_iter().find(|&n| n >= rows).unwrap();
        let capacities: Vec<_> = covers[..3].iter().copied().filter(|&n| n < compiled).chain([compiled]).collect();
        let variants = manifest["variants"].as_array().context("missing V4.1 FP8 variants")?;
        let card = if sms >= 188 { 188 } else { 170 };
        let mut names = Vec::new();
        let mut scratch = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for &(k, n) in matrices {
            ensure!(seen.insert((k, n)), "duplicate V4.1 FP8 matrix owner");
            if k == 8192 && matches!(n, 5120 | 2560) {
                let canonical: Vec<_> = variants.iter().filter(|v|
                    v["capacity"].as_u64() == Some(16)
                        && v["input_dim"].as_u64() == Some(k)
                        && v["output_dim"].as_u64() == Some(n)
                        && matches!(v["plan_sms"].as_u64(), None | Some(0))
                ).collect();
                ensure!(canonical.len() == 1, "missing or duplicate V4.1 output reduction recipe");
                let expected = canonical[0]["expected_m"].as_u64().context("missing output reduction rows")?;
                let slices = canonical[0]["split_k_slices"].as_u64().context("missing output reduction slices")?;
                let plan = cuteafd_core::coordinator_programs::OutputReductionPlan::new(
                    1, usize::try_from(expected)?, usize::try_from(slices)?)?;
                for &capacity in capacities.iter().filter(|&&capacity| capacity <= 80) {
                    let recipe: Vec<_> = variants.iter().filter(|v|
                        v["capacity"].as_u64() == Some(capacity)
                            && v["input_dim"].as_u64() == Some(k)
                            && v["output_dim"].as_u64() == Some(n)
                            && matches!(v["plan_sms"].as_u64(), None | Some(0))
                    ).collect();
                    ensure!(recipe.len() == 1, "missing or duplicate V4.1 output reduction capacity");
                    plan.validate(usize::try_from(capacity)?,
                        usize::try_from(recipe[0]["expected_m"].as_u64().context("missing output reduction rows")?)?,
                        usize::try_from(recipe[0]["split_k_slices"].as_u64().context("missing output reduction slices")?)?,
                        recipe[0]["all_rows"].as_bool().context("missing output reduction coverage")?)?;
                }
            }
            for &capacity in &capacities {
                let matching: Vec<_> = variants.iter().filter(|v| {
                    let plan = v["plan_sms"].as_u64().unwrap_or(0);
                    v["capacity"].as_u64() == Some(capacity)
                        && v["input_dim"].as_u64() == Some(k)
                        && v["output_dim"].as_u64() == Some(n)
                        && (plan == 0 || plan == card)
                }).collect();
                ensure!(matching.len() == 1,
                    "missing or duplicate V4.1 FP8 projection {k}x{n} m{capacity} sm{card}");
                let variant = matching[0];
                let name = variant["label"].as_str().context("missing V4.1 FP8 program label")?.to_string();
                let bytes = variant["scratch_bytes"].as_u64().context("missing V4.1 FP8 scratch extent")?
                    .checked_add(255).context("V4.1 FP8 scratch alignment overflow")? & !255;
                names.push(name.clone());
                scratch.push((name, bytes));
            }
        }
        Ok(Self { families: vec!["v41".into()], names, scratch })
    }

    pub fn contains(&self, name: &str) -> bool {
        cuteafd_core::coordinator_programs::program_namespace(name)
            .is_some_and(|prefix| self.families.iter().any(|f| f == prefix))
    }

    /// Module bytes the family's programs take on one device: measured
    /// 35,651,584 B for a family's selected set (MiMo, 2026-10-09) on SM120;
    /// a 2 MiB-granular charge per 16 programs bounds the observed sets.
    pub fn module_bytes(&self) -> u64 {
        (self.names.len() as u64).div_ceil(16).max(1) * 2 * MIB + 32 * MIB
    }
}

/// Display group for graph reserves; ready comparison uses each item's lifetime.
pub const GRAPH_GROWTH: &str = "graph growth";

/// When an item exists relative to readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lifetime {
    /// Allocated or captured before the server reports ready.
    Startup,
    /// Reserved, but allocated after ready (lazy graph captures, lazy
    /// workspaces); outside the ready ledger.
    Growth,
}

/// Compatibility of one rank's exact startup executable census. A graph count
/// is not an identity: all launch keys and the code/device provenance must match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphCensusIdentity {
    pub declared_graphs: String,
    pub programs_sha256: String,
    pub native_sha256: String,
    pub toolchain: String,
    pub gpu_class: String,
    pub driver: String,
    pub layout: String,
}

impl GraphCensusIdentity {
    pub fn validate(&self) -> Result<(), String> {
        if [&self.declared_graphs, &self.programs_sha256, &self.native_sha256,
            &self.toolchain, &self.gpu_class, &self.driver, &self.layout]
            .iter().any(|value| value.trim().is_empty()) {
            return Err("incomplete startup graph census identity".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphCensus {
    pub identity: GraphCensusIdentity,
    pub executables: u64,
    pub bytes: u64,
    pub evidence: String,
    pub producing_commit: String,
}

/// A synchronized capture boundary. Tracked allocations are separate owners;
/// their change must never be absorbed into executable residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphMemorySample {
    pub total: u64,
    pub free: u64,
    pub tracked_device: u64,
    pub tracked_pinned: u64,
    pub tracked_managed: u64,
}

impl GraphMemorySample {
    pub fn retained_since(self, before: Self) -> Result<u64, String> {
        if self.total != before.total || self.free > self.total || before.free > before.total {
            return Err("incompatible startup graph memory samples".into());
        }
        if self.tracked_pinned != before.tracked_pinned || self.tracked_managed != before.tracked_managed {
            return Err("mapped or managed storage changed during graph capture; prepare it before sampling".into());
        }
        let bytes = i128::from(before.free) - i128::from(self.free)
            - (i128::from(self.tracked_device) - i128::from(before.tracked_device));
        u64::try_from(bytes).map_err(|_| "negative startup graph residency after excluding tracked allocations".into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturedGraphRank {
    pub declared_graphs: String,
    pub executables: u64,
    pub bytes: u64,
}

impl CapturedGraphRank {
    pub fn measure<K>(rank: u8, declared: &DeclaredGraphSet<K>, actual: &[K],
        before: GraphMemorySample, after: GraphMemorySample) -> Result<Self, String>
        where K: std::hash::Hash + Eq + Clone + std::fmt::Debug + Serialize {
        declared.verify(actual.iter())?;
        Ok(Self { declared_graphs: declared.fingerprint(rank)?, executables: actual.len() as u64,
            bytes: after.retained_since(before)? })
    }
}

fn estimated_graph_basis() -> Basis { Basis::Estimated }

/// One rank's graph executables. Unqualified historical arithmetic remains an
/// estimate until an exact compatible census or this process's capture replaces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphRank {
    pub executables: u64,
    pub bytes: u64,
    /// Legacy startup consumers still split a margin into growth; migrated
    /// startup owners have zero margin. Lazy budgets retain their own lifetime.
    pub margin: u64,
    #[serde(default = "estimated_graph_basis")]
    pub basis: Basis,
}

impl GraphRank {
    pub fn estimated(executables: u64, bytes: u64) -> Self {
        Self { executables, bytes, margin: 0, basis: Basis::Estimated }
    }

    /// Historical whole-capture arithmetic is only a fallback estimate. It
    /// carries no percentage, minimum or maximum-of-samples adjustment.
    pub fn estimate(executables: u64, observed_bytes: u64, observed_executables: u64) -> Result<Self, String> {
        if observed_executables == 0 { return Err("empty startup graph observation".into()); }
        let bytes = (u128::from(executables) * u128::from(observed_bytes))
            .div_ceil(u128::from(observed_executables));
        Ok(Self::estimated(executables, bytes.try_into().map_err(|_| "startup graph estimate overflow")?))
    }

    /// A stale or absent row never becomes calibrated merely because counts
    /// agree. The fallback is explicit estimate-then-exact, without padding.
    pub fn from_census(identity: &GraphCensusIdentity, executables: u64,
        census: &[GraphCensus], estimated_bytes: u64) -> Result<Self, String> {
        identity.validate()?;
        let matches: Vec<_> = census.iter().filter(|row|
            row.identity == *identity && row.executables == executables).collect();
        match matches.as_slice() {
            [] => Ok(Self::estimated(executables, estimated_bytes)),
            [row] if !row.evidence.trim().is_empty() && !row.producing_commit.trim().is_empty() =>
                Ok(Self { bytes: row.bytes, basis: Basis::Calibrated, ..Self::estimated(executables, 0) }),
            _ => Err("duplicate or unattributed startup graph census".into()),
        }
    }
}

/// The decode graph set a family captures, per rank (lead first). Exact census
/// rows or explicit estimates forecast residency; synchronized capture supplies
/// the live owner. Legacy consumers still retain margins until migrated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSet {
    pub ranks: Vec<GraphRank>,
    pub lifetime: Lifetime,
    /// Distinct step shapes (rows x table geometry) the set covers.
    pub shapes: u64,
}

/// Exact startup keys paired with the admission ledger. Keys describe immutable
/// launch geometry, never process-local pointer addresses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupGraphs<K> {
    pub inventory: GraphSet,
    pub keys: Vec<K>,
}

/// An independently declared exact executable-key set. Memory observations and
/// node counts never participate in its identity. Keys must serialize stable
/// launch geometry, with no pointer addresses or process-local identifiers.
#[derive(Debug, Clone)]
pub struct DeclaredGraphSet<K> {
    keys: std::collections::HashSet<K>,
}

impl<K: std::hash::Hash + Eq + Clone + std::fmt::Debug> DeclaredGraphSet<K> {
    pub fn new(keys: &[K]) -> Result<Self, String> {
        let declared = keys.iter().cloned().collect::<std::collections::HashSet<_>>();
        if declared.len() != keys.len() { return Err("duplicate startup graph key".into()); }
        Ok(Self { keys: declared })
    }

    pub fn contains(&self, key: &K) -> bool { self.keys.contains(key) }
    pub fn len(&self) -> usize { self.keys.len() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }

    pub fn verify<'a>(&self, actual: impl IntoIterator<Item = &'a K>) -> Result<(), String>
        where K: 'a {
        let mut seen = std::collections::HashSet::new();
        for key in actual {
            if !seen.insert(key) || !self.contains(key) {
                return Err(format!("unexpected or duplicate startup graph key: {key:?}"));
            }
        }
        if seen.len() != self.len() {
            return Err(format!("startup graph coverage differs: declared {}, captured {}", self.len(), seen.len()));
        }
        Ok(())
    }

    /// Sort the serialized keys and length-frame every entry, so declaration
    /// order and randomized HashSet iteration do not affect a rank's identity.
    pub fn fingerprint(&self, rank: u8) -> Result<String, String> where K: Serialize {
        use sha2::{Digest, Sha256};
        let mut encoded = self.keys.iter().map(serde_json::to_vec)
            .collect::<Result<Vec<_>, _>>().map_err(|error| error.to_string())?;
        encoded.sort_unstable();
        if encoded.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("distinct startup graph keys serialize identically".into());
        }
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-declared-graphs-v1\0");
        hash.update([rank]);
        hash.update((encoded.len() as u64).to_le_bytes());
        for key in encoded {
            hash.update((key.len() as u64).to_le_bytes());
            hash.update(key);
        }
        Ok(format!("{:x}", hash.finalize()))
    }
}

/// A measured V4.1 executable class on one lane/device. bindings counts stable
/// source/table geometries of this exact row shape, not a padding bucket.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct V41GraphShape {
    pub device: u8,
    pub lane: u8,
    pub site: String,
    pub layer: Option<u8>,
    pub rows: u32,
    pub bindings: u32,
}

impl GraphSet {
    /// Replace planning estimates with this process's synchronized capture. No
    /// reserve or count-only match can stand in for the selected executable keys.
    pub fn captured(&self, declared: &[String], captured: &[CapturedGraphRank]) -> Result<Self, String> {
        if self.lifetime != Lifetime::Startup || declared.len() != self.ranks.len()
            || captured.len() != self.ranks.len() {
            return Err("startup graph capture rank inventory differs from admission".into());
        }
        let ranks = self.ranks.iter().zip(declared).zip(captured).map(|((rank, expected), actual)| {
            if expected.is_empty() || actual.declared_graphs != *expected || actual.executables != rank.executables {
                return Err("startup graph capture keys differ from admission".into());
            }
            Ok(GraphRank { executables: actual.executables, bytes: actual.bytes,
                margin: 0, basis: Basis::Calibrated })
        }).collect::<Result<_, String>>()?;
        Ok(Self { ranks, lifetime: Lifetime::Startup, shapes: self.shapes })
    }

    /// Couple a family's exact warm keys to the very ledger charged by its
    /// planner. Refuse a growth ledger or a shape-count mismatch at startup.
    pub fn startup<K>(self, keys: Vec<K>) -> Result<StartupGraphs<K>, String> {
        if self.lifetime != Lifetime::Startup {
            return Err("cannot warm a graph-growth inventory".into());
        }
        if self.shapes != keys.len() as u64 {
            return Err(format!("startup shape inventory {} differs from warm list {}", self.shapes, keys.len()));
        }
        Ok(StartupGraphs { inventory: self, keys })
    }

    /// Build V4.1's admission from a measured census; deliberately no guessed
    /// default census or bytes-per-executable. The serving warm-up and planner
    /// consume this same result once the hardware census is qualified.
    pub fn v41_startup(shapes: Vec<V41GraphShape>, per_executable: &[u64], margin_percent: u64,
        minimum_margin: u64) -> Result<StartupGraphs<V41GraphShape>, String> {
        let mut counts = vec![0u64; per_executable.len()];
        let mut seen = std::collections::HashSet::new();
        for shape in &shapes {
            if shape.rows == 0 || shape.rows > 4096 || shape.bindings == 0
                || shape.lane > 1 || !seen.insert((shape.device, shape.lane, shape.site.clone(), shape.layer, shape.rows)) {
                return Err("invalid or duplicate V4.1 startup graph shape".into());
            }
            let count = counts.get_mut(shape.device as usize).ok_or("V4.1 startup graph device outside inventory")?;
            *count = count.checked_add(u64::from(shape.bindings)).ok_or("V4.1 graph count overflow")?;
        }
        let ranks = counts.into_iter().zip(per_executable).map(|(count, &each)| {
            let set = Self::new(&[count], each, margin_percent, minimum_margin, 0, Lifetime::Startup);
            set.ranks[0]
        }).collect();
        Self { ranks, lifetime: Lifetime::Startup, shapes: shapes.len() as u64 }.startup(shapes)
    }

    /// `executables` per rank at `per_executable` bytes, plus `margin_percent`
    /// of that (at least `minimum_margin`) per rank.
    pub fn new(executables: &[u64], per_executable: u64, margin_percent: u64, minimum_margin: u64,
        shapes: u64, lifetime: Lifetime) -> Self {
        let ranks = executables.iter().map(|&count| {
            let measured = count.saturating_mul(per_executable);
            let margin = measured.saturating_mul(margin_percent).div_ceil(100).max(minimum_margin);
            GraphRank { executables: count, bytes: measured.saturating_add(margin), margin, basis: Basis::Estimated }
        }).collect();
        Self { ranks, lifetime, shapes }
    }

    /// A lazily captured set bounded by a byte budget per rank.
    pub fn budget(per_rank: &[u64]) -> Self {
        Self { ranks: per_rank.iter().map(|&bytes| GraphRank { executables: 0, bytes, margin: bytes, basis: Basis::Exact }).collect(),
            lifetime: Lifetime::Growth, shapes: 0 }
    }

    pub fn bytes(&self, rank: usize) -> u64 {
        self.ranks.get(rank).map_or(0, |r| r.bytes)
    }

    /// The graph bytes `rank`'s ready ledger holds: the captured estimate of a
    /// startup set, nothing of a growth set (lazy captures happen after ready).
    /// These graph bytes are separate from context, selected modules and BLAS owners.
    pub fn at_ready(&self, rank: usize) -> u64 {
        match (self.lifetime, self.ranks.get(rank)) {
            (Lifetime::Startup, Some(r)) => r.bytes - r.margin,
            _ => 0,
        }
    }

    /// Reserved past ready on `rank`: a startup set's margin, a growth set's
    /// whole budget. `at_ready + growth == bytes`.
    pub fn growth(&self, rank: usize) -> u64 {
        self.bytes(rank) - self.at_ready(rank)
    }

    /// The solver demand for this set on `rank`; lazy budgets are post-ready reserves.
    pub fn demand(&self, rank: usize) -> super::Demand {
        use cuteafd_core::memory_layout::ItemLifetime;
        let group = match self.lifetime { Lifetime::Startup => "graphs", Lifetime::Growth => "graph growth" };
        super::Demand::new(rank as u8, Category::Runtime, group, self.bytes(rank),
            self.ranks.get(rank).map_or(Basis::Estimated, |rank| rank.basis))
            .with_lifetime(match self.lifetime {
                Lifetime::Startup => ItemLifetime::ReadyResident,
                Lifetime::Growth => ItemLifetime::PostReadyReserve,
            })
    }

    /// The planner items for `rank`: the captured estimate at ready ("graphs")
    /// and the margin (or a lazy set's budget) as "graph growth".
    pub fn items(&self, rank: usize) -> Vec<cuteafd_core::memory_layout::Item> {
        use cuteafd_core::memory_layout::{Item, ItemLifetime};
        if rank >= self.ranks.len() { return Vec::new() }
        [("graphs", self.at_ready(rank), ItemLifetime::ReadyResident),
            (GRAPH_GROWTH, self.growth(rank), ItemLifetime::PostReadyReserve)]
            .into_iter().filter(|(_, bytes, _)| *bytes > 0)
            .map(|(group, bytes, lifetime)| Item::new(Category::Runtime, group, "", bytes, self.ranks[rank].basis)
                .with_lifetime(lifetime)).collect()
    }
}

/// Exact scratch of an `fp8moe` expert package (FP8, NVFP4, NVFP4-A4) at the
/// smallest compiled capacity holding `rows`, from its `manifest.json`
/// (`layouts.<layout>.capacities[].scratch_bytes`), as the runtime's
/// `Fp8MoeModule::scratch_bytes` returns it. No CUDA.
pub fn fp8moe_scratch_bytes(manifest: &Value, layout: &str, rows: u64) -> Option<u64> {
    let capacities = manifest["layouts"][layout]["capacities"].as_array()?;
    capacities.iter().filter_map(|c| Some((c["capacity"].as_u64()?, c["scratch_bytes"].as_u64()?)))
        .filter(|&(capacity, _)| capacity >= rows).min_by_key(|&(capacity, _)| capacity)
        .map(|(_, bytes)| bytes.max(256))
}

/// The `fp8moe` package a coordinator loads for `format` experts of geometry
/// `family` (`fp8-<family>[-nvfp4a4]`, mandatory W4A4 for NVFP4, as
/// `shared::experts::fp8::package_directory`), under `lib`'s `fp8/` tree, and
/// its scratch for `rows` rows.
pub fn fp8moe_package_scratch(lib: &std::path::Path, family: &str,
    format: crate::formats::fp8_experts::ExpertFormat, rows: u64) -> Option<(String, u64)> {
    let name = format.package_name(family);
    let manifest: Value = serde_json::from_slice(&std::fs::read(lib.join("fp8").join(&name).join("manifest.json")).ok()?).ok()?;
    Some((name, fp8moe_scratch_bytes(&manifest, "tp1", rows)?))
}

/// Mandatory native W4A4 dense package; missing exports never select W4A16.
pub fn dense_package_directory(lib: &std::path::Path, geometry: &str) -> std::path::PathBuf {
    lib.join("fp8").join(crate::formats::fp8_experts::ExpertFormat::Nvfp4.package_name(geometry)).join("tp1")
}

pub fn dense_package_scratch(lib: &std::path::Path, geometry: &str, rows: u64) -> Option<u64> {
    let directory = dense_package_directory(lib, geometry);
    let manifest: Value = serde_json::from_slice(&std::fs::read(directory.parent()?.join("manifest.json")).ok()?).ok()?;
    fp8moe_scratch_bytes(&manifest, "tp1", rows)
}

/// The image's `lib/` directory next to a `share/PROGRAMS.json` manifest
/// (`/opt/cuteafd/share/PROGRAMS.json` -> `/opt/cuteafd/lib`).
pub fn image_lib(manifest: Option<&std::path::Path>) -> std::path::PathBuf {
    let manifest = manifest.unwrap_or(std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    manifest.parent().map(|share| share.join("../lib")).unwrap_or_else(|| "/opt/cuteafd/lib".into())
}

/// The EXL3 capacities a local executor compiles for at most `rows` live rows:
/// every exported capacity up to the first at or above `rows`, each once.
/// Multi-row decode also admits m80 for canonical m1 reduction geometry.
pub fn exl3_capacities(rows: u64) -> Vec<u64> {
    let capacities = cuteafd_core::EXL3_CAPACITIES.map(u64::from);
    let required = if rows > 1 { rows.max(80) } else { 1 };
    let top = capacities.into_iter().find(|&c| c >= required).unwrap_or(4096);
    capacities.into_iter().filter(|&c| c <= top).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn startup_graph_census_requires_exact_attributed_identity() {
        let identity = GraphCensusIdentity { declared_graphs: "keys".into(), programs_sha256: "programs".into(),
            native_sha256: "native".into(), toolchain: "compiler".into(), gpu_class: "sm120/188/96G".into(),
            driver: "13020".into(), layout: "layout".into() };
        let census = GraphCensus { identity: identity.clone(), executables: 2, bytes: 101,
            evidence: "synthetic capture".into(), producing_commit: "producer-A".into() };
        let used = GraphRank::from_census(&identity, 2, &[census.clone()], 200).unwrap();
        assert_eq!((used.bytes, used.margin, used.basis), (101, 0, Basis::Calibrated));
        for field in 0..7 {
            let mut stale = identity.clone();
            match field {
                0 => stale.declared_graphs.push('x'),
                1 => stale.programs_sha256.push('x'),
                2 => stale.native_sha256.push('x'),
                3 => stale.toolchain.push('x'),
                4 => stale.gpu_class.push('x'),
                5 => stale.driver.push('x'),
                _ => stale.layout.push('x'),
            }
            assert_eq!(GraphRank::from_census(&stale, 2, &[census.clone()], 200).unwrap(),
                GraphRank::estimated(2, 200));
        }
        assert_eq!(GraphRank::from_census(&identity, 3, &[census.clone()], 200).unwrap(),
            GraphRank::estimated(3, 200));
        assert!(GraphRank::from_census(&identity, 2, &[census.clone(), census.clone()], 200).is_err());
        let mut unattributed = census.clone();
        unattributed.evidence.clear();
        assert!(GraphRank::from_census(&identity, 2, &[unattributed], 200).is_err());
        let mut unattributed = census.clone();
        unattributed.producing_commit.clear();
        assert!(GraphRank::from_census(&identity, 2, &[unattributed], 200).is_err());
        let mut incomplete = identity.clone();
        incomplete.native_sha256.clear();
        assert!(GraphRank::from_census(&incomplete, 2, &[], 200).is_err());
        assert_eq!(serde_json::from_slice::<GraphCensus>(&serde_json::to_vec(&census).unwrap()).unwrap(), census);
        assert_eq!(GraphRank::estimate(3, 10, 2).unwrap(), GraphRank::estimated(3, 15));
        assert!(GraphRank::estimate(3, 10, 0).is_err());
        assert!(GraphRank::estimate(u64::MAX, u64::MAX, 1).is_err());
    }

    #[test]
    fn startup_capture_excludes_tracked_storage_and_checks_actual_keys() {
        let before = GraphMemorySample { total: 1000, free: 900, tracked_device: 100,
            tracked_pinned: 10, tracked_managed: 0 };
        let after = GraphMemorySample { free: 600, tracked_device: 300, ..before };
        assert_eq!(after.retained_since(before).unwrap(), 100);
        assert_eq!(GraphMemorySample { free: 950, tracked_device: 0, ..before }
            .retained_since(before).unwrap(), 50);
        for bad in [GraphMemorySample { total: 1001, ..after },
            GraphMemorySample { free: 1001, ..after }, GraphMemorySample { tracked_pinned: 11, ..after },
            GraphMemorySample { tracked_managed: 1, ..after }, GraphMemorySample { free: 950, ..before }] {
            assert!(bad.retained_since(before).is_err());
        }
        let keys = [1u32, 2];
        let declared = DeclaredGraphSet::new(&keys).unwrap();
        let captured = CapturedGraphRank::measure(0, &declared, &[2, 1], before, after).unwrap();
        assert_eq!(captured.bytes, 100);
        for actual in [vec![1], vec![1, 3], vec![1, 1]] {
            assert!(CapturedGraphRank::measure(0, &declared, &actual, before, after).is_err());
        }
        let estimated = GraphSet { ranks: vec![GraphRank::estimated(2, 10)], shapes: 1, lifetime: Lifetime::Startup };
        let fingerprints = [declared.fingerprint(0).unwrap()];
        let measured = estimated.captured(&fingerprints, &[captured.clone()]).unwrap();
        assert_eq!(measured.ranks[0], GraphRank { executables: 2, bytes: 100, margin: 0, basis: Basis::Calibrated });
        assert!(estimated.captured(&["other keys".into()], &[captured.clone()]).is_err());
        assert!(estimated.captured(&fingerprints, &[]).is_err());
        assert!(GraphSet::budget(&[10]).captured(&fingerprints, &[captured]).is_err());
    }

    fn measured_figure() -> MemoryFigure {
        MemoryFigure { bytes: 101, attribution: FigureAttribution::Measured,
            evidence: "synthetic census", producing_commit: "producer-A",
            identity: FigureIdentity { declared_graphs: "declared", programs_sha256: "programs",
                native_sha256: "native", toolchain: "compiler", gpu_class: "card",
                driver: "driver", layout: "layout" } }
    }

    #[test]
    fn maintained_figures_use_unchanged_bytes_and_ignore_unrelated_producing_commits() {
        let figure = measured_figure();
        let used = |figure| FigureUse { name: "selected", figure, current: measured_figure().identity };
        assert_eq!(census_recommendation([used(figure)]), None);
        assert_eq!(census_recommendation([used(MemoryFigure { producing_commit: "producer-B", ..figure })]), None);
        let mut current = figure.identity;
        for part in 0..7 {
            match part {
                0 => current.declared_graphs = "changed",
                1 => current.programs_sha256 = "changed",
                2 => current.native_sha256 = "changed",
                3 => current.toolchain = "changed",
                4 => current.gpu_class = "changed",
                5 => current.driver = "changed",
                _ => current.layout = "changed",
            }
            let stale = figure.identity.stale_parts(current);
            assert_eq!(stale.len(), 1);
            let message = census_recommendation([FigureUse { current, ..used(figure) }]).unwrap();
            assert!(message.contains(stale[0]));
            assert!(message.contains("using maintained bytes unchanged"));
            assert_eq!(figure.bytes, 101);
            current = figure.identity;
        }
    }

    #[test]
    fn maintained_figures_aggregate_only_consumed_estimates_and_keep_derivations() {
        let measured = measured_figure();
        let estimated = measured.scaled(3, 2, "declared key-count ratio").unwrap();
        assert_eq!(estimated.bytes, 152);
        assert_eq!(estimated.identity, measured.identity);
        assert_eq!(estimated.evidence, measured.evidence);
        assert!(measured.scaled(1, 0, "ratio").is_err());
        assert!(measured.scaled(1, 1, "").is_err());
        assert!(MemoryFigure { bytes: u64::MAX, ..measured }.scaled(2, 1, "ratio").is_err());
        assert_eq!(census_recommendation([]), None);
        let used = FigureUse { name: "selected estimate", figure: estimated, current: measured.identity };
        let message = census_recommendation([used, used]).unwrap();
        assert_eq!(message.matches("selected estimate").count(), 1);
        assert!(message.contains("synthetic census"));
        assert!(message.contains("declared key-count ratio"));
        // An unconsumed alternative never enters this after-solve aggregation.
        assert_eq!(census_recommendation([FigureUse { name: "selected measured",
            figure: measured, current: measured.identity }]), None);
    }

    #[test]
    fn native_package_names_and_scratch_share_the_format_contract() {
        use crate::formats::fp8_experts::ExpertFormat;
        let dir = tempfile::tempdir().unwrap();
        for family in ["glm", "glmf", "qwen4", "step5"] {
            let name = ExpertFormat::Nvfp4.package_name(family);
            assert_eq!(name, format!("fp8-{family}-nvfp4a4"));
            let root = dir.path().join("fp8").join(&name);
            std::fs::create_dir_all(root.join("tp1")).unwrap();
            std::fs::write(root.join("manifest.json"), json!({"layouts":{"tp1":{
                "capacities":[{"capacity":16,"scratch_bytes":1024}]}}}).to_string()).unwrap();
            assert_eq!(fp8moe_package_scratch(dir.path(), family, ExpertFormat::Nvfp4, 16),
                Some((name, 1024)));
            assert_eq!(dense_package_directory(dir.path(), family), root.join("tp1"));
            assert!(fp8moe_package_scratch(dir.path(), family, ExpertFormat::Nvfp4, 17).is_none());
            std::fs::remove_file(root.join("manifest.json")).unwrap();
            assert!(fp8moe_package_scratch(dir.path(), family, ExpertFormat::Nvfp4, 1).is_none());
        }
    }

    #[test]
    fn declared_graph_identity_is_order_independent_ranked_and_exact() {
        let keys = [(0u32, 1u32, 16u32), (1, 4, 32), (2, 64, 256)];
        let declared = DeclaredGraphSet::new(&keys).unwrap();
        let reversed = DeclaredGraphSet::new(&keys.into_iter().rev().collect::<Vec<_>>()).unwrap();
        let fingerprint = declared.fingerprint(0).unwrap();
        assert_eq!(fingerprint, reversed.fingerprint(0).unwrap());
        assert_ne!(fingerprint, declared.fingerprint(1).unwrap());
        declared.verify(keys.iter().rev()).unwrap();
        assert!(declared.verify(keys[..2].iter()).is_err());
        assert!(declared.verify(keys.iter().chain([&(3, 1, 16)])).is_err());
        assert!(declared.verify(keys.iter().chain(keys[..1].iter())).is_err());
        assert!(DeclaredGraphSet::new(&[keys[0], keys[0]]).is_err());
        let changed = DeclaredGraphSet::new(&[(0, 1, 16), (1, 4, 32), (2, 64, 512)]).unwrap();
        assert_ne!(fingerprint, changed.fingerprint(0).unwrap());
        let empty = DeclaredGraphSet::<u32>::new(&[]).unwrap();
        assert!(empty.is_empty());
        empty.verify([].iter()).unwrap();
        assert_ne!(empty.fingerprint(0).unwrap(), empty.fingerprint(1).unwrap());
    }

    #[test]
    fn declared_graph_identity_rejects_lossy_key_serialization() {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
        struct Lossy { stable: u32, #[serde(skip)] omitted: u32 }
        let keys = [Lossy { stable: 1, omitted: 0 }, Lossy { stable: 1, omitted: 1 }];
        assert!(DeclaredGraphSet::new(&keys).unwrap().fingerprint(0).is_err());
    }

    #[test]
    fn startup_keys_match_planner_shapes_and_v41_rank_bytes() {
        let shape = |device, lane, rows| V41GraphShape {
            device, lane, site: "query".into(), layer: Some(3), rows, bindings: 2,
        };
        let keys = vec![shape(0, 0, 1), shape(0, 1, 6), shape(1, 0, 1)];
        let startup = GraphSet::v41_startup(keys.clone(), &[10, 20], 10, 0).unwrap();
        assert_eq!(startup.keys, keys);
        assert_eq!(startup.inventory.ranks.iter().map(|r| (r.executables, r.bytes, r.margin)).collect::<Vec<_>>(),
            vec![(4, 44, 4), (2, 44, 4)]);
        assert!(GraphSet::budget(&[100]).startup(vec![1]).is_err());
        assert!(GraphSet::new(&[2], 10, 0, 0, 2, Lifetime::Startup).startup(vec![1]).is_err());
        assert!(GraphSet::v41_startup(vec![shape(0, 0, 1), shape(0, 0, 1)], &[10], 0, 0).is_err());
        assert!(GraphSet::v41_startup(vec![shape(2, 0, 1)], &[10], 0, 0).is_err());
    }

    #[test]
    fn startup_and_lazy_graph_owners_partition_ready_and_growth() {
        // Qwen EXL3 1 RTX, v3-p2 ready ledger: untracked 5,936,332,180 B with 30,380 startup graphs
        // of 4,569,694,208 B. Startup graphs count at ready, their margin is growth; a lazy set
        // counts nothing at ready and its whole budget is growth.
        let startup = GraphSet::new(&[30_380], 150_418, 10, 256 << 20, 620, Lifetime::Startup);
        let lazy = GraphSet::budget(&[400 << 20]);
        for set in [&startup, &lazy] {
            assert_eq!(set.at_ready(0) + set.growth(0), set.bytes(0));
            let items = set.items(0);
            let sum = |group: &str| items.iter().filter(|i| i.group == group).map(|i| i.bytes).sum::<u64>();
            assert_eq!((sum("graphs"), sum(GRAPH_GROWTH)), (set.at_ready(0), set.growth(0)));
        }
        assert_eq!(lazy.at_ready(0), 0);

    }

    #[test]
    fn uncalibrated_packages_use_positive_manifest_formula() {
        let arch = &ARCH_CONTEXTS[1];
        let programs = ProgramSet::from_names(&["qwen4"], ["qwen4_decode", "other_decode"]);
        assert_eq!(programs.names, ["qwen4_decode"]);
        let allowance = loaded_code_allowance("qwen4", "unmeasured", false, 0, arch, Some(&programs));
        assert_eq!(allowance.basis, Basis::Estimated);
        assert_eq!(allowance.bytes, programs.module_bytes());
        assert!(allowance.bytes > 0);
        assert_eq!(allowance.pending(arch.context_bytes, arch.context_bytes), allowance.bytes);
        assert_eq!(allowance.pending(arch.context_bytes + 100, arch.context_bytes), allowance.bytes - 100);
        assert_eq!(allowance.pending(u64::MAX, arch.context_bytes), 0);
    }

    #[test]
    fn step5_code_allowance_does_not_duplicate_conditional_vocabulary_handles() {
        let arch = &ARCH_CONTEXTS[1];
        let programs = ProgramSet::from_names(&["step5", "step52"], ["step5_norm", "step52_norm"]);
        for family in ["step5", "step52"] {
            let allowance = loaded_code_allowance(family, "exl3", true, 0, arch, Some(&programs));
            assert_eq!(allowance.bytes, programs.module_bytes());
            assert_eq!(allowance.pending(u64::MAX, arch.context_bytes), 0);
        }
    }

    #[test]
    fn loaded_modules_cannot_spend_a_future_library_reservation() {
        let arch = &ARCH_CONTEXTS[1];
        let programs = ProgramSet::from_names(&["unmeasured"], ["unmeasured_decode"]);
        let allowance = loaded_code_allowance("unmeasured", "exl3", true, 0, arch, Some(&programs));
        // Recorded prefix run: already-loaded modules exceeded the formula, but
        // the thread-local BLAS and private head handles were still future owners.
        for (rank, sampled, old_pending, admitted, steady) in [
            (0, 672_399_360, 29_360_128, 8_129_791_636, 8_166_309_888),
            (1, 680_787_968, 20_971_520, 7_457_793_680, 7_495_221_248),
        ] {
            let pending = allowance.pending(sampled, arch.context_bytes);
            assert_eq!(pending, arch.cublas_bytes);
            assert!(admitted + pending - old_pending >= steady, "rank {rank}");
        }
        let selected = loaded_code_allowance("mimo", "fp8", false, 0, arch, None);
        assert_eq!(selected.pending(u64::MAX, arch.context_bytes), 0);
    }

    #[test]
    fn migrated_code_estimates_exclude_blas_and_peer_allowances() {
        let arch = &ARCH_CONTEXTS[1];
        let programs = ProgramSet::from_names(&["mimo", "qwen4", "dsv4f", "glmf", "kimik3"],
            ["mimo_decode", "qwen4_decode", "dsv4f_decode", "glmf_decode", "kimik3_decode"]);
        for family in ["mimo", "mimop2", "qwen4", "glmf", "glmf-k6", "kimik3", "muse"] {
            for package in ["fp8", "exl3", "none", "unmeasured"] {
                for (split, rank) in [(false, 0), (true, 0), (true, 1)] {
                    let code = loaded_code_allowance(family, package, split, rank, arch, Some(&programs));
                    assert_eq!(code.basis, Basis::Estimated);
                    assert_eq!(code.bytes, programs.module_bytes());
                    assert_eq!(code.pending(u64::MAX, arch.context_bytes), 0);
                    let missing = loaded_code_allowance(family, package, split, rank, arch, None);
                    let empty = ProgramSet::from_names(&[family], std::iter::empty());
                    let runtime = loaded_code_allowance(family, package, split, rank, arch, Some(&empty));
                    assert_eq!(missing.bytes, runtime.bytes);
                    assert_eq!(missing.bytes, 34 * MIB);
                    assert_eq!(missing.basis, Basis::Estimated);
                }
            }
        }
        // GLM's unrelated conversion/upload residuals and existing formula are unchanged.
        for rank in 0..2 {
            let code = loaded_code_allowance("glm", "exl3", true, rank, arch, Some(&programs));
            assert_eq!(code.basis, Basis::Formula);
            assert_eq!(code.bytes, arch.cublas_bytes + programs.module_bytes() + arch.peer_context_bytes);
        }
    }

    #[test]
    fn v4_charges_selected_modules_not_a_mixed_residency_row() {
        let arch = &ARCH_CONTEXTS[1];
        let programs = ProgramSet::from_names(&["dsv4f", "dsv4p"], ["dsv4f_decode", "dsv4p_decode"]);
        for family in ["dsv4f", "dsv4f2", "dsv4p", "dsv4p2"] {
            for package in ["*", "exl3", "unmeasured"] {
                for (split, rank) in [(false, 0), (true, 0), (true, 1)] {
                    let code = loaded_code_allowance(family, package, split, rank, arch, Some(&programs));
                    assert_eq!(code.basis, Basis::Estimated);
                    assert_eq!(code.bytes, programs.module_bytes());
                    assert_eq!(code.pending(u64::MAX, arch.context_bytes), 0);
                    let missing = loaded_code_allowance(family, package, split, rank, arch, None);
                    assert_eq!(missing.bytes, 34 * MIB);
                    assert_eq!(missing.basis, Basis::Estimated);
                }
            }
        }
    }

    #[test]
    fn exl3_capacities_list_4096_once() {
        assert_eq!(exl3_capacities(4096), [1, 16, 80, 256, 1024, 2048, 2752, 4096]);
        assert_eq!(exl3_capacities(2752), [1, 16, 80, 256, 1024, 2048, 2752]);
        assert_eq!(exl3_capacities(64), [1, 16, 80]);
        assert_eq!(exl3_capacities(1), [1]);
        assert_eq!(exl3_capacities(2), [1, 16, 80]);
        assert_eq!(exl3_capacities(16), [1, 16, 80]);
    }

    #[test]
    fn package_scratch_takes_the_smallest_capacity_at_or_above_rows() {
        let manifest = json!({"layouts": {"tp1": {"capacities": [
            {"capacity": 1, "scratch_bytes": 99328}, {"capacity": 1024, "scratch_bytes": 91839488},
            {"capacity": 4096, "scratch_bytes": 661072896}]}}});
        assert_eq!(fp8moe_scratch_bytes(&manifest, "tp1", 4096), Some(661072896));
        assert_eq!(fp8moe_scratch_bytes(&manifest, "tp1", 1000), Some(91839488));
        assert_eq!(fp8moe_scratch_bytes(&manifest, "tp1", 8192), None);
    }

    #[test]
    fn graph_set_margins_and_lifetimes() {
        let set = GraphSet::new(&[12_397, 0], 149_712, 10, 256 << 20, 253, Lifetime::Startup);
        assert_eq!(set.bytes(0), 1_855_979_664 + (256 << 20));
        assert_eq!(set.items(0).iter().map(|i| (i.group.as_str(), i.bytes)).collect::<Vec<_>>(),
            [("graphs", 1_855_979_664), ("graph growth", 256 << 20)]);
        use cuteafd_core::memory_layout::ItemLifetime::{PostReadyReserve, ReadyResident};
        assert_eq!(set.items(0).iter().map(|i| i.lifetime).collect::<Vec<_>>(),
            [ReadyResident, PostReadyReserve]);
        assert_eq!((set.demand(0).group.as_str(), set.demand(0).lifetime), ("graphs", ReadyResident));
        let lazy = GraphSet::budget(&[1 << 30]);
        assert_eq!((lazy.demand(0).group.as_str(), lazy.demand(0).lifetime), ("graph growth", PostReadyReserve));
        assert_eq!(lazy.items(0)[0].lifetime, PostReadyReserve);
    }

    #[test]
    fn program_sets_select_the_family_and_its_split() {
        let manifest = json!({"programs": [{"name": "dsv4f_wo_m64", "scratch_bytes_at_capacity": {"s": 7}},
            {"name": "dsv4f2_wo_m64"}, {"name": "glmf_kda_w8_m4096", "scratch_bytes_at_capacity": {"s": 782}},
            {"name": "dsv4p_wo_m64"}]});
        let set = ProgramSet::from_manifest(&manifest, &["dsv4f", "dsv4f2"]);
        assert_eq!(set.names, ["dsv4f_wo_m64", "dsv4f2_wo_m64"]);
        assert_eq!(set.scratch, [("dsv4f_wo_m64".to_string(), 7)]);
        assert!(!set.contains("glmf_kda_w8_m4096") && set.contains("dsv4f2_x"));
    }

    #[test]
    fn plain_glm_excludes_native_stage_names_scratch_and_module_charge() {
        let manifest = json!({"programs": std::iter::once(json!({"name": "glm_producer_m64",
            "scratch_bytes_at_capacity": {"scratch": 7}})).chain((0..32).map(|i| json!({
                "name": format!("glm_stage_native_{i}"), "scratch_bytes_at_capacity": {"scratch": 999}
            }))).chain(std::iter::once(json!({"name": "glm2_stage_swiglu_i1024"}))).collect::<Vec<_>>()});
        for families in [&["glm"][..], &["glm", "glm2"][..]] {
            let set = ProgramSet::from_manifest(&manifest, families);
            let from_names = ProgramSet::from_names(families,
                manifest["programs"].as_array().unwrap().iter().filter_map(|p| p["name"].as_str()));
            assert_eq!(set.names, ["glm_producer_m64"]);
            assert_eq!(set.names, from_names.names);
            assert_eq!(set.scratch, [("glm_producer_m64".into(), 7)]);
            assert_eq!(set.module_bytes(), 34 * MIB);
            assert!(!set.contains("glm_stage_native_0") && !set.contains("glm2_stage_swiglu_i1024"));
        }
        let native = ProgramSet::from_manifest(&manifest, &["glm", "glm2", "glm_stage", "glm2_stage"]);
        assert_eq!(native.names.len(), 34);
        assert_eq!(native.module_bytes(), 38 * MIB);
        assert!(native.contains("glm_stage_native_0") && native.contains("glm2_stage_swiglu_i1024"));
    }

    #[test]
    fn v41_fp8_retains_aligned_capacity_arenas_and_selects_physical_sms() {
        let variants = [1, 16, 64, 80, 256, 1024, 4096].into_iter().flat_map(|capacity| {
            let plans: &[u64] = if capacity <= 16 { &[170, 188] } else { &[0] };
            plans.iter().map(move |&sms| json!({"capacity": capacity, "plan_sms": sms,
                "input_dim": 5120, "output_dim": 1280,
                "label": format!("v41_query_m{capacity}_sm{sms}"),
                "scratch_bytes": capacity * 257 + sms})).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let manifest = json!({"variants": variants});
        let expected = |rows: u64, sms: u64| {
            let covers = if matches!(rows, 40 | 80) { [1,16,80,256,1024,4096] } else { [1,16,64,256,1024,4096] };
            let compiled = covers.into_iter().find(|&n| n >= rows).unwrap();
            covers[..3].iter().copied().filter(|&n| n < compiled).chain([compiled])
                .map(|n| (n * 257 + if n <= 16 { sms } else { 0 }).div_ceil(256) * 256)
                .sum::<u64>()
        };
        for rows in [1,2,16,17,40,80,81,256,257,1024,1025,4096] {
            for sms in [170,188] {
                let set = ProgramSet::v41_fp8(&manifest, &[(5120,1280)], rows, sms).unwrap();
                assert_eq!(set.scratch.iter().map(|(_,n)| n).sum::<u64>(), expected(rows, sms as u64));
                assert!(set.scratch.iter().all(|(_,n)| n % 256 == 0));
                assert!(set.names.iter().filter(|name| !name.ends_with("sm0"))
                    .all(|name| name.ends_with(&format!("sm{sms}"))));
                if rows > 1 {
                    assert!(set.scratch.iter().map(|(_,n)| n).sum::<u64>()
                        > *set.scratch.iter().map(|(_,n)| n).max().unwrap());
                }
            }
        }
        assert!(ProgramSet::v41_fp8(&manifest, &[(5120,1280)], 0,188).is_err());
        assert!(ProgramSet::v41_fp8(&manifest, &[(5120,1280)], 4097,188).is_err());
        assert!(ProgramSet::v41_fp8(&manifest, &[(5120,1280)], 1,0).is_err());
        assert!(ProgramSet::v41_fp8(&manifest, &[(5120,1280);2], 1,188).is_err());
        let mut missing = manifest.clone();
        missing["variants"].as_array_mut().unwrap().remove(0);
        assert!(ProgramSet::v41_fp8(&missing, &[(5120,1280)], 80,170).is_err());
        let mut duplicate = manifest.clone();
        duplicate["variants"].as_array_mut().unwrap().push(manifest["variants"][0].clone());
        assert!(ProgramSet::v41_fp8(&duplicate, &[(5120,1280)], 80,170).is_err());
        let mut overflow = manifest.clone();
        overflow["variants"][0]["scratch_bytes"] = u64::MAX.into();
        assert!(ProgramSet::v41_fp8(&overflow, &[(5120,1280)], 1,170).is_err());
    }

    #[test]
    fn v41_output_reduction_is_immutable_across_retained_capacities() {
        for n in [2560, 5120] {
            let variants = [1,16,64,80,256,1024,4096].into_iter().map(|capacity|
                json!({"capacity":capacity, "plan_sms":0, "input_dim":8192, "output_dim":n,
                    "label":format!("v41_output_{n}_m{capacity}"), "scratch_bytes":capacity * 257,
                    "expected_m":if capacity <= 80 {16} else {capacity},
                    "split_k_slices":1, "all_rows":capacity <= 80})
            ).collect::<Vec<_>>();
            let manifest = json!({"variants":variants});
            for rows in [1,16,40,80,81,256,4096] {
                assert!(ProgramSet::v41_fp8(&manifest, &[(8192,n)], rows,188).is_ok());
            }
            for (field, value) in [("expected_m", json!(80)), ("split_k_slices", json!(2)),
                ("all_rows", json!(false))] {
                let mut changed = manifest.clone();
                changed["variants"][2][field] = value.clone();
                changed["variants"][3][field] = value;
                assert!(ProgramSet::v41_fp8(&changed, &[(8192,n)], 80,188).is_err(), "{field}");
                assert!(ProgramSet::v41_fp8(&changed, &[(8192,n)], 4096,188).is_err(), "{field}");
            }
            let mut missing_canonical = manifest.clone();
            missing_canonical["variants"].as_array_mut().unwrap().remove(1);
            assert!(ProgramSet::v41_fp8(&missing_canonical, &[(8192,n)], 1,188).is_err());
        }
    }

    #[test]
    fn simulated_small_cards_keep_the_physical_sm_count() {
        let pro = ArchContext::for_device("sm_120", 96 << 30);
        assert_eq!(pro.sms, 188);
        assert_eq!(ArchContext::coordinator(32 << 30, Some(pro)).sms, 188);
        assert_eq!(ArchContext::coordinator(32 << 30, None).sms, 170);
    }
}
