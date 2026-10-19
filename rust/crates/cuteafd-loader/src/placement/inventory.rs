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
/// `--coordinator-gpu-budget-gib`: the RTX PRO 6000's CUDA-reported total
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
    /// simulated small card (`--coordinator-gpu-budget-gib` on a PRO) keeps the
    /// PRO's SM count: the simulation caps memory only.
    pub fn coordinator(total_bytes: u64, simulated_on: Option<&'static ArchContext>) -> ArchContext {
        let class = *Self::for_device("sm_120", total_bytes);
        match simulated_on { Some(physical) => ArchContext { sms: physical.sms, ..class }, None => class }
    }
}

/// Native code a family's serving process holds on a coordinator GPU at ready,
/// beyond [`ArchContext::context_bytes`] and outside its tracked allocations
/// and captured startup graphs: the selected program modules
/// (`CUDA_MODULE_LOADING=LAZY` loads each function's code on first launch),
/// the cuBLAS handle, expert-package modules (fp8moe/EXL3 libraries), the
/// transport's device mappings and the runtime's per-stream bookkeeping.
/// One definition for both sides: the ready ledger (the `stage: "ready"`
/// report serve logs at readiness, before any request) splits its untracked
/// bytes as `context + loaded code + GraphSet::at_ready`, so
/// `loaded code = untracked at ready - startup graphs - context`. Lazily
/// captured graphs (V4's attention graphs, any `Lifetime::Growth` set) come
/// after ready: never in this number, always charged as "graph growth".
/// Measured per serving configuration (v3-p2 cards, RTX PRO 6000 SM120,
/// driver 595.91.07, CUDA 13.2, 2026-10-10). The two-RTX `dsv4` entries are
/// TP2 expert halves (P4), measured at tagged ready on 2026-10-11.
///
/// The planner charges `bytes` (with the context) as the GPU's runtime
/// baseline. The runtime's admission sample already holds part of it (the
/// modules, an expert package, cuBLAS when a weight load ran a GEMM), so serve
/// reserves only what is still missing at its sample ([`LoadedCode::pending`]).
/// Both sides then reach the same ready ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LoadedCode {
    /// Program family: `qwen4`, `mimo` (all MiMo sizes), `dsv4` (Flash and
    /// Pro), `glmf`.
    pub family: &'static str,
    /// Local expert package on this GPU: `exl3`, `fp8` (fp8moe/NVFP4), `none`
    /// (Spark experts), or `*` (measured not to depend on it).
    pub experts: &'static str,
    /// Head-split layout (two coordinator GPUs).
    pub split: bool,
    /// 0: lead GPU, 1: the head-split peer.
    pub rank: u8,
    pub bytes: u64,
    pub source: &'static str,
}

/// The measured table ([`loaded_code`]).
pub const LOADED_CODE: &[LoadedCode] = &[
    LoadedCode { family: "qwen4", experts: "exl3", split: false, rank: 0, bytes: 780_221_844,
        source: "qwen38-exl3-min: untracked 5,936,332,180 - graphs 4,569,694,208 - context 586,416,128" },
    LoadedCode { family: "qwen4", experts: "fp8", split: false, rank: 0, bytes: 1_089_736_388,
        source: "qwen38-nvfp4-min: untracked 6,245,846,724 - graphs 4,569,694,208 - context 586,416,128" },
    LoadedCode { family: "mimo", experts: "*", split: false, rank: 0, bytes: 261_483_520,
        source: "mimo26-flash-min: untracked 847,899,648 - context 586,416,128" },
    LoadedCode { family: "mimo", experts: "*", split: true, rank: 0, bytes: 365_259_264,
        source: "mimo26-flash-max rtx0: untracked 951,675,392 - context 586,416,128" },
    LoadedCode { family: "mimo", experts: "*", split: true, rank: 1, bytes: 165_460_992,
        source: "mimo26-flash-max rtx1: untracked 751,877,120 - context 586,416,128" },
    LoadedCode { family: "dsv4", experts: "*", split: false, rank: 0, bytes: 322_050_368,
        source: "v4-flash-sim5090 (0 RTX layers): untracked 908,466,496 - context 586,416,128; \
            v4-flash-min p0 (18 FP8 layers) 309,476,320" },
    LoadedCode { family: "dsv4", experts: "exl3", split: true, rank: 0, bytes: 1_059_185_592,
        source: "v4-pro-exl3 TP2 tagged ready rtx0 (b367a34f, 2026-10-11): untracked 1,645,601,720 - context 586,416,128" },
    LoadedCode { family: "dsv4", experts: "exl3", split: true, rank: 1, bytes: 885_694_212,
        source: "v4-pro-exl3 TP2 tagged ready rtx1 (b367a34f, 2026-10-11): untracked 1,472,110,340 - context 586,416,128" },
    LoadedCode { family: "dsv4", experts: "exl3", split: false, rank: 0, bytes: 724_349_124,
        source: "v4-pro-exl3-min (1 RTX + 4 Sparks, 3 EXL3 layers): untracked 1,310,765,252 - context 586,416,128" },
    LoadedCode { family: "dsv4", experts: "*", split: true, rank: 0, bytes: 1_007_333_728,
        source: "v4-flash TP2 tagged ready rtx0 (b367a34f, 2026-10-11): untracked 1,593,749,856 - context 586,416,128" },
    LoadedCode { family: "dsv4", experts: "*", split: true, rank: 1, bytes: 803_054_544,
        source: "v4-flash TP2 tagged ready rtx1 (b367a34f, 2026-10-11): untracked 1,389,470,672 - context 586,416,128" },
    LoadedCode { family: "glmf", experts: "*", split: false, rank: 0, bytes: 436_389_024,
        source: "glm53f-exl3-min: untracked 2,968,962,208 - graphs 1,946,157,056 - context 586,416,128" },
    LoadedCode { family: "glmf", experts: "exl3-tp2-local", split: true, rank: 0, bytes: 441_094_988,
        source: "P6 staged EXL3 TP2 0-Spark startup, 2026-10-12: named program/load/exchange/package/workspace terms; pre-capture 1,027,511,116 - context 586,416,128" },
    LoadedCode { family: "glmf", experts: "exl3-tp2-local", split: true, rank: 1, bytes: 314_257_228,
        source: "P6 staged EXL3 TP2 0-Spark startup, 2026-10-12: named second-context/program/load/exchange/package/workspace terms; pre-capture 900,673,356 - context 586,416,128" },
    LoadedCode { family: "glmf", experts: "*", split: true, rank: 0, bytes: 500_143_488,
        source: "glm53f-exl3-max rtx0: untracked 3,317,929,344 - graphs 2,231,369,728 - context 586,416,128" },
    LoadedCode { family: "glmf", experts: "*", split: true, rank: 1, bytes: 215_091_456,
        source: "glm53f-exl3-max rtx1: untracked 3,085,306,112 - graphs 2,283,798,528 - context 586,416,128" },
];

/// Physical untracked startup deltas for the P6 EXL3 TP2 / zero-Spark path.
/// SM120 PRO 6000, CUDA 13.2 LAZY, row128 KDA/output shard/FP8 head, C8,
/// 4096 prefill, 64 decode; staged probe 2026-10-12, source f5193e78.
/// Package term includes the route exchange's 524,544 B driver-page reclaim;
/// it is not resident cubin size (embedded packages total only 2,841,960 B).
/// Graph capture is measured separately after all expert capacities are primed.
pub fn glmf_local_tp2_runtime_items(rank: usize) -> Vec<cuteafd_core::memory_layout::Item> {
    use cuteafd_core::memory_layout::{Basis, Category, Item};
    assert!(rank < 2, "GLM Flash TP2 has exactly two ranks");
    let terms = [
        ("CUDA context", 586_416_128),
        ("second-rank context", [0, 10_485_760][rank]),
        ("selected program setup", 12_582_912),
        ("target-load BLAS and driver state", [185_912_576, 183_725_312][rank]),
        ("peer exchange mappings and setup", 97_364_992),
        ("EXL3 and route package driver state", 6_817_612),
        ("workspace BLAS and driver state", [138_416_896, 3_280_640][rank]),
    ];
    terms.into_iter().map(|(group, bytes)| Item::new(Category::Runtime, group, "", bytes, Basis::Calibrated)).collect()
}

/// GLM Flash startup executable means (SM120 RTX PRO 6000, CUDA driver
/// 595.91.07, EXL3 K3.25, 64 decode rows, v3-p2 ready ledgers, 2026-10-10).
/// One GPU: 1,946,157,056 / 13,800; split lead: 2,231,369,728 / 13,800;
/// split peer: 2,283,798,528 / 13,500. Values retain P2's rounded arithmetic.
pub const GLMF_GRAPH_BYTES_BY_ROLE: [u64; 3] = [141_027, 161_694, 169_171];

/// GLM owned-phase SM120 executable mean, RTX PRO 6000, EXL3 K4, 2026-10-15.
/// 49,416 executables/rank used 5,790,236,672 / 5,762,973,696 B on two launches.
/// This is a phase-class calibration, not a whole-engine budget. Recheck ready
/// deltas on RTX 5090 or after changing the phase/shape census.
pub const GLM_OWNED_SM120_GRAPH_BYTES: u64 = 117_000;

/// The table's family key of a program family (`dsv4f`, `mimop2`, ...).
pub fn loaded_code_family(program_family: &str) -> &str {
    let base = program_family.trim_end_matches('2');
    if base.starts_with("mimo") { "mimo" } else if base.starts_with("dsv4") { "dsv4" } else { base }
}

/// The measured loaded code of `program_family` with `experts` on coordinator
/// `rank` of a (`split`) layout: exact package, then an explicit `*` entry.
/// Unmeasured packages retain the manifest formula, never another package's calibration.
pub fn loaded_code(program_family: &str, experts: &str, split: bool, rank: u8) -> Option<&'static LoadedCode> {
    let family = loaded_code_family(program_family);
    let at = |c: &&LoadedCode| c.family == family && c.split == split && c.rank == rank;
    LOADED_CODE.iter().filter(at).find(|c| c.experts == experts)
        .or_else(|| LOADED_CODE.iter().filter(at).find(|c| c.experts == "*"))
}

impl LoadedCode {
    /// The part serve still reserves at an admission sample whose used bytes
    /// beyond this process's tracked allocations are `untracked_at_sample`
    /// (`total - free - tracked`; the CUDA context included).
    pub fn pending(&self, untracked_at_sample: u64, context_bytes: u64) -> u64 {
        self.bytes.saturating_sub(untracked_at_sample.saturating_sub(context_bytes))
    }
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

/// Exact/explicit-wildcard calibration, otherwise cuBLAS plus selected modules
/// and peer context. Live admission subtracts bytes already in its sample.
pub fn loaded_code_allowance(program_family: &str, experts: &str, split: bool, rank: u8,
    arch: &ArchContext, programs: Option<&ProgramSet>) -> CodeAllowance {
    match loaded_code(program_family, experts, split, rank) {
        Some(code) => CodeAllowance { bytes: code.bytes, basis: Basis::Calibrated, future_library_bytes: 0 },
        None => CodeAllowance { bytes: arch.cublas_bytes
            + programs.map_or(32 * MIB, ProgramSet::module_bytes)
            + if split { arch.peer_context_bytes } else { 0 }, basis: Basis::Formula,
            future_library_bytes: arch.cublas_bytes },
    }
}

pub use cuteafd_core::RuntimeSample;

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
            names: names.into_iter().filter(|name| name.split_once('_')
                .is_some_and(|(prefix, _)| families.contains(&prefix))).map(str::to_string).collect(),
            scratch: Vec::new() }
    }

    /// Programs of `families` in `manifest` (PROGRAMS.json).
    pub fn from_manifest(manifest: &Value, families: &[&str]) -> Self {
        let keep = |name: &str| name.split_once('_').is_some_and(|(prefix, _)| families.contains(&prefix));
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
    /// still comes from LoadedCode; `scratch` here describes allocation owners.
    pub fn v41_fp8(manifest: &Value, matrices: &[(u64, u64)], rows: u64, sms: u32)
        -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        ensure!((1..=4096).contains(&rows) && sms > 0, "invalid V4.1 FP8 plan geometry");
        let compiled = [1, 16, 80, 256, 1024, 4096].into_iter().find(|&n| n >= rows).unwrap();
        let capacities: Vec<_> = [1, 16, 80].into_iter().filter(|&n| n < compiled).chain([compiled]).collect();
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
        name.split_once('_').is_some_and(|(prefix, _)| self.families.iter().any(|f| f == prefix))
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

/// One rank's graph executables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphRank {
    pub executables: u64,
    /// Reserved: the captured estimate plus the driver margin.
    pub bytes: u64,
    /// Of `bytes`, the margin (driver variance, late captures); growth.
    pub margin: u64,
}

/// The decode graph set a family captures, per rank (lead first). The
/// runtime's warm-up and the planner build it from the same function; bytes
/// are executables x the arch's bytes per executable, plus a margin.
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
            GraphRank { executables: count, bytes: measured.saturating_add(margin), margin }
        }).collect();
        Self { ranks, lifetime, shapes }
    }

    /// A lazily captured set bounded by a byte budget per rank.
    pub fn budget(per_rank: &[u64]) -> Self {
        Self { ranks: per_rank.iter().map(|&bytes| GraphRank { executables: 0, bytes, margin: bytes }).collect(),
            lifetime: Lifetime::Growth, shapes: 0 }
    }

    pub fn bytes(&self, rank: usize) -> u64 {
        self.ranks.get(rank).map_or(0, |r| r.bytes)
    }

    /// The graph bytes `rank`'s ready ledger holds: the captured estimate of a
    /// startup set, nothing of a growth set (lazy captures happen after ready).
    /// With the context and [`LoadedCode`] this is the ready ledger's untracked
    /// total: `untracked = context + loaded code + at_ready`.
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
        super::Demand::new(rank as u8, Category::Runtime, group, self.bytes(rank), Basis::Formula)
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
            .map(|(group, bytes, lifetime)| Item::new(Category::Runtime, group, "", bytes, Basis::Formula)
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
    use crate::formats::fp8_experts::ExpertFormat;
    let a4 = (format == ExpertFormat::Nvfp4).then(|| format!("fp8-{family}-nvfp4a4"));
    let name = a4.unwrap_or_else(|| format!("fp8-{family}{}", format.package_suffix()));
    let manifest: Value = serde_json::from_slice(&std::fs::read(lib.join("fp8").join(&name).join("manifest.json")).ok()?).ok()?;
    Some((name, fp8moe_scratch_bytes(&manifest, "tp1", rows)?))
}

/// Mandatory native W4A4 dense package; missing exports never select W4A16.
pub fn dense_package_directory(lib: &std::path::Path, geometry: &str) -> std::path::PathBuf {
    lib.join("fp8").join(format!("fp8-{geometry}-nvfp4a4")).join("tp1")
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
    const CAPACITIES: [u64; 6] = [1, 16, 80, 256, 1024, 4096];
    let required = if rows > 1 { rows.max(80) } else { 1 };
    let top = CAPACITIES.into_iter().find(|&c| c >= required).unwrap_or(4096);
    CAPACITIES.into_iter().filter(|&c| c <= top).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
    fn ready_untracked_is_context_plus_code_plus_startup_graphs() {
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
        let code = loaded_code("qwen4", "exl3", false, 0).unwrap().bytes;
        let context = ARCH_CONTEXTS[1].context_bytes;
        let untracked = 5_936_332_180u64;
        let planned = context + code + startup.at_ready(0);
        assert!(planned.abs_diff(untracked) < 4 << 20, "planned {planned} vs ledger {untracked}");
    }

    #[test]
    fn uncalibrated_packages_use_positive_manifest_formula() {
        let arch = &ARCH_CONTEXTS[1];
        let programs = ProgramSet::from_names(&["qwen4"], ["qwen4_decode", "other_decode"]);
        assert_eq!(programs.names, ["qwen4_decode"]);
        let allowance = loaded_code_allowance("qwen4", "unmeasured", false, 0, arch, Some(&programs));
        assert_eq!(allowance.basis, Basis::Formula);
        assert_eq!(allowance.bytes, arch.cublas_bytes + programs.module_bytes());
        assert!(allowance.bytes > 0);
        assert_eq!(allowance.pending(arch.context_bytes, arch.context_bytes), allowance.bytes);
        assert_eq!(allowance.pending(arch.context_bytes + 100, arch.context_bytes), allowance.bytes - 100);
        assert_eq!(allowance.pending(u64::MAX, arch.context_bytes), arch.cublas_bytes);
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
        let calibrated = loaded_code_allowance("mimo", "fp8", false, 0, arch, None);
        assert_eq!(calibrated.pending(u64::MAX, arch.context_bytes), 0);
    }

    #[test]
    fn measured_fp8_and_glm_exl3_planner_bytes_do_not_change() {
        let arch = &ARCH_CONTEXTS[1];
        let programs = ProgramSet::from_names(&["glm", "glm2"], ["glm_decode", "glm2_decode"]);
        for (family, package, split, rank, old_bytes) in [
            ("mimo", "fp8", false, 0, 261_483_520),
            ("mimo", "fp8", true, 0, 365_259_264),
            ("mimo", "fp8", true, 1, 165_460_992),
            ("qwen4", "fp8", false, 0, 1_089_736_388),
        ] {
            let allowance = loaded_code_allowance(family, package, split, rank, arch, Some(&programs));
            assert_eq!(allowance.basis, Basis::Calibrated);
            assert_eq!(allowance.bytes, old_bytes);
        }
        for rank in 0..2 {
            let allowance = loaded_code_allowance("glm", "exl3", true, rank, arch, Some(&programs));
            assert_eq!(allowance.basis, Basis::Formula);
            assert_eq!(arch.context_bytes + allowance.bytes,
                arch.context_bytes + arch.cublas_bytes + programs.module_bytes() + arch.peer_context_bytes);
        }
    }

    #[test]
    fn loaded_code_keys_family_split_rank_and_experts() {
        assert_eq!(loaded_code("qwen4", "fp8", false, 0).unwrap().bytes, 1_089_736_388);
        assert_eq!(loaded_code("qwen4", "exl3", false, 0).unwrap().bytes, 780_221_844);
        // No arbitrary expert package can donate its calibration.
        assert!(loaded_code("qwen4", "none", false, 0).is_none());
        assert_eq!(loaded_code("dsv4p2", "fp8", true, 1).unwrap().bytes, 803_054_544);
        // An exact package row wins over the family's any-package row, wherever it sits in the table.
        assert_eq!(loaded_code("dsv4p", "exl3", false, 0).unwrap().bytes, 724_349_124);
        assert_eq!(loaded_code("dsv4f", "*", false, 0).unwrap().bytes, 322_050_368);
        assert_eq!(loaded_code("dsv4p", "exl3", true, 1).unwrap().bytes, 885_694_212);
        assert_eq!(loaded_code("mimop", "none", true, 0).unwrap().bytes, 365_259_264);
        assert!(loaded_code("glm", "fp8", false, 0).is_none());
        let code = loaded_code("glmf", "exl3", false, 0).unwrap();
        assert_eq!(code.pending(586_416_128 + 100, 586_416_128), code.bytes - 100);
        assert_eq!(code.pending(586_416_128 + (1 << 40), 586_416_128), 0);
    }

    #[test]
    fn v4_tp2_code_matches_tagged_ready_untracked_bytes() {
        let context = ARCH_CONTEXTS[1].context_bytes;
        for (experts, untracked) in [("*", [1_593_749_856, 1_389_470_672]),
            ("exl3", [1_645_601_720, 1_472_110_340])] {
            for (rank, bytes) in untracked.into_iter().enumerate() {
                let code = loaded_code("dsv4", experts, true, rank as u8).unwrap();
                assert_eq!(context + code.bytes, bytes);
                assert!(code.source.contains("tagged ready"));
            }
        }
    }

    #[test]
    fn exl3_capacities_list_4096_once() {
        assert_eq!(exl3_capacities(4096), [1, 16, 80, 256, 1024, 4096]);
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
    fn v41_fp8_retains_aligned_capacity_arenas_and_selects_physical_sms() {
        let variants = [1, 16, 80, 256, 1024, 4096].into_iter().flat_map(|capacity| {
            let plans: &[u64] = if capacity <= 16 { &[170, 188] } else { &[0] };
            plans.iter().map(move |&sms| json!({"capacity": capacity, "plan_sms": sms,
                "input_dim": 5120, "output_dim": 1280,
                "label": format!("v41_query_m{capacity}_sm{sms}"),
                "scratch_bytes": capacity * 257 + sms})).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let manifest = json!({"variants": variants});
        let expected = |rows: u64, sms: u64| {
            let compiled = [1,16,80,256,1024,4096].into_iter().find(|&n| n >= rows).unwrap();
            [1,16,80].into_iter().filter(|&n| n < compiled).chain([compiled])
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
            let variants = [1,16,80,256,1024,4096].into_iter().map(|capacity|
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
                changed["variants"][2][field] = value;
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
