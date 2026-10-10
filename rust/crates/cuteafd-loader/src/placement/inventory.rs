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
    LoadedCode { family: "glmf", experts: "*", split: true, rank: 0, bytes: 500_143_488,
        source: "glm53f-exl3-max rtx0: untracked 3,317,929,344 - graphs 2,231,369,728 - context 586,416,128" },
    LoadedCode { family: "glmf", experts: "*", split: true, rank: 1, bytes: 215_091_456,
        source: "glm53f-exl3-max rtx1: untracked 3,085,306,112 - graphs 2,283,798,528 - context 586,416,128" },
];

/// The table's family key of a program family (`dsv4f`, `mimop2`, ...).
pub fn loaded_code_family(program_family: &str) -> &str {
    let base = program_family.trim_end_matches('2');
    if base.starts_with("mimo") { "mimo" } else if base.starts_with("dsv4") { "dsv4" } else { base }
}

/// The measured loaded code of `program_family` with `experts` on coordinator
/// `rank` of a (`split`) layout: the exact entry, else the family's entry for
/// any experts on that rank; `None` for a family the table does not measure
/// (the caller keeps its module formula).
pub fn loaded_code(program_family: &str, experts: &str, split: bool, rank: u8) -> Option<&'static LoadedCode> {
    let family = loaded_code_family(program_family);
    let at = |c: &&LoadedCode| c.family == family && c.split == split && c.rank == rank;
    LOADED_CODE.iter().filter(at).find(|c| c.experts == experts)
        .or_else(|| LOADED_CODE.iter().filter(at).find(|c| c.experts == "*"))
        .or_else(|| LOADED_CODE.iter().find(at))
}

impl LoadedCode {
    /// The part serve still reserves at an admission sample whose used bytes
    /// beyond this process's tracked allocations are `untracked_at_sample`
    /// (`total - free - tracked`; the CUDA context included).
    pub fn pending(&self, untracked_at_sample: u64, context_bytes: u64) -> u64 {
        self.bytes.saturating_sub(untracked_at_sample.saturating_sub(context_bytes))
    }
}

/// One GPU's measured inventory: one sample after the CUDA context, cuBLAS and
/// the family's program modules, before any weight.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSample {
    pub device: i32,
    /// CUDA total under the coordinator budget (what admission may use).
    pub total_bytes: u64,
    pub free_bytes: u64,
    /// Device bytes not owned by this process's allocations: context, cuBLAS,
    /// modules, and anything another process holds on the GPU.
    pub context_bytes: u64,
    /// Of `context_bytes`, the family's program modules (measured delta).
    pub module_bytes: u64,
    pub sms: u32,
    pub arch: String,
    pub driver: String,
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

/// The planner item group of reserved bytes allocated only after ready (lazy
/// graph captures, graph margins): outside the ready-ledger compare.
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
                || shape.lane > 1 || !seen.insert(shape.clone()) {
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

    /// The solver demand for this set on `rank` (Runtime "graphs"; growth sets
    /// are labelled so the ledger compare can leave them out).
    pub fn demand(&self, rank: usize) -> super::Demand {
        let group = match self.lifetime { Lifetime::Startup => "graphs", Lifetime::Growth => "graph growth" };
        super::Demand::new(rank as u8, Category::Runtime, group, self.bytes(rank), Basis::Formula)
    }

    /// The planner items for `rank`: the captured estimate at ready ("graphs")
    /// and the margin (or a lazy set's budget) as "graph growth".
    pub fn items(&self, rank: usize) -> Vec<cuteafd_core::memory_layout::Item> {
        use cuteafd_core::memory_layout::Item;
        if rank >= self.ranks.len() { return Vec::new() }
        [("graphs", self.at_ready(rank)), (GRAPH_GROWTH, self.growth(rank))].into_iter().filter(|(_, b)| *b > 0)
            .map(|(group, bytes)| Item::new(Category::Runtime, group, "", bytes, Basis::Formula)).collect()
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
/// `family` (`fp8-<family>[-nvfp4[a4]]`, W4A4 preferred where built, as
/// `shared::experts::fp8::package_directory`), under `lib`'s `fp8/` tree, and
/// its scratch for `rows` rows.
pub fn fp8moe_package_scratch(lib: &std::path::Path, family: &str,
    format: crate::formats::fp8_experts::ExpertFormat, rows: u64) -> Option<(String, u64)> {
    use crate::formats::fp8_experts::ExpertFormat;
    let a4 = (format == ExpertFormat::Nvfp4).then(|| format!("fp8-{family}-nvfp4a4"));
    let name = a4.filter(|n| lib.join("fp8").join(n).is_dir())
        .unwrap_or_else(|| format!("fp8-{family}{}", format.package_suffix()));
    let manifest: Value = serde_json::from_slice(&std::fs::read(lib.join("fp8").join(&name).join("manifest.json")).ok()?).ok()?;
    Some((name, fp8moe_scratch_bytes(&manifest, "tp1", rows)?))
}

/// Dense NVFP4 package selection shared with the coordinator loader. A4 is
/// preferred only when its tp1 package exists; A16 selects the NVFP4 package.
pub fn dense_package_directory(lib: &std::path::Path, geometry: &str, a4: bool) -> std::path::PathBuf {
    let root = lib.join("fp8");
    let preferred = root.join(format!("fp8-{geometry}-nvfp4a4")).join("tp1");
    if a4 && preferred.is_dir() { preferred }
    else { root.join(format!("fp8-{geometry}-nvfp4")).join("tp1") }
}

pub fn dense_package_scratch(lib: &std::path::Path, geometry: &str, rows: u64, a4: bool) -> Option<u64> {
    let directory = dense_package_directory(lib, geometry, a4);
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
pub fn exl3_capacities(rows: u64) -> Vec<u64> {
    const CAPACITIES: [u64; 6] = [1, 16, 80, 256, 1024, 4096];
    let top = CAPACITIES.into_iter().find(|&c| c >= rows.max(1)).unwrap_or(4096);
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
    fn loaded_code_keys_family_split_rank_and_experts() {
        assert_eq!(loaded_code("qwen4", "fp8", false, 0).unwrap().bytes, 1_089_736_388);
        assert_eq!(loaded_code("qwen4", "exl3", false, 0).unwrap().bytes, 780_221_844);
        // An unmeasured package falls back to the family's rank entry.
        assert!(loaded_code("qwen4", "none", false, 0).is_some());
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
        assert_eq!(set.demand(0).group, "graphs");
        assert_eq!(GraphSet::budget(&[1 << 30]).demand(0).group, "graph growth");
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
    fn simulated_small_cards_keep_the_physical_sm_count() {
        let pro = ArchContext::for_device("sm_120", 96 << 30);
        assert_eq!(pro.sms, 188);
        assert_eq!(ArchContext::coordinator(32 << 30, Some(pro)).sms, 188);
        assert_eq!(ArchContext::coordinator(32 << 30, None).sms, 170);
    }
}
