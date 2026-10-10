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
    /// Streaming multiprocessors (native scratch and grid sizes key on it).
    pub sms: u32,
    /// CUDA context + cuBLAS handle + runtime bookkeeping, measured on an
    /// otherwise empty device after the first allocation (no program modules).
    pub context_bytes: u64,
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
    ArchContext { arch: "sm_120", max_total_bytes: 34 << 30, sms: 170, context_bytes: 657_719_296,
        graph_executable_bytes: 149_712, driver: "595.91.07",
        source: "RTX 5090 class: SM120 PRO probe on the same driver/CUDA 13.2; SM count from the 5090 spec" },
    ArchContext { arch: "sm_120", max_total_bytes: u64::MAX, sms: 188, context_bytes: 657_719_296,
        graph_executable_bytes: 149_712, driver: "595.91.07",
        source: "RTX PRO 6000 Blackwell probe (CUDA 13.2); Qwen 12,397 graphs = 1,855,979,520 B" },
    ArchContext { arch: "sm_121", max_total_bytes: u64::MAX, sms: 48, context_bytes: 420 << 20,
        graph_executable_bytes: 149_712, driver: "580.178.04",
        source: "GB10 probe (CUDA 13.0, MemAvailable deltas): context 189-257 MiB + cuBLAS 150-160 MiB; graph bytes from SM120" },
];

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

impl GraphSet {
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
        let Some(r) = self.ranks.get(rank) else { return Vec::new() };
        let captured = match self.lifetime { Lifetime::Startup => r.bytes - r.margin, Lifetime::Growth => 0 };
        [("graphs", captured), (GRAPH_GROWTH, r.bytes - captured)].into_iter().filter(|(_, b)| *b > 0)
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
