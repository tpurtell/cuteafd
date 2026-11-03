//! Native dense facts and composed execution admission, independent of a family solver.
use super::{dense::{DensePlan, DenseRecord, DenseRole, DenseSelection, scratch::LinearScratchLayout}, residency::MetadataInput};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::exlr8::KernelFamily;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
#[path = "native_tests.rs"]
mod tests;

pub fn family_geometry(config: &Value) -> Result<(usize, usize, KernelFamily)> {
    let text = config.get("text_config").unwrap_or(config);
    match text["model_type"].as_str() {
        Some("glm_moe_dsa") => {
            let cfg = crate::families::glm5::GlmDsaConfig::from_hf(config)?;
            ensure!(cfg.hidden == 6144 && cfg.layers == 78 && cfg.first_moe_layer == 3,
                "EXLR8 native full-forward requires GLM-5.3 78 layers/hidden6144/first routed3");
            Ok((cfg.layers, cfg.first_moe_layer, KernelFamily::Glm))
        }
        Some("glm5_next_text") => {
            let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(config)?;
            let first = cfg.dense.iter().position(|&dense| !dense).context("Flash lacks routed backbone layers")?;
            ensure!(cfg.dense[first..].iter().all(|&dense| !dense), "EXLR8 requires a contiguous routed suffix");
            let kernels = KernelFamily::from_geometry(cfg.hidden, cfg.moe_intermediate / 128, cfg.topk, cfg.experts)?;
            ensure!(kernels == KernelFamily::GlmFlash && cfg.moe_intermediate % 128 == 0,
                "Flash EXLR8 geometry lacks native expert exports");
            Ok((cfg.layers, first, kernels))
        }
        other => anyhow::bail!("unsupported EXLR8 native model_type {other:?}"),
    }
}

/// Format facts feed ordinary family solver fields; ownership decisions stay external.
pub struct DenseFacts {
    pub records: Vec<DenseRecord>,
    pub selection: DenseSelection,
    pub layers: usize,
    pub shared_bytes: Vec<u64>,
}
impl DenseFacts {
    pub fn new(records: Vec<DenseRecord>, selection: DenseSelection, layers: usize, shared_bytes: Vec<u64>) -> Result<Self> {
        ensure!(matches!(shared_bytes.len(), 1 | 2), "native shared expert owner count");
        selection.backbone(&records, layers)?;
        Ok(Self { records, selection, layers, shared_bytes })
    }
    pub fn plans(&self, whole_mla: &BTreeMap<usize, usize>) -> Result<Vec<DensePlan>> {
        let records = self.records.iter().filter_map(|record| match self.selection.role(&record.tensor, self.layers) {
            Ok(DenseRole::Excluded) => None, Ok(_) => Some(Ok(record.clone())), Err(error) => Some(Err(error)),
        }).collect::<Result<Vec<_>>>()?;
        (0..self.shared_bytes.len()).map(|rank| DensePlan::for_rank_owned(&records,
            self.shared_bytes.len(), rank, whole_mla).map_err(Into::into)).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashStageLayout { pub extents: BTreeMap<&'static str, usize> }
impl FlashStageLayout {
    pub fn new(cfg: &crate::families::glm5_flash::GlmNextConfig, rows: usize, ranks: usize,
        owned_mla: bool, auxiliary_width: usize, scratch: usize) -> Result<Self> {
        let mut extents = crate::families::glm5_flash::core_admission::CoreStageLayout::new(
            cfg, rows, ranks, owned_mla, scratch)?.extents;
        let width = cfg.kda_heads.checked_mul(cfg.kda_head_dim).context("native KDA width overflow")? / ranks;
        let bytes = |n: usize| rows.checked_mul(n).and_then(|n| n.checked_mul(2)).context("native KDA stage overflow");
        extents.insert("decay", bytes(width)?.max(256));
        extents.insert("output_gate", bytes(width)?.max(256));
        extents.insert("beta", bytes(cfg.kda_heads / ranks)?.max(256));
        // Auxiliary low-rank projections run sequentially before f_b/g_b.
        extents.insert("aux", bytes(auxiliary_width)?.max(256));
        // The indexer is replicated even when attention heads are split.
        extents.insert("index_query", bytes(cuteafd_ffi::projected_index::HEADS * 128)?.max(256));
        extents.insert("index_packed", bytes(256 + cuteafd_ffi::projected_index::HEADS)?.max(256));
        Ok(Self { extents })
    }
    pub fn bytes(&self) -> Result<usize> {
        self.extents.values().try_fold(0usize, |sum, &bytes| sum.checked_add(bytes).context("native Flash stage sum overflow"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashRankLayout {
    pub linear: [LinearScratchLayout; 2],
    pub stages: [FlashStageLayout; 2],
    pub shared_workspace: u64,
}
impl FlashRankLayout {
    pub fn bytes(&self) -> Result<u64> {
        let mut bytes = self.shared_workspace;
        for layout in &self.linear { bytes = bytes.checked_add(layout.bytes()? as u64).context("native Flash linear total")?; }
        for layout in &self.stages { bytes = bytes.checked_add(layout.bytes()? as u64).context("native Flash stage total")?; }
        // NativePrograms transforms/narrows each rank before the ordinary BF16
        // exchange; K6's retained core rows and receive suffix are not used.
        Ok(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashCharges { pub ranks: Vec<FlashRankLayout>, pub programs: BTreeSet<String> }
impl FlashCharges {
    #[allow(clippy::too_many_arguments)]
    pub fn from_manifest(facts: &DenseFacts, cfg: &crate::families::glm5_flash::GlmNextConfig,
        whole_mla: &BTreeMap<usize, usize>, rows: usize, waves: usize, decode: usize,
        index_cache: crate::serving_capacity::GlmfIndexCache, manifest: &Value) -> Result<Self> {
        ensure!(facts.layers == cfg.layers && (1..=4).contains(&waves) && matches!(decode, 64 | 128),
            "native Flash execution geometry");
        let ranks = facts.shared_bytes.len();
        ensure!(ranks == 1 || decode == 64, "native Flash split requires decode capacity 64");
        let backbone = facts.selection.backbone(&facts.records, cfg.layers)?;
        let plans = (0..ranks).map(|rank| DensePlan::for_rank_owned(&backbone, ranks, rank, whole_mla))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let owned = !whole_mla.is_empty();
        let auxiliary_width = backbone.iter().filter(|record| record.tensor.ends_with(".self_attn.f_a_proj.weight")
            || record.tensor.ends_with(".self_attn.g_a_proj.weight"))
            .map(|record| record.shape[0]).max().context("native Flash low-rank KDA projections missing")?;
        let mut programs = BTreeSet::new();
        let mut stage_scratch = [0; 2];
        let entries = manifest["programs"].as_array().context("native Flash manifest programs")?;
        for (index, is_decode) in [true, false].into_iter().enumerate() {
            for name in crate::families::glm5_flash::core_admission::route_stage_names(cfg, ranks, owned, is_decode, decode) {
                let name = name.replace("_kda_core_", "_kda_projected_")
                    .replace("_absorb_m", "_absorb_fp8_m").replace("_values_m", "_values_fp8_m");
                let spec = super::workspace::program_spec(manifest, &name)?;
                let entry = entries.iter().find(|e| e["name"] == name).context("native Flash stage entry")?;
                if name.contains("_kda_projected_") {
                    cuteafd_ffi::projected_kda::validate_entry(&spec, entry)?;
                } else if name.contains("_absorb_fp8_") || name.contains("_values_fp8_") {
                    cuteafd_ffi::glm_stage::validate_entry(&spec, entry)?;
                }
                stage_scratch[index] = stage_scratch[index].max(usize::try_from(spec.scratch.get("scratch").copied().unwrap_or(0))?);
                programs.insert(name);
            }
            let prefix = if ranks == 1 { "glmf_stage" } else { "glmf2_stage" };
            let capacities = if !is_decode { vec![4096] } else if decode == 128 { vec![64, 128] } else { vec![64] };
            for capacity in capacities {
                let (name, scratch) = flash_index_stage(manifest, prefix, capacity, index_cache)?;
                stage_scratch[index] = stage_scratch[index].max(scratch);
                programs.insert(name);
            }
        }
        let shared = super::shared_workspace(manifest, KernelFamily::GlmFlash)?;
        let mut layouts = Vec::new();
        for plan in &plans {
            programs.extend(super::dense::programs::select_manifest(manifest, &facts.selection, &backbone, cfg.layers, plan, false)?);
            layouts.push(FlashRankLayout {
                linear: [LinearScratchLayout::from_manifest(manifest, &facts.selection, &backbone, cfg.layers, plan, decode, false)?,
                    LinearScratchLayout::from_manifest(manifest, &facts.selection, &backbone, cfg.layers, plan, rows, false)?],
                stages: [FlashStageLayout::new(cfg, decode, ranks, owned, auxiliary_width, stage_scratch[0])?,
                    FlashStageLayout::new(cfg, rows, ranks, owned, auxiliary_width, stage_scratch[1])?],
                shared_workspace: shared,
            });
        }
        Ok(Self { ranks: layouts, programs })
    }
    pub fn bytes(&self) -> Result<Vec<u64>> { self.ranks.iter().map(FlashRankLayout::bytes).collect() }
}

fn flash_index_stage(manifest: &Value, prefix: &str, capacity: usize,
    cache: crate::serving_capacity::GlmfIndexCache) -> Result<(String, usize)> {
    use crate::serving_capacity::GlmfIndexCache;
    let kind = match cache { GlmfIndexCache::Compact => "c", GlmfIndexCache::Keys => "keys" };
    let name = format!("{prefix}_index_projected_{kind}_m{capacity}");
    let spec = super::workspace::program_spec(manifest, &name)?;
    let entry = manifest["programs"].as_array().context("native Flash programs")?.iter()
        .find(|entry| entry["name"] == name).context("native Flash index stage entry")?;
    let scratch = match cache {
        GlmfIndexCache::Compact => cuteafd_ffi::projected_index::validate_entry(&spec, entry)?.scratch,
        GlmfIndexCache::Keys => { cuteafd_ffi::projected_index::validate_keys_entry(&spec, entry)?; 0 },
    };
    Ok((name, scratch))
}

/// Both ranks execute one stream-serialized routed arena; pair lane payloads are separate.
pub fn flash_routed_workspace(metadata: &MetadataInput, cfg: &crate::families::glm5_flash::GlmNextConfig,
    manifest: &Value, ranks: usize, rows: u64) -> Result<u64> {
    let first = cfg.dense.iter().position(|&dense| !dense).context("Flash routed suffix")?;
    let mut units = 0;
    for layer in first..cfg.layers { for rank in 0..ranks {
        units = units.max(metadata.plan(layer, ranks, rank, super::residency::ExpertSet::Routed)?.compact_unit_limit());
    } }
    super::routed_workspace(manifest, KernelFamily::GlmFlash, rows, units, super::PrefillActivations::A16)
}
