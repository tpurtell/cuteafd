//! Exact native K6 storage beside the Flash execution slots.
use super::{core::CoreInventory, GlmNextConfig};
use crate::formats::{dense_trellis::TrellisSide, trellis_core::{CoreWeightLayout, ProjectionScratchLayout}};
use anyhow::{ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreStageLayout { pub extents: BTreeMap<&'static str, usize> }
impl CoreStageLayout {
    pub fn new(cfg: &GlmNextConfig, rows: usize, ranks: usize, owned: bool, scratch: usize) -> Result<Self> {
        ensure!((1..=4096).contains(&rows) && matches!(ranks, 1 | 2), "Flash K6 stage geometry");
        let d = cfg.kda_heads * cfg.kda_head_dim / ranks;
        let heads = cfg.heads / if owned { 1 } else { ranks };
        let inter = cfg.dense_intermediate.max(cfg.moe_intermediate) / ranks;
        let bf16 = |n: usize| rows.checked_mul(n).and_then(|n| n.checked_mul(2)).context("Flash K6 stage overflow");
        Ok(Self { extents: BTreeMap::from([
            ("q", bf16(d.max(heads * cfg.qk_nope_dim).max(cfg.q_lora_rank))?),
            ("k", bf16(d.max(cfg.kv_lora_rank))?), ("v", bf16(d)?),
            ("values", bf16(d.max(heads * cfg.v_head_dim))?),
            ("gate", bf16(inter)?), ("up", bf16(inter)?), ("hidden", bf16(inter)?),
            ("scratch", scratch.max(256)),
        ]) })
    }
    pub fn bytes(&self) -> Result<usize> {
        self.extents.values().try_fold(0usize, |n, &v| n.checked_add(v).context("Flash K6 stage sum overflow"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreRankLayout {
    pub body: Vec<CoreWeightLayout>,
    pub stages: Option<[CoreStageLayout; 2]>,
    pub decode: Option<ProjectionScratchLayout>,
    pub prefill: Option<ProjectionScratchLayout>,
    pub target_head: Option<ProjectionScratchLayout>,
    pub draft_head: Option<ProjectionScratchLayout>,
    pub retained_rows: usize,
    pub receive_suffix: usize,
}
impl CoreRankLayout {
    pub fn bytes(&self, hidden: usize) -> Result<u64> {
        let mut bytes = self.receive_suffix;
        for layout in self.decode.iter().chain(&self.prefill).chain(&self.target_head).chain(&self.draft_head) {
            bytes = bytes.checked_add(layout.bytes()?).context("Flash K6 scratch sum overflow")?;
        }
        if let Some(stages) = &self.stages { for stage in stages {
            bytes = bytes.checked_add(stage.bytes()?).context("Flash K6 stage sum overflow")?;
        } }
        let retained = self.retained_rows.checked_mul(hidden).and_then(|n| n.checked_mul(12))
            .context("Flash K6 retained rows overflow")?;
        Ok(bytes.checked_add(retained).context("Flash K6 allocation total overflow")? as u64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreCharges { pub ranks: Vec<CoreRankLayout>, pub programs: Vec<String> }
impl CoreCharges {
    #[allow(clippy::too_many_arguments)]
    pub fn from_package(cfg: &GlmNextConfig, core: &CoreInventory, layers: usize, ranks: usize,
        owners: Option<&[u8]>, unit: usize, slots: usize, decode_rows: usize,
        draft_rows: Option<usize>, manifest: &serde_json::Value) -> Result<Self> {
        ensure!(layers > 0 && layers <= cfg.layers && matches!(ranks, 1 | 2)
            && (1..=4096).contains(&unit) && (1..=4).contains(&slots)
            && matches!(decode_rows, 64 | 128) && (ranks == 1 || decode_rows == 64)
            && owners.is_none_or(|o| ranks == 2 && o.len() == layers), "Flash K6 admission geometry");
        let mut bodies = vec![Vec::new(); ranks];
        for layer in 0..layers {
            let owner = owners.and_then(|o| (cfg.attention[layer] == super::GlmNextAttention::Mla).then_some(o[layer]));
            for (rank, weights) in core.layer_weights(cfg, layer, ranks, owner)?.into_iter().enumerate() {
                bodies[rank].extend(weights.into_values());
            }
        }
        let head = core.get_weight("lm_head").map(|p| CoreWeightLayout::new(p, TrellisSide::Output, 0, 1)).transpose()?;
        let mut programs = BTreeSet::new();
        for weight in bodies.iter().flatten().chain(head.iter()) {
            let book = weight.codebook.marker();
            programs.extend(cuteafd_ffi::dense_trellis::validate_package_geometry(manifest,
                weight.shape[1], weight.shape[0], if weight.side == TrellisSide::Input { 2 } else { 1 }, book,
                unit.max(decode_rows).max(draft_rows.unwrap_or(1)),
                if weight.stem == "lm_head" { cuteafd_ffi::dense_trellis::TrellisDType::F32 }
                else { cuteafd_ffi::dense_trellis::TrellisDType::Bf16 })?);
        }
        let entries = manifest["programs"].as_array().context("Flash K6 manifest programs")?;
        let lookup = |name: &str| -> Result<usize> {
            let matches = entries.iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
            ensure!(matches.len() == 1, "Flash K6 requires one {name}");
            Ok(matches[0]["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0).try_into()?)
        };
        let scratch = |rows: usize| -> Result<usize> {
            let mut bytes = 0;
            for name in stage_names(cfg, ranks, owners.is_some(), rows, decode_rows) {
                bytes = bytes.max(lookup(&name)?);
            }
            Ok(bytes)
        };
        if bodies.iter().any(|b| !b.is_empty()) {
            for rows in [decode_rows, unit] {
                for name in stage_names(cfg, ranks, owners.is_some(), rows, decode_rows) {
                    lookup(&name)?; programs.insert(name);
                }
            }
        }
        let projection = |body: &[CoreWeightLayout], rows| -> Result<Option<ProjectionScratchLayout>> {
            let mut result: Option<ProjectionScratchLayout> = None;
            for w in body {
                let layout = ProjectionScratchLayout::new(w.shape[1], w.shape[0], rows)?;
                if let Some(result) = &mut result { result.merge(&layout)?; } else { result = Some(layout); }
            }
            Ok(result)
        };
        let head_scratch = |rows| head.as_ref().map(|h| ProjectionScratchLayout::new(h.shape[1], h.shape[0], rows)).transpose();
        let split = bodies.iter().flatten().any(|w| w.side == TrellisSide::Input);
        let retained_rows = decode_rows.checked_add(unit.checked_mul(slots).context("Flash K6 live rows overflow")?)
            .context("Flash K6 live rows overflow")?;
        let receive = 4usize.checked_mul(slots).and_then(|n| n.checked_mul(unit.max(decode_rows)))
            .and_then(|n| n.checked_mul(cfg.hidden)).and_then(|n| n.checked_mul(4)).context("Flash K6 receive overflow")?;
        let mut layouts = Vec::new();
        for (rank, body) in bodies.into_iter().enumerate() {
            layouts.push(CoreRankLayout {
                stages: if body.is_empty() { None } else { Some([
                    CoreStageLayout::new(cfg, decode_rows, ranks, owners.is_some(), scratch(decode_rows)?)?,
                    CoreStageLayout::new(cfg, unit, ranks, owners.is_some(), scratch(unit)?)?]) },
                decode: projection(&body, decode_rows)?, prefill: projection(&body, unit)?,
                target_head: if rank == 0 { head_scratch(decode_rows.max(unit))? } else { None },
                draft_head: if rank == 0 { draft_rows.map(head_scratch).transpose()?.flatten() } else { None },
                retained_rows: if body.is_empty() { 0 } else { retained_rows },
                receive_suffix: if split { receive } else { 0 }, body,
            });
        }
        let result = Self { ranks: layouts, programs: programs.into_iter().collect() };
        result.bytes(cfg.hidden)?;
        Ok(result)
    }
    pub fn bytes(&self, hidden: usize) -> Result<Vec<u64>> { self.ranks.iter().map(|r| r.bytes(hidden)).collect() }
}

/// Stage package names are separate from the existing fused programs.
pub fn stage_names(cfg: &GlmNextConfig, ranks: usize, owned: bool, rows: usize, decode_rows: usize) -> Vec<String> {
    let cap = if rows <= decode_rows { format!("m{decode_rows}") } else { "m4096".into() };
    let mut names = Vec::new();
    for prefix in if ranks == 1 { vec!["glmf_stage"] } else if owned { vec!["glmf_stage", "glmf2_stage"] }
        else { vec!["glmf2_stage"] } {
        for stage in ["kda_core", "rank_norm", "absorb", "values"] {
            names.push(format!("{prefix}_{stage}_{cap}"));
        }
    }
    let prefix = if ranks == 2 { "glmf2_stage" } else { "glmf_stage" };
    for inter in [cfg.dense_intermediate / ranks, cfg.moe_intermediate / ranks] {
        names.push(format!("{prefix}_swiglu_i{inter}_{cap}"));
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_native_bytes_include_every_owner_and_no_retained_alias() -> Result<()> {
        let p = ProjectionScratchLayout::new(4096, 8192, 64)?;
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45))?;
        let stage = CoreStageLayout::new(&cfg, 64, 2, false, 4096)?;
        let rank = CoreRankLayout { body: vec![], stages: Some([stage.clone(), stage.clone()]),
            decode: Some(p.clone()), prefill: Some(p.clone()), target_head: Some(p.clone()), draft_head: Some(p.clone()),
            retained_rows: 64 + 3 * 4096, receive_suffix: 4 * 3 * 4096 * 4096 * 4 };
        assert_eq!(rank.bytes(4096)?, (4 * p.bytes()? + 2 * stage.bytes()? + rank.receive_suffix
            + rank.retained_rows * 4096 * 12) as u64);
        let mut overflow = rank; overflow.retained_rows = usize::MAX;
        assert!(overflow.bytes(4096).is_err());
        Ok(())
    }
}
