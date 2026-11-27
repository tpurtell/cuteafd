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
    /// The rank's independent device requests behind [`Self::bytes`], in bytes: every
    /// projection scratch owner, every stage extent and every retained slot buffer.
    /// The receive suffix is a view inside the peer exchange slot that the transport
    /// charges, so it is none of them (their sum is `bytes(hidden) - receive_suffix`).
    pub allocations: Vec<u64>,
    pub receive_suffix: usize,
}
impl CoreRankLayout {
    /// The rank's own device requests: [`Self::bytes`] without the receive suffix,
    /// which is a view inside the peer exchange slot the transport charges. Exactly
    /// the sum of [`Self::allocations`], so admission charges this rank once.
    pub fn owned_bytes(&self, hidden: usize) -> Result<u64> {
        self.bytes(hidden)?.checked_sub(self.receive_suffix as u64)
            .context("Flash K6 owned extent below its receive suffix")
    }

    pub fn bytes(&self, hidden: usize) -> Result<u64> {
        let mut bytes = self.receive_suffix;
        for layout in self.decode.iter().chain(&self.prefill).chain(&self.target_head).chain(&self.draft_head) {
            bytes = bytes.checked_add(layout.bytes()?).context("Flash K6 scratch sum overflow")?;
        }
        if let Some(stages) = &self.stages { for stage in stages {
            bytes = bytes.checked_add(stage.bytes()?).context("Flash K6 stage sum overflow")?;
        } }
        let retained = self.retained_rows.checked_mul(hidden).and_then(|n| n.checked_mul(8))
            .context("Flash K6 retained rows overflow")?;
        Ok(bytes.checked_add(retained).context("Flash K6 allocation total overflow")? as u64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreCharges { pub ranks: Vec<CoreRankLayout>, pub programs: Vec<String> }

/// Whole MLA stages exist only when an admitted layer actually contains MLA.
pub fn owned_mla(cfg: &GlmNextConfig, layers: usize, ranks: usize, whole: bool) -> bool {
    ranks == 2 && whole && cfg.attention[..layers].contains(&super::GlmNextAttention::Mla)
}

pub fn receive_suffix_bytes(rows: usize, hidden: usize, waves: usize, output_shard: bool) -> Result<usize> {
    (if output_shard { 6usize } else { 4 }).checked_mul(waves)
        .and_then(|n| n.checked_mul(rows)).and_then(|n| n.checked_mul(hidden))
        .and_then(|n| n.checked_mul(2)).context("Flash K6 receive overflow")
}
impl CoreCharges {
    #[allow(clippy::too_many_arguments)]
    pub fn from_package(cfg: &GlmNextConfig, core: &CoreInventory, layers: usize, ranks: usize,
        owners: Option<&[u8]>, unit: usize, slots: usize, decode_rows: usize,
        draft_rows: Option<usize>, output_shard: bool, manifest: &serde_json::Value) -> Result<Self> {
        ensure!(layers > 0 && layers <= cfg.layers && matches!(ranks, 1 | 2)
            && (1..=4096).contains(&unit) && (1..=4).contains(&slots)
            && (1..=128).contains(&decode_rows) && (ranks == 1 || decode_rows <= 64)
            && owners.is_none_or(|o| ranks == 2 && o.len() == layers), "Flash K6 admission geometry");
        let owned = owned_mla(cfg, layers, ranks, owners.is_some());
        let mut bodies = vec![Vec::new(); ranks];
        let mut native_names = vec![[BTreeSet::new(), BTreeSet::new()]; ranks];
        let mut bf16_shapes = BTreeSet::new();
        let mut native_bodies = vec![false; ranks];
        let mut native_split = false;
        for layer in 0..layers {
            let owner = owners.and_then(|o| (cfg.attention[layer] == super::GlmNextAttention::Mla).then_some(o[layer]));
            for (rank, weights) in core.layer_weights(cfg, layer, ranks, owner)?.into_iter().enumerate() {
                bodies[rank].extend(weights.into_values());
            }
            for (rank, weights) in core.layer_native(cfg, layer, ranks, owner)?.into_iter().enumerate() {
                for weight in weights.values() {
                    native_bodies[rank] = true;
                    native_split |= weight.ranks > 1 && weight.split == super::core::CoreSplit::Input;
                    if !weight.fp8 {
                        bf16_shapes.insert((weight.shape[1], weight.shape[0]));
                        continue;
                    }
                    let prefix = fp8_projection_prefix(weight);
                    let decode_cap = if decode_rows > 64 { 128 } else { 64 };
                    for (index, caps) in [vec![64, decode_cap], vec![4096]].into_iter().enumerate() {
                        for cap in caps {
                            native_names[rank][index].insert(fp8_projection_name(prefix, weight.shape, cap));
                        }
                    }
                }
            }
        }
        let head = core.get_weight("lm_head").map(|p| CoreWeightLayout::new(p, TrellisSide::Output, 0, 1)).transpose()?;
        let mut programs = BTreeSet::new();
        for (k, n) in bf16_shapes {
            programs.insert(cuteafd_ffi::draft_projection::validate_package_geometry(
                manifest, k, n, unit.max(decode_rows))?);
        }
        let mut fused_keys = BTreeSet::new();
        for weight in bodies.iter().flatten().chain(head.iter()) {
            let book = weight.codebook.marker();
            let side = if weight.side == TrellisSide::Input { 2 } else { 1 };
            let is_head = weight.stem == "lm_head";
            let resolved = cuteafd_ffi::dense_trellis::validate_package_geometry_bits(manifest,
                weight.shape[1], weight.shape[0], side, weight.bits, book,
                if is_head { decode_rows.max(draft_rows.unwrap_or(1)) } else { unit.max(decode_rows) },
                if is_head { cuteafd_ffi::dense_trellis::TrellisDType::F32 }
                else { cuteafd_ffi::dense_trellis::TrellisDType::Bf16 })?;
            if resolved.len() == 4 {
                fused_keys.insert((weight.shape[1], weight.shape[0], side, weight.bits, book, is_head));
            }
            programs.extend(resolved);
        }
        let entries = manifest["programs"].as_array().context("Flash K6 manifest programs")?;
        let lookup = |name: &str| -> Result<usize> {
            let matches = entries.iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
            ensure!(matches.len() == 1, "Flash K6 requires one {name}");
            Ok(matches[0]["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0).try_into()?)
        };
        if core.native.get("lm_head").is_some_and(|head| head.fp8) {
            lookup("glmf_head_block_fp8")?;
            programs.insert("glmf_head_block_fp8".into());
        }
        let scratch = |rank: usize, decode: bool| -> Result<usize> {
            let mut bytes = 0;
            for name in route_stage_names(cfg, ranks, owned, decode, decode_rows).iter()
                .chain(native_names[rank][usize::from(!decode)].iter()) {
                bytes = bytes.max(lookup(name)?);
            }
            Ok(bytes)
        };
        if bodies.iter().any(|b| !b.is_empty()) || native_bodies.iter().any(|&b| b) {
            for decode in [true, false] {
                for name in route_stage_names(cfg, ranks, owned, decode, decode_rows) {
                    lookup(&name)?; programs.insert(name);
                }
            }
        }
        for name in native_names.iter().flatten().flatten() { lookup(name)?; programs.insert(name.clone()); }
        let has_fused = |w: &CoreWeightLayout| fused_keys.contains(&(
            w.shape[1], w.shape[0], if w.side == TrellisSide::Input { 2 } else { 1 },
            w.bits, w.codebook.marker(), w.stem == "lm_head",
        ));
        let projection = |body: &[CoreWeightLayout], rows| -> Result<Option<ProjectionScratchLayout>> {
            let mut result: Option<ProjectionScratchLayout> = None;
            for w in body {
                let layout = ProjectionScratchLayout::new_with_fused(w.shape[1], w.shape[0], rows, has_fused(w))?;
                if let Some(result) = &mut result { result.merge(&layout)?; } else { result = Some(layout); }
            }
            Ok(result)
        };
        let head_scratch = |rows| head.as_ref().map(|h|
            ProjectionScratchLayout::new_with_fused(h.shape[1], h.shape[0], rows, has_fused(h))).transpose();
        let split = ranks > 1 && (native_split || bodies.iter().flatten().any(|w| w.side == TrellisSide::Input));
        let retained_rows = decode_rows.checked_add(unit.checked_mul(slots).context("Flash K6 live rows overflow")?)
            .context("Flash K6 live rows overflow")?;
        let receive = receive_suffix_bytes(unit.max(decode_rows), cfg.hidden, slots, output_shard)?;
        // Retained slot buffers: `[rows, rows, 2 * rows]` of `hidden * 8` bytes, one slot
        // at the decode width and `slots` at the prefill width (`glmf/core-retained`).
        let retained = |rows: usize| -> Result<Vec<u64>> {
            let row = rows.checked_mul(cfg.hidden).and_then(|n| n.checked_mul(2))
                .context("Flash K6 retained row overflow")? as u64;
            Ok(vec![row, row, row.checked_mul(2).context("Flash K6 retained row overflow")?])
        };
        let mut layouts = Vec::new();
        for (rank, body) in bodies.into_iter().enumerate() {
            let active = !body.is_empty() || native_bodies[rank];
            let stages = if !active { None } else { Some([
                CoreStageLayout::new(cfg, decode_rows, ranks, owned, scratch(rank, true)?)?,
                CoreStageLayout::new(cfg, unit, ranks, owned, scratch(rank, false)?)?]) };
            let (decode, prefill) = (projection(&body, decode_rows)?, projection(&body, unit)?);
            let target_head = if rank == 0 { head_scratch(decode_rows)? } else { None };
            let draft_head = if rank == 0 { draft_rows.map(head_scratch).transpose()?.flatten() } else { None };
            let mut allocations = Vec::new();
            for layout in decode.iter().chain(&prefill).chain(&target_head).chain(&draft_head) {
                allocations.extend(layout.storage()?.owners.values().map(|&bytes| bytes as u64));
            }
            if let Some(stages) = &stages {
                allocations.extend(stages.iter().flat_map(|stage|
                    stage.extents.values().map(|&bytes| bytes as u64)));
            }
            if active {
                allocations.extend(retained(decode_rows)?);
                for _ in 0..slots { allocations.extend(retained(unit)?); }
            }
            layouts.push(CoreRankLayout {
                stages, decode, prefill, target_head, draft_head,
                retained_rows: if !active { 0 } else { retained_rows },
                allocations, receive_suffix: if split { receive } else { 0 }, body,
            });
        }
        let result = Self { ranks: layouts, programs: programs.into_iter().collect() };
        result.bytes(cfg.hidden)?;
        Ok(result)
    }
    pub fn bytes(&self, hidden: usize) -> Result<Vec<u64>> { self.ranks.iter().map(|r| r.bytes(hidden)).collect() }

    /// Each rank's independently allocated extent, exactly its [`Self::requests`] sum.
    pub fn owned_bytes(&self, hidden: usize) -> Result<Vec<u64>> { self.ranks.iter().map(|r| r.owned_bytes(hidden)).collect() }

    /// Each rank's independent device requests, paired with [`Self::bytes`] per rank.
    pub fn requests(&self) -> impl Iterator<Item = &[u64]> {
        self.ranks.iter().map(|rank| rank.allocations.as_slice())
    }
}

pub fn fp8_projection_prefix(layout: &super::core::NativeLayout) -> &'static str {
    if layout.ranks == 1 { "glmf_stage" } else { "glmf2_stage" }
}

pub fn fp8_projection_name(prefix: &str, [n, k]: [usize; 2], cap: usize) -> String {
    format!("{prefix}_fp8_linear_k{k}_n{n}_m{cap}")
}

/// Decode and prefill stages differ even when their live row counts coincide.
pub fn route_stage_names(cfg: &GlmNextConfig, ranks: usize, owned: bool, decode: bool, decode_rows: usize) -> Vec<String> {
    let mut names = stage_names(cfg, ranks, owned, if decode { "m64" } else { "m4096" });
    if decode && decode_rows > 64 { names.extend(stage_names(cfg, ranks, owned, "m128")); }
    names.sort(); names.dedup(); names
}

/// Stage package names are separate from the existing fused programs.
fn stage_names(cfg: &GlmNextConfig, ranks: usize, owned: bool, cap: &str) -> Vec<String> {
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
    fn prefill_uses_its_stage_route_even_at_decode_width() -> Result<()> {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45))?;
        let decode = route_stage_names(&cfg, 2, true, true, 64);
        let prefill = route_stage_names(&cfg, 2, true, false, 64);
        assert!(decode.iter().all(|n| n.ends_with("_m64")));
        assert!(prefill.iter().all(|n| n.ends_with("_m4096")));
        assert!(prefill.iter().any(|n| n.starts_with("glmf_stage_rank_norm")));
        assert!(prefill.iter().any(|n| n.starts_with("glmf2_stage_kda_core")));
        let wide = route_stage_names(&cfg, 1, false, true, 128);
        assert!(wide.iter().any(|n| n.ends_with("_m64")));
        assert!(wide.iter().any(|n| n.ends_with("_m128")));
        let full = CoreStageLayout::new(&cfg, 64, 2, true, 256)?;
        assert!(full.extents["values"] >= 64 * cfg.heads * cfg.v_head_dim * 2);
        Ok(())
    }

    #[test]
    fn exact_native_bytes_include_every_owner_and_no_retained_alias() -> Result<()> {
        let p = ProjectionScratchLayout::new(4096, 8192, 64)?;
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45))?;
        let stage = CoreStageLayout::new(&cfg, 64, 2, false, 4096)?;
        let rank = CoreRankLayout { body: vec![], stages: Some([stage.clone(), stage.clone()]),
            decode: Some(p.clone()), prefill: Some(p.clone()), target_head: Some(p.clone()), draft_head: Some(p.clone()),
            retained_rows: 64 + 3 * 4096, allocations: vec![], receive_suffix: 4 * 3 * 4096 * 4096 * 2 };
        assert_eq!(rank.bytes(4096)?, (4 * p.bytes()? + 2 * stage.bytes()? + rank.receive_suffix
            + rank.retained_rows * 4096 * 8) as u64);
        let mut overflow = rank; overflow.retained_rows = usize::MAX;
        assert!(overflow.bytes(4096).is_err());
        Ok(())
    }
}
