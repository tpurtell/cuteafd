//! Header/geometry-only counterpart of V4 `workspace_here`/`step_buffers`.
//! Device allocations are rounded to at least 256 bytes exactly as in the
//! engine. Prefill has the admitted wave count and decode one lane; both arenas stay resident.
use super::{product, sum, CacheGeometryError};
use crate::families::deepseek_v4::DeepseekV4Config;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Maximum queued prompts sharing a V4 prefill unit and its private tap tails.
pub const V4_PACKED_MEMBERS: usize = 16;

/// Per-member compressor headers plus padding for 16-byte-aligned table views.
pub const V4_PACKED_TABLE_EXTRA_ROWS: usize = 6 * V4_PACKED_MEMBERS;

/// Decode retains one window; packed prefill admits one window per member.
pub fn deepseek_v4_tap_rows(rows: usize, prefill: bool) -> usize {
    rows.min(128 * if prefill { V4_PACKED_MEMBERS } else { 1 })
}

/// C128 rows use the stride baked into the exported sparse MLA program,
/// independently of the explicit or pool-clamped serving context.
pub fn compiled_c128_width(manifest: &Value, family: &str) -> Result<u64, CacheGeometryError> {
    let unsupported = || CacheGeometryError::Unsupported {
        family: "deepseek_v4",
        what: "program manifest lacks consistent compiled C128 index strides",
    };
    let programs = manifest["programs"].as_array().ok_or_else(unsupported)?;
    let mut width = None;
    for program in programs.iter().filter(|p| p["family"].as_str() == Some(family)
        && p["name"].as_str().is_some_and(|n| n.contains("_c128_m"))) {
        let value = program["params"]["indexed_width"].as_u64().ok_or_else(unsupported)?;
        if value == 0 || value % 64 != 0 || width.is_some_and(|w| w != value) {
            return Err(unsupported());
        }
        width = Some(value);
    }
    width.ok_or_else(unsupported)
}

/// Four peer slots per admitted prefill lane, release flags, and
/// the local TP2 pair's persistent abort word.
pub fn deepseek_v4_peer_exchange_bytes(hidden: u64, prefill_rows: u64, decode_rows: u64,
    tp2: bool, exchange_f32: bool, waves: u64) -> Result<u64, CacheGeometryError> {
    let slot = if exchange_f32 { product("V4 F32 peer slot", &[prefill_rows.max(decode_rows), hidden, 4])? }
        else { product("V4 prefill peer slot", &[prefill_rows, hidden, 2])?
            .max(product("V4 decode peer slot", &[decode_rows, hidden, if tp2 { 4 } else { 2 }])?) };
    sum("V4 peer exchange", &[product("V4 peer slots", &[4 * waves, slot])?, 256, if tp2 { 256 } else { 0 }])
}

/// TP2 route-broadcast fallback: admitted lanes, two parities, per GPU. No
/// hidden/shared packing buffer, peer wire, or direction multiplier.
pub fn deepseek_v4_expert_exchange_bytes(_hidden: u64, topk: u64, prefill: u64, decode: u64, _rank: usize, waves: u64)
    -> Result<u64, CacheGeometryError> {
    let payload = sum("V4 route payload", &[product("V4 route rows", &[prefill.max(decode), topk, 8])?, 16])?;
    sum("V4 route exchange", &[product("V4 route slots", &[2 * waves, payload])?, 256, 256])
}

/// Payload exists only when the solver places a TP2 expert layer. Charge it
/// with that conditional arena, not with the always-resident head-split steps.
pub fn deepseek_v4_tp2_payload_bytes(hidden: u64, prefill: u64, decode: u64, waves: u64)
    -> Result<u64, CacheGeometryError> {
    sum("V4 TP2 payload", &[
        product("V4 prefill payload", &[prefill, hidden, 4, waves])?.max(256),
        product("V4 decode payload", &[decode, hidden, 4])?.max(256),
    ])
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct V4WorkspaceScratch {
    /// Prefill arena's non-index-topk scratch, including full-head draft programs.
    pub shared_bytes: u64,
    pub index_topk_bytes: u64,
    pub decode_shared_bytes: u64,
    pub decode_index_topk_bytes: u64,
}

/// Per-lane C4 page-table geometry: the pool-sized allocation every admitted
/// lane keeps (private packed member lists per prefill lane, one per decode lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V4LanePageTable {
    pub prefill_unit_bytes: u64,
    pub prefill_fixed_bytes: u64,
    pub decode_unit_bytes: u64,
}

pub fn deepseek_v4_lane_page_tables(decode_rows: u64) -> Result<V4LanePageTable, CacheGeometryError> {
    Ok(V4LanePageTable {
        prefill_unit_bytes: product("V4 prefill table slope", &[V4_PACKED_MEMBERS as u64, 4])?,
        prefill_fixed_bytes: (V4_PACKED_MEMBERS * 3 * 4) as u64,
        decode_unit_bytes: product("V4 decode table slope", &[decode_rows, 4])?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V4WorkspaceRank {
    /// Every pool-independent cudaMalloc this rank's workspace builders issue,
    /// exactly as requested: `workspace_here`'s planes, lane buffers and fixed
    /// step buffers plus every lane's private `step_buffers` tables.
    pub requests: Vec<u64>,
    /// Sum of `requests`, the demand admission charges for this rank.
    pub fixed_device_bytes: u64,
    pub prefill_waves: u64,
    /// C4 page tables: private packed member lists per prefill slot, one list per decode row.
    pub pool_unit_device_bytes: u64,
    /// Pinned router staging, reported outside GPU admission.
    pub pinned_host_bytes: u64,
    /// Per-lane table slopes and the packing padding each prefill lane keeps;
    /// the 256-byte allocation floor covers custom manifests with fewer than
    /// 64 rows.
    pub prefill_lane_table_unit_bytes: u64,
    pub prefill_lane_table_fixed_bytes: u64,
    pub decode_lane_table_unit_bytes: u64,
}

impl V4WorkspaceRank {
    pub fn device_bytes(&self, units: u64) -> Result<u64, CacheGeometryError> {
        // One C4 page table per admitted lane; each is its own allocation.
        let lane = |unit: u64, fixed: u64| {
            product("V4 lane table", &[units, unit])?
                .checked_add(fixed)
                .ok_or(CacheGeometryError::Overflow("V4 lane table padding"))
                .map(|bytes| bytes.max(256))
        };
        sum(
            "V4 workspace total",
            &[
                self.fixed_device_bytes,
                product(
                    "V4 prefill pool tables",
                    &[
                        self.prefill_waves,
                        lane(self.prefill_lane_table_unit_bytes, self.prefill_lane_table_fixed_bytes)?,
                    ],
                )?,
                lane(self.decode_lane_table_unit_bytes, 0)?,
            ],
        )
    }
}

/// Uses the runtime's namespace selector. Keep the unsplit family even under
/// a head split: dSpark target launches still use its full-head programs.
pub fn deepseek_v4_workspace_scratch(
    manifest: &Value,
    family: &str,
    split: bool,
    prefill_rows: u64,
    decode_rows: u64,
) -> Result<V4WorkspaceScratch, CacheGeometryError> {
    let unsupported = || CacheGeometryError::Unsupported {
        family: "deepseek_v4",
        what: "program manifest lacks concrete V4 scratch geometry",
    };
    let programs = manifest["programs"].as_array().ok_or_else(unsupported)?;
    let split_family = format!("{family}2");
    let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms {
        family, split_family: split.then_some(split_family.as_str()),
    };
    let mut sizes = Vec::new();
    for program in programs {
        let name = program["name"].as_str().ok_or_else(unsupported)?;
        if selected.contains(name) {
            if let Some(values) = program["scratch_bytes_at_capacity"].as_object() {
                for value in values.values() {
                    sizes.push((name, value.as_u64().ok_or_else(unsupported)?));
                }
            }
        }
    }
    let prefill_kernel_rows = manifest["capacities"]["prefill_rows"].as_u64().unwrap_or(prefill_rows);
    let shared = |rows, prefill| selected.shared_scratch(sizes.iter().copied()
        .filter(|(name, _)| selected.workspace_program(name, rows, prefill)));
    let topk = |mode: &str, rows| -> Result<u64, CacheGeometryError> {
        let name = format!("{family}_index_topk_{mode}_m{rows}");
        programs.iter().find(|p| p["name"].as_str() == Some(name.as_str()))
            .and_then(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64()).ok_or_else(unsupported)
    };
    Ok(V4WorkspaceScratch {
        shared_bytes: shared(prefill_kernel_rows, true),
        index_topk_bytes: topk("prefill", prefill_kernel_rows)?,
        decode_shared_bytes: shared(decode_rows, false),
        decode_index_topk_bytes: topk("decode", decode_rows)?,
    })
}

pub fn deepseek_v4_workspace_geometry(
    cfg: &DeepseekV4Config,
    prefill_rows: u64,
    decode_rows: u64,
    max_context: u64,
    ranks: usize,
    waves: u64,
    scratch: V4WorkspaceScratch,
) -> Result<Vec<V4WorkspaceRank>, CacheGeometryError> {
    if ![1, 2].contains(&ranks)
        || prefill_rows == 0
        || decode_rows == 0
        || max_context == 0
        || cfg.head_dim != 512
        || cfg.window_size != 128
        || cfg.n_heads % ranks != 0
    {
        return Err(CacheGeometryError::Unsupported {
            family: "deepseek_v4",
            what: "workspace rows, context, heads or coordinator rank geometry",
        });
    }
    let width = max_context
        .div_ceil(128)
        .div_ceil(64)
        .checked_mul(64)
        .ok_or(CacheGeometryError::Overflow("V4 C128 table extent"))?;
    (0..ranks)
        .map(|rank| {
            let prefill = step(cfg, prefill_rows, waves, width, rank, ranks, scratch, true)?;
            let decode = step(cfg, decode_rows, 1, width, rank, ranks, V4WorkspaceScratch {
                shared_bytes: scratch.decode_shared_bytes,
                index_topk_bytes: scratch.decode_index_topk_bytes,
                ..scratch
            }, false)?;
            let mut requests = prefill.0;
            requests.extend(decode.0);
            let tables = deepseek_v4_lane_page_tables(decode_rows)?;
            Ok(V4WorkspaceRank {
                prefill_waves: waves,
                fixed_device_bytes: sum("V4 prefill/decode workspaces", &requests)?,
                pool_unit_device_bytes: sum("V4 prefill/decode table slope", &[
                    product("V4 prefill table slope", &[waves, tables.prefill_unit_bytes])?,
                    tables.decode_unit_bytes,
                ])?,
                pinned_host_bytes: sum("V4 prefill/decode pinned staging", &[prefill.1, decode.1])?,
                requests,
                prefill_lane_table_unit_bytes: tables.prefill_unit_bytes,
                prefill_lane_table_fixed_bytes: tables.prefill_fixed_bytes,
                decode_lane_table_unit_bytes: tables.decode_unit_bytes,
            })
        })
        .collect()
}

/// One workspace shape's exact pool-independent device requests in
/// `workspace_here`'s allocation order, plus its pinned router staging.
fn step(
    cfg: &DeepseekV4Config,
    rows: u64,
    lanes: u64,
    c128_width: u64,
    rank: usize,
    ranks: usize,
    scratch: V4WorkspaceScratch,
    prefill: bool,
) -> Result<(Vec<u64>, u64), CacheGeometryError> {
    let mul = |values: &[u64]| product("V4 workspace allocation", values);
    let h = cfg.dim as u64;
    let heads = cfg.n_heads as u64 / if rank == 1 { 2 } else { 1 };
    let topk = cfg.n_activated_experts as u64;
    let lead = |bytes: u64| if rank == 0 { bytes } else { 256 };
    let draft = |bytes: u64| if rank == 0 && cfg.n_mtp_layers > 0 { bytes } else { 0 };
    // Packed fields retain private headers and 16-byte-aligned member views.
    let table_rows = rows.checked_add(if prefill { V4_PACKED_TABLE_EXTRA_ROWS as u64 } else { 0 })
        .ok_or(CacheGeometryError::Overflow("V4 packed table rows"))?;
    // Eighteen metadata arrays (C4/C128 x9) plus four length/visible arrays.
    let metadata_rows = table_rows
        .checked_add(2)
        .ok_or(CacheGeometryError::Overflow("V4 metadata rows"))?;
    let mut requests = Vec::new();
    // The five peer/staging planes are one allocation each, `lanes` wide.
    for bytes in [
        mul(&[rows, 4, h, 2])?,
        mul(&[rows, 4, h, 2])?,
        mul(&[rows, 4, 4])?,
        mul(&[rows, 16, 4])?,
        mul(&[rows, h, 2])?,
    ] {
        requests.push(product("V4 stream plane", &[lanes, bytes])?.max(256));
    }
    let tokens = mul(&[rows, 4])?.max(256);
    let taps = draft(mul(&[
        deepseek_v4_tap_rows(rows as usize, prefill) as u64,
        cfg.dspark_target_layer_ids.len() as u64,
        h,
        2,
    ])?)
    .max(256);
    // `step_buffers`' pool-independent tables: positions, main slots, the SWA
    // and C128 indices, nine C4 and nine C128 metadata arrays, and four
    // length/visible arrays, each lane's private allocation.
    let mut tables = vec![
        mul(&[table_rows, 8])?,
        mul(&[table_rows, 8])?,
        mul(&[table_rows, 128, 4])?,
        mul(&[table_rows, 4])?,
    ];
    for _ in 0..18 {
        tables.push(mul(&[metadata_rows, 4])?);
    }
    tables.extend([
        mul(&[table_rows, 4])?,
        mul(&[table_rows, 4])?,
        // c128_indices is a compiled context-sized table, not a pool table.
        mul(&[table_rows, c128_width, 4])?,
        mul(&[table_rows, 4])?,
    ]);
    for _ in 0..lanes {
        requests.push(tokens);
        requests.extend(tables.iter().map(|&bytes| bytes.max(256)));
        requests.push(taps);
    }
    let fixed = [
        mul(&[rows, h, 2])?,
        mul(&[rows, heads, 512, 2])?,
        mul(&[rows, cfg.q_lora_rank as u64, 2])?,
        mul(&[rows, heads, 512, 2])?,
        mul(&[rows, h, 2])?,
        mul(&[rows, cfg.index_n_heads as u64, cfg.index_head_dim as u64])?,
        mul(&[rows, cfg.index_n_heads as u64, 4])?,
        mul(&[rows, cfg.index_topk as u64, 4])?,
        scratch.index_topk_bytes,
        mul(&[rows, cfg.n_routed_experts as u64, 4])?,
        mul(&[rows, topk, 4])?,
        mul(&[rows, topk, 4])?,
        mul(&[rows, h + h / 32])?,
        scratch.shared_bytes,
        4096,
        lead(mul(&[rows.min(64), cfg.vocab_size as u64, 4])?),
        draft(mul(&[rows.min(128), h, 2])?),
        draft(mul(&[rows.min(128), h, 4])?),
        draft(mul(&[rows, 4])?),
        draft(mul(&[rows, 4])?),
        // native dsv4_dspark.cu: kArgmaxBlocks=256, value/id pairs x8.
        draft(mul(&[rows, 256, 8])?),
        draft(mul(&[rows, 4])?),
        if ranks == 2 { mul(&[rows, h, 2])? } else { 256 },
        lead(4 << 20), // VocabularyHead persistent workspace.
    ];
    requests.extend(fixed.map(|bytes| bytes.max(256)));
    Ok((
        requests,
        lead(mul(&[rows, topk * 8 + h + h / 32])?).max(256),
    ))
}

#[cfg(test)]
mod expert_exchange_tests {
    use super::*;
    #[test]
    fn reservation_matches_persistent_slots_lanes_and_checks() {
        for h in [4096, 7168] {
            for (prefill, decode) in [(4096, 64), (16, 64)] {
                let payload = prefill.max(decode) * 6 * 8 + 16;
                let lead = 4 * payload + 512;
                let peer = lead;
                assert_eq!(deepseek_v4_peer_exchange_bytes(h, prefill, decode, true, false, 2).unwrap(),
                    8 * (prefill * h * 2).max(decode * h * 4) + 512);
                assert_eq!(deepseek_v4_peer_exchange_bytes(h, prefill, decode, true, true, 2).unwrap(),
                    8 * prefill.max(decode) * h * 4 + 512);
                assert_eq!(deepseek_v4_peer_exchange_bytes(h, prefill, decode, false, false, 2).unwrap(),
                    8 * prefill.max(decode) * h * 2 + 256);
                assert_eq!(deepseek_v4_expert_exchange_bytes(h, 6, prefill, decode, 0, 2).unwrap(), lead);
                assert_eq!(deepseek_v4_expert_exchange_bytes(h, 6, prefill, decode, 1, 2).unwrap(), peer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(pro: bool) -> DeepseekV4Config {
        DeepseekV4Config::from_model_args(
            &serde_json::json!({
                "vocab_size": 129280, "dim": if pro {7168} else {4096},
                "moe_inter_dim": if pro {3072} else {2048}, "n_layers": 1,
                "n_heads": if pro {128} else {64}, "n_routed_experts": if pro {384} else {256},
                "n_shared_experts": 1, "n_activated_experts": 6, "score_func": "sqrtsoftplus",
                "route_scale": 1.5, "swiglu_limit": 10.0, "q_lora_rank": 1024,
                "head_dim": 512, "rope_head_dim": 64, "o_groups": 8, "o_lora_rank": 1024,
                "window_size": 128, "original_seq_len": 65536, "rope_theta": 10000,
                "rope_factor": 16, "beta_fast": 32, "beta_slow": 1, "index_n_heads": 64,
                "index_head_dim": 128, "index_topk": 512, "hc_mult": 4,
                "hc_sinkhorn_iters": 20, "compress_rope_theta": 160000, "compress_ratios": [4],
                "dspark_target_layer_ids": [0, 0, 0]
            }),
            0,
        )
        .unwrap()
    }

    #[test]
    fn pro_workspace_tracks_wider_heads_and_hidden_without_flash_constants() {
        // Sum of the engine's allocations, at 128K compiled context, 4096
        // prefill rows x2 lanes, 64 decode rows and 1032 physical pool units.
        let scratch = V4WorkspaceScratch {
            shared_bytes: 490_734_592,
            index_topk_bytes: 558_007_296,
            decode_shared_bytes: 490_734_592,
            decode_index_topk_bytes: 558_007_296,
        };
        let packed_tables = 2 * (96 * (16 + 512 + 16 + 1024 * 4 + 18 * 4) + 15 * 1032 * 4 + 16 * 3 * 4);
        let flash =
            deepseek_v4_workspace_geometry(&config(false), 4096, 64, 131072, 2, 2, scratch).unwrap();
        assert_eq!(flash[0].device_bytes(1032).unwrap(), 3_677_619_184 - (2 * 4096 + 64) * 4096 * 4 + packed_tables);
        assert_eq!(flash[1].device_bytes(1032).unwrap(), 3_330_410_480 - (2 * 4096 + 64) * 4096 * 4 + packed_tables);
        assert_eq!(flash[0].pool_unit_device_bytes, (2 * V4_PACKED_MEMBERS as u64 + 64) * 4);
        let pro = deepseek_v4_workspace_geometry(
            &config(true),
            4096,
            64,
            131072,
            2,
            2,
            V4WorkspaceScratch {
                index_topk_bytes: 574_784_512,
                decode_index_topk_bytes: 574_784_512,
                ..scratch
            },
        )
        .unwrap();
        assert_eq!(pro[0].device_bytes(1032).unwrap(), 4_906_392_560 - (2 * 4096 + 64) * 7168 * 4 + packed_tables);
        assert_eq!(pro[1].device_bytes(1032).unwrap(), 4_286_554_096 - (2 * 4096 + 64) * 7168 * 4 + packed_tables);
        assert_eq!(
            pro[0].device_bytes(1033).unwrap() - pro[0].device_bytes(1032).unwrap(),
            pro[0].pool_unit_device_bytes
        );
        // The table floors are still exact for a custom one-row manifest.
        let tiny =
            deepseek_v4_workspace_geometry(&config(false), 1, 1, 128, 1, 2, scratch).unwrap()[0].clone();
        for units in [0, 1, 2, 4, 5, 1032] {
            let runtime_prefill = (V4_PACKED_MEMBERS as u64 * (units + 3) * 4).max(256);
            let runtime_decode = (units * 4).max(256);
            assert_eq!(tiny.device_bytes(units).unwrap() - tiny.fixed_device_bytes,
                2 * runtime_prefill + runtime_decode);
        }
        assert_eq!(tiny.device_bytes(1).unwrap() - tiny.fixed_device_bytes, 768);
    }

    #[test]
    fn draft_workspace_is_owned_only_by_the_selected_lead_rank() {
        for pro in [false, true] {
            let mut cfg = config(pro);
            let scratch = V4WorkspaceScratch::default();
            let off = deepseek_v4_workspace_geometry(&cfg, 4096, 64, 131072, 2, 2, scratch).unwrap();
            cfg.n_mtp_layers = 3;
            let selected = deepseek_v4_workspace_geometry(&cfg, 4096, 64, 131072, 2, 2, scratch).unwrap();
            // Runtime workspace_here allocates taps per lane plus six shared
            // draft buffers per step, each with the same 256-byte floor.
            let h = cfg.dim as u64;
            let removed: u64 = [(4096u64, 2u64, true), (64, 1, false)].into_iter().map(|(rows, lanes, prefill)| {
                let taps = deepseek_v4_tap_rows(rows as usize, prefill) as u64 * cfg.dspark_target_layer_ids.len() as u64 * h * 2;
                let shared = [rows.min(128) * h * 2, rows.min(128) * h * 4,
                    rows * 4, rows * 4, rows * 256 * 8, rows * 4];
                lanes * (taps.max(256) - 256)
                    + shared.into_iter().map(|bytes| bytes.max(256) - 256).sum::<u64>()
            }).sum();
            assert_eq!(selected[0].device_bytes(1032).unwrap() - off[0].device_bytes(1032).unwrap(), removed);
            assert_eq!(selected[1], off[1]);
            assert_eq!(selected[0].pool_unit_device_bytes, off[0].pool_unit_device_bytes);
        }
    }

    #[test]
    fn runtime_and_planner_scratch_agree_for_every_namespace_and_split() {
        use cuteafd_core::coordinator_programs::CoordinatorPrograms;
        for family in ["dsv4f", "dsv4p", "glm", "glmf", "mimo", "mimof", "mimop", "qwen4"] {
            let split_family = format!("{family}2");
            let sizes = [(format!("{family}_producer_m4096"), 100),
                (format!("{split_family}_producer_m4096"), 200),
                (format!("{family}_index_topk_prefill_m4096"), 300),
                (format!("{family}_index_topk_decode_m64"), 40),
                ("unrelated_huge_m4096".to_string(), 99999)];
            let manifest = serde_json::json!({"programs": sizes.iter().map(|(name, bytes)|
                serde_json::json!({"name": name, "scratch_bytes_at_capacity": {"scratch": bytes}}))
                .collect::<Vec<_>>()});
            for split in [false, true] {
                let runtime = CoordinatorPrograms { family, split_family: split.then_some(split_family.as_str()) }
                    .shared_scratch(sizes.iter().map(|(name, bytes)| (name.as_str(), *bytes)));
                let planned = deepseek_v4_workspace_scratch(&manifest, family, split, 4096, 64).unwrap();
                assert_eq!(planned.shared_bytes, runtime);
                assert_eq!(runtime, if split { 200 } else { 100 });
                assert_eq!(planned.index_topk_bytes, 300);
                assert_eq!(planned.decode_shared_bytes, 0);
                assert_eq!(planned.decode_index_topk_bytes, 40);
            }
        }
    }

    #[test]
    fn decode_arena_excludes_prefill_scratch_and_requires_its_own_topk() {
        let mut manifest = serde_json::json!({"capacities": {"prefill_rows": 4096}, "programs": [
            {"name": "dsv4f_producer_m4096", "scratch_bytes_at_capacity": {"scratch": 9000}},
            {"name": "dsv4f_producer_m64", "scratch_bytes_at_capacity": {"scratch": 600}},
            {"name": "dsv4f_compressor_continuation_c4", "scratch_bytes_at_capacity": {"scratch": 400}},
            {"name": "dsv4f_index_topk_prefill_m4096", "scratch_bytes_at_capacity": {"scratch": 10000}},
            {"name": "dsv4f_index_topk_decode_m64", "scratch_bytes_at_capacity": {"scratch": 700}}
        ]});
        let scratch = deepseek_v4_workspace_scratch(&manifest, "dsv4f", false, 2752, 64).unwrap();
        assert_eq!((scratch.shared_bytes, scratch.index_topk_bytes), (9000, 10000));
        assert_eq!((scratch.decode_shared_bytes, scratch.decode_index_topk_bytes), (600, 700));
        manifest["programs"].as_array_mut().unwrap().pop();
        assert!(deepseek_v4_workspace_scratch(&manifest, "dsv4f", false, 2752, 64).is_err());
    }

    #[test]
    fn tp2_payload_matches_planes_and_is_not_in_head_split_steps() {
        for hidden in [4096, 7168] {
            assert_eq!(deepseek_v4_tp2_payload_bytes(hidden, 2752, 64, 3).unwrap(),
                (3 * 2752 + 64) * hidden * 4);
        }
        assert_eq!(deepseek_v4_tp2_payload_bytes(1, 1, 1, 3).unwrap(), 512);
        assert_eq!(deepseek_v4_tp2_payload_bytes(32, 1, 1, 3).unwrap(), 384 + 256);
        assert!(deepseek_v4_tp2_payload_bytes(u64::MAX, 2752, 64, 3).is_err());
        let engine = include_str!("../../../cuteafd-daemon/src/families/deepseek_v4/engine.rs");
        assert!(engine.contains("let payload = self.tp2.as_ref().map("));
        assert!(!engine.contains("let payload = self.split_family.map("));
    }

    #[test]
    fn live_v4_units_select_the_compiled_topk_scratch() {
        let manifest = serde_json::json!({"capacities": {"prefill_rows": 4096}, "programs": [
            {"name": "dsv4p_index_topk_prefill_m4096", "scratch_bytes_at_capacity": {"scratch": 300}},
            {"name": "dsv4p_index_topk_decode_m64", "scratch_bytes_at_capacity": {"scratch": 40}}
        ]});
        for rows in [512, 1024, 2048, 2752, 4096] {
            assert_eq!(deepseek_v4_workspace_scratch(&manifest, "dsv4p", false, rows, 64)
                .unwrap().index_topk_bytes, 300);
        }
        let scratch = deepseek_v4_workspace_scratch(&manifest, "dsv4p", false, 2752, 64).unwrap();
        for ranks in [1, 2] {
            let cfg = config(true);
            let with = deepseek_v4_workspace_geometry(&cfg, 2752, 64, 131072, ranks, 3, scratch).unwrap();
            let without = deepseek_v4_workspace_geometry(&cfg, 2752, 64, 131072, ranks, 3,
                V4WorkspaceScratch::default()).unwrap();
            for (actual, baseline) in with.iter().zip(&without) {
                // Only prefill needs the 300-byte top-k export; decode's 40
                // bytes fit the allocation floor, independent of rows or waves.
                assert_eq!(actual.fixed_device_bytes - baseline.fixed_device_bytes, 300 - 256);
                assert_eq!(actual.prefill_waves, 3);
                assert_eq!(actual.pool_unit_device_bytes, baseline.pool_unit_device_bytes);
                assert_eq!(actual.pinned_host_bytes, baseline.pinned_host_bytes);
            }
        }
    }

    #[test]
    fn scratch_matches_runtime_selector_and_selected_family_topk() {
        let manifest = serde_json::json!({"programs": [
            {"name": "dsv4f_wo", "scratch_bytes_at_capacity": {"scratch": 100}},
            {"name": "glm_wo", "scratch_bytes_at_capacity": {"scratch": 400}},
            {"name": "dsv4p_index_topk_prefill_m4096", "scratch_bytes_at_capacity": {"scratch": 99999}},
            {"name": "dsv4f_index_topk_prefill_m4096", "scratch_bytes_at_capacity": {"scratch": 200}},
            {"name": "dsv4f_index_topk_decode_m64", "scratch_bytes_at_capacity": {"scratch": 50}}
        ]});
        assert_eq!(
            deepseek_v4_workspace_scratch(&manifest, "dsv4f", false, 4096, 64).unwrap(),
            V4WorkspaceScratch {
                shared_bytes: 100,
                index_topk_bytes: 200,
                decode_shared_bytes: 100,
                decode_index_topk_bytes: 50,
            }
        );
        assert!(deepseek_v4_workspace_scratch(&manifest, "dsv4p", false, 4096, 64).is_err());
    }
}
