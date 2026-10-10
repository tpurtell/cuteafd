//! Header/geometry-only counterpart of V4 `workspace_here`/`step_buffers`.
//! Device allocations are rounded to at least 256 bytes exactly as in the
//! engine. Prefill has two lanes and decode one; both arenas stay resident.
use super::{product, sum, CacheGeometryError};
use crate::families::deepseek_v4::DeepseekV4Config;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Small cards reserve workspaces and graphs explicitly, with the shared
/// free-memory floor. PRO cards retain the historical total reserve envelope.
pub fn deepseek_v4_headroom_bytes(total_bytes: u64, reserve_bytes: u64, workspace_bytes: u64,
    graph_bytes: u64) -> u64 {
    let small = cuteafd_core::serving_capacity::small_card_headroom_bytes(total_bytes);
    let remainder = reserve_bytes.saturating_sub(workspace_bytes.saturating_add(graph_bytes));
    if small > 0 {
        // Preserve larger caller-requested envelopes; only the default shrinks.
        if reserve_bytes > 10 << 30 { small.max(remainder) } else { small }
    } else { remainder.max(3 << 30) }
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

/// Four peer slots per each of the two prefill lanes, plus release flags.
pub fn deepseek_v4_peer_exchange_bytes(hidden: u64, prefill_rows: u64, decode_rows: u64)
    -> Result<u64, CacheGeometryError> {
    sum("V4 peer exchange", &[product("V4 peer slots", &[8, prefill_rows.max(decode_rows), hidden, 2])?, 256])
}

/// Additional expert slots and the lead's packing buffer. Per-lane/parity
/// slots carry route IDs, route weights and the lead shared-expert half.
pub fn deepseek_v4_expert_exchange_bytes(hidden: u64, topk: u64, prefill: u64, decode: u64, rank: usize)
    -> Result<u64, CacheGeometryError> {
    // Each section of the packed payload starts 16-byte aligned (+32 B).
    let payload = sum("V4 expert payload", &[product("V4 expert rows", &[prefill.max(decode), hidden * 2 + topk * 8])?, 32])?;
    sum("V4 expert exchange", &[product("V4 expert slots", &[4, payload])?,
        256, if rank == 0 { payload.max(256) } else { 0 },
        (hidden + hidden / 32).max(256),
        if rank == 1 { product("V4 peer wire", &[2 * prefill + decode, hidden + hidden / 32])? } else { 3 * 256 }])
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct V4WorkspaceScratch {
    /// Largest non-index-topk scratch of the serving and head-split families.
    pub shared_bytes: u64,
    /// Largest selected family's decode/prefill index-topk scratch.
    pub index_topk_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct V4WorkspaceRank {
    /// Everything except the pool-unit-sized C4 page tables.
    pub fixed_device_bytes: u64,
    /// C4 page tables: one row/unit entry in each persistent workspace lane.
    pub pool_unit_device_bytes: u64,
    /// Pinned router staging, reported outside GPU admission.
    pub pinned_host_bytes: u64,
    /// Per-lane table slopes preserve the 256-byte allocation floor for
    /// custom manifests with fewer than 64 rows.
    pub prefill_lane_table_unit_bytes: u64,
    pub decode_lane_table_unit_bytes: u64,
}

impl V4WorkspaceRank {
    pub fn device_bytes(self, units: u64) -> Result<u64, CacheGeometryError> {
        sum(
            "V4 workspace total",
            &[
                self.fixed_device_bytes,
                product(
                    "V4 prefill pool tables",
                    &[
                        2,
                        product(
                            "V4 prefill lane table",
                            &[units, self.prefill_lane_table_unit_bytes],
                        )?
                        .max(256),
                    ],
                )?,
                product(
                    "V4 decode pool table",
                    &[units, self.decode_lane_table_unit_bytes],
                )?
                .max(256),
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
    let shared_bytes = selected.shared_scratch(sizes);
    let mut index_topk_bytes = 0;
    for name in [
        format!("{family}_index_topk_prefill_m{prefill_rows}"),
        format!("{family}_index_topk_decode_m{decode_rows}"),
    ] {
        let program = programs
            .iter()
            .find(|p| p["name"].as_str() == Some(name.as_str()))
            .ok_or_else(unsupported)?;
        index_topk_bytes = index_topk_bytes.max(
            program["scratch_bytes_at_capacity"]["scratch"]
                .as_u64()
                .ok_or_else(unsupported)?,
        );
    }
    Ok(V4WorkspaceScratch {
        shared_bytes,
        index_topk_bytes,
    })
}

pub fn deepseek_v4_workspace_geometry(
    cfg: &DeepseekV4Config,
    prefill_rows: u64,
    decode_rows: u64,
    max_context: u64,
    ranks: usize,
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
            let prefill = step(cfg, prefill_rows, 2, width, rank, ranks, scratch)?;
            let decode = step(cfg, decode_rows, 1, width, rank, ranks, scratch)?;
            Ok(V4WorkspaceRank {
                fixed_device_bytes: sum("V4 prefill/decode workspaces", &[prefill.0, decode.0])?,
                pool_unit_device_bytes: sum(
                    "V4 prefill/decode table slope",
                    &[prefill.1, decode.1],
                )?,
                pinned_host_bytes: sum("V4 prefill/decode pinned staging", &[prefill.2, decode.2])?,
                prefill_lane_table_unit_bytes: product(
                    "V4 prefill table slope",
                    &[prefill_rows, 4],
                )?,
                decode_lane_table_unit_bytes: product("V4 decode table slope", &[decode_rows, 4])?,
            })
        })
        .collect()
}

fn step(
    cfg: &DeepseekV4Config,
    rows: u64,
    lanes: u64,
    c128_width: u64,
    rank: usize,
    ranks: usize,
    scratch: V4WorkspaceScratch,
) -> Result<(u64, u64, u64), CacheGeometryError> {
    let mul = |values: &[u64]| product("V4 workspace allocation", values);
    let h = cfg.dim as u64;
    let heads = cfg.n_heads as u64 / if rank == 1 { 2 } else { 1 };
    let topk = cfg.n_activated_experts as u64;
    let lead = |bytes: u64| if rank == 0 { bytes } else { 256 };
    // Eighteen metadata arrays (C4/C128 x9); four length/visible arrays.
    let metadata_rows = rows
        .checked_add(2)
        .ok_or(CacheGeometryError::Overflow("V4 metadata rows"))?;
    let lane_allocations = [
        mul(&[rows, 4, h, 2])?,
        mul(&[rows, 4, h, 2])?,
        mul(&[rows, 4, 4])?,
        mul(&[rows, 16, 4])?,
        mul(&[rows, 4])?,
        mul(&[rows, h, 2])?,
        mul(&[
            rows.min(128),
            cfg.dspark_target_layer_ids.len() as u64,
            h,
            2,
        ])?,
        mul(&[rows, 8])?,
        mul(&[rows, 8])?,
        mul(&[rows, 128, 4])?,
        // c128_indices is a compiled context-sized table, not a pool table.
        mul(&[rows, c128_width, 4])?,
    ];
    let lane = sum("V4 lane allocations", &lane_allocations.map(|n| n.max(256)))?;
    let lane = sum("V4 lane payload", &[lane,
        if ranks == 2 { mul(&[rows, h, 4])?.max(256) } else { 0 }])?;
    let lane = sum(
        "V4 lane metadata",
        &[
            lane,
            mul(&[18, mul(&[metadata_rows, 4])?.max(256)])?,
            mul(&[4, mul(&[rows, 4])?.max(256)])?,
        ],
    )?;
    let fixed_allocations = [
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
        mul(&[rows.min(128), h, 2])?,
        mul(&[rows.min(128), h, 4])?,
        mul(&[rows, 4])?,
        mul(&[rows, 4])?,
        // native dsv4_dspark.cu: kArgmaxBlocks=256, value/id pairs x8.
        mul(&[rows, 256, 8])?,
        mul(&[rows, 4])?,
        if ranks == 2 { mul(&[rows, h, 2])? } else { 256 },
        lead(4 << 20), // VocabularyHead persistent workspace.
    ];
    let fixed = sum(
        "V4 step allocations",
        &fixed_allocations.map(|n| n.max(256)),
    )?;
    Ok((
        sum("V4 step and lanes", &[fixed, mul(&[lanes, lane])?])?,
        mul(&[rows, lanes, 4])?,
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
                let payload = prefill.max(decode) * (h * 2 + 6 * 8) + 32;
                let check = h + h / 32;
                let lead = 4 * payload + 256 + payload + check + 3 * 256;
                let peer = 4 * payload + 256 + check + (2 * prefill + decode) * check;
                assert_eq!(deepseek_v4_expert_exchange_bytes(h, 6, prefill, decode, 0).unwrap(), lead);
                assert_eq!(deepseek_v4_expert_exchange_bytes(h, 6, prefill, decode, 1).unwrap(), peer);
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
    fn small_card_headroom_keeps_exact_workspaces_and_pro_envelope() {
        use cuteafd_core::serving_capacity::{GpuMemoryBudget, SMALL_CARD_HEADROOM_BYTES};
        let reserve = 10 << 30;
        let workspace = 4 << 30;
        let graph = 400 << 20;
        for gib in [31.8, 32.0] {
            let total = GpuMemoryBudget::from_gib(gib).unwrap().0;
            assert_eq!(deepseek_v4_headroom_bytes(total, reserve, workspace, graph),
                SMALL_CARD_HEADROOM_BYTES);
            assert_eq!(deepseek_v4_headroom_bytes(total, 20 << 30, workspace, graph),
                (20 << 30) - workspace - graph);
        }
        let pro = GpuMemoryBudget::from_gib(95.5).unwrap().0;
        assert_eq!(deepseek_v4_headroom_bytes(pro, reserve, workspace, graph),
            reserve - workspace - graph);
        assert_eq!(deepseek_v4_headroom_bytes(pro, reserve, 9 << 30, graph), 3 << 30);
    }

    #[test]
    fn pro_workspace_tracks_wider_heads_and_hidden_without_flash_constants() {
        // Sum of the engine's allocations, at 128K compiled context, 4096
        // prefill rows x2 lanes, 64 decode rows and 1032 physical pool units.
        let scratch = V4WorkspaceScratch {
            shared_bytes: 490_734_592,
            index_topk_bytes: 558_007_296,
        };
        let flash =
            deepseek_v4_workspace_geometry(&config(false), 4096, 64, 131072, 2, scratch).unwrap();
        assert_eq!(flash[0].device_bytes(1032).unwrap(), 3_732_576_176);
        assert_eq!(flash[1].device_bytes(1032).unwrap(), 3_385_367_472);
        assert_eq!(flash[0].pool_unit_device_bytes, (4096 * 2 + 64) * 4);
        let pro = deepseek_v4_workspace_geometry(
            &config(true),
            4096,
            64,
            131072,
            2,
            V4WorkspaceScratch {
                index_topk_bytes: 574_784_512,
                ..scratch
            },
        )
        .unwrap();
        assert_eq!(pro[0].device_bytes(1032).unwrap(), 4_970_786_736);
        assert_eq!(pro[1].device_bytes(1032).unwrap(), 4_350_948_272);
        assert_eq!(
            pro[0].device_bytes(1033).unwrap() - pro[0].device_bytes(1032).unwrap(),
            pro[0].pool_unit_device_bytes
        );
        // The table floors are still exact for a custom one-row manifest.
        let tiny =
            deepseek_v4_workspace_geometry(&config(false), 1, 1, 128, 1, scratch).unwrap()[0];
        assert_eq!(
            tiny.device_bytes(1).unwrap() - tiny.fixed_device_bytes,
            3 * 256
        );
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
                index_topk_bytes: 200
            }
        );
        assert!(deepseek_v4_workspace_scratch(&manifest, "dsv4p", false, 4096, 64).is_err());
    }
}
