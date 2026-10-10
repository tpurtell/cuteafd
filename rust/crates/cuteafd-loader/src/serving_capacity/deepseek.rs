//! Storage costs from the DeepSeek engines' cache allocation formulas.
//! Workspaces, mapped-table staging and drafter experts need separate reservations.
use super::{
    product, sum, CacheGeometryError, FamilyCacheGeometry, KvPlacement, RankCacheGeometry,
};
use crate::families::deepseek_v4::DeepseekV4Config;
use serde_json::Value;

/// V4's 256-source-token unit contains each C4 layer's compressed/index
/// pages and each C128 layer's compressed page. The head split replicates
/// these records and the compressor carry; it does not partition them.
/// Persistent pool bytes of one 256-token unit in one V4 layer, by compress
/// ratio: C4 records plus index keys, C128 records, none for window-only
/// layers. metadata::compressed_page_bytes rounds (rows * 584) to 576 bytes.
pub fn deepseek_v4_layer_unit_bytes(ratio: usize) -> u64 {
    match ratio {
        4 => 37_440 + 8_448,
        128 => 1_728,
        _ => 0,
    }
}

pub fn deepseek_v4_cache_geometry(
    cfg: &DeepseekV4Config,
    ranks: usize,
    prefill_rows: u64,
    draft_stages: usize,
) -> Result<FamilyCacheGeometry, CacheGeometryError> {
    if ![1, 2].contains(&ranks)
        || cfg.n_layers == 0
        || cfg.compress_ratios.len() != cfg.n_layers
        || cfg
            .compress_ratios
            .iter()
            .any(|r| !matches!(r, 0 | 4 | 128))
        || cfg.head_dim != 512
        || cfg.rope_head_dim != 64
        || cfg.index_head_dim != 128
        || cfg.window_size != 128
        || prefill_rows == 0
        || draft_stages > cfg.n_mtp_layers
    {
        return Err(CacheGeometryError::Unsupported {
            family: "deepseek_v4",
            what: "window/compressor record, prefill, drafter or coordinator rank geometry",
        });
    }
    let c4 = cfg.compress_ratios.iter().filter(|&&r| r == 4).count() as u64;
    let c128 = cfg.compress_ratios.iter().filter(|&&r| r == 128).count() as u64;
    let pages = cfg.compress_ratios.iter().try_fold(0u64, |total, &ratio| {
        sum("V4 compressed units", &[total, deepseek_v4_layer_unit_bytes(ratio)])
    })?;
    // engine::pool_layer_for: four FP32 carry arrays on C4; two on C128.
    let carry = sum(
        "V4 compressor carry",
        &[
            product("V4 C4 carry", &[c4, 2 * 16 * (1024 + 256) * 4])?,
            product("V4 C128 carry", &[c128, 2 * 256 * 512 * 4])?,
        ],
    )?;
    let ring_pages = prefill_rows
        .checked_add(128)
        .ok_or(CacheGeometryError::Overflow("V4 ring rows"))?
        .div_ceil(256);
    let window = product("V4 window ring", &[ring_pages, 149_760])?;
    let mut costs = Vec::with_capacity(ranks);
    for rank in 0..ranks {
        // The peer allocates backbone caches only. The lead also holds the
        // drafter stages' window rings, even though their experts are separate.
        let layers = (cfg.n_layers as u64)
            .checked_add(if rank == 0 { draft_stages as u64 } else { 0 })
            .ok_or(CacheGeometryError::Overflow("V4 cached layers"))?;
        costs.push(RankCacheGeometry {
            persistent_unit_bytes: pages,
            // PoolShape::units_for adds one partial compressed unit per slot.
            active_state_per_sequence_bytes: sum(
                "V4 active state",
                &[product("V4 window rings", &[layers, window])?, carry, pages],
            )?,
            retained_mark_bytes: sum(
                "V4 prefix mark",
                &[product("V4 prefix windows", &[layers, 128, 584])?, carry],
            )?,
            context_table_bytes_per_token: 2 * 64 * 4,
            ..Default::default()
        });
    }
    Ok(FamilyCacheGeometry {
        logical_unit_rows: 256,
        placement: if ranks == 1 {
            KvPlacement::SingleDevice
        } else {
            KvPlacement::Replicated
        },
        ranks: costs,
    })
}

/// V4.1's target-only CED cache. The qualified dual-GPU placement assigns
/// layers 0..20 and sources 2/8/14 to RTX0, layers 20..40 and source 20 to
/// RTX1. One 512-token source group is three C2 pages and two C1 pages.
/// Source page tables are bounded by 4096 entries per sequence; their maximum
/// footprint is reserved here (at most 64 KiB per sequence of overestimate
/// for a small pool). Drafter state is deliberately outside this geometry.
pub fn deepseek_v41_cache_geometry(
    config: &Value,
    ranks: usize,
) -> Result<FamilyCacheGeometry, CacheGeometryError> {
    let text = config.get("text_config").unwrap_or(config);
    let list = |key: &str| -> Option<Vec<u64>> {
        text.get(key)?
            .as_array()?
            .iter()
            .map(Value::as_u64)
            .collect()
    };
    let ratios: Vec<u64> = (0..40)
        .map(|l| {
            if l < 2 {
                0
            } else if l < 20 {
                2
            } else {
                1
            }
        })
        .collect();
    let actual_ratios = list("compress_ratios");
    if ![1, 2].contains(&ranks)
        || text["num_hidden_layers"].as_u64() != Some(40)
        || text["head_dim"].as_u64() != Some(512)
        || text["qk_rope_head_dim"].as_u64() != Some(64)
        || text["sliding_window"].as_u64() != Some(128)
        || list("kv_source_layer_ids").as_deref() != Some(&[2, 8, 14, 20][..])
        || actual_ratios
            .as_ref()
            .is_none_or(|r| r.get(..40) != Some(ratios.as_slice()))
    {
        return Err(CacheGeometryError::Unsupported {
            family: "deepseek_v41",
            what: "only the 40-layer CED cache with sources 2/8/14/20 and its 20/20 placement is implemented",
        });
    }
    // SourceCache: per row 64 packed index keys + FP32 scale, and FP4
    // KV values (256 bytes) + 32 bytes of block scales.
    let source_page = 256 * (68 + 288);
    let rank = |windows: u64, c2: u64, c1: u64| RankCacheGeometry {
        persistent_unit_bytes: (c2 + 2 * c1) * source_page,
        active_state_per_sequence_bytes: windows * (128 * 528 + 8)
            + (c2 + c1) * (4096 * 4 + 8)
            + c2 * 4096,
        // BackbonePrefix reserves a 4096-byte compressor tail even for C1.
        retained_mark_bytes: windows * (128 * 528 + 8) + (c2 + c1) * 4096,
        ..Default::default()
    };
    Ok(FamilyCacheGeometry {
        logical_unit_rows: 512,
        placement: if ranks == 1 {
            KvPlacement::SingleDevice
        } else {
            KvPlacement::PartitionedLayers
        },
        ranks: if ranks == 1 {
            vec![rank(40, 3, 1)]
        } else {
            vec![rank(20, 3, 0), rank(20, 0, 1)]
        },
    })
}

/// Exact target cache allocation, including bounded source page tables. Owners
/// and records match `BackboneCache`; the geometry above reserves maximum
/// tables so it remains independent of the selected pool.
pub fn deepseek_v41_cache_bytes(config: &Value, ranks: usize, slots: u64, groups: u64,
    replicated: bool) -> Result<Vec<u64>, CacheGeometryError> {
    deepseek_v41_cache_geometry(config, ranks)?;
    if !(1..=16).contains(&slots) || !(1..=131_072).contains(&groups)
        || replicated && ranks != 2 {
        return Err(CacheGeometryError::Unsupported { family: "deepseek_v41",
            what: "active slots, source pool groups or replicated cache placement" });
    }
    let rank = |windows: u64, c2: u64, c1: u64| -> Result<u64, CacheGeometryError> {
        sum("V4.1 target cache", &[
            product("V4.1 source records", &[(c2 + 2 * c1), groups, 256 * 356])?,
            product("V4.1 window state", &[windows, slots, 128 * 528 + 8])?,
            product("V4.1 C2 source tables", &[c2, slots, groups.min(4096) * 4 + 8])?,
            product("V4.1 C1 source tables", &[c1, slots, (2 * groups).min(4096) * 4 + 8])?,
            product("V4.1 compressor carry", &[c2, slots, 4096])?,
        ])
    };
    if ranks == 1 { Ok(vec![rank(40, 3, 1)?]) }
    else if replicated {
        // Replica payloads contain KV only; index keys and compressor carry
        // remain on the source owner. Replica tables follow the source tables.
        let peer_c1 = groups * 2 * 256 * 288 + slots * ((2 * groups).min(4096) * 4 + 8);
        let peer_c2 = 3 * (groups * 256 * 288 + slots * (groups.min(4096) * 4 + 8));
        Ok(vec![rank(40, 3, 0)? + peer_c1, rank(40, 0, 1)? + peer_c2])
    } else { Ok(vec![rank(20, 3, 0)?, rank(20, 0, 1)?]) }
}

/// Resolve a shared source pool from residual per-device budgets, after all
/// fixed owners and runtime headroom have been reserved. Admission adds active
/// and retained copy-on-write tails to the requested logical capacity. An
/// explicit token request must fit; auto may reduce aggregate capacity.
pub fn deepseek_v41_pool_groups(config: &Value, slots: u64, retained_turns: u64,
    available: &[u64], tokens: u64, automatic: bool, replicated: bool)
    -> Result<u64, CacheGeometryError> {
    if retained_turns > 128 || tokens == 0 {
        return Err(CacheGeometryError::Unsupported { family: "deepseek_v41",
            what: "retained turn limit or target pool tokens" });
    }
    let minimum = product("V4.1 private tails", &[2, slots + retained_turns])?;
    let desired = tokens.div_ceil(512).checked_add(slots + 2 * retained_turns)
        .ok_or(CacheGeometryError::Overflow("V4.1 source groups"))?.max(minimum);
    let fits = |groups| -> Result<bool, CacheGeometryError> {
        Ok(deepseek_v41_cache_bytes(config, available.len(), slots, groups, replicated)?
            .iter().zip(available).all(|(used, free)| used <= free))
    };
    if desired > 131_072 || !fits(minimum)? {
        return Err(CacheGeometryError::Unsupported { family: "deepseek_v41",
            what: "source pool exceeds physical capacity or minimum admission budget" });
    }
    let (mut low, mut high) = (minimum, desired);
    while low < high {
        let mid = (low + high).div_ceil(2);
        if fits(mid)? { low = mid; } else { high = mid - 1; }
    }
    if !automatic && low != desired {
        return Err(CacheGeometryError::Unsupported { family: "deepseek_v41",
            what: "requested source pool does not fit the per-device memory budget" });
    }
    Ok(low)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v4_config(pro: bool) -> DeepseekV4Config {
        let layers = if pro { 61 } else { 43 };
        let ratios: Vec<usize> = (0..layers)
            .map(|l| {
                if l < 2 {
                    if pro {
                        128
                    } else {
                        0
                    }
                } else if l % 2 == 0 {
                    4
                } else {
                    128
                }
            })
            .collect();
        DeepseekV4Config::from_model_args(&serde_json::json!({
            "vocab_size": 129280, "dim": if pro { 7168 } else { 4096 },
            "moe_inter_dim": if pro { 3072 } else { 2048 }, "n_layers": layers,
            "n_heads": if pro { 128 } else { 64 }, "n_routed_experts": if pro { 384 } else { 256 },
            "n_shared_experts": 1, "n_activated_experts": 6, "score_func": "sqrtsoftplus",
            "route_scale": 1.5, "swiglu_limit": 10.0, "q_lora_rank": 1024,
            "head_dim": 512, "rope_head_dim": 64, "o_groups": 8, "o_lora_rank": 1024,
            "window_size": 128, "original_seq_len": 65536, "rope_theta": 10000,
            "rope_factor": 16, "beta_fast": 32, "beta_slow": 1, "index_n_heads": 64,
            "index_head_dim": 128, "index_topk": 512, "hc_mult": 4,
            "hc_sinkhorn_iters": 20, "compress_rope_theta": 160000, "compress_ratios": ratios
        }), 3).unwrap()
    }

    #[test]
    fn v4_flash_and_pro_include_padded_compressed_pages_and_slot_tails() {
        for (pro, expected_unit) in [(false, 998_208), (true, 1_430_208)] {
            let cfg = v4_config(pro);
            let geometry = deepseek_v4_cache_geometry(&cfg, 2, 4096, 0).unwrap();
            assert_eq!(geometry.logical_unit_rows, 256);
            assert_eq!(geometry.ranks[0], geometry.ranks[1]);
            let rank = geometry.ranks[0];
            assert_eq!(rank.persistent_unit_bytes, expected_unit);
            // For a 256-row prefill the ring has two pages per layer;
            // for 4096 rows it has seventeen. The compressed units and
            // carry remain identical across both serving shapes.
            let small = deepseek_v4_cache_geometry(&cfg, 1, 256, 0).unwrap().ranks[0];
            assert_eq!(
                rank.active_state_per_sequence_bytes - small.active_state_per_sequence_bytes,
                cfg.n_layers as u64 * 15 * 149_760
            );
            assert_eq!(rank.retained_mark_bytes, small.retained_mark_bytes);
            let carry = rank.retained_mark_bytes - cfg.n_layers as u64 * 128 * 584;
            assert_eq!(
                small.active_state_per_sequence_bytes,
                cfg.n_layers as u64 * 2 * 149_760 + carry + expected_unit
            );
            let draft = deepseek_v4_cache_geometry(&cfg, 2, 4096, 3).unwrap();
            assert_eq!(draft.ranks[1], rank);
            assert_eq!(
                draft.ranks[0].active_state_per_sequence_bytes
                    - rank.active_state_per_sequence_bytes,
                3 * 17 * 149_760
            );
            assert_eq!(
                draft.ranks[0].retained_mark_bytes - rank.retained_mark_bytes,
                3 * 128 * 584
            );
            assert!(deepseek_v4_cache_geometry(&cfg, 1, 0, 0).is_err());
            assert!(deepseek_v4_cache_geometry(&cfg, 1, 4096, 4).is_err());
        }
    }

    #[test]
    fn v41_source_groups_follow_ced_owners_without_replication() {
        let config: Value = serde_json::from_str(include_str!(
            "../families/deepseek_v41/official-v41-config.json"
        ))
        .unwrap();
        let single = deepseek_v41_cache_geometry(&config, 1).unwrap();
        let dual = deepseek_v41_cache_geometry(&config, 2).unwrap();
        let a = single.ranks[0];
        let (b, c) = (dual.ranks[0], dual.ranks[1]);
        assert_eq!(a.persistent_unit_bytes, 455_680);
        assert_eq!(b.persistent_unit_bytes, 273_408);
        assert_eq!(c.persistent_unit_bytes, 182_272);
        assert_eq!(
            a.active_state_per_sequence_bytes,
            b.active_state_per_sequence_bytes + c.active_state_per_sequence_bytes
        );
        assert_eq!(
            a.retained_mark_bytes,
            b.retained_mark_bytes + c.retained_mark_bytes
        );
        assert_eq!(dual.placement, KvPlacement::PartitionedLayers);
        let mut bad = config.clone();
        bad["text_config"]["kv_source_layer_ids"] = serde_json::json!([2, 8, 14, 24]);
        assert!(matches!(
            deepseek_v41_cache_geometry(&bad, 2),
            Err(CacheGeometryError::Unsupported { .. })
        ));
        assert!(deepseek_v41_cache_geometry(&config, 3).is_err());
    }

    #[test]
    fn v41_exact_tables_and_pool_admission_use_the_tighter_owner() {
        let config: Value = serde_json::from_str(include_str!(
            "../families/deepseek_v41/official-v41-config.json")).unwrap();
        let groups = 4096 + 16 + 40;
        let exact = deepseek_v41_cache_bytes(&config, 2, 16, groups, false).unwrap();
        let single = deepseek_v41_cache_bytes(&config, 1, 16, groups, false).unwrap();
        assert_eq!(single[0], exact.iter().sum::<u64>());
        let geometry = deepseek_v41_cache_geometry(&config, 2).unwrap();
        for (bytes, rank) in exact.iter().zip(&geometry.ranks) {
            assert_eq!(*bytes, groups * rank.persistent_unit_bytes + 16 * rank.active_state_per_sequence_bytes);
        }
        assert_eq!(deepseek_v41_pool_groups(&config, 16, 20, &exact, 2 << 20, true, false).unwrap(), groups);
        let mut short = exact.clone();
        short[1] -= 1;
        assert_eq!(deepseek_v41_pool_groups(&config, 16, 20, &short, 2 << 20, true, false).unwrap(), groups - 1);
        assert!(deepseek_v41_pool_groups(&config, 16, 20, &short, 2 << 20, false, false).is_err());
        assert!(deepseek_v41_pool_groups(&config, 16, 20, &[0, 0], 2 << 20, true, false).is_err());
        let small = deepseek_v41_cache_bytes(&config, 1, 16, 72, false).unwrap()[0];
        assert_eq!(small, 72 * 455680 + 16 * (40 * 67592 + 3 * (72 * 4 + 8 + 4096) + 144 * 4 + 8));
    }
}
