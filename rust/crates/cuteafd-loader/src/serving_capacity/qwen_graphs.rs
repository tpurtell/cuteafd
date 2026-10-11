//! Qwen 3.8 Flash Next's startup decode graph set: the one definition the
//! engine's warm-up captures and admission (runtime and planner) reserves.
//! Each shape is a (rows, speculative, table geometry) step; every shape is
//! captured once per layer segment plus the head (`layers + 1`).
use crate::placement::inventory::{GraphSet, Lifetime};

/// Rows of the decode-shaped programs (`_m64`).
pub const QWEN_DECODE_ROWS: usize = 64;
/// Plain decode row buckets.
pub const QWEN_PLAIN_BUCKETS: &[usize] = &[1, 4, 8, 16];
/// Speculative verify row buckets.
pub const QWEN_SPEC_BUCKETS: &[usize] = &[2, 4, 8, 16, 24, 32, 64];
/// Tokens per allocation unit (four 64-row record pages) and pages per unit.
pub const QWEN_UNIT_ROWS: usize = 256;
pub const QWEN_UNIT_PAGES: usize = 4;
/// SM120 PRO, driver 595.91.07: 4,569,694,208 B / 30,380 startup graphs at the
/// 1-RTX qualified layout (v3-p2 ready ledgers, EXL3 and NVFP4 alike,
/// 2026-10-10), rounded up; 2026-10-07 TP4 measured 149,712 (12,397 graphs).
pub const QWEN_GRAPH_BYTES_PER_GRAPH: u64 = 150_418;
pub const QWEN_GRAPH_MARGIN_PERCENT: u64 = 10;
pub const QWEN_GRAPH_MARGIN_BYTES: u64 = 256 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QwenGraphGeometry {
    pub pool_width: usize,
    pub page_stride: usize,
    pub pool_stride: usize,
    pub long: bool,
}

/// The decode row bucket of `rows` live rows.
pub fn qwen_decode_bucket(rows: usize, spec: bool) -> usize {
    let buckets = if spec { QWEN_SPEC_BUCKETS } else { QWEN_PLAIN_BUCKETS };
    buckets.iter().copied().find(|&bucket| rows <= bucket).unwrap_or(rows)
}

/// Preserve the established short-context allocation buckets, but do not key a
/// long decode by unused pages reserved beyond its live power-of-two bucket.
pub fn decode_allocation_units(allocated: usize, live_tokens: usize, unit: usize) -> usize {
    allocated.min(live_tokens.max(131_072).div_ceil(unit).next_power_of_two())
}

/// Every table geometry a decode step over `pages` record pages can take for
/// sequences up to `context` tokens, short (at most `dense`) and long.
pub fn qwen_graph_geometries(context: usize, pages: usize, dense: usize) -> Vec<QwenGraphGeometry> {
    let pools = pages / QWEN_UNIT_PAGES;
    let mut geometries = Vec::new();
    for units in 1..=context.div_ceil(QWEN_UNIT_ROWS).min(pools) {
        let capacity = (units * QWEN_UNIT_ROWS).min(context);
        let mut width = 1;
        while width / 2 * QWEN_UNIT_ROWS < capacity {
            let low = if width == 1 { 1 } else { width / 2 * QWEN_UNIT_ROWS + 1 };
            let high = (width * QWEN_UNIT_ROWS).min(capacity);
            for long in [false, true] {
                if (!long && low <= high.min(dense)) || (long && low.max(dense + 1) <= high) {
                    let live_units = decode_allocation_units(units, high, QWEN_UNIT_ROWS);
                    let pool_stride = live_units.next_power_of_two().min(pools);
                    let page_stride = (live_units * QWEN_UNIT_PAGES).next_power_of_two().min(pages);
                    let geometry = QwenGraphGeometry { pool_width: width.min(pool_stride), page_stride, pool_stride, long };
                    if !geometries.contains(&geometry) { geometries.push(geometry); }
                }
            }
            width *= 2;
        }
    }
    geometries
}

/// The startup shapes: every geometry x the plain buckets up to `sequences`
/// and, with speculation, every verify bucket.
pub fn qwen_serving_graph_shapes(context: usize, pages: usize, dense: usize, sequences: usize, speculation: bool)
    -> Vec<(usize, bool, QwenGraphGeometry)> {
    let plain = qwen_decode_bucket(sequences.min(QWEN_DECODE_ROWS), false);
    qwen_graph_geometries(context, pages, dense).into_iter().flat_map(|geometry| {
        QWEN_PLAIN_BUCKETS.iter().copied().filter(move |&rows| rows <= plain).map(move |rows| (rows, false, geometry))
            .chain(QWEN_SPEC_BUCKETS.iter().copied().filter(move |_| speculation).map(move |rows| (rows, true, geometry)))
    }).collect()
}

/// Record pages of a pool of `pool_tokens` tokens.
pub fn qwen_pool_pages(pool_tokens: usize) -> Option<usize> {
    pool_tokens.div_ceil(QWEN_UNIT_ROWS).checked_mul(QWEN_UNIT_PAGES)
}

/// The startup graph set at `pool_tokens`: shapes x (layers + 1) executables,
/// measured bytes each, +10% (at least 256 MiB).
pub fn qwen_startup_graphs(context: usize, pool_tokens: usize, dense: usize, sequences: usize, speculation: bool,
    layers: usize) -> Option<GraphSet> {
    let shapes = qwen_serving_graph_shapes(context, qwen_pool_pages(pool_tokens)?, dense, sequences, speculation).len();
    let executables = (shapes as u64).checked_mul(layers as u64 + 1)?;
    Some(GraphSet::new(&[executables], QWEN_GRAPH_BYTES_PER_GRAPH, QWEN_GRAPH_MARGIN_PERCENT, QWEN_GRAPH_MARGIN_BYTES,
        shapes as u64, Lifetime::Startup))
}

/// Owner-local layer fronts, owner0 head, and (for a dual owner chain) the
/// materialized owner0 cutover post and owner1 final post. This is the same
/// segment inventory as the executor, not a duplicated whole set per GPU.
pub fn qwen_graph_segments(owners: &[usize]) -> Option<Vec<u64>> {
    if owners.is_empty() || owners[0] != 0 || owners.iter().any(|&owner| owner > 1) { return None; }
    let dual = owners.contains(&1);
    if dual && (owners.last() != Some(&1)
        || owners.windows(2).filter(|pair| pair[0] != pair[1]).count() != 1) { return None; }
    let mut counts = vec![0u64; if dual { 2 } else { 1 }];
    for &owner in owners { counts[owner] = counts[owner].checked_add(1)?; }
    counts[0] = counts[0].checked_add(if dual { 2 } else { 1 })?;
    if dual { counts[1] = counts[1].checked_add(1)?; }
    Some(counts)
}

pub fn qwen_startup_graphs_placed(context: usize, pool_tokens: usize, dense: usize, sequences: usize,
    speculation: bool, owners: &[usize]) -> Option<GraphSet> {
    let shapes = qwen_serving_graph_shapes(context, qwen_pool_pages(pool_tokens)?, dense, sequences, speculation).len();
    let executables = qwen_graph_segments(owners)?.into_iter()
        .map(|segments| (shapes as u64).checked_mul(segments)).collect::<Option<Vec<_>>>()?;
    Some(GraphSet::new(&executables, QWEN_GRAPH_BYTES_PER_GRAPH, QWEN_GRAPH_MARGIN_PERCENT, QWEN_GRAPH_MARGIN_BYTES,
        shapes as u64, Lifetime::Startup))
}

#[cfg(test)]
mod owner_graph_tests {
    use super::*;

    #[test]
    fn qwen_owner_segments_match_cutover_and_exit() {
        assert_eq!(qwen_graph_segments(&[0; 48]), Some(vec![49]));
        for cut in [1, 24, 47] {
            let owners: Vec<_> = (0..48).map(|layer| usize::from(layer >= cut)).collect();
            assert_eq!(qwen_graph_segments(&owners), Some(vec![cut as u64 + 2, (48 - cut) as u64 + 1]));
            for speculation in [false, true] {
                let graphs = qwen_startup_graphs_placed(131072, 2097152, 32768, 16, speculation, &owners).unwrap();
                assert_eq!(graphs.ranks[0].executables, graphs.shapes * (cut as u64 + 2));
                assert_eq!(graphs.ranks[1].executables, graphs.shapes * ((48 - cut) as u64 + 1));
                assert_eq!(graphs.ranks.iter().map(|rank| rank.executables).sum::<u64>(), graphs.shapes * 51);
                for rank in 0..2 { assert_eq!(graphs.at_ready(rank) + graphs.growth(rank), graphs.bytes(rank)); }
            }
        }
    }

    #[test]
    fn qwen_owner_cache_preserves_global_totals_and_mtp_home() {
        let mut config = crate::plan::testing::qwen4_config(48);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        let mut cfg = crate::families::qwen4::Qwen4Config::from_hf(&config).unwrap();
        cfg.ple_layers = vec![1];
        for cut in [1, 2, 24, 47] {
            let owners: Vec<_> = (0..48).map(|layer| usize::from(layer >= cut)).collect();
            for (mtp, kv_format) in [false, true].into_iter().flat_map(|mtp|
                [crate::families::qwen4::Qwen4KvCache::Bf16, crate::families::qwen4::Qwen4KvCache::Fp8]
                    .into_iter().map(move |format| (mtp, format))) {
                let whole = super::super::qwen_cache_geometry(&cfg, 48, mtp, kv_format).unwrap();
                let placed = qwen_cache_geometry_placed(&cfg, &owners, mtp, kv_format).unwrap();
                assert_eq!(placed.placement, super::super::KvPlacement::PartitionedLayers);
                assert_eq!(placed.ranks.iter().map(|rank| rank.persistent_unit_bytes).sum::<u64>(), whole.ranks[0].persistent_unit_bytes);
                assert_eq!(placed.ranks.iter().map(|rank| rank.active_state_per_sequence_bytes).sum::<u64>(), whole.ranks[0].active_state_per_sequence_bytes);
                assert_eq!(placed.ranks.iter().map(|rank| rank.retained_mark_bytes).sum::<u64>(), whole.ranks[0].retained_mark_bytes);
                assert_eq!(placed.ranks.iter().map(|rank| rank.speculative_replay_bytes).sum::<u64>(), whole.ranks[0].speculative_replay_bytes);
                assert_eq!(placed.ranks[0].fixed_state_bytes, whole.ranks[0].fixed_state_bytes);
                assert_eq!(placed.ranks[1].fixed_state_bytes, 3 * 64 * 4);
                let no_mtp = qwen_cache_geometry_placed(&cfg, &owners, false, kv_format).unwrap();
                assert_eq!(placed.ranks[1], no_mtp.ranks[1]);
            }
        }
    }

    #[test]
    fn qwen_owner_segments_refuse_unsupported_topologies() {
        for owners in [vec![], vec![1], vec![0, 2], vec![0, 1, 0], vec![0, 1, 0, 1]] {
            assert!(qwen_graph_segments(&owners).is_none());
        }
    }
}

/// Partition the authoritative cache geometry by whole-layer owner, while
/// retaining logical global unit ids. Commit tables exist on each owner;
/// the deferred MTP id buffer and MTP pools remain on the head's owner0.
pub fn qwen_cache_geometry_placed(cfg: &crate::families::qwen4::Qwen4Config, owners: &[usize], mtp: bool,
    kv_format: crate::families::qwen4::Qwen4KvCache)
    -> anyhow::Result<super::FamilyCacheGeometry> {
    use super::{KvPlacement, RankCacheGeometry};
    let segments = qwen_graph_segments(owners).ok_or_else(|| anyhow::anyhow!("invalid Qwen cache owners"))?;
    anyhow::ensure!(owners.len() <= cfg.layers, "Qwen cache owner count exceeds the backbone");
    let mut geometry = super::qwen_cache_geometry(cfg, owners.len(), mtp, kv_format)?;
    if segments.len() == 1 { return Ok(geometry); }
    let mut ranks = vec![RankCacheGeometry::default(); 2];
    let mut previous = RankCacheGeometry::default();
    let add = |dst: &mut u64, bytes: u64| -> anyhow::Result<()> {
        *dst = dst.checked_add(bytes).ok_or_else(|| anyhow::anyhow!("Qwen owner cache overflow"))?;
        Ok(())
    };
    for (layer, &owner) in owners.iter().enumerate() {
        let current = super::qwen_cache_geometry(cfg, layer + 1, false, kv_format)?.ranks[0];
        let rank = &mut ranks[owner];
        add(&mut rank.persistent_unit_bytes, current.persistent_unit_bytes - previous.persistent_unit_bytes)?;
        add(&mut rank.active_state_per_sequence_bytes,
            current.active_state_per_sequence_bytes - previous.active_state_per_sequence_bytes)?;
        add(&mut rank.retained_mark_bytes, current.retained_mark_bytes - previous.retained_mark_bytes)?;
        add(&mut rank.speculative_replay_bytes, current.speculative_replay_bytes - previous.speculative_replay_bytes)?;
        previous = current;
    }
    let all = geometry.ranks[0];
    add(&mut ranks[0].persistent_unit_bytes, all.persistent_unit_bytes - previous.persistent_unit_bytes)?;
    add(&mut ranks[0].active_state_per_sequence_bytes,
        all.active_state_per_sequence_bytes - previous.active_state_per_sequence_bytes)?;
    ranks[0].pool_metadata_unit_bytes = all.pool_metadata_unit_bytes;
    ranks[0].fixed_state_bytes = all.fixed_state_bytes;
    ranks[1].fixed_state_bytes = 3 * QWEN_DECODE_ROWS as u64 * 4;
    geometry.placement = KvPlacement::PartitionedLayers;
    geometry.ranks = ranks;
    Ok(geometry)
}

/// The largest pool (whole units, at most `target`) whose startup graph set
/// fits `available` bytes beside `fixed` at `per_token` bytes a token, with
/// that set: descend from the no-graph bound until the pool's own set fits
/// (a smaller pool never needs more graphs). `requested` is checked, not sized.
#[allow(clippy::too_many_arguments)]
pub fn qwen_graph_pool(available: u64, fixed: u64, per_token: u64, target: u64, requested: Option<u64>,
    mut graphs: impl FnMut(u64) -> Option<GraphSet>) -> Result<(u64, GraphSet), String> {
    let unit = QWEN_UNIT_ROWS as u64;
    if per_token == 0 { return Err("Qwen KV admission needs positive token bytes".into()); }
    let rounded = requested.map(|tokens| tokens.div_ceil(unit) * unit);
    let mut tokens = rounded.unwrap_or_else(|| (available.saturating_sub(fixed) / per_token / unit * unit)
        .min(target / unit * unit));
    loop {
        if tokens < unit { return Err("no room for Qwen KV after graph and fixed reserves".into()); }
        let set = graphs(tokens).ok_or("Qwen graph count overflow")?;
        let capacity = available.saturating_sub(fixed.saturating_add(set.bytes(0))) / per_token / unit * unit;
        if tokens <= capacity { return Ok((tokens, set)); }
        if rounded.is_some() {
            return Err(format!("fixed Qwen KV pool of {tokens} tokens does not fit with {} startup graphs ({} graph \
                reserve bytes)", set.ranks[0].executables, set.bytes(0)));
        }
        tokens = capacity;
    }
}

/// Device bytes of one Qwen step workspace (`Qwen4Engine::workspace`) of `t`
/// rows: every buffer rounded up to 256 bytes, the shared program scratch
/// (`scratch`, the largest of the step's programs) and the index top-k
/// scratch, `logit_rows` vocabulary rows and the 4 MiB head workspace. `pages`
/// is the pool's record pages (decode tables hold one row per step row).
#[allow(clippy::too_many_arguments)]
pub fn qwen_workspace_bytes(cfg: &crate::families::qwen4::Qwen4Config, t: u64, decode: bool, logit_rows: u64,
    pages: u64, mapped_ple_row_bytes: Option<u64>, scratch: u64, topk_scratch: u64) -> u64 {
    let a = |bytes: u64| bytes.max(256);
    let (h, hc) = (cfg.hidden as u64, 4u64);
    let table_rows = if decode { t } else { 1 };
    let pool_pages = pages / QWEN_UNIT_PAGES as u64;
    let ple_rows = cfg.ple_rows() as u64;
    let (heads, hd) = (cfg.heads as u64, cfg.head_dim as u64);
    let blocks = (cfg.index_budget / 4) as u64;
    let topk = cfg.topk as u64;
    let mut total = [a(t * hc * h * 2), a(t * hc * h * 2), a(t * hc * 2), a(t * h * 2), a(t * h * 2), a(t * h * 2),
        a(t * h * 2), a(t * 8), a(t * 12), a(t * 12), a(t * 8), a(t * 4), a(t * 4), a(t * 8), a(t * 4),
        a(table_rows * pages * 4), a(table_rows * pool_pages * 4), a(t * ple_rows * 8),
        a(t * heads * hd * 2), a(t * heads * hd * 2), a(t * cfg.index_heads as u64 * 128 * 2), a(t * heads * hd * 2),
        a(t * blocks * 4), a(t * 2112 * 4), a(t * 4), a(scratch), a(topk_scratch),
        a(logit_rows * cfg.vocab_size as u64 * 4), a(t * cfg.experts as u64 * 4), a(t * topk * 4), a(t * topk * 4),
        a(t * (h + h / 32)), a(t * 4), a(logit_rows * 8), a(t * 4), a(logit_rows * 8), 4 << 20].iter().sum::<u64>();
    if let Some(row_bytes) = mapped_ple_row_bytes {
        total += a(t * ple_rows * 8) + a(t * ple_rows * row_bytes);
    }
    total
}

/// The largest scratch among the programs one Qwen step launches at `cap`
/// (`m64` / `m4096`), and the index top-k scratch, from the program manifest.
pub fn qwen_step_scratch(manifest: &serde_json::Value, decode: bool, ple_fp8: bool,
    kv_format: crate::families::qwen4::Qwen4KvCache) -> (u64, u64) {
    let cap = if decode { "m64" } else { "m4096" };
    let suffix = if kv_format == crate::families::qwen4::Qwen4KvCache::Fp8 { "_kv_fp8" } else { "" };
    let scratch = |name: &str| manifest["programs"].as_array().into_iter().flatten()
        .find(|p| p["name"] == name).and_then(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64()).unwrap_or(0);
    let names = ["qwen4_hc_pre".to_string(), "qwen4_hc_post_pre".into(), "qwen4_head".into(), "qwen4_shared".into(),
        if ple_fp8 { "qwen4_ple_fp8" } else { "qwen4_ple_bf16" }.into(), "qwen4_mtp_feedback".into(),
        format!("qwen4_gdn_{cap}"), format!("qwen4_attn_producer{suffix}_{cap}"), format!("qwen4_gdn_w8_{cap}"),
        format!("qwen4_attn_producer_w8{suffix}_{cap}"), format!("qwen4_attn_o_w8_{cap}"), format!("qwen4_sparse_gqa{suffix}_{cap}"),
        format!("qwen4_attn_o_{cap}")];
    (names.iter().map(|n| scratch(n)).max().unwrap_or(0), scratch(&format!("qwen4_index_topk_{cap}")))
}

/// Qwen's KV admission inputs, shared by serve-qwen4 and `cuteafd plan`:
/// pool bytes per token and every byte the coordinator GPU still allocates
/// beside the pool after its weights, PLE table and resident experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenAdmission {
    pub per_token: u64,
    /// (category, group, bytes): step workspaces, state, replay, marks,
    /// probe logits, lazily loaded experts.
    pub items: Vec<(cuteafd_core::memory_layout::Category, &'static str, u64)>,
    /// Kept free: allocator slack and growth (`max(--headroom, 3 GiB)`).
    pub headroom: u64,
}

impl QwenAdmission {
    pub fn fixed(&self) -> u64 { self.items.iter().map(|i| i.2).sum::<u64>() + self.headroom }
}

pub struct QwenAdmissionInputs<'a> {
    pub cfg: &'a crate::families::qwen4::Qwen4Config,
    pub layers: usize,
    pub mtp: bool,
    pub kv_format: crate::families::qwen4::Qwen4KvCache,
    pub manifest: Option<&'a serde_json::Value>,
    pub prefill_rows: u64,
    pub slots: u64,
    pub mark_bytes: u64,
    pub full_prefill_logits: u64,
    /// Mapped PLE table: row bytes and whether rows are FP8.
    pub ple: Option<(u64, bool)>,
    /// Lazily loaded EXL3 expert window (peak), 0 when experts are resident.
    pub future_expert_bytes: u64,
    pub headroom: u64,
}

pub fn qwen_admission(inputs: &QwenAdmissionInputs<'_>) -> Result<QwenAdmission, super::CacheGeometryError> {
    use cuteafd_core::memory_layout::Category;
    let geometry = super::qwen_cache_geometry(inputs.cfg, inputs.layers, inputs.mtp, inputs.kv_format)?;
    let rank = &geometry.ranks[0];
    let unit = geometry.logical_unit_rows;
    // Prefill owns one page table and decode one per row, each with four
    // record page ids plus one pool page id per allocation unit.
    let tables_per_unit = (1 + QWEN_DECODE_ROWS as u64) * 5 * 4;
    let per_token = (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes + tables_per_unit).div_ceil(unit);
    let ple_fp8 = inputs.ple.is_some_and(|p| p.1);
    let mapped = inputs.ple.map(|p| p.0);
    let workspace = |decode: bool| {
        let (scratch, topk) = inputs.manifest.map_or((0, 0), |m| qwen_step_scratch(m, decode, ple_fp8, inputs.kv_format));
        let (t, logits) = if decode { (QWEN_DECODE_ROWS as u64, QWEN_DECODE_ROWS as u64) } else { (inputs.prefill_rows.max(1), 1) };
        // Page tables follow the pool (per_token above).
        qwen_workspace_bytes(inputs.cfg, t, decode, logits, 0, mapped, scratch, topk)
    };
    let items = vec![
        (Category::Workspace, "decode step", workspace(true)),
        (Category::Workspace, "prefill step", workspace(false)),
        (Category::Workspace, "probe prefill logits", inputs.full_prefill_logits),
        (Category::Kv, "state", rank.active_state_per_sequence_bytes * inputs.slots + rank.fixed_state_bytes
            + rank.speculative_replay_bytes),
        (Category::Prefix, "marks", inputs.mark_bytes),
        (Category::Experts, "lazy EXL3 window", inputs.future_expert_bytes),
    ].into_iter().filter(|i| i.2 > 0).collect();
    Ok(QwenAdmission { per_token, items, headroom: inputs.headroom })
}

#[cfg(test)]
mod tests {
    use super::qwen_step_scratch;
    use crate::families::qwen4::Qwen4KvCache;

    #[test]
    fn scratch_admission_selects_the_kv_format_and_capacity() {
        let mut programs = Vec::new();
        for (cap, factor) in [("m64", 1_u64), ("m4096", 10)] {
            for (stem, bytes) in [("attn_producer", 300), ("attn_producer_w8", 400),
                ("sparse_gqa", 500), ("attn_producer_kv_fp8", 600),
                ("attn_producer_w8_kv_fp8", 700), ("sparse_gqa_kv_fp8", 800),
                ("index_topk", 200)] {
                programs.push(serde_json::json!({"name": format!("qwen4_{stem}_{cap}"),
                    "scratch_bytes_at_capacity": {"scratch": bytes * factor}}));
            }
        }
        let manifest = serde_json::json!({"programs": programs});
        for (decode, factor) in [(true, 1), (false, 10)] {
            assert_eq!(qwen_step_scratch(&manifest, decode, false, Qwen4KvCache::Bf16),
                (500 * factor, 200 * factor));
            assert_eq!(qwen_step_scratch(&manifest, decode, false, Qwen4KvCache::Fp8),
                (800 * factor, 200 * factor));
        }
    }
}
