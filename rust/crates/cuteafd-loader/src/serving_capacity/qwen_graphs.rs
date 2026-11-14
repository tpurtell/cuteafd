//! Qwen 3.8 Flash Next's decode graph shapes and bootstrap prewarm: the one
//! definition the engine's warm-up captures and admission (runtime and
//! planner) reserves. Each shape is a (rows, speculative, table geometry)
//! step; every shape is captured once per layer segment plus the head.
//!
//! The checked-in list in `qwen_prewarm.json` is the provisional prewarm and
//! reserve until a `GRAPH_TRACE` at each card layout replaces it; the runtime
//! filters it against the shapes this launch may actually capture.
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Rows of the decode-shaped programs (`_m64`).
pub const QWEN_DECODE_ROWS: usize = 64;
/// Plain decode row buckets.
pub const QWEN_PLAIN_BUCKETS: &[usize] = &[1, 4, 8, 16, 24, 32, 64];
/// Speculative verify row buckets.
pub const QWEN_SPEC_BUCKETS: &[usize] = &[2, 4, 8, 16, 24, 32, 64];
/// Tokens per allocation unit (four 64-row record pages) and pages per unit.
pub const QWEN_UNIT_ROWS: usize = 256;
pub const QWEN_UNIT_PAGES: usize = 4;
/// Measured driver bytes of one startup graph executable, rounded up: the
/// 1-RTX min config (`qwen38-exl3-min`, EXL3 K4.25 PLE FP8, MTP depth 3)
/// captured 33,418 executables (682 shapes x 49 segments) for 5,102,370,816 B
/// on SM120 PRO, 2026-11-08 — 152,683.31 B each, charged as the next byte. The
/// capture measures allocatable memory before and after the whole loop, so it
/// absorbs the modules and stream slabs that capturing the set creates
/// (pinned-host page tables are charged as their own owner); multiplying it by
/// the exact executable count is the whole charge, with no margin.
pub const QWEN_GRAPH_BYTES_PER_GRAPH: u64 = 152_684;

/// One captured decode step: its row bucket, speculation, and table geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QwenGraphShape {
    pub rows: usize,
    pub spec: bool,
    pub long: bool,
    pub pool_width: usize,
    pub page_stride: usize,
    pub pool_stride: usize,
}

impl QwenGraphShape {
    /// Full-attention capture geometry; GDN and head keys normalize further.
    pub fn capture_shape(self) -> Self {
        Self { pool_width: 0, pool_stride: if self.long { self.pool_stride } else { 0 }, ..self }
    }

    pub fn geometry(&self) -> QwenGraphGeometry {
        QwenGraphGeometry { pool_width: self.pool_width, page_stride: self.page_stride, pool_stride: self.pool_stride, long: self.long }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenPrewarmEntry {
    #[serde(flatten)]
    pub shape: QwenGraphShape,
    /// Hold this shape resident across eviction pressure (trace-chosen).
    pub pin: bool,
}

/// The resolved capture shape: how many devices own layers, and whether Sparks
/// are attached. A 0-Spark card captures no onboarding graphs and a TP1 card
/// captures no TP2 peer segments, so the four shapes capture different sets;
/// each has its own prewarm and reserve, and an untraced shape uses the
/// bootstrap set rather than borrowing another shape's list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QwenGraphShapeKey {
    #[serde(rename = "single_0spark")] Single0Spark,
    #[serde(rename = "single_sparks")] SingleSparks,
    #[serde(rename = "dual_0spark")] Dual0Spark,
    #[serde(rename = "dual_sparks")] DualSparks,
}

impl QwenGraphShapeKey {
    pub fn resolve(owners: usize, sparks: bool) -> Self {
        match (owners > 1, sparks) {
            (false, false) => Self::Single0Spark,
            (false, true) => Self::SingleSparks,
            (true, false) => Self::Dual0Spark,
            (true, true) => Self::DualSparks,
        }
    }
    /// Layer-owning devices this shape captures on.
    pub fn owners(self) -> usize { if self.is_dual() { 2 } else { 1 } }
    pub fn sparks(self) -> bool { matches!(self, Self::SingleSparks | Self::DualSparks) }
    fn is_dual(self) -> bool { matches!(self, Self::Dual0Spark | Self::DualSparks) }
    fn index(self) -> usize {
        match self { Self::Single0Spark => 0, Self::SingleSparks => 1, Self::Dual0Spark => 2, Self::DualSparks => 3 }
    }
}

/// A `GRAPH_TRACE` run's captured set for one shape.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenTrace {
    /// Host, card, model and date the trace ran.
    pub provenance: String,
    pub entries: Vec<QwenPrewarmEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QwenShapePrewarm {
    pub shape: QwenGraphShapeKey,
    /// Per-device prewarm arena: the selected shapes' measured capture bytes,
    /// summed and rounded once to the 2 MiB chunk. Never the trace's charged
    /// high-water.
    pub reserve_base_bytes: Vec<u64>,
    /// Per-device capture headroom, a fixed 2 MiB; the shared graph reserve
    /// extra adds room for runtime captures.
    pub reserve_extra_bytes: Vec<u64>,
    /// The shape's own trace; `None` until one runs, which falls back to the
    /// bootstrap startup set with a warning.
    pub trace: Option<QwenTrace>,
}

#[derive(Debug, Deserialize)]
pub struct QwenBootstrap {
    pub entries: Vec<QwenPrewarmEntry>,
}

#[derive(Debug, Deserialize)]
pub struct QwenPrewarmList {
    pub schema_version: u32,
    pub family: String,
    /// Where the reserves and entries came from (measurement or trace).
    pub provenance: String,
    /// Every shape's provisional prewarm before its own trace.
    pub bootstrap: QwenBootstrap,
    pub shapes: Vec<QwenShapePrewarm>,
}

pub fn qwen_prewarm_list() -> &'static QwenPrewarmList {
    static LIST: OnceLock<QwenPrewarmList> = OnceLock::new();
    LIST.get_or_init(|| {
        let list: QwenPrewarmList = serde_json::from_str(include_str!("qwen_prewarm.json"))
            .expect("qwen_prewarm.json is well formed");
        assert_eq!(list.schema_version, 1, "qwen prewarm schema");
        assert_eq!(list.family, "qwen4", "qwen prewarm family");
        assert!(!list.bootstrap.entries.is_empty(), "qwen prewarm bootstrap entries");
        for key in [QwenGraphShapeKey::Single0Spark, QwenGraphShapeKey::SingleSparks,
            QwenGraphShapeKey::Dual0Spark, QwenGraphShapeKey::DualSparks] {
            let rows = list.shapes.iter().filter(|prewarm| prewarm.shape == key).count();
            assert_eq!(rows, 1, "qwen prewarm shape {key:?} appears once");
        }
        for prewarm in &list.shapes {
            assert_eq!(prewarm.reserve_base_bytes.len(), prewarm.shape.owners(), "qwen prewarm reserve ranks");
            assert_eq!(prewarm.reserve_extra_bytes.len(), prewarm.shape.owners(), "qwen prewarm reserve ranks");
        }
        list
    })
}

pub fn qwen_shape_prewarm(key: QwenGraphShapeKey) -> &'static QwenShapePrewarm {
    qwen_prewarm_list().shapes.iter().find(|prewarm| prewarm.shape == key)
        .expect("qwen prewarm shape")
}

/// The reserve of `rank` on `key`: base + extra capture headroom.
pub fn qwen_reserve(key: QwenGraphShapeKey, rank: usize) -> Option<(u64, u64)> {
    let prewarm = qwen_shape_prewarm(key);
    Some((*prewarm.reserve_base_bytes.get(rank)?, *prewarm.reserve_extra_bytes.get(rank)?))
}

/// The prewarm of `key`, filtered to the shapes this launch can capture: an
/// entry whose geometry or bucket is outside the allowed set is dropped (a
/// narrower context prewarms a subset; a wider one captures the rest lazily).
/// An untraced shape uses the bootstrap startup set and says so once.
pub fn qwen_prewarm_shapes(key: QwenGraphShapeKey, context: usize, pages: usize, dense: usize,
    sequences: usize, speculation: bool) -> Vec<QwenGraphShape> {
    use std::sync::Once;
    static WARNED: [Once; 4] = [Once::new(), Once::new(), Once::new(), Once::new()];
    let prewarm = qwen_shape_prewarm(key);
    if prewarm.trace.is_none() {
        WARNED[key.index()].call_once(|| {
            tracing::warn!(shape = ?key,
                "qwen4 has no graph trace for this shape; prewarming the bootstrap startup set");
        });
    }
    let entries = prewarm.trace.as_ref().map_or(&qwen_prewarm_list().bootstrap.entries, |trace| &trace.entries);
    let allowed: std::collections::HashSet<_> = qwen_serving_graph_shapes(context, pages, dense, sequences, speculation)
        .into_iter().map(|(rows, spec, geometry)| QwenGraphShape { rows, spec,
            long: geometry.long, pool_width: geometry.pool_width, page_stride: geometry.page_stride,
            pool_stride: geometry.pool_stride }.capture_shape()).collect();
    let pins = qwen_prewarm_pinned(key);
    let mut seen = std::collections::HashSet::new();
    let mut shapes: Vec<_> = entries.iter().map(|entry| entry.shape.capture_shape())
        .filter(|shape| allowed.contains(shape) && seen.insert(*shape)).collect();
    // A pinned old geometry pins its capture-equivalent normalized identity.
    shapes.sort_by_key(|shape| !pins.contains(shape));
    shapes
}

/// The shapes `key` prewarms that pin their cache entry.
pub fn qwen_prewarm_pinned(key: QwenGraphShapeKey) -> Vec<QwenGraphShape> {
    let prewarm = qwen_shape_prewarm(key);
    let entries = prewarm.trace.as_ref().map_or(&qwen_prewarm_list().bootstrap.entries, |trace| &trace.entries);
    let mut seen = std::collections::HashSet::new();
    entries.iter().filter(|entry| entry.pin).map(|entry| entry.shape.capture_shape())
        .filter(|shape| seen.insert(*shape)).collect()
}


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

/// Every serving shape: each geometry x the plain buckets up to `sequences`
/// and, with speculation, every verify bucket.
pub fn qwen_serving_graph_shapes(context: usize, pages: usize, dense: usize, sequences: usize, speculation: bool)
    -> Vec<(usize, bool, QwenGraphGeometry)> {
    let plain = qwen_decode_bucket(sequences, false);
    qwen_graph_geometries(context, pages, dense).into_iter().flat_map(|geometry| {
        QWEN_PLAIN_BUCKETS.iter().copied().filter(move |&rows| rows <= plain).map(move |rows| (rows, false, geometry))
            .chain(QWEN_SPEC_BUCKETS.iter().copied().filter(move |_| speculation).map(move |rows| (rows, true, geometry)))
    }).collect()
}

/// Record pages of a pool of `pool_tokens` tokens.
pub fn qwen_pool_pages(pool_tokens: usize) -> Option<usize> {
    pool_tokens.div_ceil(QWEN_UNIT_ROWS).checked_mul(QWEN_UNIT_PAGES)
}

/// Owner-local layer fronts and owner0 head. Cutover and exit posts run eagerly;
/// each physical rank owns only its layer fronts, not a duplicated whole set.
pub fn qwen_graph_segments(owners: &[usize]) -> Option<Vec<u64>> {
    if owners.is_empty() || owners[0] != 0 || owners.iter().any(|&owner| owner > 1) { return None; }
    let dual = owners.contains(&1);
    if dual && (owners.last() != Some(&1)
        || owners.windows(2).filter(|pair| pair[0] != pair[1]).count() != 1) { return None; }
    let mut counts = vec![0u64; if dual { 2 } else { 1 }];
    for &owner in owners { counts[owner] = counts[owner].checked_add(1)?; }
    counts[0] = counts[0].checked_add(1)?;
    Some(counts)
}

#[cfg(test)]
mod owner_graph_tests {
    use super::*;

    #[test]
    fn qwen_owner_segments_match_cutover_and_exit() {
        assert_eq!(qwen_graph_segments(&[0; 48]), Some(vec![49]));
        for cut in [1, 24, 47] {
            let owners: Vec<_> = (0..48).map(|layer| usize::from(layer >= cut)).collect();
            let segments = qwen_graph_segments(&owners).unwrap();
            assert_eq!(segments, vec![cut as u64 + 1, (48 - cut) as u64]);
            assert_eq!(segments.iter().sum::<u64>(), 49);
        }
    }

    #[test]
    fn qwen_prewarm_is_the_measured_startup_set() {
        use super::*;
        let shapes = qwen_serving_graph_shapes(262_144, 8192, 2051, 16, true);
        assert_eq!(shapes.len(), 682);
        let entries = &qwen_prewarm_list().bootstrap.entries;
        assert_eq!(entries.len(), shapes.len());
        for (entry, shape) in entries.iter().zip(&shapes) {
            assert_eq!((entry.shape.rows, entry.shape.spec, entry.shape.geometry()), *shape);
            assert!(!entry.pin, "the bootstrap prewarm pins nothing");
        }
        for key in [QwenGraphShapeKey::SingleSparks, QwenGraphShapeKey::DualSparks] {
            // Untraced: every shape falls back to the bootstrap set, never another shape's list.
            assert!(qwen_shape_prewarm(key).trace.is_none(), "{key:?}");
            assert_eq!(qwen_prewarm_shapes(key, 262_144, 8192, 2051, 16, true).len(), 187, "{key:?}");
            assert_eq!(qwen_prewarm_shapes(key, 8192, 8192, 2051, 16, true).len(), 88, "{key:?}");
            assert!(qwen_prewarm_shapes(key, 8192, 8192, 2051, 1, false).len() < 253, "{key:?}");
            assert!(qwen_prewarm_pinned(key).is_empty(), "{key:?}");
        }
        // Traced shapes install their own measured list; every entry must stay
        // reachable at the card it was traced on (the trace cards' pool pages),
        // so prewarm never silently drops a measured shape.
        for (key, pages) in [(QwenGraphShapeKey::Single0Spark, 10_744),
            (QwenGraphShapeKey::Dual0Spark, 32_768)] {
            let prewarm = qwen_shape_prewarm(key);
            let trace = prewarm.trace.as_ref().expect("traced shape");
            let retained = qwen_prewarm_shapes(key, 262_144, pages, 2051, 16, true);
            assert_eq!(retained.len(), if key == QwenGraphShapeKey::Single0Spark { 13 } else { 15 });
            for entry in &trace.entries {
                assert!(retained.contains(&entry.shape.capture_shape()), "{key:?}: {entry:?}");
            }
            assert!(qwen_prewarm_pinned(key).is_empty(), "{key:?}");
        }
        assert_eq!(QwenGraphShapeKey::resolve(1, false), QwenGraphShapeKey::Single0Spark);
        assert_eq!(QwenGraphShapeKey::resolve(1, true), QwenGraphShapeKey::SingleSparks);
        assert_eq!(QwenGraphShapeKey::resolve(2, false), QwenGraphShapeKey::Dual0Spark);
        assert_eq!(QwenGraphShapeKey::resolve(2, true), QwenGraphShapeKey::DualSparks);
        for (key, rank0, rank1) in [
            (QwenGraphShapeKey::Single0Spark, (109_051_904, 2_097_152), None),
            (QwenGraphShapeKey::SingleSparks, (5_102_370_816, 2_097_152), None),
            (QwenGraphShapeKey::Dual0Spark, (67_108_864, 2_097_152), Some((73_400_320, 2_097_152))),
            (QwenGraphShapeKey::DualSparks, (2_707_423_232, 2_097_152), Some((2_604_662_784, 2_097_152))),
        ] {
            assert_eq!(qwen_reserve(key, 0), Some(rank0), "{key:?}");
            assert_eq!(qwen_reserve(key, 1), rank1, "{key:?}");
            assert_eq!(qwen_reserve(key, 2), None, "{key:?}");
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
    add(&mut ranks[0].retained_mark_bytes, all.retained_mark_bytes - previous.retained_mark_bytes)?;
    ranks[0].pool_metadata_unit_bytes = all.pool_metadata_unit_bytes;
    ranks[0].fixed_state_bytes = all.fixed_state_bytes;
    ranks[1].fixed_state_bytes = 3 * QWEN_DECODE_ROWS as u64 * 4;
    geometry.placement = KvPlacement::PartitionedLayers;
    geometry.ranks = ranks;
    Ok(geometry)
}

/// Device bytes of one Qwen step workspace (`Qwen4Engine::workspace`) of `t`
/// rows: every buffer has a 256-byte allocation minimum, the shared program scratch
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
    // Packed attention uses a zero-origin member view; PLE retains global origins.
    if !decode { total += a(t * 4); }
    if let Some(row_bytes) = mapped_ple_row_bytes {
        total += a(t * ple_rows * 8) + a(t * ple_rows * row_bytes);
    }
    total
}

/// Unit-owned live rows plus hook-local scratch; retained terminal rows feed
/// the canonical MTP without keeping a whole span of every intermediate plane.
pub fn qwen_unit_workspace_bytes(cfg: &crate::families::qwen4::Qwen4Config, shape: super::PrefillShape,
    terminal_rows: u64, logit_rows: u64, mapped: Option<u64>, scratch: u64, topk: u64) -> u64 {
    let unit = shape.unit_rows;
    let h = cfg.hidden as u64;
    let slot_bytes = |rows: u64| 2 * rows * 4 * h * 2 + rows * 4 * 2 + 3 * rows * h * 2;
    qwen_workspace_bytes(cfg, unit, false, logit_rows, 0, mapped, scratch, topk)
        - slot_bytes(unit) + slot_bytes(shape.slot_rows()) + terminal_rows * 4 * h * 2
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
    pub prefill_shape: Option<super::PrefillShape>,
    pub slots: u64,
    pub mark_bytes: u64,
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
    let separate_mtp = inputs.prefill_shape.is_some() && inputs.mtp;
    let tables_per_unit = (1 + QWEN_DECODE_ROWS as u64 + u64::from(separate_mtp)) * 5 * 4;
    let per_token = (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes + tables_per_unit).div_ceil(unit);
    let ple_fp8 = inputs.ple.is_some_and(|p| p.1);
    let mapped = inputs.ple.map(|p| p.0);
    let workspace = |decode: bool| {
        let (scratch, topk) = inputs.manifest.map_or((0, 0), |m| qwen_step_scratch(m, decode, ple_fp8, inputs.kv_format));
        let (t, logits) = if decode { (QWEN_DECODE_ROWS as u64, QWEN_DECODE_ROWS as u64) } else { (inputs.prefill_rows.max(1), 1) };
        // Page tables follow the pool (per_token above).
        if !decode {
            if let Some(shape) = inputs.prefill_shape {
                let terminal = if inputs.mtp { inputs.prefill_rows } else { 0 };
                return qwen_unit_workspace_bytes(inputs.cfg, shape, terminal, logits, mapped, scratch, topk);
            }
        }
        qwen_workspace_bytes(inputs.cfg, t, decode, logits, 0, mapped, scratch, topk)
    };
    let mtp_workspace = if separate_mtp {
        let (scratch, topk) = inputs.manifest.map_or((0, 0), |m| qwen_step_scratch(m, false, ple_fp8, inputs.kv_format));
        qwen_workspace_bytes(inputs.cfg, inputs.prefill_shape.unwrap().unit_rows, false, 1, 0, mapped, scratch, topk)
    } else { 0 };
    let items = vec![
        (Category::Workspace, "decode step", workspace(true)),
        (Category::Workspace, "prefill step", workspace(false)),
        (Category::Workspace, "prefill MTP step", mtp_workspace),
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
    fn unit_workspace_scales_only_live_planes_and_retained_terminal_rows() {
        use super::{qwen_unit_workspace_bytes, qwen_workspace_bytes};
        use crate::serving_capacity::PrefillShape;
        let cfg = crate::families::qwen4::Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        let h = cfg.hidden as u64;
        for rows in [512, 1024, 2048, 2752, 4096] {
            let base = qwen_workspace_bytes(&cfg, rows, false, 1, 0, Some(160), 12345, 6789);
            let extra_slot = rows * (2 * 4 * h * 2 + 4 * 2 + 3 * h * 2);
            for waves in 1..=4 {
                let shape = PrefillShape { unit_rows: rows, waves }.validate().unwrap();
                let slot_bytes = qwen_unit_workspace_bytes(&cfg, shape, 0, 1, Some(160), 12345, 6789);
                assert_eq!(slot_bytes, base + (waves - 1) * extra_slot);
                for terminal in [4096, shape.slot_rows(), 16384] {
                    assert_eq!(qwen_unit_workspace_bytes(&cfg, shape, terminal, 1, Some(160), 12345, 6789),
                        slot_bytes + terminal * 4 * h * 2);
                }
            }
        }
        assert_eq!(crate::serving_capacity::prefill_return_bytes(2, 2048, 4, 2048), 4
            * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(2 * 2048 * 2048 * 2).unwrap());
    }

    #[test]
    fn full_prefill_logits_charge_matches_workspace_minimum_allocations() {
        use super::qwen_workspace_bytes;
        let cfg = crate::families::qwen4::Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        let baseline = qwen_workspace_bytes(&cfg, 2048, false, 1, 0, Some(160), 0, 0);
        for logits in [1, 16, 32, 33, 2048, 2752, 8192, 8256, 16384] {
            let runtime = qwen_workspace_bytes(&cfg, 2048, false, logits, 0, Some(160), 0, 0) - baseline;
            assert_eq!(crate::plan::layout::full_prefill_logits_bytes("qwen4", logits, cfg.vocab_size as u64),
                runtime, "{logits} logit rows");
        }
    }

    #[test]
    fn unit_mtp_scratch_and_page_tables_are_admitted_separately() {
        use super::{qwen_admission, qwen_workspace_bytes, QwenAdmissionInputs};
        use crate::serving_capacity::PrefillShape;
        let mut config = crate::plan::testing::qwen4_config(48);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        let cfg = crate::families::qwen4::Qwen4Config::from_hf(&config).unwrap();
        let mut inputs = QwenAdmissionInputs { cfg: &cfg, layers: 48, mtp: true, kv_format: Qwen4KvCache::Fp8,
            manifest: None, prefill_rows: 4096, prefill_shape: None, slots: 8, mark_bytes: 0,
             ple: Some((160, true)), future_expert_bytes: 0, headroom: 0 };
        let legacy = qwen_admission(&inputs).unwrap();
        inputs.prefill_shape = Some(PrefillShape { unit_rows: 2048, waves: 2 });
        let units = qwen_admission(&inputs).unwrap();
        let bytes = |name| units.items.iter().find(|item| item.1 == name).unwrap().2;
        assert_eq!(bytes("prefill MTP step"), qwen_workspace_bytes(&cfg, 2048, false, 1, 0, Some(160), 0, 0));
        let geometry = crate::serving_capacity::qwen_cache_geometry(&cfg, 48, true, Qwen4KvCache::Fp8).unwrap();
        let rank = &geometry.ranks[0];
        let pool_bytes = rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes;
        assert_eq!(legacy.per_token, (pool_bytes + 65 * 5 * 4).div_ceil(geometry.logical_unit_rows));
        assert_eq!(units.per_token, (pool_bytes + 66 * 5 * 4).div_ceil(geometry.logical_unit_rows));
        inputs.mtp = false;
        let no_mtp = qwen_admission(&inputs).unwrap();
        assert!(!no_mtp.items.iter().any(|item| item.1 == "prefill MTP step"));
        assert_eq!(bytes("prefill step") - no_mtp.items.iter().find(|item| item.1 == "prefill step").unwrap().2,
            4096 * 4 * cfg.hidden as u64 * 2);
    }

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
