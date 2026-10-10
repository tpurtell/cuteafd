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
/// 2026-10-07 TP4 SM120: 1,855,979,520 physical bytes / 12,397 graphs,
/// rounded up; the margin also covers the earlier 21,560-graph measurement.
pub const QWEN_GRAPH_BYTES_PER_GRAPH: u64 = 149_712;
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
pub fn qwen_step_scratch(manifest: &serde_json::Value, decode: bool, ple_fp8: bool) -> (u64, u64) {
    let cap = if decode { "m64" } else { "m4096" };
    let scratch = |name: &str| manifest["programs"].as_array().into_iter().flatten()
        .find(|p| p["name"] == name).and_then(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64()).unwrap_or(0);
    let names = ["qwen4_hc_pre".to_string(), "qwen4_hc_post_pre".into(), "qwen4_head".into(), "qwen4_shared".into(),
        if ple_fp8 { "qwen4_ple_fp8" } else { "qwen4_ple_bf16" }.into(), "qwen4_mtp_feedback".into(),
        format!("qwen4_gdn_{cap}"), format!("qwen4_attn_producer_{cap}"), format!("qwen4_gdn_w8_{cap}"),
        format!("qwen4_attn_producer_w8_{cap}"), format!("qwen4_attn_o_w8_{cap}"), format!("qwen4_sparse_gqa_{cap}"),
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
    let geometry = super::qwen_cache_geometry(inputs.cfg, inputs.layers, inputs.mtp)?;
    let rank = &geometry.ranks[0];
    let unit = geometry.logical_unit_rows;
    // Prefill owns one page table and decode one per row, each with four
    // record page ids plus one pool page id per allocation unit.
    let tables_per_unit = (1 + QWEN_DECODE_ROWS as u64) * 5 * 4;
    let per_token = (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes + tables_per_unit).div_ceil(unit);
    let ple_fp8 = inputs.ple.is_some_and(|p| p.1);
    let mapped = inputs.ple.map(|p| p.0);
    let workspace = |decode: bool| {
        let (scratch, topk) = inputs.manifest.map_or((0, 0), |m| qwen_step_scratch(m, decode, ple_fp8));
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
