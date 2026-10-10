//! GLM Flash startup decode graph inventory, shared by capture and CPU admission.
use super::glmf_table_pages;
pub const PAGE_ROWS: usize = 64;
pub const UNIT_PAGES: usize = 4;
pub const UNIT_ROWS: usize = PAGE_ROWS * UNIT_PAGES;
pub const DECODE_ROWS: usize = 64;
pub const PLAIN_DECODE_BUCKETS: [usize; 6] = [1, 4, 8, 16, 32, 64];
pub const SPEC_DECODE_BUCKETS: [usize; 6] = [2, 4, 8, 16, 32, 64];
// SM120 measured bytes per executable, plus admission margins (WP9, 2026-10-06).
pub const MEASURED_GRAPH_BYTES: u64 = 146_459;
pub const GRAPH_MARGIN_PERCENT: u64 = 20;
pub const GRAPH_RANK_MARGIN_BYTES: u64 = 64 << 20;
// WP9 SM120 measurements; retained until a selected-program ledger separates cuBLAS ownership.
pub const WORKSPACE_RUNTIME_OVERHEAD_BYTES: u64 = 72 << 20;
pub fn workspace_runtime_overhead(lanes: usize, drafter: bool) -> u64 {
    WORKSPACE_RUNTIME_OVERHEAD_BYTES * (1 + lanes + usize::from(drafter)) as u64
}

pub fn verify_budget(decode_rows: usize, sms: usize) -> usize {
    if decode_rows <= DECODE_ROWS { decode_rows } else { (3 * sms / 4).clamp(DECODE_ROWS, decode_rows) }
}

/// The row buckets padded decode steps run at (startup graphs): the plain buckets (a plain step has
/// one row per sequence, at most 64) and the speculative ones, which with the wide programs end at
/// the verify budget: `[2, 4, 8, 16, 32, 64, 127]` on an RTX 5090, `[.., 64, 128]` on 188 SMs,
/// `[.., 64, 99]` on 132. A budget of 64 keeps the 64-row sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeBuckets {
    pub plain: Vec<usize>,
    pub spec: Vec<usize>,
}

impl DecodeBuckets {
    /// The sets of a verify budget of `verify_rows` rows ([`verify_budget`]).
    pub fn new(verify_rows: usize) -> Self {
        let mut spec = SPEC_DECODE_BUCKETS.to_vec();
        if verify_rows > DECODE_ROWS {
            spec.push(verify_rows);
        }
        Self { plain: PLAIN_DECODE_BUCKETS.to_vec(), spec }
    }

    /// The bucket a step of `rows` rows pads to (the rows themselves past the largest).
    pub fn bucket(&self, rows: usize, spec: bool) -> usize {
        let set = if spec { &self.spec } else { &self.plain };
        set.iter().copied().find(|&bucket| bucket >= rows).unwrap_or(rows)
    }
}

pub fn graph_reserve_bytes(graphs: usize) -> u64 {
    let measured = graphs as u64 * MEASURED_GRAPH_BYTES;
    measured + (measured * GRAPH_MARGIN_PERCENT).div_ceil(100) + GRAPH_RANK_MARGIN_BYTES
}

/// The startup decode graph set's bytes per rank: `buckets` (the engine's sets) over every geometry.
#[allow(clippy::too_many_arguments)]
pub fn serving_graph_reserve(context: usize, pool_tokens: usize, dense: usize,
    sequences: usize, speculation: bool, layers: usize, peer: bool, buckets: &DecodeBuckets) -> Vec<u64> {
    let shapes = serving_graph_shapes(context, pool_tokens.div_ceil(PAGE_ROWS), dense, sequences, speculation,
        buckets).len();
    let graphs = std::iter::once(shapes * (layers + 1))
        .chain(peer.then_some(shapes * layers)).collect::<Vec<_>>();
    let bytes: Vec<_> = graphs.iter().map(|&count| graph_reserve_bytes(count)).collect();
    bytes
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphGeometry {
    pub pool_width: usize,
    pub page_stride: usize,
    pub pool_stride: usize,
    pub long: bool,
}

impl GraphGeometry {
    /// The table geometry a decode step's graphs are keyed by. The pool top-k (`index_topk`, which
    /// runs only when a row is long) is the only launch that reads the pool table's width and
    /// stride, so a short step keys neither: short steps that differ only there launch the same
    /// programs with the same pointers and scalars, and share their graphs.
    pub fn keyed(pool_width: usize, page_stride: usize, pool_stride: usize, long: bool) -> Self {
        if long { Self { pool_width, page_stride, pool_stride, long } }
        else { Self { pool_width: 0, page_stride, pool_stride: 0, long } }
    }
}

/// The narrowest decode page-table stride, in MLA pages (4,096 tokens): sequences up to that size
/// share one table shape, and so their decode graphs. The index expansion reads only a row's own
/// pages, so a wider row changes nothing but its upload. A long step (a row past the 2,051-token
/// dense context) already has at least 36 pages, so the floor never moves its stride.
pub const MIN_PAGE_STRIDE: usize = 64;

/// A decode step's (page-table, pool-table) strides for sequences of at most `pages` and
/// `pool_pages` pages: powers of two (they bound the graphs a growing batch captures), the page
/// stride from its floor, both at most the pool's pages and a table row's columns (a sequence
/// holds at most `max_context` tokens' pages). Only a long step keys the pool stride.
pub fn decode_strides(pages: usize, pool_pages: usize, pool: (usize, usize), table: (usize, usize)) -> (usize, usize) {
    (pages.max(1).next_power_of_two().max(MIN_PAGE_STRIDE).min(pool.0).min(table.0),
        pool_pages.max(1).next_power_of_two().min(pool.1).min(table.1))
}

pub fn graph_geometries(context: usize, pages: usize, dense: usize) -> Vec<GraphGeometry> {
    let pools = pages / UNIT_PAGES;
    let (table_pages, table_pools) = glmf_table_pages(context as u64);
    let table = (table_pages as usize, table_pools as usize);
    let mut geometries = Vec::new();
    for units in 1..=context.div_ceil(UNIT_ROWS).min(pools) {
        let capacity = (units * UNIT_ROWS).min(context);
        let mut width = 1;
        while width / 2 * UNIT_ROWS < capacity {
            let low = if width == 1 { 1 } else { width / 2 * UNIT_ROWS + 1 };
            let high = (width * UNIT_ROWS).min(capacity);
            for long in [false, true] {
                if (!long && low <= high.min(dense)) || (long && low.max(dense + 1) <= high) {
                    let live_units = units.min(high.max(131_072).div_ceil(UNIT_ROWS).next_power_of_two());
                    let (page_stride, pool_stride) = decode_strides(
                        live_units * UNIT_PAGES, live_units, (pages, pools), table);
                    let geometry = GraphGeometry::keyed(width.min(pool_stride), page_stride, pool_stride, long);
                    if !geometries.contains(&geometry) { geometries.push(geometry); }
                }
            }
            width *= 2;
        }
    }
    geometries
}

pub fn serving_graph_shapes(context: usize, pages: usize, dense: usize, sequences: usize, speculation: bool,
    buckets: &DecodeBuckets) -> Vec<(usize, bool, GraphGeometry)> {
    let plain = buckets.bucket(sequences.clamp(16, DECODE_ROWS), false);
    graph_geometries(context, pages, dense).into_iter().flat_map(|geometry| {
        buckets.plain.iter().copied().filter(move |&rows| rows <= plain)
            .map(move |rows| (rows, false, geometry))
            .chain(buckets.spec.iter().copied().filter(move |_| speculation)
                .map(move |rows| (rows, true, geometry)))
    }).collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn graph_solver_shrinks_without_oscillation_and_preserves_lazy_fallback() {
        let mut seen = Vec::new();
        let answer = solve_graph_pool(2048, |pool| -> Result<_, ()> {
            seen.push(pool);
            Ok((if pool > 1024 { 768 } else { 1280 }, true))
        }).unwrap();
        assert_eq!(answer, (768, true));
        assert_eq!(seen, [2048, 768]);
        assert_eq!(solve_graph_pool(2048, |_| -> Result<_, ()> { Ok((512, false)) }).unwrap(), (512, false));
        assert_eq!(admitted_graph_context(2048, 1024, true), 768);
        assert_eq!(admitted_graph_context(2048, 1024, false), 2048);
    }

    #[test]
    fn graph_admission_retries_only_memory_refusals() {
        let startup = Some(StartupGraphReserve { reserve: 200, allowance: 100 });
        let mut calls = Vec::new();
        assert_eq!(admit_beside_decode_graphs(startup, |graphs| {
            calls.push(graphs);
            if graphs.is_some() { Err("memory") } else { Ok(1024) }
        }, |error| *error == "memory", |retry, _| retry), Ok((1024, false)));
        assert_eq!(calls, [Some(200), None]);
        assert_eq!(admit_beside_decode_graphs(startup, |_| Err("geometry"),
            |error| *error == "memory", |retry, _| retry), Err("geometry"));
        assert_eq!(admit_beside_decode_graphs(startup, |graphs| Err(if graphs.is_some() { "startup memory" } else { "lazy memory" }),
            |error| error.ends_with("memory"), |retry, _| retry), Err("startup memory"));
    }

    #[test]
    fn exchange_matches_peer_slots_and_control_floor() {
        assert_eq!(peer_exchange_bytes(2, 4096, 4096, 2, true), 402_653_440);
        assert_eq!(peer_exchange_bytes(2, 4096, 4096, 2, false), 268_435_712);
    }

    #[test]
    fn startup_counts_match_rc3_and_peer_has_no_head_graph() {
        let buckets = DecodeBuckets::new(verify_budget(64, 188));
        let shapes = serving_graph_shapes(1_048_576, 2_097_152 / PAGE_ROWS, 2051, 16, true, &buckets);
        assert_eq!(shapes.len(), 300);
        let bytes = serving_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, true, &buckets);
        assert_eq!(bytes, vec![graph_reserve_bytes(13_800), graph_reserve_bytes(13_500)]);
        assert_eq!(verify_budget(128, 170), 127);
        assert_eq!(verify_budget(128, 132), 99);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn serving_graph_counts(context: usize, pool_tokens: usize, dense: usize, sequences: usize,
    speculation: bool, layers: usize, peer: bool, buckets: &DecodeBuckets) -> Vec<u64> {
    let shapes = serving_graph_shapes(context, pool_tokens.div_ceil(PAGE_ROWS), dense, sequences,
        speculation, buckets).len() as u64;
    std::iter::once(shapes * (layers + 1) as u64).chain(peer.then_some(shapes * layers as u64)).collect()
}

/// Every exchange slot owns its payload, completion word and native control storage.
pub fn peer_exchange_bytes(lanes: u64, rows: u64, hidden: u64, partial_bytes: u64, output_shard: bool) -> u64 {
    let slots = lanes * if output_shard { 6 } else { 4 };
    slots * (rows.max(64) * hidden * partial_bytes).max(256) + ((slots + 1) * 16).max(256)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartupGraphReserve {
    pub reserve: u64,
    pub allowance: u64,
}

/// Retry only a memory refusal caused by startup bytes above the lazy allowance.
/// Error classification and contextual diagnostics belong to the caller's edge.
pub fn admit_beside_decode_graphs<E>(
    startup: Option<StartupGraphReserve>,
    mut admit: impl FnMut(Option<u64>) -> Result<usize, E>,
    shortfall: impl Fn(&E) -> bool,
    retry_error: impl FnOnce(E, E) -> E,
) -> Result<(usize, bool), E> {
    let Some(graphs) = startup else { return Ok((admit(None)?, false)) };
    let refused = match admit(Some(graphs.reserve)) {
        Ok(tokens) => return Ok((tokens, true)),
        Err(error) if graphs.reserve > graphs.allowance && shortfall(&error) => error,
        Err(error) => return Err(error),
    };
    match admit(None) {
        Ok(tokens) => Ok((tokens, false)),
        Err(retry) if shortfall(&retry) => Err(refused),
        Err(retry) => Err(retry_error(retry, refused)),
    }
}

/// Re-evaluate an automatic admission at its own graph geometry. Shrinking only avoids
/// oscillation across graph-shape thresholds; the returned reservation describes the pool.
/// The closure returns `(pool_tokens, startup_graphs)`, preserving lazy-fallback policy.
pub fn solve_graph_pool<E>(initial: usize, mut admit: impl FnMut(usize) -> Result<(usize, bool), E>)
    -> Result<(usize, bool), E> {
    let mut candidate = initial;
    loop {
        let (pool, startup) = admit(candidate)?;
        if !startup || pool >= candidate { return Ok((candidate.min(pool), startup)); }
        candidate = pool;
    }
}

/// Context chosen by the automatic serving policy after pool admission.
pub fn admitted_graph_context(context: usize, pool: usize, automatic: bool) -> usize {
    if automatic && context > pool { pool.saturating_sub(UNIT_ROWS) } else { context }
}
