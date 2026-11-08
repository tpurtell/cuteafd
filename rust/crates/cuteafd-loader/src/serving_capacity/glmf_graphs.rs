//! GLM Flash startup decode graph inventory, shared by capture and CPU admission.
use super::glmf_table_pages;
pub const PAGE_ROWS: usize = 64;
pub const UNIT_PAGES: usize = 4;
pub const UNIT_ROWS: usize = PAGE_ROWS * UNIT_PAGES;
pub const DECODE_ROWS: usize = 64;
pub const PLAIN_DECODE_BUCKETS: [usize; 6] = [1, 4, 8, 16, 32, 64];
pub const SPEC_DECODE_BUCKETS: [usize; 6] = [2, 4, 8, 16, 32, 64];
use crate::placement::inventory::{CapturedGraphRank, DeclaredGraphSet, GraphRank};
use serde::{Deserialize, Serialize};

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

/// Whole historical observations, kept only for CPU unit fixtures that plan a
/// synthetic geometry no census covers. No production path scales a count:
/// admission charges a measured census row for the exact declared fingerprint
/// ([`census_rank`]) and refuses any other set (#1577).
#[cfg(test)]
pub(crate) fn estimated_graph_rank(graphs: u64, role: usize) -> GraphRank {
    let (bytes, count) = [(1_946_157_056, 13_800), (2_231_369_728, 13_800),
        (2_283_798_528, 13_500)][role.min(2)];
    GraphRank::estimate(graphs, bytes, count).expect("GLMF historical graph estimate fits u64")
}

/// The startup decode graph set's bytes per rank: `buckets` (the engine's sets) over every geometry.
#[allow(clippy::too_many_arguments)]
pub fn serving_graph_reserve(context: usize, pool_tokens: usize, dense: usize,
    sequences: usize, speculation: bool, layers: usize, peer: bool, buckets: &DecodeBuckets)
    -> Result<Vec<u64>, String> {
    Ok(startup_set(context, pool_tokens, dense, sequences, speculation, layers, peer, buckets)?
        .inventory.ranks.iter().map(|rank| rank.bytes).collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    fn exchange_matches_peer_slots_and_control_floor() {
        assert_eq!(peer_exchange_bytes(2, 4096, 4096, 2, true), 402_653_440);
        assert_eq!(peer_exchange_bytes(2, 4096, 4096, 2, false), 268_435_712);
    }

    #[test]
    fn measured_census_row_calibrates_the_serving_key_set_through_the_full_pool() {
        let buckets = DecodeBuckets::new(verify_budget(64, 188));
        let shapes = |pool: usize| serving_graph_shapes(1_048_576, pool.div_ceil(PAGE_ROWS), 2051, 16, true, &buckets);
        let declared = declared_fingerprints(&shapes(2_000_128), 45, 1).unwrap();
        assert_eq!(declared.len(), 1);
        // The exact key set the census row was captured on; a count alone would
        // also match a different set, which is why the fingerprint is framed.
        assert_eq!(declared[0], STARTUP_GRAPH_CENSUS[0].declared_graphs);
        // A pool that fills a 96 GiB card keys the same set: the stride floors
        // and the page-stride/pool-stride caps do not move between 2M and 100M.
        assert_eq!(declared_fingerprints(&shapes(100_000_000), 45, 1).unwrap()[0], declared[0]);
        // A one-GPU reserve is charged from the row, never from a scaled count.
        let graphs = startup_set(1_048_576, 2_000_128, 2051, 16, true, 45, false, &buckets).unwrap();
        assert_eq!(graphs.inventory.ranks[0].bytes, STARTUP_GRAPH_CENSUS[0].bytes);
        assert_eq!(graphs.inventory.ranks[0].margin, 0);
        assert_eq!(graphs.inventory.ranks[0].basis, cuteafd_core::memory_layout::Basis::Calibrated);
        assert_eq!(graphs.keys.len(), shapes(2_000_128).len());
        assert_eq!(graphs.inventory.ranks[0].executables, 13_800);
    }

    #[test]
    fn uncensused_declared_sets_refuse_with_their_fingerprint() {
        let buckets = DecodeBuckets::new(verify_budget(64, 188));
        let (context, pool) = (131_072, 2_097_152);
        refuse_unmeasured_graphs();
        let error = startup_set(context, pool, 2051, 16, true, 45, false, &buckets).unwrap_err();
        assert!(error.contains("no measured instantiate row"), "{error}");
        assert!(error.contains(&declared_fingerprints(
            &serving_graph_shapes(context, pool / PAGE_ROWS, 2051, 16, true, &buckets), 45, 1).unwrap()[0]),
            "{error}");
        assert!(error.contains("STARTUP_GRAPH_CENSUS"), "{error}");
        super::SYNTHETIC_GRAPHS.with(|flag| flag.set(true));
    }

    #[test]
    fn startup_counts_match_rc3_and_peer_has_no_head_graph() {
        let buckets = DecodeBuckets::new(verify_budget(64, 188));
        let shapes = serving_graph_shapes(1_048_576, 2_097_152 / PAGE_ROWS, 2051, 16, true, &buckets);
        assert_eq!(shapes.len(), 300);
        // Rank 0 is the censused set; rank 1's peer set has no row, and the
        // test-only fixture fallback supplies its historical estimate.
        let bytes = serving_graph_reserve(1_048_576, 2_097_152, 2051, 16, true, 45, true, &buckets).unwrap();
        assert_eq!(bytes, vec![STARTUP_GRAPH_CENSUS[0].bytes, estimated_graph_rank(13_500, 2).bytes]);
        assert_eq!(verify_budget(128, 170), 127);
        assert_eq!(verify_budget(128, 132), 99);
    }
}

/// Stable launch geometry shared by admission and the executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphKey {
    pub segment: usize,
    pub rows: usize,
    pub spec: bool,
    pub long: bool,
    pub pool_width: usize,
    pub page_stride: usize,
    pub pool_stride: usize,
}

impl GraphKey {
    pub fn new(segment: usize, rows: usize, spec: bool, geometry: GraphGeometry) -> Self {
        let geometry = GraphGeometry::keyed(geometry.pool_width, geometry.page_stride, geometry.pool_stride, geometry.long);
        Self { segment, rows, spec, long: geometry.long, pool_width: geometry.pool_width,
            page_stride: geometry.page_stride, pool_stride: geometry.pool_stride }
    }
}

pub fn executable_keys(shapes: &[(usize, bool, GraphGeometry)], layers: usize, rank: usize) -> Vec<GraphKey> {
    shapes.iter().flat_map(|&(rows, spec, geometry)| (0..layers + usize::from(rank == 0))
        .map(move |segment| GraphKey::new(segment, rows, spec, geometry))).collect()
}

pub fn declared_fingerprints(shapes: &[(usize, bool, GraphGeometry)], layers: usize, ranks: usize)
    -> Result<Vec<String>, String> {
    (0..ranks).map(|rank| DeclaredGraphSet::new(&executable_keys(shapes, layers, rank))?
        .fingerprint(rank as u8)).collect()
}

/// Capture bytes and exact retained executable identities; never a count-only proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupCapture {
    pub ranks: Vec<CapturedGraphRank>,
}

/// One immutable geometry list and its per-rank ledger for both admission and
/// capture. Each rank's bytes are its measured census row for the declared
/// fingerprint; an uncensused set is a refusal naming the fingerprint.
#[allow(clippy::too_many_arguments)]
pub fn startup_set(context: usize, pool_tokens: usize, dense: usize, sequences: usize,
    speculation: bool, layers: usize, peer: bool, buckets: &DecodeBuckets)
    -> Result<crate::placement::inventory::StartupGraphs<(usize, bool, GraphGeometry)>, String> {
    use crate::placement::inventory::{GraphSet, Lifetime};
    let keys = serving_graph_shapes(context, pool_tokens.div_ceil(PAGE_ROWS), dense, sequences, speculation, buckets);
    let shapes = keys.len() as u64;
    let gpus = 1 + usize::from(peer);
    let declared = declared_fingerprints(&keys, layers, gpus)?;
    let ranks = (0..gpus).map(|rank| census_rank(
        shapes * (layers + usize::from(rank == 0)) as u64, &declared[rank], rank, gpus))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(GraphSet { ranks, shapes, lifetime: Lifetime::Startup }.startup(keys)
        .expect("startup geometry list and ledger are constructed together"))
}

#[allow(clippy::too_many_arguments)]
pub fn serving_graph_counts(context: usize, pool_tokens: usize, dense: usize, sequences: usize,
    speculation: bool, layers: usize, peer: bool, buckets: &DecodeBuckets) -> Vec<u64> {
    let shapes = serving_graph_shapes(context, pool_tokens.div_ceil(PAGE_ROWS), dense, sequences,
        speculation, buckets).len() as u64;
    std::iter::once(shapes * (layers + 1) as u64).chain(peer.then_some(shapes * layers as u64)).collect()
}

/// A measured startup decode graph residency for one exact declared key set.
///
/// `declared_graphs` fingerprints every captured executable (segment, rows,
/// speculation, long/short and page-table geometry) for one coordinator rank,
/// so a row matches only that set, never a count. `gpus` is the coordinator
/// rank count the capture ran with, recorded as provenance: the fingerprint is
/// per rank, so a rank-0 row serves one- and two-coordinator captures alike and
/// a second rank needs its own row. Provenance records the binary and hardware
/// that produced the measurement; a row is added by capturing the set in one
/// launch (#1577), never by scaling a different count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartupGraphCensus {
    pub declared_graphs: &'static str,
    pub executables: u64,
    pub gpus: u8,
    pub bytes: u64,
    pub programs_sha256: &'static str,
    pub native_sha256: &'static str,
    pub toolchain: &'static str,
    pub gpu_class: &'static str,
    pub driver: &'static str,
    pub evidence: &'static str,
    pub producing_commit: &'static str,
}

/// Measured startup graph sets. A declared set without a row is refused by
/// [`census_rank`], naming its fingerprint: a startup key-set change needs a
/// fresh capture and a new row before that layout can serve.
pub const STARTUP_GRAPH_CENSUS: &[StartupGraphCensus] = &[StartupGraphCensus {
    declared_graphs: "8757d4da6763b4536c2916bb84f20fc221d86681d487ecae2f0e296a03a0e292",
    executables: 13_800,
    gpus: 1,
    bytes: 2_313_158_656,
    programs_sha256: "60a42f767fcad1695df928c6d7ce1c8621490354127c001c2f096c9040b9194b",
    native_sha256: "d6f7646a781d54dccd3aa96118e088e1e58f0291961ac2bf75744bb9e7149811",
    toolchain: "rustc 1.98.1 (48a229cea 2026-09-01); binary: rustc; commit-hash: 48a229ceaefd4985c50990b14116b6d856af0985; commit-date: 2026-09-01; host: x86_64-unknown-linux-gnu; release: 1.98.1; LLVM version: 22.1.8",
    gpu_class: "sm_120/188/101973491712",
    driver: "13.2",
    evidence: "GLM Flash synchronized startup capture after exact masked-segment prewarm; tracked device growth excluded",
    producing_commit: "bfe5f6babe3768bd9640d7adaff467fc630850b9",
}];

/// Unit fixtures plan synthetic geometries no launch has censused. The
/// fallback is compiled into test builds only; production refuses instead.
#[cfg(test)]
thread_local! {
    static SYNTHETIC_GRAPHS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

#[cfg(test)]
pub(crate) fn refuse_unmeasured_graphs() { SYNTHETIC_GRAPHS.with(|flag| flag.set(false)); }

/// One rank's calibrated ledger from the measured row for its exact declared
/// fingerprint. Graph instantiate storage is driver memory whose size follows
/// the captured kernels, not the key count, so a row is never scaled to another
/// set: an uncensused fingerprint is a refusal naming it and the measurement
/// that would cover it.
pub fn census_rank(executables: u64, fingerprint: &str, rank: usize, gpus: usize)
    -> Result<GraphRank, String> {
    if let Some(row) = STARTUP_GRAPH_CENSUS.iter().find(|row|
        row.declared_graphs == fingerprint && row.executables == executables) {
        return Ok(GraphRank { executables, bytes: row.bytes, margin: 0,
            basis: cuteafd_core::memory_layout::Basis::Calibrated });
    }
    #[cfg(test)]
    if SYNTHETIC_GRAPHS.with(std::cell::Cell::get) {
        return Ok(estimated_graph_rank(executables, if gpus == 1 { 0 } else { rank + 1 }));
    }
    Err(format!("GLM Flash startup graphs: no measured instantiate row for declared key set \
        {fingerprint} ({executables} executables, rank {rank} of {gpus}); capture this layout with a \
        memory census and add its device {rank} instantiate bytes to STARTUP_GRAPH_CENSUS"))
}

/// Every exchange slot owns its payload, completion word and native control storage.
pub fn peer_exchange_bytes(lanes: u64, rows: u64, hidden: u64, partial_bytes: u64, output_shard: bool) -> u64 {
    let slots = lanes * if output_shard { 6 } else { 4 };
    slots * (rows.max(64) * hidden * partial_bytes).max(256) + ((slots + 1) * 16).max(256)
}

// Startup graphs are the only decode graphs a serving engine captures: admission
// charges every executable before the KV pool is sized, and a shape outside the
// declared startup set runs eagerly without allocating.
