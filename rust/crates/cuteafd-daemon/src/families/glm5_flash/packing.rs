//! Packed admission prefill (`serve-glmf --prefill-batch`): the layout of one prefill pass over
//! the next chunks of several sequences ([`super::engine::GlmfEngine::prefill_packed`]).
//!
//! The sequences' rows sit back to back in one step, each sequence's rows contiguous and in
//! position order. Every coordinator program whose arithmetic depends on the rows it is given
//! (the mHC sites, which switch to TF32 mixes at 384 rows; the router scores, a skinny GEMV up to
//! 160 rows; the KDA layer, whose chunked recurrence above 64 rows holds one sequence; the DSA
//! indexer, top-k and selection, which read one sequence's tables; the LM head) runs once per
//! sequence over that sequence's rows, as its own pass would run it. The MLA producer, sparse
//! MLA, o, the dense and shared-expert MLPs, the router's top-8 and the wire rows run once over
//! all rows (one route for every row count, each row computed alone), and each MoE layer sends
//! one Spark wave for all of them: the expert ranks read each expert once for the whole burst
//! instead of once per prompt.
//!
//! Each sequence's page tables are cut to the columns its rows reach and laid back to back in the
//! step's one table row, and each sequence's drafter taps (its last rows, as a pass of its own
//! takes them) to their own tap rows, so a pass needs no buffer a one-sequence pass does not.
use super::engine::{KPOOL, PAGE_ROWS};
use anyhow::{ensure, Result};

/// Tokens of one pool-cache page (64 pools of [`KPOOL`] tokens).
const POOL_PAGE_TOKENS: usize = KPOOL * PAGE_ROWS;

/// What one packed pass may hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    /// Rows of the prefill workspace the pass runs in.
    pub rows: usize,
    /// The longest chunk a sequence's own pass runs as one step (pipelined Spark prefill cuts
    /// longer ones into lanes, which a packed pass does not): a sequence's rows in a packed pass
    /// are then exactly the rows its own pass gives each per-sequence program.
    pub chunk_rows: usize,
    /// Columns of the workspace's MLA page table and pool-page table (one sequence of the
    /// longest context).
    pub table_pages: usize,
    pub table_pool_pages: usize,
    /// The drafter's tap rows, when a drafter is attached.
    pub taps: Option<usize>,
    /// Longest context whose rows select every earlier token (longer rows run the pool top-k).
    pub dense_context: usize,
    pub max_context: usize,
}

/// One sequence's rows in a packed pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Segment {
    /// Its first step row, and its rows.
    pub first: usize,
    pub rows: usize,
    /// Its first position (the sequence's length before the pass).
    pub start: usize,
    /// Its MLA record pages: from `page_offset` in the step's page table, `page_columns` of them.
    pub page_offset: usize,
    pub page_columns: usize,
    /// Its pool pages: from `pool_offset` in the step's pool-page table, `pool_columns` of them
    /// (also the columns its top-k reads).
    pub pool_offset: usize,
    pub pool_columns: usize,
    /// A row of it sees more than `dense_context` tokens: its pool top-k runs.
    pub long: bool,
    /// Its last `tap_rows` rows go to drafter tap rows `tap_offset..` (none without a drafter).
    pub tap_offset: usize,
    pub tap_rows: usize,
}

impl Segment {
    /// One past its last step row.
    pub fn end_row(&self) -> usize {
        self.first + self.rows
    }
}

/// The pass over `chunks` (each sequence's length before the pass and the rows it adds, in step
/// order), or why it does not fit.
pub(crate) fn plan(chunks: &[(usize, usize)], limits: &Limits) -> Result<Vec<Segment>> {
    ensure!(!chunks.is_empty(), "a packed prefill needs a sequence");
    let mut out = Vec::with_capacity(chunks.len());
    let (mut first, mut pages, mut pools, mut taps) = (0, 0, 0, 0);
    for &(start, rows) in chunks {
        ensure!(rows > 0, "a packed prefill chunk of no rows");
        ensure!(rows <= limits.chunk_rows, "a packed prefill chunk of {rows} rows (its own pass cuts it past {})",
            limits.chunk_rows);
        let end = start + rows;
        ensure!(end <= limits.max_context, "a packed prefill to position {end} past the context {}", limits.max_context);
        let (page_columns, pool_columns) = (end.div_ceil(PAGE_ROWS), end.div_ceil(POOL_PAGE_TOKENS));
        let tap_rows = limits.taps.map_or(0, |cap| rows.min(cap));
        out.push(Segment { first, rows, start, page_offset: pages, page_columns, pool_offset: pools, pool_columns,
            long: end > limits.dense_context, tap_offset: taps, tap_rows });
        first += rows;
        pages += page_columns;
        pools += pool_columns;
        taps += tap_rows;
    }
    ensure!(first <= limits.rows, "a packed prefill of {first} rows exceeds the {}-row workspace", limits.rows);
    ensure!(pages <= limits.table_pages && pools <= limits.table_pool_pages,
        "a packed prefill's page tables ({pages} record and {pools} pool pages) exceed one table row ({} and {})",
        limits.table_pages, limits.table_pool_pages);
    if let Some(cap) = limits.taps {
        ensure!(taps <= cap, "a packed prefill's drafter taps ({taps} rows) exceed the {cap} tap rows");
    }
    Ok(out)
}

/// Whether `chunks` fit one packed pass.
pub(crate) fn fits(chunks: &[(usize, usize)], limits: &Limits) -> bool {
    plan(chunks, limits).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> Limits {
        // A 131,072-token context: 2,048 record-page and 512 pool-page columns.
        Limits { rows: 4096, chunk_rows: 4096, table_pages: 2048, table_pool_pages: 512, taps: Some(2048),
            dense_context: 2051, max_context: 131_072 }
    }

    #[test]
    fn sequences_sit_back_to_back_with_their_own_tables_and_taps() {
        let segments = plan(&[(0, 40), (0, 27), (300, 5)], &limits()).unwrap();
        assert_eq!(segments.iter().map(|s| (s.first, s.rows, s.start)).collect::<Vec<_>>(),
            [(0, 40, 0), (40, 27, 0), (67, 5, 300)]);
        // Record pages: 1, 1 and ceil(305 / 64) = 5; pool pages: 1, 1 and ceil(305 / 256) = 2.
        assert_eq!(segments.iter().map(|s| (s.page_offset, s.page_columns)).collect::<Vec<_>>(), [(0, 1), (1, 1), (2, 5)]);
        assert_eq!(segments.iter().map(|s| (s.pool_offset, s.pool_columns)).collect::<Vec<_>>(), [(0, 1), (1, 1), (2, 2)]);
        // Each sequence taps its own rows (all of them, under the 2,048 tap rows) at its own offset.
        assert_eq!(segments.iter().map(|s| (s.tap_offset, s.tap_rows)).collect::<Vec<_>>(), [(0, 40), (40, 27), (67, 5)]);
        assert!(segments.iter().all(|s| !s.long));
        assert_eq!((segments[2].end_row(), segments[2].start + segments[2].rows), (72, 305));
    }

    #[test]
    fn boundaries_fall_exactly_on_pages_and_pools() {
        // 64 rows fill one record page; 65 reach a second. 256 tokens fill one pool page.
        let segments = plan(&[(0, 64), (0, 65), (192, 64), (192, 65)], &limits()).unwrap();
        assert_eq!(segments.iter().map(|s| s.page_columns).collect::<Vec<_>>(), [1, 2, 4, 5]);
        assert_eq!(segments.iter().map(|s| s.pool_columns).collect::<Vec<_>>(), [1, 1, 1, 2]);
        assert_eq!(segments.iter().map(|s| s.page_offset).collect::<Vec<_>>(), [0, 1, 3, 7]);
        assert_eq!(segments.iter().map(|s| s.pool_offset).collect::<Vec<_>>(), [0, 1, 2, 3]);
    }

    #[test]
    fn a_row_past_the_dense_context_runs_its_own_top_k() {
        // Positions 0..2051 select every earlier token; position 2051 (2,052 tokens) is long.
        let segments = plan(&[(0, 2051), (2000, 52), (0, 8)], &Limits { taps: None, ..limits() }).unwrap();
        assert_eq!(segments.iter().map(|s| s.long).collect::<Vec<_>>(), [false, true, false]);
    }

    #[test]
    fn a_single_long_prompt_fills_the_workspace() {
        let segments = plan(&[(0, 4096)], &Limits { taps: None, ..limits() }).unwrap();
        assert_eq!((segments.len(), segments[0].rows, segments[0].page_columns, segments[0].pool_columns),
            (1, 4096, 64, 16));
        assert!(segments[0].long && segments[0].tap_rows == 0);
        assert!(!fits(&[(0, 4097)], &Limits { taps: None, ..limits() }));
        // With a drafter one sequence taps at most the tap rows (its last 2,048), as alone.
        let segments = plan(&[(0, 4096)], &limits()).unwrap();
        assert_eq!((segments[0].tap_offset, segments[0].tap_rows), (0, 2048));
    }

    #[test]
    fn mixed_lengths_are_held_to_rows_tables_and_taps() {
        assert!(fits(&[(0, 2000), (0, 2000), (0, 96)], &Limits { taps: None, ..limits() }));
        assert!(!fits(&[(0, 2000), (0, 2000), (0, 97)], &Limits { taps: None, ..limits() }), "4,097 rows");
        // Drafter taps: every row of each sequence, at most 2,048 in all.
        assert!(fits(&[(0, 1000), (0, 1048)], &limits()));
        assert!(!fits(&[(0, 1000), (0, 1049)], &limits()));
        // Long cached prefixes need long tables: two sequences at 100K positions take 1,563 record
        // pages each, more than one 2,048-column row holds.
        assert!(fits(&[(100_000, 8)], &limits()));
        assert!(!fits(&[(100_000, 8), (100_000, 8)], &limits()));
        // With record-page columns to spare, pool pages bind on their own: ceil(32,769 / 256) = 129
        // each, so three take 387 of the 512 columns and four would take 516.
        let wide = Limits { table_pages: 16_384, ..limits() };
        assert!(fits(&[(32_760, 9), (32_760, 9), (32_760, 9)], &wide));
        assert!(!fits(&[(32_760, 9), (32_760, 9), (32_760, 9), (32_760, 9)], &wide));
    }

    #[test]
    fn a_chunk_its_own_pass_would_cut_into_lanes_stays_out() {
        // Two Spark lanes cut a lone chunk from 512 rows: a packed pass takes 511 at most.
        let lanes = Limits { chunk_rows: 511, ..limits() };
        assert!(fits(&[(0, 511), (0, 511)], &lanes));
        assert!(!fits(&[(0, 512), (0, 7)], &lanes));
        assert!(!fits(&[(4096, 7), (0, 512)], &lanes));
    }

    #[test]
    fn chunks_past_the_context_or_empty_are_refused() {
        assert!(!fits(&[], &limits()));
        assert!(!fits(&[(0, 0)], &limits()));
        assert!(!fits(&[(131_070, 3)], &limits()));
        assert!(fits(&[(131_070, 2)], &Limits { table_pages: 4096, ..limits() }));
    }

    #[test]
    fn chunk_plans_cut_at_any_row_keep_their_positions() {
        // A prompt's second chunk (from 4,096) beside a fresh prompt: its own positions and pages.
        let segments = plan(&[(4096, 100), (0, 100)], &limits()).unwrap();
        assert_eq!((segments[0].start, segments[0].page_columns, segments[0].pool_columns), (4096, 66, 17));
        assert!(segments[0].long && !segments[1].long);
        assert_eq!((segments[1].first, segments[1].page_offset, segments[1].pool_offset), (100, 66, 17));
    }
}
