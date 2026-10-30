//! The V4.1 Engram look-ahead: the shared token-table look-ahead
//! ([`crate::formats::mapped_table::lookahead`]) over layers 1 and 14. Each
//! request's rows come from a forked Engram history (nothing is committed),
//! hashed once per piece for both layers; layer 14 of a unit is read once layer
//! 1 has landed through the units in flight (the shared N-layer deadline rule).
use super::engram_prefetch::RowRead;
use crate::formats::mapped_table::lookahead::{self, LookaheadTable, RowLanding, RowSource};
use crate::formats::mapped_table::row_cache::RowCache;
use crate::formats::mapped_table::{placement::Placement, GatherMode, ReadPriority, TableStats};
use crate::EngramTable;
use anyhow::Result;
use cuteafd_core::EngramHistory;
use std::ops::Range;
use std::sync::Arc;

pub use lookahead::{window_units, LookaheadHandle, UnitPlan};

/// A V4.1 table read for the look-ahead: one reactor buffer per plane, never waited for.
struct EngramLanding { read: RowRead, table: Arc<EngramTable> }
impl RowLanding for EngramLanding {
    fn poll(&mut self) -> Result<bool> { self.read.poll(&self.table) }
    fn row(&self, index: usize) -> [&[u8]; 2] { self.read.row(index) }
}

/// One Engram layer as a look-ahead table.
struct EngramLayer(Arc<EngramTable>);
impl LookaheadTable for EngramLayer {
    fn cache(&self) -> &Arc<RowCache> { self.0.cache() }
    fn start_read(&self, rows: &[u64], priority: ReadPriority) -> Result<Box<dyn RowLanding>> {
        let read = self.0.start_read_window(rows, priority, 1, GatherMode { no_wait: true })?;
        Ok(Box::new(EngramLanding { read, table: Arc::clone(&self.0) }))
    }
    fn stats(&self) -> &TableStats { self.0.weights().stats() }
    fn rows_per_token(&self) -> usize { 24 }
    fn placement(&self) -> Placement {
        let weights = self.0.weights().placement();
        let scales = self.0.scales().map_or(Placement::Local, |scales| scales.placement());
        if weights == Placement::Local && scales == Placement::Local { Placement::Local } else { Placement::Remote }
    }
}

/// A prompt's Engram rows: compressed ids (None for image rows) after a forked history.
struct EngramSource { history: EngramHistory, tokens: Vec<Option<u32>> }
impl RowSource for EngramSource {
    fn tokens(&self) -> usize { self.tokens.len() }
    fn rows(&mut self, tokens: Range<usize>) -> Result<Vec<Vec<u64>>> {
        let count = tokens.len();
        let batch = self.history.prepare(self.history.position(), &self.tokens[tokens], count.max(1))?;
        self.history.commit(&batch, count)?;
        Ok(vec![batch.prefetch_rows(0)?, batch.prefetch_rows(1)?])
    }
}

/// The planner over both Engram layers; dropping it stops the look-ahead.
pub struct EngramLookahead(lookahead::Lookahead);

impl EngramLookahead {
    pub fn new(tables: [Arc<EngramTable>; 2]) -> Result<Self> {
        let tables = tables.into_iter().map(|table| Arc::new(EngramLayer(table)) as Arc<dyn LookaheadTable>).collect();
        Ok(Self(lookahead::Lookahead::new("engram", tables)?))
    }

    /// Read a decode round's drafted rows (per layer) into the decode share the
    /// moment the drafter emits them. Never blocks.
    pub fn decode(&self, rows: [Vec<u64>; 2]) { self.0.decode(rows.into()) }

    /// Read ahead `tokens` (compressed ids, None for image rows) that follow
    /// `history`'s frontier. `history` is a fork: its batches are never committed.
    pub fn start(&self, history: EngramHistory, tokens: Vec<Option<u32>>, units: UnitPlan) -> LookaheadHandle {
        self.start_tagged(0, history, tokens, units)
    }
    /// [`Self::start`] with the caller's request id for the per-request summary.
    pub fn start_tagged(&self, tag: u64, history: EngramHistory, tokens: Vec<Option<u32>>, units: UnitPlan)
        -> LookaheadHandle {
        let start = history.position();
        self.0.start(tag, start, Box::new(EngramSource { history, tokens }), units)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mapped_table::row_cache::{Lookup, RowClass};
    use std::sync::mpsc;
    /// A V4.1 piece: one reactor batch of 24-row tokens (2048 / 24, in 64-token steps).
    const PIECE_TOKENS: usize = 64;
    use std::os::unix::fs::FileExt;

    fn table(layer: usize, rows: &[u64]) -> Result<Arc<EngramTable>> {
        let weights = tempfile::NamedTempFile::new()?;
        let scales = tempfile::NamedTempFile::new()?;
        let count = cuteafd_core::ENGRAM_ROWS[layer];
        weights.as_file().set_len(count * 256)?;
        scales.as_file().set_len(count * 8)?;
        for &row in rows {
            weights.as_file().write_all_at(&[(row % 251) as u8; 256], row * 256)?;
            scales.as_file().write_all_at(&[(row % 13) as u8; 8], row * 8)?;
        }
        let map = |file: &tempfile::NamedTempFile, width| unsafe {
            // SAFETY: the fixture files outlive the mapping (unlinked names keep their inode).
            crate::MappedTable::single(file.path(), 0, count,
                crate::RowFormat { dtype: cuteafd_core::DType::U8, width, row_bytes: width })
        };
        let mut table = EngramTable::new(map(&weights, 256)?, map(&scales, 8)?)?;
        table.set_cache_shares([16384, 1024])?;
        Ok(Arc::new(table))
    }

    #[test]
    fn look_ahead_fills_both_caches_and_releases_at_commit() -> Result<()> {
        let tokens: Vec<Option<u32>> = (0..300).map(|i| Some(i * 7 % 99_000)).collect();
        let history = EngramHistory::new(2)?;
        let batch = history.prepare(0, &tokens, 300)?;
        let rows = [batch.prefetch_rows(0)?, batch.prefetch_rows(1)?];
        let tables = [table(0, &rows[0])?, table(1, &rows[1])?];
        let ahead = EngramLookahead::new(tables.clone())?;
        let units = || UnitPlan::new(vec![128, 256, 300], 3);
        let handle = ahead.start(history.fork()?, tokens.clone(), units());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        // Pieces of 64 tokens (one reactor batch of 24-row tokens): 0-64, 64-128, 128-192, 192-256, 256-300.
        while tables.iter().any(|t| t.weights().stats().snapshot().lookahead_pieces < 5) {
            assert!(std::time::Instant::now() < deadline, "look-ahead did not finish");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        for (layer, table) in tables.iter().enumerate() {
            let mut distinct = rows[layer].clone();
            distinct.sort_unstable();
            distinct.dedup();
            let (pins, lookups) = table.cache().acquire(&distinct, RowClass::Decode);
            for (lookup, &row) in lookups.iter().zip(&distinct) {
                let Lookup::Ready(slot) = *lookup else { panic!("row {row} not cached: {lookup:?}") };
                assert_eq!(pins.row(slot)[0], (row % 251) as u8);
                assert_eq!(pins.row(slot)[256], (row % 13) as u8);
            }
        }
        // A second request over the same prompt pins the cached rows, reading none
        // (its pieces complete without a read).
        let read = tables[0].weights().stats().snapshot().lookahead_rows;
        let stale = ahead.start(history.fork()?, tokens.clone(), units());
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(tables[0].weights().stats().snapshot().lookahead_rows, read);
        assert_eq!(tables[0].weights().stats().snapshot().lookahead_pieces, 10);
        // Commits past the end release every piece; the handle reports the end.
        assert!(!handle.commit(300));
        drop(handle);
        // A handle outliving the look-ahead must not keep its planner alive.
        drop(ahead);
        drop(stale);
        for table in &tables {
            assert_eq!(table.cache().room(RowClass::Prefill), 16384);
        }
        Ok(())
    }

    #[test]
    fn layer_14_of_a_unit_waits_for_layer_1_of_the_units_in_flight() -> Result<()> {
        // Five units of 256 tokens at depth 3: unit 0's layer 14 is read only
        // after layer 1 of units 0, 1 and 2 has landed, and before unit 3's
        // layer 1 (whose deadline comes after unit 0 commits).
        let tokens: Vec<Option<u32>> = (0..1280u32).map(|i| Some(i % 90_000)).collect();
        let history = EngramHistory::new(2)?;
        let mut owned = [table(0, &[])?, table(1, &[])?].map(|t| Arc::try_unwrap(t).ok().expect("sole owner"));
        for table in &mut owned { table.set_cache_shares([2 * 3 * 256 * 24, 1024])?; }
        let tables = owned.map(Arc::new);
        let ahead = EngramLookahead::new(tables.clone())?;
        let _handle = ahead.start(history.fork()?, tokens, UnitPlan::new(vec![256, 512, 768, 1024, 1280], 3));
        let snapshot = |layer: usize| tables[layer].weights().stats().snapshot();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while snapshot(1).lookahead_rows == 0 {
            assert!(std::time::Instant::now() < deadline, "layer 14 never started");
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        // Six layer-1 pieces (units 0-2, two pieces each) precede any layer-14 read.
        assert!(snapshot(0).lookahead_pieces >= 6, "{}", snapshot(0).lookahead_pieces);
        // Every unit eventually lands in both layers (the window is 2 d = 6 units).
        while snapshot(0).lookahead_pieces < 10 || snapshot(1).lookahead_pieces < 10 {
            assert!(std::time::Instant::now() < deadline, "look-ahead did not finish");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        Ok(())
    }

    #[test]
    fn units_bound_the_window_to_two_full_prefills() -> Result<()> {
        // Depth 1, units of 128: the window is 2 units past the committed one.
        let tokens: Vec<Option<u32>> = (0..640u32).map(|i| Some(i + 9)).collect();
        let history = EngramHistory::new(2)?;
        let tables = [table(0, &[])?, table(1, &[])?];
        let ahead = EngramLookahead::new(tables.clone())?;
        let handle = ahead.start(history.fork()?, tokens, UnitPlan::new(vec![128, 256, 384, 512, 640], 1));
        let pieces = || tables[0].weights().stats().snapshot().lookahead_pieces;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        // Two units of 128 tokens are four 64-token pieces.
        while pieces() < 4 { assert!(std::time::Instant::now() < deadline); std::thread::sleep(std::time::Duration::from_millis(2)); }
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(pieces(), 4, "two units past the frontier, no more");
        assert!(handle.commit(128));
        while pieces() < 6 { assert!(std::time::Instant::now() < deadline); std::thread::sleep(std::time::Duration::from_millis(2)); }
        Ok(())
    }

    #[test]
    fn drafted_rows_land_in_the_decode_share_and_stay_cached() -> Result<()> {
        let history = EngramHistory::new(2)?;
        let batch = history.prepare(0, &[Some(11), Some(12), Some(13)], 3)?;
        let rows = [batch.prefetch_rows(0)?, batch.prefetch_rows(1)?];
        let tables = [table(0, &rows[0])?, table(1, &rows[1])?];
        let ahead = EngramLookahead::new(tables.clone())?;
        ahead.decode(rows.clone());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while tables.iter().any(|t| t.weights().stats().snapshot().lookahead_pieces < 1) {
            assert!(std::time::Instant::now() < deadline, "draft look-ahead did not land");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        for (layer, table) in tables.iter().enumerate() {
            let (pins, lookups) = table.cache().acquire(&rows[layer][..1], RowClass::Decode);
            let Lookup::Ready(slot) = lookups[0] else { panic!("drafted row not cached: {:?}", lookups[0]) };
            assert_eq!(pins.row(slot)[0], (rows[layer][0] % 251) as u8);
        }
        Ok(())
    }

    #[test]
    fn look_ahead_stays_inside_the_cache_and_window() -> Result<()> {
        // A prefill share of two pieces: the planner needs room for both layers
        // of a piece, so the second piece waits for the first to commit.
        let tokens: Vec<Option<u32>> = (0..256).map(|i| Some(i + 5)).collect();
        let history = EngramHistory::new(2)?;
        let tables = [table(0, &[])?, table(1, &[])?];
        let mut owned = tables.map(|t| Arc::try_unwrap(t).ok().expect("sole owner"));
        for table in &mut owned { table.set_cache_shares([2 * PIECE_TOKENS * 24, 64])?; }
        let tables = owned.map(Arc::new);
        let ahead = EngramLookahead::new(tables.clone())?;
        let handle = ahead.start(history.fork()?, tokens, UnitPlan::new(vec![128, 256], 3));
        let pieces = || tables[0].weights().stats().snapshot().lookahead_pieces;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while pieces() < 1 { assert!(std::time::Instant::now() < deadline); std::thread::sleep(std::time::Duration::from_millis(5)); }
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(pieces(), 1, "no room until the first piece commits");
        assert!(handle.commit(PIECE_TOKENS as u64));
        while pieces() < 2 { assert!(std::time::Instant::now() < deadline); std::thread::sleep(std::time::Duration::from_millis(5)); }
        Ok(())
    }

    /// The cand4 hang shape: a flood of decode commands whose rows all miss
    /// the page cache must not leave the planner holding reactor buffers while
    /// it blocks for more, so a gather that needs a buffer still finishes.
    #[test]
    fn decode_flood_never_starves_a_gather_of_reactor_buffers() -> Result<()> {
        let history = EngramHistory::new(2)?;
        let tables = [table(0, &[])?, table(1, &[])?];
        let ahead = EngramLookahead::new(tables.clone())?;
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        // Far more pieces than the reactor has buffers, every row cold.
        for _ in 0..64 {
            let rows = [0, 1].map(|layer| (0..24).map(|_| next() % cuteafd_core::ENGRAM_ROWS[layer]).collect::<Vec<_>>());
            ahead.decode(rows);
        }
        let batch = std::sync::Arc::new(history.prepare(0, &[Some(41), Some(42), Some(43)], 3)?);
        let gather_tables = tables.clone();
        let (send, receive) = mpsc::channel();
        std::thread::spawn(move || {
            let mut staging = crate::EngramBatchStaging::new(3).expect("staging");
            let result = [0, 1].into_iter().try_for_each(|layer| {
                let prepared = staging.prepare(&gather_tables[layer], &[batch.as_ref()], layer, ReadPriority::Decode)?;
                staging.finish(&gather_tables[layer], prepared)
            });
            let _ = send.send(result.map_err(|error| error.to_string()));
        });
        let Ok(outcome) = receive.recv_timeout(std::time::Duration::from_secs(20)) else {
            // A starved planner never stops: fail without joining it.
            std::mem::forget(ahead);
            anyhow::bail!("a decode gather starved behind the look-ahead's reactor buffers");
        };
        outcome.map_err(anyhow::Error::msg)?;
        drop(ahead);
        Ok(())
    }
}
