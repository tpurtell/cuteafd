//! A single-plane token table (Qwen PLE n-grams) behind the shared row cache and
//! look-ahead: prompts are read ahead from admission, decode rows as soon as
//! the token is known, and each gather is a cache lookup plus a copy into the
//! caller's staging, reading only rows nobody has fetched.
use super::lookahead::{Lookahead, LookaheadHandle, LookaheadTable, RowLanding, RowSource, UnitPlan};
use super::row_cache::{Lookup, RowCache, RowClass};
use super::{GatherMode, MappedTable, PreparedTableGather, ReadPriority, TableStats};
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// A mapped table with its row cache.
pub struct TokenTable { table: Arc<MappedTable>, cache: Arc<RowCache>, rows_per_token: usize }

/// A look-ahead read of one table: its rows land in a private buffer.
struct TableLanding { gather: PreparedTableGather, table: Arc<MappedTable>, bytes: Vec<u8> }
impl RowLanding for TableLanding {
    fn poll(&mut self) -> Result<bool> { Ok(self.gather.poll(&self.table, &mut self.bytes)?) }
    fn row(&self, index: usize) -> [&[u8]; 2] {
        let width = self.table.row_bytes();
        [&self.bytes[index * width..(index + 1) * width], &[]]
    }
}

impl LookaheadTable for TokenTable {
    fn cache(&self) -> &Arc<RowCache> { &self.cache }
    fn start_read(&self, rows: &[u64], priority: ReadPriority) -> Result<Box<dyn RowLanding>> {
        let mut bytes = vec![0; rows.len() * self.table.row_bytes()];
        let slots: Vec<usize> = (0..rows.len()).collect();
        let gather = self.table.prepare_gather(rows, &slots, &mut bytes, priority, 1, GatherMode { no_wait: true })?;
        Ok(Box::new(TableLanding { gather, table: Arc::clone(&self.table), bytes }))
    }
    fn stats(&self) -> &TableStats { self.table.stats() }
    fn rows_per_token(&self) -> usize { self.rows_per_token }
}

impl TokenTable {
    /// `shares` are the cache's (prefill, decode) rows; `rows_per_token` bounds
    /// a prompt token's rows.
    pub fn new(table: MappedTable, shares: [usize; 2], rows_per_token: usize) -> Result<Arc<Self>> {
        let cache = RowCache::new(table.row_bytes(), shares, table.shared_stats())?;
        Ok(Arc::new(Self { table: Arc::new(table), cache, rows_per_token }))
    }
    /// Cache shares for prefill units of `unit_rows` tokens `depth` deep (the
    /// look-ahead's window plus the unit in flight) and `decode_rows` decode rows
    /// (two verify windows), at `rows_per_token` rows a token.
    pub fn shares(unit_rows: usize, depth: usize, decode_rows: usize, rows_per_token: usize) -> [usize; 2] {
        [(super::lookahead::window_units(depth) + 1) * unit_rows * rows_per_token, 2 * decode_rows * rows_per_token]
    }
    /// Host bytes of a cache of `shares` rows of `row_bytes`.
    pub fn cache_bytes(row_bytes: usize, shares: [usize; 2]) -> usize { RowCache::host_bytes(row_bytes, shares) }
    pub fn table(&self) -> &MappedTable { &self.table }
    pub fn row_bytes(&self) -> usize { self.table.row_bytes() }

    /// Gather `ids` (in order, duplicates allowed) into `output`, `row_bytes`
    /// each: cached rows are copied, rows another reader is filling are waited
    /// for, and the rest are read here (and published when the cache admits them).
    pub fn gather(&self, ids: &[u64], output: &mut [u8], priority: ReadPriority) -> Result<()> {
        let width = self.table.row_bytes();
        ensure!(output.len() == ids.len() * width, "token-table gather destination differs from its rows");
        // Distinct rows in order; duplicates are copied from their first slot.
        let mut first = HashMap::with_capacity(ids.len());
        let (mut distinct, mut slots, mut dups) = (Vec::new(), Vec::new(), Vec::new());
        for (slot, &id) in ids.iter().enumerate() {
            ensure!(id < self.table.rows(), "token-table row {id} outside its {} rows", self.table.rows());
            match first.entry(id) {
                std::collections::hash_map::Entry::Occupied(canonical) => dups.push((*canonical.get(), slot)),
                std::collections::hash_map::Entry::Vacant(entry) => { entry.insert(slot); distinct.push(id); slots.push(slot); }
            }
        }
        let class = if priority.is_decode() { RowClass::Decode } else { RowClass::Prefill };
        let (mut pins, lookups) = self.cache.acquire(&distinct, class);
        let (mut misses, mut pending) = (Vec::new(), Vec::new());
        for ((lookup, &id), &slot) in lookups.iter().zip(&distinct).zip(&slots) {
            match *lookup {
                Lookup::Ready(cached) => output[slot * width..(slot + 1) * width].copy_from_slice(pins.row(cached)),
                Lookup::Wait(cached) => pending.push((cached, slot, id)),
                Lookup::Fill(cached) => misses.push((id, Some(cached), slot)),
                Lookup::Uncached => misses.push((id, None, slot)),
            }
        }
        let read = |misses: &[(u64, Option<u32>, usize)], output: &mut [u8]| -> Result<Vec<u8>> {
            let ids: Vec<u64> = misses.iter().map(|m| m.0).collect();
            let mut bytes = vec![0; ids.len() * width];
            let dense: Vec<usize> = (0..ids.len()).collect();
            self.table.prepare_gather(&ids, &dense, &mut bytes, priority, 2, GatherMode::default())?
                .finish(&self.table, &mut bytes)?;
            for (i, &(_, _, slot)) in misses.iter().enumerate() {
                output[slot * width..(slot + 1) * width].copy_from_slice(&bytes[i * width..(i + 1) * width]);
            }
            Ok(bytes)
        };
        let mut fallback = misses.len();
        if !misses.is_empty() {
            let bytes = read(&misses, output)?;
            pins.publish_all(misses.iter().enumerate().filter_map(|(i, &(_, cached, _))|
                cached.map(|c| (c, [&bytes[i * width..(i + 1) * width], &[][..]]))))?;
        }
        let mut failed = Vec::new();
        for &(cached, slot, id) in &pending {
            if pins.wait(cached) { output[slot * width..(slot + 1) * width].copy_from_slice(pins.row(cached)); }
            else { failed.push((id, None, slot)); }
        }
        if !failed.is_empty() {
            // Their filler stopped (a cancelled request or failed read): read them here.
            read(&failed, output)?;
            fallback += failed.len();
        }
        let waited = pending.len() - failed.len();
        if fallback > 0 || waited > 0 { self.table.stats().record_gather_misses(class == RowClass::Decode, fallback, waited); }
        drop(pins);
        for (source, slot) in dups {
            output.copy_within(source * width..(source + 1) * width, slot * width);
        }
        Ok(())
    }
}

/// A prompt's rows for a one-table look-ahead: `hash(tokens)` appends each
/// token's rows (in token order) and advances its own history.
pub struct HashedSource<H: FnMut(&[u32], &mut Vec<u64>) -> Result<()> + Send + 'static> {
    pub tokens: Vec<u32>,
    pub hash: H,
}
impl<H: FnMut(&[u32], &mut Vec<u64>) -> Result<()> + Send + 'static> RowSource for HashedSource<H> {
    fn tokens(&self) -> usize { self.tokens.len() }
    fn rows(&mut self, tokens: std::ops::Range<usize>) -> Result<Vec<Vec<u64>>> {
        let mut rows = Vec::new();
        (self.hash)(&self.tokens[tokens], &mut rows)?;
        Ok(vec![rows])
    }
}

/// A token table with its look-ahead planner.
pub struct TokenTableLookahead { table: Arc<TokenTable>, planner: Lookahead }
impl TokenTableLookahead {
    pub fn new(name: &str, table: Arc<TokenTable>) -> Result<Self> {
        let planner = Lookahead::new(name, vec![Arc::clone(&table) as Arc<dyn LookaheadTable>])
            .context("starting a token-table look-ahead")?;
        Ok(Self { table, planner })
    }
    pub fn table(&self) -> &Arc<TokenTable> { &self.table }
    /// Read ahead a prompt's rows (`source`, from absolute position `start`) in its units.
    pub fn start(&self, tag: u64, start: u64, source: Box<dyn RowSource>, units: UnitPlan) -> LookaheadHandle {
        self.planner.start(tag, start, source, units)
    }
    /// Read a decode round's rows into the decode share now. Never blocks.
    pub fn decode(&self, rows: Vec<u64>) { if !rows.is_empty() { self.planner.decode(vec![rows]); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mapped_table::RowFormat;
    use std::os::unix::fs::FileExt;

    fn fixture(rows: u64, width: usize, shares: [usize; 2]) -> Result<(tempfile::NamedTempFile, Arc<TokenTable>)> {
        let file = tempfile::NamedTempFile::new()?;
        file.as_file().set_len(rows * width as u64)?;
        for row in 0..rows { file.as_file().write_all_at(&vec![(row % 251) as u8; width], row * width as u64)?; }
        // SAFETY: the fixture file outlives the mapping.
        let table = unsafe { MappedTable::single(file.path(), 0, rows, RowFormat { dtype: cuteafd_core::DType::U8, width, row_bytes: width })? };
        Ok((file, TokenTable::new(table, shares, 4)?))
    }

    #[test]
    fn gather_copies_duplicates_and_publishes_into_the_cache() -> Result<()> {
        let (_file, table) = fixture(1000, 160, [64, 16])?;
        let ids = [7u64, 3, 7, 999, 3, 0];
        let mut out = vec![0u8; ids.len() * 160];
        table.gather(&ids, &mut out, ReadPriority::Prefill)?;
        for (i, &id) in ids.iter().enumerate() { assert!(out[i * 160..(i + 1) * 160].iter().all(|&b| b == (id % 251) as u8)); }
        // Every distinct row is now cached.
        let (pins, lookups) = table.cache().acquire(&[0, 3, 7, 999], RowClass::Decode);
        assert!(lookups.iter().all(|l| matches!(l, Lookup::Ready(_))), "{lookups:?}");
        drop(pins);
        // A gather past the share reads uncached rows privately and stays exact.
        let ids: Vec<u64> = (100..300).collect();
        let mut out = vec![0u8; ids.len() * 160];
        table.gather(&ids, &mut out, ReadPriority::Prefill)?;
        for (i, &id) in ids.iter().enumerate() { assert_eq!(out[i * 160], (id % 251) as u8); }
        Ok(())
    }

    #[test]
    fn look_ahead_fills_a_prompt_and_its_gathers_hit() -> Result<()> {
        let (_file, table) = fixture(4096, 160, [4 * 512, 64])?;
        let ahead = TokenTableLookahead::new("test-ple", Arc::clone(&table))?;
        let tokens: Vec<u32> = (0..256).collect();
        // Four rows a token from a toy hash; the source keeps no history.
        let hash = |t: &[u32], rows: &mut Vec<u64>| -> Result<()> {
            rows.extend(t.iter().flat_map(|&t| (0..4u64).map(move |h| (u64::from(t) * 4 + h) % 4096)));
            Ok(())
        };
        let handle = ahead.start(1, 0, Box::new(HashedSource { tokens: tokens.clone(), hash }),
            UnitPlan { ends: vec![128, 256], depth: 1 });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while table.stats().snapshot().lookahead_rows < 1024 {
            assert!(std::time::Instant::now() < deadline, "look-ahead did not land");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let before = table.stats().snapshot();
        let ids: Vec<u64> = (0..1024).collect();
        let mut out = vec![0u8; ids.len() * 160];
        table.gather(&ids, &mut out, ReadPriority::Prefill)?;
        let delta = table.stats().snapshot().since(&before);
        assert_eq!((delta.cache_hit_rows, delta.stall_prefill_fallback_rows), (1024, 0));
        assert!(!handle.commit(256));
        // Decode rows land in the decode share.
        ahead.decode(vec![4000, 4001]);
        while table.stats().snapshot().lookahead_rows < 1026 {
            assert!(std::time::Instant::now() < deadline, "decode look-ahead did not land");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        Ok(())
    }

    /// Qwen PLE prefill-unit gathers on the real table (QWEN_BENCH_SNAPSHOT):
    /// per-unit gather time, direct (the old path: rows read at gather time) vs
    /// through the row cache after the look-ahead, cold (QWEN_BENCH_DROP runs a
    /// true drop, e.g. `nest drop-caches --host raptor`) and warm.
    #[test]
    #[ignore = "CPU/IO microbench on a real snapshot, run in release"]
    fn qwen_ple_unit_gather_microbench() -> Result<()> {
        let Some(snapshot) = std::env::var_os("QWEN_BENCH_SNAPSHOT") else { return Ok(()) };
        let snapshot = std::path::Path::new(&snapshot);
        let checkpoint = crate::plan::Checkpoint::open(snapshot)?;
        let cfg = crate::families::qwen4::Qwen4Config::from_hf(&checkpoint.config)?;
        let mut shards: Vec<_> = checkpoint.tensors.iter()
            .filter(|t| t.meta.name.contains(".ngram_embedding.shard_") && t.meta.name.ends_with(".weight")).collect();
        shards.sort_by_key(|t| t.meta.name.rsplit("shard_").next().and_then(|s| s.split('.').next())
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(usize::MAX));
        let tensors: Vec<_> = shards.iter().map(|t| (t.shard.as_str(), &t.meta)).collect();
        let (unit, depth, units) = (2752usize, 3usize, 8usize);
        let per_token = cfg.ple_rows();
        let open = || -> Result<Arc<TokenTable>> {
            // SAFETY: the snapshot is immutable for the bench's lifetime.
            let table = unsafe { MappedTable::from_tensors(snapshot, &tensors)? };
            table.prepare_reads();
            TokenTable::new(table, TokenTable::shares(unit, depth, 64, per_token), per_token)
        };
        let rows = open()?.table().rows();
        let mut seed = std::env::var("QWEN_BENCH_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0x9E37_79B9u64);
        let prompt: Vec<Vec<u64>> = (0..units).map(|_| (0..unit * per_token).map(|_| {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % rows }).collect()).collect();
        let drop = || -> Result<()> {
            if let Ok(command) = std::env::var("QWEN_BENCH_DROP") {
                let status = std::process::Command::new("sh").arg("-c").arg(&command).status()?;
                anyhow::ensure!(status.success(), "drop command failed: {command}");
            }
            Ok(())
        };
        let cold = std::env::var_os("QWEN_BENCH_COLD").is_some();
        for arm in ["direct", "cached"] {
            let table = open()?;
            if cold { drop()?; } else {
                for ids in &prompt { let mut out = vec![0u8; ids.len() * table.row_bytes()]; table.table().gather_into(ids, &mut out)?; }
                // Warm in the page cache only: a fresh table has an empty row cache.
            }
            let table = if cold { table } else { open()? };
            let ahead = TokenTableLookahead::new("bench-ple", Arc::clone(&table))?;
            let mut out = vec![0u8; unit * per_token * table.row_bytes()];
            let started = std::time::Instant::now();
            let handle = (arm == "cached").then(|| {
                let units = UnitPlan { ends: (1..=units).map(|u| (u * unit) as u64).collect(), depth };
                let rows = prompt.clone();
                let mut next = 0usize;
                let hash = move |t: &[u32], out: &mut Vec<u64>| -> Result<()> {
                    let flat = rows.iter().flatten();
                    out.extend(flat.skip(next * per_token).take(t.len() * per_token));
                    next += t.len();
                    Ok(())
                };
                ahead.start(0, 0, Box::new(HashedSource { tokens: vec![0; units.ends.len() * unit], hash }), units)
            });
            let mut times = Vec::new();
            for (u, ids) in prompt.iter().enumerate() {
                // Each unit's consumer arrives one compute slot (~30 ms here) after the previous.
                std::thread::sleep(std::time::Duration::from_millis(30));
                let t = std::time::Instant::now();
                if arm == "cached" { table.gather(ids, &mut out, ReadPriority::Prefill)?; }
                else { table.table().gather_into_priority(ids, &mut out, ReadPriority::Prefill)?; }
                times.push(t.elapsed().as_secs_f64() * 1e3);
                if let Some(handle) = &handle { handle.commit(((u + 1) * unit) as u64); }
            }
            let total = started.elapsed().as_secs_f64() * 1e3;
            let snap = table.stats().snapshot();
            println!("qwen ple {} {arm}: {units} units of {} rows, gather ms per unit {:?}, sum {:.1}, wall {:.1}; hits {} fallback {} lookahead {}",
                if cold { "COLD" } else { "warm" }, unit * per_token,
                times.iter().map(|t| (t * 10.0).round() / 10.0).collect::<Vec<_>>(), times.iter().sum::<f64>(), total,
                snap.cache_hit_rows, snap.stall_prefill_fallback_rows, snap.lookahead_rows);
        }
        Ok(())
    }
}
