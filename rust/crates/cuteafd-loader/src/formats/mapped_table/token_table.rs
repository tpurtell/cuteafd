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
pub struct TokenTable {
    table: Arc<MappedTable>,
    cache: Arc<RowCache>,
    rows_per_token: usize,
    /// Tests only: a placement other than the files' own.
    #[cfg(test)]
    placement: Option<super::placement::Placement>,
}

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
    fn placement(&self) -> super::placement::Placement {
        #[cfg(test)]
        if let Some(placement) = self.placement { return placement; }
        self.table.placement()
    }
}

impl TokenTable {
    /// `shares` are the cache's (prefill, decode) rows; `rows_per_token` bounds
    /// a prompt token's rows.
    pub fn new(table: MappedTable, shares: [usize; 2], rows_per_token: usize) -> Result<Arc<Self>> {
        let cache = RowCache::new(table.row_bytes(), shares, table.shared_stats())?;
        Ok(Arc::new(Self { table: Arc::new(table), cache, rows_per_token, #[cfg(test)] placement: None }))
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
    /// A demand gather of the first table reads ahead of look-ahead reads.
    pub fn gather(&self, ids: &[u64], output: &mut [u8], priority: ReadPriority) -> Result<()> {
        let priority = priority.first();
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

    /// A remote-only table reads a prompt ahead only from the call after the one
    /// running: the running call's rows are left to its demand gathers.
    #[test]
    fn remote_tables_read_ahead_from_the_next_call() -> Result<()> {
        let (_file, table) = fixture(4096, 160, [4 * 512, 64])?;
        let mut owned = Arc::try_unwrap(table).ok().expect("sole owner");
        owned.placement = Some(super::super::placement::Placement::Remote);
        let table = Arc::new(owned);
        let ahead = TokenTableLookahead::new("test-remote", Arc::clone(&table))?;
        assert!(ahead.planner.remote());
        let hash = |t: &[u32], rows: &mut Vec<u64>| -> Result<()> {
            rows.extend(t.iter().flat_map(|&t| (0..4u64).map(move |h| u64::from(t) * 4 + h)));
            Ok(())
        };
        // Two calls of one unit: tokens 0..128 (running), 128..256 (next).
        let handle = ahead.start(1, 0, Box::new(HashedSource { tokens: (0..256).collect(), hash }),
            UnitPlan { ends: vec![128, 256], depth: 1, calls: vec![0, 1] });
        // Its prefill is running: call 0 belongs to its demand gathers.
        handle.running();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while table.stats().snapshot().lookahead_rows < 512 {
            assert!(std::time::Instant::now() < deadline, "the next call was not read ahead");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(table.stats().snapshot().lookahead_rows, 512, "only the next call's rows");
        let (pins, running) = table.cache().acquire(&(0..512).collect::<Vec<_>>(), RowClass::Decode);
        // (The decode share is small: past it rows come back Uncached; none is Ready or in flight.)
        let cached = running.iter().filter(|l| matches!(l, Lookup::Ready(_) | Lookup::Wait(_))).count();
        assert_eq!(cached, 0, "the running call's rows are not cached");
        drop(pins);
        let (pins, next) = table.cache().acquire(&(512..1024).collect::<Vec<_>>(), RowClass::Decode);
        assert!(next.iter().all(|l| matches!(l, Lookup::Ready(_))));
        drop(pins);
        assert!(handle.last_table_gate().load(std::sync::atomic::Ordering::Acquire) >= 128, "the skipped call's gate is open");
        // Decode rows still land.
        ahead.decode(vec![4000, 4001]);
        while table.stats().snapshot().lookahead_rows < 514 {
            assert!(std::time::Instant::now() < deadline, "decode rows did not land");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
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
            UnitPlan::new(vec![128, 256], 1));
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

    /// The look-ahead reserves a batch only as it submits its read: rows of a
    /// piece it has not reached yet are free for a demand gather to read itself.
    #[test]
    fn look_ahead_never_reserves_rows_ahead_of_its_reads() -> Result<()> {
        // 160 B rows: one batch is 2048 rows; a piece of 384 tokens x 16 rows is 6144.
        let (_file, table) = fixture(1 << 16, 160, [1 << 16, 1024])?;
        let ahead = TokenTableLookahead::new("test-ple", Arc::clone(&table))?;
        let hash = |t: &[u32], rows: &mut Vec<u64>| -> Result<()> {
            rows.extend(t.iter().flat_map(|&t| (0..16u64).map(move |h| u64::from(t) * 16 + h)));
            Ok(())
        };
        let tokens: Vec<u32> = (0..384).collect();
        let _handle = ahead.start(1, 0, Box::new(HashedSource { tokens, hash }), UnitPlan::new(vec![384], 1));
        // Whatever the planner has done so far, at most one batch per in-flight read is Filling.
        for _ in 0..50 {
            let rows: Vec<u64> = (0..384 * 16).collect();
            let (pins, lookups) = table.cache().acquire(&rows, RowClass::Decode);
            let waiting = lookups.iter().filter(|l| matches!(l, Lookup::Wait(_))).count();
            assert!(waiting <= 3 * 2048, "{waiting} rows reserved but not being read");
            drop(pins);
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        Ok(())
    }

    /// Prefill gathers from a staged token table (STAGE_DIR: every 2-D tensor of
    /// STAGE_ROW_BYTES-byte rows in its safetensors files, in file and tensor
    /// order), STAGE_ROWS_PER_TOKEN rows a token (PLE 16 x 160 B, strided Engram
    /// 24 x 264 B). One prompt of BENCH_TOKENS (default 8192) random tokens in
    /// units of 2752, 3 deep, BENCH_COMPUTE_MS (default 30) between units.
    /// BENCH_ARM `direct` reads each unit's rows at gather time (the dev path);
    /// `cached` adds the row cache and look-ahead from the prompt's start, its
    /// profile from LOOKAHEAD_BENCH_INFLIGHT / LOOKAHEAD_BENCH_PIECE. BENCH_DROP
    /// runs first (a true drop, e.g. `nest drop-caches`). Prints wall, gather time,
    /// device reads, mean service per read and reads in flight.
    #[test]
    #[ignore = "CPU/IO microbench on staged tables, run in release"]
    fn staged_table_gather_microbench() -> Result<()> {
        let Some(dir) = std::env::var_os("STAGE_DIR") else { return Ok(()) };
        let env = |name: &str, default: usize| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
        let (row_bytes, per_token) = (env("STAGE_ROW_BYTES", 160), env("STAGE_ROWS_PER_TOKEN", 16));
        let mut files: Vec<_> = std::fs::read_dir(&dir)?.filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect();
        files.sort();
        let mut parts = Vec::new();
        for file in &files {
            let mut tensors = crate::read_safetensors_metadata(file)?;
            tensors.retain(|t| t.shape.len() == 2 && t.byte_length == (t.shape[0] * row_bytes) as u64);
            tensors.sort_by_key(|t| t.byte_offset);
            parts.extend(tensors.iter().map(|t| super::super::TablePart { path: file.clone(), offset: t.byte_offset, rows: t.shape[0] as u64 }));
        }
        anyhow::ensure!(!parts.is_empty(), "no {row_bytes}-byte-row tensors under {dir:?}");
        let (unit, depth) = (2752usize, 3usize);
        let tokens = env("BENCH_TOKENS", 8192);
        let compute = std::time::Duration::from_millis(env("BENCH_COMPUTE_MS", 30) as u64);
        let arm = std::env::var("BENCH_ARM").unwrap_or_else(|_| "cached".into());
        // SAFETY: the staged files are sealed and immutable for the bench's lifetime.
        let table = unsafe { MappedTable::open(&parts, RowFormat { dtype: cuteafd_core::DType::U8, width: row_bytes, row_bytes })? };
        table.prepare_reads();
        let table = TokenTable::new(table, TokenTable::shares(unit, depth, 64, per_token), per_token)?;
        let rows = table.table().rows();
        let mut seed = env("BENCH_SEED", 0x9E37_79B9) as u64;
        let ids: Vec<u64> = (0..tokens * per_token).map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed % rows }).collect();
        let ends: Vec<usize> = (1..=tokens.div_ceil(unit)).map(|u| (u * unit).min(tokens)).collect();
        if let Ok(command) = std::env::var("BENCH_DROP") {
            let status = std::process::Command::new("sh").arg("-c").arg(&command).status()?;
            anyhow::ensure!(status.success(), "drop command failed: {command}");
        }
        let ahead = TokenTableLookahead::new("bench-table", Arc::clone(&table))?;
        let before = table.stats().snapshot();
        let started = std::time::Instant::now();
        let handle = (arm == "cached").then(|| {
            let (rows, mut next) = (ids.clone(), 0usize);
            let hash = move |t: &[u32], out: &mut Vec<u64>| -> Result<()> {
                out.extend_from_slice(&rows[next * per_token..(next + t.len()) * per_token]);
                next += t.len();
                Ok(())
            };
            ahead.start(0, 0, Box::new(HashedSource { tokens: vec![0; tokens], hash }),
                UnitPlan::new(ends.iter().map(|&e| e as u64).collect(), depth))
        });
        let (mut gather, mut at) = (0f64, 0);
        let mut out = vec![0u8; unit * per_token * row_bytes];
        for &end in &ends {
            let unit_ids = &ids[at * per_token..end * per_token];
            let bytes = &mut out[..unit_ids.len() * row_bytes];
            let t = std::time::Instant::now();
            if arm == "cached" { table.gather(unit_ids, bytes, ReadPriority::Prefill)?; }
            else { table.table().gather_into_priority(unit_ids, bytes, ReadPriority::PrefillFirst)?; }
            gather += t.elapsed().as_secs_f64() * 1e3;
            if let Some(handle) = &handle { handle.commit(end as u64); }
            std::thread::sleep(compute);
            at = end;
        }
        let wall = started.elapsed().as_secs_f64();
        drop(handle);
        let d = table.stats().snapshot().since(&before);
        println!("RESULT dir={} arm={arm} tokens={tokens} inflight={} piece={} wall_ms={:.0} gather_ms={:.0} device_reads={} \
            service_ms_per_read={:.2} reads_in_flight={:.1} hits={} fallback={} pending={} lookahead={}",
            std::path::Path::new(&dir).file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned()),
            std::env::var("LOOKAHEAD_BENCH_INFLIGHT").unwrap_or_else(|_| "3".into()),
            std::env::var("LOOKAHEAD_BENCH_PIECE").unwrap_or_else(|_| "auto".into()),
            wall * 1e3, gather, d.page_misses, d.miss_ns as f64 / 1e6 / d.page_misses.max(1) as f64,
            d.miss_ns as f64 / 1e9 / wall, d.cache_hit_rows, d.stall_prefill_fallback_rows, d.stall_prefill_pending_rows,
            d.lookahead_rows);
        Ok(())
    }
}
