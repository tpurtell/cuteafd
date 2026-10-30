//! Engram look-ahead: as soon as a prefill's prompt is known, read its rows into
//! each layer's row cache ahead of the gathers, bounded by the cache.
//!
//! One planner thread hashes forward from each request's frontier (a forked
//! history: nothing is committed), reserves the rows of the next piece in both
//! layers' caches and starts their reads: resident rows are copied on this
//! thread, misses go to the process-wide read reactor, so the reactor is the
//! I/O concurrency. Completed pieces are published and stay pinned until the
//! request's commits pass them; the look-ahead never runs further than
//! [`AHEAD_TOKENS`] past the committed frontier or past what the cache admits.
//! Rows a gather already reserved are pinned, not read again: whoever reserves
//! a row first reads it, so the look-ahead and the gathers share one fetch.
use super::engram_prefetch::RowRead;
use crate::formats::mapped_table::row_cache::{Lookup, Pins, RowClass};
use crate::formats::mapped_table::ReadPriority;
use crate::EngramTable;
use anyhow::{Context, Result};
use cuteafd_core::EngramHistory;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

/// Two full prefill reserves past the committed frontier (TJ, 2026-10-30).
pub const AHEAD_TOKENS: usize = 2 * crate::serving_capacity::prefill_waves::RESERVE_ROWS as usize;
/// Tokens per piece: both layers' rows of a piece fit one reactor buffer per table.
const PIECE_TOKENS: usize = 128;
/// Pieces reading at once. Each holds up to one reactor buffer per table and
/// layer (4 of the reactor's 14 prefill buffers), which keeps the reactor's
/// queue full while leaving buffers for the gathers.
const MAX_INFLIGHT: usize = 2;

enum Command {
    Start { id: u64, history: EngramHistory, tokens: Vec<Option<u32>>, committed: Arc<AtomicU64> },
    Wake,
    Cancel(u64),
    Stop,
}

/// A request's look-ahead. Publish commits with [`Self::commit`]; dropping it
/// stops the look-ahead and releases its rows.
pub struct LookaheadHandle {
    id: u64,
    end: u64,
    committed: Arc<AtomicU64>,
    send: mpsc::Sender<Command>,
}

impl LookaheadHandle {
    /// The request committed through `position`; false once that is the prompt's end.
    pub fn commit(&self, position: u64) -> bool {
        self.committed.fetch_max(position, Ordering::Release);
        let _ = self.send.send(Command::Wake);
        position < self.end
    }
}

impl Drop for LookaheadHandle {
    fn drop(&mut self) { let _ = self.send.send(Command::Cancel(self.id)); }
}

struct Piece { end: u64, pins: [Pins; 2], reads: [Option<(RowRead, Vec<u32>)>; 2] }

struct Plan {
    id: u64,
    history: EngramHistory,
    tokens: Vec<Option<u32>>,
    /// Absolute position of `tokens[0]`.
    start: u64,
    /// Index of the next token to plan.
    next: usize,
    committed: Arc<AtomicU64>,
    inflight: VecDeque<Piece>,
    held: VecDeque<Piece>,
}

/// The planner thread; dropping it stops the look-ahead and releases every row.
pub struct EngramLookahead {
    send: mpsc::Sender<Command>,
    ids: AtomicU64,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl EngramLookahead {
    pub fn new(tables: [Arc<EngramTable>; 2]) -> Result<Self> {
        let (send, receive) = mpsc::channel();
        let thread = std::thread::Builder::new().name("engram-lookahead".into())
            .spawn(move || run(tables, receive)).context("starting the Engram look-ahead")?;
        Ok(Self { send, ids: AtomicU64::new(1), thread: Some(thread) })
    }

    /// Read ahead `tokens` (compressed ids, None for image rows) that follow
    /// `history`'s frontier. `history` is a fork: its batches are never committed.
    pub fn start(&self, history: EngramHistory, tokens: Vec<Option<u32>>) -> LookaheadHandle {
        let id = self.ids.fetch_add(1, Ordering::Relaxed);
        let committed = Arc::new(AtomicU64::new(history.position()));
        let end = history.position() + tokens.len() as u64;
        let _ = self.send.send(Command::Start { id, history, tokens, committed: Arc::clone(&committed) });
        LookaheadHandle { id, end, committed, send: self.send.clone() }
    }
}

impl Drop for EngramLookahead {
    fn drop(&mut self) {
        // Handles may outlive the pipeline: stop explicitly. The planner drains
        // its reads and fails unread rows before it exits.
        let _ = self.send.send(Command::Stop);
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

fn run(tables: [Arc<EngramTable>; 2], receive: mpsc::Receiver<Command>) {
    let mut plans: Vec<Plan> = Vec::new();
    loop {
        // Apply every queued command without waiting.
        loop {
            match receive.try_recv() {
                Ok(Command::Stop) | Err(mpsc::TryRecvError::Disconnected) => return,
                Ok(command) => apply(&mut plans, command),
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }
        let mut progressed = false;
        for plan in &mut plans {
            let committed = plan.committed.load(Ordering::Acquire);
            while plan.held.front().is_some_and(|piece| piece.end <= committed) {
                plan.held.pop_front();
            }
            match poll(plan, &tables) {
                Ok(landed) => progressed |= landed,
                Err(error) => { abandon(plan, &error); continue; }
            }
        }
        let inflight: usize = plans.iter().map(|plan| plan.inflight.len()).sum();
        if inflight < MAX_INFLIGHT {
            // Start pieces for the request closest to needing them first.
            if let Some(plan) = plans.iter_mut().filter(|plan| ready_to_plan(plan, &tables))
                .min_by_key(|plan| plan.start + plan.next as u64 - plan.committed.load(Ordering::Acquire)) {
                match start_piece(plan, &tables) {
                    Ok(()) => progressed = true,
                    Err(error) => abandon(plan, &error),
                }
            }
        }
        plans.retain(|plan| plan.next < plan.tokens.len() || !plan.inflight.is_empty() || !plan.held.is_empty());
        if progressed { continue; }
        // Nothing to start: wait for the oldest read, or for a command.
        let oldest = plans.iter_mut().find(|plan| !plan.inflight.is_empty());
        match oldest {
            Some(plan) => if let Err(error) = land_oldest(plan, &tables) { abandon(plan, &error) },
            None => match receive.recv() {
                Ok(Command::Stop) | Err(_) => return,
                Ok(command) => apply(&mut plans, command),
            },
        }
    }
}

fn apply(plans: &mut Vec<Plan>, command: Command) {
    match command {
        Command::Start { id, history, tokens, committed } => {
            let start = history.position();
            plans.push(Plan { id, history, tokens, start, next: 0, committed, inflight: VecDeque::new(),
                held: VecDeque::new() });
        }
        Command::Wake | Command::Stop => {}
        // Dropping the plan drains its reads, then fails unread rows and unpins.
        Command::Cancel(id) => plans.retain(|plan| plan.id != id),
    }
}

fn abandon(plan: &mut Plan, error: &anyhow::Error) {
    tracing::warn!(%error, "Engram look-ahead stopped for a request; its rows load on demand");
    plan.next = plan.tokens.len();
    plan.inflight.clear();
}

/// More tokens, inside the window past the committed frontier, and cache room
/// for a whole piece in both layers.
fn ready_to_plan(plan: &Plan, tables: &[Arc<EngramTable>; 2]) -> bool {
    let position = plan.start + plan.next as u64;
    plan.next < plan.tokens.len()
        && position < plan.committed.load(Ordering::Acquire) + AHEAD_TOKENS as u64
        && tables.iter().all(|table| table.cache().room(RowClass::Prefill) >= PIECE_TOKENS * 24)
}

fn start_piece(plan: &mut Plan, tables: &[Arc<EngramTable>; 2]) -> Result<()> {
    let end = (plan.next + PIECE_TOKENS).min(plan.tokens.len());
    let batch = plan.history.prepare(plan.history.position(), &plan.tokens[plan.next..end], PIECE_TOKENS)?;
    plan.history.commit(&batch, end - plan.next)?;
    plan.next = end;
    let mut pins = Vec::with_capacity(2);
    let mut reads = [None, None];
    for (layer, table) in tables.iter().enumerate() {
        let mut rows = batch.prefetch_rows(layer)?;
        rows.sort_unstable();
        rows.dedup();
        let (held, lookups) = table.cache().acquire(&rows, RowClass::Prefill);
        let mut ids = Vec::new();
        let mut slots = Vec::new();
        for (lookup, &row) in lookups.iter().zip(&rows) {
            if let Lookup::Fill(slot) = *lookup { ids.push(row); slots.push(slot); }
        }
        if !ids.is_empty() {
            reads[layer] = Some((table.start_read(&ids, ReadPriority::Prefill)?, slots));
        }
        pins.push(held);
    }
    let [first, second]: [Pins; 2] = pins.try_into().map_err(|_| anyhow::anyhow!("two engram layers"))?;
    plan.inflight.push_back(Piece { end: plan.start + end as u64, pins: [first, second], reads });
    Ok(())
}

/// Publish every read that has landed, in order, without waiting.
fn poll(plan: &mut Plan, tables: &[Arc<EngramTable>; 2]) -> Result<bool> {
    let mut landed = false;
    for piece in &mut plan.inflight {
        for (layer, table) in tables.iter().enumerate() {
            if let Some((read, _)) = piece.reads[layer].as_mut() {
                if read.poll(table)? { publish(piece, layer, table)?; landed = true; }
            }
        }
    }
    while plan.inflight.front().is_some_and(|piece| piece.reads.iter().all(Option::is_none)) {
        let piece = plan.inflight.pop_front().expect("checked front");
        plan.held.push_back(piece);
    }
    Ok(landed)
}

fn land_oldest(plan: &mut Plan, tables: &[Arc<EngramTable>; 2]) -> Result<()> {
    let piece = plan.inflight.front_mut().expect("an in-flight piece");
    for (layer, table) in tables.iter().enumerate() {
        if let Some((read, _)) = piece.reads[layer].as_mut() {
            read.finish(table)?;
            publish(piece, layer, table)?;
        }
    }
    let piece = plan.inflight.pop_front().expect("checked front");
    plan.held.push_back(piece);
    Ok(())
}

fn publish(piece: &mut Piece, layer: usize, table: &EngramTable) -> Result<()> {
    let (read, slots) = piece.reads[layer].take().expect("a landed read");
    let rows = piece.pins[layer].publish_all(slots.iter().enumerate().map(|(i, &slot)| (slot, read.row(i))))?;
    table.weights().stats().record_lookahead(rows);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let handle = ahead.start(history.fork()?, tokens.clone());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while tables.iter().any(|t| t.weights().stats().snapshot().lookahead_pieces < 3) {
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
        // A second request over the same prompt pins the cached rows, reading none.
        let stale = ahead.start(history.fork()?, tokens.clone());
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(tables[0].weights().stats().snapshot().lookahead_pieces, 3);
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
    fn look_ahead_stays_inside_the_cache_and_window() -> Result<()> {
        // A prefill share of one piece: the second piece waits for a commit.
        let tokens: Vec<Option<u32>> = (0..256).map(|i| Some(i + 5)).collect();
        let history = EngramHistory::new(2)?;
        let tables = [table(0, &[])?, table(1, &[])?];
        let mut owned = tables.map(|t| Arc::try_unwrap(t).ok().expect("sole owner"));
        for table in &mut owned { table.set_cache_shares([PIECE_TOKENS * 24, 64])?; }
        let tables = owned.map(Arc::new);
        let ahead = EngramLookahead::new(tables.clone())?;
        let handle = ahead.start(history.fork()?, tokens);
        let pieces = || tables[0].weights().stats().snapshot().lookahead_pieces;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while pieces() < 1 { assert!(std::time::Instant::now() < deadline); std::thread::sleep(std::time::Duration::from_millis(5)); }
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(pieces(), 1, "no room until the first piece commits");
        assert!(handle.commit(128));
        while pieces() < 2 { assert!(std::time::Instant::now() < deadline); std::thread::sleep(std::time::Duration::from_millis(5)); }
        Ok(())
    }
}
