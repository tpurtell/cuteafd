//! Engram look-ahead: as soon as a prefill's prompt is known, read its rows into
//! each layer's row cache ahead of the gathers, bounded by the cache.
//!
//! One planner thread hashes forward from each request's frontier (a forked
//! history: nothing is committed), reserves the rows of the next piece in the
//! layers' caches and starts their reads in deadline order: layer 1 runs ahead,
//! and a piece's layer-14 reads start only once every layer-1 row through one
//! prefill call ([`L14_LAG_TOKENS`]) past it has landed, whoever read it, so
//! layer-14 reads never queue at the device ahead of an earlier layer-1 deadline.
//! Reads: resident rows are copied on this
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
/// Layer reads (one piece, one layer) in flight at once; each holds up to one
/// reactor buffer per table (6 of the reactor's 14 prefill buffers), leaving
/// buffers for the gathers.
const MAX_INFLIGHT: usize = 3;
/// A piece's layer-14 rows wait until layer 1 has landed this far past it: one
/// full prefill call, so a unit's layer-14 reads follow its call's layer-1 reads.
const L14_LAG_TOKENS: u64 = crate::serving_capacity::prefill_waves::RESERVE_ROWS;

enum Command {
    Start { id: u64, history: EngramHistory, tokens: Vec<Option<u32>>, committed: Arc<AtomicU64> },
    /// A decode round's drafted rows per layer: read into the decode share now,
    /// published and left evictable for the verify gather.
    Decode([Vec<u64>; 2]),
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

/// One prefill piece: its layer-1 pins and read, and its layer-14 rows until
/// their deadline turn (then their pins and read).
struct PlanPiece {
    end: u64,
    l1: Pins,
    l1_read: Option<(RowRead, Vec<u32>)>,
    l14_rows: Option<Vec<u64>>,
    l14: Option<Pins>,
    l14_read: Option<(RowRead, Vec<u32>)>,
}
impl PlanPiece {
    fn reading(&self) -> usize { usize::from(self.l1_read.is_some()) + usize::from(self.l14_read.is_some()) }
    fn done(&self) -> bool { self.reading() == 0 && self.l14_rows.is_none() }
}

struct Plan {
    id: u64,
    history: EngramHistory,
    tokens: Vec<Option<u32>>,
    /// Absolute position of `tokens[0]`.
    start: u64,
    /// Index of the next token to plan.
    next: usize,
    committed: Arc<AtomicU64>,
    /// Planned pieces, oldest first, until commits pass them.
    pieces: VecDeque<PlanPiece>,
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

    /// Read a decode round's drafted rows (per layer) into the decode share the
    /// moment the drafter emits them; the verify gather then finds them landed
    /// or in flight. Never blocks.
    pub fn decode(&self, rows: [Vec<u64>; 2]) {
        let _ = self.send.send(Command::Decode(rows));
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
    let mut decode: VecDeque<Piece> = VecDeque::new();
    loop {
        // Apply every queued command without waiting; decode rows start at once.
        loop {
            match receive.try_recv() {
                Ok(Command::Stop) | Err(mpsc::TryRecvError::Disconnected) => return,
                Ok(Command::Decode(rows)) => match start_decode(&rows, &tables) {
                    Ok(Some(piece)) => decode.push_back(piece),
                    Ok(None) => {}
                    Err(error) => tracing::warn!(%error, "Engram draft look-ahead read failed; rows load on demand"),
                },
                Ok(command) => apply(&mut plans, command),
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }
        let mut progressed = false;
        // Published decode rows stay cached (evictable); their pins drop here.
        for piece in &mut decode {
            for (layer, table) in tables.iter().enumerate() {
                if let Some((read, _)) = piece.reads[layer].as_mut() {
                    match read.poll(table) {
                        Ok(true) => { if publish(piece, layer, table).is_ok() { progressed = true; } }
                        Ok(false) => {}
                        Err(error) => {
                            tracing::warn!(%error, "Engram draft look-ahead read failed; rows load on demand");
                            piece.reads[layer] = None;
                        }
                    }
                }
            }
        }
        decode.retain(|piece| piece.reads.iter().any(Option::is_some));
        for plan in &mut plans {
            let committed = plan.committed.load(Ordering::Acquire);
            // Committed pieces release their rows; unread layer-14 rows are no longer needed.
            while plan.pieces.front().is_some_and(|piece| piece.end <= committed && piece.reading() == 0) {
                plan.pieces.pop_front();
            }
            match poll(plan, &tables) {
                Ok(landed) => progressed |= landed,
                Err(error) => { abandon(plan, &error); continue; }
            }
        }
        let reading = |plans: &[Plan]| plans.iter().flat_map(|plan| &plan.pieces).map(PlanPiece::reading).sum::<usize>();
        // Layer-14 reads that are due go first: their deadline is earlier than
        // any layer-1 piece the cursor has not started.
        while reading(&plans) < MAX_INFLIGHT {
            let Some((plan, index)) = plans.iter_mut().find_map(|plan| due_l14(plan, &tables).map(|index| (plan, index)))
                else { break };
            if let Err(error) = start_l14(plan, index, &tables) { abandon(plan, &error); }
            progressed = true;
        }
        if reading(&plans) < MAX_INFLIGHT {
            // Start pieces for the request closest to needing them first.
            if let Some(plan) = plans.iter_mut().filter(|plan| ready_to_plan(plan, &tables))
                .min_by_key(|plan| plan.start + plan.next as u64 - plan.committed.load(Ordering::Acquire)) {
                match start_piece(plan, &tables) {
                    Ok(()) => progressed = true,
                    Err(error) => abandon(plan, &error),
                }
            }
        }
        plans.retain(|plan| plan.next < plan.tokens.len() || !plan.pieces.is_empty());
        if progressed { continue; }
        if !decode.is_empty() {
            // Decode reads are short: poll them again soon rather than block on a prefill piece.
            std::thread::sleep(std::time::Duration::from_micros(50));
            continue;
        }
        // Layer-14 rows waiting on layer-1 rows a gather is reading: check again shortly.
        let waiting = plans.iter().any(|plan| plan.pieces.iter().any(|piece| piece.l14_rows.is_some()));
        let oldest = plans.iter_mut().find(|plan| plan.pieces.iter().any(|piece| piece.reading() > 0));
        match oldest {
            Some(plan) => if let Err(error) = land_oldest(plan, &tables) { abandon(plan, &error) },
            None if waiting => match receive.recv_timeout(std::time::Duration::from_millis(1)) {
                Ok(Command::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => return,
                Ok(Command::Decode(rows)) => match start_decode(&rows, &tables) {
                    Ok(Some(piece)) => decode.push_back(piece),
                    Ok(None) => {}
                    Err(error) => tracing::warn!(%error, "Engram draft look-ahead read failed; rows load on demand"),
                },
                Ok(command) => apply(&mut plans, command),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            },
            None => match receive.recv() {
                Ok(Command::Stop) | Err(_) => return,
                Ok(Command::Decode(rows)) => match start_decode(&rows, &tables) {
                    Ok(Some(piece)) => decode.push_back(piece),
                    Ok(None) => {}
                    Err(error) => tracing::warn!(%error, "Engram draft look-ahead read failed; rows load on demand"),
                },
                Ok(command) => apply(&mut plans, command),
            },
        }
    }
}

fn apply(plans: &mut Vec<Plan>, command: Command) {
    match command {
        Command::Start { id, history, tokens, committed } => {
            let start = history.position();
            plans.push(Plan { id, history, tokens, start, next: 0, committed, pieces: VecDeque::new() });
        }
        Command::Wake | Command::Stop | Command::Decode(_) => {}
        // Dropping the plan drains its reads, then fails unread rows and unpins.
        Command::Cancel(id) => plans.retain(|plan| plan.id != id),
    }
}

fn abandon(plan: &mut Plan, error: &anyhow::Error) {
    tracing::warn!(%error, "Engram look-ahead stopped for a request; its rows load on demand");
    plan.next = plan.tokens.len();
    plan.pieces.clear();
}

/// More tokens, inside the window past the committed frontier, and cache room
/// for a whole piece in both layers (layer 14's rows follow later).
fn ready_to_plan(plan: &Plan, tables: &[Arc<EngramTable>; 2]) -> bool {
    let position = plan.start + plan.next as u64;
    plan.next < plan.tokens.len()
        && position < plan.committed.load(Ordering::Acquire) + AHEAD_TOKENS as u64
        && tables.iter().all(|table| table.cache().room(RowClass::Prefill) >= 2 * PIECE_TOKENS * 24)
}

/// The oldest piece whose layer-14 rows are due: every layer-1 row through
/// [`L14_LAG_TOKENS`] past it (or the prompt's end) has landed.
fn due_l14(plan: &Plan, tables: &[Arc<EngramTable>; 2]) -> Option<usize> {
    let index = plan.pieces.iter().position(|piece| piece.l14_rows.is_some())?;
    let horizon = plan.pieces[index].end + L14_LAG_TOKENS;
    let planned_end = plan.start + plan.next as u64;
    let prompt_end = plan.start + plan.tokens.len() as u64;
    // Layer 1 stops short of the horizon at the window or the cache's room:
    // then the layer-1 rows planned so far are the ones due first.
    if planned_end < horizon.min(prompt_end) && ready_to_plan(plan, tables) { return None; }
    // Pieces starting before the horizon carry the layer-1 rows that are due first.
    plan.pieces.iter().filter(|piece| piece.end.saturating_sub(PIECE_TOKENS as u64) < horizon)
        .all(|piece| piece.l1_read.is_none() && piece.l1.settled()).then_some(index)
}

fn reserve(table: &EngramTable, rows: &[u64], class: RowClass, priority: ReadPriority)
    -> Result<(Pins, Option<(RowRead, Vec<u32>)>)> {
    let mut rows = rows.to_vec();
    rows.sort_unstable();
    rows.dedup();
    let (held, lookups) = table.cache().acquire(&rows, class);
    let (mut ids, mut slots) = (Vec::new(), Vec::new());
    for (lookup, &row) in lookups.iter().zip(&rows) {
        if let Lookup::Fill(slot) = *lookup { ids.push(row); slots.push(slot); }
    }
    let read = if ids.is_empty() { None } else { Some((table.start_read(&ids, priority)?, slots)) };
    Ok((held, read))
}

fn start_piece(plan: &mut Plan, tables: &[Arc<EngramTable>; 2]) -> Result<()> {
    let end = (plan.next + PIECE_TOKENS).min(plan.tokens.len());
    let batch = plan.history.prepare(plan.history.position(), &plan.tokens[plan.next..end], PIECE_TOKENS)?;
    plan.history.commit(&batch, end - plan.next)?;
    plan.next = end;
    let (l1, l1_read) = reserve(&tables[0], &batch.prefetch_rows(0)?, RowClass::Prefill, ReadPriority::PrefillFirst)?;
    plan.pieces.push_back(PlanPiece { end: plan.start + end as u64, l1, l1_read,
        l14_rows: Some(batch.prefetch_rows(1)?), l14: None, l14_read: None });
    Ok(())
}

fn start_l14(plan: &mut Plan, index: usize, tables: &[Arc<EngramTable>; 2]) -> Result<()> {
    let piece = &mut plan.pieces[index];
    let rows = piece.l14_rows.take().expect("due layer-14 rows");
    let (pins, read) = reserve(&tables[1], &rows, RowClass::Prefill, ReadPriority::Prefill)?;
    piece.l14 = Some(pins);
    piece.l14_read = read;
    Ok(())
}

/// Reserve a round's drafted rows in the decode share and start reading the
/// ones nobody holds; None when every row is already cached or in flight.
fn start_decode(rows: &[Vec<u64>; 2], tables: &[Arc<EngramTable>; 2]) -> Result<Option<Piece>> {
    let mut pins = Vec::with_capacity(2);
    let mut reads = [None, None];
    for (layer, table) in tables.iter().enumerate() {
        let mut rows = rows[layer].clone();
        rows.sort_unstable();
        rows.dedup();
        let (held, lookups) = table.cache().acquire(&rows, RowClass::Decode);
        let (mut ids, mut slots) = (Vec::new(), Vec::new());
        for (lookup, &row) in lookups.iter().zip(&rows) {
            if let Lookup::Fill(slot) = *lookup { ids.push(row); slots.push(slot); }
        }
        if !ids.is_empty() {
            let priority = if layer == 0 { ReadPriority::DecodeFirst } else { ReadPriority::Decode };
            reads[layer] = Some((table.start_read(&ids, priority)?, slots));
        }
        pins.push(held);
    }
    if reads.iter().all(Option::is_none) { return Ok(None); }
    let [first, second]: [Pins; 2] = pins.try_into().map_err(|_| anyhow::anyhow!("two engram layers"))?;
    Ok(Some(Piece { end: 0, pins: [first, second], reads }))
}

/// Publish every read that has landed, without waiting.
fn poll(plan: &mut Plan, tables: &[Arc<EngramTable>; 2]) -> Result<bool> {
    let mut landed = false;
    for piece in &mut plan.pieces {
        if let Some((read, _)) = piece.l1_read.as_mut() {
            if read.poll(&tables[0])? {
                publish_read(&mut piece.l1, piece.l1_read.take().expect("landed"), &tables[0])?;
                landed = true;
            }
        }
        if let Some((read, _)) = piece.l14_read.as_mut() {
            if read.poll(&tables[1])? {
                let pins = piece.l14.as_mut().expect("layer-14 pins");
                publish_read(pins, piece.l14_read.take().expect("landed"), &tables[1])?;
                landed = true;
            }
        }
    }
    Ok(landed)
}

/// Wait for the oldest outstanding read (layer 1 first) and publish it.
fn land_oldest(plan: &mut Plan, tables: &[Arc<EngramTable>; 2]) -> Result<()> {
    let piece = plan.pieces.iter_mut().find(|piece| piece.reading() > 0).expect("an in-flight read");
    if let Some((mut read, slots)) = piece.l1_read.take() {
        read.finish(&tables[0])?;
        publish_read(&mut piece.l1, (read, slots), &tables[0])?;
    } else if let Some((mut read, slots)) = piece.l14_read.take() {
        read.finish(&tables[1])?;
        publish_read(piece.l14.as_mut().expect("layer-14 pins"), (read, slots), &tables[1])?;
    }
    Ok(())
}

fn publish_read(pins: &mut Pins, (read, slots): (RowRead, Vec<u32>), table: &EngramTable) -> Result<()> {
    let rows = pins.publish_all(slots.iter().enumerate().map(|(i, &slot)| (slot, read.row(i))))?;
    table.weights().stats().record_lookahead(rows);
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
    fn layer_14_waits_for_layer_1_through_the_lag() -> Result<()> {
        // A prompt longer than the lag: piece 0's layer 14 is read only after
        // layer 1 has landed through one prefill call past it.
        let tokens: Vec<Option<u32>> = (0..(L14_LAG_TOKENS as u32 + 512)).map(|i| Some(i % 90_000)).collect();
        let history = EngramHistory::new(2)?;
        let mut owned = [table(0, &[])?, table(1, &[])?].map(|t| Arc::try_unwrap(t).ok().expect("sole owner"));
        for table in &mut owned { table.set_cache_shares([AHEAD_TOKENS * 24, 1024])?; }
        let tables = owned.map(Arc::new);
        let ahead = EngramLookahead::new(tables.clone())?;
        let _handle = ahead.start(history.fork()?, tokens);
        let rows = |layer: usize| tables[layer].weights().stats().snapshot().lookahead_rows;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while rows(1) == 0 {
            assert!(std::time::Instant::now() < deadline, "layer 14 never started");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        // Layer 1 had landed at least one lag's worth of tokens before any layer-14 row.
        let lag_pieces = (L14_LAG_TOKENS as usize).div_ceil(PIECE_TOKENS) as u64;
        assert!(tables[0].weights().stats().snapshot().lookahead_pieces >= lag_pieces);
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
