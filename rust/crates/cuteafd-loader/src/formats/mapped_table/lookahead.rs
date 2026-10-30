//! Token-table look-ahead: as soon as a prefill's prompt is known, read its rows
//! into each table's bounded row cache ahead of the gathers.
//!
//! A family supplies the tables (one per consuming layer, in layer order) and,
//! per request, a [`RowSource`] that turns prompt token ranges into each table's
//! rows (V4.1 Engram n-gram hashes over a forked history, Qwen PLE n-gram ids).
//! One planner thread reads pieces aligned to the request's prefill units in the
//! units' deadline order. Units run in pipeline order (each one layer behind the
//! previous, a new one starting when the oldest commits), so with depth `d` the
//! deadlines are table 0 of the `d` units in flight, then table 1 of the oldest,
//! and so on: table `k + 1` of unit `u` is read once table `k` has landed through
//! unit `u + d - 1` (whoever read it). The look-ahead never runs past two full
//! prefills ([`window_units`]) beyond the committed unit or past what the caches
//! admit. Across requests it follows admission order, which is execution order.
//! Reads go through each table's mapped-table reader (copy first, then the
//! residency split and the shared read reactor) and never block for a reactor
//! buffer: the planner polls. Pieces stay pinned until commits pass them.
//! Whoever reserves a row first reads it, so the look-ahead, the gathers and
//! decode share one fetch.
use super::row_cache::{Lookup, Pins, RowCache, RowClass};
use super::{ReadPriority, TableStats};
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

/// Units the look-ahead runs past the request's committed unit: two full
/// prefills of `depth` units. Row-cache prefill shares are sized from the same
/// function (plus the unit in flight), so the two cannot drift.
pub const fn window_units(depth: usize) -> usize { 2 * depth }

/// Tokens per piece.
const PIECE_TOKENS: usize = 128;
/// Prefill table reads (one piece, one table) in flight at once. A read holds
/// one reactor buffer per plane (two for a paired weight and scale table) and
/// at most one decode piece runs; reads never wait for a buffer, so the planner
/// can never block holding buffers while a gather waits for one.
const MAX_INFLIGHT: usize = 3;

/// One table the look-ahead fills: its row cache and a reader.
pub trait LookaheadTable: Send + Sync + 'static {
    fn cache(&self) -> &Arc<RowCache>;
    /// Start reading `rows` (distinct) into private buffers, holding at most one
    /// reactor buffer per plane and never waiting for one (the planner polls).
    fn start_read(&self, rows: &[u64], priority: ReadPriority) -> Result<Box<dyn RowLanding>>;
    /// Where look-ahead rows are counted.
    fn stats(&self) -> &TableStats;
    /// Rows a prompt token reads from this table (an upper bound).
    fn rows_per_token(&self) -> usize;
}

/// A read in flight.
pub trait RowLanding: Send {
    /// Land whatever has completed without waiting; true once every row is here.
    fn poll(&mut self) -> Result<bool>;
    /// Row `index` as the cache stores it (one or two parts).
    fn row(&self, index: usize) -> [&[u8]; 2];
}

/// A request's prompt as table rows. `rows` is called once per piece, in order,
/// for consecutive token ranges from 0; it returns each table's rows.
pub trait RowSource: Send + 'static {
    fn tokens(&self) -> usize;
    fn rows(&mut self, tokens: Range<usize>) -> Result<Vec<Vec<u64>>>;
}

/// The prefill units a request's prompt runs in: absolute end positions, in
/// order, and the pipeline depth (units in flight at once).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitPlan { pub ends: Vec<u64>, pub depth: usize }

enum Command {
    Start { id: u64, tag: u64, start: u64, source: Box<dyn RowSource>, units: UnitPlan,
        committed: Arc<AtomicU64>, reserved: Arc<AtomicU64> },
    /// A decode round's rows per table: read into the decode shares now,
    /// published and left evictable for the verify gather.
    Decode(Vec<Vec<u64>>),
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
    reserved: Arc<AtomicU64>,
    send: mpsc::Sender<Command>,
}

impl LookaheadHandle {
    /// Absolute position through which the look-ahead has reserved the last
    /// table's rows (`u64::MAX` once it stopped): a family that admits its
    /// last-table gathers early does so only behind it, so both readers follow
    /// one deadline order.
    pub fn last_table_gate(&self) -> Arc<AtomicU64> { Arc::clone(&self.reserved) }
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

type Read = (Box<dyn RowLanding>, Vec<u32>);

/// A decode round's pins and reads, one per table.
struct DecodePiece { pins: Vec<Pins>, reads: Vec<Option<Read>> }

/// One prefill piece: per table, its rows until their deadline turn, then their
/// pins and read.
struct PlanPiece {
    end: u64,
    /// Index of the prefill unit this piece lies in.
    unit: usize,
    rows: Vec<Option<Vec<u64>>>,
    pins: Vec<Option<Pins>>,
    reads: Vec<Option<Read>>,
}
impl PlanPiece {
    fn reading(&self) -> usize { self.reads.iter().filter(|read| read.is_some()).count() }
    /// Table `table`'s rows have landed (read here or by another reader).
    fn landed(&self, table: usize) -> bool {
        self.rows[table].is_none() && self.reads[table].is_none() && self.pins[table].as_ref().is_none_or(Pins::settled)
    }
}

struct Plan {
    id: u64,
    /// The caller's request id, for the per-request summary.
    tag: u64,
    admitted: std::time::Instant,
    source: Box<dyn RowSource>,
    /// Absolute position of the source's token 0.
    start: u64,
    /// Index of the next token to plan.
    next: usize,
    units: UnitPlan,
    committed: Arc<AtomicU64>,
    reserved: Arc<AtomicU64>,
    /// Planned pieces, oldest first, until commits pass them.
    pieces: VecDeque<PlanPiece>,
    /// Per table: distinct rows planned, already cached, being filled by
    /// another reader, read here, and read before the first commit; rows of
    /// later tables committed before their turn (the gathers read them).
    needed: Vec<u64>,
    cached: Vec<u64>,
    joined: Vec<u64>,
    read: Vec<u64>,
    read_waiting: Vec<u64>,
    skipped: u64,
}

/// The planner thread; dropping it stops the look-ahead and releases every row.
pub struct Lookahead {
    send: mpsc::Sender<Command>,
    ids: AtomicU64,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Lookahead {
    /// `tables` in the order their consumers run within a unit.
    pub fn new(name: &str, tables: Vec<Arc<dyn LookaheadTable>>) -> Result<Self> {
        anyhow::ensure!(!tables.is_empty(), "a look-ahead needs at least one table");
        let (send, receive) = mpsc::channel();
        let thread = std::thread::Builder::new().name(format!("{name}-lookahead"))
            .spawn(move || run(tables, receive)).context("starting a token-table look-ahead")?;
        Ok(Self { send, ids: AtomicU64::new(1), thread: Some(thread) })
    }

    /// Read a decode round's rows (per table) into the decode shares as soon as
    /// they are known; the verify gather then finds them landed or in flight.
    /// A newer round's rows replace an unstarted older round's. Never blocks.
    pub fn decode(&self, rows: Vec<Vec<u64>>) {
        let _ = self.send.send(Command::Decode(rows));
    }

    /// Read ahead `source`'s tokens, which start at absolute position `start`.
    /// `units` are the prompt's prefill units (absolute ends) and their depth;
    /// `tag` is the caller's request id for the per-request summary.
    pub fn start(&self, tag: u64, start: u64, source: Box<dyn RowSource>, units: UnitPlan) -> LookaheadHandle {
        let id = self.ids.fetch_add(1, Ordering::Relaxed);
        let committed = Arc::new(AtomicU64::new(start));
        let end = start + source.tokens() as u64;
        let reserved = Arc::new(AtomicU64::new(start));
        let _ = self.send.send(Command::Start { id, tag, start, source, units, committed: Arc::clone(&committed),
            reserved: Arc::clone(&reserved) });
        LookaheadHandle { id, end, committed, reserved, send: self.send.clone() }
    }
}

impl Drop for Lookahead {
    fn drop(&mut self) {
        // Handles may outlive the look-ahead: stop explicitly. The planner drains
        // its reads and fails unread rows before it exits.
        let _ = self.send.send(Command::Stop);
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

fn run(tables: Vec<Arc<dyn LookaheadTable>>, receive: mpsc::Receiver<Command>) {
    let mut plans: Vec<Plan> = Vec::new();
    let mut decode: Option<DecodePiece> = None;
    let mut next_decode: Option<Vec<Vec<u64>>> = None;
    loop {
        loop {
            match receive.try_recv() {
                Ok(Command::Stop) | Err(mpsc::TryRecvError::Disconnected) => return,
                Ok(Command::Decode(rows)) => next_decode = Some(rows),
                Ok(command) => apply(&mut plans, command, tables.len()),
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }
        let mut progressed = false;
        // One decode piece at a time (its gather reads what has not landed).
        if decode.is_none() {
            if let Some(rows) = next_decode.take() {
                match start_decode(&rows, &tables) {
                    Ok(piece) => { progressed |= piece.is_some(); decode = piece; }
                    Err(error) => tracing::warn!(%error, "token-table decode look-ahead read failed; rows load on demand"),
                }
            }
        }
        if let Some(piece) = &mut decode {
            for (index, table) in tables.iter().enumerate() {
                if let Some((read, _)) = piece.reads[index].as_mut() {
                    match read.poll() {
                        Ok(true) => {
                            let read = piece.reads[index].take().expect("landed");
                            if publish(&mut piece.pins[index], read, table.as_ref()).is_ok() { progressed = true; }
                        }
                        Ok(false) => {}
                        Err(error) => {
                            tracing::warn!(%error, "token-table decode look-ahead read failed; rows load on demand");
                            piece.reads[index] = None;
                        }
                    }
                }
            }
            if piece.reads.iter().all(Option::is_none) { decode = None; }
        }
        for plan in &mut plans {
            let committed = plan.committed.load(Ordering::Acquire);
            // Committed pieces release their rows; unread later-table rows are no longer needed.
            while plan.pieces.front().is_some_and(|piece| piece.end <= committed && piece.reading() == 0) {
                let piece = plan.pieces.pop_front().expect("checked front");
                for rows in piece.rows.into_iter().flatten() {
                    plan.skipped += distinct(rows).len() as u64;
                }
            }
            match poll(plan, &tables) {
                Ok(landed) => progressed |= landed,
                Err(error) => { abandon(plan, &error); continue; }
            }
        }
        let reading = |plans: &[Plan]| plans.iter().flat_map(|plan| &plan.pieces).map(PlanPiece::reading).sum::<usize>();
        // Later-table reads that are due go first: their deadline is earlier
        // than any table-0 piece not yet started. `plans` is in admission order.
        while reading(&plans) < MAX_INFLIGHT {
            let Some((plan, index, table)) = plans.iter_mut()
                .find_map(|plan| due(plan, &tables).map(|(index, table)| (plan, index, table))) else { break };
            if let Err(error) = start_table(plan, index, table, &tables) { abandon(plan, &error); }
            progressed = true;
        }
        if reading(&plans) < MAX_INFLIGHT {
            // The earliest-admitted request with room in its window goes first.
            if let Some(plan) = plans.iter_mut().filter(|plan| ready_to_plan(plan, &tables)).min_by_key(|plan| plan.id) {
                match start_piece(plan, &tables) {
                    Ok(()) => progressed = true,
                    Err(error) => abandon(plan, &error),
                }
            }
        }
        plans.retain(|plan| plan.next < plan.source.tokens() || !plan.pieces.is_empty());
        if progressed { continue; }
        // Reads in flight, or later-table rows waiting on rows a gather is
        // reading: check again shortly, staying responsive to decode commands.
        let busy = decode.is_some() || plans.iter().any(|plan| plan.pieces.iter()
            .any(|piece| piece.reading() > 0 || piece.rows.iter().any(Option::is_some)));
        let command = if busy {
            match receive.recv_timeout(std::time::Duration::from_micros(if decode.is_some() { 50 } else { 200 })) {
                Ok(command) => Some(command),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        } else {
            match receive.recv() { Ok(command) => Some(command), Err(_) => return }
        };
        match command {
            Some(Command::Stop) => return,
            Some(Command::Decode(rows)) => next_decode = Some(rows),
            Some(command) => apply(&mut plans, command, tables.len()),
            None => {}
        }
    }
}

fn apply(plans: &mut Vec<Plan>, command: Command, tables: usize) {
    match command {
        Command::Start { id, tag, start, source, mut units, committed, reserved } => {
            let end = start + source.tokens() as u64;
            // Malformed or short plans degrade to one unit through the prompt's end.
            units.ends.retain(|&e| e > start && e < end);
            units.ends.push(end);
            units.depth = units.depth.max(1);
            plans.push(Plan { id, tag, admitted: std::time::Instant::now(), source, start, next: 0, units,
                committed, reserved, pieces: VecDeque::new(), needed: vec![0; tables], cached: vec![0; tables],
                joined: vec![0; tables], read: vec![0; tables], read_waiting: vec![0; tables], skipped: 0 });
        }
        Command::Wake | Command::Stop | Command::Decode(_) => {}
        // Dropping the plan drains its reads, then fails unread rows and unpins.
        Command::Cancel(id) => plans.retain(|plan| plan.id != id),
    }
}

fn abandon(plan: &mut Plan, error: &anyhow::Error) {
    tracing::warn!(%error, "token-table look-ahead stopped for a request; its rows load on demand");
    plan.next = plan.source.tokens();
    plan.pieces.clear();
    plan.reserved.store(u64::MAX, Ordering::Release);
}

fn distinct(mut rows: Vec<u64>) -> Vec<u64> {
    rows.sort_unstable();
    rows.dedup();
    rows
}

impl Plan {
    /// Index of the unit holding absolute position `position`.
    fn unit_of(&self, position: u64) -> usize { self.units.ends.partition_point(|&end| end <= position) }
    fn tally(&mut self, table: usize, [needed, cached, joined]: [u64; 3]) {
        self.needed[table] += needed;
        self.cached[table] += cached;
        self.joined[table] += joined;
    }
    fn count(&mut self, table: usize, rows: u64) {
        self.read[table] += rows;
        if self.committed.load(Ordering::Acquire) <= self.start { self.read_waiting[table] += rows; }
    }
}

/// More tokens, inside two full prefills past the committed unit, and cache room
/// for a whole piece in every table (later tables follow).
fn ready_to_plan(plan: &Plan, tables: &[Arc<dyn LookaheadTable>]) -> bool {
    let position = plan.start + plan.next as u64;
    let committed = plan.committed.load(Ordering::Acquire).max(plan.start);
    plan.next < plan.source.tokens()
        && plan.unit_of(position) < plan.unit_of(committed) + window_units(plan.units.depth)
        && tables.iter().all(|table| table.cache().room(RowClass::Prefill) >= 2 * PIECE_TOKENS * table.rows_per_token())
}

/// The oldest piece and later table whose rows are due: the previous table has
/// landed through unit `u + d - 1` (or as far as planning has gone when the
/// window or the caches stop it).
fn due(plan: &Plan, tables: &[Arc<dyn LookaheadTable>]) -> Option<(usize, usize)> {
    let (index, table) = plan.pieces.iter().enumerate()
        .find_map(|(index, piece)| (1..piece.rows.len()).find(|&t| piece.rows[t].is_some()).map(|t| (index, t)))?;
    let last = (plan.pieces[index].unit + plan.units.depth - 1).min(plan.units.ends.len() - 1);
    let horizon = plan.units.ends[last];
    let planned_end = plan.start + plan.next as u64;
    if planned_end < horizon && ready_to_plan(plan, tables) { return None; }
    plan.pieces.iter().filter(|piece| piece.unit <= last).all(|piece| piece.landed(table - 1))
        .then_some((index, table))
}

/// Pin `rows` and start reading the ones nobody holds. Also returns (distinct
/// rows, already cached, being filled by another reader).
fn reserve(table: &dyn LookaheadTable, rows: Vec<u64>, class: RowClass, priority: ReadPriority)
    -> Result<(Pins, Option<Read>, [u64; 3])> {
    let rows = distinct(rows);
    let (held, lookups) = table.cache().acquire(&rows, class);
    let (mut ids, mut slots, mut counts) = (Vec::new(), Vec::new(), [rows.len() as u64, 0, 0]);
    for (lookup, &row) in lookups.iter().zip(&rows) {
        match *lookup {
            Lookup::Fill(slot) => { ids.push(row); slots.push(slot); }
            Lookup::Ready(_) => counts[1] += 1,
            Lookup::Wait(_) => counts[2] += 1,
            Lookup::Uncached => {}
        }
    }
    let read = if ids.is_empty() { None } else { Some((table.start_read(&ids, priority)?, slots)) };
    Ok((held, read, counts))
}

fn start_piece(plan: &mut Plan, tables: &[Arc<dyn LookaheadTable>]) -> Result<()> {
    // Pieces never cross a unit boundary.
    let position = plan.start + plan.next as u64;
    let unit = plan.unit_of(position);
    let unit_end = (plan.units.ends[unit] - plan.start) as usize;
    let end = (plan.next + PIECE_TOKENS).min(unit_end).min(plan.source.tokens());
    let mut rows = plan.source.rows(plan.next..end)?;
    anyhow::ensure!(rows.len() == tables.len(), "row source gave {} tables, the look-ahead has {}", rows.len(), tables.len());
    plan.next = end;
    let first = std::mem::take(&mut rows[0]);
    let (pins, read, counts) = reserve(tables[0].as_ref(), first, RowClass::Prefill, ReadPriority::PrefillFirst)?;
    plan.tally(0, counts);
    let mut piece = PlanPiece { end: plan.start + end as u64, unit, rows: rows.into_iter().map(Some).collect(),
        pins: (0..tables.len()).map(|_| None).collect(), reads: (0..tables.len()).map(|_| None).collect() };
    piece.rows[0] = None;
    piece.pins[0] = Some(pins);
    piece.reads[0] = read;
    if tables.len() == 1 { plan.reserved.fetch_max(piece.end, Ordering::Release); }
    plan.pieces.push_back(piece);
    Ok(())
}

fn start_table(plan: &mut Plan, index: usize, table: usize, tables: &[Arc<dyn LookaheadTable>]) -> Result<()> {
    let rows = plan.pieces[index].rows[table].take().expect("due rows");
    let (pins, read, counts) = reserve(tables[table].as_ref(), rows, RowClass::Prefill, ReadPriority::Prefill)?;
    plan.tally(table, counts);
    let piece = &mut plan.pieces[index];
    piece.pins[table] = Some(pins);
    piece.reads[table] = read;
    if table + 1 == tables.len() { plan.reserved.fetch_max(piece.end, Ordering::Release); }
    Ok(())
}

/// Reserve a decode round's rows in the decode shares and start reading the ones
/// nobody holds; None when every row is already cached or in flight.
fn start_decode(rows: &[Vec<u64>], tables: &[Arc<dyn LookaheadTable>]) -> Result<Option<DecodePiece>> {
    let mut piece = DecodePiece { pins: Vec::with_capacity(tables.len()), reads: Vec::with_capacity(tables.len()) };
    for (index, table) in tables.iter().enumerate() {
        let rows = distinct(rows.get(index).cloned().unwrap_or_default());
        let (held, lookups) = table.cache().acquire(&rows, RowClass::Decode);
        let (mut ids, mut slots) = (Vec::new(), Vec::new());
        for (lookup, &row) in lookups.iter().zip(&rows) {
            if let Lookup::Fill(slot) = *lookup { ids.push(row); slots.push(slot); }
        }
        let priority = if index == 0 { ReadPriority::DecodeFirst } else { ReadPriority::Decode };
        piece.reads.push(if ids.is_empty() { None } else { Some((table.start_read(&ids, priority)?, slots)) });
        piece.pins.push(held);
    }
    Ok(piece.reads.iter().any(Option::is_some).then_some(piece))
}

/// Publish every read that has landed, without waiting.
fn poll(plan: &mut Plan, tables: &[Arc<dyn LookaheadTable>]) -> Result<bool> {
    let mut landed = vec![0u64; tables.len()];
    for piece in &mut plan.pieces {
        for (index, table) in tables.iter().enumerate() {
            if let Some((read, _)) = piece.reads[index].as_mut() {
                if read.poll()? {
                    let read = piece.reads[index].take().expect("landed");
                    landed[index] += publish(piece.pins[index].as_mut().expect("pins of a read"), read, table.as_ref())?;
                }
            }
        }
    }
    for (index, rows) in landed.iter().enumerate() { plan.count(index, *rows); }
    Ok(landed.iter().any(|&rows| rows > 0))
}

fn publish(pins: &mut Pins, (read, slots): Read, table: &dyn LookaheadTable) -> Result<u64> {
    let rows = pins.publish_all(slots.iter().enumerate().map(|(i, &slot)| (slot, read.row(i))))?;
    table.stats().record_lookahead(rows);
    Ok(rows as u64)
}

impl Drop for Plan {
    fn drop(&mut self) {
        // No more reservations: the gathers read the rest themselves.
        self.reserved.store(u64::MAX, Ordering::Release);
        let unplanned = self.source.tokens() - self.next;
        tracing::info!(request_id = self.tag, prompt_tokens = self.source.tokens(), units = self.units.ends.len(),
            unplanned_tokens = unplanned, needed = ?self.needed, cached = ?self.cached, joined = ?self.joined,
            read = ?self.read, read_waiting = ?self.read_waiting, later_tables_skipped = self.skipped,
            lifetime_ms = self.admitted.elapsed().as_millis() as u64, "token-table look-ahead request");
    }
}
