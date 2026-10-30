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
//! buffer: the planner polls. A piece reserves its rows one reactor batch at a
//! time, only as it submits that batch's read, so it never holds rows nobody is
//! reading: a demand gather that starts on a unit reads the unreserved rest of
//! it itself and joins only reads actually in flight. Pieces stay pinned until
//! commits pass them. Whoever reserves a row first reads it, so the look-ahead,
//! the gathers and decode share one fetch.
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

/// Fewest tokens per piece (V4.1 Engram: 128 x 24 rows x 264 B ~ 0.8 MiB).
const MIN_PIECE_TOKENS: usize = 128;

/// Tokens per piece: the most whose rows still fit one reactor buffer in every
/// table (so each read streams at full-buffer efficiency), in 64-token steps
/// and never fewer than [`MIN_PIECE_TOKENS`]; and at most a quarter of the
/// smallest prefill share, so two pieces always fit beside what is pinned.
fn piece_tokens(tables: &[Arc<dyn LookaheadTable>]) -> usize {
    let row_bytes = tables.iter().map(|table| table.rows_per_token() * table.cache().width()).max().unwrap_or(1).max(1);
    let share = tables.iter().map(|table| table.cache().share(RowClass::Prefill) / (4 * table.rows_per_token().max(1)))
        .min().unwrap_or(usize::MAX);
    #[cfg(test)]
    if let Some(tokens) = bench_override("LOOKAHEAD_BENCH_PIECE") { return tokens; }
    (super::reactor::BUFFER_BYTES / row_bytes / 64 * 64).min(share / 64 * 64).max(MIN_PIECE_TOKENS)
}

/// Placement microbenches only: a profile parameter from the environment.
#[cfg(test)]
fn bench_override(name: &str) -> Option<usize> { std::env::var(name).ok().and_then(|v| v.parse().ok()) }

/// Reads in flight (the placement microbench sweeps it).
fn max_inflight() -> usize {
    #[cfg(test)]
    if let Some(reads) = bench_override("LOOKAHEAD_BENCH_INFLIGHT") { return reads.max(1); }
    MAX_INFLIGHT
}
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
    /// Where the table's rows are read from ([`super::placement`]): prompts are
    /// read ahead only when every table is local.
    fn placement(&self) -> super::placement::Placement;
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
/// order, the pipeline depth (units in flight at once) and the index of each
/// call's first unit (the units one prefill call runs; empty: every `depth` units).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnitPlan { pub ends: Vec<u64>, pub depth: usize, pub calls: Vec<usize> }
impl UnitPlan {
    /// Calls of `depth` units each.
    pub fn new(ends: Vec<u64>, depth: usize) -> Self { Self { ends, depth, calls: Vec::new() } }
    /// The first unit of the call after the one holding unit `unit`.
    fn next_call(&self, unit: usize) -> usize {
        if self.calls.is_empty() { return (unit / self.depth.max(1) + 1) * self.depth.max(1); }
        self.calls.iter().copied().find(|&first| first > unit).unwrap_or(self.ends.len())
    }
}

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

/// One table's share of a prefill piece: its distinct rows not yet reserved
/// (until its deadline turn, `started` is false), the pins of the batches
/// reserved so far and the batch being read.
struct Step { started: bool, queue: VecDeque<u64>, pins: Vec<Pins>, read: Option<Read> }
impl Step {
    /// Every row reserved, read and settled (published here or by another reader).
    fn landed(&self) -> bool { self.started && self.queue.is_empty() && self.read.is_none() && self.pins.iter().all(Pins::settled) }
}

/// One prefill piece: a step per table.
struct PlanPiece {
    end: u64,
    /// Index of the prefill unit this piece lies in.
    unit: usize,
    steps: Vec<Step>,
}
impl PlanPiece {
    fn reading(&self) -> usize { self.steps.iter().filter(|step| step.read.is_some()).count() }
    fn landed(&self, table: usize) -> bool { self.steps[table].landed() }
    /// A started step with rows still to reserve and no read in flight.
    fn next_batch(&self) -> Option<usize> {
        self.steps.iter().position(|step| step.started && step.read.is_none() && !step.queue.is_empty())
    }
}

struct Plan {
    id: u64,
    /// The caller's request id, for the per-request summary.
    tag: u64,
    admitted: std::time::Instant,
    /// Read only from the call after the one running (a remote table).
    remote: bool,
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
    /// Tokens of running calls passed without reading (remote tables).
    skipped_tokens: u64,
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
    /// Some table has every copy on other hosts: prompts are read ahead only
    /// from the call after the one running (the demand gathers own the current
    /// call's rows on the capped remote read path).
    remote: bool,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Lookahead {
    /// `tables` in the order their consumers run within a unit.
    pub fn new(name: &str, tables: Vec<Arc<dyn LookaheadTable>>) -> Result<Self> {
        anyhow::ensure!(!tables.is_empty(), "a look-ahead needs at least one table");
        let remote = tables.iter().any(|table| table.placement() == super::placement::Placement::Remote);
        tracing::info!(name, remote, placements = ?tables.iter().map(|table| table.placement()).collect::<Vec<_>>(),
            "token-table look-ahead");
        let (send, receive) = mpsc::channel();
        let thread = std::thread::Builder::new().name(format!("{name}-lookahead"))
            .spawn(move || run(tables, receive, remote)).context("starting a token-table look-ahead")?;
        Ok(Self { send, ids: AtomicU64::new(1), remote, thread: Some(thread) })
    }
    /// Whether prompts are read ahead only from the next call (a remote table).
    pub fn remote(&self) -> bool { self.remote }

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

fn run(tables: Vec<Arc<dyn LookaheadTable>>, receive: mpsc::Receiver<Command>, remote: bool) {
    let mut plans: Vec<Plan> = Vec::new();
    let mut decode: Option<DecodePiece> = None;
    let mut next_decode: Option<Vec<Vec<u64>>> = None;
    loop {
        loop {
            match receive.try_recv() {
                Ok(Command::Stop) | Err(mpsc::TryRecvError::Disconnected) => return,
                Ok(Command::Decode(rows)) => next_decode = Some(rows),
                Ok(command) => apply(&mut plans, command, tables.len(), remote),
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
                            if publish(&mut piece.pins[index], read, table.as_ref(), true).is_ok() { progressed = true; }
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
                plan.skipped += piece.steps.iter().map(|step| step.queue.len() as u64).sum::<u64>();
            }
            match poll(plan, &tables) {
                Ok(landed) => progressed |= landed,
                Err(error) => { abandon(plan, &error); continue; }
            }
        }
        let reading = |plans: &[Plan]| plans.iter().flat_map(|plan| &plan.pieces).map(PlanPiece::reading).sum::<usize>();
        // The next batch of a started step goes first (oldest piece of the
        // earliest-admitted request): its deadline is the earliest.
        while reading(&plans) < max_inflight() {
            let Some((plan, index, table)) = plans.iter_mut().find_map(|plan| plan.pieces.iter().enumerate()
                .find_map(|(index, piece)| piece.next_batch().map(|table| (index, table))).map(|(index, table)| (plan, index, table)))
                else { break };
            if let Err(error) = next_batch(plan, index, table, &tables) { abandon(plan, &error); }
            progressed = true;
        }
        // Later-table reads that are due go next: their deadline is earlier
        // than any table-0 piece not yet started. `plans` is in admission order.
        while reading(&plans) < max_inflight() {
            let Some((plan, index, table)) = plans.iter_mut()
                .find_map(|plan| due(plan, &tables).map(|(index, table)| (plan, index, table))) else { break };
            if let Err(error) = start_table(plan, index, table, &tables) { abandon(plan, &error); }
            progressed = true;
        }
        if reading(&plans) < max_inflight() {
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
            .any(|piece| piece.steps.iter().any(|step| step.read.is_some() || !step.queue.is_empty() || !step.started)));
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
            Some(command) => apply(&mut plans, command, tables.len(), remote),
            None => {}
        }
    }
}

fn apply(plans: &mut Vec<Plan>, command: Command, tables: usize, remote: bool) {
    match command {
        Command::Start { id, tag, start, source, mut units, committed, reserved } => {
            let end = start + source.tokens() as u64;
            // Malformed or short plans degrade to one unit through the prompt's end.
            units.ends.retain(|&e| e > start && e < end);
            units.ends.push(end);
            units.depth = units.depth.max(1);
            units.calls.retain(|&first| first < units.ends.len());
            plans.push(Plan { id, tag, admitted: std::time::Instant::now(), source, start, next: 0, units, remote,
                committed, reserved, pieces: VecDeque::new(), needed: vec![0; tables], cached: vec![0; tables],
                joined: vec![0; tables], read: vec![0; tables], read_waiting: vec![0; tables], skipped: 0, skipped_tokens: 0 });
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
        && tables.iter().all(|table| table.cache().room(RowClass::Prefill) >= 2 * piece_tokens(tables) * table.rows_per_token())
}

/// The oldest piece and later table whose rows are due: the previous table has
/// landed through unit `u + d - 1` (or as far as planning has gone when the
/// window or the caches stop it).
fn due(plan: &Plan, tables: &[Arc<dyn LookaheadTable>]) -> Option<(usize, usize)> {
    let (index, table) = plan.pieces.iter().enumerate()
        .find_map(|(index, piece)| (1..piece.steps.len()).find(|&t| !piece.steps[t].started).map(|t| (index, t)))?;
    let last = (plan.pieces[index].unit + plan.units.depth - 1).min(plan.units.ends.len() - 1);
    let horizon = plan.units.ends[last];
    let planned_end = plan.start + plan.next as u64;
    if planned_end < horizon && ready_to_plan(plan, tables) { return None; }
    plan.pieces.iter().filter(|piece| piece.unit <= last).all(|piece| piece.landed(table - 1))
        .then_some((index, table))
}

/// Rows one look-ahead read reserves and submits: one reactor batch.
fn batch_rows(table: &dyn LookaheadTable) -> usize {
    super::uring::MAX_BATCH.min(super::reactor::BUFFER_BYTES / table.cache().width().max(1)).max(1)
}

/// Reserve rows from the front of `step`'s queue until one reactor batch needs
/// reading (or the queue empties), and start that read. Rows already cached or
/// being filled by another reader are pinned, not read. Returns (rows taken,
/// already cached, being filled elsewhere).
fn reserve_batch(step: &mut Step, table: &dyn LookaheadTable) -> Result<[u64; 3]> {
    // One acquire per batch (its read publishes through these pins): at most a
    // batch of rows, fewer read when some are cached or filled elsewhere.
    let take = batch_rows(table).min(step.queue.len());
    let rows: Vec<u64> = step.queue.drain(..take).collect();
    let (held, lookups) = table.cache().acquire(&rows, RowClass::Prefill);
    let (mut ids, mut slots, mut counts) = (Vec::new(), Vec::new(), [rows.len() as u64, 0, 0]);
    for (lookup, &row) in lookups.iter().zip(&rows) {
        match *lookup {
            Lookup::Fill(slot) => { ids.push(row); slots.push(slot); }
            Lookup::Ready(_) => counts[1] += 1,
            Lookup::Wait(_) => counts[2] += 1,
            Lookup::Uncached => {}
        }
    }
    step.pins.push(held);
    // Look-ahead reads are never due before a demand gather (which reads its
    // first table at PrefillFirst): every table's batches queue at Prefill.
    if !ids.is_empty() { step.read = Some((table.start_read(&ids, ReadPriority::Prefill)?, slots)); }
    Ok(counts)
}

fn start_piece(plan: &mut Plan, tables: &[Arc<dyn LookaheadTable>]) -> Result<()> {
    if plan.remote {
        // A remote table's current call belongs to its demand gathers: pass its
        // tokens (the source hashes them, nothing is read) and its gate opens.
        let committed = plan.committed.load(Ordering::Acquire).max(plan.start);
        let first = plan.units.next_call(plan.unit_of(committed));
        let from = if first == 0 { 0 } else { (plan.units.ends[first - 1] - plan.start) as usize };
        let from = from.min(plan.source.tokens());
        if plan.next < from {
            plan.source.rows(plan.next..from)?;
            plan.skipped_tokens += (from - plan.next) as u64;
            plan.next = from;
            plan.reserved.fetch_max(plan.start + from as u64, Ordering::Release);
            if plan.next >= plan.source.tokens() { return Ok(()); }
        }
    }
    // Pieces never cross a unit boundary.
    let position = plan.start + plan.next as u64;
    let unit = plan.unit_of(position);
    let unit_end = (plan.units.ends[unit] - plan.start) as usize;
    let end = (plan.next + piece_tokens(tables)).min(unit_end).min(plan.source.tokens());
    let rows = plan.source.rows(plan.next..end)?;
    anyhow::ensure!(rows.len() == tables.len(), "row source gave {} tables, the look-ahead has {}", rows.len(), tables.len());
    plan.next = end;
    let steps = rows.into_iter().map(|rows| Step { started: false, queue: distinct(rows).into(), pins: Vec::new(), read: None })
        .collect();
    plan.pieces.push_back(PlanPiece { end: plan.start + end as u64, unit, steps });
    let index = plan.pieces.len() - 1;
    start_table(plan, index, 0, tables)
}

/// Start table `table`'s step of piece `index`: its first batch.
fn start_table(plan: &mut Plan, index: usize, table: usize, tables: &[Arc<dyn LookaheadTable>]) -> Result<()> {
    plan.pieces[index].steps[table].started = true;
    if table + 1 == tables.len() { plan.reserved.fetch_max(plan.pieces[index].end, Ordering::Release); }
    next_batch(plan, index, table, tables)
}

/// Reserve and read the next batch of a started step.
fn next_batch(plan: &mut Plan, index: usize, table: usize, tables: &[Arc<dyn LookaheadTable>]) -> Result<()> {
    let counts = reserve_batch(&mut plan.pieces[index].steps[table], tables[table].as_ref())?;
    plan.tally(table, counts);
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
            let step = &mut piece.steps[index];
            if let Some((read, _)) = step.read.as_mut() {
                if read.poll()? {
                    let read = step.read.take().expect("landed");
                    let last = step.queue.is_empty();
                    landed[index] += publish(step.pins.last_mut().expect("pins of a read"), read, table.as_ref(), last)?;
                }
            }
        }
    }
    for (index, rows) in landed.iter().enumerate() { plan.count(index, *rows); }
    Ok(landed.iter().any(|&rows| rows > 0))
}

fn publish(pins: &mut Pins, (read, slots): Read, table: &dyn LookaheadTable, piece: bool) -> Result<u64> {
    let rows = pins.publish_all(slots.iter().enumerate().map(|(i, &slot)| (slot, read.row(i))))?;
    table.stats().record_lookahead(rows, piece);
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
            remote = self.remote, running_call_tokens_passed = self.skipped_tokens,
            lifetime_ms = self.admitted.elapsed().as_millis() as u64, "token-table look-ahead request");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Shape { cache: Arc<RowCache>, rows: usize, stats: Arc<TableStats> }
    impl LookaheadTable for Shape {
        fn cache(&self) -> &Arc<RowCache> { &self.cache }
        fn start_read(&self, _: &[u64], _: ReadPriority) -> Result<Box<dyn RowLanding>> { unreachable!() }
        fn stats(&self) -> &TableStats { &self.stats }
        fn rows_per_token(&self) -> usize { self.rows }
        fn placement(&self) -> super::super::placement::Placement { super::super::placement::Placement::Local }
    }
    fn shape(width: usize, rows: usize, shares: [usize; 2]) -> Arc<dyn LookaheadTable> {
        let stats = Arc::new(TableStats::default());
        Arc::new(Shape { cache: RowCache::new(width, shares, Arc::clone(&stats)).unwrap(), rows, stats })
    }

    /// One read fills at most a reactor buffer in every table; at the admitted
    /// 2752 x 3 shape the window (2 d units) and a planning step's two pieces fit
    /// the prefill share, so the look-ahead never stops early on room.
    #[test]
    fn pieces_fill_a_reactor_buffer_and_fit_the_admitted_share() {
        let (unit, depth) = (2752, 3);
        for (width, rows, expected) in [(264, 24, 128), (160, 16, 384)] {
            let share = (window_units(depth) + 1) * unit * rows;
            let tables = vec![shape(width, rows, [share, 2 * 64 * rows])];
            let tokens = piece_tokens(&tables);
            assert_eq!(tokens, expected, "{width} B x {rows} rows");
            assert!(tokens * rows * width <= super::super::reactor::BUFFER_BYTES || tokens == MIN_PIECE_TOKENS);
            assert!(window_units(depth) * unit * rows + 2 * tokens * rows <= share, "{width} B: window plus a step fits");
        }
        // Several tables: the widest row decides.
        let both = vec![shape(160, 16, [1 << 20, 1024]), shape(264, 24, [1 << 20, 1024])];
        assert_eq!(piece_tokens(&both), 128);
    }
}
