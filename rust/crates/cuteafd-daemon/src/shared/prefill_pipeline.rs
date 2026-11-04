//! Causal chunk ordering for prefill lanes polled on one executor thread.
//!
//! These are level-triggered dependencies, not ownership of execution storage.
//! Reserve/commit waits and their marks are separate so the family can perform its
//! synchronous cache mutation before releasing the successor. Stage-only users
//! can take `chunk` handles without participating in reservation or commit order.
//!
//! Dropping a permit cancels it without marking any unfinished dependency as
//! successful. The caller must propagate errors with `try_join!`, which drops a
//! peer waiting on that cancelled predecessor. A peer awaited independently can
//! remain pending forever; cancellation does not authorize it to use stale state.
//! Family guards remain responsible for revoking admission and draining work.

use std::cell::Cell;
use tokio::sync::Notify;

struct Event {
    ready: Cell<bool>,
    notify: Notify,
}

impl Event {
    fn new() -> Self {
        Self {
            ready: Cell::new(false),
            notify: Notify::new(),
        }
    }

    async fn wait(&self) {
        while !self.ready.get() {
            // No other lane can publish between the check and polling this wait:
            // all lanes are polled on the same executor thread.
            self.notify.notified().await;
        }
    }

    fn publish(&self) {
        self.ready.set(true);
        self.notify.notify_waiters();
    }
}

pub(crate) struct PipelineOrder {
    chunks: usize,
    stages: usize,
    // Per chunk: reservation, each stage, then commit. One allocation per prompt.
    events: Vec<Event>,
}

impl PipelineOrder {
    pub fn new(chunks: usize, stages: usize) -> Self {
        let event_count = stages
            .checked_add(2)
            .and_then(|width| chunks.checked_mul(width))
            .expect("prefill pipeline dimensions overflow");
        Self {
            chunks,
            stages,
            events: (0..event_count).map(|_| Event::new()).collect(),
        }
    }

    /// A stage-only handle, or a handle whose cache reservation is external.
    pub fn chunk(&self, index: usize) -> ChunkPermit<'_> {
        assert!(index < self.chunks, "prefill chunk index out of bounds");
        ChunkPermit { order: self, index }
    }

    /// Wait before reserving cache storage; mark only after that mutation succeeds.
    pub async fn wait_reserve_turn(&self, index: usize) -> ChunkPermit<'_> {
        let permit = self.chunk(index);
        permit.wait_previous(0).await;
        permit
    }

    fn event(&self, index: usize, offset: usize) -> &Event {
        &self.events[index * (self.stages + 2) + offset]
    }
}

pub(crate) struct ChunkPermit<'a> {
    order: &'a PipelineOrder,
    index: usize,
}

impl ChunkPermit<'_> {
    pub fn index(&self) -> usize {
        self.index
    }

    async fn wait_previous(&self, offset: usize) {
        if let Some(previous) = self.index.checked_sub(1) {
            self.order.event(previous, offset).wait().await;
        }
    }

    pub fn mark_reserved(&self) {
        assert!(
            self.index == 0 || self.order.event(self.index - 1, 0).ready.get(),
            "prefill reservation out of order"
        );
        self.order.event(self.index, 0).publish();
    }

    pub async fn wait_predecessor(&self, stage: usize) {
        assert!(stage < self.order.stages, "prefill stage out of bounds");
        self.wait_previous(stage + 1).await;
    }

    pub fn publish(&self, stage: usize) {
        assert!(stage < self.order.stages, "prefill stage out of bounds");
        self.order.event(self.index, stage + 1).publish();
    }

    /// Wait before suffix capture, boundary publication and history commit.
    pub async fn wait_commit_turn(&self) {
        self.wait_previous(self.order.stages + 1).await;
    }

    /// Release the successor only after the family's whole commit succeeds.
    pub fn commit(self) {
        let offset = self.order.stages + 1;
        assert!(
            self.index == 0 || self.order.event(self.index - 1, offset).ready.get(),
            "prefill commit out of order"
        );
        self.order.event(self.index, offset).publish();
    }
}

/// The shared prefill executor (PLAN "Prefill contract"): a prompt runs as a
/// continuous stream of row units on one lane and one Spark transport, with up
/// to `waves` units' expert waves outstanding. Each in-flight unit owns a slot:
/// its slice of every buffer that stays live across a hook, and one return
/// slot on the transport. Hook-local scratch is shared.
///
/// Ordering is `PipelineOrder`'s, generalized and checked synchronously:
/// unit k+1 of a sequence attends layer L only after unit k attended L (its KV
/// and recurrent state for L are published), each unit runs its own layers in
/// order, waves land in dispatch order, and units commit in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Unit {
    /// Order within the call.
    pub index: usize,
    /// Causal chain: units of one sequence attend in order at every layer.
    pub sequence: usize,
    /// Explicit members of a packed unit; empty for the unchanged singleton path.
    pub segments: Vec<UnitSegment>,
    /// Rows of the call's token stream.
    pub rows: std::ops::Range<usize>,
    /// Layers this unit runs (normally all; a partial replay may run a suffix).
    pub layers: std::ops::Range<usize>,
    /// In-flight slot: owns this unit's slices and its return slot.
    pub slot: usize,
    /// Start only after every earlier unit committed.
    pub barrier: bool,
}

/// A request's contribution to a unit. `rows` are unit-local, `tokens` are
/// request-local. Identities are stable for the lifetime of this submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnitSegment {
    pub sequence: usize,
    pub rows: std::ops::Range<usize>,
    pub tokens: std::ops::Range<usize>,
    pub prompt_end: bool,
    pub snapshot: bool,
    pub head: bool,
}

/// Source metadata owned by a deferred Spark request builder.
pub(crate) struct ReplyOrigins(Vec<(u64, u64, cuteafd_transport::ExpertV2SourceKind)>);
impl ReplyOrigins {
    pub(crate) fn combined_reply_eligible(&self) -> bool {
        crate::shared::spark_waves::combined_reply_eligible(self.0.iter().map(|&(id, _, kind)|
            (id, kind == cuteafd_transport::ExpertV2SourceKind::Prefill)))
    }

    pub(crate) fn apply(self, request: &mut cuteafd_transport::ExpertProtocolV2Request) -> anyhow::Result<()> {
        anyhow::ensure!(request.rows.len() == self.0.len(), "packed Spark origin count differs from request rows");
        for (row, (id, position, kind)) in request.rows.iter_mut().zip(self.0) {
            row.source_request_id = id;
            row.token_position = position;
            row.source_kind = kind;
        }
        Ok(())
    }
}

impl Unit {
    /// Packed units carry their real members in every mode; leave singleton
    /// descriptors unchanged. Worker math does not consume this metadata.
    pub(crate) fn reply_origins(&self) -> anyhow::Result<Option<ReplyOrigins>> {
        if self.segments.len() <= 1 { return Ok(None); }
        self.packed_reply_origins().map(Some)
    }

    fn packed_reply_origins(&self) -> anyhow::Result<ReplyOrigins> {
        let mut origins = Vec::with_capacity(self.rows.len());
        for segment in &self.segments {
            anyhow::ensure!(segment.rows.start == origins.len() && segment.rows.len() == segment.tokens.len(),
                "packed Spark segments must be contiguous with matching token extents");
            let id = u64::try_from(segment.sequence)?.checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("packed Spark source id overflow"))?;
            for token in segment.tokens.clone() {
                origins.push((id, u64::try_from(token)?, cuteafd_transport::ExpertV2SourceKind::Prefill));
            }
        }
        anyhow::ensure!(origins.len() == self.rows.len(), "packed Spark segments do not cover the unit");
        Ok(ReplyOrigins(origins))
    }

    fn sequences(&self) -> Vec<usize> {
        if self.segments.is_empty() { vec![self.sequence] }
        else { self.segments.iter().map(|s| s.sequence).collect() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FfnRoute {
    /// Dense or RTX-resident experts, completed inside one hook.
    Local,
    /// A Spark wave: sent in one hook, landed in a later one.
    Spark,
}

/// Restore submission-local host state on every exit, without waiting for GPU
/// work. The enclosing submission owner still aborts/drains queued work on error.
pub(crate) fn with_cleanup<T, R>(state: &mut T,
    body: impl FnOnce(&mut T) -> anyhow::Result<R>,
    cleanup: impl FnOnce(&mut T) -> anyhow::Result<()>) -> anyhow::Result<R> {
    struct Guard<'a, T, F: FnOnce(&mut T) -> anyhow::Result<()>> {
        state: &'a mut T,
        cleanup: Option<F>,
    }
    impl<T, F: FnOnce(&mut T) -> anyhow::Result<()>> Drop for Guard<'_, T, F> {
        fn drop(&mut self) {
            if let Some(cleanup) = self.cleanup.take() {
                if let Err(error) = cleanup(self.state) {
                    tracing::error!(%error, "prefill scope cleanup failed during unwinding");
                }
            }
        }
    }
    let mut guard = Guard { state, cleanup: Some(cleanup) };
    let result = body(guard.state);
    let restored = guard.cleanup.take().expect("prefill cleanup is armed")(guard.state);
    match result {
        Ok(value) => restored.map(|()| value),
        Err(error) => {
            if let Err(cleanup) = restored {
                tracing::error!(%cleanup, "prefill scope cleanup failed after submission error");
            }
            Err(error)
        }
    }
}

/// A family's per-unit, per-layer arithmetic. Every hook enqueues work on the
/// lane's streams and returns; the executor owns order and waves in flight.
/// Anything a hook leaves for a later hook lives in the unit's slot, never at a
/// shared fixed offset. Peer partial exchanges complete inside one hook.
pub(crate) trait PrefillHooks {
    type Ticket;
    fn route(&self, layer: usize) -> FfnRoute;
    /// Before the unit's first layer: embed, inject media, stage the slot's
    /// attention metadata and side inputs.
    fn begin_unit(&mut self, unit: &Unit) -> anyhow::Result<()>;
    /// Attention for the unit's rows, including planned hops; publishes the
    /// unit's KV/state for `layer`.
    fn attention(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<()>;
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<()>;
    /// Router, shared expert and the wave's dispatch into the unit's return
    /// slot. The host may wait only for this unit's route copies.
    fn send_experts(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<Self::Ticket>;
    /// Receive, reduce, combine, residual update and drafter taps.
    fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: Self::Ticket) -> anyhow::Result<()>;
    /// After the last layer: exit hops, final norm and the requested head rows.
    fn end_unit(&mut self, unit: &Unit) -> anyhow::Result<()>;
    /// In unit order: frontier/state commit and drafter hand-off. Frees the slot.
    fn commit_unit(&mut self, unit: &Unit) -> anyhow::Result<()>;
}

/// Async owners follow the same schedule and storage contract as
/// [`PrefillHooks`]. Hooks are polled on the CUDA-owning thread, not spawned:
/// their futures need not be `Send` and may retain device-owner borrows.
pub(crate) trait AsyncPrefillHooks {
    type Ticket;
    fn route(&self, layer: usize) -> FfnRoute;
    async fn begin_unit(&mut self, unit: &Unit) -> anyhow::Result<()>;
    async fn attention(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<()>;
    async fn local_ffn(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<()>;
    async fn send_experts(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<Self::Ticket>;
    async fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: Self::Ticket) -> anyhow::Result<()>;
    async fn end_unit(&mut self, unit: &Unit) -> anyhow::Result<()>;
    async fn commit_unit(&mut self, unit: &Unit) -> anyhow::Result<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    Begin { unit: usize },
    Attention { layer: usize, unit: usize },
    Local { layer: usize, unit: usize },
    Send { layer: usize, unit: usize },
    Land { layer: usize, unit: usize },
    End { unit: usize },
    Commit { unit: usize },
}

/// Fewest rows per wave worth splitting a prompt that fits one unit (TJ,
/// 2026-10-25: splitting very short prompts buys nothing). Below `2 *` this a
/// prompt runs as one unit, so its expert and projection kernels see the
/// whole prompt's rows rather than switching route on a half.
pub(crate) const MIN_SPLIT_ROWS: usize = 1024;

/// Cut `span` rows into 64-row-aligned units of at most `unit_rows`.
///
/// A span longer than one unit cuts into full units of exactly `unit_rows`
/// and one remainder tail: families capture graphs at the configured unit
/// width, and a balanced width (`span / count`) would run every unit of a
/// prompt eagerly (#815). A span that fits one unit splits into equal halves
/// (or more) only when each wave keeps [`MIN_SPLIT_ROWS`] and the units stay
/// half full (#762, `prefill_waves::waves_used`).
pub(crate) fn prefill_units(span: usize, unit_rows: usize, waves: usize, min_rows: usize)
    -> anyhow::Result<Vec<std::ops::Range<usize>>> {
    anyhow::ensure!(span > 0 && unit_rows > 0 && unit_rows % 64 == 0 && waves > 0 && min_rows > 0,
        "prefill unit geometry must be positive and page aligned");
    if span > unit_rows {
        return Ok((0..span).step_by(unit_rows).map(|first| first..span.min(first + unit_rows)).collect());
    }
    let floor = min_rows.max(MIN_SPLIT_ROWS);
    let used = cuteafd_loader::serving_capacity::prefill_waves::waves_used(
        span as u64, waves as u64, unit_rows as u64) as usize;
    let count = (1..=used).rev().find(|&n| n == 1 || span / n >= floor).unwrap_or(1);
    // ceil(span / count) <= unit_rows, and unit_rows is page aligned.
    let width = span.div_ceil(count).next_multiple_of(64);
    Ok((0..span).step_by(width).map(|first| first..span.min(first + width)).collect())
}

/// Plan a call whose output survives only in the final unit's workspace.
/// Scoring callers submit at most one unit; keep it whole even when the normal
/// serving planner would split it across waves. Last-row serving is unchanged.
pub(crate) fn retained_logits_units(span: usize, unit_rows: usize, waves: usize, min_rows: usize,
    logit_rows: usize, logit_capacity: usize) -> anyhow::Result<Vec<std::ops::Range<usize>>> {
    let units = prefill_units(span, unit_rows, waves, min_rows)?;
    anyhow::ensure!(logit_rows <= span && logit_rows <= logit_capacity,
        "prefill requests {logit_rows} logits rows with span {span} and logits capacity {logit_capacity}");
    if logit_rows > units.last().map_or(0, |unit| unit.len()) {
        anyhow::ensure!(span <= unit_rows,
            "prefill logits must lie in the last retained unit; score in calls of at most {unit_rows} rows");
        return Ok(vec![0..span]);
    }
    Ok(units)
}

/// Units for one sequence's span starting at row 0, slots assigned round-robin.
pub(crate) fn sequence_units(span: usize, unit_rows: usize, waves: usize, layers: usize)
    -> anyhow::Result<Vec<Unit>> {
    Ok(prefill_units(span, unit_rows, waves, 256)?.into_iter().enumerate().map(|(index, rows)| Unit {
        index, sequence: 0, segments: Vec::new(), rows, layers: 0..layers, slot: index % waves, barrier: false }).collect())
}

/// The executor's exact order. Units enter free slots in order. Each turn
/// serves one unit: land its pending wave, then run its layers until its next
/// wave is sent, it ends, or it ran a local layer followed by another local
/// layer while another unit is live (runs of local layers interleave across
/// units, as the paired encoder lanes did; the layer before a wave keeps the
/// turn so the wave is sent at once). Turns rotate in send order, so waves land in
/// dispatch order. A unit admitted into a freed slot is served before any
/// older unit lands again, so its first wave queues on the Sparks behind the
/// older units' last waves: the transport never drains between units.
pub(crate) fn prefill_schedule(routes: &[FfnRoute], units: &[Unit], waves: usize)
    -> anyhow::Result<Vec<Step>> {
    anyhow::ensure!(waves > 0, "prefill needs at least one wave in flight");
    for (index, unit) in units.iter().enumerate() {
        anyhow::ensure!(unit.index == index && unit.slot < waves && !unit.rows.is_empty()
            && unit.layers.start < unit.layers.end && unit.layers.end <= routes.len(),
            "prefill unit {index} geometry differs from the call");
    }
    let mut latest = std::collections::HashMap::new();
    let mut predecessor = Vec::with_capacity(units.len());
    for unit in units {
        let mut first = 0;
        let mut members = std::collections::HashSet::new();
        for segment in &unit.segments {
            anyhow::ensure!(segment.rows.start == first && !segment.rows.is_empty()
                && segment.rows.len() == segment.tokens.len()
                && members.insert(segment.sequence), "invalid packed member in unit {}", unit.index);
            first = segment.rows.end;
        }
        anyhow::ensure!(unit.segments.is_empty() || first == unit.rows.len(),
            "packed members do not cover unit {}", unit.index);
        let mut previous = Vec::new();
        for sequence in unit.sequences() {
            if let Some(earlier) = latest.insert(sequence, unit.index) {
                if !previous.contains(&earlier) { previous.push(earlier); }
            }
        }
        predecessor.push(previous);
    }
    // Next layer each unit attends: it has attended every layer below it.
    let mut attend = units.iter().map(|unit| unit.layers.start).collect::<Vec<_>>();
    let mut pending = vec![false; units.len()];
    let mut ended = vec![false; units.len()];
    let mut committed = 0;
    let mut next = 0;
    let mut live = Vec::<usize>::new();
    let mut fresh = std::collections::VecDeque::<usize>::new();
    let mut turns = std::collections::VecDeque::<usize>::new();
    let mut idle_turns = 0;
    let mut steps = Vec::new();
    while committed < units.len() {
        while live.len() < waves && next < units.len() && (!units[next].barrier || committed == next) {
            anyhow::ensure!(live.iter().all(|&other| units[other].slot != units[next].slot),
                "prefill unit {next} slot {} is still in flight", units[next].slot);
            steps.push(Step::Begin { unit: next });
            live.push(next);
            fresh.push_back(next);
            next += 1;
        }
        let unit = fresh.pop_front().or_else(|| turns.pop_front())
            .ok_or_else(|| anyhow::anyhow!("prefill schedule has no unit to serve"))?;
        let mut progressed = false;
        if pending[unit] {
            steps.push(Step::Land { layer: attend[unit] - 1, unit });
            pending[unit] = false;
            progressed = true;
        }
        while !pending[unit] && attend[unit] < units[unit].layers.end {
            let layer = attend[unit];
            // Unit k+1 attends layer L only after unit k attended L.
            if predecessor[unit].iter().copied().filter(|&earlier| earlier >= committed)
                .any(|earlier| attend[earlier] <= layer.min(units[earlier].layers.end - 1)) { break; }
            steps.push(Step::Attention { layer, unit });
            attend[unit] += 1;
            progressed = true;
            if routes[layer] == FfnRoute::Spark {
                steps.push(Step::Send { layer, unit });
                pending[unit] = true;
            } else {
                steps.push(Step::Local { layer, unit });
                // Yield the turn after a local layer whose successor is local
                // too, when another unit is live: dense runs interleave across
                // units. The last local layer before a Spark layer keeps the
                // turn, so the unit's next wave is sent without delay.
                // Only to a unit that can run a local layer itself (no wave to
                // land first): yielding to one that must land would wait on
                // its Sparks before this unit's next GPU-only layer.
                let other_local = live.iter().any(|&other| other != unit && !pending[other]
                    && attend[other] < units[other].layers.end && routes[attend[other]] == FfnRoute::Local);
                if other_local && attend[unit] < units[unit].layers.end
                    && routes[attend[unit]] == FfnRoute::Local { break; }
            }
        }
        if !pending[unit] && attend[unit] == units[unit].layers.end {
            steps.push(Step::End { unit });
            ended[unit] = true;
            progressed = true;
        } else {
            turns.push_back(unit);
        }
        while committed < next && ended[committed] {
            steps.push(Step::Commit { unit: committed });
            live.retain(|&other| other != committed);
            committed += 1;
        }
        idle_turns = if progressed { 0 } else { idle_turns + 1 };
        anyhow::ensure!(idle_turns <= live.len(), "prefill schedule deadlocked");
    }
    Ok(steps)
}

// Expand only the hook calls' await syntax so both drivers use one step walk
// and exactly the same ticket, admission, FIFO and error handling.
macro_rules! drive_prefill_units {
    ($hooks:ident, $units:ident, $waves:ident $(, $await:tt)?) => {{
        let layers = $units.iter().map(|unit| unit.layers.end).max().unwrap_or(0);
        let routes = (0..layers).map(|layer| $hooks.route(layer)).collect::<Vec<_>>();
        let steps = prefill_schedule(&routes, $units, $waves)?;
        let mut tickets: Vec<Option<(usize, H::Ticket)>> = (0..$units.len()).map(|_| None).collect();
        let mut outstanding = std::collections::VecDeque::new();
        let timed = tracing::enabled!(target: "cuteafd::prefill_hooks", tracing::Level::DEBUG);
        for step in steps {
            // Per-hook host time (RUST_LOG=cuteafd::prefill_hooks=debug): how long
            // the issuing thread spends inside each hook, the drains included.
            let started = timed.then(std::time::Instant::now);
            match step {
                Step::Begin { unit } => $hooks.begin_unit(&$units[unit])$(.$await)??,
                Step::Attention { layer, unit } => $hooks.attention(layer, &$units[unit])$(.$await)??,
                Step::Local { layer, unit } => $hooks.local_ffn(layer, &$units[unit])$(.$await)??,
                Step::Send { layer, unit } => {
                    anyhow::ensure!(outstanding.len() < $waves, "prefill wave admission exceeded");
                    tickets[unit] = Some((layer, $hooks.send_experts(layer, &$units[unit])$(.$await)??));
                    outstanding.push_back(unit);
                }
                Step::Land { layer, unit } => {
                    anyhow::ensure!(outstanding.pop_front() == Some(unit), "prefill waves land out of order");
                    let (sent, ticket) = tickets[unit].take().ok_or_else(|| anyhow::anyhow!("prefill wave missing"))?;
                    anyhow::ensure!(sent == layer, "prefill wave layer differs");
                    $hooks.land_experts(layer, &$units[unit], ticket)$(.$await)??;
                }
                Step::End { unit } => $hooks.end_unit(&$units[unit])$(.$await)??,
                Step::Commit { unit } => $hooks.commit_unit(&$units[unit])$(.$await)??,
            }
            if let Some(started) = started {
                let (kind, unit, layer) = match step {
                    Step::Begin { unit } => ("begin", unit, None),
                    Step::Attention { layer, unit } => ("attention", unit, Some(layer)),
                    Step::Local { layer, unit } => ("local", unit, Some(layer)),
                    Step::Send { layer, unit } => ("send", unit, Some(layer)),
                    Step::Land { layer, unit } => ("land", unit, Some(layer)),
                    Step::End { unit } => ("end", unit, None),
                    Step::Commit { unit } => ("commit", unit, None),
                };
                tracing::debug!(target: "cuteafd::prefill_hooks", kind, unit, layer, slot = $units[unit].slot,
                    host_us = started.elapsed().as_micros() as u64, "prefill hook");
            }
        }
        anyhow::ensure!(outstanding.is_empty(), "prefill finished with waves outstanding");
        Ok(())
    }};
}

/// Runs `units` through `hooks` in [`prefill_schedule`] order. Errors return
/// immediately; the family's submission guard aborts in-flight waves and
/// drains before storage is released (the executor holds no owners).
pub(crate) fn run_prefill_units<H: PrefillHooks>(hooks: &mut H, units: &[Unit], waves: usize)
    -> anyhow::Result<()> {
    drive_prefill_units!(hooks, units, waves)
}

/// The async counterpart of [`run_prefill_units`], using its exact step walk.
/// Cancellation drops tickets; the family's submission guard must abort and
/// drain every queued device operation before releasing its owners.
pub(crate) async fn run_prefill_units_async<H: AsyncPrefillHooks>(hooks: &mut H, units: &[Unit], waves: usize)
    -> anyhow::Result<()> {
    drive_prefill_units!(hooks, units, waves, await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::time::Duration;

    #[test]
    fn packed_reply_metadata_enforces_each_members_threshold() -> anyhow::Result<()> {
        use cuteafd_core::SparkReplyFormat;
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertV2Dtype, ExpertV2SourceKind};
        for first_rows in [64, 65] {
            let rows = first_rows + 65;
            let unit = Unit { index: 0, sequence: 7, segments: vec![
                UnitSegment { sequence: 7, rows: 0..first_rows, tokens: 91..91 + first_rows,
                    prompt_end: false, snapshot: false, head: false },
                UnitSegment { sequence: 12, rows: first_rows..rows, tokens: 203..268,
                    prompt_end: false, snapshot: false, head: false },
            ], rows: 400..400 + rows, layers: 0..1, slot: 0, barrier: false };
            let mut request = ExpertProtocolV2Request::new(1, 1, 0, 32, ExpertV2Dtype::Bf16,
                (0..rows).map(|row| cuteafd_transport::ExpertProtocolV2RowDescriptor {
                    row_id: row as u64, source_kind: ExpertV2SourceKind::Prefill, source_request_id: 1,
                    token_position: row as u64, route_offset: 0, route_count: 0,
                }).collect(), vec![], vec![0; rows * 64])?;
            unit.reply_origins()?.unwrap().apply(&mut request)?;
            for (row, descriptor) in request.rows.iter().enumerate() {
                let (id, token) = if row < first_rows { (8, 91 + row) }
                    else { (13, 203 + row - first_rows) };
                assert_eq!((descriptor.source_request_id, descriptor.token_position, descriptor.source_kind),
                    (id, token as u64, ExpertV2SourceKind::Prefill));
            }
            let decision = cuteafd_core::spark_reply::SparkReplyFormats { prefill: Some(SparkReplyFormat::Nvfp4), decode: None }.combined(request.rows.iter().map(|row|
                (row.source_request_id, row.source_kind == ExpertV2SourceKind::Prefill)));
            assert_eq!(decision.format, SparkReplyFormat::Nvfp4);
            request.rows[0].source_kind = ExpertV2SourceKind::Decode;
            assert_eq!(cuteafd_core::spark_reply::SparkReplyFormats { prefill: Some(SparkReplyFormat::Nvfp4), decode: None }.combined(request.rows.iter().map(|row|
                (row.source_request_id, row.source_kind == ExpertV2SourceKind::Prefill))).format, SparkReplyFormat::Bf16);
            let mut singleton = unit.clone();
            singleton.segments.clear();
            assert!(singleton.reply_origins()?.is_none());
            let mut invalid = unit.clone();
            invalid.segments[1].rows.start += 1;
            assert!(invalid.reply_origins().is_err());
        }
        Ok(())
    }

    #[test]
    fn cleanup_restores_state_on_body_errors_and_preserves_the_cause() {
        for fail_body in [false, true] {
            for fail_cleanup in [false, true] {
                let mut active = true;
                let result = with_cleanup(&mut active, |active| {
                    assert!(*active);
                    anyhow::ensure!(!fail_body, "injected unit-shape failure");
                    Ok(7)
                }, |active| {
                    *active = false;
                    anyhow::ensure!(!fail_cleanup, "injected cleanup failure");
                    Ok(())
                });
                assert!(!active);
                match (fail_body, fail_cleanup) {
                    (true, _) => assert_eq!(result.unwrap_err().to_string(), "injected unit-shape failure"),
                    (false, true) => assert_eq!(result.unwrap_err().to_string(), "injected cleanup failure"),
                    (false, false) => assert_eq!(result.unwrap(), 7),
                }
            }
        }
        // Negative control: an unguarded early return leaks the active state.
        let mut active = true;
        let result = with_cleanup(&mut active, |_| Err::<(), _>(anyhow::anyhow!("shape")), |_| Ok(()));
        assert!(result.is_err() && active);
    }

    #[test]
    fn cleanup_restores_state_on_unwinding() {
        let mut active = true;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: anyhow::Result<()> = with_cleanup(&mut active, |_| panic!("injected panic"),
                |active| { *active = false; Ok(()) });
        }));
        assert!(result.is_err());
        assert!(!active);
    }

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    fn random_routes(layers: usize, seed: &mut u64) -> Vec<FfnRoute> {
        (0..layers).map(|layer| if layer == 0 || lcg(seed) % 4 == 0 { FfnRoute::Local } else { FfnRoute::Spark })
            .collect()
    }

    /// Checks every contract ordering rule over a schedule, with a model of
    /// slot-owned rows (residual per slot, written by Begin/Local/Land, read by
    /// Attention/Send) and one shared hook-local scratch region that must never
    /// be read across a hook boundary.
    fn check_schedule(routes: &[FfnRoute], units: &[Unit], waves: usize, steps: &[Step]) {
        let mut attended = vec![Vec::<usize>::new(); units.len()];
        let mut landed_or_local = vec![Vec::<usize>::new(); units.len()];
        let mut outstanding = std::collections::VecDeque::new();
        let mut begun = vec![false; units.len()];
        let mut ended = vec![false; units.len()];
        let mut commits = Vec::new();
        // Residual slot contents: (unit, last layer completed) per slot.
        let mut residual: Vec<Option<(usize, Option<usize>)>> = vec![None; waves];
        for &step in steps {
            match step {
                Step::Begin { unit } => {
                    assert!(!begun[unit]);
                    begun[unit] = true;
                    let slot = units[unit].slot;
                    if let Some((owner, _)) = residual[slot] {
                        assert!(commits.contains(&owner), "slot {slot} reused before unit {owner} committed");
                    }
                    if units[unit].barrier { assert_eq!(commits.len(), unit, "barrier before earlier commits"); }
                    residual[slot] = Some((unit, None));
                }
                Step::Attention { layer, unit } => {
                    let u = &units[unit];
                    assert!(begun[unit] && !ended[unit]);
                    // Own predecessor layer complete.
                    let expected = if layer == u.layers.start { None } else { Some(layer - 1) };
                    assert_eq!(residual[u.slot], Some((unit, expected)), "unit {unit} layer {layer} read stale rows");
                    assert_eq!(landed_or_local[unit].last().copied(), expected);
                    // Every participating sequence's predecessor must publish first.
                    for sequence in u.sequences() {
                        if let Some(earlier) = units[..unit].iter().rev().find(|o| o.sequences().contains(&sequence)) {
                            if earlier.layers.contains(&layer) {
                                assert!(attended[earlier.index].contains(&layer) || commits.contains(&earlier.index),
                                    "unit {unit} attended layer {layer} before unit {}", earlier.index);
                            }
                        }
                    }
                    attended[unit].push(layer);
                }
                Step::Local { layer, unit } => {
                    assert_eq!(routes[layer], FfnRoute::Local);
                    assert_eq!(attended[unit].last(), Some(&layer));
                    landed_or_local[unit].push(layer);
                    residual[units[unit].slot] = Some((unit, Some(layer)));
                }
                Step::Send { layer, unit } => {
                    assert_eq!(routes[layer], FfnRoute::Spark);
                    assert_eq!(attended[unit].last(), Some(&layer));
                    outstanding.push_back((unit, layer));
                    assert!(outstanding.len() <= waves, "more than {waves} waves outstanding");
                }
                Step::Land { layer, unit } => {
                    assert_eq!(outstanding.pop_front(), Some((unit, layer)), "waves land out of dispatch order");
                    assert_eq!(residual[units[unit].slot].map(|(owner, _)| owner), Some(unit),
                        "unit {unit} slot overwritten while its wave was out");
                    landed_or_local[unit].push(layer);
                    residual[units[unit].slot] = Some((unit, Some(layer)));
                }
                Step::End { unit } => {
                    let u = &units[unit];
                    assert!(!ended[unit]);
                    assert_eq!(landed_or_local[unit], u.layers.clone().collect::<Vec<_>>());
                    ended[unit] = true;
                }
                Step::Commit { unit } => {
                    assert!(ended[unit]);
                    assert_eq!(commits.len(), unit, "units commit out of order");
                    commits.push(unit);
                }
            }
        }
        assert!(outstanding.is_empty());
        assert_eq!(commits, (0..units.len()).collect::<Vec<_>>());
    }

    #[test]
    fn packed_fork_join_waits_for_every_member_and_rejects_bad_layouts() -> anyhow::Result<()> {
        let routes = [FfnRoute::Local, FfnRoute::Local, FfnRoute::Spark, FfnRoute::Spark];
        let mut units = sequence_units(5 * 64, 64, 3, routes.len())?;
        let members = [vec![10], vec![20], vec![10, 20], vec![10], vec![20]];
        for (u, ids) in units.iter_mut().zip(&members) {
            let width = u.rows.len() / ids.len();
            u.segments = ids.iter().enumerate().map(|(i, &sequence)| UnitSegment {
                sequence, rows: i * width..(i + 1) * width, tokens: 0..width,
                prompt_end: false, snapshot: false, head: false }).collect();
        }
        let steps = prefill_schedule(&routes, &units, 3)?;
        check_schedule(&routes, &units, 3, &steps);
        // A checker that only watched the first member would accept this corruption.
        let mut broken = steps.clone();
        let before = broken.iter().position(|s| *s == Step::Attention { layer: 0, unit: 1 }).unwrap();
        let after = broken.iter().position(|s| *s == Step::Attention { layer: 0, unit: 2 }).unwrap();
        broken.swap(before, after);
        assert!(std::panic::catch_unwind(|| check_schedule(&routes, &units, 3, &broken)).is_err());
        for bad in 0..4 {
            let mut invalid = units.clone();
            match bad {
                0 => invalid[2].segments[1].sequence = 10,
                1 => invalid[2].segments[1].rows.start -= 1,
                2 => invalid[2].segments[1].rows.end += 1,
                _ => invalid[2].segments[1].tokens.end += 1,
            }
            assert!(prefill_schedule(&routes, &invalid, 3).is_err());
        }
        Ok(())
    }

    #[test]
    fn explicit_singleton_members_preserve_the_exact_schedule() -> anyhow::Result<()> {
        for span in [1, 65, 2048, 4097] {
            let units = sequence_units(span, 2048, 3, 4)?;
            let mut explicit = units.clone();
            for u in &mut explicit {
                u.segments.push(UnitSegment { sequence: 0, rows: 0..u.rows.len(), tokens: u.rows.clone(),
                    prompt_end: u.rows.end == span, snapshot: false, head: u.rows.end == span });
            }
            let routes = [FfnRoute::Local, FfnRoute::Local, FfnRoute::Spark, FfnRoute::Spark];
            assert_eq!(prefill_schedule(&routes, &units, 3)?, prefill_schedule(&routes, &explicit, 3)?);
        }
        Ok(())
    }

    fn random_units(seed: &mut u64, layers: usize, waves: usize) -> Vec<Unit> {
        let count = 1 + (lcg(seed) % 9) as usize;
        let mut sequence = 0;
        (0..count).map(|index| {
            if index > 0 && lcg(seed) % 5 == 0 { sequence += 1; }
            let barrier = index > 0 && lcg(seed) % 9 == 0;
            let start = if barrier { (lcg(seed) as usize) % layers } else { 0 };
            Unit { index, sequence, segments: Vec::new(), rows: index * 64..(index + 1) * 64, layers: start..layers,
                slot: index % waves, barrier }
        }).collect()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unit_schedule_keeps_every_contract_rule() -> anyhow::Result<()> {
        let mut seed = 7;
        for case in 0..3000 {
            let layers = 1 + (lcg(&mut seed) % 24) as usize;
            let waves = 1 + (lcg(&mut seed) % 3) as usize;
            let routes = random_routes(layers, &mut seed);
            let units = random_units(&mut seed, layers, waves);
            let steps = prefill_schedule(&routes, &units, waves)
                .map_err(|error| anyhow::anyhow!("case {case}: {error}"))?;
            check_schedule(&routes, &units, waves, &steps);
            check_drivers(&routes, &units, waves, &steps).await?;
        }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn one_wave_is_the_plain_serial_order() -> anyhow::Result<()> {
        let routes = [FfnRoute::Local, FfnRoute::Spark, FfnRoute::Spark, FfnRoute::Local];
        let units = sequence_units(4096, 2048, 1, 4)?;
        let steps = prefill_schedule(&routes, &units, 1)?;
        let mut expected = Vec::new();
        for unit in 0..2 {
            expected.push(Step::Begin { unit });
            for layer in 0..4 {
                expected.push(Step::Attention { layer, unit });
                expected.push(if routes[layer] == FfnRoute::Spark { Step::Send { layer, unit } }
                    else { Step::Local { layer, unit } });
                if routes[layer] == FfnRoute::Spark { expected.push(Step::Land { layer, unit }); }
            }
            expected.push(Step::End { unit });
            expected.push(Step::Commit { unit });
        }
        assert_eq!(steps, expected);
        check_drivers(&routes, &units, 1, &expected).await?;
        Ok(())
    }

    #[test]
    fn retained_full_logits_reject_cross_unit_requests_but_keep_serving_waves() -> anyhow::Result<()> {
        // #749: two 256-row waves cannot retain a 512-row scoring result.
        assert!(retained_logits_units(512, 256, 2, 256, 512, 512).is_err());
        assert!(retained_logits_units(1024, 256, 2, 256, 1024, 1024).is_err());
        assert!(retained_logits_units(256, 256, 2, 256, 256, 64).is_err());
        assert!(retained_logits_units(0, 256, 2, 256, 0, 256).is_err());
        for waves in 1..=4 {
            for rows in [256, 2048, 4096] {
                // All-row scoring is serial even when a wide unit normally splits.
                assert_eq!(retained_logits_units(rows, rows, waves, 256, rows, rows)?, [0..rows]);
                for span in [rows, rows * 2, rows * 4 + 1] {
                    assert_eq!(retained_logits_units(span, rows, waves, 256, 1, 64)?,
                        prefill_units(span, rows, waves, 256)?);
                }
            }
            let mut scored = 0;
            for tokens in (0..1024).collect::<Vec<_>>().chunks(256) {
                let units = retained_logits_units(tokens.len(), 256, waves, 256, tokens.len(), 256)?;
                assert_eq!(units, [0..256]);
                scored += tokens.len();
            }
            assert_eq!(scored, 1024);
        }
        Ok(())
    }

    #[test]
    fn spans_past_one_unit_cut_full_captured_units_and_one_tail() -> anyhow::Result<()> {
        // #815: every unit but the tail is exactly the configured (captured) width.
        assert_eq!(prefill_units(4096, 2048, 2, 256)?, [0..2048, 2048..4096]);
        assert_eq!(prefill_units(5000, 2048, 2, 256)?, [0..2048, 2048..4096, 4096..5000]);
        assert_eq!(prefill_units(8192, 2752, 3, 256)?, [0..2752, 2752..5504, 5504..8192]);
        assert_eq!(prefill_units(8192, 1024, 3, 256)?.len(), 8);
        assert_eq!(prefill_units(8192, 4096, 2, 256)?, [0..4096, 4096..8192]);
        assert_eq!(prefill_units(65536, 4096, 2, 256)?.len(), 16);
        assert!(prefill_units(0, 2048, 2, 256).is_err() && prefill_units(10, 100, 2, 256).is_err());
        for span in 1..=20_000 {
            for (rows, waves) in [(2048, 2), (1024, 2), (512, 1), (2048, 3), (2752, 3), (2048, 4)] {
                let units = prefill_units(span, rows, waves, 256)?;
                assert_eq!((units.first().unwrap().start, units.last().unwrap().end), (0, span));
                assert!(units.windows(2).all(|pair| pair[0].end == pair[1].start && pair[1].start % 64 == 0));
                assert!(units.iter().all(|unit| unit.len() <= rows));
                if span > rows {
                    assert!(units[..units.len() - 1].iter().all(|unit| unit.len() == rows), "{span}: {units:?}");
                }
            }
        }
        Ok(())
    }

    /// A span that fits one unit splits only when each wave keeps
    /// MIN_SPLIT_ROWS, onto the fewest waves that keep units half full (#762).
    #[test]
    fn spans_within_one_unit_split_for_overlap_only_when_worth_it() -> anyhow::Result<()> {
        use cuteafd_loader::serving_capacity::prefill_waves::{unit_rows, waves_used, MAX_WAVES};
        assert_eq!(prefill_units(300, 2048, 2, 256)?, [0..300]);
        for span in [512, 1033, 1354, 2047] {
            for waves in 1..=4 {
                assert_eq!(prefill_units(span, 4096, waves, 256)?, [0..span]);
                assert_eq!(prefill_units(span, 2048, waves, 256)?, [0..span]);
            }
        }
        assert_eq!(prefill_units(2048, 4096, 2, 256)?, [0..2048]);
        assert_eq!(prefill_units(4095, 4096, 2, 256)?, [0..4095]);
        assert_eq!(prefill_units(4096, 4096, 2, 256)?, [0..2048, 2048..4096]);
        assert_eq!(prefill_units(2752, 2752, 3, 256)?, [0..1408, 1408..2752]);
        for waves in 2..=MAX_WAVES as usize {
            let unit = unit_rows(waves as u64) as usize;
            for span in 1..=unit {
                let units = prefill_units(span, unit, waves, 256)?;
                assert_eq!((units.first().unwrap().start, units.last().unwrap().end), (0, span));
                let used = waves_used(span as u64, waves as u64, unit as u64) as usize;
                assert!(units.len() <= used.max(1) && units.len() <= waves, "{span}: {}", units.len());
                if units.len() > 1 {
                    assert!(units[..units.len() - 1].iter().all(|u| u.len() >= MIN_SPLIT_ROWS), "{span}: {units:?}");
                }
            }
        }
        Ok(())
    }

    /// Discrete-event model: one GPU stream runs the schedule's hooks in order
    /// at fixed costs; the Sparks serve waves FIFO, one at a time, `spark` each.
    /// Returns (total time, Spark idle time between the first send and the last
    /// land, excluding ramp before the first wave).
    fn simulate(routes: &[FfnRoute], steps: &[Step], attention: f64, send: f64, land: f64, spark: f64)
        -> (f64, f64) {
        let (mut gpu, mut spark_free, mut idle) = (0.0f64, 0.0f64, 0.0f64);
        let mut finish = std::collections::VecDeque::new();
        let mut first_send = None;
        let _ = routes;
        for step in steps {
            match *step {
                Step::Attention { .. } => gpu += attention,
                Step::Local { .. } => gpu += attention,
                Step::Send { .. } => {
                    gpu += send;
                    let start = gpu.max(spark_free);
                    if first_send.is_some() { idle += (start - spark_free).max(0.0); }
                    first_send.get_or_insert(start);
                    spark_free = start + spark;
                    finish.push_back(spark_free);
                }
                Step::Land { .. } => {
                    gpu = gpu.max(finish.pop_front().unwrap()) + land;
                }
                _ => {}
            }
        }
        (gpu, idle)
    }

    /// Two-RTX V4.1 (layers 0-18 local, 19-39 Spark): units interleave their
    /// local layers instead of one unit running all nineteen before the next
    /// starts (hunt A rank 3). Unit k+1 still attends a layer only after unit k.
    #[tokio::test(flavor = "current_thread")]
    async fn local_layer_runs_interleave_across_units() -> anyhow::Result<()> {
        let routes = (0..40).map(|l| if l < 19 { FfnRoute::Local } else { FfnRoute::Spark }).collect::<Vec<_>>();
        let units = sequence_units(8256, 2752, 3, 40)?;
        let steps = prefill_schedule(&routes, &units, 3)?;
        check_drivers(&routes, &units, 3, &steps).await?;
        let at = |want: Step| steps.iter().position(|step| *step == want).expect("step scheduled");
        // Unit 1 starts its local layers before unit 0 finished its own.
        assert!(at(Step::Attention { layer: 0, unit: 1 }) < at(Step::Attention { layer: 2, unit: 0 }),
            "unit 1 waited for unit 0's local layers");
        assert!(at(Step::Attention { layer: 0, unit: 2 }) < at(Step::Attention { layer: 3, unit: 0 }));
        // Causal order: unit k+1 attends layer L after unit k.
        for layer in 0..40 { for unit in 1..units.len() {
            assert!(at(Step::Attention { layer, unit: unit - 1 }) < at(Step::Attention { layer, unit }));
        } }
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn two_waves_keep_the_sparks_busy_whenever_the_gpu_keeps_up() -> anyhow::Result<()> {
        let units = sequence_units(16384, 2048, 2, 60)?;
        for dense in [0usize, 1, 3] {
            let routes = (0..60).map(|l| if l < dense { FfnRoute::Local } else { FfnRoute::Spark }).collect::<Vec<_>>();
            let serial = prefill_schedule(&routes, &sequence_units(16384, 2048, 1, 60)?, 1)?;
            let two = prefill_schedule(&routes, &units, 2)?;
            check_drivers(&routes, &units, 2, &two).await?;
            // GPU time per unit-layer (land + attention + send) below the
            // Sparks' time per wave: within a unit the Sparks never wait. When
            // a unit ends, its successor starts with GPU-only work (dense
            // layers and its first attention); only that lead beyond the
            // Sparks' last queued wave can show as a gap, at most once per new
            // unit. The Sparks stay busy for nearly the whole prefill.
            for (attention, send, land, spark) in [(4.0, 1.0, 1.0, 10.0), (6.0, 1.0, 1.5, 9.0), (1.0, 0.5, 0.5, 20.0)] {
                assert!(attention + send + land < spark);
                let (one_time, one_idle) = simulate(&routes, &serial, attention, send, land, spark);
                let (time, idle) = simulate(&routes, &two, attention, send, land, spark);
                let lead = attention * (2 * dense + 1) as f64 + send + land;
                let bound = (units.len() - 1) as f64 * (lead - spark).max(0.0);
                assert!(idle <= bound + 1e-9,
                    "Spark idle {idle} > {bound} with {dense} dense layers, gpu {attention}/{send}/{land} spark {spark}");
                // With no GPU-only lead the only Spark wait is the first wave.
                let waves = (units.len() * (60 - dense)) as f64;
                assert!(dense > 0 || waves * spark / time > 0.995, "Spark busy {:.4} of the prefill", waves * spark / time);
                assert!(one_idle > idle && time < one_time, "{time} vs serial {one_time}");
            }
        }
        Ok(())
    }

    struct Recorder { routes: Vec<FfnRoute>, log: Vec<Step>, fail_at: Option<usize>, tickets: usize }
    impl Recorder {
        fn record(&mut self, step: Step) -> anyhow::Result<()> {
            if self.fail_at == Some(self.log.len()) { anyhow::bail!("injected failure at {step:?}"); }
            self.log.push(step);
            Ok(())
        }
    }
    impl PrefillHooks for Recorder {
        type Ticket = usize;
        fn route(&self, layer: usize) -> FfnRoute { self.routes[layer] }
        fn begin_unit(&mut self, unit: &Unit) -> anyhow::Result<()> { self.record(Step::Begin { unit: unit.index }) }
        fn attention(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<()> {
            self.record(Step::Attention { layer, unit: unit.index })
        }
        fn local_ffn(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<()> {
            self.record(Step::Local { layer, unit: unit.index })
        }
        fn send_experts(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<usize> {
            self.record(Step::Send { layer, unit: unit.index })?;
            self.tickets += 1;
            Ok(self.tickets)
        }
        fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: usize) -> anyhow::Result<()> {
            assert!(ticket > 0);
            self.record(Step::Land { layer, unit: unit.index })
        }
        fn end_unit(&mut self, unit: &Unit) -> anyhow::Result<()> { self.record(Step::End { unit: unit.index }) }
        fn commit_unit(&mut self, unit: &Unit) -> anyhow::Result<()> { self.record(Step::Commit { unit: unit.index }) }
    }

    impl AsyncPrefillHooks for Recorder {
        type Ticket = usize;
        fn route(&self, layer: usize) -> FfnRoute { PrefillHooks::route(self, layer) }
        async fn begin_unit(&mut self, unit: &Unit) -> anyhow::Result<()> {
            tokio::task::yield_now().await;
            PrefillHooks::begin_unit(self, unit)
        }
        async fn attention(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<()> {
            tokio::task::yield_now().await;
            PrefillHooks::attention(self, layer, unit)
        }
        async fn local_ffn(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<()> {
            tokio::task::yield_now().await;
            PrefillHooks::local_ffn(self, layer, unit)
        }
        async fn send_experts(&mut self, layer: usize, unit: &Unit) -> anyhow::Result<usize> {
            tokio::task::yield_now().await;
            PrefillHooks::send_experts(self, layer, unit)
        }
        async fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: usize) -> anyhow::Result<()> {
            tokio::task::yield_now().await;
            PrefillHooks::land_experts(self, layer, unit, ticket)
        }
        async fn end_unit(&mut self, unit: &Unit) -> anyhow::Result<()> {
            tokio::task::yield_now().await;
            PrefillHooks::end_unit(self, unit)
        }
        async fn commit_unit(&mut self, unit: &Unit) -> anyhow::Result<()> {
            tokio::task::yield_now().await;
            PrefillHooks::commit_unit(self, unit)
        }
    }

    async fn check_drivers(routes: &[FfnRoute], units: &[Unit], waves: usize, expected: &[Step]) -> anyhow::Result<()> {
        let mut sync = Recorder { routes: routes.to_vec(), log: Vec::new(), fail_at: None, tickets: 0 };
        let mut asynchronous = Recorder { routes: routes.to_vec(), log: Vec::new(), fail_at: None, tickets: 0 };
        run_prefill_units(&mut sync, units, waves)?;
        run_prefill_units_async(&mut asynchronous, units, waves).await?;
        assert_eq!(sync.log, expected);
        assert_eq!(asynchronous.log, expected);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn executor_runs_the_schedule_and_stops_at_the_first_error() -> anyhow::Result<()> {
        let routes = vec![FfnRoute::Local, FfnRoute::Spark, FfnRoute::Spark, FfnRoute::Local, FfnRoute::Spark];
        let units = sequence_units(6000, 2048, 2, routes.len())?;
        let expected = prefill_schedule(&routes, &units, 2)?;
        check_drivers(&routes, &units, 2, &expected).await?;
        for fail in 0..expected.len() {
            let mut sync = Recorder { routes: routes.clone(), log: Vec::new(), fail_at: Some(fail), tickets: 0 };
            let mut asynchronous = Recorder { routes: routes.clone(), log: Vec::new(), fail_at: Some(fail), tickets: 0 };
            assert!(run_prefill_units(&mut sync, &units, 2).is_err());
            assert!(run_prefill_units_async(&mut asynchronous, &units, 2).await.is_err());
            assert_eq!(sync.log, expected[..fail], "sync hooks ran after the failure at step {fail}");
            assert_eq!(asynchronous.log, expected[..fail], "async hooks ran after the failure at step {fail}");
        }
        Ok(())
    }

    async fn jitter(seed: &mut u64) {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for _ in 0..(*seed >> 32) % 4 {
            tokio::task::yield_now().await;
        }
    }

    async fn lane(
        order: &PipelineOrder,
        parity: usize,
        stride: usize,
        mut seed: u64,
        log: &RefCell<Vec<(usize, usize)>>,
    ) {
        for index in (parity..order.chunks).step_by(stride) {
            jitter(&mut seed).await;
            let permit = order.wait_reserve_turn(index).await;
            assert_eq!(permit.index(), index);
            jitter(&mut seed).await;
            log.borrow_mut().push((index, 0));
            permit.mark_reserved();
            for stage in 0..order.stages {
                jitter(&mut seed).await;
                permit.wait_predecessor(stage).await;
                log.borrow_mut().push((index, stage + 1));
                permit.publish(stage);
            }
            jitter(&mut seed).await;
            permit.wait_commit_turn().await;
            jitter(&mut seed).await;
            log.borrow_mut().push((index, order.stages + 1));
            permit.commit();
        }
    }

    fn check_order(order: &PipelineOrder, log: &RefCell<Vec<(usize, usize)>>) {
        let log = log.borrow();
        assert_eq!(log.len(), order.chunks * (order.stages + 2));
        for dependency in 0..order.stages + 2 {
            let indices: Vec<_> = log
                .iter()
                .filter_map(|&(index, event)| (event == dependency).then_some(index))
                .collect();
            assert_eq!(indices, (0..order.chunks).collect::<Vec<_>>());
        }
        for index in 0..order.chunks {
            let events: Vec<_> = log
                .iter()
                .filter_map(|&(chunk, event)| (chunk == index).then_some(event))
                .collect();
            assert_eq!(events, (0..order.stages + 2).collect::<Vec<_>>());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn two_lanes_order_five_chunks_with_randomized_yields() {
        for seed in 0..32 {
            let order = PipelineOrder::new(5, 4);
            let log = RefCell::new(Vec::new());
            tokio::time::timeout(Duration::from_secs(1), async {
                tokio::join!(
                    lane(&order, 0, 2, seed, &log),
                    lane(&order, 1, 2, seed + 1, &log)
                );
            })
            .await
            .expect("pipeline deadlocked");
            check_order(&order, &log);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn three_lanes_preserve_dependency_order() {
        let order = PipelineOrder::new(8, 3);
        let log = RefCell::new(Vec::new());
        tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(
                lane(&order, 0, 3, 7, &log),
                lane(&order, 1, 3, 11, &log),
                lane(&order, 2, 3, 13, &log)
            );
        })
        .await
        .expect("pipeline deadlocked");
        check_order(&order, &log);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stage_publication_is_level_triggered_and_stage_specific() {
        let order = PipelineOrder::new(2, 2);
        let first = order.chunk(0);
        let second = order.chunk(1);
        first.wait_predecessor(1).await;
        first.publish(0);
        second.wait_predecessor(0).await;
        second.wait_predecessor(0).await;
        let waiting = second.wait_predecessor(1);
        tokio::pin!(waiting);
        tokio::select! {
            biased;
            _ = &mut waiting => panic!("another stage released the wait"),
            _ = tokio::task::yield_now() => {}
        }
        first.publish(1);
        tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .expect("stage publication was lost");
        second.wait_predecessor(1).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn reservation_and_commit_marks_follow_external_mutations() {
        let order = PipelineOrder::new(2, 0);
        let first = order.wait_reserve_turn(0).await;
        let second = order.wait_reserve_turn(1);
        tokio::pin!(second);
        tokio::select! {
            biased;
            _ = &mut second => panic!("reservation released before its mark"),
            _ = tokio::task::yield_now() => {}
        }
        first.mark_reserved();
        let second = second.await;
        second.mark_reserved();
        {
            let waiting = second.wait_commit_turn();
            tokio::pin!(waiting);
            tokio::select! {
                biased;
                _ = &mut waiting => panic!("commit released before its mark"),
                _ = tokio::task::yield_now() => {}
            }
            first.commit();
            waiting.await;
        }
        second.wait_commit_turn().await;
        second.commit();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn error_mid_chunk_drops_two_peer_lanes() {
        let order = PipelineOrder::new(3, 1);
        let entered = Cell::new(0);
        let result = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::try_join!(
                biased;
                async {
                    {
                        let first = order.wait_reserve_turn(0).await;
                        first.mark_reserved();
                        tokio::task::yield_now().await;
                        assert_eq!(entered.get(), 2);
                    }
                    Err::<(), _>("encoder failed mid-chunk")
                },
                async {
                    let second = order.wait_reserve_turn(1).await;
                    second.mark_reserved();
                    entered.set(entered.get() + 1);
                    second.wait_predecessor(0).await;
                    Ok::<(), &str>(())
                },
                async {
                    let third = order.wait_reserve_turn(2).await;
                    third.mark_reserved();
                    entered.set(entered.get() + 1);
                    third.wait_commit_turn().await;
                    Ok::<(), &str>(())
                }
            )
        })
        .await
        .expect("try_join did not cancel both peer lanes");
        assert_eq!(result, Err("encoder failed mid-chunk"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn erroring_lane_drops_peer_waiting_on_each_dependency() {
        for dependency in 0..3 {
            let order = PipelineOrder::new(2, 1);
            let entered = Cell::new(false);
            let result = tokio::time::timeout(Duration::from_secs(1), async {
                tokio::try_join!(
                    biased;
                    async {
                        {
                            let first = order.wait_reserve_turn(0).await;
                            if dependency != 0 {
                                first.mark_reserved();
                            }
                            tokio::task::yield_now().await;
                            assert!(entered.get());
                        }
                        Err::<(), _>("encoder failed")
                    },
                    async {
                        entered.set(true);
                        let second = order.wait_reserve_turn(1).await;
                        if dependency == 1 {
                            second.wait_predecessor(0).await;
                        } else {
                            second.wait_commit_turn().await;
                        }
                        panic!("cancelled predecessor released its successor");
                        #[allow(unreachable_code)]
                        Ok::<(), &str>(())
                    }
                )
            })
            .await
            .expect("try_join did not propagate the lane error");
            assert_eq!(result, Err("encoder failed"));
        }
    }
}
