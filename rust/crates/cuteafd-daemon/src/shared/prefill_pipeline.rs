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
    /// Rows of the call's token stream.
    pub rows: std::ops::Range<usize>,
    /// Layers this unit runs (normally all; a partial replay may run a suffix).
    pub layers: std::ops::Range<usize>,
    /// In-flight slot: owns this unit's slices and its return slot.
    pub slot: usize,
    /// Start only after every earlier unit committed.
    pub barrier: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FfnRoute {
    /// Dense or RTX-resident experts, completed inside one hook.
    Local,
    /// A Spark wave: sent in one hook, landed in a later one.
    Spark,
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

/// Cut `span` rows into equal 64-row-aligned units of at most `unit_rows`,
/// a multiple of `waves` of them while each keeps at least `min_rows`, so the
/// in-flight units finish together and no unit runs a short tail alone.
pub(crate) fn prefill_units(span: usize, unit_rows: usize, waves: usize, min_rows: usize)
    -> anyhow::Result<Vec<std::ops::Range<usize>>> {
    anyhow::ensure!(span > 0 && unit_rows > 0 && unit_rows % 64 == 0 && waves > 0 && min_rows > 0,
        "prefill unit geometry must be positive and page aligned");
    let mut count = span.div_ceil(unit_rows);
    let rounded = count.next_multiple_of(waves);
    if span / rounded >= min_rows { count = rounded; }
    // ceil(span / count) <= unit_rows, and unit_rows is page aligned.
    let width = span.div_ceil(count).next_multiple_of(64);
    Ok((0..span).step_by(width).map(|first| first..span.min(first + width)).collect())
}

/// Units for one sequence's span starting at row 0, slots assigned round-robin.
pub(crate) fn sequence_units(span: usize, unit_rows: usize, waves: usize, layers: usize)
    -> anyhow::Result<Vec<Unit>> {
    Ok(prefill_units(span, unit_rows, waves, 256)?.into_iter().enumerate().map(|(index, rows)| Unit {
        index, sequence: 0, rows, layers: 0..layers, slot: index % waves, barrier: false }).collect())
}

/// The executor's exact order. Units enter free slots in order. Each turn
/// serves one unit: land its pending wave, then run its layers until its next
/// wave is sent or it ends. Turns rotate in send order, so waves land in
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
    let predecessor = (0..units.len()).map(|index| units[..index].iter().rev()
        .find(|other| other.sequence == units[index].sequence).map(|other| other.index)).collect::<Vec<_>>();
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
            if let Some(earlier) = predecessor[unit].filter(|&earlier| earlier >= committed) {
                if attend[earlier] <= layer.min(units[earlier].layers.end - 1) { break; }
            }
            steps.push(Step::Attention { layer, unit });
            attend[unit] += 1;
            progressed = true;
            if routes[layer] == FfnRoute::Spark {
                steps.push(Step::Send { layer, unit });
                pending[unit] = true;
            } else {
                steps.push(Step::Local { layer, unit });
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

/// Runs `units` through `hooks` in [`prefill_schedule`] order. Errors return
/// immediately; the family's submission guard aborts in-flight waves and
/// drains before storage is released (the executor holds no owners).
pub(crate) fn run_prefill_units<H: PrefillHooks>(hooks: &mut H, units: &[Unit], waves: usize)
    -> anyhow::Result<()> {
    let layers = units.iter().map(|unit| unit.layers.end).max().unwrap_or(0);
    let routes = (0..layers).map(|layer| hooks.route(layer)).collect::<Vec<_>>();
    let steps = prefill_schedule(&routes, units, waves)?;
    let mut tickets: Vec<Option<(usize, H::Ticket)>> = (0..units.len()).map(|_| None).collect();
    let mut outstanding = std::collections::VecDeque::new();
    for step in steps {
        match step {
            Step::Begin { unit } => hooks.begin_unit(&units[unit])?,
            Step::Attention { layer, unit } => hooks.attention(layer, &units[unit])?,
            Step::Local { layer, unit } => hooks.local_ffn(layer, &units[unit])?,
            Step::Send { layer, unit } => {
                anyhow::ensure!(outstanding.len() < waves, "prefill wave admission exceeded");
                tickets[unit] = Some((layer, hooks.send_experts(layer, &units[unit])?));
                outstanding.push_back(unit);
            }
            Step::Land { layer, unit } => {
                anyhow::ensure!(outstanding.pop_front() == Some(unit), "prefill waves land out of order");
                let (sent, ticket) = tickets[unit].take().ok_or_else(|| anyhow::anyhow!("prefill wave missing"))?;
                anyhow::ensure!(sent == layer, "prefill wave layer differs");
                hooks.land_experts(layer, &units[unit], ticket)?;
            }
            Step::End { unit } => hooks.end_unit(&units[unit])?,
            Step::Commit { unit } => hooks.commit_unit(&units[unit])?,
        }
    }
    anyhow::ensure!(outstanding.is_empty(), "prefill finished with waves outstanding");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::time::Duration;

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
                    // Same-sequence predecessor attended this layer first.
                    if let Some(earlier) = units[..unit].iter().rev().find(|o| o.sequence == u.sequence) {
                        if earlier.layers.contains(&layer) {
                            assert!(attended[earlier.index].contains(&layer) || commits.contains(&earlier.index),
                                "unit {unit} attended layer {layer} before unit {}", earlier.index);
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

    fn random_units(seed: &mut u64, layers: usize, waves: usize) -> Vec<Unit> {
        let count = 1 + (lcg(seed) % 9) as usize;
        let mut sequence = 0;
        (0..count).map(|index| {
            if index > 0 && lcg(seed) % 5 == 0 { sequence += 1; }
            let barrier = index > 0 && lcg(seed) % 9 == 0;
            let start = if barrier { (lcg(seed) as usize) % layers } else { 0 };
            Unit { index, sequence, rows: index * 64..(index + 1) * 64, layers: start..layers,
                slot: index % waves, barrier }
        }).collect()
    }

    #[test]
    fn unit_schedule_keeps_every_contract_rule() -> anyhow::Result<()> {
        let mut seed = 7;
        for case in 0..3000 {
            let layers = 1 + (lcg(&mut seed) % 24) as usize;
            let waves = 1 + (lcg(&mut seed) % 3) as usize;
            let routes = random_routes(layers, &mut seed);
            let units = random_units(&mut seed, layers, waves);
            let steps = prefill_schedule(&routes, &units, waves)
                .map_err(|error| anyhow::anyhow!("case {case}: {error}"))?;
            check_schedule(&routes, &units, waves, &steps);
        }
        Ok(())
    }

    #[test]
    fn one_wave_is_the_plain_serial_order() -> anyhow::Result<()> {
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
        Ok(())
    }

    #[test]
    fn units_cut_equal_page_aligned_and_in_multiples_of_waves() -> anyhow::Result<()> {
        assert_eq!(prefill_units(4096, 2048, 2, 256)?, [0..2048, 2048..4096]);
        assert_eq!(prefill_units(5000, 2048, 2, 256)?, [0..1280, 1280..2560, 2560..3840, 3840..5000]);
        assert_eq!(prefill_units(300, 2048, 2, 256)?, [0..300]);
        assert_eq!(prefill_units(8192, 2048, 2, 256)?.len(), 4);
        assert!(prefill_units(0, 2048, 2, 256).is_err() && prefill_units(10, 100, 2, 256).is_err());
        for span in 1..=9000 {
            for (rows, waves) in [(2048, 2), (1024, 2), (512, 1), (2048, 3)] {
                let units = prefill_units(span, rows, waves, 256)?;
                assert_eq!(units.first().unwrap().start, 0);
                assert_eq!(units.last().unwrap().end, span);
                assert!(units.windows(2).all(|pair| pair[0].end == pair[1].start && pair[1].start % 64 == 0));
                assert!(units.iter().all(|unit| unit.len() <= rows));
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

    #[test]
    fn two_waves_keep_the_sparks_busy_whenever_the_gpu_keeps_up() -> anyhow::Result<()> {
        let units = sequence_units(16384, 2048, 2, 60)?;
        for dense in [0usize, 1, 3] {
            let routes = (0..60).map(|l| if l < dense { FfnRoute::Local } else { FfnRoute::Spark }).collect::<Vec<_>>();
            let serial = prefill_schedule(&routes, &sequence_units(16384, 2048, 1, 60)?, 1)?;
            let two = prefill_schedule(&routes, &units, 2)?;
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

    #[test]
    fn mimo_continuous_units_preserve_routes_slots_and_spark_occupancy() -> anyhow::Result<()> {
        for layers in [48, 78] {
            for onboard in [0, 4, 12] {
                let routes = (0..layers).map(|layer| if layer < 1 + onboard {
                    FfnRoute::Local
                } else { FfnRoute::Spark }).collect::<Vec<_>>();
                let serial_units = sequence_units(65536 + 127, 4096, 1, layers)?;
                let serial = prefill_schedule(&routes, &serial_units, 1)?;
                let (serial_time, _) = simulate(&routes, &serial, 1.0, 0.5, 0.5, 20.0);
                for waves in [2, 3] {
                    let units = sequence_units(65536 + 127, 4096, waves, layers)?;
                    let steps = prefill_schedule(&routes, &units, waves)?;
                    check_schedule(&routes, &units, waves, &steps);
                    assert!(units.len() > waves);
                    let (time, idle) = simulate(&routes, &steps, 1.0, 0.5, 0.5, 20.0);
                    let lead = (2 * (1 + onboard) + 1) as f64 + 1.0;
                    let bound = (units.len() - waves) as f64 * (lead - 20.0).max(0.0);
                    assert!(idle <= bound + 1e-9, "layers={layers} onboard={onboard} waves={waves}: idle={idle}, bound={bound}");
                    assert!(time < serial_time);
                }
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

    #[test]
    fn executor_runs_the_schedule_and_stops_at_the_first_error() -> anyhow::Result<()> {
        let routes = vec![FfnRoute::Local, FfnRoute::Spark, FfnRoute::Spark, FfnRoute::Local, FfnRoute::Spark];
        let units = sequence_units(6000, 2048, 2, routes.len())?;
        let expected = prefill_schedule(&routes, &units, 2)?;
        let mut hooks = Recorder { routes: routes.clone(), log: Vec::new(), fail_at: None, tickets: 0 };
        run_prefill_units(&mut hooks, &units, 2)?;
        assert_eq!(hooks.log, expected);
        for fail in 0..expected.len() {
            let mut hooks = Recorder { routes: routes.clone(), log: Vec::new(), fail_at: Some(fail), tickets: 0 };
            assert!(run_prefill_units(&mut hooks, &units, 2).is_err());
            assert_eq!(hooks.log, expected[..fail], "hooks ran after the failure at step {fail}");
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
