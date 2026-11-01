//! Lane-local rounds; shared owners are borrowed only during synchronous work.
use super::*;
use crate::shared::serve::lanes::{self, LaneCommit, LaneLease, LaneSet};
use std::cell::RefCell;

pub(super) fn run<'a, P: VerificationTarget<'a>, C: DraftChain<'a>>(runtime: &tokio::runtime::Runtime,
    first: &mut P, second: &mut P, requests: &mut Requests<'a>,
    first_transport: &mut P::Transport, second_transport: &mut P::Transport,
    active: &mut [Option<Active<'a>>], draft: Option<&mut DraftRuntime<'_, 'a, C>>,
    prefixes: &mut PrefixCache<'a>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>,
) -> Result<()> {
    // Units released since both lanes started (a retirement on either lane, in any round) may
    // resume a stalled request: one epoch for both, sampled before either lane runs.
    let released_at = requests.release_epoch();
    let requests = RefCell::new(requests);
    let active = RefCell::new(active);
    let draft = RefCell::new(draft);
    let prefixes = RefCell::new(prefixes);
    let set = LaneSet::default();
    runtime.block_on(set.run(
        |lease| lane(lease, first, first_transport, &requests, &active, &draft, &prefixes, receive, wake, released_at),
        |lease| lane(lease, second, second_transport, &requests, &active, &draft, &prefixes, receive, wake, released_at),
    ))
}

/// One lane round's target and draft commit.
struct Commit<'r, 'q, 'w, 'd, 'a, P: VerificationTarget<'a>, C: DraftChain<'a>> {
    lane: usize,
    pass: &'r mut P,
    requests: &'r RefCell<&'q mut Requests<'a>>,
    draft: &'r RefCell<Option<&'d mut DraftRuntime<'w, 'a, C>>>,
    batch: &'r mut RequestBatch,
    accepted: &'r [u32],
}
impl<'a, P: VerificationTarget<'a>, C: DraftChain<'a>> LaneCommit for Commit<'_, '_, '_, '_, 'a, P, C> {
    fn begin(&mut self) -> Result<()> {
        if let Some(draft) = self.draft.borrow_mut().as_deref_mut() {
            draft.begin_queued_commit(self.lane, self.pass, &self.requests.borrow(), self.batch, self.accepted)?;
        }
        self.pass.enqueue_cache_commit(&self.requests.borrow(), self.batch, self.accepted)
    }
    fn ready(&mut self) -> Result<bool> {
        let draft = self.draft.borrow().as_deref().map(|draft| draft.poll_queued_commit(self.lane))
            .transpose()?.unwrap_or(true);
        Ok(self.pass.poll_cache_commit()? && draft)
    }
    fn publish(&mut self) -> Result<()> {
        match self.draft.borrow_mut().as_deref_mut() {
            Some(draft) => draft.finish_queued_commit(self.lane, self.pass, &mut self.requests.borrow_mut(),
                self.batch, self.accepted),
            None => self.pass.commit(&mut self.requests.borrow_mut(), self.batch, self.accepted),
        }
    }
    fn abort_and_drain(&mut self) -> Result<()> {
        let target = self.pass.abort_cache_commit(&mut self.requests.borrow_mut());
        if let Err(cleanup) = &target { tracing::error!(%cleanup, "draining failed lane window commit"); }
        let draft = match self.draft.borrow_mut().as_deref_mut() {
            Some(draft) => draft.abort_queued_commit(self.lane, &mut self.requests.borrow_mut(), self.batch),
            None => { self.requests.borrow_mut().revoke_batch(self.batch); Ok(()) }
        };
        if let Err(cleanup) = &draft { tracing::error!(%cleanup, "draining failed lane draft commit"); }
        target.and(draft)
    }
}

async fn lane<'a, P: VerificationTarget<'a>, C: DraftChain<'a>>(lease: LaneLease<'_>, pass: &mut P,
    transport: &mut P::Transport, requests: &RefCell<&mut Requests<'a>>,
    active: &RefCell<&mut [Option<Active<'a>>]>, draft: &RefCell<Option<&mut DraftRuntime<'_, 'a, C>>>,
    prefixes: &RefCell<&mut PrefixCache<'a>>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>,
    released_at: u64,
) -> Result<()> {
    let lane = lease.index();
    // Whether this lane did work this call: a round or a retirement.
    let mut worked = false;
    {
        let mut round_id = 0u64;
        loop {
            if lease.stopping() { return Ok(()); }
            // This lane has completed all of its own GPU/transport work. Retire
            // only its requests; the peer need not stop or migrate survivors.
            let retired: Vec<_> = active.borrow().iter().enumerate().filter_map(|(slot, entry)|
                entry.as_ref().filter(|r| r.lane() == lane && (r.finished || r.job.events.is_closed()))
                    .map(|_| slot)).collect();
            for slot in retired {
                worked = true;
                let request = active.borrow_mut()[slot].take().unwrap();
                let request_id = request.id();
                retire(lane, request, requests, prefixes, draft).await?;
                tracing::debug!(target: "cuteafd::lane_schedule", lane, request_id, round_id,
                    "independent lane request retired");
            }
            // Admission/prefill still uses both execution lanes.
            let members: Vec<_> = {
                let active = active.borrow();
                if wake.poll_media(round_id)
                    || (!wake.media_pending && wake.ready(active.iter().flatten().count(), active.len(), !receive.is_empty(),
                        requests.borrow().release_epoch())) {
                    lease.stop(); return Ok(());
                }
                let stalled = active.iter().flatten().filter(|r| r.stalled).map(|r| r.job.events.is_closed());
                match boundary(round_id + u64::from(worked), stalled, requests.borrow().release_epoch(), released_at,
                    active.iter().enumerate().filter_map(|(slot, r)|
                        r.as_ref().filter(|r| r.lane() == lane && !r.stalled).map(|_| slot))) {
                    Boundary::Stop => { lease.stop(); return Ok(()); }
                    Boundary::Done => return Ok(()),
                    Boundary::Run(members) => members,
                }
            };
            ensure!(members.len() <= 8, "independent lane exceeds eight requests");
            // A member whose next round would pass its reservation needs a top-up, which the
            // outer loop plans at a boundary where both lanes have drained.
            {
                let active = active.borrow();
                let requests = requests.borrow();
                let verify_rows = draft.borrow().as_ref().map_or(1, |d| d.max_verify_rows());
                let mut short = false;
                for &slot in &members {
                    let r = active[slot].as_ref().unwrap();
                    let need = requests.cache().committed_end(r.lease)? as usize
                        + (r.job.max_tokens - r.generated).min(verify_rows);
                    short |= need > requests.units(r.lease)?.len() * UNIT_TOKENS;
                }
                if short { lease.stop(); return Ok(()); }
            }
            let _capture_watch = crate::shared::decode_graph::CaptureWatch::round();
            let started = Instant::now();
            // Reserve target capacity and snapshot this lane's seeds, then release
            // all bank borrows before waiting on this lane's draft workspace.
            let seeds = {
                let active = active.borrow();
                let requests = requests.borrow();
                let verify_rows = draft.borrow().as_ref().map_or(1, |d| d.max_verify_rows());
                let capacity: Vec<_> = members.iter().map(|&slot| {
                    let r = active[slot].as_ref().unwrap();
                    (r.lease, (r.job.max_tokens-r.generated).min(verify_rows) as u32)
                }).collect();
                // The boundary check above reserved every member's round: validate the bindings only.
                requests.cache().check_append_capacity(&capacity)?;
                members.iter().map(|&slot| {
                    let r = active[slot].as_ref().unwrap();
                    Ok((r.id(), r.anchor, requests.cache().committed_end(r.lease)?, r.job.max_tokens-r.generated))
                }).collect::<Result<Vec<_>>>()?
            };
            let drafting = seeds;
            // The anchors' rows are known since the last commit: fetch them across the draft phase.
            super::prefetch_drafts(&requests.borrow(), &active.borrow(), &members,
                &drafting.iter().map(|seed| vec![seed.1]).collect::<Vec<_>>());
            let copied = vec![false; members.len()];
            let mut clock = RoundClock::at(started);
            let (drafted, draft_us) = if drafting.is_empty() { (Vec::new(), 0) } else { loop {
                let proposed = {
                    let mut draft = draft.borrow_mut();
                    if let Some(draft) = draft.as_deref_mut() { draft.poll_propose(lane, &drafting)? }
                    else { Some((drafting.iter().map(|r| vec![r.1]).collect(), 0)) }
                };
                if let Some(inputs) = proposed { break inputs; }
                // Even when a peer requests a cohort drain, finish our pending
                // draft and transaction before retirement can recycle its slots.
                tokio::task::yield_now().await;
            } };
            // The polled draft times itself from issue to completion.
            if !drafting.is_empty() { clock.drafted_us(draft_us); }
            let mut inputs = drafted;
            if let Some(draft) = draft.borrow_mut().as_deref_mut() {
                extend_inputs(draft, &active.borrow(), &members, &mut inputs)?;
            }
            super::prefetch_drafts(&requests.borrow(), &active.borrow(), &members, &inputs);
            let proposal = console::Proposal::capture(&inputs, console::live());
            let shared = active.borrow().iter().flatten().any(|r| r.lane() != lane);
            let (inputs, mut batch, capture, retain_shapes) = {
                let active = active.borrow();
                let mut requests = requests.borrow_mut();
                let mut draft = draft.borrow_mut();
                for (&slot, input) in members.iter().zip(&mut inputs) {
                    let r = active[slot].as_ref().unwrap();
                    if let Some(constraint) = &r.constraint { constraint.truncate_proposal(input)?; }
                    if crate::shared::probe::no_speculation(&r.job.probe) { input.truncate(1); }
                }
                // Lengths are chosen from this lane's proposals and routes only;
                // the peer lane's activity selects the shared-regime fit.
                if let Some(draft) = draft.as_deref_mut() {
                    let candidates: Vec<_> = members.iter().zip(&inputs).zip(&copied)
                        .map(|((&slot, input), &copied)| (active[slot].as_ref().unwrap().id(), if copied { 0 } else { input.len()-1 })).collect();
                    if let Some(lengths) = draft.select_lengths(lane, &candidates, shared)? {
                        for ((input, &copied), length) in inputs.iter_mut().zip(&copied).zip(lengths) {
                            if !copied { input.truncate(length+1); }
                        }
                    }
                }
                bound_inputs(&mut inputs);
                let batch = prepare_decode_lane(&mut requests, &active, &members, &inputs, draft.is_some())?;
                (inputs, Some(batch), draft.as_deref().map(DraftRuntime::capture).unwrap_or_default(),
                    draft.as_deref().is_some_and(DraftRuntime::has_policy))
            };
            let prepared_us = started.elapsed().as_micros() as u64;
            round_id += 1;
            super::stall::tick();
            tracing::debug!(target: "cuteafd::lane_schedule", lane, round_id, requests=members.len(),
                "independent verifier issued");
            let operation: Result<()> = async {
                if retain_shapes { pass.retain_small_graph_shapes()?; }
                pass.set_capture(capture)?;
                // Device-ordered passes only while the other lane is idle (two
                // busy lanes interleave on host waits; see PLAN.md).
                pass.set_device_order(!shared);
                // The peer lane's work would fill this lane's exchanges and evict an L2 prefetch.
                pass.set_l2_prefetch(!shared);
                let current = batch.as_mut().unwrap();
                let batch_id = current.cache()?.identity();
                let selected: Vec<_> = (0..current.cache()?.positions().len()).collect();
                let active_borrow = active.borrow();
                let compact = !tracing::enabled!(target: "cuteafd::logit_trace", tracing::Level::DEBUG)
                    && !super::probe_rows_wanted(&active_borrow, &members)
                    && members.iter().all(|&slot| {
                        let request = active_borrow[slot].as_ref().unwrap();
                        request.constraint.is_none() && request.job.sampling.is_greedy()
                    });
                drop(active_borrow);
                // Chunk 4a: a lane round is device-selected whenever the layout
                // supports the terminal and at least one row is servable; each
                // row's parameters then decide its kernels, and an unservable
                // row is re-sampled on the CPU inside the same round. A round
                // whose every row is unservable, or a layout without the
                // terminal, keeps the whole-round CPU path.
                let mut round = None;
                let next = if compact {
                    let greedy = unsafe { pass.execute_shared_greedy(requests, current, transport, 0, &selected).await? };
                    if tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG) {
                        tracing::debug!(target: "cuteafd::timing", lane,
                            pass_us=started.elapsed().as_micros() as u64 - prepared_us, "independent greedy pass returned");
                    }
                    ScoreRows::from_greedy(scores::VOCAB, greedy)?
                } else if use_sampled_terminal(P::SUPPORTS_SAMPLED_TERMINAL,
                    round_has_device_rows(&active.borrow(), &members)) {
                    let built = build_sampling_round(&active.borrow(), &members, &inputs,
                        tracing::enabled!(target: "cuteafd::logit_trace", tracing::Level::DEBUG))?;
                    let scored = execute_shared_sampled_rows(pass, requests, current,
                        transport, &built).await?;
                    round = Some(built);
                    scored
                } else {
                    // No device terminal ran, so no row was device-selected: the
                    // counter must not claim the servable rows as device work.
                    sampling_stats::record_round(false, 0, 0);
                    unsafe { pass.execute_shared(requests, current, transport, 0, &selected).await?; }
                    ScoreRows::new(scores::VOCAB, pass.download_logits(current, &selected).await?)?
                };
                let verify_us = started.elapsed().as_micros() as u64 - prepared_us;
                tracing::debug!(target: "cuteafd::cost_model", batch=batch_id, lane, round_id,
                    requests=members.len(), rows=selected.len(), prepared_us, verify_us,
                    "verification round cost");
                let retain_enabled = prefixes.borrow().turn_bank_enabled();
                let mut decision = prepare_commit_lane(lane, &requests.borrow(),
                    &active.borrow(), &members, &inputs, &next, draft.borrow().as_deref(),
                    verify_us, round.as_ref(), retain_enabled)?;
                // A frontier row whose bytes are already packed (a fallback row
                // the CPU re-sampled) is retained in place: the pre-chunk-4b
                // code downloaded it here a second time even though it was
                // already on the host. This lane paid that transfer, so the
                // saving is counted here and not on the serial lane.
                let packed = std::mem::take(&mut decision.frontier_packed);
                sampling_stats::record_frontier_packed(packed.len(),
                    scores::ROW_BYTES, true);
                for (member, row, mask, retain) in packed {
                    decision.next_after_commit[member] =
                        Some(retain_packed_frontier(&next, row, retain, mask.as_deref())?);
                }
                if !decision.frontier_downloads.is_empty() {
                    let rows: Vec<_> = decision.frontier_downloads.iter().map(|&(_, row, _, _)| row).collect();
                    let bytes = pass.download_logits(batch.as_ref().unwrap(), &rows).await?;
                    ensure!(bytes.len() == rows.len() * scores::ROW_BYTES, "retained frontier download extent differs");
                    for ((member, row, mask, retain), bytes) in decision.frontier_downloads.drain(..)
                        .zip(bytes.chunks_exact(scores::ROW_BYTES)) {
                        decision.next_after_commit[member] =
                            Some(resolve_frontier(retain, &next, row, bytes, mask.as_deref())?);
                    }
                }
                let commit_timed = tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG).then(Instant::now);
                lanes::commit(&mut Commit { lane, pass: &mut *pass, requests, draft, batch: batch.as_mut().unwrap(),
                    accepted: &decision.accepted }).await?;
                if let Some(t) = commit_timed {
                    tracing::debug!(target: "cuteafd::timing", lane, commit_us=t.elapsed().as_micros() as u64,
                        decide_us=t.duration_since(started).as_micros() as u64 - prepared_us - verify_us,
                        "independent lane commit");
                }
                let (accepted, emitted, emissions, accepted_inputs) = publish_commit_lane(&mut active.borrow_mut(),
                    &members, &mut batch, decision)?;
                let live = console::live();
                let tally = console::Tally::new(proposal, &inputs, &accepted_inputs, &emissions, live);
                { let active = active.borrow();
                    tally.usage(members.iter().map(|&slot| active[slot].as_ref().and_then(|r| r.job.usage.as_ref()))); }
                tracing::debug!(target: "cuteafd::lane_schedule", lane, round_id,
                    "independent verifier committed");
                for (&slot, tokens) in members.iter().zip(emissions) {
                    let sender = active.borrow()[slot].as_ref().unwrap().job.events.clone();
                    let delivered: Result<()> = async {
                        for token in tokens {
                            let (chunks, finished) = {
                                let mut active = active.borrow_mut();
                                let request = active[slot].as_mut().unwrap();
                                (request.emit_one(token)?, request.finished)
                            };
                            for chunk in chunks.into_iter().flatten() { sender.send(Ok(chunk))?; }
                            if finished { break; }
                        }
                        Ok(())
                    }.await;
                    if let Err(error) = delivered {
                        let failed = !sender.is_closed();
                        let mut active = active.borrow_mut();
                        let request = active[slot].as_mut().unwrap();
                        request.finished = true;
                        request.failed = failed;
                        drop(active);
                        let _ = sender.send(Err(format!("{error:#}").into()));
                    }
                }
                let layer_us = pass.captured_layer_us();
                observe_lane_round(draft.borrow_mut().as_deref_mut(), lane, shared,
                    pass.captured_routes(), &layer_us, &active.borrow(), &members, &inputs,
                    &accepted_inputs, &copied, &clock);
                if let Some(live) = live {
                    let active = active.borrow();
                    live.push(console_round(tally, lane, shared, started, draft_us,
                        prepared_us.saturating_sub(draft_us), verify_us, &layer_us, pass.captured_ffn_split(),
                        &active, &members, &inputs, round.as_ref()));
                    if live.gauges_due() {
                        live.push(console_gauges(&active, &requests.borrow(), &prefixes.borrow(), receive.len(), None));
                    }
                }
                tracing::debug!(target: "cuteafd::timing", lane, requests=members.len(),
                    proposed=inputs.iter().map(|r| r.len()-1).sum::<usize>(), accepted, emitted,
                    draft_us, prepared_us, verify_us, total_us=started.elapsed().as_micros() as u64,
                    "native independent lane round");
                Ok(())
            }.await;
            let stopped_capture = pass.set_route_capture(false);
            let discarded = if let Some(batch) = &mut batch {
                // Successful commits relinquish ownership. Failed execution or
                // commit must drain/discard this lane before the outer reset.
                pass.discard(batch)
            } else { Ok(()) };
            operation?;
            stopped_capture?;
            discarded?;
            // Give already-ready remote completions an opportunity to run before
            // queuing another draft on the shared RTX.
            tokio::task::yield_now().await;
        }
    }
}

async fn retire<'a, C: DraftChain<'a>>(lane: usize, mut request: Active<'a>, requests: &RefCell<&mut Requests<'a>>,
    prefixes: &RefCell<&mut PrefixCache<'a>>, draft: &RefCell<Option<&mut DraftRuntime<'_, 'a, C>>>) -> Result<()> {
    request.console_retire();
    let state = if !request.failed && requests.borrow().cache().request_id(request.lease).is_ok() {
        requests.borrow_mut().record_state(request.lease, &request.job.probe, "turn")
    } else { Ok(()) };
    // Chunk-4b: consult the turn bank before requiring the retained frontier, so
    // a cache-disabled deployment does not warn about a frontier it correctly
    // chose not to download.
    let cacheable = request.cacheable && prefixes.borrow().turn_bank_enabled()
        && requests.borrow().cache().request_id(request.lease).is_ok();
    if cacheable {
        let retained: Result<()> = async {
            let next = request.next_after_commit.as_ref().context("finished request has no retained logits")?;
            prefixes.borrow_mut().capture_session(request.job.usage.as_ref().map(|u| u.session_id().to_owned()));
            let id = super::super::prefix::draft_id(&*draft.borrow(), request.id());
            let keys = super::super::prefix::keyed(&request.image_keys, &request.tokens);
            let queued = prefixes.borrow_mut().queue_retain(lane, &mut requests.borrow_mut(),
                draft.borrow_mut().as_deref_mut(), SnapshotKind::Turn, &keys, request.image_keys.spans(), next,
                request.lease, id)?;
            if queued {
                tracing::debug!(target: "cuteafd::lane_schedule", lane, request_id=request.id(),
                    "independent snapshot queued");
                loop {
                    let ready = prefixes.borrow_mut().poll_retain(lane, &mut requests.borrow_mut(),
                        draft.borrow_mut().as_deref_mut())?;
                    if ready { break; }
                    tokio::task::yield_now().await;
                }
                tracing::debug!(target: "cuteafd::lane_schedule", lane, request_id=request.id(),
                    "independent snapshot published");
            }
            Ok(())
        }.await;
        if let Err(error) = retained {
            if let Err(cleanup) = prefixes.borrow_mut().abort_retain(lane, &mut requests.borrow_mut(),
                draft.borrow_mut().as_deref_mut()) {
                tracing::error!(%cleanup, "draining failed independent snapshot");
            }
            tracing::warn!(%error, "completed request prefix was not retained");
        }
    }
    let target = requests.borrow_mut().release_if_present(request.lease);
    let speculative = draft.borrow_mut().as_deref_mut().map(|d| d.release(request.id())).transpose();
    if let Some(probe) = request.job.probe.as_ref().filter(|probe| probe.spec.record_state) {
        if let Err(error) = &state { probe.fail(format!("state diagnostic: {error:#}")); }
        probe.finish_state();
    }
    target.and(speculative.map(|_| ())).and(state)
}

/// A lane's decision at a completed boundary (after its own retirements).
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Boundary {
    /// Stop both lanes: a stalled request can resume at the drained boundary.
    Stop,
    /// Nothing runnable on this lane: return without stopping the peer, which keeps running its
    /// members and stops at its own boundary when a stalled request can resume.
    Done,
    /// Run a round over these slots.
    Run(Vec<usize>),
}

/// The boundary decision every V4.1 lane makes ([`resume_due`], then its runnable members).
pub(super) fn boundary(work: u64, stalled: impl Iterator<Item = bool>, epoch: u64, released_at: u64,
    members: impl Iterator<Item = usize>) -> Boundary {
    if resume_due(work, stalled, epoch, released_at) { return Boundary::Stop; }
    let members: Vec<usize> = members.collect();
    if members.is_empty() { Boundary::Done } else { Boundary::Run(members) }
}

/// Whether a lane that has done `work` (rounds plus retirements) stops so a stalled request may
/// resume at the drained boundary: there is a stalled request (`closed`: its client left), and
/// units were released since both lanes started (`released_at`) or a stalled client left. A
/// lane does some work first, so every call makes progress.
pub(super) fn resume_due(rounds: u64, mut closed: impl Iterator<Item = bool>, epoch: u64, released_at: u64) -> bool {
    let mut closed = closed.by_ref().peekable();
    rounds > 0 && closed.peek().is_some() && (epoch != released_at || closed.any(|c| c))
}

#[cfg(test)]
mod tests {
    use super::{boundary, Boundary};
    use crate::shared::serve::lanes::{LaneLease, LaneSet};
    use std::cell::{Cell, RefCell};

    /// A request of the lane model: its lane, stalled or not, client closed, rounds it still runs
    /// before finishing (its turn then frees units).
    #[derive(Clone, Debug)]
    struct Req { lane: usize, stalled: bool, closed: bool, left: u32 }

    /// The two lanes of one `independent::run` call over shared state, driven by the real
    /// `LaneSet` (one stop cell, `tokio::join!` polling lane 0 first) and the real `boundary`
    /// decision. Rounds yield like the device waits do. Returns rounds run per lane.
    async fn model_lane(lease: LaneLease<'_>, active: &RefCell<Vec<Option<Req>>>, epoch: &Cell<u64>,
        released_at: u64, rounds: &[Cell<u32>; 2]) -> anyhow::Result<()> {
        let index = lease.index();
        let (mut round_id, mut worked) = (0u64, false);
        loop {
            if lease.stopping() { return Ok(()); }
            // Retire this lane's finished or closed requests (no await on the close path).
            let slots = active.borrow().len();
            for slot in 0..slots {
                let done = active.borrow()[slot].as_ref()
                    .is_some_and(|r| r.lane == index && !r.stalled && (r.left == 0 || r.closed));
                if done {
                    active.borrow_mut()[slot] = None;
                    epoch.set(epoch.get() + 1);
                    worked = true;
                }
            }
            let decision = {
                let active = active.borrow();
                boundary(round_id + u64::from(worked),
                    active.iter().flatten().filter(|r| r.stalled).map(|r| r.closed), epoch.get(), released_at,
                    active.iter().enumerate().filter_map(|(slot, r)|
                        r.as_ref().filter(|r| r.lane == index && !r.stalled).map(|_| slot)))
            };
            let members = match decision {
                Boundary::Stop => { lease.stop(); return Ok(()); }
                Boundary::Done => return Ok(()),
                Boundary::Run(members) => members,
            };
            for slot in members { active.borrow_mut()[slot].as_mut().unwrap().left -= 1; }
            rounds[index].set(rounds[index].get() + 1);
            round_id += 1;
            tokio::task::yield_now().await;
        }
    }

    /// The two lanes of one `independent::run` call over shared state, driven by the real
    /// `LaneSet` (one stop cell, `tokio::join!` polling lane 0 first) and the real `boundary`
    /// decision. Rounds yield like the device waits do. Returns rounds run per lane.
    fn run_lanes(active: &RefCell<Vec<Option<Req>>>, epoch: &Cell<u64>) -> [u32; 2] {
        let rounds = [Cell::new(0u32), Cell::new(0u32)];
        let released_at = epoch.get();
        let set = LaneSet::default();
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(set.run(|lease| model_lane(lease, active, epoch, released_at, &rounds),
            |lease| model_lane(lease, active, epoch, released_at, &rounds))).unwrap();
        [rounds[0].get(), rounds[1].get()]
    }

    /// One stalled request on lane 0 and one runnable request on lane 1. The empty lane returns
    /// without stopping its peer; the runnable request runs to its end (freeing units), and the
    /// lanes stop at that boundary so the stall is planned again. Never a call with no work.
    #[test]
    fn a_stalled_and_a_runnable_request_make_progress_until_units_return() {
        let active = RefCell::new(vec![Some(Req { lane: 0, stalled: true, closed: false, left: 50 }),
            Some(Req { lane: 1, stalled: false, closed: false, left: 30 })]);
        let epoch = Cell::new(7);
        let rounds = run_lanes(&active, &epoch);
        assert_eq!(rounds, [0, 30], "the runnable lane ran its request to the end");
        assert_eq!(epoch.get(), 8, "its retirement freed units");
        assert!(active.borrow()[1].is_none() && active.borrow()[0].as_ref().unwrap().stalled);
        // Next call: nothing runnable, the stalled request has not resumed (the outer loop's
        // reservation round resumes it): no lane runs, no round, no spin inside the call.
        assert_eq!(run_lanes(&active, &epoch), [0, 0]);
    }

    /// Lane 0's only member is a closed client retired in round 0, which frees units; the peer
    /// lane (polled second) must see that release and stop at its next boundary instead of
    /// running until its own top-up need.
    #[test]
    fn a_round_zero_retirement_on_one_lane_resumes_a_stall_on_the_other() {
        let active = RefCell::new(vec![Some(Req { lane: 0, stalled: false, closed: true, left: 50 }),
            Some(Req { lane: 1, stalled: true, closed: false, left: 50 }),
            Some(Req { lane: 1, stalled: false, closed: false, left: 1000 })]);
        let epoch = Cell::new(3);
        let rounds = run_lanes(&active, &epoch);
        assert_eq!(epoch.get(), 4);
        assert!(rounds[1] <= 1, "lane 1 stopped at its first boundary, not after {} rounds", rounds[1]);
        assert!(active.borrow()[2].as_ref().unwrap().left >= 999);
    }

    /// A closed stalled client also stops the lanes at the next boundary after some work.
    #[test]
    fn a_closed_stalled_client_stops_the_lanes_after_a_round() {
        let active = RefCell::new(vec![Some(Req { lane: 0, stalled: true, closed: true, left: 5 }),
            Some(Req { lane: 1, stalled: false, closed: false, left: 100 })]);
        let epoch = Cell::new(0);
        assert_eq!(run_lanes(&active, &epoch), [0, 1]);
        assert_eq!(boundary(0, [true].into_iter(), 0, 0, [1].into_iter()), Boundary::Run(vec![1]));
        assert_eq!(boundary(5, std::iter::empty(), 9, 0, std::iter::empty()), Boundary::Done);
    }
}
