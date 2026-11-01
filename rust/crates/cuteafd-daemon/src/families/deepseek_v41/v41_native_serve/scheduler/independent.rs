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
    let requests = RefCell::new(requests);
    let active = RefCell::new(active);
    let draft = RefCell::new(draft);
    let prefixes = RefCell::new(prefixes);
    let set = LaneSet::default();
    runtime.block_on(drive_lanes(&set, || requests.borrow().release_epoch(),
        |lease, control| lane(lease, control, first, first_transport, &requests, &active, &draft, &prefixes, receive, wake),
        |lease, control| lane(lease, control, second, second_transport, &requests, &active, &draft, &prefixes, receive, wake),
    ))
}

/// Run both lanes of one call. Units released since both lanes started (a retirement on
/// either lane, in any round) may resume a stalled request, so the release epoch (`epoch`
/// reads it) is sampled once, here, before either lane runs, and every lane's [`LaneControl`]
/// compares against it.
pub(super) async fn drive_lanes<'s, F, G>(set: &'s LaneSet, epoch: impl Fn() -> u64,
    first: impl FnOnce(LaneLease<'s>, LaneControl) -> F, second: impl FnOnce(LaneLease<'s>, LaneControl) -> G)
    -> Result<()>
where F: std::future::Future<Output = Result<()>>, G: std::future::Future<Output = Result<()>> {
    let control = LaneControl::start(epoch());
    set.run(|lease| first(lease, control), |lease| second(lease, control)).await
}

/// One lane's boundary bookkeeping: the call's start epoch, rounds issued and whether it retired
/// a request. A lane records its work only through it and decides only through [`LaneControl::decide`].
#[derive(Clone, Copy, Debug)]
pub(super) struct LaneControl {
    released_at: u64,
    rounds: u64,
    retired: bool,
}

impl LaneControl {
    fn start(released_at: u64) -> Self { Self { released_at, rounds: 0, retired: false } }
    /// This lane retired a request (its units released): work, like a round.
    pub fn retired(&mut self) { self.retired = true; }
    pub fn round_issued(&mut self) { self.rounds += 1; }
    pub fn rounds(&self) -> u64 { self.rounds }
    /// The boundary decision after this lane's retirements: stop for a due prefill round (the
    /// shared decode share's deadline, `prefill_due`), else [`boundary`] over its work so far.
    pub fn decide(&self, prefill_due: bool, stalled: impl Iterator<Item = bool>, epoch: u64,
        members: impl Iterator<Item = usize>) -> Boundary {
        if prefill_due { return Boundary::Stop; }
        boundary(self.rounds + u64::from(self.retired), stalled, epoch, self.released_at, members)
    }
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

async fn retire_completed<M: round_failure::RoundMember, F: std::future::Future<Output = Result<()>>>(
    active: &RefCell<&mut [Option<M>]>, slots: impl IntoIterator<Item = usize>, lane: usize, round_id: u64,
    mut release: impl FnMut(M) -> F,
) {
    for slot in slots {
        let request = active.borrow_mut()[slot].take().unwrap();
        let request_id = request.id();
        if let Err(error) = release(request).await {
            tracing::error!(lane, request_id, %error,
                "independent lane request release failed; the serve loop continues");
        } else {
            tracing::debug!(target: "cuteafd::lane_schedule", lane, request_id, round_id,
                "independent lane request retired");
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn lane<'a, P: VerificationTarget<'a>, C: DraftChain<'a>>(lease: LaneLease<'_>, mut control: LaneControl,
    pass: &mut P, transport: &mut P::Transport, requests: &RefCell<&mut Requests<'a>>,
    active: &RefCell<&mut [Option<Active<'a>>]>, draft: &RefCell<Option<&mut DraftRuntime<'_, 'a, C>>>,
    prefixes: &RefCell<&mut PrefixCache<'a>>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>,
) -> Result<()> {
    let lane = lease.index();
    {
        let mut round_id = 0u64;
        loop {
            if lease.stopping() { return Ok(()); }
            // This lane has completed all of its own GPU/transport work. Retire
            // only its requests; the peer need not stop or migrate survivors.
            let retired: Vec<_> = active.borrow().iter().enumerate().filter_map(|(slot, entry)|
                entry.as_ref().filter(|r| retire_due(lane, r.lane(), r.finished, r.job.events.is_closed()))
                    .map(|_| slot)).collect();
            if !retired.is_empty() { control.retired(); }
            retire_completed(active, retired, lane, round_id,
                |request| retire(lane, request, requests, prefixes, draft)).await;
            // Admission and prefill rounds run between lane sets: both lanes stop.
            let members: Vec<_> = {
                let active = active.borrow();
                if wake.poll_media(round_id)
                    || (!wake.media_pending && wake.ready(active.iter().flatten().count(), active.len(), !receive.is_empty(),
                        requests.borrow().release_epoch())) {
                    lease.stop(); return Ok(());
                }
                let stalled = active.iter().flatten().filter(|r| r.stalled).map(|r| r.job.events.is_closed());
                match control.decide(wake.prefill_due(round_id), stalled, requests.borrow().release_epoch(),
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
            control.round_issued();
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

/// Whether lane `lane` retires a request owned by `owner` at its boundary: a finished one, or one
/// whose client left (stalled or not).
pub(super) fn retire_due(lane: usize, owner: usize, finished: bool, closed: bool) -> bool {
    owner == lane && (finished || closed)
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
    use super::{boundary, drive_lanes, retire_completed, retire_due, round_failure, Boundary, LaneControl};
    use crate::shared::serve::lanes::{LaneLease, LaneSet};
    use std::cell::{Cell, RefCell};

    /// A request of the lane model: its lane, stalled or not, client closed, rounds it still runs
    /// before finishing (its turn then frees units).
    #[derive(Clone, Debug)]
    struct Req { lane: usize, stalled: bool, closed: bool, left: u32 }

    /// One lane over the model, structured as the production `lane`: the production retire
    /// filter, retirements and rounds recorded only through [`LaneControl`], the decision only
    /// through it, a yield per round like the device waits.
    async fn model_lane(lease: LaneLease<'_>, mut control: LaneControl, active: &RefCell<Vec<Option<Req>>>,
        epoch: &Cell<u64>, rounds: &[Cell<u32>; 2]) -> anyhow::Result<()> {
        let index = lease.index();
        loop {
            if lease.stopping() { return Ok(()); }
            let retired: Vec<usize> = active.borrow().iter().enumerate().filter_map(|(slot, r)|
                r.as_ref().filter(|r| retire_due(index, r.lane, r.left == 0, r.closed)).map(|_| slot)).collect();
            for slot in retired {
                control.retired();
                active.borrow_mut()[slot] = None;
                epoch.set(epoch.get() + 1);
            }
            let decision = {
                let active = active.borrow();
                control.decide(false, active.iter().flatten().filter(|r| r.stalled).map(|r| r.closed), epoch.get(),
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
            control.round_issued();
            tokio::task::yield_now().await;
        }
    }

    /// One `independent::run` call over the model: the production [`drive_lanes`] (one epoch
    /// sampled for both lanes, the real `LaneSet` with one stop cell and `tokio::join!` polling
    /// lane 0 first). Returns rounds run per lane.
    fn run_lanes(active: &RefCell<Vec<Option<Req>>>, epoch: &Cell<u64>) -> [u32; 2] {
        let rounds = [Cell::new(0u32), Cell::new(0u32)];
        let set = LaneSet::default();
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(drive_lanes(&set, || epoch.get(),
            |lease, control| model_lane(lease, control, active, epoch, &rounds),
            |lease, control| model_lane(lease, control, active, epoch, &rounds))).unwrap();
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
        // Next call: nothing runnable and the stall not yet replanned: no lane runs a round.
        assert_eq!(run_lanes(&active, &epoch), [0, 0]);
    }

    /// Lane 0's only member is a closed client retired in round 0 (no await), which frees units;
    /// the peer lane (polled second) must see that release and stop at its first boundary
    /// instead of running until its own top-up need. Fails if `drive_lanes` samples the epoch
    /// per lane or if a retirement is not recorded as work.
    #[test]
    fn a_round_zero_retirement_on_one_lane_resumes_a_stall_on_the_other() {
        let active = RefCell::new(vec![Some(Req { lane: 0, stalled: false, closed: true, left: 50 }),
            Some(Req { lane: 1, stalled: true, closed: false, left: 50 }),
            Some(Req { lane: 1, stalled: false, closed: false, left: 1000 })]);
        let epoch = Cell::new(3);
        let rounds = run_lanes(&active, &epoch);
        assert_eq!(epoch.get(), 4);
        assert_eq!(rounds, [0, 0], "lane 0 stopped at its retirement boundary; lane 1 never ran");
        assert_eq!(active.borrow()[2].as_ref().unwrap().left, 1000);
    }

    /// `drive_lanes` samples the release epoch once, before either lane is polled: a release on
    /// lane 0 before lane 1's first poll is still a release for lane 1. (Fails if each lane
    /// samples the epoch when it starts.)
    #[test]
    fn both_lanes_compare_against_one_epoch_sampled_before_either_runs() {
        let epoch = Cell::new(10);
        let seen = [Cell::new(0u64), Cell::new(0u64)];
        let set = LaneSet::default();
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(drive_lanes(&set, || epoch.get(),
            |lease, control: LaneControl| { let (epoch, seen) = (&epoch, &seen); async move {
                // Lane 0 releases units before lane 1 is first polled.
                epoch.set(epoch.get() + 1);
                seen[lease.index()].set(control.released_at);
                anyhow::Ok(())
            } },
            |lease, control: LaneControl| { let seen = &seen; async move {
                seen[lease.index()].set(control.released_at);
                anyhow::Ok(())
            } })).unwrap();
        assert_eq!((seen[0].get(), seen[1].get()), (10, 10));
        // And lane 1's decision sees lane 0's release.
        let control = LaneControl { released_at: 10, rounds: 1, retired: false };
        assert_eq!(control.decide(false, [false].into_iter(), 11, [3].into_iter()), Boundary::Stop);
        // A due prefill round stops the lanes whatever else holds.
        let fresh = LaneControl { released_at: 11, rounds: 0, retired: false };
        assert_eq!(fresh.decide(true, std::iter::empty(), 11, [3].into_iter()), Boundary::Stop);
        assert_eq!(fresh.decide(false, std::iter::empty(), 11, [3].into_iter()), Boundary::Run(vec![3]));
    }

    /// A stalled request whose client left is retired by its owning lane at its first boundary
    /// (the production filter retires closed requests, stalled or not). Nothing is stalled after
    /// that, so there is nothing to replan: the peer lane keeps running its request to the end.
    /// A second stalled request on the peer's side resumes at the next boundary instead.
    #[test]
    fn a_closed_stalled_client_is_retired_by_its_lane() {
        let active = RefCell::new(vec![Some(Req { lane: 0, stalled: true, closed: true, left: 5 }),
            Some(Req { lane: 1, stalled: false, closed: false, left: 100 })]);
        let epoch = Cell::new(0);
        assert_eq!(run_lanes(&active, &epoch), [0, 100]);
        assert!(active.borrow()[0].is_none(), "retired by lane 0");
        assert_eq!(epoch.get(), 2, "both retirements released units");
        let active = RefCell::new(vec![Some(Req { lane: 0, stalled: true, closed: true, left: 5 }),
            Some(Req { lane: 1, stalled: true, closed: false, left: 5 }),
            Some(Req { lane: 1, stalled: false, closed: false, left: 100 })]);
        let epoch = Cell::new(0);
        assert_eq!(run_lanes(&active, &epoch), [0, 0], "the release resumes the other stall at once");
        assert_eq!(active.borrow()[2].as_ref().unwrap().left, 100);
        assert!(retire_due(0, 0, false, true) && !retire_due(1, 0, true, true) && !retire_due(0, 0, false, false));
        assert_eq!(boundary(0, [true].into_iter(), 0, 0, [1].into_iter()), Boundary::Run(vec![1]));
        assert_eq!(boundary(5, std::iter::empty(), 9, 0, std::iter::empty()), Boundary::Done);
    }

    #[test]
    fn retirement_error_does_not_fail_either_lane_or_skip_next_retirement() {
        let (first, mut first_output, _) = round_failure::tests::active_fixture(1, 0);
        let (second, mut second_output, _) = round_failure::tests::active_fixture(2, 0);
        let (peer, mut peer_output, _) = round_failure::tests::active_fixture(3, 1);
        let mut active = [Some(first), Some(second), Some(peer)];
        let mut released = Vec::new();
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(retire_completed(&RefCell::new(active.as_mut_slice()), [0, 1], 0, 1, |request| {
            let id = request.id();
            released.push(id);
            std::future::ready(if id == 1 { Err(anyhow::anyhow!("injected release failure")) } else { Ok(()) })
        }));
        assert_eq!(released, [1, 2], "cleanup continues after a per-request error");
        assert!(active[0].is_none() && active[1].is_none());
        assert!(!active[2].as_ref().unwrap().failed, "peer lane survives");
        assert!(first_output.try_recv().is_err());
        assert!(second_output.try_recv().is_err());
        assert!(peer_output.try_recv().is_err());
    }
}
