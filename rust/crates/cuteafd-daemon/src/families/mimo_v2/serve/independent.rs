//! Lane-local DFlash rounds; only extracted request owners cross Spark waits.
use super::*;
use super::super::engine::{DecodeLane, RoundBank};

#[allow(clippy::too_many_arguments)]
pub(super) async fn lane_rounds<'a>(lane: DecodeLane<'_, '_, 'a>, selector: &mut TokenSelector<'a>,
    bank: &RoundBank<Active<'a>>, shared: &mut Option<crate::shared::draft::runtime::SharedPolicy>,
    width: usize, policy: Policy, prefix: &PrefixArgs, cache_enabled: bool,
    receive: &mpsc::Receiver<NativeRequest>, admission_pending: bool,
    trace: &mut Option<crate::shared::speculation_trace::Trace>) -> Result<()> {
    let (mut observed_rounds, mut skipped_rounds) = (0u64, 0u64);
    let (mut verify_s, mut draft_s, mut emit_s) = (0f64, 0f64, 0f64);
    loop {
        if lane.stopping() { return Ok(()); }
        if let Some(reason) = cuteafd_transport::health::failure_reason() {
            anyhow::bail!("expert wire unavailable until restart: {reason}");
        }
        let mut round = bank.extract(|a| a.lane == lane.index());
        if round.is_empty() { return Ok(()); }
        let mut active: Vec<_> = round.iter_mut().collect();
        let cycle = Instant::now();
        let mut round_clock = RoundClock::at(cycle);
        let mut drafted_requests = 0;
        let mut drafted_width = 0;
        let mut tally = console::Step::begin(lane.index());
        let (draft0, emit0) = (draft_s, emit_s);
        let engine_before = tally.live().then(|| lane.profile());
        // Rows each sequence may add after its next token.
        let room = DECODE_ROWS.saturating_sub(active.len());
        let limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else { room.min(a.job.max_tokens - a.generated - 1)
            .min(a.capacity - a.placement.len - 1) }).collect();
        let ready: Vec<_> = active.iter().map(|a| a.slot.is_some()).collect();
        let draft_rooms = eligible_draft_rooms(&limits, &ready, width);
        let action_requests: Vec<_> = active.iter().zip(&limits).map(|(a, &room)| (a.ticket.id(), room)).collect();
        let run_draft = match shared.as_mut() {
            Some(shared) => select_draft_action(&mut shared.policy, &action_requests, &draft_rooms,
                width, DECODE_ROWS, policy.fixed)?,
            None => false,
        };
        // DFlash drafts after every next token (sequences with a ring slot),
        // then the adaptive plan's counts.
        let drafted: Vec<Option<super::super::dflash::Draft>> = if run_draft {
            let seqs: Vec<_> = active.iter().enumerate().filter_map(|(i, a)|
                a.slot.filter(|_| draft_rooms[i] > 0).map(|slot| (i, DraftSeq { slot,
                    anchor: a.next, position: a.placement.len, valid_from: a.draft_from }))).collect();
            for &(i, _) in &seqs { active[i].counts[5] += 1; }
            drafted_requests = seqs.len();
            drafted_width = width;
            let timer = Instant::now();
            let drafts = round_clock.draft(|| lane.draft(&seqs.iter().map(|(_, s)| *s).collect::<Vec<_>>()))?;
            draft_s += timer.elapsed().as_secs_f64();
            let mut out = vec![None; active.len()];
            for ((i, _), draft) in seqs.into_iter().zip(drafts) { out[i] = Some(draft); }
            out
        } else { vec![None; active.len()] };
        let plan_timer = Instant::now();
        let priors: Vec<Vec<f64>> = active.iter().zip(&drafted).map(|(a, draft)| {
            let (Some(shared), Some(draft)) = (shared.as_ref(), draft.as_ref()) else { return Vec::new() };
            shared.prior(&a.drafts, &resource::DraftEvidence::Selector(&draft.features),
                crate::shared::draft::copy::COPY_EXTENSION)
        }).collect();
        let copy_agreement = trace.as_ref().map(|_| active.iter().zip(&drafted).map(|(a, d)|
            d.as_ref().map_or(0, |d| crate::shared::draft::copy::agreement(&a.history, &d.tokens)))
            .collect::<Vec<_>>());
        let copy_windows = trace.as_ref().map(|_| active.iter().zip(&drafted).map(|(a, d)|
            crate::shared::draft::copy::trace_window(&a.history,
                d.as_ref().map_or(0, |d| d.tokens.len() + crate::shared::draft::copy::COPY_EXTENSION)))
            .collect::<Vec<_>>());
        let mut priors = priors;
        let copies: Vec<_> = active.iter().zip(&drafted).zip(&limits).zip(&mut priors)
            .map(|(((a, d), &limit), prior)| match (shared.as_ref(), d.as_ref()) {
                (Some(shared), Some(d)) if policy.fixed.is_none() => resource::copy_continuation(&shared.policy, active.len(),
                    a.ticket.id(), &a.history, &d.tokens, prior, limit, a.job.sampling.is_greedy()),
                _ => None,
            }).collect();
        for (i, prior) in priors.iter_mut().enumerate() {
            let available = copies[i].as_ref().map_or_else(
                || drafted[i].as_ref().map_or(0, |draft| draft.tokens.len()), |copy| copy.tokens.len());
            prior.truncate(available);
        }
        let mut predicted = None;
        let mut planned = if let Some(fixed) = policy.fixed {
            drafted.iter().zip(&limits).map(|(d, &limit)| d.as_ref().map_or(0, |d| fixed.min(limit).min(d.tokens.len()))).collect()
        } else if let Some(shared) = shared.as_mut().filter(|_| drafted_width > 0) {
            let candidates: Vec<_> = active.iter().zip(&priors).zip(&limits).map(|((a, prior), &limit)|
                resource::Candidate { id: a.ticket.id(), prior, limit, selector: false, cold: a.drafts.cold() }).collect();
            let selection = resource::select_drafted(&mut shared.policy, &candidates, drafted_width, DECODE_ROWS, drafted_requests);
            predicted = selection.predicted;
            selection.lengths
        } else { vec![0; active.len()] };
        if resource::trim_budget(&mut planned, DECODE_ROWS) { predicted = None; }
        if policy.fixed.is_some() {
            if let Some(shared) = shared.as_ref() {
                let candidates: Vec<_> = active.iter().zip(&priors).map(|(a, prior)|
                    cuteafd_core::DraftCandidate { id: a.ticket.id(), confidence: prior }).collect();
                crate::shared::draft::binding::trace_selection(&shared.policy, &candidates,
                    drafted_width, drafted_requests, DECODE_ROWS, &planned);
            }
        }
        let sequences: Vec<Vec<u32>> = active.iter().enumerate().map(|(i, a)| {
            let full = copies[i].as_ref().map(|c| c.tokens.as_slice())
                .unwrap_or_else(|| drafted[i].as_ref().map_or(&[][..], |d| d.tokens.as_slice()));
            let mut rows: Vec<u32> = std::iter::once(a.next).chain(full.iter().take(planned[i]).copied()).collect();
            if let Some(state) = a.constraint.as_ref() { state.truncate_proposal(&mut rows)?; }
            Ok(rows)
        }).collect::<Result<_>>()?;
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
        let plan_us = console::us(plan_timer);
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        let timer = Instant::now();
        // A sequence whose grammar fails is failed alone after the step; an
        // error inside `submit` would end the engine.
        let mut poisoned: Vec<Option<String>> = vec![None; active.len()];
        cuteafd_transport::route_log::step(|| active.iter().zip(&sequences).map(|(a, s)| (a.ticket.id(), s.len() as u32)).collect());
        lane.probe_begin(tokens.len());
        let mut rows: Vec<(&mut MimoPlacement, usize)> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.len())).collect();
        let logits = lane.verify(&mut rows, &tokens).await?;
        drop(rows);
        let mut batch = SelectBatch::default();
        for (i, ((a, rows), &start)) in active.iter().zip(&sequences).zip(&starts).enumerate() {
            poisoned[i] = batch.push_sequence_isolated(a.job.sampling, a.constraint.as_ref(), rows, start as u64 + 1);
        }
        let selected = lane.sample(selector, &logits, &batch)?;
        lane.probe_end();
        let step_s = timer.elapsed().as_secs_f64();
        verify_s += step_s;
        let timer = Instant::now();
        let mut offset = 0;
        let mut context = Vec::new();
        let before: Vec<usize> = active.iter().map(|a| a.history.len()).collect();
        let mut verified = vec![(0usize, 1usize, Some(Censor::Cancelled)); active.len()];
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(&starts).enumerate()
            .map(|(i, ((request, rows), &start))| {
            if let Some(error) = poisoned[i].take() {
                let _ = request.job.events.send(Err(NativeFailure::Worker(error)));
                verified[i] = (rows.len(), 1, Some(Censor::Cancelled));
                offset += rows.len();
                return true;
            }
            let mut finished = false;
            let mut cancelled = false;
            let mut last = None;
            for j in 0..rows.len() {
                // Rows 0..=j are committed; the token row j produces is next.
                request.placement.len = start + j + 1;
                probe::decode_row(lane.library(), &request.job.probe, &logits, offset + j, request.generated,
                    request.history.len());
                match take(request.constraint.as_mut(), &selected[offset + j]).and_then(|t| Ok((t, request.emit(t)?))) {
                    Ok((token, done)) => {
                        last = Some(token);
                        finished = done;
                        if done && cache_enabled && !probe::cold(&request.job.probe) {
                            // A normal finish (the client took the last chunk): the row that
                            // produced the last token follows the turn snapshot.
                            request.turn = logits.row_host(lane.library(), offset + j).ok();
                        }
                        if done || rows.get(j + 1) != Some(&token) {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        finished = true;
                        cancelled = true;
                        break;
                    }
                }
            }
            let committed = request.placement.len - start;
            if let Some(slot) = request.slot.filter(|_| !finished || (prefix.mimo_prefix_draft && request.turn.is_some())) {
                context.extend((0..committed).map(|r| ContextRow { tap_row: offset + r, slot, position: start + r }));
            }
            offset += rows.len();
            let (proposed, accepted) = (rows.len() - 1, committed - 1);
            if prefix.mimo_prefix_draft && request.counts[0] == 0 {
                tracing::info!(position = start, valid_from = request.draft_from, drafted = proposed, accepted,
                    copy = false, neural_calls = request.counts[5], "first draft verification");
            }
            request.counts[0] += 1;
            let censor = if cancelled { Some(Censor::Cancelled) } else {
                resource::finish_censor(finished, committed, rows, last, &request.job.stop_token_ids) };
            verified[i] = (rows.len(), committed, censor);
            request.counts[1] += proposed;
            request.counts[2] += accepted;
            let neural = proposed.min(copies[i].as_ref().map_or_else(
                || drafted[i].as_ref().map_or(0, |d| d.tokens.len()), |c| c.neural));
            request.drafts.observe(neural.min(accepted + usize::from(censor.is_none())), accepted.min(neural));
            finished
        }).collect();
        if !context.is_empty() { lane.update_context(&context)?; }
        emit_s += timer.elapsed().as_secs_f64();
        if let Some(shared) = shared.as_mut() {
            let requests: Vec<_> = active.iter().enumerate().map(|(i, a)| {
                let (rows, accepted, censor) = verified[i];
                let neural = drafted[i].as_ref().map_or(0, |draft| draft.tokens.len())
                    .min(copies[i].as_ref().map_or(usize::MAX, |copy| copy.neural))
                    .min(rows.saturating_sub(1)).min(priors[i].len());
                resource::Verified { id: a.ticket.id(), rows, accepted,
                    source: if neural > 0 { DraftSource::Neural } else { DraftSource::Undrafted },
                    prior: (neural > 0).then_some(&priors[i][..neural]),
                    features: drafted[i].as_ref().map(|d| &d.features[..neural]), censor }
            }).collect();
            let times = round_clock.observe();
            match lane.probe_finish(|routes, layer_us| resource::observe_drafted(&mut shared.policy,
                &requests, routes, layer_us, times, drafted_width, predicted, Some(drafted_requests))) {
                Some(Ok(())) => observed_rounds += 1,
                Some(Err(error)) => {
                    skipped_rounds += 1;
                    tracing::debug!(%error, "MiMo draft round not observed");
                }
                None => {
                    skipped_rounds += 1;
                    tracing::debug!("MiMo draft round probe incomplete");
                }
            }
            resource::publish(&shared.policy, observed_rounds, skipped_rounds, None);
        }
        if let Some(trace) = trace.as_mut() {
            let line = serde_json::json!({"rows": sequences.iter().map(Vec::len).sum::<usize>(),
                "draft_width": drafted_width, "drafted_requests": drafted_requests,
                "lengths": sequences.iter().map(|s| s.len()-1).collect::<Vec<_>>(),
                "accepted": verified.iter().map(|v| v.1.saturating_sub(1)).collect::<Vec<_>>(),
                "neural_lengths": drafted.iter().map(|d| d.as_ref().map_or(0, |d| d.tokens.len())).collect::<Vec<_>>(),
                "request_ids": active.iter().map(|a| a.ticket.id()).collect::<Vec<_>>(),
                "neural_tokens": drafted.iter().map(|d| d.as_ref().map(|d| &d.tokens)).collect::<Vec<_>>(),
                "copy_windows": copy_windows.as_ref().map(|windows| windows.iter().map(|(start, suffix, tokens)|
                    serde_json::json!({"start": start, "matched_suffix": suffix, "tokens": tokens})).collect::<Vec<_>>()),
                "committed_tokens": active.iter().zip(&before).map(|(a, &n)| &a.history[n..]).collect::<Vec<_>>(),
                "copy_agreement": copy_agreement,
                "copy_neural_prefix": copies.iter().map(|c| c.as_ref().map(|c| c.neural)).collect::<Vec<_>>(),
                "copy_lengths": copies.iter().map(|c| c.as_ref().map_or(0, |c| c.tokens.len())).collect::<Vec<_>>(),
                "prior": priors, "predicted_us": predicted, "policy": resource::snapshot()});
            if let Err(error) = trace.write(&line) {
                tracing::warn!(%error, "MiMo speculation trace write failed");
            }
        }
        for (i, request) in active.iter().enumerate() {
            let proposal = &sequences[i][1..];
            tally.member(&request.ticket, proposal, sequences[i].len() - 1, &request.history[before[i]..],
                request.constraint.is_some(), finished[i]);
        }
        tally.end(|| {
            let engine_now = lane.profile();
            let gpu = |i: usize| engine_before.map_or(f64::NAN, |before| 1e6 * (engine_now[i] - before[i]));
            vec![("draft", 1e6 * (draft_s - draft0)), ("plan", plan_us), ("verify", 1e6 * step_s),
                ("emit", 1e6 * (emit_s - emit0)), ("gpu", gpu(0)), ("experts", gpu(1))]
        });
        let stop = finished.iter().any(|done| *done) || admission_pending || !receive.is_empty()
            || active.iter().any(|a| a.placement.len.saturating_add(DECODE_ROWS) > a.capacity
                || a.placement.len.saturating_add(DECODE_ROWS) > a.placement.pages.len() * 64);
        for (request, done) in active.iter_mut().zip(finished) { request.finished = done; }
        // The guard restores the exact stable bank slots before the next loop,
        // a normal stop, or the enclosing whole-future failure boundary.
        drop(active);
        drop(round);
        if stop { lane.stop(); return Ok(()); }
    }
}
