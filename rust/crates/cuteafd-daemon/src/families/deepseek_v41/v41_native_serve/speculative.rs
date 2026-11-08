use super::*;
use crate::families::deepseek_v41::v41_dspark_cache::{DsparkWindow, WindowLease};
use cuteafd_ffi::CuteafdDeviceBuffer;
/// Bytes of one request's retained rings in a draft mark: three windows' full slots.
pub(crate) const DRAFT_MARK_BYTES: usize = 3 * cuteafd_ffi::V41DsparkCache::SLOT_BYTES;
use crate::families::deepseek_v41::v41_experts::dspark::{DsparkChain, DsparkMainContext, DsparkWeights};
use crate::families::deepseek_v41::v41_requests::RequestBatch;
use crate::families::deepseek_v41::v41_backbone_lane::Capture;
use crate::shared::draft::binding::LaneRound;
use crate::shared::draft::clock::RoundTimes;
use crate::shared::draft::evidence::sigmoid;
mod chain;
mod policy;
pub(crate) use policy::LaneRequest;

static TAIL_REQUESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TAIL_PROPOSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TAIL_REACHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TAIL_ACCEPTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAFT_FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One lane round's draft seeds: (request identity, anchor, committed end,
/// remaining output budget).
type DraftSeed = (u64, u32, u64, usize);

/// Reset a lane's draft state before a new proposal attempt. A proposal that
/// fails in its guards must not leave the previous round's width, drafted
/// request count or confidence trace behind for `select_lengths`/`observe`.
pub(crate) fn reset_lane_draft(lane: &mut LaneRound, confidence_trace: &mut std::collections::BTreeMap<u64, Vec<f32>>,
    neural_lengths: &mut std::collections::BTreeMap<u64, usize>, inputs: &[DraftSeed]) {
    lane.width = 0;
    lane.drafted_requests = 0;
    for &(id, _, _, _) in inputs {
        confidence_trace.remove(&id);
        neural_lengths.remove(&id);
    }
}

/// The anchor-only inputs a failed proposal verifies.
pub(crate) fn anchor_inputs(seeds: &[DraftSeed]) -> Vec<Vec<u32>> {
    seeds.iter().map(|seed| vec![seed.1]).collect()
}

/// Whether a pending proposal still holds request `id` (and so blocks release).
pub(crate) fn pending_draft_blocks(pending: &[Option<(Vec<DraftSeed>, Instant)>], id: u64) -> bool {
    pending.iter().flatten().any(|(seeds, _)| seeds.iter().any(|seed| seed.0 == id))
}

pub(crate) fn policy_snapshot() -> serde_json::Value {
    use std::sync::atomic::Ordering::Relaxed;
    let mut snapshot = policy::snapshot();
    snapshot["copy_tail"] = serde_json::json!({
        "requests": TAIL_REQUESTS.load(Relaxed), "proposed": TAIL_PROPOSED.load(Relaxed),
        "reached": TAIL_REACHED.load(Relaxed), "accepted": TAIL_ACCEPTED.load(Relaxed),
    });
    snapshot["draft_fallbacks"] = serde_json::json!(DRAFT_FALLBACKS.load(Relaxed));
    snapshot
}

fn tail_outcome(rows: usize, accepted: usize, neural: usize) -> (usize, usize, usize) {
    let proposed = rows.saturating_sub(neural + 1);
    let reached = if accepted <= neural { 0 } else { proposed.min(accepted - neural) };
    let accepted = accepted.saturating_sub(neural + 1).min(proposed);
    (proposed, reached, accepted)
}
pub(crate) use chain::{DraftChain, ProposeChain};
mod distributed;

/// The process-wide terminal signal for the expert wire. A draft that runs on
/// a terminal engine is not a reusable target-only fallback: the lane must
/// fail the round instead of verifying anchors forever.
pub(crate) fn require_peer_live() -> Result<()> {
    if let Some(reason) = cuteafd_transport::health::failure_reason() {
        anyhow::bail!("draft peer wire unavailable until restart: {reason}");
    }
    Ok(())
}

/// Execution workspaces share a request-indexed bank of persistent draft state.
pub(crate) struct DraftRuntime<'w, 'a, C = DsparkChain<'w, 'a>> {
    mains: Vec<DsparkMainContext<'w, 'a>>,
    pending_commit_ids: Vec<Vec<u64>>,
    pending_prefix_ids: [Option<u64>; 2],
    chains: Vec<C>,
    windows: [DsparkWindow<'a>; 3],
    requests: std::collections::BTreeMap<u64, DraftRequest>,
    pending: Vec<Option<(Vec<(u64, u32, u64, usize)>, Instant)>>,
    request_limit: usize,
    draft_limit: usize,
    draft_width: usize,
    // Downloaded whenever the length policy is bound, or for diagnostics.
    confidence_trace: std::collections::BTreeMap<u64, Vec<f32>>,
    neural_lengths: std::collections::BTreeMap<u64, usize>,
    /// Shared fixed-length override, absent for adaptive selection.
    length: crate::shared::draft::length::DraftLength,
    /// Online bandwidth-balance length policy, bound to the installed placement.
    policy: Option<cuteafd_core::DraftPolicy>,
    published: Option<Instant>,
    /// Per lane: the round's draft width, the last selection's regime and
    /// prediction, and the route storage its observation reuses.
    lanes: Vec<LaneRound>,
}
struct DraftRequest {
    leases: [WindowLease; 3],
    rng: cuteafd_core::DsparkRng,
    slot: usize,
}
impl<'w, 'a> DraftRuntime<'w, 'a> {
    pub fn with_requests(
        lib: &'a NativeLibrary, weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>, head: &'w VocabularyHead<'a>,
        capacity: u32, requests: u32,
    ) -> Result<Self> {
        ensure!((1..=cuteafd_core::V41_MAX_REQUESTS as u32).contains(&requests), "invalid draft request limit");
        let window = || DsparkWindow::new(lib, requests as usize, capacity,
            DsparkWindow::device_bytes(requests as usize, capacity)?);
        let lane_count = if requests == 1 { 1 } else { 2 };
        let lane_requests = requests.div_ceil(lane_count as u32);
        let shared_bytes = weights.draft_bytes(lane_requests)?;
        let lane_bytes = weights.draft_bytes(lane_requests)?;
        let mut mains = vec![weights.main_context(capacity, DsparkMainContext::device_bytes(lib, capacity)?)?];
        if lane_count > 1 { mains.push(weights.main_context(cuteafd_core::V41_VERIFY_ROWS as u32, DsparkMainContext::device_bytes(lib, cuteafd_core::V41_VERIFY_ROWS as u32)?)?); }
        let chains = (0..lane_count).map(|_| weights.draft(table, head, lane_requests, lane_bytes))
            .collect::<Result<Vec<_>>>()?;
        tracing::info!(lanes=lane_count, lane_requests, draft_width=weights.draft_width(), shared_workspace_bytes=shared_bytes,
            lane_workspace_bytes=lane_bytes, total_workspace_bytes=lane_bytes*lane_count,
            additional_workspace_bytes=(lane_bytes*lane_count).saturating_sub(shared_bytes),
            "lane-local dSpark draft workspaces (weights shared)");
        Ok(Self {
            mains,
            pending_commit_ids: vec![Vec::new(); lane_count],
            pending_prefix_ids: [None, None],
            chains,
            windows: [window()?, window()?, window()?],
            requests: Default::default(),
            pending: vec![None; lane_count],
            request_limit: requests as usize,
            draft_limit: 5,
            draft_width: weights.draft_width(),
            confidence_trace: Default::default(),
            neural_lengths: Default::default(),
            length: Default::default(),
            policy: None,
            published: None,
            lanes: (0..lane_count).map(|_| LaneRound::new(policy::LAYERS, policy::TOPK)).collect(),
        })
    }
}
impl<'w, 'a> DraftRuntime<'w, 'a, DsparkChain<'w, 'a>> {
    /// Seal the draft proposal inventory: every admitted count and both widths
    /// are literal keys, so a sealed miss runs eagerly and allocates nothing.
    pub(crate) fn capture_startup(&mut self) -> Result<()> {
        ensure!(self.requests.is_empty() && self.pending.iter().all(Option::is_none),
            "draft startup requires no admitted requests");
        let windows = self.windows.each_ref();
        for chain in &mut self.chains { chain.capture_startup(windows)?; }
        Ok(())
    }
}
impl<'w, 'a, C: DraftChain<'a>> DraftRuntime<'w, 'a, C> {
    pub fn admit(&mut self, id: u64) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(!self.requests.contains_key(&id), "draft request already admitted");
        let slot = (0..self.request_limit).find(|slot| self.requests.values().all(|request| request.slot != *slot))
            .context("draft request capacity exhausted")?;
        let mut leases = Vec::new();
        for stage in 0..3 {
            match self.windows[stage].begin_request(slot, id) {
                Ok(lease) => leases.push(lease),
                Err(error) => {
                    for (stage, lease) in leases.into_iter().enumerate() {
                        let _ = self.windows[stage].release(lease);
                    }
                    return Err(error);
                }
            }
        }
        self.requests.insert(id, DraftRequest {
            leases: leases.try_into().ok().context("draft admission incomplete")?,
            rng: cuteafd_core::DsparkRng::new(id),
            slot,
        });
        Ok(())
    }
    pub(super) fn prefill_tap_capacity(&self) -> usize { self.mains[0].capacity() }
    pub fn max_verify_rows(&self) -> usize { self.draft_width + crate::shared::draft::copy::COPY_EXTENSION + 1 }
    pub fn attention_drafts(&self) -> usize { self.draft_limit.min(self.draft_width) }
    pub fn set_length(&mut self, length: crate::shared::draft::length::DraftLength) -> Result<()> {
        self.draft_limit = length.draft_length.unwrap_or(5).min(self.draft_width);
        self.length = length;
        Ok(())
    }
    /// Bind the length policy to the installed expert placement before serving.
    pub fn configure_policy(&mut self, transport: &NativeTp4Wave<'_>, nvfp4: bool) -> Result<()> {
        let geometry = policy::placement(transport, nvfp4, self.draft_width)?;
        tracing::info!(fixed=self.length.draft_length.is_some(), draft_limit=self.draft_limit, local_layers=policy::local_layers(&geometry),
            remote_expert_bytes=policy::remote_expert_bytes(&geometry),
            "dSpark bandwidth-balance length policy bound");
        self.bind_policy(geometry)
    }
    /// Bind the policy to a V4.1 geometry. Width 5 is the drafter's trained
    /// block; a width-7 load can also draft 5 each round, chosen by the policy.
    pub(crate) fn bind_policy(&mut self, mut geometry: cuteafd_core::PolicyGeometry) -> Result<()> {
        ensure!(geometry.widths.last() == Some(&self.draft_width), "policy widths differ from the loaded draft width");
        geometry.max_positions = self.draft_width + crate::shared::draft::copy::COPY_EXTENSION;
        let seed = cuteafd_core::PolicySeed::weak(self.draft_width);
        let policy = self.length.policy(geometry, seed, self.draft_width)?;
        policy::publish(&policy, self.draft_limit);
        self.policy = Some(policy);
        Ok(())
    }
    /// A length policy is bound (it observes every verification round).
    pub fn has_policy(&self) -> bool { self.policy.is_some() }
    /// What a verification round records: see [`round_capture`].
    pub fn capture(&self) -> Capture {
        round_capture(self.policy.as_ref(), super::console::live().is_some(),
            crate::shared::draft::binding::replay_trace().is_some(), crate::shared::draft::timing_profile())
    }
    fn wants_confidence(&self) -> bool {
        self.policy.is_some() || tracing::enabled!(target: "cuteafd::draft_policy", tracing::Level::DEBUG)
    }
    /// Choose each request's draft length for one lane round. `requests` are
    /// (identity, available drafts) in lane order; `shared` means the other lane
    /// has active requests. `None` verifies every available draft while a fit warms up.
    pub fn select_lengths(&mut self, lane: usize, requests: &[(u64, usize)], shared: bool)
        -> Result<Option<Vec<usize>>> {
        let round = self.lanes.get_mut(lane).context("invalid draft policy lane")?;
        round.predicted = None;
        round.shared = shared;
        let width = round.width;
        // Every verifier request is present; undrafted/copy members have zero room.
        if requests.is_empty() { return Ok(None); }
        let Some(policy) = &mut self.policy else { return Ok(None) };
        let probabilities = requests.iter().map(|&(id, maximum)| {
            if maximum == 0 { return Ok(Vec::new()); }
            let logits = self.confidence_trace.get(&id).context("missing draft confidence")?;
            ensure!(maximum <= logits.len(), "draft confidence prefix exceeds output");
            Ok(logits[..maximum].iter().map(|&x| sigmoid(x)).collect::<Vec<_>>())
        }).collect::<Result<Vec<_>>>()?;
        let candidates: Vec<_> = requests.iter().zip(&probabilities)
            .map(|(&(id, _), confidence)| cuteafd_core::DraftCandidate { id, confidence }).collect();
        let started = Instant::now();
        let drafted_requests = round.drafted_requests;
        if let Some(lengths) = crate::shared::draft::resource::fixed_selection(policy, &candidates,
            cuteafd_core::V41_VERIFY_ROWS) { return Ok(Some(lengths)); }
        let selection = policy.select_drafted_within(shared, &candidates, width, cuteafd_core::V41_VERIFY_ROWS, drafted_requests)?;
        Ok(match selection {
            Some(selection) => {
                tracing::debug!(target: "cuteafd::draft_policy", lane, shared,
                    available=?requests.iter().map(|r| r.1).collect::<Vec<_>>(), selected=?selection.lengths,
                    expected_tokens=selection.expected_tokens, predicted_us=selection.predicted_us,
                    evaluated=selection.evaluated, selection_us=started.elapsed().as_micros() as u64,
                    "native length selection");
                self.lanes[lane].predicted = Some(selection.predicted_us);
                Some(selection.lengths)
            }
            None => {
                let ids: Vec<_> = requests.iter().map(|r| r.0).collect();
                let full: Vec<_> = requests.iter().map(|r| r.1).collect();
                self.lanes[lane].predicted = policy.predict_drafted(shared, &ids, &full, width, drafted_requests);
                None
            }
        })
    }
    /// Feed one completed lane round to the policy. `requests` are (identity,
    /// verifier rows, accepted inputs, copied) in lane order; `times` span the
    /// round start to its observation and the draft call alone. Observation
    /// problems are logged, never fatal.
    pub fn observe_round(&mut self, lane: usize, shared: bool, routes: &[Vec<[u32; 6]>],
        layer_us: &[Option<f64>], requests: &[policy::LaneRequest], times: RoundTimes) {
        use std::sync::atomic::Ordering::Relaxed;
        for request in requests {
            if let Some(&neural) = self.neural_lengths.get(&request.id) {
                let (proposed, reached, accepted) = tail_outcome(request.rows, request.accepted as usize, neural);
                TAIL_REQUESTS.fetch_add(u64::from(proposed > 0), Relaxed);
                TAIL_PROPOSED.fetch_add(proposed as u64, Relaxed);
                TAIL_REACHED.fetch_add(reached as u64, Relaxed);
                TAIL_ACCEPTED.fetch_add(accepted as u64, Relaxed);
            }
        }
        let Some(round) = self.lanes.get_mut(lane) else { return };
        let Some(policy) = &mut self.policy else { round.predicted = None; return };
        let predicted = round.predicted;
        if let Err(error) = policy::observe(policy, &self.confidence_trace, round, shared, routes, layer_us,
            requests, times) {
            tracing::warn!(%error, lane, "dSpark policy round observation skipped");
        }
        tracing::debug!(target: "cuteafd::draft_policy", lane, shared, predicted_us=predicted,
            total_us=times.total_us, draft_us=times.draft_us,
            rows=requests.iter().map(|r| r.rows).sum::<usize>(), "native length policy observation");
        if self.published.is_none_or(|at| at.elapsed() >= std::time::Duration::from_millis(250)) {
            policy::publish(policy, self.draft_limit);
            self.published = Some(Instant::now());
        }
    }
    /// Choose a width using every verifier request; nonparticipants have zero
    /// draft room. The neural pass still contains only participating requests.
    fn apply_width(&mut self, lane: usize, chain: usize, requests: &[(u64, usize)]) -> Result<usize> {
        let maximum = self.draft_width;
        let limit = self.draft_limit;
        let width = match &mut self.policy {
            Some(policy) => {
                let requests: Vec<_> = requests.iter()
                    .map(|&(id, room)| (id, room.min(limit))).collect();
                policy.choose_width(self.lanes[lane].shared, &requests)
            }
            None => maximum,
        };
        if let Err(error) = self.chains[chain].set_width(width) {
            self.chains[chain].note_failure(&error);
            return Err(error);
        }
        self.lanes[lane].width = width;
        self.lanes[lane].drafted_requests = requests.iter().filter(|r| r.1 > 0).count();
        Ok(width)
    }
    pub fn release(&mut self, id: u64) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(!self.pending_prefix_ids.contains(&Some(id)), "cannot release a request with pending snapshot copies");
        ensure!(!pending_draft_blocks(&self.pending, id),
            "cannot release a request with a pending draft");
        ensure!(!self.pending_commit_ids.iter().any(|ids| ids.contains(&id)),
            "cannot release a request with a pending commit");
        if let Some(policy) = &mut self.policy { policy.release(id); }
        self.confidence_trace.remove(&id);
        self.neural_lengths.remove(&id);
        let mut failure = None;
        if let Some(request) = self.requests.remove(&id) {
            for (window, lease) in self.windows.iter_mut().zip(request.leases) {
                if window.request_id(lease).is_ok() {
                    if let Err(error) = window.release(lease) {
                        failure = Some(error);
                    }
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
    /// The device every draft window (and so every draft mark) lives on.
    pub fn mark_device(&self) -> crate::shared::memory::device::Device<'a> {
        self.windows[0].device()
    }
    fn ring_at(mark: CuteafdDeviceBuffer, stage: usize) -> CuteafdDeviceBuffer {
        let bytes = cuteafd_ffi::V41DsparkCache::SLOT_BYTES;
        CuteafdDeviceBuffer { ptr: unsafe { mark.ptr.cast::<u8>().add(stage * bytes).cast() }, bytes, ..mark }
    }
    fn check_frontier(&self, id: u64, end: u64) -> Result<[WindowLease; 3]> {
        let request = self.requests.get(&id).context("draft request not admitted")?;
        for (window, lease) in self.windows.iter().zip(request.leases) {
            ensure!(window.committed_end(lease)? == Some(end), "draft and target prefix frontiers differ");
        }
        Ok(request.leases)
    }
    /// Copy request `id`'s three rings at frontier `end` into `mark` (`MARK_BYTES` on
    /// `mark_device`) synchronously.
    pub fn capture_mark(&mut self, id: u64, end: u64, mark: CuteafdDeviceBuffer) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(mark.bytes == DRAFT_MARK_BYTES, "draft mark storage size differs");
        let leases = self.check_frontier(id, end)?;
        let device = self.windows[0].device();
        device.run(|| {
            for (stage, (window, lease)) in self.windows.iter_mut().zip(leases).enumerate() {
                window.capture_ring(lease, Self::ring_at(mark, stage))?;
            }
            Ok(())
        })
    }
    /// [`DraftRuntime::capture_mark`] queued on capture lane `lane`.
    pub fn queue_mark(&mut self, lane: usize, id: u64, end: u64, mark: CuteafdDeviceBuffer) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(self.pending_prefix_ids.get(lane).context("invalid draft snapshot lane")?.is_none(),
            "draft snapshot lane is occupied");
        ensure!(!self.pending_prefix_ids.contains(&Some(id)), "draft snapshot already pending");
        ensure!(mark.bytes == DRAFT_MARK_BYTES, "draft mark storage size differs");
        let leases = self.check_frontier(id, end)?;
        self.pending_prefix_ids[lane] = Some(id);
        let device = self.windows[0].device();
        let queued = device.run(|| {
            for (stage, (window, lease)) in self.windows.iter_mut().zip(leases).enumerate() {
                window.queue_ring(lane, lease, Self::ring_at(mark, stage))?;
            }
            Ok(())
        });
        if let Err(error) = queued {
            if let Err(cleanup) = self.abort_mark(lane) {
                tracing::error!(%cleanup, "draining failed draft snapshot enqueue");
            }
            return Err(error);
        }
        Ok(())
    }
    /// Whether a ring copy is queued on capture lane `lane`.
    pub fn mark_pending(&self, lane: usize) -> bool {
        self.pending_prefix_ids.get(lane).is_some_and(Option::is_some)
    }
    /// The common frontier of request `id`'s three rings.
    pub fn committed_end(&self, id: u64) -> Result<u64> {
        let request = self.requests.get(&id).context("draft request not admitted")?;
        let mut end = None;
        for (window, lease) in self.windows.iter().zip(request.leases) {
            let current = window.committed_end(lease)?.context("draft window is not seeded")?;
            ensure!(end.is_none_or(|e| e == current), "draft windows disagree on the frontier");
            end = Some(current);
        }
        end.context("no draft windows")
    }
    /// Whether lane `lane`'s queued rings all landed; the lane frees once they have.
    pub fn mark_ready(&mut self, lane: usize) -> Result<bool> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(self.pending_prefix_ids.get(lane).is_some_and(Option::is_some), "no draft snapshot is pending");
        let mut ready = true;
        for window in &mut self.windows {
            // A window whose copy already landed freed its lane; poll only the rest.
            if window.lane_pending(lane) { ready &= window.ring_ready(lane)?; }
        }
        if ready { self.pending_prefix_ids[lane] = None; }
        Ok(ready)
    }
    pub fn abort_mark(&mut self, lane: usize) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(lane < self.pending_prefix_ids.len(), "invalid draft snapshot lane");
        let mut failure = None;
        for window in &mut self.windows {
            if let Err(error) = window.abort_ring(lane) { failure.get_or_insert(error); }
        }
        self.pending_prefix_ids[lane] = None;
        failure.map_or(Ok(()), Err)
    }
    /// Restore request `id`'s rings at frontier `end` from `mark`. A failure releases the draft
    /// request.
    pub fn restore_mark(&mut self, id: u64, end: u64, mark: CuteafdDeviceBuffer) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(mark.bytes == DRAFT_MARK_BYTES && mark.device_id == self.windows[0].device().id,
            "draft mark storage differs from the runtime's");
        let leases = self.requests.get(&id).context("draft request not admitted")?.leases;
        let device = self.windows[0].device();
        let restored = device.run(|| {
            for (stage, lease) in leases.into_iter().enumerate() {
                self.windows[stage].restore_ring(lease, end, Self::ring_at(mark, stage))?;
            }
            Ok(())
        });
        if let Err(error) = restored {
            if let Err(cleanup) = self.release(id) {
                tracing::error!(%cleanup, "releasing failed draft prefix restore");
            }
            return Err(error);
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn validate_position(&self, id: u64, end: u64) -> Result<()> {
        let request = self.requests.get(&id).context("draft request not admitted")?;
        for (window, lease) in self.windows.iter().zip(request.leases) {
            ensure!(window.request_id(lease)? == id, "draft request identity differs");
            ensure!(window.committed_end(lease)? == Some(end), "draft position differs");
        }
        Ok(())
    }
    pub fn commit(
        &mut self,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: u32,
    ) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        self.commit_batch(pass, requests, batch, &[accepted])
    }
    pub fn commit_batch(
        &mut self,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: &[u32],
    ) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        let ids = batch.cache()?.request_ids();
        ensure!(ids.len() == accepted.len(), "draft acceptance count differs");
        let leases = ids.iter().map(|id| self.requests.get(id)
            .map(|request| request.leases).context("draft request not admitted"))
            .collect::<Result<Vec<_>>>()?;
        let stage_leases: [Vec<WindowLease>; 3] = std::array::from_fn(|stage|
            leases.iter().map(|request| request[stage]).collect());
        let taps = pass.taps(batch)?;
        let mut proposal = unsafe {
            self.mains[0]
                .execute_rows(taps.values(), taps.batch_identity(), taps.rows())?
        };
        let [a, b, c] = self.windows.as_mut_slice() else {
            anyhow::bail!("draft windows incomplete");
        };
        unsafe {
            pass.commit_with_dspark(
                requests,
                batch,
                &mut proposal,
                &mut [a, b, c],
                [&stage_leases[0], &stage_leases[1], &stage_leases[2]],
                accepted,
            )
        }
    }
    /// Begin a completed decode batch's accepted-cache transaction on its own lane.
    pub fn begin_queued_commit(&mut self, lane: usize, pass: &impl crate::families::deepseek_v41::v41_target_pass::TargetCache<'a>,
        requests: &Requests<'a>, batch: &RequestBatch, accepted: &[u32]) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(lane < self.mains.len() && self.pending_commit_ids[lane].is_empty(), "commit lane busy or invalid");
        requests.validate_acceptance(batch, accepted)?;
        ensure!(accepted.iter().any(|&n| n > 0), "queued decode commit has no accepted rows");
        let ids = batch.cache()?.request_ids();
        let leases = ids.iter().map(|id| self.requests.get(id)
            .map(|request| request.leases).context("draft commit request not admitted"))
            .collect::<Result<Vec<_>>>()?;
        let stage_leases: [Vec<WindowLease>; 3] = std::array::from_fn(|stage|
            leases.iter().map(|request| request[stage]).collect());
        let taps = pass.taps(batch)?;
        let chunks = crate::families::deepseek_v41::v41_experts::dspark::prepare_commit_rows(taps.rows(), self.windows.each_ref(),
            stage_leases.each_ref().map(|leases| leases.as_slice()), accepted)?;
        let writes = [self.windows[0].prepare_async_write(&chunks[0], taps.rows().len() as u32)?,
            self.windows[1].prepare_async_write(&chunks[1], taps.rows().len() as u32)?,
            self.windows[2].prepare_async_write(&chunks[2], taps.rows().len() as u32)?];
        self.pending_commit_ids[lane] = ids.to_vec();
        unsafe { self.mains[lane].enqueue_commit(taps.values(), &batch.cache()?.positions(),
            self.windows.each_ref(), writes) }
    }
    pub fn poll_queued_commit(&self, lane: usize) -> Result<bool> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        self.mains[lane].poll_commit()
    }
    pub fn finish_queued_commit(&mut self, lane: usize, pass: &mut impl crate::families::deepseek_v41::v41_target_pass::TargetCache<'a>,
        requests: &mut Requests<'a>, batch: &mut RequestBatch, accepted: &[u32]) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        self.mains[lane].publish_commit(self.windows.as_mut_slice())?;
        pass.commit(requests, batch, accepted)?;
        self.pending_commit_ids[lane].clear();
        Ok(())
    }
    /// Drain before revoking any target/cache storage, including partial enqueue.
    pub fn abort_queued_commit(&mut self, lane: usize, requests: &mut Requests<'a>,
        batch: &mut RequestBatch) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        // Target publication may have cancelled the batch before returning an
        // error. Keep cleanup independent of its now-unavailable cache view.
        let ids = if self.pending_commit_ids[lane].is_empty() {
            batch.cache().map(|cache| cache.request_ids().to_vec()).unwrap_or_default()
        } else { self.pending_commit_ids[lane].clone() };
        let mut result = self.mains[lane].abort_commit(self.windows.as_mut_slice());
        self.pending_commit_ids[lane].clear();
        requests.revoke_batch(batch);
        for id in ids { if let Err(error) = self.release(id) { result = Err(error); } }
        result
    }

    pub fn calibrated_confidence(&self, requests: usize, id: u64, extent: usize) -> Result<Option<Vec<f64>>> {
        let Some(policy) = self.policy.as_ref().filter(|policy| !policy.fixed()) else { return Ok(None); };
        let logits = self.confidence_trace.get(&id).context("missing neural draft confidence")?;
        ensure!(extent <= logits.len(), "neural confidence extent mismatch");
        let prior: Vec<_> = logits[..extent].iter().map(|&x| sigmoid(x)).collect();
        Ok(Some(policy.probabilities(requests, cuteafd_core::DraftCandidate { id, confidence: &prior })))
    }

    /// Copied continuation uses the same raw prior and observation as other families.
    pub fn extend_confidence(&mut self, id: u64, neural: usize, extra: usize) -> Result<()> {
        let confidence = self.confidence_trace.get_mut(&id).context("missing neural draft confidence")?;
        ensure!(neural <= confidence.len(), "neural confidence extent mismatch");
        self.neural_lengths.insert(id, neural);
        confidence.truncate(neural);
        confidence.extend((0..extra).map(|i| {
            let p = crate::shared::draft::copy::tail_prior(i);
            (p / (1. - p)).ln() as f32
        }));
        Ok(())
    }

    /// Each lane owns its draft scratch and stream; both can replay concurrently.
    /// Short RefCell borrows never survive a scheduler yield.
    pub fn poll_propose(&mut self, lane: usize,
        inputs: &[(u64, u32, u64, usize)],
    ) -> Result<Option<(Vec<Vec<u32>>, u64)>> {
        ensure!(lane < self.chains.len(), "invalid draft lane");
        let _device = self.chains[lane].execution_device().map(|device| device.enter()).transpose()?;
        if let Some((seeds, started)) = self.pending[lane].take() {
            if seeds != inputs {
                // W64-1: the pending proposal no longer matches this round's
                // inputs. Cancel it and drop the marker so `release()` can free
                // the requests; the caller then verifies anchors only.
                self.chains[lane].cancel_replay();
                if let Some(round) = self.lanes.get_mut(lane) {
                    // Clear the cancelled proposal's ids, not the new round's:
                    // the new round resets its own state when it starts.
                    reset_lane_draft(round, &mut self.confidence_trace, &mut self.neural_lengths, &seeds);
                }
                anyhow::bail!("pending draft request inputs changed");
            }
            let completed = match self.chains[lane].poll_replay() {
                Ok(None) => { self.pending[lane] = Some((seeds, started)); return Ok(None); },
                Ok(Some(value)) => value,
                Err(error) => {
                    self.chains[lane].note_failure(&error);
                    if let Some(round) = self.lanes.get_mut(lane) {
                        reset_lane_draft(round, &mut self.confidence_trace, &mut self.neural_lengths, inputs);
                    }
                    return Err(error);
                },
            };
            let (packed, values) = completed;
            let active: Vec<_> = inputs.iter().enumerate()
                .filter(|(_, (_, _, end, remaining))| *end >= 2 && *remaining > 1).collect();
            let count = active.len();
            let width = self.chains[lane].width();
            ensure!(packed.len() == (width+1)*count && values.len() == width*count, "draft output extent differs");
            let mut outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
            for (row, &(output, &(id, anchor, _, remaining))) in active.iter().enumerate() {
                if self.wants_confidence() {
                    self.confidence_trace.insert(id, (0..width).map(|step| values[step*count+row]).collect());
                }
                let tokens: Vec<_> = (0..=width).map(|step| packed[step*count+row]).collect();
                ensure!(tokens[0] == anchor && tokens.iter().all(|&t| t < 129280), "invalid draft tokens");
                outputs[output] = tokens[..remaining.min(self.draft_limit.min(width)+1)].to_vec();
            }
            return Ok(Some((outputs, started.elapsed().as_micros() as u64)));
        }
        let started = Instant::now();
        ensure!(self.pending[lane].is_none(), "shared draft proposal still pending");
        // W64-1: reset before the guards so a proposal that fails them cannot
        // leave the previous round's width/counts/confidence behind.
        if let Some(round) = self.lanes.get_mut(lane) {
            reset_lane_draft(round, &mut self.confidence_trace, &mut self.neural_lengths, inputs);
        }
        ensure!(!inputs.is_empty() && inputs.len() <= cuteafd_core::V41_MAX_LANE_MEMBERS, "invalid draft batch size");
        let mut seen = std::collections::BTreeSet::new();
        for &(id, anchor, _, remaining) in inputs {
            ensure!(seen.insert(id) && self.requests.contains_key(&id), "invalid draft request identity");
            ensure!(anchor < 129280 && remaining > 0, "invalid draft request input");
        }
        let active: Vec<_> = inputs.iter().enumerate()
            .filter(|(_, (_, _, end, remaining))| *end >= 2 && *remaining > 1).collect();
        let outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
        if active.is_empty() {
            return Ok(Some((outputs, started.elapsed().as_micros() as u64)));
        }
        let widths: Vec<_> = inputs.iter().map(|&(id, _, end, remaining)|
            (id, if end >= 2 { remaining.saturating_sub(1) } else { 0 })).collect();
        self.apply_width(lane, lane, &widths)?;
        let count = active.len();
        let tokens: Vec<_> = active.iter().map(|(_, (_, anchor, _, _))| *anchor as i32).collect();
        let bindings: [Vec<_>; 3] = std::array::from_fn(|stage| active.iter()
            .map(|(_, (id, _, end, _))| (self.requests[id].leases[stage], *end)).collect());
        if let Err(error) = self.chains[lane].stage_tokens(&tokens) {
            self.chains[lane].note_failure(&error);
            return Err(error);
        }
        let mut rngs: Vec<_> = self.requests.iter_mut().filter_map(|(id, request)|
            active.iter().position(|(_, (active_id, _, _, _))| active_id == id)
                .map(|index| (index, &mut request.rng))).collect();
        rngs.sort_by_key(|(index, _)| *index);
        if let Err(error) = self.chains[lane].stage_sampling(&mut rngs.into_iter().map(|(_, rng)| rng).collect::<Vec<_>>(),
            &vec![0.0; count]) {
            self.chains[lane].note_failure(&error);
            return Err(error);
        }
        let windows = self.windows.each_ref();
        let bindings = bindings.each_ref().map(|rows| rows.as_slice());
        // SAFETY: admitted leases and windows stay owned by the runtime through completion/cancellation.
        if let Err(error) = unsafe { self.chains[lane].begin_replay(windows, bindings) } {
            self.chains[lane].note_failure(&error);
            return Err(error);
        }
        self.pending[lane] = Some((inputs.to_vec(), started));
        Ok(None)
    }

    pub fn confidence_trace(&self, id: u64) -> Option<&[f32]> {
        self.confidence_trace.get(&id).map(Vec::as_slice)
    }
}

impl<'w, 'a, C: DraftChain<'a>> DraftRuntime<'w, 'a, C> {
    pub fn windows(&self) -> &[DsparkWindow<'a>] {
        &self.windows
    }
    /// Liveness gate for the anchor-only fallback: a terminal draft propagates
    /// its error instead of silently degrading every round to anchors.
    pub fn require_live(&self, lane: usize) -> Result<()> {
        self.chains.get(lane).context("invalid draft lane")?.require_live()
    }
    /// A failed draft proposal: verify each member's anchor alone, but only
    /// while the draft is still live. A terminal draft propagates its error so
    /// the round fails instead of degrading to anchors forever. The lane's
    /// draft state is reset so the anchor-only round is observed as undrafted.
    pub fn anchor_fallback(&mut self, lane: usize, seeds: &[DraftSeed], error: &anyhow::Error)
        -> Result<Vec<Vec<u32>>> {
        self.require_live(lane)?;
        if let Some(round) = self.lanes.get_mut(lane) {
            reset_lane_draft(round, &mut self.confidence_trace, &mut self.neural_lengths, seeds);
        }
        let total = DRAFT_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        tracing::warn!(lane, requests = seeds.len(), total, error = %format!("{error:#}"),
            "dSpark draft proposal failed; this lane round verifies anchors only");
        Ok(anchor_inputs(seeds))
    }

}
impl<'w, 'a, C: ProposeChain<'a>> DraftRuntime<'w, 'a, C> {
    /// Inputs are (request identity, anchor, committed end, remaining output budget).
    /// Returned rows retain input order; short histories/budgets use the anchor only.
    ///
    /// The lane's own chain drafts and latches: `propose` and `require_live` must
    /// agree on which chain a lane owns, or a terminal failure on one chain is
    /// never seen by the lane that drafts on it.
    pub fn propose(&mut self, lane: usize,
        inputs: &[(u64, u32, u64, usize)],
    ) -> Result<Vec<Vec<u32>>> {
        ensure!(lane < self.lanes.len() && lane < self.chains.len(), "invalid draft lane");
        ensure!(self.pending.iter().all(Option::is_none), "shared draft proposal still pending");
        // W64-1: reset before the guards so a proposal that fails them cannot
        // leave the previous round's width/counts/confidence behind.
        reset_lane_draft(&mut self.lanes[lane], &mut self.confidence_trace, &mut self.neural_lengths, inputs);
        ensure!(!inputs.is_empty() && inputs.len() <= cuteafd_core::V41_MAX_LANE_MEMBERS, "invalid draft batch size");
        let mut seen = std::collections::BTreeSet::new();
        for &(id, anchor, _, remaining) in inputs {
            ensure!(seen.insert(id) && self.requests.contains_key(&id), "invalid draft request identity");
            ensure!(anchor < 129280 && remaining > 0, "invalid draft request input");
        }
        let active: Vec<_> = inputs.iter().enumerate()
            .filter(|(_, (_, _, end, remaining))| *end >= 2 && *remaining > 1).collect();
        let mut outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
        if active.is_empty() { return Ok(outputs); }
        let widths: Vec<_> = inputs.iter().map(|&(id, _, end, remaining)|
            (id, if end >= 2 { remaining.saturating_sub(1) } else { 0 })).collect();
        let width = self.apply_width(lane, lane, &widths)?;
        let count = active.len();
        let tokens: Vec<_> = active.iter().map(|(_, (_, anchor, _, _))| *anchor as i32).collect();
        let bindings: [Vec<_>; 3] = std::array::from_fn(|stage| active.iter()
            .map(|(_, (id, _, end, _))| (self.requests[id].leases[stage], *end)).collect());
        if let Err(error) = self.chains[lane].set_tokens(&tokens) {
            self.chains[lane].note_failure(&error);
            return Err(error);
        }
        let mut rngs: Vec<_> = self.requests.iter_mut().filter_map(|(id, request)|
            active.iter().position(|(_, (active_id, _, _, _))| active_id == id)
                .map(|index| (index, &mut request.rng))).collect();
        rngs.sort_by_key(|(index, _)| *index);
        if let Err(error) = self.chains[lane].prepare_sampling(
            &mut rngs.into_iter().map(|(_, rng)| rng).collect::<Vec<_>>(), &vec![0.0; count]) {
            self.chains[lane].note_failure(&error);
            return Err(error);
        }
        let windows = self.windows.each_ref();
        let bindings = bindings.each_ref().map(|rows| rows.as_slice());
        let launched = {
            let chain = &mut self.chains[lane];
            if chain.has_graph(count) {
                // SAFETY: same stable window/binding lifetime as capture below.
                unsafe { chain.replay(windows, bindings) }
            } else if chain.capture_allowed(count) {
                // SAFETY: admitted leases and windows remain owned until the synchronous operation drains.
                let captured = unsafe { chain.capture(windows, bindings) };
                // SAFETY: same stable window/binding lifetime as capture above.
                captured.and_then(|()| unsafe { chain.replay(windows, bindings) })
            } else {
                // A shape outside the sealed startup inventory runs eagerly and
                // allocates no executable.
                // SAFETY: same stable window/binding lifetime as capture above.
                unsafe { chain.run(windows, bindings) }
            }
        };
        if let Err(error) = launched {
            self.chains[lane].note_failure(&error);
            return Err(error);
        }
        let buffer = match self.chains[lane].draft_output() {
            Ok(output) => output[0],
            Err(error) => { self.chains[lane].note_failure(&error); return Err(error); }
        };
        let mut bytes = vec![0; buffer.bytes];
        if let Err(error) = self.chains[lane].copy_d2h(&mut bytes, buffer) {
            self.chains[lane].note_failure(&error);
            return Err(error);
        }
        ensure!(bytes.len() == (width + 1) * count * 4, "draft token extent differs");
        let packed: Vec<_> = bytes.chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap())).collect();
        if self.wants_confidence() {
            let confidence = match self.chains[lane].draft_output() {
                Ok(output) => output[2],
                Err(error) => { self.chains[lane].note_failure(&error); return Err(error); }
            };
            ensure!(confidence.bytes == width * count * 4, "draft confidence extent differs");
            let mut bytes = vec![0; confidence.bytes];
            if let Err(error) = self.chains[lane].copy_d2h(&mut bytes, confidence) {
                self.chains[lane].note_failure(&error);
                return Err(error);
            }
            let values: Vec<_> = bytes.chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();
            for (row, &(_, &(id, _, _, _))) in active.iter().enumerate() {
                self.confidence_trace.insert(id,
                    (0..width).map(|step| values[step * count + row]).collect());
            }
        }
        for (row, &(output, &(_, anchor, _, remaining))) in active.iter().enumerate() {
            let tokens: Vec<_> = (0..=width).map(|step| packed[step * count + row]).collect();
            ensure!(tokens[0] == anchor && tokens.iter().all(|&token| token < 129280), "invalid draft tokens");
            outputs[output] = tokens[..remaining.min(self.draft_limit.min(width) + 1)].to_vec();
        }
        Ok(outputs)
    }

}

/// Observers use host round timestamps. Replay routes are read after the pass;
/// only resource pricing or an explicit timing profile requests layer events.
pub(crate) fn round_capture(policy: Option<&cuteafd_core::DraftPolicy>, _console_live: bool,
    replay: bool, profiling: bool) -> Capture {
    let Some(policy) = policy else { return Capture { routes: false, timing: profiling } };
    Capture { routes: policy.needs_routes() || replay, timing: policy.needs_routes() || profiling }
}

#[cfg(test)]
mod restored_prefix_tests {
    use super::*;

    #[test]
    fn observation_and_replay_do_not_enable_layer_timing() {
        use cuteafd_core::{DraftMode, DraftPolicy, PolicySeed};
        let geometry = super::policy::remote_geometry(7);
        for (mode, fixed, priced) in [(DraftMode::Simple, false, false), (DraftMode::SimpleRc, false, false),
            (DraftMode::SimpleDerived, false, false), (DraftMode::Shared, true, false), (DraftMode::Shared, false, true)] {
            let mut policy = DraftPolicy::new(geometry.clone(), PolicySeed::weak(7), fixed).unwrap();
            policy.set_draft_mode(mode);
            for watched in [false, true] {
                for replay in [false, true] {
                    for profiling in [false, true] {
                        assert_eq!(round_capture(Some(&policy), watched, replay, profiling),
                            Capture { routes: priced || replay, timing: priced || profiling },
                            "{mode:?} fixed={fixed} watched={watched} replay={replay} profiling={profiling}");
                    }
                }
            }
        }
        assert_eq!(round_capture(None, true, false, false), Capture::default());
        assert_eq!(round_capture(None, true, false, true), Capture { routes: false, timing: true });
    }
    #[tokio::test]
    async fn console_subscriber_keeps_capture_timing_off_unless_profiled() {
        use tower::ServiceExt;
        let hub = cuteafd_api::openai::ConsoleHub::new(false);
        let (queue, _receive) = tokio::sync::mpsc::channel(1);
        let router = cuteafd_api::openai::router_with_console(queue,
            cuteafd_api::openai::NativeLimits::default(),
            std::sync::Arc::new(std::sync::Mutex::new(serde_json::Value::Null)),
            std::time::Duration::from_secs(1), hub.clone());
        let response = router.oneshot(axum::http::Request::builder()
            .uri("/v1/console/events").body(axum::body::Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        assert_eq!(hub.viewers(), 1, "a real console subscriber must be counted");
        assert!(!round_capture(None, hub.viewers() > 0, false, false).timing);
        assert!(round_capture(None, hub.viewers() > 0, false, true).timing);
        drop(response);
        assert_eq!(hub.viewers(), 0);
    }

    #[test]
    fn long_tail_stats_count_only_judged_positions() {
        assert_eq!(tail_outcome(38, 1, 5), (32, 0, 0));
        assert_eq!(tail_outcome(38, 5, 5), (32, 0, 0));
        assert_eq!(tail_outcome(38, 6, 5), (32, 1, 0));
        assert_eq!(tail_outcome(38, 20, 5), (32, 15, 14));
        assert_eq!(tail_outcome(38, 38, 5), (32, 32, 32));
        assert_eq!(tail_outcome(4, 4, 5), (0, 0, 0));
    }
}

#[cfg(test)]
mod anchor_fallback_tests {
    use super::*;
    use crate::families::deepseek_v41::v41_experts::dspark::DraftFailure;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use std::cell::Cell;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum At { Width, Tokens, Sampling, Capture, Replay, Output, TokensCopy, ConfidenceCopy, Begin, Poll, Never }
    struct FailingChain {
        failure: DraftFailure,
        at: At,
        device_error: bool,
        tokens: Vec<u32>,
        polls: usize,
        cancelled: bool,
        downloads: Cell<usize>,
    }
    impl FailingChain {
        fn new(at: At, device_error: bool) -> Self {
            Self { failure: Default::default(), at, device_error, tokens: Vec::new(), polls: 0,
                cancelled: false, downloads: Cell::new(0) }
        }
        // Deliberately does not record: only the real runtime can latch this mock.
        fn step(&self, at: At) -> Result<()> {
            if self.at != at { return Ok(()); }
            if self.device_error {
                Err(anyhow::Error::new(cuteafd_ffi::NativeStatusError {
                    context: format!("injected {at:?}"), status: 98, detail: "invalid device function".into(),
                }).context("draft operation"))
            } else { anyhow::bail!("injected host guard at {at:?}") }
        }
        fn packed(&self) -> Vec<u32> {
            (0..=5).flat_map(|step| self.tokens.iter().map(move |&token| token + step)).collect()
        }
    }
    impl<'a> DraftChain<'a> for FailingChain {
        fn execution_device(&self) -> Option<crate::shared::memory::device::Device<'a>> { None }
        fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> {
            self.tokens = tokens.iter().map(|&token| token as u32).collect();
            self.step(At::Tokens)
        }
        fn stage_sampling(&mut self, _: &mut [&mut cuteafd_core::DsparkRng], _: &[f32]) -> Result<()> { self.step(At::Sampling) }
        unsafe fn begin_replay(&mut self, _: [&DsparkWindow<'_>; 3], _: [&[(WindowLease, u64)]; 3]) -> Result<()> { self.step(At::Begin) }
        fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
            self.step(At::Poll)?;
            self.polls += 1;
            Ok((self.polls > 1).then(|| (self.packed(), vec![0.5; self.tokens.len() * 5])))
        }
        fn cancel_replay(&mut self) { self.cancelled = true; }
        fn note_failure(&mut self, error: &anyhow::Error) { self.failure.record(error); }
        fn width(&self) -> usize { 5 }
        fn set_width(&mut self, _: usize) -> Result<()> { self.step(At::Width) }
        fn require_live(&self) -> Result<()> { self.failure.check("test draft chain") }
    }
    impl<'a> ProposeChain<'a> for FailingChain {
        fn has_graph(&self, _: usize) -> bool { self.at != At::Capture }
        fn capture_allowed(&self, _: usize) -> bool { true }
        unsafe fn run(&mut self, _: [&DsparkWindow<'_>; 3], _: [&[(WindowLease, u64)]; 3]) -> Result<()> { self.step(At::Replay) }
        fn set_tokens(&mut self, tokens: &[i32]) -> Result<()> { self.stage_tokens(tokens) }
        fn prepare_sampling(&mut self, _: &mut [&mut cuteafd_core::DsparkRng], _: &[f32]) -> Result<()> { self.step(At::Sampling) }
        unsafe fn capture(&mut self, _: [&DsparkWindow<'_>; 3], _: [&[(WindowLease, u64)]; 3]) -> Result<()> { self.step(At::Capture) }
        unsafe fn replay(&mut self, _: [&DsparkWindow<'_>; 3], _: [&[(WindowLease, u64)]; 3]) -> Result<()> { self.step(At::Replay) }
        fn draft_output(&self) -> Result<[CuteafdDeviceBuffer; 3]> {
            self.step(At::Output)?;
            Ok([CuteafdDeviceBuffer { ptr: 1usize as *mut _, bytes: self.tokens.len() * 6 * 4, device_id: 0, flags: 0 },
                CuteafdDeviceBuffer::default(),
                CuteafdDeviceBuffer { ptr: 2usize as *mut _, bytes: self.tokens.len() * 5 * 4, device_id: 0, flags: 0 }])
        }
        fn copy_d2h(&self, dst: &mut [u8], buffer: CuteafdDeviceBuffer) -> Result<()> {
            self.downloads.set(self.downloads.get() + 1);
            let confidence = buffer.ptr as usize == 2;
            self.step(if confidence { At::ConfidenceCopy } else { At::TokensCopy })?;
            let bytes: Vec<_> = if confidence { vec![0.5f32; self.tokens.len() * 5].iter().flat_map(|v| v.to_ne_bytes()).collect() }
                else { self.packed().iter().flat_map(|v| v.to_ne_bytes()).collect() };
            dst.copy_from_slice(&bytes);
            Ok(())
        }
    }
    fn fixture() -> Result<Fixture> {
        Fixture::build_for_tests()
    }
    fn runtime<'a>(lib: &'a NativeLibrary, at: At, device_error: bool) -> Result<DraftRuntime<'a, 'a, FailingChain>> {
        let window = || DsparkWindow::new(lib, 4, 8, DsparkWindow::device_bytes(4, 8)?);
        let mut runtime = DraftRuntime {
            mains: Vec::new(), chains: (0..2).map(|_| FailingChain::new(at, device_error)).collect(),
            windows: [window()?, window()?, window()?], pending_commit_ids: vec![Vec::new(); 2],
            pending_prefix_ids: [None, None], requests: Default::default(), pending: vec![None; 2],
            request_limit: 4, draft_limit: 5, draft_width: 5, confidence_trace: Default::default(),
            neural_lengths: Default::default(), length: Default::default(), policy: None, published: None,
            lanes: (0..2).map(|_| LaneRound::new(policy::LAYERS, policy::TOPK)).collect(),
        };
        for id in 1..=4 { runtime.admit(id)?; }
        runtime.bind_policy(policy::remote_geometry(5))?;
        Ok(runtime)
    }
    fn seeds(ids: &[u64]) -> Vec<DraftSeed> {
        ids.iter().enumerate().map(|(i, &id)| (id, 11 + i as u32, 5, 9)).collect()
    }
    #[test]
    fn synchronous_proposal_records_each_failure_on_its_own_lane() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        for lane in 0..2 {
            for at in [At::Width, At::Tokens, At::Sampling, At::Capture, At::Replay, At::Output, At::TokensCopy, At::ConfidenceCopy] {
                for device in [false, true] {
                    let mut runtime = runtime(&lib, at, device)?;
                    let round = seeds(&[1, 2]);
                    let error = runtime.propose(lane, &round).unwrap_err();
                    let fallback = runtime.anchor_fallback(lane, &round, &error);
                    assert_eq!(fallback.is_err(), device, "lane {lane}, {at:?}");
                    assert!(runtime.require_live(1 - lane).is_ok(), "other lane must stay live");
                    if !device { assert_eq!(fallback?, vec![vec![11], vec![12]]); }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn independent_proposal_records_submission_and_poll_failures() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        for lane in 0..2 {
            for at in [At::Width, At::Tokens, At::Sampling, At::Begin, At::Poll] {
                for device in [false, true] {
                    let mut runtime = runtime(&lib, at, device)?;
                    let round = seeds(&[1, 2]);
                    let result = match runtime.poll_propose(lane, &round) {
                        Ok(None) => runtime.poll_propose(lane, &round),
                        result => result,
                    };
                    let error = result.unwrap_err();
                    assert!(runtime.pending[lane].is_none());
                    assert_eq!(runtime.anchor_fallback(lane, &round, &error).is_err(), device, "lane {lane}, {at:?}");
                    assert!(runtime.require_live(1 - lane).is_ok());
                }
            }
        }
        Ok(())
    }
    #[test]
    fn independent_pending_blocks_release_and_cancellation_resets_old_ids() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        let mut runtime = runtime(&lib, At::Never, false)?;
        let round = seeds(&[1, 2]);
        assert!(runtime.poll_propose(1, &round)?.is_none());
        assert!(runtime.release(1).is_err());
        assert!(runtime.poll_propose(1, &round)?.is_none());
        assert!(runtime.release(2).is_err());
        runtime.confidence_trace.insert(1, vec![0.5; 5]);
        runtime.neural_lengths.insert(1, 3);
        assert!(runtime.poll_propose(1, &seeds(&[3, 4])).is_err());
        assert!(runtime.chains[1].cancelled);
        assert_eq!((runtime.lanes[1].width, runtime.lanes[1].drafted_requests), (0, 0));
        assert!(runtime.confidence_trace(1).is_none() && !runtime.neural_lengths.contains_key(&1));
        runtime.release(1)?;
        runtime.release(2)?;
        Ok(())
    }
    #[test]
    fn successful_poll_preserves_order_and_invalid_lane_is_an_error() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        let mut runtime = runtime(&lib, At::Never, false)?;
        assert!(runtime.poll_propose(2, &seeds(&[1])).is_err());
        let round = seeds(&[2, 1]);
        assert!(runtime.poll_propose(1, &round)?.is_none());
        assert!(runtime.poll_propose(1, &round)?.is_none());
        let (tokens, _) = runtime.poll_propose(1, &round)?.context("proposal should finish")?;
        assert_eq!(tokens, vec![vec![11, 12, 13, 14, 15, 16], vec![12, 13, 14, 15, 16, 17]]);
        runtime.release(1)?;
        runtime.release(2)?;
        Ok(())
    }
    #[test]
    fn four_then_two_failed_proposal_is_an_undrafted_policy_round() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        for independent in [false, true] {
            let mut runtime = runtime(&lib, At::Never, false)?;
            let previous = seeds(&[1, 2, 3, 4]);
            runtime.propose(0, &previous)?;
            assert_eq!(runtime.lanes[0].drafted_requests, 4);
            runtime.neural_lengths.insert(1, 3);
            let mut round = seeds(&[1, 2]);
            round[0].1 = 129280; // Host validation fails before drafting starts.
            let error = if independent { runtime.poll_propose(0, &round).unwrap_err() }
                else { runtime.propose(0, &round).unwrap_err() };
            assert_eq!((runtime.lanes[0].width, runtime.lanes[0].drafted_requests), (0, 0));
            assert!(runtime.confidence_trace(1).is_none() && !runtime.neural_lengths.contains_key(&1));
            round[0].1 = 11;
            assert_eq!(runtime.anchor_fallback(0, &round, &error)?, vec![vec![11], vec![12]]);
            runtime.select_lengths(0, &[(1, 0), (2, 0)], false)?;
        }
        Ok(())
    }
}
