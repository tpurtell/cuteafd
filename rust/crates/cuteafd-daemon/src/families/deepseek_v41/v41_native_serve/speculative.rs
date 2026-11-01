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

pub(crate) fn policy_snapshot() -> serde_json::Value {
    use std::sync::atomic::Ordering::Relaxed;
    let mut snapshot = policy::snapshot();
    snapshot["copy_tail"] = serde_json::json!({
        "requests": TAIL_REQUESTS.load(Relaxed), "proposed": TAIL_PROPOSED.load(Relaxed),
        "reached": TAIL_REACHED.load(Relaxed), "accepted": TAIL_ACCEPTED.load(Relaxed),
    });
    snapshot
}

fn tail_outcome(rows: usize, accepted: usize, neural: usize) -> (usize, usize, usize) {
    let proposed = rows.saturating_sub(neural + 1);
    let reached = if accepted <= neural { 0 } else { proposed.min(accepted - neural) };
    let accepted = accepted.saturating_sub(neural + 1).min(proposed);
    (proposed, reached, accepted)
}
pub(crate) use chain::DraftChain;
mod distributed;

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
    /// Verify every available draft instead of selecting lengths.
    fixed: bool,
    /// Online bandwidth-balance length policy, bound to the installed placement.
    policy: Option<cuteafd_core::DraftPolicy>,
    published: Option<Instant>,
    /// Per lane: the round's draft width, the last selection's regime and
    /// prediction, and the route storage its observation reuses.
    lanes: [LaneRound; 2],
}
struct DraftRequest {
    leases: [WindowLease; 3],
    rng: cuteafd_core::DsparkRng,
    slot: usize,
}
impl<'w, 'a> DraftRuntime<'w, 'a> {
    pub fn new(
        lib: &'a NativeLibrary,
        weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>,
        head: &'w VocabularyHead<'a>,
        capacity: u32,
    ) -> Result<Self> {
        Self::with_requests(lib, weights, table, head, capacity, 16)
    }
    pub fn with_requests(
        lib: &'a NativeLibrary, weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>, head: &'w VocabularyHead<'a>,
        capacity: u32, requests: u32,
    ) -> Result<Self> {
        ensure!((1..=16).contains(&requests), "invalid draft request limit");
        let window = || DsparkWindow::new(lib, requests as usize, capacity,
            DsparkWindow::device_bytes(requests as usize, capacity)?);
        let lane_count = if requests == 1 { 1 } else { 2 };
        let lane_requests = requests.div_ceil(lane_count as u32);
        let shared_bytes = weights.draft_bytes(requests)?;
        let lane_bytes = weights.draft_bytes(lane_requests)?;
        let mut mains = vec![weights.main_context(capacity, DsparkMainContext::device_bytes(lib, capacity)?)?];
        if lane_count > 1 { mains.push(weights.main_context(80, DsparkMainContext::device_bytes(lib, 80)?)?); }
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
            fixed: false,
            policy: None,
            published: None,
            lanes: std::array::from_fn(|_| LaneRound::new(policy::LAYERS, policy::TOPK)),
        })
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
    pub fn set_draft_limit(&mut self, limit: u8) -> Result<()> {
        ensure!((1..=self.draft_width as u8).contains(&limit), "draft limit exceeds generated width");
        self.draft_limit = limit as usize;
        Ok(())
    }
    /// Verify every available draft; the policy still records statistics.
    pub fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed;
        if let Some(policy) = &mut self.policy {
            *policy = policy.restarted(fixed);
        }
    }
    /// Bind the length policy to the installed expert placement before serving.
    pub fn configure_policy(&mut self, transport: &NativeTp4Wave<'_>, nvfp4: bool) -> Result<()> {
        let geometry = policy::placement(transport, nvfp4, self.draft_width)?;
        tracing::info!(fixed=self.fixed, draft_limit=self.draft_limit, local_layers=policy::local_layers(&geometry),
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
        let mut policy = cuteafd_core::DraftPolicy::new(geometry, seed, self.fixed)?;
        crate::shared::draft::apply_length_rule(&mut policy);
        policy::publish(&policy, self.draft_limit);
        self.policy = Some(policy);
        Ok(())
    }
    /// A length policy is bound (it observes every verification round).
    pub fn has_policy(&self) -> bool { self.policy.is_some() }
    /// What a verification round records: see [`round_capture`].
    pub fn capture(&self) -> Capture {
        round_capture(self.policy.as_ref(), super::console::live().is_some(),
            crate::shared::expert_usage::enabled())
    }
    fn wants_confidence(&self) -> bool {
        self.policy.is_some() || tracing::enabled!(target: "cuteafd::draft_policy", tracing::Level::DEBUG)
    }
    /// Choose each request's draft length for one lane round. `requests` are
    /// (identity, available drafts) in lane order; `shared` means the other lane
    /// has active requests. `None` verifies every available draft (fixed mode
    /// or a fit still warming up).
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
        self.chains[chain].set_width(width)?;
        self.lanes[lane].width = width;
        self.lanes[lane].drafted_requests = requests.iter().filter(|r| r.1 > 0).count();
        Ok(width)
    }
    pub fn release(&mut self, id: u64) -> Result<()> {
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(!self.pending_prefix_ids.contains(&Some(id)), "cannot release a request with pending snapshot copies");
        ensure!(!self.pending.iter().flatten().any(|(seeds, _)| seeds.iter().any(|seed| seed.0 == id)),
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
        let [a, b, c] = &mut self.windows;
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
        self.mains[lane].publish_commit(&mut self.windows)?;
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
        let mut result = self.mains[lane].abort_commit(&mut self.windows);
        self.pending_commit_ids[lane].clear();
        requests.revoke_batch(batch);
        for id in ids { if let Err(error) = self.release(id) { result = Err(error); } }
        result
    }

    pub fn calibrated_confidence(&self, requests: usize, id: u64, extent: usize) -> Result<Option<Vec<f64>>> {
        let Some(policy) = self.policy.as_ref().filter(|_| !self.fixed) else { return Ok(None); };
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
        let _device = self.chains[0].execution_device().map(|device| device.enter()).transpose()?;
        ensure!(lane < self.chains.len(), "invalid draft lane");
        if let Some((seeds, started)) = &self.pending[lane] {
            let started = *started;
            ensure!(seeds == inputs, "pending draft request inputs changed");
            let completed = match self.chains[lane].poll_replay() {
                Ok(None) => return Ok(None),
                Ok(Some(value)) => value,
                Err(error) => { self.pending[lane] = None; return Err(error); },
            };
            self.pending[lane] = None;
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
        ensure!(!inputs.is_empty() && inputs.len() <= 16, "invalid draft batch size");
        let mut seen = std::collections::BTreeSet::new();
        for &(id, anchor, _, remaining) in inputs {
            ensure!(seen.insert(id) && self.requests.contains_key(&id), "invalid draft request identity");
            ensure!(anchor < 129280 && remaining > 0, "invalid draft request input");
        }
        for &(id, _, _, _) in inputs {
            self.confidence_trace.remove(&id);
        self.neural_lengths.remove(&id);
        }
        let active: Vec<_> = inputs.iter().enumerate()
            .filter(|(_, (_, _, end, remaining))| *end >= 2 && *remaining > 1).collect();
        let outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
        if active.is_empty() {
            self.lanes[lane].width = 0; self.lanes[lane].drafted_requests = 0;
            return Ok(Some((outputs, started.elapsed().as_micros() as u64)));
        }
        let widths: Vec<_> = inputs.iter().map(|&(id, _, end, remaining)|
            (id, if end >= 2 { remaining.saturating_sub(1) } else { 0 })).collect();
        self.apply_width(lane, lane, &widths)?;
        let count = active.len();
        let tokens: Vec<_> = active.iter().map(|(_, (_, anchor, _, _))| *anchor as i32).collect();
        let bindings: [Vec<_>; 3] = std::array::from_fn(|stage| active.iter()
            .map(|(_, (id, _, end, _))| (self.requests[id].leases[stage], *end)).collect());
        self.chains[lane].stage_tokens(&tokens)?;
        let mut rngs: Vec<_> = self.requests.iter_mut().filter_map(|(id, request)|
            active.iter().position(|(_, (active_id, _, _, _))| active_id == id)
                .map(|index| (index, &mut request.rng))).collect();
        rngs.sort_by_key(|(index, _)| *index);
        self.chains[lane].stage_sampling(&mut rngs.into_iter().map(|(_, rng)| rng).collect::<Vec<_>>(),
            &vec![0.0; count])?;
        let windows = self.windows.each_ref();
        let bindings = bindings.each_ref().map(|rows| rows.as_slice());
        unsafe { self.chains[lane].begin_replay(windows, bindings)?; }
        self.pending[lane] = Some((inputs.to_vec(), started));
        Ok(None)
    }

    pub fn confidence_trace(&self, id: u64) -> Option<&[f32]> {
        self.confidence_trace.get(&id).map(Vec::as_slice)
    }
}

impl<'w, 'a, C: DraftChain<'a>> DraftRuntime<'w, 'a, C> {
    pub fn windows(&self) -> &[DsparkWindow<'a>; 3] {
        &self.windows
    }
}
impl<'w, 'a> DraftRuntime<'w, 'a> {
    /// Inputs are (request identity, anchor, committed end, remaining output budget).
    /// Returned rows retain input order; short histories/budgets use the anchor only.
    pub fn propose(&mut self, lib: &'a NativeLibrary, lane: usize,
        inputs: &[(u64, u32, u64, usize)],
    ) -> Result<Vec<Vec<u32>>> {
        ensure!(lane < self.lanes.len(), "invalid draft lane");
        ensure!(self.pending.iter().all(Option::is_none), "shared draft proposal still pending");
        ensure!(!inputs.is_empty() && inputs.len() <= 16, "invalid draft batch size");
        let mut seen = std::collections::BTreeSet::new();
        for &(id, anchor, _, remaining) in inputs {
            ensure!(seen.insert(id) && self.requests.contains_key(&id), "invalid draft request identity");
            ensure!(anchor < 129280 && remaining > 0, "invalid draft request input");
        }
        for &(id, _, _, _) in inputs {
            self.confidence_trace.remove(&id);
        self.neural_lengths.remove(&id);
        }
        let active: Vec<_> = inputs.iter().enumerate()
            .filter(|(_, (_, _, end, remaining))| *end >= 2 && *remaining > 1).collect();
        let mut outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
        if active.is_empty() { self.lanes[lane].width = 0; self.lanes[lane].drafted_requests = 0; return Ok(outputs); }
        let widths: Vec<_> = inputs.iter().map(|&(id, _, end, remaining)|
            (id, if end >= 2 { remaining.saturating_sub(1) } else { 0 })).collect();
        let width = self.apply_width(lane, 0, &widths)?;
        let count = active.len();
        let tokens: Vec<_> = active.iter().map(|(_, (_, anchor, _, _))| *anchor as i32).collect();
        let bindings: [Vec<_>; 3] = std::array::from_fn(|stage| active.iter()
            .map(|(_, (id, _, end, _))| (self.requests[id].leases[stage], *end)).collect());
        self.chains[0].set_tokens(&tokens)?;
        let mut rngs: Vec<_> = self.requests.iter_mut().filter_map(|(id, request)|
            active.iter().position(|(_, (active_id, _, _, _))| active_id == id)
                .map(|index| (index, &mut request.rng))).collect();
        rngs.sort_by_key(|(index, _)| *index);
        self.chains[0].prepare_sampling(&mut rngs.into_iter().map(|(_, rng)| rng).collect::<Vec<_>>(),
            &vec![0.0; count])?;
        let windows = self.windows.each_ref();
        let bindings = bindings.each_ref().map(|rows| rows.as_slice());
        if !self.chains[0].has_graph(count) {
            unsafe { self.chains[0].capture(windows, bindings)?; }
        }
        unsafe { self.chains[0].replay(windows, bindings)?; }
        let buffer = self.chains[0].draft_output()?[0];
        let mut bytes = vec![0; buffer.bytes];
        lib.copy_d2h(&mut bytes, buffer)?;
        ensure!(bytes.len() == (width + 1) * count * 4, "draft token extent differs");
        let packed: Vec<_> = bytes.chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap())).collect();
        if self.wants_confidence() {
            let confidence = self.chains[0].draft_output()?[2];
            ensure!(confidence.bytes == width * count * 4, "draft confidence extent differs");
            let mut bytes = vec![0; confidence.bytes];
            lib.copy_d2h(&mut bytes, confidence)?;
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

/// A policy-bound verification round captures route ids only for resource
/// pricing or the replay trace, whatever the draft mode; simple and fixed modes
/// skip them. Per-layer CUDA-event times and the FFN stage split come with the
/// routes and also whenever a console viewer is connected (its layer profile
/// and expert-stage panels), so a default launch keeps them while watched and
/// skips them while nobody is.
pub(crate) fn round_capture(policy: Option<&cuteafd_core::DraftPolicy>, console_live: bool,
    usage: bool) -> Capture {
    let Some(policy) = policy else { return Capture { routes: usage, timing: false } };
    let routes = crate::shared::draft::binding::needs_routes(policy) || usage;
    Capture { routes, timing: routes || console_live }
}

#[cfg(test)]
mod restored_prefix_tests {
    use super::*;

    /// Simple and fixed rounds never capture route ids; a connected console
    /// still gets per-layer times and the FFN stage split in every mode, and an
    /// unwatched simple or fixed round records neither. The priced policy
    /// captures both, watched or not.
    #[test]
    fn console_keeps_layer_times_in_every_draft_mode() {
        use cuteafd_core::{DraftMode, DraftPolicy, PolicySeed};
        assert_eq!(round_capture(None, true, false), Capture::default());
        // usage capture records routes without layer timing.
        assert_eq!(round_capture(None, false, true), Capture { routes: true, timing: false });
        let geometry = super::policy::remote_geometry(7);
        for (mode, fixed, priced) in [(DraftMode::Simple, false, false), (DraftMode::SimpleRc, false, false),
            (DraftMode::SimpleDerived, false, false), (DraftMode::Shared, true, false), (DraftMode::Shared, false, true)] {
            let mut policy = DraftPolicy::new(geometry.clone(), PolicySeed::weak(7), fixed).unwrap();
            policy.set_draft_mode(mode);
            // The replay trace (SPECULATION_TRACE) forces routes; this test runs without it.
            let traced = crate::shared::draft::binding::needs_routes(&policy) && !priced;
            assert!(!traced, "{mode:?} fixed={fixed}: unexpected route need");
            let watched = round_capture(Some(&policy), true, false);
            let unwatched = round_capture(Some(&policy), false, false);
            assert_eq!(watched, Capture { routes: priced, timing: true }, "{mode:?} fixed={fixed} watched");
            assert_eq!(unwatched, Capture { routes: priced, timing: priced }, "{mode:?} fixed={fixed} unwatched");
            // usage forces routes (and therefore timing) on every mode.
            assert_eq!(round_capture(Some(&policy), false, true),
                Capture { routes: true, timing: true }, "{mode:?} fixed={fixed} usage");
        }
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
