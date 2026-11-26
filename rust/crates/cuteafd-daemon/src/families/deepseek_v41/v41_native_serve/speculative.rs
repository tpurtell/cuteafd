use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::BackboneCache;
use crate::families::deepseek_v41::v41_dspark_cache::{DsparkWindow, WindowLease};
use crate::families::deepseek_v41::v41_experts::dfly2::{Dfly2Chain, Dfly2TapStore, Dfly2Vocab, Dfly2Weights};
use crate::families::deepseek_v41::v41_experts::dspark::{DraftFailure, DsparkChain, DsparkMainContext, DsparkWeights};
use crate::families::deepseek_v41::v41_target_pass::Dfly2CommitSources;
use crate::shared::memory::device::{Device, Event, Stream};
use cuteafd_ffi::CuteafdDeviceBuffer;
/// Bytes of one request's retained rings in a draft mark: three windows' full slots.
pub(crate) const DRAFT_MARK_BYTES: usize = 3 * cuteafd_ffi::V41DsparkCache::SLOT_BYTES;
use crate::families::deepseek_v41::v41_requests::RequestBatch;
use crate::families::deepseek_v41::v41_backbone_lane::Capture;
use crate::shared::draft::history::{self, Policy, Proposal};
mod chain;

/// One verified request of a lane round, in lane order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LaneRequest {
    pub id: u64,
    /// Verifier rows executed, anchor included.
    pub rows: usize,
    /// Accepted inputs, anchor included.
    pub accepted: u32,
    /// Verification stopped (stop token, output limit) without judging the next draft.
    pub censored: bool,
}

/// What a lane carries from its plan to its proposal.
#[derive(Clone, Debug, Default)]
pub(crate) struct LaneDraft {
    /// Drafts planned per lane member, in lane order (empty: no draft pass).
    depths: Vec<usize>,
    /// Indices actually submitted to the native chain, in lane order.
    selected: Vec<usize>,
}

/// Layers whose experts run on the coordinator, for the console's layer classes.
static LOCAL_LAYERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

static TAIL_REQUESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TAIL_PROPOSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TAIL_REACHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TAIL_ACCEPTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DRAFT_FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One lane round's draft seeds: (request identity, anchor, committed end,
/// remaining output budget).
type DraftSeed = (u64, u32, u64, usize);

/// Reset a lane's draft state before a new proposal attempt. A proposal that
/// fails in its guards must not leave the previous round's plan or confidence
/// trace behind for `shape`/`observe_round`.
pub(crate) fn reset_lane_draft(lane: &mut LaneDraft, confidence_trace: &mut std::collections::BTreeMap<u64, Vec<f32>>,
    neural_lengths: &mut std::collections::BTreeMap<u64, usize>, inputs: &[DraftSeed]) {
    lane.depths.clear();
    lane.selected.clear();
    for &(id, _, _, _) in inputs {
        confidence_trace.remove(&id);
        neural_lengths.remove(&id);
    }
}

/// Preserve the ordinary draft batch when it fits; under pressure select at
/// most one full-width draft per available request slot, ranked by history.
/// Every selected member needs at least one free target verify row.
fn select_draft_members(seeds: &[DraftSeed], depths: &[usize], width: usize, verify_rows: usize,
    probability: impl Fn(u64) -> f64) -> Vec<usize> {
    let mut active: Vec<_> = seeds.iter().enumerate().filter_map(|(i, &(_, _, end, remaining))|
        (end >= 2 && remaining > 1).then_some(i)).collect();
    let capacity = cuteafd_core::v41_draft_members(seeds.len(), width);
    if seeds.len() > capacity {
        active.retain(|&i| depths[i] > 0);
        active.sort_by(|&i, &j| probability(seeds[j].0).total_cmp(&probability(seeds[i].0))
            .then_with(|| seeds[i].0.cmp(&seeds[j].0)));
        active.truncate(capacity.min(verify_rows.saturating_sub(seeds.len())));
        active.sort_unstable();
    }
    active
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
    let mut snapshot = history::snapshot();
    let local = LOCAL_LAYERS.load(Relaxed);
    if local != u64::MAX { snapshot["local_layers"] = serde_json::json!(local); }
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

enum DraftBackend<'w, 'a, C> {
    Dspark {
        mains: Vec<DsparkMainContext<'w, 'a>>,
        chains: Vec<C>,
        windows: [DsparkWindow<'a>; 3],
        requests: std::collections::BTreeMap<u64, DraftRequest>,
    },
    Dfly2 {
        chains: Vec<Dfly2Chain<'w, 'a>>,
        taps: Dfly2TapStore<'a>,
        commit_streams: Vec<Stream<'a>>,
        commit_done: Vec<Event<'a>>,
        pending_updates: Vec<Vec<(u64, Option<u64>)>>,
        failure: DraftFailure,
    },
}

fn slice_dev_buf(buf: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset <= buf.bytes && bytes <= buf.bytes - offset, "DFly2 commit slice out of range");
    Ok(CuteafdDeviceBuffer {
        ptr: unsafe { buf.ptr.cast::<u8>().add(offset).cast() },
        bytes,
        ..buf
    })
}

#[allow(clippy::type_complexity)]
fn prepare_dfly2_tap_updates(
    sources: &Dfly2CommitSources<'_>,
    batch: &RequestBatch,
    accepted: &[u32],
) -> Result<(Vec<(u64, u64, CuteafdDeviceBuffer, [CuteafdDeviceBuffer; 3])>, Vec<(u64, Option<u64>)>)> {
    let ids = batch.cache()?.request_ids();
    ensure!(ids.len() == accepted.len(), "DFly2 acceptance count differs");
    let mut copies = Vec::with_capacity(ids.len());
    let mut frontiers = Vec::with_capacity(ids.len());
    let mut offset = 0usize;
    for (&id, &acc) in ids.iter().zip(accepted) {
        let start = offset;
        while offset < sources.rows.len() && sources.rows[offset].request_id == id {
            offset += 1;
        }
        let count = offset - start;
        ensure!((acc as usize) <= count, "DFly2 accepted prefix exceeds batch rows");
        if acc == 0 {
            continue;
        }
        let last_row = start + acc as usize - 1;
        let new_end = sources.rows[last_row].position + 1;
        if let Some(sel_idx) = sources.selected_rows.iter().position(|&r| r == last_row) {
            let h = slice_dev_buf(sources.normalized, sel_idx * 10240, 10240)?;
            let r0 = slice_dev_buf(sources.recall[0], last_row * 2048, 2048)?;
            let r1 = slice_dev_buf(sources.recall[1], last_row * 2048, 2048)?;
            let r2 = slice_dev_buf(sources.recall[2], last_row * 2048, 2048)?;
            copies.push((id, new_end, h, [r0, r1, r2]));
            frontiers.push((id, Some(new_end)));
        } else {
            frontiers.push((id, None));
        }
    }
    Ok((copies, frontiers))
}

/// Execution workspaces share a request-indexed bank of persistent draft state.
pub(crate) struct DraftRuntime<'w, 'a, C = DsparkChain<'w, 'a>> {
    backend: DraftBackend<'w, 'a, C>,
    pending_commit_ids: Vec<Vec<u64>>,
    pending_prefix_ids: [Option<u64>; 2],
    pending: Vec<Option<(Vec<(u64, u32, u64, usize)>, Instant)>>,
    request_limit: usize,
    verify_rows: usize,
    draft_limit: usize,
    draft_width: usize,
    // The confidence head's raw score per drafted position (dSpark logit or
    // DFly2 candidate probability), downloaded whenever the draft policy is
    // bound, or for diagnostics.
    confidence_trace: std::collections::BTreeMap<u64, Vec<f32>>,
    neural_lengths: std::collections::BTreeMap<u64, usize>,
    /// `DRAFT_POLICY` and the `DRAFT_LENGTH` fixed-width override.
    length: crate::shared::draft::length::DraftLength,
    /// The shared history draft policy.
    policy: Option<Policy>,
    /// Per lane: the round's plan.
    lanes: Vec<LaneDraft>,
}
struct DraftRequest {
    leases: [WindowLease; 3],
    rng: cuteafd_core::DsparkRng,
    slot: usize,
}
impl<'w, 'a> DraftRuntime<'w, 'a> {
    pub(crate) fn capture_startup(&mut self) -> Result<()> {
        ensure!(self.pending.iter().all(Option::is_none),
            "draft startup requires no pending proposals");
        match &mut self.backend {
            DraftBackend::Dspark { chains, windows, requests, .. } => {
                ensure!(requests.is_empty(), "draft startup requires no admitted requests");
                let windows = windows.each_ref();
                for chain in chains { chain.capture_startup(windows)?; }
                self.graph_completed()
            }
            DraftBackend::Dfly2 { .. } => Ok(()),
        }
    }

    pub fn with_requests(
        lib: &'a NativeLibrary, weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>, head: &'w VocabularyHead<'a>,
        capacity: u32, decode_rows: u32, requests: u32, lanes: u8,
    ) -> Result<Self> {
        let (lane_count, lane_requests) = crate::families::deepseek_v41::v41_experts::dspark::draft_lanes(requests, lanes)?;
        let lane_count = lane_count as usize;
        let window = || DsparkWindow::new(lib, requests as usize, capacity,
            DsparkWindow::device_bytes(requests as usize, capacity)?);
        let shared_bytes = weights.draft_bytes(lane_requests)?;
        let lane_bytes = weights.draft_bytes(lane_requests)?;
        let mut mains = vec![weights.main_context(capacity, DsparkMainContext::device_bytes(lib, capacity)?)?];
        if lane_count > 1 { mains.push(weights.main_context(decode_rows, DsparkMainContext::device_bytes(lib, decode_rows)?)?); }
        let chains = (0..lane_count).map(|_| weights.draft(table, head, lane_requests, lane_bytes))
            .collect::<Result<Vec<_>>>()?;
        tracing::info!(lanes=lane_count, lane_requests, draft_width=weights.draft_width(), shared_workspace_bytes=shared_bytes,
            lane_workspace_bytes=lane_bytes, total_workspace_bytes=lane_bytes*lane_count,
            additional_workspace_bytes=(lane_bytes*lane_count).saturating_sub(shared_bytes),
            "lane-local dSpark draft workspaces (weights shared)");
        Ok(Self {
            backend: DraftBackend::Dspark {
                mains,
                chains,
                windows: [window()?, window()?, window()?],
                requests: Default::default(),
            },
            pending_commit_ids: vec![Vec::new(); lane_count],
            pending_prefix_ids: [None, None],
            pending: vec![None; lane_count],
            request_limit: requests as usize,
            verify_rows: decode_rows as usize,
            draft_limit: weights.draft_width(),
            draft_width: weights.draft_width(),
            confidence_trace: Default::default(),
            neural_lengths: Default::default(),
            length: Default::default(),
            policy: None,
            lanes: vec![LaneDraft::default(); lane_count],
        })
    }

    pub fn with_dfly2_single(
        weights: &'w Dfly2Weights<'a>,
        table: &'w NativeRtxTensors<'a>,
        head: &'w VocabularyHead<'a>,
        decode_rows: u32,
        requests: u32,
        lanes: u8,
    ) -> Result<Self> {
        let (lane_count, lane_requests) = crate::families::deepseek_v41::v41_experts::dspark::draft_lanes(requests, lanes)?;
        let lane_count = lane_count as usize;
        let device = weights.device();
        let chains = (0..lane_count)
            .map(|_| Dfly2Chain::new(weights, table, Dfly2Vocab::Single(head), lane_requests as usize))
            .collect::<Result<Vec<_>>>()?;
        let taps = Dfly2TapStore::new(device, requests as usize)?;
        let commit_streams = (0..lane_count).map(|_| Stream::new(device)).collect::<Result<Vec<_>>>()?;
        let commit_done = (0..lane_count).map(|_| Event::new(device)).collect::<Result<Vec<_>>>()?;
        tracing::info!(
            lanes = lane_count,
            lane_requests,
            block_size = weights.block_size(),
            draft_width = weights.draft_width(),
            "lane-local V4.1 DFly2 draft workspaces (weights shared)"
        );
        Ok(Self {
            backend: DraftBackend::Dfly2 {
                chains,
                taps,
                commit_streams,
                commit_done,
                pending_updates: vec![Vec::new(); lane_count],
                failure: DraftFailure::default(),
            },
            pending_commit_ids: vec![Vec::new(); lane_count],
            pending_prefix_ids: [None, None],
            pending: vec![None; lane_count],
            request_limit: requests as usize,
            verify_rows: decode_rows as usize,
            draft_limit: weights.draft_width(),
            draft_width: weights.draft_width(),
            confidence_trace: Default::default(),
            neural_lengths: Default::default(),
            length: Default::default(),
            policy: None,
            lanes: vec![LaneDraft::default(); lane_count],
        })
    }
}
impl<'w, 'a, C: DraftChain<'a>> DraftRuntime<'w, 'a, C> {
    fn execution_device(&self, lane: usize) -> Option<Device<'a>> {
        match &self.backend {
            DraftBackend::Dspark { chains, .. } => chains[lane].execution_device(),
            DraftBackend::Dfly2 { chains, .. } => Some(chains[lane].device()),
        }
    }
    pub fn is_dfly2(&self) -> bool {
        matches!(self.backend, DraftBackend::Dfly2 { .. })
    }
    pub fn has_marks(&self) -> bool {
        !self.is_dfly2()
    }
    pub fn invalidate_prefix(&mut self, id: u64) {
        if let DraftBackend::Dfly2 { taps, .. } = &mut self.backend {
            taps.invalidate(id);
        }
    }
    pub fn admit(&mut self, id: u64) -> Result<()> {
        if let Some(reason) = cuteafd_transport::health::failure_reason() {
            anyhow::bail!("draft runtime unavailable until restart: {reason}");
        }
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        match &mut self.backend {
            DraftBackend::Dspark { windows, requests, .. } => {
                ensure!(!requests.contains_key(&id), "draft request already admitted");
                let slot = (0..self.request_limit).find(|slot| requests.values().all(|request| request.slot != *slot))
                    .context("draft request capacity exhausted")?;
                let mut leases = Vec::new();
                for stage in 0..3 {
                    match windows[stage].begin_request(slot, id) {
                        Ok(lease) => leases.push(lease),
                        Err(error) => {
                            for (stage, lease) in leases.into_iter().enumerate() {
                                let _ = windows[stage].release(lease);
                            }
                            return Err(error);
                        }
                    }
                }
                requests.insert(id, DraftRequest {
                    leases: leases.try_into().ok().context("draft admission incomplete")?,
                    rng: cuteafd_core::DsparkRng::new(id),
                    slot,
                });
                Ok(())
            }
            DraftBackend::Dfly2 { taps, .. } => {
                taps.admit(id)?;
                Ok(())
            }
        }
    }
    pub(super) fn prefill_tap_capacity(&self) -> Option<usize> {
        match &self.backend {
            DraftBackend::Dspark { mains, .. } => Some(mains[0].capacity()),
            DraftBackend::Dfly2 { .. } => None,
        }
    }
    pub fn max_verify_rows(&self) -> usize { self.draft_width + crate::shared::draft::copy::COPY_EXTENSION + 1 }
    pub fn attention_drafts(&self) -> usize { self.draft_limit.min(self.draft_width) }
    pub fn set_length(&mut self, length: crate::shared::draft::length::DraftLength) -> Result<()> {
        self.draft_limit = length.effective(self.draft_width);
        self.length = length;
        Ok(())
    }
    /// Bind the shared history draft policy before serving. The drafter declares
    /// its native block and its confidence head score kind; `local_layers` is
    /// the installed placement's RTX layer count, for the console.
    pub(crate) fn bind_policy(&mut self, local_layers: usize) {
        LOCAL_LAYERS.store(local_layers as u64, std::sync::atomic::Ordering::Relaxed);
        let head = if self.is_dfly2() {
            cuteafd_core::HeadScore::CandidateProbability
        } else {
            cuteafd_core::HeadScore::Logit
        };
        self.policy = Some(Policy::new(self.length, self.draft_width, head, true));
    }
    /// What a verification round records besides its outputs: layer events under the timing profile.
    pub fn capture(&self) -> Capture {
        Capture { routes: false, timing: crate::shared::draft::timing_profile() }
    }
    fn wants_confidence(&self) -> bool {
        self.policy.is_some() || tracing::enabled!(target: "cuteafd::draft_policy", tracing::Level::DEBUG)
    }
    /// Plan the lane round's drafts before the draft pass. `seeds` are every
    /// lane member; a short history or an exhausted budget has no room.
    fn plan(&mut self, lane: usize, seeds: &[DraftSeed]) -> Vec<usize> {
        let requests: Vec<_> = seeds.iter().map(|&(id, _, end, remaining)| {
            let ready = match &self.backend {
                DraftBackend::Dspark { .. } => end >= 2,
                DraftBackend::Dfly2 { taps, .. } => end >= 1 && taps.ready_at(id, end),
            };
            (id, if ready { remaining.saturating_sub(1) } else { 0 })
        }).collect();
        let depths = match &mut self.policy {
            Some(policy) => policy.plan(&requests, self.verify_rows),
            None => requests.iter().map(|&(_, room)| room.min(self.draft_limit)).collect(),
        };
        let mut selected = select_draft_members(seeds, &depths, self.draft_width, self.verify_rows,
            |id| self.policy.as_ref().map_or(0., |policy| policy.probability(id, 0)));
        if let DraftBackend::Dfly2 { taps, .. } = &self.backend {
            selected.retain(|&i| {
                let (id, _, end, _) = seeds[i];
                taps.ready_at(id, end)
            });
        }
        self.lanes[lane].depths = depths.clone();
        self.lanes[lane].selected = selected;
        depths
    }
    /// Fit the lane's proposals to their plan after the draft pass: keep the
    /// calibrated prefix and its explore, extend an agreeing copy, and let the
    /// verifier row budget caps the round. `members` are (identity, the
    /// request's tokens with its anchor last, copies allowed) in lane order;
    /// `inputs` are each member's anchor followed by its drafts.
    pub fn shape(&mut self, members: &[(u64, &[u32], bool)], inputs: &mut [Vec<u32>]) {
        let Some(policy) = self.policy.as_mut() else { return };
        let mut drafts: Vec<Vec<u32>> = inputs.iter().map(|input| input[1..].to_vec()).collect();
        let mut proposals: Vec<_> = members.iter().zip(drafts.iter_mut()).map(|(&(id, history, greedy), tokens)|
            Proposal { id, tokens, head: self.confidence_trace.get(&id).map(Vec::as_slice), history, greedy }).collect();
        let neural = policy.shape(&mut proposals, self.verify_rows);
        drop(proposals);
        for (((input, draft), neural), &(id, _, _)) in inputs.iter_mut().zip(drafts).zip(neural).zip(members) {
            if draft.len() > neural { self.neural_lengths.insert(id, neural); }
            input.truncate(1);
            input.extend(draft);
        }
    }
    /// Feed one completed lane round to the policy; `requests` are in lane order.
    pub fn observe_round(&mut self, requests: &[LaneRequest], timing: Option<history::RoundTiming>) {
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
        let Some(policy) = &mut self.policy else { return };
        let outcomes: Vec<_> = requests.iter().map(|request|
            history::outcome(request.id, request.rows, request.accepted as usize, request.censored)).collect();
        policy.observe_timed(&outcomes, timing);
    }
    pub fn release(&mut self, id: u64) -> Result<()> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        ensure!(!self.pending_prefix_ids.contains(&Some(id)), "cannot release a request with pending snapshot copies");
        ensure!(!pending_draft_blocks(&self.pending, id),
            "cannot release a request with a pending draft");
        ensure!(!self.pending_commit_ids.iter().any(|ids| ids.contains(&id)),
            "cannot release a request with a pending commit");
        if let Some(policy) = &mut self.policy { policy.release(id); }
        self.confidence_trace.remove(&id);
        self.neural_lengths.remove(&id);
        let mut failure = None;
        match &mut self.backend {
            DraftBackend::Dspark { windows, requests, .. } => {
                if let Some(request) = requests.remove(&id) {
                    for (window, lease) in windows.iter_mut().zip(request.leases) {
                        if window.request_id(lease).is_ok() {
                            if let Err(error) = window.release(lease) {
                                failure = Some(error);
                            }
                        }
                    }
                }
            }
            DraftBackend::Dfly2 { taps, .. } => {
                taps.release(id);
            }
        }
        failure.map_or(Ok(()), Err)
    }
    /// The device every draft window (and so every draft mark) lives on; `None` for self-stateless DFly2.
    pub fn mark_device(&self) -> Option<crate::shared::memory::device::Device<'a>> {
        match &self.backend {
            DraftBackend::Dspark { windows, .. } => Some(windows[0].device()),
            DraftBackend::Dfly2 { .. } => None,
        }
    }
    fn ring_at(mark: CuteafdDeviceBuffer, stage: usize) -> CuteafdDeviceBuffer {
        let bytes = cuteafd_ffi::V41DsparkCache::SLOT_BYTES;
        CuteafdDeviceBuffer { ptr: unsafe { mark.ptr.cast::<u8>().add(stage * bytes).cast() }, bytes, ..mark }
    }
    fn check_frontier(&self, id: u64, end: u64) -> Result<[WindowLease; 3]> {
        let DraftBackend::Dspark { windows, requests, .. } = &self.backend else {
            anyhow::bail!("DFly2 does not store draft marks");
        };
        let request = requests.get(&id).context("draft request not admitted")?;
        for (window, lease) in windows.iter().zip(request.leases) {
            ensure!(window.committed_end(lease)? == Some(end), "draft and target prefix frontiers differ");
        }
        Ok(request.leases)
    }
    /// Copy request `id`'s three rings at frontier `end` into `mark` (`MARK_BYTES` on
    /// `mark_device`) synchronously.
    pub fn capture_mark(&mut self, id: u64, end: u64, mark: CuteafdDeviceBuffer) -> Result<()> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        ensure!(mark.bytes == DRAFT_MARK_BYTES, "draft mark storage size differs");
        let leases = self.check_frontier(id, end)?;
        let DraftBackend::Dspark { windows, .. } = &mut self.backend else {
            anyhow::bail!("DFly2 does not store draft marks");
        };
        let device = windows[0].device();
        device.run(|| {
            for (stage, (window, lease)) in windows.iter_mut().zip(leases).enumerate() {
                window.capture_ring(lease, Self::ring_at(mark, stage))?;
            }
            Ok(())
        })
    }
    /// [`DraftRuntime::capture_mark`] queued on capture lane `lane`.
    pub fn queue_mark(&mut self, lane: usize, id: u64, end: u64, mark: CuteafdDeviceBuffer) -> Result<()> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        ensure!(self.pending_prefix_ids.get(lane).context("invalid draft snapshot lane")?.is_none(),
            "draft snapshot lane is occupied");
        ensure!(!self.pending_prefix_ids.contains(&Some(id)), "draft snapshot already pending");
        ensure!(mark.bytes == DRAFT_MARK_BYTES, "draft mark storage size differs");
        let leases = self.check_frontier(id, end)?;
        self.pending_prefix_ids[lane] = Some(id);
        let queued = match &mut self.backend {
            DraftBackend::Dspark { windows, .. } => {
                let device = windows[0].device();
                device.run(|| {
                    for (stage, (window, lease)) in windows.iter_mut().zip(leases).enumerate() {
                        window.queue_ring(lane, lease, Self::ring_at(mark, stage))?;
                    }
                    Ok(())
                })
            }
            DraftBackend::Dfly2 { .. } => Err(anyhow::anyhow!("DFly2 does not store draft marks")),
        };
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
        match &self.backend {
            DraftBackend::Dspark { windows, requests, .. } => {
                let request = requests.get(&id).context("draft request not admitted")?;
                let mut end = None;
                for (window, lease) in windows.iter().zip(request.leases) {
                    let current = window.committed_end(lease)?.context("draft window is not seeded")?;
                    ensure!(end.is_none_or(|e| e == current), "draft windows disagree on the frontier");
                    end = Some(current);
                }
                end.context("no draft windows")
            }
            DraftBackend::Dfly2 { taps, .. } => {
                taps.committed_end(id).context("DFly2 request taps not seeded")
            }
        }
    }
    /// Whether lane `lane`'s queued rings all landed; the lane frees once they have.
    pub fn mark_ready(&mut self, lane: usize) -> Result<bool> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        ensure!(self.pending_prefix_ids.get(lane).is_some_and(Option::is_some), "no draft snapshot is pending");
        let mut ready = true;
        if let DraftBackend::Dspark { windows, .. } = &mut self.backend {
            for window in windows {
                if window.lane_pending(lane) { ready &= window.ring_ready(lane)?; }
            }
        }
        if ready { self.pending_prefix_ids[lane] = None; }
        Ok(ready)
    }
    pub fn abort_mark(&mut self, lane: usize) -> Result<()> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        ensure!(lane < self.pending_prefix_ids.len(), "invalid draft snapshot lane");
        let mut failure = None;
        if let DraftBackend::Dspark { windows, .. } = &mut self.backend {
            for window in windows {
                if let Err(error) = window.abort_ring(lane) { failure.get_or_insert(error); }
            }
        }
        self.pending_prefix_ids[lane] = None;
        failure.map_or(Ok(()), Err)
    }
    /// Drain every restore stream before discard can recycle a draft slot.
    pub fn drain_prefix(&self) -> Result<()> {
        if let DraftBackend::Dspark { windows, .. } = &self.backend {
            for window in windows { window.device().run(|| window.synchronize())?; }
        }
        Ok(())
    }
    /// Restore request `id`'s rings at frontier `end` from `mark`. The caller owns rollback;
    /// errors must not release or readmit a slot whose ring copies may still be queued.
    pub fn restore_mark(&mut self, id: u64, end: u64, mark: CuteafdDeviceBuffer) -> Result<()> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        let DraftBackend::Dspark { windows, requests, .. } = &mut self.backend else {
            anyhow::bail!("DFly2 does not restore draft marks");
        };
        ensure!(mark.bytes == DRAFT_MARK_BYTES && mark.device_id == windows[0].device().id,
            "draft mark storage differs from the runtime's");
        let leases = requests.get(&id).context("draft request not admitted")?.leases;
        let device = windows[0].device();
        device.run(|| {
            for (stage, lease) in leases.into_iter().enumerate() {
                windows[stage].restore_ring(lease, end, Self::ring_at(mark, stage))?;
            }
            Ok(())
        })
    }
    #[cfg(test)]
    pub(crate) fn validate_position(&self, id: u64, end: u64) -> Result<()> {
        let DraftBackend::Dspark { windows, requests, .. } = &self.backend else {
            ensure!(self.committed_end(id)? == end, "DFly2 draft position differs");
            return Ok(());
        };
        let request = requests.get(&id).context("draft request not admitted")?;
        for (window, lease) in windows.iter().zip(request.leases) {
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
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        self.commit_batch(pass, requests, batch, &[accepted])
    }
    pub fn commit_batch(
        &mut self,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: &[u32],
    ) -> Result<()> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        match &mut self.backend {
            DraftBackend::Dspark { mains, windows, requests: draft_requests, .. } => {
                let ids = batch.cache()?.request_ids();
                ensure!(ids.len() == accepted.len(), "draft acceptance count differs");
                let leases = ids.iter().map(|id| draft_requests.get(id)
                    .map(|request| request.leases).context("draft request not admitted"))
                    .collect::<Result<Vec<_>>>()?;
                let stage_leases: [Vec<WindowLease>; 3] = std::array::from_fn(|stage|
                    leases.iter().map(|request| request[stage]).collect());
                let taps = pass.taps(batch)?;
                let mut proposal = unsafe {
                    mains[0]
                        .execute_rows(taps.values(), taps.batch_identity(), taps.rows())?
                };
                let [a, b, c] = windows.as_mut_slice() else {
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
            DraftBackend::Dfly2 { taps, commit_streams, .. } => {
                let sources = pass.dfly2_commit_sources(batch)?;
                let (copies, frontiers) = prepare_dfly2_tap_updates(&sources, batch, accepted)?;
                let device = taps.device();
                device.run(|| {
                    if !copies.is_empty() {
                        unsafe {
                            taps.enqueue_commit_rows(device.library, &copies, commit_streams[0].raw)?;
                        }
                        commit_streams[0].drain()?;
                    }
                    taps.publish_frontiers(&frontiers)
                })?;
                pass.commit(requests, batch, accepted)
            }
        }
    }
    /// Begin a completed decode batch's accepted-cache transaction on its own lane.
    pub fn begin_queued_commit(&mut self, lane: usize, pass: &impl crate::families::deepseek_v41::v41_target_pass::TargetCache<'a>,
        requests: &Requests<'a>, batch: &RequestBatch, accepted: &[u32]) -> Result<()> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        ensure!(self.pending_commit_ids[lane].is_empty(), "commit lane busy or invalid");
        requests.validate_acceptance(batch, accepted)?;
        ensure!(accepted.iter().any(|&n| n > 0), "queued decode commit has no accepted rows");
        let ids = batch.cache()?.request_ids();
        match &mut self.backend {
            DraftBackend::Dspark { mains, windows, requests: draft_requests, .. } => {
                ensure!(lane < mains.len(), "commit lane invalid");
                let leases = ids.iter().map(|id| draft_requests.get(id)
                    .map(|request| request.leases).context("draft commit request not admitted"))
                    .collect::<Result<Vec<_>>>()?;
                let stage_leases: [Vec<WindowLease>; 3] = std::array::from_fn(|stage|
                    leases.iter().map(|request| request[stage]).collect());
                let taps = pass.taps(batch)?;
                let chunks = crate::families::deepseek_v41::v41_experts::dspark::prepare_commit_rows(taps.rows(), windows.each_ref(),
                    stage_leases.each_ref().map(|leases| leases.as_slice()), accepted)?;
                let writes = [windows[0].prepare_async_write(&chunks[0], taps.rows().len() as u32)?,
                    windows[1].prepare_async_write(&chunks[1], taps.rows().len() as u32)?,
                    windows[2].prepare_async_write(&chunks[2], taps.rows().len() as u32)?];
                self.pending_commit_ids[lane] = ids.to_vec();
                unsafe { mains[lane].enqueue_commit(taps.values(), &batch.cache()?.positions(),
                    windows.each_ref(), writes) }
            }
            DraftBackend::Dfly2 { taps, commit_streams, commit_done, pending_updates, .. } => {
                ensure!(lane < commit_streams.len(), "DFly2 commit lane invalid");
                let sources = pass.dfly2_commit_sources(batch)?;
                let (copies, frontiers) = prepare_dfly2_tap_updates(&sources, batch, accepted)?;
                let device = taps.device();
                device.run(|| {
                    if !copies.is_empty() {
                        unsafe {
                            taps.enqueue_commit_rows(device.library, &copies, commit_streams[lane].raw)?;
                        }
                    }
                    commit_done[lane].record(&commit_streams[lane])
                })?;
                pending_updates[lane] = frontiers;
                self.pending_commit_ids[lane] = ids.to_vec();
                Ok(())
            }
        }
    }
    pub fn poll_queued_commit(&self, lane: usize) -> Result<bool> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        match &self.backend {
            DraftBackend::Dspark { mains, .. } => mains[lane].poll_commit(),
            DraftBackend::Dfly2 { taps, commit_done, .. } => taps.device().run(|| unsafe {
                taps.device().library.cuda_event_query(commit_done[lane].raw)
            }),
        }
    }
    pub fn finish_queued_commit(&mut self, lane: usize, pass: &mut impl crate::families::deepseek_v41::v41_target_pass::TargetCache<'a>,
        requests: &mut Requests<'a>, batch: &mut RequestBatch, accepted: &[u32]) -> Result<()> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        match &mut self.backend {
            DraftBackend::Dspark { mains, windows, .. } => {
                mains[lane].publish_commit(windows.as_mut_slice())?;
            }
            DraftBackend::Dfly2 { taps, pending_updates, .. } => {
                let updates = std::mem::take(&mut pending_updates[lane]);
                taps.publish_frontiers(&updates)?;
            }
        }
        pass.commit(requests, batch, accepted)?;
        self.pending_commit_ids[lane].clear();
        Ok(())
    }
    /// Drain before revoking any target/cache storage, including partial enqueue.
    pub fn abort_queued_commit(&mut self, lane: usize, requests: &mut Requests<'a>,
        batch: &mut RequestBatch) -> Result<()> {
        let _device = self.execution_device(0).map(|device| device.enter()).transpose()?;
        // Target publication may have cancelled the batch before returning an
        // error. Keep cleanup independent of its now-unavailable cache view.
        let ids = if self.pending_commit_ids[lane].is_empty() {
            batch.cache().map(|cache| cache.request_ids().to_vec()).unwrap_or_default()
        } else { self.pending_commit_ids[lane].clone() };
        let mut result = match &mut self.backend {
            DraftBackend::Dspark { mains, windows, .. } => mains[lane].abort_commit(windows.as_mut_slice()),
            DraftBackend::Dfly2 { commit_streams, pending_updates, .. } => {
                pending_updates[lane].clear();
                commit_streams[lane].drain()
            }
        };
        self.pending_commit_ids[lane].clear();
        requests.revoke_batch(batch);
        for id in ids { if let Err(error) = self.release(id) { result = Err(error); } }
        result
    }

    /// Each lane owns its draft scratch and stream; both can replay concurrently.
    /// Short RefCell borrows never survive a scheduler yield.
    pub(crate) fn visit_graph_banks(&mut self, f: &mut dyn FnMut(i32, &'static str, bool, &mut dyn crate::shared::decode_graph::CacheAttached) -> Result<()>) -> Result<()> {
        if let DraftBackend::Dspark { mains, chains, .. } = &mut self.backend {
            for main in mains { main.visit_graph_banks(f)?; }
            for chain in chains { chain.visit_graph_banks(f)?; }
        }
        Ok(())
    }
    pub(crate) fn visit_lane_graphs(&mut self, lane: usize,
        f: &mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()> {
        if let DraftBackend::Dspark { mains, chains, .. } = &mut self.backend {
            if let Some(main) = mains.get_mut(lane) { main.visit_graph_banks(f)?; }
            if let Some(chain) = chains.get_mut(lane) { chain.visit_graph_banks(f)?; }
        }
        Ok(())
    }
    fn graph_cycle(&mut self, lane: usize) -> Result<()> {
        if self.is_dfly2() { return Ok(()); }
        crate::families::deepseek_v41::graph_policy::cycle(
            crate::shared::decode_graph::device::Cycle::Verify, |f| self.visit_lane_graphs(lane, f))
    }
    fn graph_completed(&mut self) -> Result<()> {
        let library = match &self.backend {
            DraftBackend::Dspark { windows, .. } => windows[0].device().library,
            DraftBackend::Dfly2 { .. } => return Ok(()),
        };
        unsafe { // SAFETY: V4.1 graph owners borrow persistent engine storage only.
            crate::shared::decode_graph::device::observe_completed(library, |f| self.visit_graph_banks(f)) }
    }
    fn graph_lane_completed(&mut self, lane: usize) -> Result<()> {
        let library = match &self.backend {
            DraftBackend::Dspark { windows, .. } => windows[0].device().library,
            DraftBackend::Dfly2 { .. } => return Ok(()),
        };
        // SAFETY: this completed draft owns only persistent captured storage.
        unsafe { crate::shared::decode_graph::device::observe_completed(library,
            |f| self.visit_lane_graphs(lane, f)) }
    }
    pub fn poll_propose(&mut self, lane: usize,
        inputs: &[(u64, u32, u64, usize)],
    ) -> Result<Option<(Vec<Vec<u32>>, u64)>> {
        self.poll_propose_with_cache(lane, inputs, None)
    }
    pub fn poll_propose_with_cache(&mut self, lane: usize,
        inputs: &[(u64, u32, u64, usize)],
        cache: Option<&BackboneCache<'a>>,
    ) -> Result<Option<(Vec<Vec<u32>>, u64)>> {
        let chain_count = match &self.backend {
            DraftBackend::Dspark { chains, .. } => chains.len(),
            DraftBackend::Dfly2 { chains, .. } => chains.len(),
        };
        ensure!(lane < chain_count, "invalid draft lane");
        let _device = self.execution_device(lane).map(|device| device.enter()).transpose()?;
        if let Some((seeds, started)) = self.pending[lane].take() {
            if seeds != inputs {
                match &mut self.backend {
                    DraftBackend::Dspark { chains, .. } => chains[lane].cancel_replay(),
                    DraftBackend::Dfly2 { chains, .. } => chains[lane].cancel_pending(),
                }
                if let Some(round) = self.lanes.get_mut(lane) {
                    reset_lane_draft(round, &mut self.confidence_trace, &mut self.neural_lengths, &seeds);
                }
                anyhow::bail!("pending draft request inputs changed");
            }
            match &mut self.backend {
                DraftBackend::Dspark { chains, .. } => {
                    let completed = match chains[lane].poll_replay() {
                        Ok(None) => { self.pending[lane] = Some((seeds, started)); return Ok(None); },
                        Ok(Some(value)) => value,
                        Err(error) => {
                            chains[lane].note_failure(&error);
                            if let Some(round) = self.lanes.get_mut(lane) {
                                reset_lane_draft(round, &mut self.confidence_trace, &mut self.neural_lengths, inputs);
                            }
                            return Err(error);
                        },
                    };
                    self.graph_lane_completed(lane)?;
                    let (packed, values) = completed;
                    let active: Vec<_> = self.lanes[lane].selected.iter().map(|&i| (i, &inputs[i])).collect();
                    let count = active.len();
                    let width = self.draft_width;
                    ensure!(packed.len() == (width+1)*count && values.len() == width*count, "draft output extent differs");
                    let mut outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
                    for (row, &(output, &(id, anchor, _, remaining))) in active.iter().enumerate() {
                        if self.wants_confidence() {
                            self.confidence_trace.insert(id, (0..width).map(|step| values[step*count+row]).collect());
                        }
                        let tokens: Vec<_> = (0..=width).map(|step| packed[step*count+row]).collect();
                        ensure!(tokens[0] == anchor && tokens.iter().all(|&t| t < 129280), "invalid draft tokens");
                        let depth = self.lanes[lane].depths.get(output).copied().unwrap_or(0);
                        outputs[output] = tokens[..remaining.min(depth.min(width)+1)].to_vec();
                    }
                    return Ok(Some((outputs, started.elapsed().as_micros() as u64)));
                }
                DraftBackend::Dfly2 { chains, failure, .. } => {
                    let completed = match chains[lane].poll() {
                        Ok(None) => { self.pending[lane] = Some((seeds, started)); return Ok(None); }
                        Ok(Some(value)) => value,
                        Err(error) => {
                            failure.record(&error);
                            if let Some(round) = self.lanes.get_mut(lane) {
                                reset_lane_draft(round, &mut self.confidence_trace, &mut self.neural_lengths, inputs);
                            }
                            return Err(error);
                        }
                    };
                    let (tokens_rd, probs_rd) = completed;
                    let active: Vec<_> = self.lanes[lane].selected.iter().map(|&i| (i, &inputs[i])).collect();
                    let count = active.len();
                    let width = self.draft_width;
                    ensure!(tokens_rd.len() == width * count && probs_rd.len() == width * count, "DFly2 output extent differs");
                    let mut outputs: Vec<_> = inputs.iter().map(|(_, anchor, _, _)| vec![*anchor]).collect();
                    for (row, &(output, &(id, anchor, _, remaining))) in active.iter().enumerate() {
                        let req_tokens = &tokens_rd[row * width..(row + 1) * width];
                        let req_probs = &probs_rd[row * width..(row + 1) * width];
                        ensure!(req_tokens.iter().all(|&t| t < 129280), "invalid DFly2 draft tokens");
                        if self.wants_confidence() {
                            self.confidence_trace.insert(id, req_probs.to_vec());
                        }
                        let depth = self.lanes[lane].depths.get(output).copied().unwrap_or(0);
                        let take = remaining.saturating_sub(1).min(depth.min(width));
                        let mut seq = Vec::with_capacity(1 + take);
                        seq.push(anchor);
                        seq.extend_from_slice(&req_tokens[..take]);
                        outputs[output] = seq;
                    }
                    return Ok(Some((outputs, started.elapsed().as_micros() as u64)));
                }
            }
        }
        let started = Instant::now();
        ensure!(self.pending[lane].is_none(), "shared draft proposal still pending");
        if let Some(round) = self.lanes.get_mut(lane) {
            reset_lane_draft(round, &mut self.confidence_trace, &mut self.neural_lengths, inputs);
        }
        ensure!(!inputs.is_empty() && inputs.len() <= cuteafd_core::v41_lane_members(self.lanes.len()), "invalid draft batch size");
        let mut seen = std::collections::BTreeSet::new();
        for &(id, anchor, _, remaining) in inputs {
            let admitted = match &self.backend {
                DraftBackend::Dspark { requests, .. } => requests.contains_key(&id),
                DraftBackend::Dfly2 { taps, .. } => taps.contains(id),
            };
            ensure!(seen.insert(id) && admitted, "invalid draft request identity");
            ensure!(anchor < 129280 && remaining > 0, "invalid draft request input");
        }
        let outputs = anchor_inputs(inputs);
        let depths = self.plan(lane, inputs);
        let active: Vec<_> = self.lanes[lane].selected.iter().map(|&i| (i, &inputs[i])).collect();
        if active.is_empty() || depths.iter().all(|&depth| depth == 0) {
            return Ok(Some((outputs, started.elapsed().as_micros() as u64)));
        }
        let count = active.len();
        self.graph_cycle(lane)?;
        match &mut self.backend {
            DraftBackend::Dspark { chains, windows, requests, .. } => {
                let tokens: Vec<_> = active.iter().map(|(_, (_, anchor, _, _))| *anchor as i32).collect();
                let bindings: [Vec<_>; 3] = std::array::from_fn(|stage| active.iter()
                    .map(|(_, (id, _, end, _))| (requests[id].leases[stage], *end)).collect());
                if let Err(error) = chains[lane].stage_tokens(&tokens) {
                    chains[lane].note_failure(&error);
                    return Err(error);
                }
                let mut rngs: Vec<_> = requests.iter_mut().filter_map(|(id, request)|
                    active.iter().position(|(_, (active_id, _, _, _))| active_id == id)
                        .map(|index| (index, &mut request.rng))).collect();
                rngs.sort_by_key(|(index, _)| *index);
                if let Err(error) = chains[lane].stage_sampling(&mut rngs.into_iter().map(|(_, rng)| rng).collect::<Vec<_>>(),
                    &vec![0.0; count]) {
                    chains[lane].note_failure(&error);
                    return Err(error);
                }
                let windows = windows.each_ref();
                let bindings = bindings.each_ref().map(|rows| rows.as_slice());
                if let Err(error) = unsafe { chains[lane].begin_replay(windows, bindings) } {
                    chains[lane].note_failure(&error);
                    return Err(error);
                }
            }
            DraftBackend::Dfly2 { chains, taps, failure, .. } => {
                let cache = cache.context("DFly2 draft requires live backbone cache")?;
                let items: Vec<(u64, u32, u64)> = active.iter().map(|&(_, &(id, anchor, end, _))| (id, anchor, end)).collect();
                let mut ctx = [Vec::with_capacity(count), Vec::with_capacity(count), Vec::with_capacity(count)];
                for stage in 0..3 {
                    for &(id, _, _) in &items {
                        let recall = taps.recall_row(id, stage)?;
                        ctx[stage].push(cache.dfly2_context_by_id(id, stage, recall)?);
                    }
                }
                let refs = [ctx[0].as_slice(), ctx[1].as_slice(), ctx[2].as_slice()];
                if let Err(error) = unsafe { chains[lane].begin(taps, &items, refs) } {
                    failure.record(&error);
                    return Err(error);
                }
            }
        }
        self.pending[lane] = Some((inputs.to_vec(), started));
        Ok(None)
    }

    pub fn confidence_trace(&self, id: u64) -> Option<&[f32]> {
        self.confidence_trace.get(&id).map(Vec::as_slice)
    }
}

impl<'w, 'a, C: DraftChain<'a>> DraftRuntime<'w, 'a, C> {
    /// Liveness gate for the anchor-only fallback: a terminal draft propagates
    /// its error instead of silently degrading every round to anchors.
    pub fn require_live(&self, lane: usize) -> Result<()> {
        match &self.backend {
            DraftBackend::Dspark { chains, .. } => chains.get(lane).context("invalid draft lane")?.require_live(),
            DraftBackend::Dfly2 { chains, failure, .. } => {
                ensure!(lane < chains.len(), "invalid draft lane");
                failure.check("V4.1 DFly2 draft chain")?;
                require_peer_live()
            }
        }
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
            "draft proposal failed; this lane round verifies anchors only");
        Ok(anchor_inputs(seeds))
    }

}
impl<'w, 'a, C: ProposeChain<'a>> DraftRuntime<'w, 'a, C> {
    pub fn propose(&mut self, lane: usize,
        inputs: &[(u64, u32, u64, usize)],
    ) -> Result<Vec<Vec<u32>>> {
        self.propose_with_cache(lane, inputs, None)
    }
    /// Inputs are (request identity, anchor, committed end, remaining output budget).
    /// Returned rows retain input order; short histories/budgets use the anchor only.
    pub fn propose_with_cache(&mut self, lane: usize,
        inputs: &[(u64, u32, u64, usize)],
        cache: Option<&BackboneCache<'a>>,
    ) -> Result<Vec<Vec<u32>>> {
        let chain_count = match &self.backend {
            DraftBackend::Dspark { chains, .. } => chains.len(),
            DraftBackend::Dfly2 { chains, .. } => chains.len(),
        };
        ensure!(lane < self.lanes.len() && lane < chain_count, "invalid draft lane");
        ensure!(self.pending.iter().all(Option::is_none), "shared draft proposal still pending");
        reset_lane_draft(&mut self.lanes[lane], &mut self.confidence_trace, &mut self.neural_lengths, inputs);
        ensure!(!inputs.is_empty() && inputs.len() <= cuteafd_core::v41_lane_members(self.lanes.len()), "invalid draft batch size");
        let mut seen = std::collections::BTreeSet::new();
        for &(id, anchor, _, remaining) in inputs {
            let admitted = match &self.backend {
                DraftBackend::Dspark { requests, .. } => requests.contains_key(&id),
                DraftBackend::Dfly2 { taps, .. } => taps.contains(id),
            };
            ensure!(seen.insert(id) && admitted, "invalid draft request identity");
            ensure!(anchor < 129280 && remaining > 0, "invalid draft request input");
        }
        let mut outputs = anchor_inputs(inputs);
        let depths = self.plan(lane, inputs);
        let active: Vec<_> = self.lanes[lane].selected.iter().map(|&i| (i, &inputs[i])).collect();
        if active.is_empty() || depths.iter().all(|&depth| depth == 0) { return Ok(outputs); }
        let width = self.draft_width;
        let count = active.len();
        self.graph_cycle(lane)?;
        let wants_conf = self.wants_confidence();
        match &mut self.backend {
            DraftBackend::Dspark { chains, windows, requests, .. } => {
                let tokens: Vec<_> = active.iter().map(|(_, (_, anchor, _, _))| *anchor as i32).collect();
                let bindings: [Vec<_>; 3] = std::array::from_fn(|stage| active.iter()
                    .map(|(_, (id, _, end, _))| (requests[id].leases[stage], *end)).collect());
                if let Err(error) = chains[lane].set_tokens(&tokens) {
                    chains[lane].note_failure(&error);
                    return Err(error);
                }
                let mut rngs: Vec<_> = requests.iter_mut().filter_map(|(id, request)|
                    active.iter().position(|(_, (active_id, _, _, _))| active_id == id)
                        .map(|index| (index, &mut request.rng))).collect();
                rngs.sort_by_key(|(index, _)| *index);
                if let Err(error) = chains[lane].prepare_sampling(
                    &mut rngs.into_iter().map(|(_, rng)| rng).collect::<Vec<_>>(), &vec![0.0; count]) {
                    chains[lane].note_failure(&error);
                    return Err(error);
                }
                let windows = windows.each_ref();
                let bindings = bindings.each_ref().map(|rows| rows.as_slice());
                if !chains[lane].has_graph(count) {
                    if let Err(error) = unsafe { chains[lane].capture(windows, bindings) } {
                        chains[lane].note_failure(&error);
                        return Err(error);
                    }
                }
                if let Err(error) = unsafe { chains[lane].replay(windows, bindings) } {
                    chains[lane].note_failure(&error);
                    return Err(error);
                }
                let buffer = match chains[lane].draft_output() {
                    Ok(output) => output[0],
                    Err(error) => { chains[lane].note_failure(&error); return Err(error); }
                };
                let mut bytes = vec![0; buffer.bytes];
                if let Err(error) = chains[lane].copy_d2h(&mut bytes, buffer) {
                    chains[lane].note_failure(&error);
                    return Err(error);
                }
                ensure!(bytes.len() == (width + 1) * count * 4, "draft token extent differs");
                let packed: Vec<_> = bytes.chunks_exact(4)
                    .map(|b| u32::from_ne_bytes(b.try_into().unwrap())).collect();
                if wants_conf {
                    let confidence = match chains[lane].draft_output() {
                        Ok(output) => output[2],
                        Err(error) => { chains[lane].note_failure(&error); return Err(error); }
                    };
                    ensure!(confidence.bytes == width * count * 4, "draft confidence extent differs");
                    let mut bytes = vec![0; confidence.bytes];
                    if let Err(error) = chains[lane].copy_d2h(&mut bytes, confidence) {
                        chains[lane].note_failure(&error);
                        return Err(error);
                    }
                    let values: Vec<_> = bytes.chunks_exact(4)
                        .map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();
                    for (row, &(_, &(id, _, _, _))) in active.iter().enumerate() {
                        self.confidence_trace.insert(id,
                            (0..width).map(|step| values[step * count + row]).collect());
                    }
                }
                self.graph_lane_completed(lane)?;
                for (row, &(output, &(_, anchor, _, remaining))) in active.iter().enumerate() {
                    let tokens: Vec<_> = (0..=width).map(|step| packed[step * count + row]).collect();
                    ensure!(tokens[0] == anchor && tokens.iter().all(|&token| token < 129280), "invalid draft tokens");
                    outputs[output] = tokens[..remaining.min(depths[output].min(width) + 1)].to_vec();
                }
            }
            DraftBackend::Dfly2 { chains, taps, failure, .. } => {
                let cache = cache.context("DFly2 draft requires live backbone cache")?;
                let items: Vec<(u64, u32, u64)> = active.iter().map(|&(_, &(id, anchor, end, _))| (id, anchor, end)).collect();
                let mut ctx = [Vec::with_capacity(count), Vec::with_capacity(count), Vec::with_capacity(count)];
                for stage in 0..3 {
                    for &(id, _, _) in &items {
                        let recall = taps.recall_row(id, stage)?;
                        ctx[stage].push(cache.dfly2_context_by_id(id, stage, recall)?);
                    }
                }
                let refs = [ctx[0].as_slice(), ctx[1].as_slice(), ctx[2].as_slice()];
                let (tokens_rd, probs_rd) = match unsafe { chains[lane].propose_sync(taps, &items, refs) } {
                    Ok(v) => v,
                    Err(error) => { failure.record(&error); return Err(error); }
                };
                ensure!(tokens_rd.len() == width * count && probs_rd.len() == width * count, "DFly2 output extent differs");
                for (row, &(output, &(id, anchor, _, remaining))) in active.iter().enumerate() {
                    let req_tokens = &tokens_rd[row * width..(row + 1) * width];
                    let req_probs = &probs_rd[row * width..(row + 1) * width];
                    ensure!(req_tokens.iter().all(|&t| t < 129280), "invalid DFly2 draft tokens");
                    if wants_conf {
                        self.confidence_trace.insert(id, req_probs.to_vec());
                    }
                    let take = remaining.saturating_sub(1).min(depths[output].min(width));
                    let mut seq = Vec::with_capacity(1 + take);
                    seq.push(anchor);
                    seq.extend_from_slice(&req_tokens[..take]);
                    outputs[output] = seq;
                }
            }
        }
        Ok(outputs)
    }

}

#[cfg(test)]
mod restored_prefix_tests {
    use super::*;

    #[test]
    fn wide_lane_draft_members_follow_history_and_verify_budget() {
        let seeds: Vec<_> = (0..64).map(|i| (i, 1, 10, 8)).collect();
        let depths = vec![7; 64];
        for width in [5, 7] {
            assert_eq!(select_draft_members(&seeds, &depths, width, 128, |id| id as f64), (32..64).collect::<Vec<_>>());
            assert!(select_draft_members(&seeds, &depths, width, 64, |_| 1.).is_empty());
            assert_eq!(select_draft_members(&seeds[..48], &depths[..48], width, 64, |id| id as f64), (32..48).collect::<Vec<_>>());
            assert_eq!(select_draft_members(&seeds[..32], &depths[..32], width, 64, |_| 1.), (0..32).collect::<Vec<_>>());
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

#[cfg(test)]
mod anchor_fallback_tests {
    use super::*;
    use crate::families::deepseek_v41::v41_experts::dspark::DraftFailure;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use std::cell::Cell;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum At { Tokens, Sampling, Capture, Replay, Output, TokensCopy, ConfidenceCopy, Begin, Poll, Never }
    struct FailingChain {
        failure: DraftFailure,
        at: At,
        device_error: bool,
        tokens: Vec<u32>,
        polls: usize,
        cancelled: bool,
        downloads: Cell<usize>,
        graph_bank: Option<crate::shared::decode_graph::GraphBank<usize, ()>>,
    }
    impl FailingChain {
        fn new(at: At, device_error: bool) -> Self {
            Self { failure: Default::default(), at, device_error, tokens: Vec::new(), polls: 0,
                cancelled: false, downloads: Cell::new(0), graph_bank: None }
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
        fn visit_graph_banks(&mut self, f: &mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()> {
            if let Some(bank) = &mut self.graph_bank { f(0, "dspark_chain", false, bank)?; }
            Ok(())
        }
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
        fn require_live(&self) -> Result<()> { self.failure.check("test draft chain") }
    }
    impl<'a> ProposeChain<'a> for FailingChain {
        fn has_graph(&self, _: usize) -> bool { self.at != At::Capture }
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
            backend: DraftBackend::Dspark {
                mains: Vec::new(),
                chains: (0..2).map(|_| FailingChain::new(at, device_error)).collect(),
                windows: [window()?, window()?, window()?],
                requests: Default::default(),
            },
            pending_commit_ids: vec![Vec::new(); 2],
            pending_prefix_ids: [None, None], pending: vec![None; 2],
            request_limit: 4, verify_rows: cuteafd_core::V41_VERIFY_ROWS, draft_limit: 5, draft_width: 5, confidence_trace: Default::default(),
            neural_lengths: Default::default(), length: Default::default(), policy: None,
            lanes: vec![LaneDraft::default(); 2],
        };
        for id in 1..=4 { runtime.admit(id)?; }
        runtime.bind_policy(0);
        Ok(runtime)
    }
    fn dspark_parts<'r, 'w, 'a, C>(runtime: &'r mut DraftRuntime<'w, 'a, C>) -> (&'r mut Vec<C>, &'r mut [DsparkWindow<'a>; 3], &'r mut std::collections::BTreeMap<u64, DraftRequest>) {
        match &mut runtime.backend {
            DraftBackend::Dspark { chains, windows, requests, .. } => (chains, windows, requests),
            DraftBackend::Dfly2 { .. } => unreachable!("expected dSpark backend in test"),
        }
    }
    fn seeds(ids: &[u64]) -> Vec<DraftSeed> {
        ids.iter().enumerate().map(|(i, &id)| (id, 11 + i as u32, 5, 9)).collect()
    }
    #[test]
    fn failed_ring_restore_drain_rejects_readmission_and_idle_health() -> Result<()> {
        const CHILD: &str = "CUTEAFD_DRAFT_RESTORE_DRAIN_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "families::deepseek_v41::v41_native_serve::speculative::anchor_fallback_tests::failed_ring_restore_drain_rejects_readmission_and_idle_health", "--nocapture"])
                .env(CHILD, "1").status()?;
            ensure!(status.success(), "isolated draft-drain regression failed");
            return Ok(());
        }
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        let mut runtime = runtime(&lib, At::Never, false)?;
        let mark = crate::shared::memory::DeviceAllocation::new(&lib, DRAFT_MARK_BYTES,
            crate::shared::memory::TEST)?;
        let lease = dspark_parts(&mut runtime).2[&1].leases[0];
        tokio::runtime::Runtime::new()?.block_on(async {
            use axum::{body::Body, http::{Request, StatusCode}};
            use tower::ServiceExt;
            let (queue, mut receive) = tokio::sync::mpsc::channel(1);
            let mut profile = cuteafd_api::openai::ModelProfile::default();
            profile.engine_health = Some(cuteafd_api::openai::health::HealthWitness(
                std::sync::Arc::new(cuteafd_transport::health::failure_reason)));
            let app = cuteafd_api::openai::router_for_model(queue, Default::default(),
                std::sync::Arc::new(std::sync::Mutex::new(serde_json::Value::Null)),
                std::time::Duration::from_secs(1), cuteafd_api::openai::ConsoleHub::disabled(), profile);
            let get = || Request::get("/health").body(Body::empty()).unwrap();
            assert_eq!(app.clone().oneshot(get()).await?.status(), StatusCode::OK);
            fixture.configure_pack(&lib, 0, 1)?;
            assert!(runtime.restore_mark(1, 63, mark.buffer).is_err());
            assert!(dspark_parts(&mut runtime).2.contains_key(&1), "restore must retain failed request ownership");
            assert!(runtime.drain_prefix().is_err());
            runtime.release(1)?; // Poisoned component ownership remains held, not recycled.
            assert!(runtime.admit(1).is_err(), "failed ring slot must never be readmitted");
            assert!(dspark_parts(&mut runtime).1[0].begin_request(0, 99).is_err());
            assert!(dspark_parts(&mut runtime).1[0].request_id(lease).is_err(), "poisoned ring is unreadable");
            assert_eq!(app.oneshot(get()).await?.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(receive.try_recv().is_err());
            fixture.configure_pack(&lib, 0, 0)?;
            anyhow::Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn synchronous_proposal_records_each_failure_on_its_own_lane() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        for lane in 0..2 {
            for at in [At::Tokens, At::Sampling, At::Capture, At::Replay, At::Output, At::TokensCopy, At::ConfidenceCopy] {
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
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        for lane in 0..2 {
            for at in [At::Tokens, At::Sampling, At::Begin, At::Poll] {
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
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        let mut runtime = runtime(&lib, At::Never, false)?;
        let round = seeds(&[1, 2]);
        assert!(runtime.poll_propose(1, &round)?.is_none());
        assert!(runtime.release(1).is_err());
        assert!(runtime.poll_propose(1, &round)?.is_none());
        assert!(runtime.release(2).is_err());
        runtime.confidence_trace.insert(1, vec![0.5; 5]);
        runtime.neural_lengths.insert(1, 3);
        assert!(runtime.poll_propose(1, &seeds(&[3, 4])).is_err());
        assert!(dspark_parts(&mut runtime).0[1].cancelled);
        assert!(runtime.lanes[1].depths.is_empty());
        assert!(runtime.confidence_trace(1).is_none() && !runtime.neural_lengths.contains_key(&1));
        runtime.release(1)?;
        runtime.release(2)?;
        Ok(())
    }
    #[test]
    fn both_proposal_completion_paths_measure_pending_captures() -> Result<()> {
        use crate::shared::decode_graph::{device::{Basis, Cycle, Preference, Site}, GraphBank};
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        for independent in [false, true] {
            let mut runtime = runtime(&lib, At::Never, false)?;
            let mut bank = GraphBank::new();
            bank.attach_device(&lib, 0, 64 << 20, Site { name: "draft_completion_fixture", cycle: Cycle::Other,
                preference: Preference::Adaptive, bootstrap: 16, basis: Basis::Bytes, spans_devices: false })?;
            let ticket = bank.cache_ticket(&1, false, None).context("fixture admission")?;
            bank.cache_commit(1, (), ticket, 1, 1, false);
            assert!(!bank.cache_needs_boundary(), "commit already sampled physical growth");
            let pending = bank.cache_ticket(&2, false, None).context("pending completion fixture")?;
            dspark_parts(&mut runtime).0[1].graph_bank = Some(bank);
            let round = seeds(&[1, 2]);
            if independent {
                assert!(runtime.poll_propose(1, &round)?.is_none());
                assert!(dspark_parts(&mut runtime).0[1].graph_bank.as_ref().unwrap().cache_needs_boundary());
                assert!(runtime.poll_propose(1, &round)?.is_none());
                assert!(runtime.poll_propose(1, &round)?.is_some());
            } else {
                runtime.propose(1, &round)?;
            }
            assert!(!dspark_parts(&mut runtime).0[1].graph_bank.as_ref().unwrap().cache_needs_boundary(),
                "completed proposal must feed measured_capture, independent={independent}");
            drop(pending);
            dspark_parts(&mut runtime).0[1].graph_bank.as_mut().unwrap().cache_clear_drained();
        }
        Ok(())
    }
    #[test]
    fn successful_poll_preserves_order_and_invalid_lane_is_an_error() -> Result<()> {
        let fixture = fixture()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
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
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        for independent in [false, true] {
            let mut runtime = runtime(&lib, At::Never, false)?;
            let previous = seeds(&[1, 2, 3, 4]);
            runtime.propose(0, &previous)?;
            assert_eq!(runtime.lanes[0].depths, [5; 4]);
            runtime.neural_lengths.insert(1, 3);
            let mut round = seeds(&[1, 2]);
            round[0].1 = 129280; // Host validation fails before drafting starts.
            let error = if independent { runtime.poll_propose(0, &round).unwrap_err() }
                else { runtime.propose(0, &round).unwrap_err() };
            assert!(runtime.lanes[0].depths.is_empty());
            assert!(runtime.confidence_trace(1).is_none() && !runtime.neural_lengths.contains_key(&1));
            round[0].1 = 11;
            let mut inputs = runtime.anchor_fallback(0, &round, &error)?;
            assert_eq!(inputs, vec![vec![11], vec![12]]);
            // The anchor-only round shapes and observes as undrafted.
            runtime.shape(&[(1, &[11], true), (2, &[12], true)], &mut inputs);
            assert_eq!(inputs, vec![vec![11], vec![12]]);
            runtime.observe_round(&[LaneRequest { id: 1, rows: 1, accepted: 1, censored: false },
                LaneRequest { id: 2, rows: 1, accepted: 1, censored: false }], None);
        }
        Ok(())
    }
}
