//! Full target-pass sequencing with GPU-local lanes and short request borrows.
use super::*;
use crate::families::deepseek_v41::v41_target_head::distributed_target::{DistributedTargetHead, DistributedTargetLogits};
mod encoder_units;
use crate::families::deepseek_v41::v41_backbone_cache::CachePlacement;
use crate::families::deepseek_v41::v41_backbone_execution::DistributedExecution;
use crate::shared::peer_split::hop::DirectInputHop;
use crate::families::deepseek_v41::v41_engram::placement::PlacedEngram;
use crate::shared::memory::device::DeviceOwner;

#[cfg(test)]
fn trace_query_component(query: &crate::families::deepseek_v41::v41_attention_query::AttentionQueryOutput<'_>)
    -> Result<(cuteafd_ffi::CuteafdDeviceBuffer, usize)> {
    Ok(match std::env::var("CUTEAFD_QUERY_COMPONENT").as_deref().unwrap_or("rotated") {
        "hidden" => (query.hidden, 10240),
        "raw_rank" => (query.raw_rank, 2560),
        "normalized_rank" => (query.normalized_rank, 2560),
        "projected" | "qb_pair" => (query.projected, 65536),
        "rotated" => (query.rotated, 65536),
        _ => anyhow::bail!("unknown query trace component"),
    })
}


pub(crate) type DistributedTargetPass<'w, 'a> = Pass<'a, PlacedResources<'w, 'a>>;

pub(crate) struct PlacedResources<'w, 'a> {
    map: CachePlacement,
    embedding: DeviceOwner<'a, TargetEmbeddingWave<'w, 'a>>,
    lanes: [DeviceOwner<'a, BackboneLane<'w, 'a>>; 2],
    indices: [Option<DeviceOwner<'a, IndexLane<'w, 'a>>>; 2],
    execution: DistributedExecution<'w, 'a>,
    engram: PlacedEngram<'w, 'a>,
    head: DistributedTargetHead<'w, 'a>,
    taps: DeviceOwner<'a, TargetTapWave<'a>>,
    transfers: [DirectInputHop<'a>; 2],
    suffix_stream: DeviceOwner<'a, crate::shared::memory::LoadStream<'a>>,
    timeout: Duration,
    capture_routes: bool,
    route_capture: Vec<Vec<[u32; 6]>>,
    verification: bool,
    #[cfg(test)]
    trace: bool,
    #[cfg(test)]
    trace_records: Vec<(u64, usize, usize)>,
    #[cfg(test)]
    trace_queries: [DeviceOwner<'a, crate::shared::memory::DeviceAllocation<'a>>; 2],
    #[cfg(test)]
    observed_layer: Option<std::rc::Rc<std::cell::Cell<usize>>>,
    #[cfg(test)]
    trace_buffers: [DeviceOwner<'a, crate::shared::memory::DeviceAllocation<'a>>; 2],
}
/// Execution futures drain their borrowed GPU work before this guard revokes
/// reserved chunks, including cancellation during queued early publication.
struct ReservedPassGuard<'p, 'r, 'q, 'w, 'a> {
    pass: &'p mut DistributedTargetPass<'w, 'a>,
    requests: &'r std::cell::RefCell<&'q mut Requests<'a>>,
    batch: &'p mut RequestBatch,
    complete: bool,
}
impl Drop for ReservedPassGuard<'_, '_, '_, '_, '_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        let mut requests = self.requests.borrow_mut();
        if let Err(error) = self.pass.abort_cache_commit(&mut requests) {
            tracing::error!(%error, "draining cancelled distributed encoder publication");
        }
        requests.revoke_batch(self.batch);
        if let Err(error) = self.pass.discard(self.batch) {
            tracing::error!(%error, "discarding cancelled distributed encoder pass");
        }
    }
}
impl<'w, 'a> DistributedTargetPass<'w, 'a> {
    pub fn bind_execution_plan(&mut self, placement: &cuteafd_loader::placement::Placement) -> Result<()> {
        use cuteafd_loader::placement::{HopKind, HopPoint, LayerExecutionPlan};
        let plan = LayerExecutionPlan::new(placement,
            cuteafd_loader::placement::families::deepseek_v41::execution_hops(2))?;
        let map = CachePlacement::from_execution_plan(&plan)?;
        ensure!(map == self.resources.map, "distributed target differs from admitted ownership");
        for hop in plan.hops() {
            let HopPoint::BeforeLayer(layer) = hop.at else {
                anyhow::bail!("V4.1 block transfer requires an adjacent-layer hop")
            };
            ensure!((1..40).contains(&layer) && hop.kind == HopKind::Boundary
                && map.attention(layer - 1)? == usize::from(hop.from)
                && map.attention(layer)? == usize::from(hop.to),
                "V4.1 block transfer differs from admitted hop");
        }
        for (gpu, transfer) in self.resources.transfers.iter_mut().enumerate() {
            let hops: Vec<_> = plan.hops().iter().filter(|hop| usize::from(hop.to) == gpu).copied().collect();
            transfer.bind(&hops, plan.spec())?;
        }
        Ok(())
    }
    pub fn enable_dual_attention(&mut self)->Result<()> {
        for lane in &mut self.resources.lanes {
            let device=lane.device;
            let peer=crate::shared::memory::device::Device { library:device.library,id:1-device.id };
            let budget=lane.dual_attention_bytes()?;
            device.run(||lane.enable_dual_attention(peer,budget))?;
        }
        Ok(())
    }
    pub fn configure_cache_replicas(&mut self,bank:&crate::families::deepseek_v41::v41_backbone_cache::BackboneCache<'a>)->Result<()> {
        bank.configure_producer_replicas(&mut self.resources.execution.producers.windows,
            &mut self.resources.execution.producers.sources)
    }
    pub fn encoder_device(&self) -> Result<crate::shared::memory::device::Device<'a>> {
        Ok(self.resources.lanes[self.resources.map.attention(19)?].device)
    }
    pub fn new(
        map: CachePlacement,
        embedding: DeviceOwner<'a, TargetEmbeddingWave<'w, 'a>>,
        lanes: [DeviceOwner<'a, BackboneLane<'w, 'a>>; 2],
        indices: [Option<DeviceOwner<'a, IndexLane<'w, 'a>>>; 2],
        execution: DistributedExecution<'w, 'a>,
        engram: PlacedEngram<'w, 'a>,
        head: DistributedTargetHead<'w, 'a>,
        taps: DeviceOwner<'a, TargetTapWave<'a>>,
        timeout: Duration,
    ) -> Result<Self> {
        ensure!(
            !timeout.is_zero() && lanes[0].device.id == 0 && lanes[1].device.id == 1,
            "invalid distributed target lane devices/timeout"
        );
        ensure!(
            embedding.device.id == map.attention(0)? as i32
                && head.device().id == map.attention(39)? as i32
                && taps.device.id == head.device().id,
            "distributed embedding/head/tap placement differs"
        );
        for layer in [2, 8, 14, 20, 24, 28, 32, 36] {
            let gpu = map.attention(layer)?;
            ensure!(
                indices[gpu]
                    .as_ref()
                    .is_some_and(|i| i.device.id == gpu as i32),
                "distributed index owner absent"
            );
        }
        let spec = cuteafd_loader::placement::families::deepseek_v41::execution_hops(2);
        let transfers = [
            DirectInputHop::new(lanes[1].device, lanes[0].device, spec)?,
            DirectInputHop::new(lanes[0].device, lanes[1].device, spec)?,
        ];
        let device = lanes[map.attention(19)?].device;
        let suffix_stream = device.own(|| {
            Ok(crate::shared::memory::LoadStream {
                library: device.library,
                raw: unsafe { device.library.cuda_stream_create()? },
            })
        })?;
        #[cfg(test)]
        let trace_buffers = [
            lanes[0].device.own(|| crate::shared::memory::DeviceAllocation::new(device.library, 40 * 16 * 40976))?,
            lanes[1].device.own(|| crate::shared::memory::DeviceAllocation::new(device.library, 40 * 16 * 40976))?,
        ];
        #[cfg(test)]
        let trace_queries = [
            lanes[0].device.own(|| crate::shared::memory::DeviceAllocation::new(device.library, 60 * 16 * 131072))?,
            lanes[1].device.own(|| crate::shared::memory::DeviceAllocation::new(device.library, 60 * 16 * 131072))?,
        ];
        let chain = crate::shared::memory::chain::enabled()
            .then(|| crate::shared::memory::chain::StageChain::on_devices(lanes[0].device.library, &[0, 1]))
            .transpose()?;
        let mut lanes = lanes;
        // Reserve host history at planning time. The per-layer IDs are already
        // present in each router's completed pinned staging; collecting them
        // introduces no device transfer, wait, or cross-lane dependency.
        for layer in 0..40 {
            lanes[map.attention(layer)?].reserve_route_capture(layer..layer + 1, 4096)?;
        }
        Ok(Self {
            resources: PlacedResources {
                map,
                embedding,
                lanes,
                indices,
                execution,
                engram,
                head,
                taps,
                transfers,
                suffix_stream,
                timeout,
                capture_routes: false,
                route_capture: (0..40).map(|_| Vec::with_capacity(4096)).collect(),
                verification: false,
                #[cfg(test)]
                trace: false,
                #[cfg(test)]
                trace_records: Vec::new(),
                #[cfg(test)]
                trace_queries,
                #[cfg(test)]
                observed_layer: None,
                #[cfg(test)]
                trace_buffers,
            },
            state: State::Idle,
            chain,
            device_order: true,
            device_default: true,
            l2_solo: true,
            sampled: None,
        })
    }
    /// Selects this prefill slot's rows in every component's shared row planes
    /// on both GPUs (`shared::memory::row_window`), at rest.
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(self.state != State::Idle, "prefill slot", "a pass is running or uncommitted")?;
        let resources = &mut self.resources;
        // At rest the previous unit's finished outputs are dropped (the next
        // use restarts every owner); pending work still refuses.
        for index in resources.indices.iter_mut().flatten() {
            let device = index.device;
            device.run(|| index.restart())?;
        }
        resources.execution.restart_for(CacheStage::Encoder);
        for lane in &mut resources.lanes {
            let device = lane.device;
            device.run(|| lane.get_mut().select_window(window.clone()))?;
        }
        for index in resources.indices.iter_mut().flatten() {
            let device = index.device;
            device.run(|| index.get_mut().select_window(window.clone()))?;
        }
        resources.execution.select_window(window.clone())?;
        let device = resources.embedding.device;
        device.run(|| resources.embedding.get_mut().select_window(window.clone()))?;
        resources.engram.select_window(window.clone())?;
        let device = resources.taps.device;
        device.run(|| resources.taps.get_mut().select_window(window))
    }
    /// A prefill unit's input (`Requests::prepare_unit_input`) on layer 0's GPU.
    /// # Safety
    /// As `begin_stage`: the unit owns its batch and its rows.
    pub(super) fn prepare_unit_input(&mut self, requests: &std::cell::RefCell<&mut Requests<'a>>,
        batch: &RequestBatch) -> Result<()> {
        self.resources.execution.restart_for(CacheStage::Encoder);
        for index in self.resources.indices.iter_mut().flatten() {
            let device = index.device;
            device.run(|| index.restart())?;
        }
        let gpu = self.resources.map.attention(0)?;
        let device = self.resources.lanes[gpu].device;
        ensure!(self.resources.embedding.device.id == device.id, "embedding and layer 0 differ in GPU");
        device.run(|| self.resources.lanes[gpu].restart())?;
        let (embedding, lane) = (self.resources.embedding.get_mut(), self.resources.lanes[gpu].get_mut());
        // SAFETY: the unit owns its rows of the block planes (caller contract).
        device.run(|| unsafe { requests.borrow().prepare_unit_input(batch, embedding, lane) })
    }
    /// `--l2-prefetch` on both GPUs' lanes, each for the layers it holds.
    pub fn enable_l2_prefetch(&mut self, budget: usize) -> Result<()> {
        for lane in &mut self.resources.lanes {
            let device = lane.device;
            device.run(|| lane.get_mut().enable_l2_prefetch(budget))?;
        }
        Ok(())
    }
    /// Retain decode-row graph shapes on both owning devices (sticky).
    pub fn retain_small_graph_shapes(&mut self) -> Result<()> {
        ensure!(self.state == State::Idle, "cannot change graph retention during a target pass");
        for lane in &mut self.resources.lanes {
            let device = lane.device;
            device.run(|| { lane.retain_small_graph_shapes(); Ok(()) })?;
        }
        for index in self.resources.indices.iter_mut().flatten() {
            index.get_mut().enable_small_graph_shapes();
        }
        Ok(())
    }
    pub fn set_route_capture(&mut self, enabled: bool) -> Result<()> {
        if enabled {
            return self.set_capture(crate::families::deepseek_v41::v41_backbone_lane::Capture { routes: true, timing: true });
        }
        for lane in &mut self.resources.lanes {
            let device = lane.device;
            device.run(|| { lane.set_route_capture(false); Ok(()) })?;
        }
        self.resources.capture_routes = false;
        Ok(())
    }
    /// Route ids and/or pass timing on both owning devices (see `BackboneLane::set_capture`).
    pub fn set_capture(&mut self, capture: crate::families::deepseek_v41::v41_backbone_lane::Capture) -> Result<()> {
        let enabled = capture.routes || capture.timing;
        ensure!(!enabled || self.state == State::Idle,
            "cannot reset route capture during a target pass");
        for lane in &mut self.resources.lanes {
            let device = lane.device;
            device.run(|| { lane.set_capture(capture); Ok(()) })?;
        }
        self.resources.capture_routes = capture.routes;
        if enabled {
            // Match the single-GPU pass: adaptive row changes must retain the
            // index projection and selection graphs on both owning devices.
            for index in self.resources.indices.iter_mut().flatten() {
                index.get_mut().enable_small_graph_shapes();
            }
            for rows in &mut self.resources.route_capture { rows.clear(); }
        }
        Ok(())
    }
    pub fn captured_routes(&self) -> &[Vec<[u32; 6]>] { &self.resources.route_capture }
    /// Host FFN stage split of the last captured pass, summed over both GPUs.
    pub fn captured_ffn_split(&self) -> crate::families::deepseek_v41::v41_backbone_lane::FfnSplit {
        let mut split = crate::families::deepseek_v41::v41_backbone_lane::FfnSplit::default();
        for lane in &self.resources.lanes { split.add(&lane.captured_ffn_split()); }
        split
    }
    /// The next `execute` is a verification pass (consumed by that call).
    pub(crate) fn mark_verification(&mut self) { self.resources.verification = true; }
    /// Device-select the rows of a completed non-greedy verification pass.
    /// # Safety
    /// The pass completed with `greedy = false` and its head is unconsumed.
    pub(crate) async unsafe fn sample_head(&mut self,
        requests: &[crate::families::deepseek_v41::v41_target_head::TargetSamplingRowRequest], masks: Option<&[u32]>,
        mask_words: usize, ordered_rows: bool) -> Result<()> {
        self.sampled = None;
        let rows = unsafe { self.resources.head.sample(requests, masks, mask_words, ordered_rows).await? };
        self.sampled = Some(rows);
        Ok(())
    }
    pub(crate) fn take_sampled(&mut self) -> Result<crate::families::deepseek_v41::v41_target_head::SampledTargetRows> {
        self.sampled.take().context("distributed pass has no sampled rows")
    }
    pub(crate) async fn download_sampled(&mut self, rows: &crate::families::deepseek_v41::v41_target_head::SampledTargetRows,
        selection: &[usize]) -> Result<Vec<u8>> {
        self.resources.head.download_sampled(rows, selection).await
    }
    /// Device time per layer between FFN finishes on the same GPU. The first
    /// layer after a GPU handoff is timed from its input's arrival (one clock
    /// per GPU), which omits only the peer transfer itself.
    pub fn captured_layer_us(&self) -> Vec<Option<f64>> {
        (0..40usize).map(|layer| {
            let previous = layer.checked_sub(1)?;
            let (a, b) = (self.resources.map.attention(previous).ok()?, self.resources.map.attention(layer).ok()?);
            let lane = &self.resources.lanes[b];
            let clock = lane.layer_clock();
            lane.device.run(|| Ok(if a == b { clock.between(previous, layer) }
                else { clock.since_entry(layer) })).ok().flatten()
        }).collect()
    }
    pub fn install_attention_plan(&mut self, plan: cuteafd_core::coordinator_programs::AttentionReductionPlan) -> Result<()> {
        for lane in &mut self.resources.lanes {
            let device = lane.device;
            device.run(|| lane.get_mut().install_attention_plan(plan))?;
        }
        Ok(())
    }
    pub fn set_attention_decode(&mut self, decode: bool) {
        for lane in &mut self.resources.lanes { lane.get_mut().set_attention_decode(decode); }
    }
    pub fn reserve_sparse_decode_rows(&mut self, rows: usize) -> Result<()> {
        for lane in &mut self.resources.lanes {
            let device = lane.device;
            device.run(|| lane.get_mut().reserve_sparse_decode_rows(rows))?;
        }
        Ok(())
    }
    async unsafe fn advance(&mut self, layer: usize) -> Result<()> {
        ensure!((1..40).contains(&layer), "invalid distributed next layer");
        let source = self.resources.map.attention(layer - 1)?;
        let destination = self.resources.map.attention(layer)?;
        if source == destination {
            let lane = &mut self.resources.lanes[destination];
            let device = lane.device;
            device.run(|| lane.advance())
        } else {
            let [left, right] = &mut self.resources.lanes;
            let (source, destination_lane) = if source == 0 {
                (left, right)
            } else {
                (right, left)
            };
            let output = source.output()?;
            unsafe {
                destination_lane
                    .get_mut()
                    .import_previous_cooperative(&output, &mut self.resources.transfers[destination])
                    .await
            }
        }
    }
    /// # Safety
    /// All owners describe one model/capacity/map and this admitted batch. Each
    /// concurrent request lane owns a separate pass and transport. The caller
    /// retains suffix/head/model owners through completion or drained cancellation.
    pub async unsafe fn execute(
        &mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch,
        transport: &mut DeviceOwner<'a, NativeTp4Wave<'a>>,
        placement: u64,
        selected: &[usize],
        suffix: Option<&mut DeviceOwner<'a, EncoderSuffix<'a>>>,
        encoder: Option<&BlockOutput<'_>>,
        greedy: bool,
    ) -> Result<()> {
        // Verification passes run with device-ordered stages. Peer-projection
        // modes (TP2 attention/query/output, all opt-in) keep host drains.
        let verification = std::mem::replace(&mut self.resources.verification, false);
        self.set_attention_decode(verification);
        // `--l2-prefetch` only while the other lane is idle (consumed per pass).
        let solo = std::mem::replace(&mut self.l2_solo, true);
        for lane in &mut self.resources.lanes { lane.get_mut().set_l2_prefetch(solo); }
        let chained = verification && !self.resources.lanes.iter().any(|lane| lane.uses_peer_projection());
        let Some(handle) = self.begin_chain(chained)? else {
            let result = unsafe { self.execute_unchained(requests, batch, transport, placement, selected,
                suffix, encoder, greedy).await };
            // A failed Spark wire released its waits over garbage planes.
            let result = result.and_then(|()| crate::shared::spark_waves::healthy());
            // Remote layers left their captured routes in the routers' rings.
            return self.settle_route_rings(result);
        };
        let timed = tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG).then(Instant::now);
        let result = crate::shared::memory::chain::watchdog(unsafe { handle.scope(self.execute_unchained(
            requests, batch, transport, placement, selected, suffix, encoder, greedy)) }).await;
        let issued_us = timed.map(|t| t.elapsed().as_micros() as u64);
        let drained = self.drain_chain();
        let drained_us = timed.map(|t| t.elapsed().as_micros() as u64);
        // A failed Spark wire released its waits over garbage planes.
        let result = result.and(drained).and_then(|()| crate::shared::spark_waves::healthy());
        if let (Some(t), Some(issued_us), Some(drained_us)) = (timed, issued_us, drained_us) {
            tracing::debug!(target: "cuteafd::timing", issued_us, drain_us=drained_us-issued_us,
                check_us=t.elapsed().as_micros() as u64-drained_us, "distributed verification chain");
        }
        let ring_timed = timed.map(|_| Instant::now());
        // Remote and device-ordered local layers left their routes in the routers' rings.
        let result = self.settle_route_rings(result);
        if let Some(t) = ring_timed {
            tracing::debug!(target: "cuteafd::timing", route_ring_us=t.elapsed().as_micros() as u64,
                "distributed route ring drain");
        }
        result
    }
    /// After a pass: a successful one's ring captures into the history; a
    /// failed one's dropped, so no router is left with an undrained ring (its
    /// next window change would refuse). Returns the pass's result first.
    pub(super) fn settle_route_rings(&mut self, result: Result<()>) -> Result<()> {
        if result.is_ok() { return self.drain_route_rings(); }
        self.discard_route_rings()?;
        result
    }
    /// Every lane's undrained ring captures dropped (failed or cancelled work).
    pub(super) fn discard_route_rings(&mut self) -> Result<()> {
        for lane in &mut self.resources.lanes {
            let device = lane.device;
            device.run(|| lane.get_mut().discard_route_ring())?;
        }
        Ok(())
    }
    /// After a drained pass: every ring-captured layer's routes into the capture.
    pub(super) fn drain_route_rings(&mut self) -> Result<()> {
        if !self.resources.capture_routes { return Ok(()); }
        for gpu in 0..self.resources.lanes.len() {
            let device = self.resources.lanes[gpu].device;
            let lane = &mut self.resources.lanes[gpu];
            for layer in device.run(|| lane.get_mut().drain_route_ring())? {
                self.resources.route_capture[layer].clone_from(&self.resources.lanes[gpu].captured_routes()[layer]);
            }
        }
        Ok(())
    }
    async unsafe fn execute_unchained(
        &mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch,
        transport: &mut DeviceOwner<'a, NativeTp4Wave<'a>>,
        placement: u64,
        selected: &[usize],
        suffix: Option<&mut DeviceOwner<'a, EncoderSuffix<'a>>>,
        encoder: Option<&BlockOutput<'_>>,
        greedy: bool,
    ) -> Result<()> {
        requests.with_requests(|r| r.validate(batch))?;
        ensure!(self.state == State::Idle, "distributed pass still in use");
        if batch.cache()?.is_reserved() {
            let mut guard = ReservedPassGuard {
                pass: self,
                requests,
                batch,
                complete: false,
            };
            unsafe {
                guard
                    .pass
                    .execute_inner(
                        requests,
                        guard.batch,
                        transport,
                        placement,
                        selected,
                        suffix,
                        encoder,
                        greedy,
                    )
                    .await?;
            }
            guard.complete = true;
            Ok(())
        } else {
            unsafe {
                self.execute_inner(
                    requests, batch, transport, placement, selected, suffix, encoder, greedy,
                )
                .await
            }
        }
    }
    async unsafe fn execute_inner(
        &mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch,
        transport: &mut DeviceOwner<'a, NativeTp4Wave<'a>>,
        placement: u64,
        selected: &[usize],
        mut suffix: Option<&mut DeviceOwner<'a, EncoderSuffix<'a>>>,
        encoder: Option<&BlockOutput<'_>>,
        greedy: bool,
    ) -> Result<()> {
        let begin_timed = tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG).then(Instant::now);
        requests.with_requests(|r| r.validate(batch))?;
        let stage = batch.cache()?.stage();
        let id = batch.cache()?.identity();
        let rows = batch.cache()?.positions().len();
        let reserved = batch.cache()?.is_reserved();
        ensure!(
            !reserved || stage == CacheStage::Encoder,
            "reserved stage differs"
        );
        ensure!(
            transport.device.id == self.resources.map.attention(20)? as i32,
            "decoder transport reduction GPU differs"
        );
        ensure!(
            (stage.is_encoder() || !selected.is_empty())
                && selected.len() <= 80
                && selected.iter().all(|&i| i < rows)
                && selected
                    .iter()
                    .enumerate()
                    .all(|(i, row)| !selected[..i].contains(row)),
            "invalid distributed head selection"
        );
        self.state.begin()?;
        self.resources.engram.reset_uploads()?;
        let guard = BatchGuard {
            batch,
            completed: false,
        };
        let mut guard = guard;
        unsafe { self.begin_stage(requests, guard.batch, stage, encoder).await?; }
        if let Some(t) = begin_timed {
            tracing::debug!(target: "cuteafd::timing", rows, begin_us=t.elapsed().as_micros() as u64,
                "distributed pass begin");
        }
        layer_schedule!(stage; layer, first; {
            let gpu = self.resources.map.attention(layer)?;
            let device = self.resources.lanes[gpu].device;
        }; device.library; {
            if layer != first {
                unsafe {
                    self.advance(layer).await?;
                }
                if (2..14).contains(&layer) && !self.resources.engram.staged(1)
                    && self.resources.engram.can_enqueue(1)? {
                    if let cuteafd_loader::EngramGatherPoll::Ready(lease) = requests.with_requests(|r| r.poll_engram_layer(guard.batch, 1))? {
                        self.resources.engram.enqueue_lease(lease, guard.batch.engram_wave()?)?;
                    }
                }
                if [1, 14].contains(&layer) {
                    let gate = usize::from(layer == 14);
                    if !self.resources.engram.staged(gate) {
                        guard.batch.wait_engram_gather(gate, self.resources.timeout).await?;
                    self.resources.engram.wait_enqueue(gate, self.resources.timeout).await?;
                        match requests.with_requests(|r| r.poll_engram_gather(guard.batch, &self.resources.lanes[gpu]))? {
                            cuteafd_loader::EngramGatherPoll::Ready(lease) => self.resources.engram.enqueue_lease(lease, guard.batch.engram_wave()?)?,
                            cuteafd_loader::EngramGatherPoll::Cancelled => anyhow::bail!("distributed Engram gather cancelled"),
                            cuteafd_loader::EngramGatherPoll::Pending => anyhow::bail!("engram gather wait returned pending"),
                        }
                    }
                    unsafe { self.resources.engram.apply_staged(&mut self.resources.lanes[gpu], gate).await?; }
                }
                if layer >= 37 {
                    let input = self.resources.lanes[gpu].prepared_input()?;
                    device
                        .future(unsafe {
                            self.resources.taps
                                .get_mut()
                                .capture_cooperative(guard.batch.cache()?, &input)
                        })
                        .await?;
                }
                device
                    .future(unsafe { self.resources.lanes[gpu].get_mut().begin_prepared_cooperative() })
                    .await?;
            }
            #[cfg(test)]
            let trace_stream = self.resources.lanes[gpu].trace_stream();
            #[cfg(test)]
            if self.resources.trace && std::env::var_os("CUTEAFD_TRACE_QUERY").is_some() {
                let query = self.resources.lanes[gpu].query_output()?;
                let position = query.tokens()?[0];
                ensure!(position as usize + query.rows <= 16, "query trace extent exceeded");
                let (captured, width) = trace_query_component(&query)?;
                let mut bytes = query.rows * width;
                let offset = ((layer + 20) * 16 + position as usize) * 131072;
                let buffer = self.resources.trace_queries[gpu].buffer;
                let destination = cuteafd_ffi::CuteafdDeviceBuffer {
                    ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer
                };
                device.run(|| unsafe { device.library.copy_d2d_async(destination,
                    captured, bytes, trace_stream) })?;
                if std::env::var("CUTEAFD_QUERY_COMPONENT").as_deref() == Ok("qb_pair") {
                    let extra = query.rows * 2560;
                    let destination = cuteafd_ffi::CuteafdDeviceBuffer {
                        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset + bytes).cast() }, bytes: extra, ..buffer
                    };
                    device.run(|| unsafe { device.library.copy_d2d_async(destination,
                        query.normalized_rank, extra, trace_stream) })?;
                    bytes += extra;
                }
                self.resources.trace_records.push((position, layer + 120, bytes));
                if std::env::var("CUTEAFD_QUERY_COMPONENT").as_deref() == Ok("qb_pair") {
                    let qb_scratch=query.qb_scratch.context("full query-B trace scratch absent")?;
                    let bytes = qb_scratch.bytes;
                    ensure!(bytes <= 131072, "QB scratch exceeds diagnostic slot");
                    let offset = ((layer + 40) * 16 + position as usize) * 131072;
                    let destination = cuteafd_ffi::CuteafdDeviceBuffer {
                        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer
                    };
                    device.run(|| unsafe { device.library.copy_d2d_async(destination,
                        qb_scratch, bytes, trace_stream) })?;
                    self.resources.trace_records.push((position, layer + 160, bytes));
                }
            }
            let index = if [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer) {
                self.resources.indices[gpu].as_mut()
            } else {
                None
            };
            let mut production = unsafe {
                requests.with_requests(|r| {
                    self.resources.execution.enqueue_production(
                        r.cache(),
                        guard.batch.cache()?,
                        &self.resources.lanes[gpu],
                        index,
                    )
                })?
            };
            while !unsafe {
                if reserved {
                    requests
                        .borrow_mut()
                        .poll_distributed_encoder_production(guard.batch, &mut production)?
                } else {
                    requests.with_requests(|r| production.poll(r.cache(), guard.batch.cache()?))?
                }
            } {
                tokio::task::yield_now().await;
            }
            drop(production);
            #[cfg(test)]
            if self.resources.trace && std::env::var_os("CUTEAFD_TRACE_QUERY").is_some() {
                let query = self.resources.lanes[gpu].query_output()?;
                let position = query.tokens()?[0];
                ensure!(position as usize + query.rows <= 16, "query trace extent exceeded");
                let (captured, width) = trace_query_component(&query)?;
                let mut bytes = query.rows * width;
                let offset = (layer * 16 + position as usize) * 131072;
                let buffer = self.resources.trace_queries[gpu].buffer;
                let destination = cuteafd_ffi::CuteafdDeviceBuffer {
                    ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer
                };
                device.run(|| unsafe { device.library.copy_d2d_async(destination,
                    captured, bytes, trace_stream) })?;
                if std::env::var("CUTEAFD_QUERY_COMPONENT").as_deref() == Ok("qb_pair") {
                    let extra = query.rows * 2560;
                    let destination = cuteafd_ffi::CuteafdDeviceBuffer {
                        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset + bytes).cast() }, bytes: extra, ..buffer
                    };
                    device.run(|| unsafe { device.library.copy_d2d_async(destination,
                        query.normalized_rank, extra, trace_stream) })?;
                    bytes += extra;
                }
                self.resources.trace_records.push((position, layer + 80, bytes));
            }
            let prepared = unsafe {
                requests.with_requests(|r| {
                    self.resources.execution.prepare_layer(
                        r.cache(),
                        guard.batch.cache()?,
                        &mut self.resources.lanes[gpu],
                        self.resources.indices[gpu].as_ref().map(DeviceOwner::get),
                    )
                })?
            };
            #[cfg(test)]
            let prepared = if self.resources.trace && std::env::var_os("CUTEAFD_TRACE_FFN").is_some()
                && std::env::var("CUTEAFD_TRACE_LAYER").ok().map_or(true,
                    |value| value.split(',').any(|v| v.parse::<usize>().ok() == Some(layer))) {
                let (prepared, record) = unsafe {
                    prepared.trace_ffn_input(self.resources.trace_buffers[gpu].buffer, trace_stream).await?
                };
                self.resources.trace_records.push(record);
                prepared
            } else { prepared };
            let completed = unsafe {
                prepared
                    .execute(transport.get_mut(), placement, guard.batch.image_mask())
                    .await?
            };
            unsafe {
                self.resources.execution
                    .complete_layer(guard.batch.cache()?, &mut self.resources.lanes[gpu], completed)
                    .await?;
            }
            if self.resources.capture_routes {
                self.resources.route_capture[layer].clone_from(&self.resources.lanes[gpu].captured_routes()[layer]);
                if tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG) {
                    let mut seen = [false; 384];
                    for &expert in self.resources.route_capture[layer].iter().flatten() {
                        if let Some(value) = seen.get_mut(expert as usize) { *value = true; }
                    }
                    tracing::debug!(target: "cuteafd::timing", layer, rows,
                        distinct_experts=seen.iter().filter(|&&value| value).count(),
                        "distributed routed expert reuse");
                }
            }
            #[cfg(test)]
            if let Some(progress) = &self.resources.observed_layer { progress.set(layer + 1); }
            #[cfg(test)]
            if self.resources.trace && std::env::var("CUTEAFD_TRACE_LAYER").ok()
                .map_or(true, |value| value.split(',').any(|v| v.parse::<usize>().ok() == Some(layer))) {
                let output = self.resources.lanes[gpu].output()?;
                let captured = if std::env::var_os("CUTEAFD_TRACE_PRE").is_some() {
                    output.pre
                } else { output.residual };
                let position = output.tokens[0];
                let paired = std::env::var_os("CUTEAFD_TRACE_FFN").is_some();
                let offset = ((layer + if paired { 20 } else { 0 }) * 16 + position as usize) * 40976;
                ensure!(position as usize + output.tokens.len() <= 16, "trace extent exceeded");
                let buffer = self.resources.trace_buffers[gpu].buffer;
                ensure!(offset + captured.bytes <= buffer.bytes, "trace buffer exceeded");
                let destination = cuteafd_ffi::CuteafdDeviceBuffer {
                    ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
                    bytes: captured.bytes, ..buffer
                };
                // Queue behind this producer and before its next overwrite. No
                // host wait is added between layers; read back after the run.
                device.run(|| unsafe { device.library.copy_d2d_async(destination, captured, captured.bytes,
                    self.resources.lanes[gpu].trace_stream()) })?;
                let mut captured_bytes = captured.bytes;
                if std::env::var_os("CUTEAFD_TRACE_PRE").is_none() {
                    let destination = cuteafd_ffi::CuteafdDeviceBuffer {
                        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset + captured_bytes).cast() },
                        bytes: output.pre.bytes, ..buffer
                    };
                    device.run(|| unsafe { device.library.copy_d2d_async(destination, output.pre,
                        output.pre.bytes, self.resources.lanes[gpu].trace_stream()) })?;
                    captured_bytes += output.pre.bytes;
                }
                self.resources.trace_records.push((position, layer + if paired { 40 } else { 0 }, captured_bytes));
            }
            if reserved {
                unsafe {
                    self.publish_encoder_layer(requests, guard.batch, layer)
                        .await?;
                }
            }
        });
        if stage.is_encoder() {
            let suffix = suffix.as_mut().context("distributed encoder suffix owner missing")?;
            unsafe { self.finish_encoder_stage(requests, guard.batch, stage, suffix).await?; }
            self.state = State::Encoded(id);
        } else {
            self.resources.taps.output(guard.batch.cache()?)?;
            let gpu = self.resources.map.attention(39)?;
            let output = self.resources.lanes[gpu].output()?;
            let head_timed = begin_timed.map(|_| Instant::now());
            unsafe { self.resources.head.execute_block_with_greedy(&output, selected, greedy).await?; }
            if let Some(t) = head_timed {
                tracing::debug!(target: "cuteafd::timing", rows, greedy, head_us=t.elapsed().as_micros() as u64,
                    "distributed pass head");
            }
            self.state = State::Ready(id);
        }
        guard.completed = true;
        Ok(())
    }
    /// Restart the pass's owners for `batch` at `stage` and embed its rows (or
    /// import the retained encoder suffix for the decoder replay).
    /// # Safety
    /// The pass exclusively owns its lanes; `encoder` is this batch's retained suffix.
    async unsafe fn begin_stage(&mut self, requests: &std::cell::RefCell<&mut Requests<'a>>,
        batch: &RequestBatch, stage: CacheStage, encoder: Option<&BlockOutput<'_>>) -> Result<()> {
        self.resources.execution.restart_for(stage);
        for index in self.resources.indices.iter_mut().flatten() {
            let device = index.device;
            device.run(|| {
                if stage == CacheStage::Replay {
                    index.restart_decoder()
                } else {
                    index.restart()
                }
            })?;
        }
        if !stage.is_encoder() {
            self.resources.taps.begin(batch.cache()?)?;
        }
        let first = stage.windows().start;
        let gpu = self.resources.map.attention(first)?;
        let device = self.resources.lanes[gpu].device;
        if stage == CacheStage::Replay {
            let encoder = encoder.context("distributed replay requires retained encoder suffix")?;
            ensure!(
                encoder.tokens == batch.cache()?.positions(),
                "distributed replay suffix rows differ"
            );
            if encoder.residual.device_id == device.id {
                device.run(|| self.resources.lanes[gpu].restart_decoder(encoder))?;
            } else {
                unsafe {
                    self.resources.lanes[gpu]
                        .get_mut()
                        .import_previous_cooperative(encoder, &mut self.resources.transfers[gpu])
                        .await?;
                }
            }
            device
                .future(unsafe { self.resources.lanes[gpu].get_mut().begin_prepared_cooperative() })
                .await?;
        } else {
            device.run(|| self.resources.lanes[gpu].restart())?;
            let text = requests.with_requests(|r| r.text_embedding_input(batch))?;
            if let Some((tokens, positions)) = text {
                device
                    .future(unsafe {
                        self.resources.lanes[gpu].get_mut().begin_tokens_cooperative(
                            self.resources.embedding.get_mut(),
                            tokens,
                            &positions,
                        )
                    })
                    .await?;
            } else {
                unsafe {
                    requests.with_requests(|r| {
                        device.run(|| {
                            r.begin_input(
                                batch,
                                self.resources.embedding.get_mut(),
                                self.resources.lanes[gpu].get_mut(),
                            )
                        })
                    })?;
                }
            }
        }
        Ok(())
    }
    /// The encoder tail: capture the suffix rows, then (full encoder) advance
    /// to layer 20 and produce the decoder source.
    /// # Safety
    /// The pass's encoder output for `batch` is complete; `suffix` is this prompt's.
    async unsafe fn finish_encoder_stage(&mut self, requests: &std::cell::RefCell<&mut Requests<'a>>,
        batch: &RequestBatch, stage: CacheStage, suffix: &mut DeviceOwner<'a, EncoderSuffix<'a>>) -> Result<()> {
        let reserved = batch.cache()?.is_reserved();
        let gpu = self.resources.map.attention(19)?;
        let device = self.resources.lanes[gpu].device;
        ensure!(suffix.device.id == device.id, "encoder suffix GPU differs");
        let output = self.resources.lanes[gpu].output()?;
        if crate::shared::memory::chain::deferred() {
            // Device-ordered prefill units (#815 option (i)): no host wait;
            // the unit's commit drains the chain before the suffix is read.
            let stream = self.resources.suffix_stream.raw;
            device.run(|| unsafe { suffix.get_mut().capture_chained(&output, stream) })?;
        } else {
            device.future(unsafe { suffix.get_mut().capture_cooperative(&output, &self.resources.suffix_stream) }).await?;
        }
        if stage == CacheStage::Encoder {
            unsafe { self.advance(20).await?; }
            let gpu = self.resources.map.attention(20)?;
            let device = self.resources.lanes[gpu].device;
            device.future(unsafe { self.resources.lanes[gpu].get_mut().begin_prepared_cooperative() }).await?;
            let mut production = unsafe { requests.with_requests(|r| self.resources.execution.enqueue_production(
                r.cache(), batch.cache()?, &self.resources.lanes[gpu], None))? };
            while !unsafe {
                if reserved { requests.borrow_mut().poll_distributed_encoder_production(batch, &mut production)? }
                else { requests.with_requests(|r| production.poll(r.cache(), batch.cache()?))? }
            } {
                tokio::task::yield_now().await;
            }
            drop(production);
            self.resources.execution.finish_decoder_source(batch.cache()?)?;
            // The boundary's query projection and its source-20 reader are done
            // (the production completed): free the lane's guarded scratch as
            // attention does after its join, so a later unit's acquire orders
            // after this boundary (C2).
            let lane = &self.resources.lanes[gpu];
            lane.device.run(|| lane.release_scratch())?;
        }
        Ok(())
    }
    /// A unit's encoder tail on the shared executor (units end in order).
    /// # Safety
    /// As [`Self::finish_encoder_stage`].
    pub(super) async unsafe fn finish_encoder_unit(&mut self, requests: &std::cell::RefCell<&mut Requests<'a>>,
        batch: &RequestBatch, suffix: &mut DeviceOwner<'a, EncoderSuffix<'a>>) -> Result<()> {
        unsafe { self.finish_encoder_stage(requests, batch, CacheStage::Encoder, suffix).await?; }
        self.state = State::Encoded(batch.cache()?.identity());
        Ok(())
    }
    async unsafe fn publish_encoder_layer(
        &mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>,
        batch: &RequestBatch,
        layer: usize,
    ) -> Result<()> {
        unsafe {
            requests.with_requests(|r| {
                self.resources.execution
                    .enqueue_encoder_publication(r.cache(), batch.cache()?, layer)
            })?;
        }
        while !self.resources.execution.poll_encoder_publication()? {
            tokio::task::yield_now().await;
        }
        requests
            .borrow_mut()
            .finish_distributed_encoder_publication(batch, &mut self.resources.execution)
    }
    pub fn output(&self, batch: &RequestBatch) -> Result<DistributedTargetLogits<'_>> {
        self.state.ready(batch.cache()?.identity())?;
        self.resources.head.output()
    }
    /// Download selected head rows for sampling or constrained token selection.
    /// Rows index the compact head output, not the original request batch.
    pub async fn download_logits(&mut self, batch: &RequestBatch, rows: &[usize]) -> Result<Vec<u8>> {
        self.state.ready(batch.cache()?.identity())?;
        self.resources.head.download_rows(rows).await
    }
    pub fn greedy_output(&mut self, batch: &RequestBatch) -> Result<Vec<(u32, f32)>> {
        self.state.ready(batch.cache()?.identity())?;
        Ok(self.resources.head.greedy_output()?.to_vec())
    }
    pub fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>> {
        self.state.ready(batch.cache()?.identity())?;
        self.resources.taps.output(batch.cache()?)
    }
    pub fn enqueue_cache_commit(
        &mut self,
        requests: &Requests<'a>,
        batch: &RequestBatch,
        accepted: &[u32],
    ) -> Result<()> {
        let id = batch.cache()?.identity();
        ensure!(
            self.state == State::Ready(id) || self.state == State::Encoded(id),
            "distributed pass incomplete"
        );
        requests.validate_acceptance(batch, accepted)?;
        unsafe {
            self.resources.execution
                .enqueue_cache_commit(requests.cache(), batch.cache()?, accepted)
        }
    }
    pub fn poll_cache_commit(&self) -> Result<bool> {
        self.resources.execution.poll_cache_commit()
    }
    pub fn commit(
        &mut self,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: &[u32],
    ) -> Result<()> {
        let id = batch.cache()?.identity();
        ensure!(
            self.state == State::Ready(id) || self.state == State::Encoded(id),
            "distributed pass incomplete"
        );
        self.state = State::Running;
        self.resources.taps.reset();
        requests.commit_distributed(batch, &mut self.resources.execution, accepted)?;
        self.state = State::Idle;
        Ok(())
    }
    pub fn abort_cache_commit(&mut self, requests: &mut Requests<'a>) -> Result<()> {
        requests.abort_distributed_cache_commit(&mut self.resources.execution)
    }
    /// Execution and output consumers must be dropped before discarding. Any
    /// queued cache commit must first be completed or explicitly aborted.
    pub fn discard(&mut self, batch: &mut RequestBatch) -> Result<()> {
        batch.cancel();
        self.resources.taps.reset();
        self.state = State::Running;
        for lane in &mut self.resources.lanes {
            let device = lane.device;
            device.run(|| {
                lane.invalidate();
                Ok(())
            })?;
        }
        for index in self.resources.indices.iter_mut().flatten() {
            let device = index.device;
            device.run(|| index.restart())?;
        }
        self.resources.execution.restart_for(CacheStage::Full);
        self.state = State::Idle;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::deepseek_v41::v41_backbone_cache::BackboneCache;
    use crate::families::deepseek_v41::v41_backbone_execution::CacheProducerWeights;
    use crate::families::deepseek_v41::v41_backbone_lane::BackboneLaneWeights;
    use crate::families::deepseek_v41::v41_backbone_shared::tp2::Weights as SharedWeights;
    use crate::families::deepseek_v41::v41_engram::placement::PlacedEngramWeights;
    use crate::families::deepseek_v41::v41_experts::tp2_ffn;
    use crate::families::deepseek_v41::v41_index_lane::IndexLaneWeights;
    use crate::shared::memory::device::Device;
    use crate::families::deepseek_v41::v41_target_head::TargetHeadWeights;
    use crate::families::deepseek_v41::v41_tensors::{NativeRtxTensors, VocabularyHead};
    use cuteafd_ffi::NativeLibrary;
    use cuteafd_transport::{ExpertV2SourceKind, TcpTransportConfig, expert::SparkExperts};
    use crate::families::deepseek_v41::v41_native_serve::prefill_target::PrefillTarget;
    use std::rc::Rc;
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT, CUTEAFD_DUAL_PEERS, two GPUs and live Sparks"]
    fn distributed_target_prefill_decode_commit_smoke() -> Result<()> {
        distributed_prefill_fixture(false)
    }

    #[test]
    #[ignore = "requires native library, snapshot, two GPUs and live TP4 Sparks"]
    fn packed_prefill_matches_singleton_logits() -> Result<()> {
        distributed_prefill_fixture(true)
    }

    fn packed_reference_windows() -> Result<Vec<cuteafd_bench::reference::Window>> {
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
        let manifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let reference = cuteafd_bench::fidelity_dataset::load(&root, &manifest)?;
        ensure!(reference.checkpoint == "deepseek-ai/DeepSeek-V4.1-Flash-1017" && reference.vocab == 129280,
            "packed fixture requires the published V4.1 reference");
        ["c00", "c04", "c08"].iter().map(|id| {
            let mut window = reference.windows.iter().find(|w| w.id == *id)
                .with_context(|| format!("missing short reference window {id}"))?.clone();
            ensure!(window.media.is_empty() && window.positions.len() >= 20
                && window.positions[19].pos <= 1024, "reference window does not fit packed fixture");
            window.positions.truncate(20);
            Ok(window)
        }).collect()
    }

    fn distributed_prefill_fixture(packed_check: bool) -> Result<()> {
        use crate::families::deepseek_v41::v41_native_serve::prefill_target::PrefillTarget;
        use crate::families::deepseek_v41::v41_experts::dspark::{DsparkChain, DsparkWeights};
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let snapshot = std::env::var("CUTEAFD_SNAPSHOT")?;
        let packed_windows = if packed_check { Some(packed_reference_windows()?) } else { None };
        let unit_rows = if packed_check { 2048 } else { 16 };
        let reserve_rows = 2 * unit_rows;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&snapshot),
        )?;
        lib.cuda_set_device(0)?;
        let devices = [
            Device {
                library: &lib,
                id: 0,
            },
            Device {
                library: &lib,
                id: 1,
            },
        ];
        // Exercise the deployment's 3/1 source split by default. Boundary 14
        // remains available to cover a source group on the second encoder GPU.
        let boundary = std::env::var("CUTEAFD_ATTENTION_BOUNDARY").ok()
            .map(|value| value.parse::<usize>()).transpose()?.unwrap_or(20);
        ensure!([14, 20].contains(&boundary), "unsupported fixture attention boundary");
        let map = CachePlacement::new(std::array::from_fn(|l| usize::from(l >= boundary)))?;
        eprintln!("loading placed backbone and auxiliary weights");
        let weights = BackboneLaneWeights::load_distributed(
            &lib,
            &catalog,
            map,
            BackboneLaneWeights::distributed_device_bytes(&lib, &catalog, map)?,
            16 << 20,
        )?;
        let producers = CacheProducerWeights::load_distributed(
            &lib,
            &catalog,
            map,
            CacheProducerWeights::distributed_device_bytes(&lib, &catalog, map)?,
            16 << 20,
        )?;
        let iw = IndexLaneWeights::load_distributed(
            &lib,
            &catalog,
            map,
            IndexLaneWeights::distributed_device_bytes(&lib, &catalog, map)?,
            16 << 20,
        )?;
        let ew = PlacedEngramWeights::load(
            &lib,
            &catalog,
            map,
            PlacedEngramWeights::device_bytes(&lib, &catalog, map)?,
            16 << 20,
        )?;
        let names = ["embed.weight".to_string()];
        let table = devices[0].own(|| {
            NativeRtxTensors::load(
                &lib,
                &catalog,
                &names,
                NativeRtxTensors::plan(&catalog, &names)?,
                16 << 20,
            )
        })?;
        let vocab = [
            devices[0].own(|| crate::families::deepseek_v41::v41_tensors::VocabularyShard::load(&lib, &catalog, 0..64640, 1 << 30, 16 << 20))?,
            devices[1].own(|| crate::families::deepseek_v41::v41_tensors::VocabularyShard::load(&lib, &catalog, 64640..129280, 1 << 30, 16 << 20))?,
        ];
        let hw = devices[1].own(|| {
            TargetHeadWeights::load(
                &lib,
                &catalog,
                TargetHeadWeights::device_bytes(&catalog)?,
                16 << 20,
            )
        })?;
        eprintln!("loading all 20 encoder expert layers as TP2");
        let routed = tp2_ffn::load_routed_weights(
            devices, &catalog, 20, unit_rows, [73_000_000_000; 2],
        )?;
        let shared: [Rc<Vec<SharedWeights<'_>>>; 2] = [0, 1]
            .map(|r| {
                (0..40)
                    .map(|l| {
                        SharedWeights::load(
                            devices[r],
                            &catalog,
                            l,
                            SharedWeights::load_peak_device_bytes(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(Rc::new)
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .ok()
            .expect("two ranks");
        let make_pass = || -> Result<DistributedTargetPass<'_, '_>> {
            let lanes = [
                BackboneLane::new_on_device(
                    &weights,
                    16,
                    BackboneLane::placed_workspace_bytes(&lib, 16)?,
                    0,
                )?,
                BackboneLane::new_on_device(
                    &weights,
                    16,
                    BackboneLane::placed_workspace_bytes(&lib, 16)?,
                    1,
                )?,
            ];
            let indices = [
                Some(IndexLane::new_on_device(
                    &iw,
                    16,
                    IndexLane::placed_workspace_bytes(&lib, map, 16, 0)?
                        .iter()
                        .sum(),
                    0,
                )?),
                Some(IndexLane::new_on_device(
                    &iw,
                    16,
                    IndexLane::placed_workspace_bytes(&lib, map, 16, 1)?
                        .iter()
                        .sum(),
                    1,
                )?),
            ];
            let mut pass = DistributedTargetPass::new(
                map,
                devices[0].own(|| {
                    TargetEmbeddingWave::new(
                        &lib,
                        &table,
                        16,
                        TargetEmbeddingWave::device_bytes(16)?,
                    )
                })?,
                lanes,
                indices,
                DistributedExecution::new(
                    &producers,
                    16,
                    crate::families::deepseek_v41::v41_backbone_execution::PlacedProducerWaves::device_bytes(
                        &lib, map, 16,
                    )?,
                )?,
                PlacedEngram::new(&ew, 16, PlacedEngram::device_bytes(&lib, map, 16)?)?,
                DistributedTargetHead::new(devices, &hw, [&vocab[0], &vocab[1]], 16,
                    DistributedTargetHead::device_bytes(16, 64640)?)?,
                devices[1]
                    .own(|| TargetTapWave::new(&lib, 16, TargetTapWave::device_bytes(16)?))?,
                Duration::from_secs(120),
            )?;
            let spec = cuteafd_loader::placement::families::deepseek_v41::execution_hops(2);
            let modes = (0..40).map(|layer| Ok(cuteafd_loader::placement::LayerMode::Whole {
                gpu: map.attention(layer)? as u8, ffn: cuteafd_loader::placement::FfnMode::Owner,
            })).collect::<Result<Vec<_>>>()?;
            let hops = cuteafd_loader::placement::plan_hops(&modes, &spec)?;
            for (gpu, transfer) in pass.resources.transfers.iter_mut().enumerate() {
                let incoming: Vec<_> = hops.iter().filter(|hop| usize::from(hop.to) == gpu).copied().collect();
                transfer.bind(&incoming, spec)?;
            }
            Ok(pass)
        };
        // Two prefill slots on one storage set per GPU (reserve = 2 * unit, unit conditional on reference fixture),
        // as serve assembles them (`shared::memory::row_window`).
        let slot_storage = || -> Result<_> {
            let both = [0, 1];
            Ok((
                [devices[0].own(|| crate::families::deepseek_v41::v41_backbone_lane::LaneStorage::new(&weights, reserve_rows, unit_rows, &both))?,
                 devices[1].own(|| crate::families::deepseek_v41::v41_backbone_lane::LaneStorage::new(&weights, reserve_rows, unit_rows, &both))?],
                [IndexLane::storage_on_device(&iw, reserve_rows, unit_rows, 0, &both)?, IndexLane::storage_on_device(&iw, reserve_rows, unit_rows, 1, &both)?],
                crate::families::deepseek_v41::v41_backbone_execution::PlacedProducerStorage::new(&lib, map, reserve_rows as usize, unit_rows as usize)?,
                devices[0].own(|| TargetEmbeddingWave::storage(&lib, reserve_rows as usize, unit_rows as usize))?,
                PlacedEngram::storage(&ew, reserve_rows as usize, unit_rows as usize)?,
                devices[1].own(|| TargetTapWave::storage(&lib, reserve_rows as usize, unit_rows as usize))?,
            ))
        };
        let (lane_storage, index_storage, producer_storage, embedding_storage, engram_storage, tap_storage) = slot_storage()?;
        let make_slot = || -> Result<DistributedTargetPass<'_, '_>> {
            let mut pass = DistributedTargetPass::new(
                map,
                devices[0].own(|| TargetEmbeddingWave::on_storage(&lib, &table, embedding_storage.get().clone()))?,
                [BackboneLane::on_storage_on_device(&weights, lane_storage[0].get().clone())?,
                 BackboneLane::on_storage_on_device(&weights, lane_storage[1].get().clone())?],
                [Some(IndexLane::on_storage_on_device(&iw, index_storage[0].get().clone())?),
                 Some(IndexLane::on_storage_on_device(&iw, index_storage[1].get().clone())?)],
                DistributedExecution::on_storage(&producers, &producer_storage)?,
                PlacedEngram::on_storage(&ew, &engram_storage)?,
                DistributedTargetHead::new(devices, &hw, [&vocab[0], &vocab[1]], 16,
                    DistributedTargetHead::device_bytes(16, 64640)?)?,
                devices[1].own(|| TargetTapWave::on_storage(tap_storage.get().clone()))?,
                Duration::from_secs(120),
            )?;
            let spec = cuteafd_loader::placement::families::deepseek_v41::execution_hops(2);
            let modes = (0..40).map(|layer| Ok(cuteafd_loader::placement::LayerMode::Whole {
                gpu: map.attention(layer)? as u8, ffn: cuteafd_loader::placement::FfnMode::Owner,
            })).collect::<Result<Vec<_>>>()?;
            let hops = cuteafd_loader::placement::plan_hops(&modes, &spec)?;
            for (gpu, transfer) in pass.resources.transfers.iter_mut().enumerate() {
                let incoming: Vec<_> = hops.iter().filter(|hop| usize::from(hop.to) == gpu).copied().collect();
                transfer.bind(&incoming, spec)?;
            }
            Ok(pass)
        };
        let mut pass = make_pass()?;
        let token_map = cuteafd_loader::EngramTokenMap::from_file(
            &std::path::Path::new(&snapshot).join("tokenizer.json"),
        )?;
        let pipeline =
            unsafe { cuteafd_loader::EngramPipeline::new(&lib, &catalog, token_map, cuteafd_loader::EngramStagingShape::new(unit_rows as usize, 2)?, 0)? };
        let pages = if packed_check { BackboneCache::pages_for_context(2, 1024)? } else { [4, 4, 4, 8] };
        let mut requests = Requests::new_distributed(
            &lib,
            pipeline,
            2,
            pages,
            map,
            BackboneCache::distributed_device_bytes(map, 2, pages)?,
        )?;
        let peers: [std::net::SocketAddr; 4] = std::env::var("CUTEAFD_DUAL_PEERS")?
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?
            .try_into()
            .ok()
            .context("four Spark endpoints required")?;
        // As serve: the workers' selection/admission handshake (layers 20.. on
        // the Sparks, two prefill units) before any link. Endpoints admit the
        // decode owners' verify width (64 rows); the fixture's 16-row links fit.
        crate::families::deepseek_v41::v41_native_serve::admission::spark_admission(std::path::Path::new(&snapshot),
            &catalog, 20, peers.len(), unit_rows.max(cuteafd_core::V41_VERIFY_ROWS as u32), 2,
            // The workers run TP4 x EP1 (golden-run --spark-tp 4 --spark-ep 1), as serve's admission names it.
            Some(cuteafd_transport::expert::SparkTopology::NATIVE_TP4_EP1),
            crate::families::deepseek_v41::v41_native_serve::DECODE_LANE_ROWS)?.start(&peers)?.wait()?;
        let roce = SparkExperts::new_world(cuteafd_transport::expert::SparkWorld::owned(cuteafd_transport::expert::SparkTopology::NATIVE_TP4_EP1)?, &peers, 16, crate::families::deepseek_v41::v41_native_serve::transport_config(crate::families::deepseek_v41::v41_native_serve::protocol_v2_timing()), 0)?;
        let mut transport =
            devices[1].own(|| NativeTp4Wave::new(&lib, roce, NativeTp4Wave::device_bytes(16)?))?;
        transport.install_tp2(tp2_ffn::Wave::new(routed.clone(), shared.clone(), 20, 16, [None, None])?)?;
        // As serve: each link's exchange starts at the first remote layer.
        let nvfp4 = catalog.nvfp4().is_some();
        devices[1].run(|| transport.get_mut().bootstrap(20, nvfp4))?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let draft_weights = devices[1].own(|| DsparkWeights::load(&lib, &catalog, 80, 1, 32usize << 30, 16 << 20))?;
        let mut draft = crate::families::deepseek_v41::v41_native_serve::speculative::DraftRuntime::with_distributed_requests(
            devices, &draft_weights, &table, [&vocab[0], &vocab[1]], 16, 2)?;
        draft.admit(91001)?;
        let lease = requests.admit(0, 91001)?;
        let mut tokens = vec![100u32, 200, 300, 400];
        for (step, kind) in [
            ExpertV2SourceKind::Prefill,
            ExpertV2SourceKind::Decode,
            ExpertV2SourceKind::Decode,
            ExpertV2SourceKind::Decode,
        ]
        .into_iter()
        .enumerate()
        {
            pass.set_route_capture(true)?;
            let route_capacities: Vec<_> = pass.resources.route_capture.iter().map(Vec::capacity).collect();
            eprintln!(
                "executing distributed full pass step={step} rows={}",
                tokens.len()
            );
            let mut batch = requests.prepare(&[crate::families::deepseek_v41::v41_requests::RequestTokens {
                lease,
                tokens: &tokens,
                image_mask: None,
                kind,
            }])?;
            let next = runtime.block_on(async {
                let requests = std::cell::RefCell::new(&mut requests);
                if step % 2 == 0 {
                    Ok::<_, anyhow::Error>(Some(unsafe {
                        VerificationTarget::execute_shared_greedy(&mut pass, &requests, &mut batch,
                            &mut transport, 0, &[tokens.len() - 1]).await?
                    }))
                } else {
                    unsafe {
                        pass.prefill_logits(&lib, &mut requests.borrow_mut(), &mut batch,
                            &mut transport, &[tokens.len() - 1], None).await?;
                    }
                    assert!(pass.greedy_output(&batch).is_err());
                    Ok(None)
                }
            })?;
            ensure!(pass.captured_routes().len() == 40, "distributed route history has missing layers");
            let counts: Vec<usize> = pass.captured_routes().iter().map(Vec::len).collect();
            for (layer, routes) in pass.captured_routes().iter().enumerate() {
                ensure!(routes.len() == tokens.len(),
                    "distributed route row count differs at layer {layer}: {} rows, expected {} (step {step}, per layer {counts:?})",
                    routes.len(), tokens.len());
                ensure!(routes == &pass.resources.lanes[pass.resources.map.attention(layer)?].captured_routes()[layer],
                    "distributed route history differs from its GPU owner");
                ensure!(routes.iter().all(|row| row.iter().enumerate().all(|(i, id)|
                    *id < 384 && !row[..i].contains(id))), "invalid captured top-six routes at layer {layer}");
                ensure!(routes.capacity() == route_capacities[layer], "route history allocated during execution");
            }
            assert!(pass.set_route_capture(true).is_err());
            pass.set_route_capture(false)?;
            let bytes = runtime.block_on(VerificationTarget::download_logits(&mut pass, &batch, &[0]))?;
            ensure!(bytes.len() == 129280 * 4, "distributed logit download extent differs");
            let scores: Vec<f32> = bytes.chunks_exact(4)
                .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap())).collect();
            ensure!(scores.iter().all(|score| score.is_finite()), "non-finite downloaded logits");
            let best = scores.iter().enumerate().fold(0, |best, (index, score)|
                if *score > scores[best] { index } else { best });
            let next = next.unwrap_or_else(|| vec![(best as u32, scores[best])]);
            assert_eq!((best as u32, scores[best]), next[0]);
            assert!(runtime.block_on(pass.download_logits(&batch, &[1])).is_err());
            assert_eq!(lib.cuda_get_device()?, 0);
            ensure!(
                next.len() == 1 && next[0].0 < 129280 && next[0].1.is_finite(),
                "invalid full-pass greedy output"
            );
            ensure!(
                crate::families::deepseek_v41::v41_target_pass::TargetCache::taps(&pass, &batch)?.rows().len() == tokens.len(),
                "distributed taps lost rows"
            );
            if step == 2 {
                VerificationTarget::discard(&mut pass, &mut batch)?;
                assert!(requests.validate(&batch).is_err());
                assert_eq!(requests.cache().committed_end(lease)?, 5);
                assert_eq!(lib.cuda_get_device()?, 0);
                eprintln!("PASS discarded full distributed proposal without advancing history");
                continue;
            }
            runtime.block_on(pass.commit_prefill(&mut requests, &mut batch,
                Some(draft.get_mut()), tokens.len() as u32))?;
            draft.validate_position(91001, if step == 3 { 6 } else { 4 + step as u64 })?;
            assert_eq!(
                requests.cache().committed_end(lease)?,
                if step == 3 { 6 } else { 4 + step as u64 }
            );
            eprintln!(
                "PASS full distributed step={step} token={} score={}",
                next[0].0, next[0].1
            );
            tokens = vec![next[0].0];
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        requests.release(lease)?;
        draft.release(91001)?;
        drop(draft);
        eprintln!("PASS distributed prefill commit: target and all three dSpark cache frontiers agree");
        let prompt = [100u32, 200, 300, 400, 500, 600, 700];
        let baseline = requests.admit(1, 92000)?;
        let mut baseline_batch = requests.prepare(&[crate::families::deepseek_v41::v41_requests::RequestTokens {
            lease: baseline,
            tokens: &prompt,
            image_mask: None,
            kind: ExpertV2SourceKind::Prefill,
        }])?;
        runtime.block_on(unsafe {
            pass.execute(
                &std::cell::RefCell::new(&mut requests),
                &mut baseline_batch,
                &mut transport,
                0,
                &[6],
                None,
                None,
                true,
            )
        })?;
        let expected = pass.resources.head.greedy_output()?.to_vec();
        pass.discard(&mut baseline_batch)?;
        requests.release(baseline)?;
        let lease = requests.admit(0, 92001)?;
        requests.begin_encoder(lease, prompt.len() as u64)?;
        let mut suffix = pass.new_suffix(&lib, prompt.len() as u64)?;
        for chunk in [&prompt[..3], &prompt[3..]] {
            let mut batch = requests.reserve_encoder(&[crate::families::deepseek_v41::v41_requests::RequestTokens {
                lease,
                tokens: chunk,
                image_mask: None,
                kind: ExpertV2SourceKind::Prefill,
            }])?;
            runtime.block_on(unsafe { pass.encoder_part(&mut requests, &mut batch, &mut transport, &mut suffix) })?;
            runtime.block_on(pass.commit_prefill::<DsparkChain<'_, '_>>(&mut requests, &mut batch, None, chunk.len() as u32))?;
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        assert_eq!(requests.cache().committed_end(lease)?, 7);
        assert_eq!(requests.begin_decoder_replay(lease)?, 0);
        let mut batch = requests.prepare_replay(&[crate::families::deepseek_v41::v41_backbone_cache::CacheWork {
            lease,
            tokens: 7,
            kind: ExpertV2SourceKind::Prefill,
        }])?;
        let bytes = runtime.block_on(unsafe {
            pass.prefill_logits(&lib, &mut requests, &mut batch, &mut transport, &[6], Some(&suffix))
        })?;
        let scores: Vec<_> = bytes.chunks_exact(4).map(|v| f32::from_ne_bytes(v.try_into().unwrap())).collect();
        ensure!(scores.len() == 129280 && scores.iter().all(|v| v.is_finite()), "invalid prefill replay logits");
        let best = (0..scores.len()).fold(0, |best, i| if scores[i] > scores[best] { i } else { best });
        let next = vec![(best as u32, scores[best])];
        assert!(next[0].0 < 129280 && next[0].1.is_finite());
        assert_eq!(
            next[0].0, expected[0].0,
            "reserved encoder/replay greedy token differs from full prefill"
        );
        ensure!(
            (next[0].1 - expected[0].1).abs() < 0.25,
            "reserved encoder/replay greedy score differs: {} vs {}",
            next[0].1,
            expected[0].1
        );

        pass.enqueue_cache_commit(&requests, &batch, &[7])?;
        runtime.block_on(async {
            while !pass.poll_cache_commit()? {
                tokio::task::yield_now().await;
            }
            Ok::<_, anyhow::Error>(())
        })?;
        pass.commit(&mut requests, &mut batch, &[7])?;
        requests.release(lease)?;
        eprintln!(
            "PASS distributed reserved encoder 3+4 tokens, source publication and decoder replay token={}",
            next[0].0
        );
        let mut other = make_pass()?;
        let mut other_transport = devices[1].own(|| {
            NativeTp4Wave::new(
                &lib,
                // As serve: the second decode lane's own endpoint (lane 1).
                SparkExperts::new_world(cuteafd_transport::expert::SparkWorld::owned(cuteafd_transport::expert::SparkTopology::NATIVE_TP4_EP1)?, &peers, 16, crate::families::deepseek_v41::v41_native_serve::transport_config(crate::families::deepseek_v41::v41_native_serve::protocol_v2_timing()), 1)?,
                NativeTp4Wave::device_bytes(16)?,
            )
        })?;
        other_transport.install_tp2(tp2_ffn::Wave::new(routed.clone(), shared.clone(), 20, 16, [None, None])?)?;
        devices[1].run(|| other_transport.get_mut().bootstrap(20, nvfp4))?;
        // The prefill slots: two passes on shared storage; slot 0's FFN owner
        // holds the one prefill link (execution lane 2), two return slots of
        // 16 rows; one TP2 FFN arena for both (reserve 32, unit 16).
        let ffn_storage = tp2_ffn::Storage::new(routed, shared, 20, reserve_rows, unit_rows, [None, None])?;
        let link = devices[1].own(|| crate::families::deepseek_v41::v41_experts::coordinator::V41Link::prefill(&lib,
            SparkExperts::new_world(cuteafd_transport::expert::SparkWorld::owned(cuteafd_transport::expert::SparkTopology::NATIVE_TP4_EP1)?, &peers, unit_rows,
                crate::families::deepseek_v41::v41_native_serve::transport_config(crate::families::deepseek_v41::v41_native_serve::protocol_v2_timing()), 2)?, 2, unit_rows))?;
        let mut slot_transports = [devices[1].own(|| NativeTp4Wave::with_link(&lib, link.into_inner()))?,
            devices[1].own(|| NativeTp4Wave::unlinked(&lib, unit_rows))?];
        for transport in &mut slot_transports { transport.install_tp2(tp2_ffn::Wave::on_storage(ffn_storage.clone()))?; }
        devices[1].run(|| slot_transports[0].get_mut().bootstrap(20, nvfp4))?;
        let [slot0_transport, slot1_transport] = slot_transports;
        let mut prefill_slots = crate::families::deepseek_v41::v41_native_serve::prefill_target::PrefillSlots {
            packed_suffix: pass.new_packed_storage(&lib, unit_rows as usize)?,
            prompt_suffixes: vec![pass.new_prompt_storage(&lib)?],
            slots: vec![(make_slot()?, slot0_transport), (make_slot()?, slot1_transport)],
        };
        if let Some(windows) = packed_windows {
            use clap::Parser;
            let rows = unit_rows.to_string();
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-native", "--snapshot", &snapshot,
                "--native-lib", &std::env::var("CUTEAFD_NATIVE_LIB")?, "--prefill-batch-tokens", &rows,
                "--prefill-waves", "2", "--prefill-packing", "unrestricted"])?;
            let crate::cli::Commands::ServeNative(args) = cli.command else { unreachable!() };
            let mut draft = crate::families::deepseek_v41::v41_native_serve::speculative::DraftRuntime::with_distributed_requests(
                devices, &draft_weights, &table, [&vocab[0], &vocab[1]], unit_rows, 2)?;
            let samples: Vec<_> = (0..20).flat_map(|sample| windows.iter().map(move |window|
                (window, window.positions[sample].pos))).collect();
            let mut scored = std::collections::BTreeMap::<_, (Vec<_>, Vec<_>)>::new();
            for pack in samples.chunks(2) {
                let rows = crate::families::deepseek_v41::v41_native_serve::scheduler::exercise_packed_prefill(
                    &lib, &runtime, &mut requests, &mut prefill_slots, &args, pack, Some(draft.get_mut()))?;
                for ((window, _), (packed, singleton)) in pack.iter().zip(rows) {
                    let entry = scored.entry(window.id.clone()).or_default();
                    entry.0.push(packed);
                    entry.1.push(singleton);
                }
            }
            use cuteafd_bench::reference::Fidelity;
            let mut packed_records = Vec::new();
            let mut singleton_records = Vec::new();
            for window in windows {
                let (packed, singleton) = scored.get(&window.id).context("missing packed reference rows")?;
                let (packed, singleton) = (window.score(packed), window.score(singleton));
                ensure!(packed.positions == 20 && singleton.positions == 20
                    && packed.missing == 0 && singleton.missing == 0
                    && packed.non_finite == 0 && singleton.non_finite == 0,
                    "packed/singleton reference rows missing or nonfinite");
                // Every endpoint is a member; a pooled tie cannot hide one degraded member.
                for (p, s) in packed.records.iter().zip(&singleton.records) {
                    ensure!(p.position == s.position && p.window == s.window,
                        "packed/singleton reference positions differ");
                    ensure!(p.kl - s.kl <= 0.005 && i32::from(p.agree) - i32::from(s.agree) >= 0,
                        "packed member {}:{} regressed: packed {p:?}, singleton {s:?}", p.window, p.position);
                }
                eprintln!("V4.1 packed/reference window={} positions={} packed KL={:.8} top1={:.6}; singleton KL={:.8} top1={:.6}",
                    window.id, packed.positions, packed.kl, packed.top1, singleton.kl, singleton.top1);
                packed_records.extend(packed.records);
                singleton_records.extend(singleton.records);
            }
            let packed = Fidelity::from_records(packed_records);
            let singleton = Fidelity::from_records(singleton_records);
            eprintln!("V4.1 packed/reference pooled positions={} packed KL={:.8} top1={:.6}; singleton KL={:.8} top1={:.6}; delta KL={:.8} top1={:.6}",
                packed.positions, packed.kl, packed.top1, singleton.kl, singleton.top1,
                packed.kl - singleton.kl, packed.top1 - singleton.top1);
            ensure!(packed.positions == 60 && packed.kl - singleton.kl <= 0.005
                && packed.top1 - singleton.top1 >= -0.01, "packed pooled fidelity regressed");
            return Ok(());
        }
        for chunk_rows in [3, 16] {
            let reference_lease = requests.admit(0, 92900 + chunk_rows as u64)?;
            requests.begin_encoder(reference_lease, prompt.len() as u64)?;
            let mut reference_suffix = pass.new_suffix(&lib, prompt.len() as u64)?;
            for chunk in prompt.chunks(chunk_rows) {
                let mut batch = requests.reserve_encoder(&[crate::families::deepseek_v41::v41_requests::RequestTokens {
                    lease: reference_lease, tokens: chunk, image_mask: None, kind: ExpertV2SourceKind::Prefill,
                }])?;
                runtime.block_on(unsafe { pass.encoder_part(&mut requests, &mut batch, &mut transport, &mut reference_suffix) })?;
                runtime.block_on(pass.commit_prefill::<DsparkChain<'_, '_>>(&mut requests, &mut batch, None, chunk.len() as u32))?;
            }
            let start = requests.begin_decoder_replay(reference_lease)?;
            let replay_rows = prompt.len() as u32 - start as u32;
            let mut batch = requests.prepare_replay(&[crate::families::deepseek_v41::v41_backbone_cache::CacheWork {
                lease: reference_lease, tokens: replay_rows, kind: ExpertV2SourceKind::Prefill,
            }])?;
            let reference_bytes = runtime.block_on(unsafe { pass.prefill_logits(&lib, &mut requests, &mut batch,
                &mut transport, &[replay_rows as usize - 1], Some(&reference_suffix)) })?;
            let reference_scores: Vec<_> = reference_bytes.chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap())).collect();
            let reference_anchor = (0..reference_scores.len()).fold(0, |best, i|
                if reference_scores[i] > reference_scores[best] { i } else { best }) as u32;
            runtime.block_on(pass.commit_prefill::<DsparkChain<'_, '_>>(&mut requests, &mut batch, None, replay_rows))?;
            requests.release(reference_lease)?;
            eprintln!("serving prefill reference chunk_rows={chunk_rows} sequential_anchor={reference_anchor} one_shot_anchor={}", expected[0].0);
            let id = 93000 + chunk_rows as u64;
            let lease = requests.admit(0, id)?;
            let mut draft = crate::families::deepseek_v41::v41_native_serve::speculative::DraftRuntime::with_distributed_requests(
                devices, &draft_weights, &table, [&vocab[0], &vocab[1]], 16, 2)?;
            draft.admit(id)?;
            let anchor = crate::families::deepseek_v41::v41_native_serve::prefill_target::exercise_prefill(&lib, &runtime,
                &mut requests, &mut prefill_slots, lease, &prompt, chunk_rows, 2, Some(draft.get_mut()))?;
            assert_eq!(anchor, reference_anchor, "serving prefill anchor differs from matching sequential chunks");
            draft.validate_position(id, 7)?;
            let proposed = loop {
                if let Some((rows, _)) = draft.poll_propose(0, &[(id, anchor, 7, 12)])? { break rows; }
                std::thread::yield_now();
            };
            ensure!(proposed.len() == 1 && proposed[0].len() == 6 && proposed[0][0] == anchor,
                "serving prefill to draft handoff differs");
            let continued: Vec<_> = prompt.into_iter().chain([800, 900]).collect();
            let next = crate::families::deepseek_v41::v41_native_serve::prefill_target::exercise_prefill(&lib, &runtime,
                &mut requests, &mut prefill_slots, lease, &continued, chunk_rows, 2, Some(draft.get_mut()))?;
            ensure!(next < 129280, "invalid continuation anchor");
            assert_eq!(requests.cache().committed_end(lease)?, 9);
            draft.validate_position(id, 9)?;
            crate::families::deepseek_v41::v41_native_serve::prefill_target::exercise_distributed_decode(&lib, &runtime,
                std::path::Path::new(&snapshot), &mut pass, &mut other, &mut requests, [&mut transport, &mut other_transport],
                lease, id, &continued, next, draft.get_mut())?;
            assert_eq!(lib.cuda_get_device()?, 0);
            eprintln!("PASS serving distributed decode: streamed finish, target/draft retirement, transport reset");
            eprintln!("PASS serving distributed prefill chunk_rows={chunk_rows}: encoder/replay, draft handoff, cached continuation, target/draft commits");
        }
        {
            // C1/C4/D1: a client disconnect mid-unit (queued chained work, the
            // link lent out, windows selected on passes and their TP2 FFN).
            use crate::shared::memory::row_window::RowWindow;

            let lease = requests.admit(0, 93500)?;
            let error = crate::families::deepseek_v41::v41_native_serve::prefill_target::exercise_cancelled_prefill::<_, DsparkChain<'_, '_>>(
                &lib, &runtime, &mut requests, &mut prefill_slots, lease, &prompt, 3, 2);
            ensure!(format!("{error:#}").contains("client disconnected"), "cancel path failed otherwise: {error:#}");
            if requests.cache().request_id(lease).is_ok() { requests.release(lease)?; }
            let windows: Vec<_> = prefill_slots.slots.iter().map(|(pass, transport)| (
                pass.resources.lanes.iter().map(|lane| lane.window().clone()).collect::<Vec<_>>(),
                transport.tp2_window().cloned())).collect();
            for (lanes, tp2) in windows {
                ensure!(lanes.iter().all(|window| *window == RowWindow::NONE), "a cancelled unit left a lane windowed");
                ensure!(tp2.is_none_or(|window| window == RowWindow::NONE), "a cancelled unit left its TP2 FFN windowed");
            }
            ensure!(prefill_slots.slots[0].1.get_mut().take_link().map(|link| prefill_slots.slots[0].1.get_mut().restore_link(link)).is_ok(),
                "a cancelled units call left slot 0 without its link");
            // Slot 0 serves the next prompt (units, replay) after the cancel.
            let lease = requests.admit(0, 93501)?;
            let anchor = crate::families::deepseek_v41::v41_native_serve::prefill_target::exercise_prefill::<_, DsparkChain<'_, '_>>(
                &lib, &runtime, &mut requests, &mut prefill_slots, lease, &prompt, 3, 2, None)?;
            ensure!(anchor < 129280, "post-cancel prefill anchor invalid");
            requests.release(lease)?;
            eprintln!("PASS cancelled units: windows (passes + TP2 FFN) at rows from 0, link restored, slot 0 serves again");
            // C1: every unit's TP2 FFN works in its slot's window, as its pass
            // does (negative control: unwindowed TP2 sees rows from 0).
            use self::encoder_units::{SKIP_TP2_WINDOW, UNIT_WINDOWS};
            for skip in [false, true] {
                SKIP_TP2_WINDOW.with(|flag| flag.set(skip));
                UNIT_WINDOWS.with(|seen| seen.borrow_mut().clear());
                let lease = requests.admit(0, 93510 + u64::from(skip))?;
                let ran = crate::families::deepseek_v41::v41_native_serve::prefill_target::exercise_prefill::<_, DsparkChain<'_, '_>>(
                    &lib, &runtime, &mut requests, &mut prefill_slots, lease, &prompt, 3, 2, None);
                SKIP_TP2_WINDOW.with(|flag| flag.set(false));
                ran?;
                requests.release(lease)?;
                let seen = UNIT_WINDOWS.with(|seen| std::mem::take(&mut *seen.borrow_mut()));
                ensure!(seen.len() >= 2 && seen.iter().any(|(pass, _)| pass.start() > 0), "units did not run in two windows");
                let matched = seen.iter().all(|(pass, tp2)| tp2.as_ref() == Some(pass));
                eprintln!("TP2 FFN unit windows skip={skip}: {} units, matched={matched}", seen.len());
                ensure!(matched != skip, "TP2 FFN window {} its pass's (skip={skip})", if matched { "matches" } else { "differs from" });
            }
            eprintln!("PASS TP2 FFN units windowed with their passes; the unwindowed control differs");
        }
        drop(draft_weights);
        if std::env::var_os("CUTEAFD_INDEPENDENT_ENCODER_CHECK").is_some() {
            let counts = [3usize, 7];
            let mut reference: [Option<Vec<u8>>; 2] = [None, None];
            for concurrent in [false, true, true, true, true] {
                let progress = std::rc::Rc::new(std::cell::Cell::new(0));
                pass.resources.observed_layer = Some(progress.clone());
                let leases = [requests.admit(0, 94000)?, requests.admit(1, 94001)?];
                for (lease, count) in leases.into_iter().zip(counts) { requests.begin_encoder(lease, count as u64)?; }
                let mut batches = leases.into_iter().zip(counts).map(|(lease, count)| requests.reserve_encoder(&[
                    crate::families::deepseek_v41::v41_requests::RequestTokens { lease, tokens: &prompt[..count],
                        image_mask: None, kind: ExpertV2SourceKind::Prefill }
                ])).collect::<Result<Vec<_>>>()?;
                let mut suffixes = counts.into_iter().map(|count| devices[map.attention(19)?].own(||
                    EncoderSuffix::new(&lib, count as u64, EncoderSuffix::device_bytes(count as u64)?)))
                    .collect::<Result<Vec<_>>>()?;
                {
                    let bank = std::cell::RefCell::new(&mut requests);
                    let (b0, b1) = batches.split_at_mut(1);
                    let (s0, s1) = suffixes.split_at_mut(1);
                    runtime.block_on(async {
                        if concurrent {
                            tokio::try_join!(
                                unsafe { pass.execute(&bank, &mut b0[0], &mut transport, 0,
                                    &[], Some(&mut s0[0]), None, false) },
                                async {
                                    while progress.get() < 5 { tokio::task::yield_now().await; }
                                    unsafe { other.execute(&bank, &mut b1[0], &mut other_transport, 0,
                                        &[], Some(&mut s1[0]), None, false).await }
                                },
                            )?;
                        } else {
                            unsafe { pass.execute(&bank, &mut b0[0], &mut transport, 0,
                                &[], Some(&mut s0[0]), None, false).await?; }
                            unsafe { other.execute(&bank, &mut b1[0], &mut other_transport, 0,
                                &[], Some(&mut s1[0]), None, false).await?; }
                        }
                        Ok::<_, anyhow::Error>(())
                    })?;
                }
                for (index, lane) in [&mut pass, &mut other].into_iter().enumerate() {
                    let output = suffixes[index].output()?;
                    let mut bytes = vec![0; output.residual.bytes];
                    devices[map.attention(19)?].run(|| lib.copy_d2h(&mut bytes, output.residual))?;
                    if let Some(expected) = &reference[index] {
                        ensure!(&bytes == expected,
                            "independent encoder residual differs: concurrent={concurrent}, lane={index}");
                    } else { reference[index] = Some(bytes); }
                    lane.enqueue_cache_commit(&requests, &batches[index], &[counts[index] as u32])?;
                    runtime.block_on(async {
                        while !lane.poll_cache_commit()? { tokio::task::yield_now().await; }
                        Ok::<_, anyhow::Error>(())
                    })?;
                    lane.commit(&mut requests, &mut batches[index], &[counts[index] as u32])?;
                    requests.release(leases[index])?;
                }
            }
            eprintln!("PASS independent encoder cache leases: sequential and concurrent residuals exact");
            pass.resources.observed_layer = None;
        }
        for (case, chunks) in [
            vec![&prompt[..3], &prompt[3..]],
            vec![&prompt[..1], &prompt[1..3], &prompt[3..4], &prompt[4..]],
        ]
        .into_iter()
        .cycle()
        .take(std::env::var("CUTEAFD_STREAM_CASES").ok().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(2))
        .enumerate()
        {
            let mut expected_case = None;
            let mut expected_trace = Vec::new();
            for interleaved in [false, true] {
                let tracing = std::env::var_os("CUTEAFD_TRACE_ENCODER").is_some();
                pass.resources.trace = tracing;
                other.resources.trace = tracing;
                let lease = requests.admit(0, 93000 + case as u64)?;
                requests.begin_encoder(lease, 7)?;
                let mut suffix = devices[map.attention(19)?]
                    .own(|| EncoderSuffix::new(&lib, 7, EncoderSuffix::device_bytes(7)?))?;
                if interleaved {
                    let mut link = DistributedTargetPass::take_link(&mut prefill_slots.slots[0].1)?;
                    let [(slot0, transport0), (slot1, transport1)] = &mut prefill_slots.slots[..] else { unreachable!() };
                    let result = runtime.block_on(unsafe {
                        DistributedTargetPass::execute_encoder_units(vec![(slot0, transport0), (slot1, transport1)], &mut link, &mut requests, lease, &chunks, &mut suffix, &[], &|| true, &|| Ok(()))
                    });
                    DistributedTargetPass::restore_link(&mut prefill_slots.slots[0].1, link)?;
                    result?;
                } else {
                    for chunk in &chunks {
                        let mut batch =
                            requests.reserve_encoder(&[crate::families::deepseek_v41::v41_requests::RequestTokens {
                                lease,
                                tokens: chunk,
                                image_mask: None,
                                kind: ExpertV2SourceKind::Prefill,
                            }])?;
                        runtime.block_on(unsafe {
                            pass.execute(
                                &std::cell::RefCell::new(&mut requests),
                                &mut batch,
                                &mut transport,
                                0,
                                &[],
                                Some(&mut suffix),
                                None,
                                false,
                            )
                        })?;
                        pass.enqueue_cache_commit(&requests, &batch, &[chunk.len() as u32])?;
                        assert!(pass.poll_cache_commit()?);
                        pass.commit(&mut requests, &mut batch, &[chunk.len() as u32])?;
                    }
                }
                pass.resources.trace = false;
                other.resources.trace = false;
                let mut trace = Vec::new();
                if tracing {
                    for lane in [&mut pass, &mut other] {
                        for gpu in 0..2 {
                            devices[gpu].run(|| unsafe {
                                lib.cuda_stream_synchronize(lane.resources.lanes[gpu].trace_stream())
                            })?;
                        }
                        for (position, layer, bytes) in lane.resources.trace_records.drain(..) {
                            let gpu = map.attention(layer % 40)?;
                            let (buffer, offset) = if layer >= 80 {
                                (lane.resources.trace_queries[gpu].buffer, ((if layer >= 160 { layer - 120 } else if layer >= 120 { layer - 100 } else { layer - 80 }) * 16 + position as usize) * 131072)
                            } else {
                                let stored_layer = if layer >= 40 { layer - 20 } else { layer };
                                (lane.resources.trace_buffers[gpu].buffer, (stored_layer * 16 + position as usize) * 40976)
                            };
                            let source = cuteafd_ffi::CuteafdDeviceBuffer {
                                ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer
                            };
                            let mut data = vec![0; bytes];
                            devices[gpu].run(|| lib.copy_d2h(&mut data, source))?;
                            trace.push((position, layer, data));
                        }
                    }
                }
                if !interleaved { expected_trace = trace; }
                else {
                    for (position, layer, bytes) in &trace {
                        let (_, _, expected) = expected_trace.iter().find(|(p,l,_)| p == position && l == layer).unwrap();
                        if bytes != expected {
                            if *layer >= 160 {
                                let changed = bytes.iter().zip(expected).filter(|(a,b)| a != b).count();
                                eprintln!("TRACE case={case} position={position} layer={} stage=qb_scratch changed_bytes={changed}", layer % 40);
                                continue;
                            }
                            if (80..160).contains(layer) && std::env::var("CUTEAFD_QUERY_COMPONENT").as_deref() == Ok("qb_pair") {
                                let rows = bytes.len() / (65536 + 2560);
                                let input_equal = bytes[rows*65536..] == expected[rows*65536..];
                                eprintln!("QB_PAIR case={case} layer={} position={position} rows={rows} input_equal={input_equal}", layer % 40);
                                if input_equal {
                                    let scratch_layer = layer % 40 + 160;
                                    let scratch = trace.iter().find(|(p,l,_)| p == position && *l == scratch_layer);
                                    let baseline_scratch = expected_trace.iter().find(|(p,l,_)| p == position && *l == scratch_layer);
                                    if let (Some((_,_,a)), Some((_,_,b))) = (scratch, baseline_scratch) {
                                        let changed = a.iter().zip(b).filter(|(a,b)| a != b).count();
                                        eprintln!("QB_SCRATCH case={case} layer={} position={position} changed_bytes={changed} total_bytes={}", layer % 40, a.len());
                                    }
                                    if let Some(directory) = std::env::var_os("CUTEAFD_TRACE_DUMP_DIR") {
                                        let directory = std::path::PathBuf::from(directory);
                                        std::fs::create_dir_all(&directory)?;
                                        let meta = directory.join("qb.json");
                                        if !meta.exists() {
                                            if let (Some((_,_,a)), Some((_,_,b))) = (scratch, baseline_scratch) {
                                                std::fs::write(directory.join("actual-scratch.bin"), a)?;
                                                std::fs::write(directory.join("expected-scratch.bin"), b)?;
                                            }
                                            std::fs::write(directory.join("input.bf16"), &bytes[rows*65536..])?;
                                            std::fs::write(directory.join("expected.bf16"), &expected[..rows*65536])?;
                                            std::fs::write(directory.join("actual.bf16"), &bytes[..rows*65536])?;
                                            std::fs::write(meta, format!("{{\"layer\":{},\"rows\":{rows},\"position\":{position}}}\n", layer % 40))?;
                                        }
                                    }
                                }
                            }
                            let paired = std::env::var_os("CUTEAFD_TRACE_FFN").is_some();
                            let is_query = *layer >= 80;
                            let is_output = !is_query && (!paired || *layer >= 40);
                            let pre_start = if !is_output { bytes.len() }
                                else if std::env::var_os("CUTEAFD_TRACE_PRE").is_some() { 0 }
                                else { bytes.len() / 40976 * 40960 };
                            let peak = bytes[..pre_start].chunks_exact(2).zip(expected[..pre_start].chunks_exact(2)).map(|(a,b)|
                                (f32::from_bits((u16::from_ne_bytes([a[0],a[1]]) as u32)<<16) -
                                 f32::from_bits((u16::from_ne_bytes([b[0],b[1]]) as u32)<<16)).abs()
                            ).fold(0f32,f32::max);
                            let pre_peak = bytes[pre_start..].chunks_exact(4).zip(expected[pre_start..].chunks_exact(4))
                                .map(|(a,b)| (f32::from_ne_bytes(a.try_into().unwrap()) -
                                    f32::from_ne_bytes(b.try_into().unwrap())).abs()).fold(0f32, f32::max);
                            eprintln!("TRACE case={case} position={position} layer={} stage={} bf16_peak={peak} pre_peak={pre_peak}",
                                layer % 40, if *layer >= 160 { "qb_scratch" } else if *layer >= 120 { "query_before" } else if is_query { "query_after" } else if is_output { "output" } else { "ffn_input" });
                        }
                    }
                }
                assert_eq!(requests.cache().committed_end(lease)?, 7);
                assert_eq!(pass.state, State::Idle);
                assert_eq!(other.state, State::Idle);
                assert_eq!(requests.begin_decoder_replay(lease)?, 0);
                let mut batch =
                    requests.prepare_replay(&[crate::families::deepseek_v41::v41_backbone_cache::CacheWork {
                        lease,
                        tokens: 7,
                        kind: ExpertV2SourceKind::Prefill,
                    }])?;
                let encoded = suffix.output()?;
                runtime.block_on(unsafe {
                    pass.execute(
                        &std::cell::RefCell::new(&mut requests),
                        &mut batch,
                        &mut transport,
                        0,
                        &[6],
                        None,
                        Some(&encoded),
                        true,
                    )
                })?;
                let next = pass.resources.head.greedy_output()?.to_vec();
                if let Some(expected) = expected_case {
                    assert_eq!(
                        next[0], expected,
                        "interleaved encoder greedy result differs from identical sequential chunks"
                    );
                } else {
                    expected_case = Some(next[0]);
                }
                pass.enqueue_cache_commit(&requests, &batch, &[7])?;
                runtime.block_on(async {
                    while !pass.poll_cache_commit()? {
                        tokio::task::yield_now().await;
                    }
                    Ok::<_, anyhow::Error>(())
                })?;
                pass.commit(&mut requests, &mut batch, &[7])?;
                requests.release(lease)?;
                assert_eq!(lib.cuda_get_device()?, 0);
                eprintln!(
                    "PASS distributed encoder case={case} interleaved={interleaved} chunks={} and live decoder replay",
                    chunks.len()
                );
            }
        }
        let lease = requests.admit(0, 94000)?;
        requests.begin_encoder(lease, 7)?;
        let mut cancelled_suffix = devices[map.attention(19)?]
            .own(|| EncoderSuffix::new(&lib, 7, EncoderSuffix::device_bytes(7)?))?;
        let calls = std::cell::Cell::new(0);
        let mut link = DistributedTargetPass::take_link(&mut prefill_slots.slots[0].1)?;
        let result = {
            let [(slot0, transport0), (slot1, transport1)] = &mut prefill_slots.slots[..] else { unreachable!() };
            runtime.block_on(unsafe {
                DistributedTargetPass::execute_encoder_units(vec![(slot0, transport0), (slot1, transport1)], &mut link, &mut requests, lease, &[&prompt[..3], &prompt[3..]], &mut cancelled_suffix, &[], &|| {
                        calls.set(calls.get() + 1);
                        calls.get() < 12
                    }, &|| Ok(()))
            })
        };
        DistributedTargetPass::restore_link(&mut prefill_slots.slots[0].1, link)?;
        assert!(result.is_err(), "disconnected encoder units succeeded");
        assert!(requests.cache().request_id(lease).is_err());
        assert!(prefill_slots.slots.iter().all(|(slot, _)| slot.state == State::Idle));
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!("PASS disconnected distributed encoder stream revoked both active chunks");
        // Cancel a reserved pass at a cooperative suspension. The guard revokes
        // its admission and resets both GPU-local lanes for subsequent reuse.
        let cancelled = requests.admit(0, 92002)?;
        requests.begin_encoder(cancelled, 7)?;
        let mut batch = requests.reserve_encoder(&[crate::families::deepseek_v41::v41_requests::RequestTokens {
            lease: cancelled,
            tokens: &prompt[..3],
            image_mask: None,
            kind: ExpertV2SourceKind::Prefill,
        }])?;
        {
            use std::future::Future;
            let bank = std::cell::RefCell::new(&mut requests);
            let mut pending = Box::pin(unsafe {
                pass.execute(
                    &bank,
                    &mut batch,
                    &mut transport,
                    0,
                    &[],
                    Some(&mut suffix),
                    None,
                    false,
                )
            });
            runtime.block_on(std::future::poll_fn(|cx| {
                assert!(pending.as_mut().poll(cx).is_pending());
                std::task::Poll::Ready(())
            }));
        }
        assert!(requests.cache().request_id(cancelled).is_err());
        assert!(requests.validate(&batch).is_err());
        assert_eq!(pass.state, State::Idle);
        assert_eq!(lib.cuda_get_device()?, 0);
        let recovered = requests.admit(0, 92003)?;
        let mut batch = requests.prepare(&[crate::families::deepseek_v41::v41_requests::RequestTokens {
            lease: recovered,
            tokens: &prompt[..1],
            image_mask: None,
            kind: ExpertV2SourceKind::Prefill,
        }])?;
        runtime.block_on(unsafe {
            pass.execute(
                &std::cell::RefCell::new(&mut requests),
                &mut batch,
                &mut transport,
                0,
                &[0],
                None,
                None,
                true,
            )
        })?;
        pass.enqueue_cache_commit(&requests, &batch, &[1])?;
        runtime.block_on(async {
            while !pass.poll_cache_commit()? {
                tokio::task::yield_now().await;
            }
            Ok::<_, anyhow::Error>(())
        })?;
        pass.commit(&mut requests, &mut batch, &[1])?;
        assert_eq!(requests.cache().committed_end(recovered)?, 1);
        requests.release(recovered)?;
        eprintln!(
            "PASS cancelled distributed reserved pass revoked admission and restored both GPU owners"
        );
        Ok(())
    }
}
