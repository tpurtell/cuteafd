//! Two-RTX V4.1 encoder prefill on the shared executor: the same hooks and
//! order as the single-RTX units (`super::super::encoder_units`), on placed
//! lanes. Each in-flight slot owns one distributed pass and its link.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::CacheLease;
use crate::families::deepseek_v41::v41_backbone_execution::{UnitPendingLayer, UnitPreparedLayer};
use crate::families::deepseek_v41::v41_requests::RequestTokens;
use crate::shared::prefill_pipeline::{run_prefill_units_async, AsyncPrefillHooks, FfnRoute, Unit};
use cuteafd_transport::ExpertV2SourceKind;
use std::cell::RefCell;

impl<'w, 'a> DistributedTargetPass<'w, 'a> {
    /// The per-layer preparation `execute_inner` runs before the FFN, for a
    /// reserved encoder unit: the lane advance (with its block hop), Engram at
    /// layers 1 and 14, cache production and the attention, finished.
    /// # Safety
    /// The unit's batch and this pass's owners stay live until the layer lands.
    async unsafe fn prepare_unit_layer(&mut self, requests: &RefCell<&mut Requests<'a>>,
        batch: &mut RequestBatch, layer: usize) -> Result<UnitPreparedLayer> {
        let gpu = self.resources.map.attention(layer)?;
        let device = self.resources.lanes[gpu].device;
        if layer == 0 {
            // Prepared at begin (`prepare_unit_input`): the query here.
            device.future(unsafe { self.resources.lanes[gpu].get_mut().begin_prepared_cooperative() }).await?;
        } else {
            unsafe { self.advance(layer).await?; }
            if (2..14).contains(&layer) && !self.resources.engram.staged(1)
                    && self.resources.engram.can_enqueue(1)? {
                if let cuteafd_loader::EngramGatherPoll::Ready(lease) = requests.borrow().poll_engram_layer(batch, 1)? {
                    self.resources.engram.enqueue_lease(lease, batch.engram_wave()?)?;
                }
            }
            if [1, 14].contains(&layer) {
                let gate = usize::from(layer == 14);
                if !self.resources.engram.staged(gate) {
                    batch.wait_engram_gather(gate, self.resources.timeout).await?;
                    self.resources.engram.wait_enqueue(gate, self.resources.timeout).await?;
                    let gathered = requests.with_requests(|r| r.poll_engram_gather(batch, &self.resources.lanes[gpu]))?;
                    match gathered {
                        cuteafd_loader::EngramGatherPoll::Ready(lease) => self.resources.engram.enqueue_lease(lease, batch.engram_wave()?)?,
                        _ => anyhow::bail!("Engram gather wait returned no rows"),
                    }
                }
                unsafe { self.resources.engram.apply_staged(&mut self.resources.lanes[gpu], gate).await?; }
            }
            device.future(unsafe { self.resources.lanes[gpu].get_mut().begin_prepared_cooperative() }).await?;
        }
        let index = if [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer) { self.resources.indices[gpu].as_mut() } else { None };
        let mut production = unsafe { requests.with_requests(|r| self.resources.execution.enqueue_production(
            r.cache(), batch.cache()?, &self.resources.lanes[gpu], index))? };
        while !unsafe { requests.borrow_mut().poll_distributed_encoder_production(batch, &mut production)? } {
            tokio::task::yield_now().await;
        }
        drop(production);
        let prepared = unsafe { requests.with_requests(|r| self.resources.execution.prepare_layer(r.cache(),
            batch.cache()?, &mut self.resources.lanes[gpu], self.resources.indices[gpu].as_ref().map(DeviceOwner::get)))? };
        let prepared = prepared.into_unit().await?;
        // Publish this layer's window now (as one RTX does): the next unit
        // attends it before this unit lands its FFN.
        if crate::shared::memory::chain::deferred() {
            // Device-ordered (#815 option (i)): a chained direct commit after the attention.
            requests.borrow_mut().publish_distributed_unit_window(batch, &mut self.resources.execution, layer)?;
        } else {
            unsafe { requests.with_requests(|r| self.resources.execution
                .enqueue_unit_window_publication(r.cache(), batch.cache()?, layer))?; }
            while !self.resources.execution.poll_encoder_publication()? { tokio::task::yield_now().await; }
            requests.borrow_mut().finish_distributed_encoder_publication(batch, &mut self.resources.execution)?;
        }
        Ok(prepared)
    }

    /// After a unit's attention hook joined every fork reader on the chain:
    /// release the shared scratch of the layer's lane and index lane.
    pub(super) fn release_attention_scratch(&self, layer: usize) -> Result<()> {
        let gpu = self.resources.map.attention(layer)?;
        let lane = &self.resources.lanes[gpu];
        lane.device.run(|| lane.release_scratch())?;
        if let Some(index) = self.resources.indices[gpu].as_ref() {
            index.device.run(|| index.release_scratch())?;
        }
        Ok(())
    }
    /// Completes a unit's layer from its FFN result and publishes its KV.
    /// # Safety
    /// `completed` is this unit's layer result; the batch is reserved.
    async unsafe fn complete_unit_layer(&mut self, requests: &RefCell<&mut Requests<'a>>, batch: &RequestBatch,
        layer: usize, completed: crate::families::deepseek_v41::v41_backbone_execution::CompletedLayer<'_>) -> Result<()> {
        let gpu = self.resources.map.attention(layer)?;
        let _ = requests;
        unsafe { self.resources.execution.complete_layer(batch.cache()?, &mut self.resources.lanes[gpu], completed).await }
    }
}

struct UnitSlot<'s, 'w, 'a> {
    pass: &'s mut DistributedTargetPass<'w, 'a>,
    transport: &'s mut DeviceOwner<'a, NativeTp4Wave<'a>>,
    batch: Option<RequestBatch>,
    prepared: Option<UnitPreparedLayer>,
    started: Instant,
    /// As on one RTX: this slot's per-layer publication events.
    published: super::super::encoder_units::PublishedLayers<'a>,
}
#[cfg(test)]
thread_local! {
    /// Negative control: leave the transport's TP2 FFN unwindowed (pre-C1).
    pub(crate) static SKIP_TP2_WINDOW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Each unit's (pass lane window, TP2 FFN window) as its rows are selected.
    pub(crate) static UNIT_WINDOWS: std::cell::RefCell<Vec<(crate::shared::memory::row_window::RowWindow,
        Option<crate::shared::memory::row_window::RowWindow>)>> = const { std::cell::RefCell::new(Vec::new()) };
}
impl UnitSlot<'_, '_, '_> {
    /// The slot's rows in every shared plane: the pass's components and its
    /// borrowed transport's TP2 FFN (peer inputs, shared and final outputs).
    fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        self.pass.select_window(window.clone())?;
        #[cfg(test)]
        if SKIP_TP2_WINDOW.with(|skip| skip.get()) { return Ok(()); }
        let device = self.transport.device;
        device.run(|| self.transport.get_mut().select_window(window))
    }
}
struct UnitHooks<'s, 'r, 'w, 'a> {
    slots: Vec<UnitSlot<'s, 'w, 'a>>,
    /// The call's unit: slot `k`'s window is rows `k * unit_rows` onward.
    unit_rows: usize,
    /// The one prefill transport (on the transports' GPU): every unit's wave
    /// lands in its own return slot.
    prefill: &'s mut DeviceOwner<'a, crate::families::deepseek_v41::v41_experts::coordinator::V41Link<'a>>,
    requests: RefCell<&'r mut Requests<'a>>,
    lease: CacheLease,
    packed: Option<&'s [(CacheLease, &'s [u32])]>,
    chunks: &'s [&'s [u32]],
    suffix: &'s mut DeviceOwner<'a, EncoderSuffix<'a>>,
    keep_running: &'s dyn Fn() -> bool,
    before_unit: &'s dyn Fn() -> Result<()>,
    /// The next call's leading chunks (#762 window cycles): the last units of
    /// this call prefetch their Engram rows, so the next call's units start warm.
    ahead: &'s [&'s [u32]],
    /// Every unit committed: the request stays admitted.
    complete: bool,
}
impl Drop for UnitHooks<'_, '_, '_, '_> {
    fn drop(&mut self) {
        // Device-ordered units leave queued work (and cross-slot waits) on
        // every slot's chain: drain them all before any storage is released.
        for slot in &mut self.slots {
            if slot.pass.chain.is_some() {
                crate::shared::decode_graph::fatal_drain(slot.pass.drain_chain(), "cancelled prefill unit chain");
            }
            // A completed prompt drained every unit's ring at its commit; a
            // failed or cancelled unit's captures are dropped, so the window
            // reset below finds every router at rest.
            if let Err(error) = slot.pass.discard_route_rings() {
                tracing::error!(%error, "discarding a cancelled prefill unit's route captures");
            }
        }
        // A failed or cancelled prefill revokes the whole request (its later
        // units' reservations included) after each slot's work drained.
        let mut requests = self.requests.borrow_mut();
        for slot in &mut self.slots {
            if let Some(mut batch) = slot.batch.take() {
                if let Err(error) = slot.pass.abort_cache_commit(&mut requests) {
                    tracing::error!(%error, "draining cancelled distributed encoder unit");
                }
                if let Err(error) = slot.pass.resources.execution.abort_queries() {
                    tracing::error!(%error, "aborting cancelled distributed encoder unit's producers");
                }
                requests.revoke_batch(&mut batch);
                if let Err(error) = slot.pass.discard(&mut batch) {
                    tracing::error!(%error, "discarding cancelled distributed encoder unit");
                }
            }
        }
        if !self.complete && self.packed.is_none() && requests.cache().request_id(self.lease).is_ok() {
            if let Err(error) = requests.release(self.lease) {
                tracing::error!(%error, "revoking cancelled distributed encoder units");
            }
        }
        drop(requests);
        // Every exit (completion, error, cancel) leaves each drained slot at
        // rows from 0, so later non-unit work on slot 0 finds no stale window.
        for slot in &mut self.slots {
            if let Err(error) = slot.select_window(crate::shared::memory::row_window::RowWindow::NONE) {
                tracing::error!(%error, "resetting a prefill slot's window");
            }
        }
    }
}
impl<'a> UnitHooks<'_, '_, '_, 'a> {
    fn previous_publication(&self, unit: &Unit, layer: usize) -> Option<*mut std::ffi::c_void> {
        let slot = super::super::encoder_units::previous_slot(unit, self.slots.len())?;
        self.slots[slot].published.event(layer)
    }
    async fn begin_unit_inner(&mut self, unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        (self.before_unit)()?;
        let slot = &mut self.slots[unit.slot];
        ensure!(slot.batch.is_none() && slot.prepared.is_none(), "encoder unit slot occupied");
        slot.pass.resources.engram.reset_uploads()?;
        slot.started = Instant::now();
        let chunk = self.chunks[unit.index];
        let inputs: Vec<_> = match self.packed {
            Some(members) => members.iter().map(|&(lease, tokens)| RequestTokens {
                lease, tokens, image_mask: None, kind: ExpertV2SourceKind::Prefill }).collect(),
            None => vec![RequestTokens { lease: self.lease, tokens: chunk,
                image_mask: None, kind: ExpertV2SourceKind::Prefill }],
        };
        let batch = self.requests.borrow_mut().reserve_encoder(&inputs)?;
        if self.packed.is_none() {
            let next: Vec<&[u32]> = self.chunks[unit.index + 1..].iter().chain(self.ahead).copied().collect();
            self.requests.borrow_mut().prefetch_encoder(self.lease, &next)?;
        }
        // The slot is at rest (drained in `begin_unit`): select this unit's
        // rows in every shared plane (its transport's TP2 FFN included, C1),
        // then prepare layer 0's input. Its query runs in its attention hook
        // (the query outputs are shared scratch).
        slot.batch = Some(batch);
        slot.select_window(super::super::encoder_units::unit_window(unit, self.unit_rows)?)?;
        #[cfg(test)]
        UNIT_WINDOWS.with(|seen| seen.borrow_mut().push((slot.pass.resources.lanes[0].window().clone(),
            slot.transport.tp2_window().cloned())));
        let batch = slot.batch.as_mut().context("encoder unit batch")?;
        slot.pass.state.begin()?;
        slot.pass.prepare_unit_input(&self.requests, batch)
    }
    async fn attention_inner(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        let slot = &mut self.slots[unit.slot];
        ensure!(slot.prepared.is_none(), "encoder attention still pending");
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        // SAFETY: the preceding hook consumed the previous layer's output.
        slot.prepared = Some(unsafe { slot.pass.prepare_unit_layer(&self.requests, batch, layer).await? });
        // Every fork reader of the attention's shared scratch joined the chain head.
        slot.pass.release_attention_scratch(layer)?;
        Ok(())
    }
    async fn local_ffn_inner(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let prepared = slot.prepared.take().context("encoder attention absent")?;
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        let gpu = slot.pass.resources.map.attention(layer)?;
        let device = slot.pass.resources.lanes[gpu].device;
        // SAFETY: the bound input stays in this unit's slot through the FFN.
        let completed = device.future(unsafe { prepared.execute(slot.pass.resources.lanes[gpu].get_mut(),
            slot.transport.get_mut(), 0, batch.image_mask()) }).await?;
        unsafe { slot.pass.complete_unit_layer(&self.requests, batch, layer, completed).await }
    }
    async fn send_experts_inner(&mut self, layer: usize, unit: &Unit) -> Result<UnitPendingLayer> {
        let slot = &mut self.slots[unit.slot];
        let prepared = slot.prepared.take().context("encoder attention absent")?;
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        let gpu = slot.pass.resources.map.attention(layer)?;
        let device = slot.pass.resources.lanes[gpu].device;
        // SAFETY: the slot is retained until the matching FIFO landing hook.
        device.future(unsafe { prepared.send(slot.pass.resources.lanes[gpu].get_mut(), slot.transport.get_mut(),
            self.prefill.get_mut(), 0, batch.image_mask()) }).await
    }
    async fn land_experts_inner(&mut self, layer: usize, unit: &Unit, ticket: UnitPendingLayer) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        let gpu = slot.pass.resources.map.attention(layer)?;
        let device = slot.pass.resources.lanes[gpu].device;
        // SAFETY: no other unit overwrote this slot's shared values or input.
        // As `PlacedPreparedLayer::execute`: on the lane's GPU; the collect
        // switches to the transport's GPU and returns the result here.
        let completed = device.future(unsafe { ticket.land(slot.pass.resources.lanes[gpu].get_mut(),
            slot.transport.get_mut(), self.prefill.get_mut()) }).await?;
        unsafe { slot.pass.complete_unit_layer(&self.requests, batch, layer, completed).await }
    }
    async fn end_unit_inner(&mut self, unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        // SAFETY: the unit's encoder output is complete; units end in order.
        unsafe { slot.pass.finish_encoder_unit(&self.requests, batch, self.suffix).await }
    }
    async fn commit_unit_inner(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        let rows = super::super::encoder_units::member_rows(batch)?;
        slot.pass.enqueue_cache_commit(&self.requests.borrow(), batch, &rows)?;
        let outcome = crate::shared::serve::lanes::wait_ready(|| slot.pass.poll_cache_commit(),
            crate::shared::serve::lanes::COMMIT_DEADLINE).await?;
        anyhow::ensure!(outcome == crate::shared::serve::lanes::WaitOutcome::Ready,
            "encoder unit commit did not drain within {:?}", crate::shared::serve::lanes::COMMIT_DEADLINE);
        slot.pass.commit(&mut self.requests.borrow_mut(), batch, &rows)?;
        slot.batch = None;
        crate::families::deepseek_v41::v41_native_serve::console::totals::prefill(unit.rows.len());
        crate::families::deepseek_v41::v41_native_serve::console::Prefill::done(
            crate::families::deepseek_v41::v41_native_serve::console::PrefillKind::Chunk,
            unit.slot, unit.index, self.chunks.len(), unit.rows.len(), slot.started);
        Ok(())
    }
}
/// Device order for a unit's hooks, as for the pass's verification chain:
/// off when a lane's opt-in TP2 peer transfers are host-ordered.
fn unit_chain<'a>(pass: &DistributedTargetPass<'_, 'a>)
    -> Result<Option<crate::shared::memory::chain::ChainHandle<'a>>> {
    if pass.resources.lanes.iter().any(|lane| lane.uses_peer_projection()) { return Ok(None); }
    pass.unit_chain().map(Some)
}

/// Polls `future` in `chain`'s scope when there is one.
async fn scoped<F: std::future::Future>(chain: Option<crate::shared::memory::chain::ChainHandle<'_>>, future: F)
    -> F::Output {
    match chain { Some(chain) => chain.scope(future).await, None => future.await }
}

impl<'a> AsyncPrefillHooks for UnitHooks<'_, '_, '_, 'a> {
    type Ticket = UnitPendingLayer;
    fn route(&self, layer: usize) -> FfnRoute {
        if self.slots[0].transport.has_local_layer(layer) { FfnRoute::Local } else { FfnRoute::Spark }
    }
    // As on one RTX: every hook polls in its unit's device-ordered chain (#815).
    async fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        // As on one RTX: the slot's previous unit committed without a host
        // drain; drain before this unit rewrites the pass's pinned staging.
        if self.slots[unit.slot].pass.chain.is_some() { self.slots[unit.slot].pass.drain_chain()?; }
        crate::shared::spark_waves::healthy()?;
        let chain = unit_chain(self.slots[unit.slot].pass)?;
        scoped(chain, self.begin_unit_inner(unit)).await
    }
    async fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        // As on one RTX (#815 option (i)): device-ordered, no host wait. The
        // attention orders after the previous unit's publication of this layer;
        // its window and source publications are chained direct commits.
        crate::shared::spark_waves::healthy()?;
        let Some(chain) = unit_chain(self.slots[unit.slot].pass)? else {
            // Host-ordered TP2 peer transfers keep the established host order.
            if self.slots[unit.slot].pass.chain.is_some() { self.slots[unit.slot].pass.drain_chain()?; }
            return self.attention_inner(layer, unit).await;
        };
        let event = self.previous_publication(unit, layer);
        chain.scope(async {
            super::super::encoder_units::order_after(self.slots[unit.slot].published.library, event)?;
            self.attention_inner(layer, unit).await?;
            self.slots[unit.slot].published.record(layer)
        }).await
    }
    async fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let chain = unit_chain(self.slots[unit.slot].pass)?;
        scoped(chain, self.local_ffn_inner(layer, unit)).await
    }
    async fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        let chain = unit_chain(self.slots[unit.slot].pass)?;
        scoped(chain, self.send_experts_inner(layer, unit)).await
    }
    async fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: Self::Ticket) -> Result<()> {
        let chain = unit_chain(self.slots[unit.slot].pass)?;
        scoped(chain, self.land_experts_inner(layer, unit, ticket)).await
    }
    async fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        // Device-ordered (#815 option (i)): the suffix capture and the
        // boundary source publication chain behind the unit's last layer.
        crate::shared::spark_waves::healthy()?;
        let Some(chain) = unit_chain(self.slots[unit.slot].pass)? else {
            if self.slots[unit.slot].pass.chain.is_some() { self.slots[unit.slot].pass.drain_chain()?; }
            return self.end_unit_inner(unit).await;
        };
        let boundary = super::super::encoder_units::BOUNDARY;
        let event = self.previous_publication(unit, boundary);
        chain.scope(async {
            super::super::encoder_units::order_after(self.slots[unit.slot].published.library, event)?;
            self.end_unit_inner(unit).await?;
            self.slots[unit.slot].published.record(boundary)
        }).await
    }
    async fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        // As on one RTX: every window and source was published on the device
        // during the unit's layers, so its cache commit enqueues nothing and the
        // commit is host bookkeeping. Without a device order (host-ordered peer
        // transfers) the unit's work is drained first, as before.
        crate::shared::spark_waves::healthy()?;
        let pass = &mut *self.slots[unit.slot].pass;
        // As on one RTX: a unit captured with routes drains before its ring
        // is read, and the ring is drained before the unit commits.
        if pass.chain.is_some() && (pass.resources.capture_routes || unit_chain(pass)?.is_none()) {
            pass.drain_chain()?;
        }
        pass.drain_route_rings()?;
        self.commit_unit_inner(unit).await
    }
}


impl<'w, 'a> DistributedTargetPass<'w, 'a> {
    /// Encoder prefill of `chunks` (one prompt) as row units on the shared
    /// executor, one unit per slot in flight: slot k is `slots[k]` with its link.
    /// # Safety
    /// Passes and transports have independent mutable storage and the same model,
    /// placement and capacity. Poll on one CUDA-owning executor thread.
    pub(crate) async unsafe fn execute_encoder_units(slots: Vec<(&mut Self, &mut DeviceOwner<'a, NativeTp4Wave<'a>>)>,
        prefill: &mut DeviceOwner<'a, crate::families::deepseek_v41::v41_experts::coordinator::V41Link<'a>>, requests: &mut Requests<'a>, lease: CacheLease, chunks: &[&[u32]],
        suffix: &mut DeviceOwner<'a, EncoderSuffix<'a>>, ahead: &[&[u32]], keep_running: &dyn Fn() -> bool,
        before_unit: &dyn Fn() -> Result<()>) -> Result<()> {
        // SAFETY: forwards the caller's retained slot, request and suffix contract.
        unsafe { Self::execute_encoder_units_inner(slots, prefill, requests, None,
            lease, chunks, suffix, ahead, keep_running, before_unit).await }
    }
    /// One packed short-prompt unit on the shared waves executor.
    /// # Safety
    /// Every member owns its lease and the shared suffix through drained completion.
    pub(crate) async unsafe fn execute_packed_encoder_unit(
        slots: Vec<(&mut Self, &mut DeviceOwner<'a, NativeTp4Wave<'a>>)>,
        prefill: &mut DeviceOwner<'a, crate::families::deepseek_v41::v41_experts::coordinator::V41Link<'a>>,
        requests: &mut Requests<'a>, members: &[(CacheLease, &[u32])], suffix: &mut DeviceOwner<'a, EncoderSuffix<'a>>,
    ) -> Result<()> {
        let tokens: Vec<_> = members.iter().flat_map(|(_, t)| t.iter().copied()).collect();
        // SAFETY: member leases and the packed suffix stay live through the executor drain.
        unsafe { Self::execute_encoder_units_inner(slots, prefill, requests, Some(members),
            members[0].0, &[&tokens], suffix, &[], &|| true, &|| Ok(())).await }
    }
    async unsafe fn execute_encoder_units_inner(slots: Vec<(&mut Self, &mut DeviceOwner<'a, NativeTp4Wave<'a>>)>,
        prefill: &mut DeviceOwner<'a, crate::families::deepseek_v41::v41_experts::coordinator::V41Link<'a>>, requests: &mut Requests<'a>, packed: Option<&[(CacheLease, &[u32])]>, lease: CacheLease, chunks: &[&[u32]],
        suffix: &mut DeviceOwner<'a, EncoderSuffix<'a>>, ahead: &[&[u32]], keep_running: &dyn Fn() -> bool,
        before_unit: &dyn Fn() -> Result<()>) -> Result<()> {
        if packed.is_none() { requests.prefetch_encoder(lease, &chunks.iter().chain(ahead).copied().collect::<Vec<_>>())?; }
        let waves = slots.len();
        ensure!((1..=4).contains(&waves), "V4.1 prefill runs 1..=4 units in flight");
        ensure!(slots.iter().all(|(pass, _)| pass.state == State::Idle), "distributed encoder lane still in use");
        ensure!(slots.iter().all(|(pass, _)| pass.resources.map == slots[0].0.resources.map),
            "distributed encoder lane placement differs");
        let mut units = super::super::encoder_units::encoder_units(chunks)?;
        if let Some(members) = packed { super::super::encoder_units::packed_members(&mut units[0], members.iter().map(|(_, tokens)| tokens.len())); }
        for unit in &mut units { unit.slot = unit.index % waves; }
        for layer in 0..20 {
            ensure!(slots.iter().all(|(_, transport)| transport.has_local_layer(layer) == slots[0].1.has_local_layer(layer)),
                "encoder slot expert placement differs");
        }
        let map = slots[0].0.resources.map;
        let library = slots[0].0.resources.lanes[0].device.library;
        let unit_rows = chunks.iter().map(|chunk| chunk.len()).max().unwrap_or(0);
        let mut hooks = UnitHooks {
            prefill,
            unit_rows,
            slots: slots.into_iter().map(|(pass, transport)| {
                pass.set_attention_decode(false);
                let published = super::super::encoder_units::PublishedLayers::on_devices(library,
                    |layer| Ok(pass.resources.lanes[map.attention(layer)?].device.id))?;
                Ok(UnitSlot { pass, transport, batch: None, prepared: None, started: Instant::now(), published })
            }).collect::<Result<Vec<_>>>()?,
            requests: RefCell::new(requests), lease, packed, chunks, suffix, ahead, keep_running, before_unit, complete: false,
        };
        run_prefill_units_async(&mut hooks, &units, waves).await?;
        // Every unit's chained work completes before the decoder replay reads it.
        for slot in &hooks.slots { if slot.pass.chain.is_some() { slot.pass.drain_chain()?; } }
        crate::shared::spark_waves::healthy()?;
        hooks.complete = true;
        // Non-unit work (replay, continuation, scoring) runs at rows from 0.
        // Here a refusal is loud; error and cancel exits reset in `Drop`.
        for slot in &mut hooks.slots { slot.select_window(crate::shared::memory::row_window::RowWindow::NONE)?; }
        Ok(())
    }
}
