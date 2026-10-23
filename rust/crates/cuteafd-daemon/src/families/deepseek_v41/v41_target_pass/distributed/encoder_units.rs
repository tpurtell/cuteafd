//! Single-pass shared-executor binding for the opt-in distributed encoder.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::CacheLease;
use crate::families::deepseek_v41::v41_backbone_execution::{UnitPendingLayer, UnitPreparedLayer};
use crate::families::deepseek_v41::v41_requests::RequestTokens;
use crate::shared::prefill_pipeline::{run_prefill_units_async, AsyncPrefillHooks, FfnRoute, Unit};
use cuteafd_transport::ExpertV2SourceKind;
use std::cell::RefCell;

struct UnitHooks<'s, 'r, 'w, 'a> {
    pass: &'s mut DistributedTargetPass<'w, 'a>,
    requests: RefCell<&'r mut Requests<'a>>,
    lease: CacheLease,
    chunks: &'s [&'s [u32]],
    transport: &'s mut DeviceOwner<'a, NativeTp4Wave<'a>>,
    suffix: &'s mut DeviceOwner<'a, EncoderSuffix<'a>>,
    keep_running: &'s dyn Fn() -> bool,
    before_unit: &'s dyn Fn() -> Result<()>,
    batch: Option<RequestBatch>,
    prepared: Option<UnitPreparedLayer>,
    started: Instant,
}
impl Drop for UnitHooks<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if let Some(mut batch) = self.batch.take() {
            let mut requests = self.requests.borrow_mut();
            if let Err(error) = self.pass.abort_cache_commit(&mut requests) {
                tracing::error!(%error, "draining cancelled encoder unit publication");
            }
            requests.revoke_batch(&mut batch);
            if let Err(error) = self.pass.discard(&mut batch) {
                tracing::error!(%error, "discarding cancelled distributed encoder unit");
            }
        }
    }
}
impl AsyncPrefillHooks for UnitHooks<'_, '_, '_, '_> {
    type Ticket = UnitPendingLayer;
    fn route(&self, layer: usize) -> FfnRoute {
        if self.transport.has_local_layer(layer) {
            FfnRoute::Local
        } else {
            FfnRoute::Spark
        }
    }
    async fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        ensure!(
            unit.slot == 0 && self.batch.is_none() && self.prepared.is_none(),
            "serial distributed encoder slot occupied"
        );
        ensure!((self.keep_running)(), "client disconnected");
        (self.before_unit)()?;
        self.started = Instant::now();
        self.batch = Some(
            self.requests
                .borrow_mut()
                .reserve_encoder(&[RequestTokens {
                    lease: self.lease,
                    tokens: self.chunks[unit.index],
                    image_mask: None,
                    kind: ExpertV2SourceKind::Prefill,
                }])?,
        );
        let batch = self.batch.as_mut().context("encoder unit batch absent")?;
        self.requests.borrow().validate(batch)?;
        self.pass.state.begin()?;
        self.pass
            .resources
            .execution
            .restart_for(CacheStage::Encoder);
        for index in self.pass.resources.indices.iter_mut().flatten() {
            index.device.run(|| index.restart())?;
        }
        let gpu = self.pass.resources.map.attention(0)?;
        let device = self.pass.resources.lanes[gpu].device;
        device.run(|| self.pass.resources.lanes[gpu].restart())?;
        let text = self.requests.borrow().text_embedding_input(batch)?;
        if let Some((tokens, positions)) = text {
            // SAFETY: this unit exclusively owns its input and embedding storage.
            device
                .future(unsafe {
                    self.pass.resources.lanes[gpu]
                        .get_mut()
                        .begin_tokens_cooperative(
                            self.pass.resources.embedding.get_mut(),
                            tokens,
                            &positions,
                        )
                })
                .await?;
        } else {
            // SAFETY: the reserved batch and lane remain bound to the same unit.
            unsafe {
                device.run(|| {
                    self.requests.borrow().begin_input(
                        batch,
                        self.pass.resources.embedding.get_mut(),
                        self.pass.resources.lanes[gpu].get_mut(),
                    )
                })?;
            }
        }
        Ok(())
    }
    async fn attention(&mut self, layer: usize, _unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        ensure!(
            self.prepared.is_none(),
            "distributed encoder attention pending"
        );
        let gpu = self.pass.resources.map.attention(layer)?;
        let device = self.pass.resources.lanes[gpu].device;
        let batch = self.batch.as_mut().context("encoder unit batch absent")?;
        if layer != 0 {
            // SAFETY: the previous FFN and any peer handoff have completed.
            unsafe {
                self.pass.advance(layer).await?;
            }
            if [1, 14].contains(&layer) {
                let started = Instant::now();
                loop {
                    let gathered = self
                        .requests
                        .borrow()
                        .poll_engram_gather(batch, &self.pass.resources.lanes[gpu])?;
                    match gathered {
                        cuteafd_loader::EngramGatherPoll::Ready(lease) => {
                            // SAFETY: the gather lease is retained through gate completion.
                            unsafe {
                                self.pass
                                    .resources
                                    .engram
                                    .apply(&mut self.pass.resources.lanes[gpu], &lease.view()?)
                                    .await?;
                            }
                            break;
                        }
                        cuteafd_loader::EngramGatherPoll::Cancelled => {
                            anyhow::bail!("encoder unit Engram cancelled")
                        }
                        cuteafd_loader::EngramGatherPoll::Pending => {}
                    }
                    ensure!(
                        started.elapsed() < self.pass.resources.timeout,
                        "encoder unit Engram timed out at layer {layer}"
                    );
                    tokio::task::yield_now().await;
                }
            }
            // SAFETY: every prior consumer of the unit's input has completed.
            device
                .future(unsafe {
                    self.pass.resources.lanes[gpu]
                        .get_mut()
                        .begin_prepared_cooperative()
                })
                .await?;
        }
        let index = if [2, 8, 14].contains(&layer) {
            self.pass.resources.indices[gpu].as_mut()
        } else {
            None
        };
        // SAFETY: retain cache, query and index owners through production polling.
        let mut production = unsafe {
            self.pass.resources.execution.enqueue_production(
                self.requests.borrow().cache(),
                batch.cache()?,
                &self.pass.resources.lanes[gpu],
                index,
            )?
        };
        // SAFETY: no external producer or cache mutation races this unit.
        while !unsafe {
            self.requests
                .borrow_mut()
                .poll_distributed_encoder_production(batch, &mut production)?
        } {
            tokio::task::yield_now().await;
        }
        drop(production);
        // SAFETY: completed production and selection remain live through attention.
        let prepared = unsafe {
            self.pass.resources.execution.prepare_layer(
                self.requests.borrow().cache(),
                batch.cache()?,
                &mut self.pass.resources.lanes[gpu],
                self.pass.resources.indices[gpu]
                    .as_ref()
                    .map(DeviceOwner::get),
            )?
        };
        self.prepared = Some(prepared.into_unit().await?);
        // SAFETY: attention has drained; the FFN does not read the KV proposal.
        unsafe {
            self.pass
                .resources
                .execution
                .enqueue_encoder_attention_publication(
                    self.requests.borrow().cache(),
                    batch.cache()?,
                    layer,
                )?;
        }
        while !self.pass.resources.execution.poll_encoder_publication()? {
            tokio::task::yield_now().await;
        }
        self.requests
            .borrow_mut()
            .finish_distributed_encoder_publication(batch, &mut self.pass.resources.execution)?;
        Ok(())
    }
    async fn local_ffn(&mut self, layer: usize, _unit: &Unit) -> Result<()> {
        let gpu = self.pass.resources.map.attention(layer)?;
        let device = self.pass.resources.lanes[gpu].device;
        let prepared = self.prepared.take().context("encoder attention absent")?;
        let batch = self.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: input and transport results stay live through final mHC completion.
        let completed = device
            .future(unsafe {
                prepared.execute(
                    self.pass.resources.lanes[gpu].get_mut(),
                    self.transport.get_mut(),
                    0,
                    batch.image_mask(),
                )
            })
            .await?;
        // SAFETY: completed result carries this unit's original query binding.
        unsafe {
            self.pass
                .resources
                .execution
                .complete_layer(
                    batch.cache()?,
                    &mut self.pass.resources.lanes[gpu],
                    completed,
                )
                .await?;
        }
        Ok(())
    }
    async fn send_experts(&mut self, layer: usize, _unit: &Unit) -> Result<Self::Ticket> {
        let gpu = self.pass.resources.map.attention(layer)?;
        let device = self.pass.resources.lanes[gpu].device;
        let prepared = self.prepared.take().context("encoder attention absent")?;
        let batch = self.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: shared/input storage remains reserved until the landing hook.
        device
            .future(unsafe {
                prepared.send(
                    self.pass.resources.lanes[gpu].get_mut(),
                    self.transport.get_mut(),
                    0,
                    batch.image_mask(),
                )
            })
            .await
    }
    async fn land_experts(
        &mut self,
        layer: usize,
        _unit: &Unit,
        ticket: Self::Ticket,
    ) -> Result<()> {
        let gpu = self.pass.resources.map.attention(layer)?;
        let device = self.pass.resources.lanes[gpu].device;
        let batch = self.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: the serial slot's shared values have not been overwritten.
        let completed = device
            .future(unsafe {
                ticket.land(
                    self.pass.resources.lanes[gpu].get_mut(),
                    self.transport.get_mut(),
                )
            })
            .await?;
        // SAFETY: the original batch and lane remain retained through mHC completion.
        unsafe {
            self.pass
                .resources
                .execution
                .complete_layer(
                    batch.cache()?,
                    &mut self.pass.resources.lanes[gpu],
                    completed,
                )
                .await?;
        }
        Ok(())
    }
    async fn end_unit(&mut self, _unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        let batch = self.batch.as_ref().context("encoder unit batch absent")?;
        let gpu = self.pass.resources.map.attention(19)?;
        let device = self.pass.resources.lanes[gpu].device;
        ensure!(
            self.suffix.device.id == device.id,
            "encoder unit suffix GPU differs"
        );
        let output = self.pass.resources.lanes[gpu].output()?;
        // SAFETY: completed encoder output remains immutable through suffix capture.
        device
            .future(unsafe {
                self.suffix
                    .get_mut()
                    .capture_cooperative(&output, &self.pass.resources.suffix_stream)
            })
            .await?;
        // SAFETY: output has no remaining consumers before the source-20 handoff.
        unsafe {
            self.pass.advance(20).await?;
        }
        let gpu = self.pass.resources.map.attention(20)?;
        let device = self.pass.resources.lanes[gpu].device;
        // SAFETY: the source-20 query reads this unit's completed encoder boundary.
        device
            .future(unsafe {
                self.pass.resources.lanes[gpu]
                    .get_mut()
                    .begin_prepared_cooperative()
            })
            .await?;
        // SAFETY: all owners remain retained until source publication completes.
        let mut production = unsafe {
            self.pass.resources.execution.enqueue_production(
                self.requests.borrow().cache(),
                batch.cache()?,
                &self.pass.resources.lanes[gpu],
                None,
            )?
        };
        // SAFETY: the reserved batch owns the source-20 cache proposal.
        while !unsafe {
            self.requests
                .borrow_mut()
                .poll_distributed_encoder_production(batch, &mut production)?
        } {
            tokio::task::yield_now().await;
        }
        drop(production);
        self.pass
            .resources
            .execution
            .finish_decoder_source(batch.cache()?)?;
        self.pass.state = State::Encoded(batch.cache()?.identity());
        Ok(())
    }
    async fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        let batch = self.batch.as_mut().context("encoder unit batch absent")?;
        self.pass.enqueue_cache_commit(
            &self.requests.borrow(),
            batch,
            &[unit.rows.len() as u32],
        )?;
        while !self.pass.poll_cache_commit()? {
            tokio::task::yield_now().await;
        }
        self.pass.commit(
            &mut self.requests.borrow_mut(),
            batch,
            &[unit.rows.len() as u32],
        )?;
        self.batch = None;
        crate::families::deepseek_v41::v41_native_serve::console::totals::prefill(unit.rows.len());
        crate::families::deepseek_v41::v41_native_serve::console::Prefill::done(
            crate::families::deepseek_v41::v41_native_serve::console::PrefillKind::Chunk,
            unit.slot,
            unit.index,
            self.chunks.len(),
            unit.rows.len(),
            self.started,
        );
        Ok(())
    }
}
impl<'w, 'a> DistributedTargetPass<'w, 'a> {
    /// # Safety
    /// Pass, transport, cache and suffix have matching device ownership and
    /// remain alive through completion or drained cancellation.
    pub(in crate::families::deepseek_v41) async unsafe fn execute_encoder_units(
        &mut self,
        requests: &mut Requests<'a>,
        lease: CacheLease,
        chunks: &[&[u32]],
        transport: &mut DeviceOwner<'a, NativeTp4Wave<'a>>,
        suffix: &mut DeviceOwner<'a, EncoderSuffix<'a>>,
        keep_running: &dyn Fn() -> bool,
        before_unit: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        self.set_attention_decode(false);
        let units = super::super::encoder_units::encoder_units(chunks)?;
        ensure!(
            self.state == State::Idle
                && transport.device.id == self.resources.map.attention(20)? as i32,
            "distributed encoder unit owner or state differs"
        );
        let mut hooks = UnitHooks {
            pass: self,
            requests: RefCell::new(requests),
            lease,
            chunks,
            transport,
            suffix,
            keep_running,
            before_unit,
            batch: None,
            prepared: None,
            started: Instant::now(),
        };
        run_prefill_units_async(&mut hooks, &units, 1).await
    }
}
