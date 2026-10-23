//! Opt-in single-owner encoder units. Decode keeps its independent owners.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::CacheLease;
use crate::families::deepseek_v41::v41_backbone_execution::{UnitPendingLayer, UnitPreparedLayer};
use crate::families::deepseek_v41::v41_requests::RequestTokens;
use crate::shared::prefill_pipeline::{run_prefill_units_async, AsyncPrefillHooks, FfnRoute, Unit};
use cuteafd_transport::ExpertV2SourceKind;
use std::cell::RefCell;

pub(in crate::families::deepseek_v41) fn enabled() -> Result<bool> {
    match std::env::var("CUTEAFD_PREFILL_WAVES").as_deref() {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok("1") => Ok(true),
        _ => {
            anyhow::bail!("V4.1 currently supports PREFILL_WAVES=1; omit it for the existing path")
        }
    }
}

pub(super) fn encoder_units(chunks: &[&[u32]]) -> Result<Vec<Unit>> {
    ensure!(!chunks.is_empty(), "empty encoder unit span");
    let mut start = 0;
    chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| {
            ensure!(
                !chunk.is_empty() && chunk.len() <= 4096,
                "encoder unit capacity differs"
            );
            let end = start + chunk.len();
            let unit = Unit {
                index,
                sequence: 0,
                rows: start..end,
                layers: 0..20,
                slot: 0,
                barrier: false,
            };
            start = end;
            Ok(unit)
        })
        .collect()
}

pub(in crate::families::deepseek_v41) fn decoder_unit(index: usize, rows: std::ops::Range<usize>) -> Result<Unit> {
    ensure!(!rows.is_empty() && rows.len() <= 128, "decoder replay suffix capacity differs");
    Ok(Unit { index, sequence: 0, rows, layers: 20..40, slot: 0, barrier: true })
}

struct UnitHooks<'s, 'r, 'w, 'a> {
    pass: &'s mut TargetPass<'w, 'a>,
    requests: RefCell<&'r mut Requests<'a>>,
    lease: CacheLease,
    chunks: &'s [&'s [u32]],
    transport: &'s mut NativeTp4Wave<'a>,
    suffix: &'s mut EncoderSuffix<'a>,
    keep_running: &'s dyn Fn() -> bool,
    before_unit: &'s dyn Fn() -> Result<()>,
    batch: Option<RequestBatch>,
    prepared: Option<UnitPreparedLayer>,
    started: Instant,
}
impl Drop for UnitHooks<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if let Some(mut batch) = self.batch.take() {
            self.requests.borrow_mut().revoke_batch(&mut batch);
            if let Err(error) = self.pass.discard(&mut batch) {
                tracing::error!(%error, "discarding cancelled encoder unit");
            }
        }
    }
}
impl<'a> AsyncPrefillHooks for UnitHooks<'_, '_, '_, 'a> {
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
            "serial encoder unit slot occupied"
        );
        ensure!((self.keep_running)(), "client disconnected");
        (self.before_unit)()?;
        self.started = Instant::now();
        let chunk = self.chunks[unit.index];
        self.batch = Some(
            self.requests
                .borrow_mut()
                .reserve_encoder(&[RequestTokens {
                    lease: self.lease,
                    tokens: chunk,
                    image_mask: None,
                    kind: ExpertV2SourceKind::Prefill,
                }])?,
        );
        let batch = self.batch.as_mut().context("encoder unit batch absent")?;
        self.pass.state.begin()?;
        self.pass
            .resources
            .execution
            .restart_for(CacheStage::Encoder);
        self.pass.resources.lane.restart()?;
        self.pass.resources.index.restart()?;
        self.pass.resources.taps.reset();
        // SAFETY: the unit exclusively owns its batch and the pass's input storage.
        unsafe {
            self.requests.borrow_mut().begin_input(
                batch,
                &mut self.pass.resources.embedding,
                &mut self.pass.resources.lane,
            )?;
        }
        Ok(())
    }
    async fn attention(&mut self, layer: usize, _unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        ensure!(self.prepared.is_none(), "encoder attention still pending");
        let batch = self.batch.as_mut().context("encoder unit batch absent")?;
        // SAFETY: the preceding hook has consumed the previous layer's output.
        unsafe {
            self.pass
                .prepare_encoder_query(&self.requests, batch, layer)
                .await?;
        }
        // SAFETY: attention completes before its cache/index/lane owners can change.
        let prepared = unsafe {
            self.requests.borrow_mut().prepare_encoder_layer(
                batch,
                &mut self.pass.resources.execution,
                &mut self.pass.resources.lane,
                &mut self.pass.resources.index,
            )?
        };
        self.prepared = Some(prepared.into_unit().await?);
        Ok(())
    }
    async fn local_ffn(&mut self, _layer: usize, _unit: &Unit) -> Result<()> {
        let prepared = self.prepared.take().context("encoder attention absent")?;
        let batch = self.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: the original bound input remains in the exclusive unit slot.
        let completed = unsafe {
            prepared
                .execute(
                    &mut self.pass.resources.lane,
                    self.transport,
                    0,
                    batch.image_mask(),
                )
                .await?
        };
        // SAFETY: the transport result and original unit binding remain live.
        unsafe {
            self.pass.resources.execution.complete_layer(
                batch.cache()?,
                &mut self.pass.resources.lane,
                completed,
            )?;
        }
        Ok(())
    }
    async fn send_experts(&mut self, _layer: usize, _unit: &Unit) -> Result<Self::Ticket> {
        let prepared = self.prepared.take().context("encoder attention absent")?;
        let batch = self.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: the slot is retained until the matching FIFO landing hook.
        unsafe {
            prepared
                .send(
                    &mut self.pass.resources.lane,
                    self.transport,
                    0,
                    batch.image_mask(),
                )
                .await
        }
    }
    async fn land_experts(
        &mut self,
        _layer: usize,
        _unit: &Unit,
        ticket: Self::Ticket,
    ) -> Result<()> {
        let batch = self.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: no other unit can overwrite this serial slot's shared values.
        let completed = unsafe {
            ticket
                .land(&mut self.pass.resources.lane, self.transport)
                .await?
        };
        // SAFETY: result and batch belong to the original prepared layer.
        unsafe {
            self.pass.resources.execution.complete_layer(
                batch.cache()?,
                &mut self.pass.resources.lane,
                completed,
            )?;
        }
        Ok(())
    }
    async fn end_unit(&mut self, _unit: &Unit) -> Result<()> {
        ensure!((self.keep_running)(), "client disconnected");
        let batch = self.batch.as_mut().context("encoder unit batch absent")?;
        self.suffix.capture(&self.pass.resources.lane.output()?)?;
        self.pass.resources.lane.advance()?;
        // SAFETY: encoder output is complete and this batch owns its source-20 proposal.
        unsafe {
            self.pass.resources.lane.begin_prepared()?;
            self.requests.borrow_mut().publish_encoder_boundary(
                batch,
                &mut self.pass.resources.execution,
                &self.pass.resources.lane,
            )?;
        }
        self.pass.state = State::Encoded(batch.cache()?.identity());
        Ok(())
    }
    async fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        let batch = self.batch.as_mut().context("encoder unit batch absent")?;
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
impl<'w, 'a> TargetPass<'w, 'a> {
    /// # Safety
    /// The single pass and transport share a CUDA owner and retain all unit
    /// storage until completion or drained cancellation. Decode owners are idle.
    pub(super) async unsafe fn execute_encoder_units(
        &mut self,
        requests: &mut Requests<'a>,
        lease: CacheLease,
        chunks: &[&[u32]],
        transport: &mut NativeTp4Wave<'a>,
        suffix: &mut EncoderSuffix<'a>,
        keep_running: &dyn Fn() -> bool,
        before_unit: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        self.set_attention_decode(false);
        let units = encoder_units(chunks)?;
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

impl<'a> crate::families::deepseek_v41::v41_native_serve::prefill_target::ReplayTarget<'a>
    for TargetPass<'_, 'a> {
    type Suffix = EncoderSuffix<'a>;
    async unsafe fn begin_replay_unit(&mut self, requests: &Requests<'a>,
        batch: &RequestBatch, suffix: &Self::Suffix) -> Result<()> {
        requests.validate(batch)?;
        ensure!(batch.cache()?.stage() == CacheStage::Replay, "decoder replay stage differs");
        let encoder = suffix.output()?;
        ensure!(encoder.tokens == batch.cache()?.positions(), "encoder suffix/replay row order differs");
        self.set_attention_decode(false);
        self.resources.lane.set_l2_prefetch(std::mem::replace(&mut self.l2_solo, true));
        self.state.begin()?;
        self.sampled = None;
        self.resources.taps.begin(batch.cache()?)?;
        self.resources.execution.restart_for(CacheStage::Replay);
        self.resources.lane.restart_decoder(&encoder)?;
        self.resources.index.restart_decoder()?;
        // SAFETY: retain the original suffix and prepared lane through every replay hook.
        unsafe { self.resources.lane.begin_prepared()?; }
        Ok(())
    }
    async unsafe fn replay_attention(&mut self, requests: &Requests<'a>,
        batch: &RequestBatch, layer: usize) -> Result<UnitPreparedLayer> {
        ensure!((20..40).contains(&layer), "decoder replay layer differs");
        if layer != 20 {
            self.resources.lane.advance()?;
            if layer >= 37 {
                // SAFETY: the previous layer completed and its output is immutable until preparation.
                unsafe { self.resources.taps.capture(batch.cache()?, &self.resources.lane.prepared_input()?)?; }
            }
            // SAFETY: the previous layer's output is retained until its query completes.
            unsafe { self.resources.lane.begin_prepared()?; }
        }
        // SAFETY: preserve the ordinary replay's synchronous producer/index/attention route.
        let prepared = unsafe { self.resources.execution.prepare_layer(requests.cache(), batch.cache()?,
            &mut self.resources.lane, &mut self.resources.index)? };
        prepared.into_unit().await
    }
    async unsafe fn replay_local_ffn(&mut self, batch: &RequestBatch,
        transport: &mut NativeTp4Wave<'a>, _layer: usize, prepared: UnitPreparedLayer) -> Result<()> {
        // SAFETY: this barrier exclusively owns the original query and FFN result storage.
        let completed = unsafe { prepared.execute(&mut self.resources.lane, transport, 0, batch.image_mask()).await? };
        // SAFETY: the completed FFN carries this replay batch's binding.
        unsafe { self.resources.execution.complete_layer(batch.cache()?, &mut self.resources.lane, completed) }
    }
    async unsafe fn send_replay_experts(&mut self, batch: &RequestBatch,
        transport: &mut NativeTp4Wave<'a>, _layer: usize, prepared: UnitPreparedLayer)
        -> Result<UnitPendingLayer> {
        // SAFETY: retain this lane's shared/input values until the matching landing hook.
        unsafe { prepared.send(&mut self.resources.lane, transport, 0, batch.image_mask()).await }
    }
    async unsafe fn land_replay_experts(&mut self, batch: &RequestBatch,
        transport: &mut NativeTp4Wave<'a>, _layer: usize, ticket: UnitPendingLayer) -> Result<()> {
        // SAFETY: FIFO transport and lane storage remain owned by this one barrier unit.
        let completed = unsafe { ticket.land(&mut self.resources.lane, transport).await? };
        // SAFETY: complete the original bound query without changing the replay's arithmetic order.
        unsafe { self.resources.execution.complete_layer(batch.cache()?, &mut self.resources.lane, completed) }
    }
    fn replay_route(&self, transport: &NativeTp4Wave<'a>, layer: usize) -> FfnRoute {
        if transport.has_local_layer(layer) { FfnRoute::Local } else { FfnRoute::Spark }
    }
    async unsafe fn end_replay_unit(&mut self, lib: &'a cuteafd_ffi::NativeLibrary,
        batch: &RequestBatch, selected: &[usize]) -> Result<Vec<u8>> {
        self.resources.taps.output(batch.cache()?)?;
        let output = self.resources.lane.output()?;
        // SAFETY: this is the ordinary replay head terminal on the same completed final output.
        unsafe { self.resources.head.execute_block(&output, selected)?; }
        self.state = State::Ready(batch.cache()?.identity());
        let logits = self.resources.head.output()?;
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_pipeline::{prefill_schedule, Step};

    #[test]
    fn suffix_replay_waits_for_every_encoder_commit() -> Result<()> {
        let full = vec![0; 4096];
        let tail = vec![0; 257];
        for waves in 1..=3 {
            let mut units = encoder_units(&[&full, &full, &full, &tail])?;
            for unit in &mut units { unit.slot = unit.index % waves; }
            let replay = units.len();
            units.push(decoder_unit(replay, 12417..12545)?);
            for mask in [0, (1u64 << 40) - 1, 0x5555555555, 0xaaaaaaaaaa, (1u64 << 20) - 1] {
                let routes = (0..40).map(|layer| if mask & (1 << layer) == 0 {
                    FfnRoute::Spark
                } else { FfnRoute::Local }).collect::<Vec<_>>();
                let schedule = prefill_schedule(&routes, &units, waves)?;
                let mut committed = 0;
                let mut pending = std::collections::VecDeque::new();
                let mut replay_layers = Vec::new();
                for step in schedule {
                    match step {
                        Step::Begin { unit } if unit == replay => {
                            assert_eq!(committed, replay, "replay entered before all encoder/source commits");
                            assert!(pending.is_empty());
                        }
                        Step::Attention { unit, layer } if unit == replay => replay_layers.push(layer),
                        Step::Send { unit, layer } => {
                            pending.push_back((unit, layer));
                            assert!(pending.len() <= waves);
                        }
                        Step::Land { unit, layer } => assert_eq!(pending.pop_front(), Some((unit, layer))),
                        Step::End { unit } if unit == replay => {
                            assert_eq!(replay_layers, (20..40).collect::<Vec<_>>());
                            assert!(pending.is_empty());
                        }
                        Step::Commit { unit } => {
                            assert_eq!(unit, committed);
                            committed += 1;
                        }
                        _ => {}
                    }
                }
                assert_eq!(committed, units.len());
            }
        }
        assert!(decoder_unit(0, 0..0).is_err());
        assert!(decoder_unit(0, 0..129).is_err());
        Ok(())
    }

    #[test]
    fn encoder_units_preserve_cuts_and_ced_layer_barrier() -> Result<()> {
        let first = vec![0; 4096];
        let tail = vec![0; 257];
        let units = encoder_units(&[&first, &first, &tail])?;
        assert_eq!(
            units.iter().map(|u| u.rows.clone()).collect::<Vec<_>>(),
            [0..4096, 4096..8192, 8192..8449]
        );
        assert!(units
            .iter()
            .all(|u| u.layers == (0..20) && u.slot == 0 && !u.barrier));
        for mask in [0, 1, (1u32 << 20) - 1, 0x55555, 0xaaaaa] {
            let routes = (0..20)
                .map(|layer| {
                    if mask & (1 << layer) != 0 {
                        FfnRoute::Local
                    } else {
                        FfnRoute::Spark
                    }
                })
                .collect::<Vec<_>>();
            let schedule = prefill_schedule(&routes, &units, 1)?;
            let mut committed = 0;
            let mut pending = None;
            for step in schedule {
                match step {
                    Step::Begin { unit } => assert_eq!(unit, committed),
                    Step::Send { unit, layer } => assert!(pending.replace((unit, layer)).is_none()),
                    Step::Land { unit, layer } => assert_eq!(pending.take(), Some((unit, layer))),
                    Step::Commit { unit } => {
                        assert_eq!(unit, committed);
                        committed += 1;
                    }
                    _ => {}
                }
            }
            assert_eq!(committed, units.len());
            assert!(pending.is_none());
        }
        assert!(encoder_units(&[]).is_err());
        assert!(encoder_units(&[&[]]).is_err());
        assert!(encoder_units(&[&vec![0; 4097]]).is_err());
        Ok(())
    }
}
