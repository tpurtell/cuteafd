//! Slot-owned encoder units. Decode keeps its independent owners and fixed links.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::CacheLease;
use crate::families::deepseek_v41::v41_backbone_execution::{UnitPendingLayer, UnitPreparedLayer};
use crate::families::deepseek_v41::v41_requests::RequestTokens;
use crate::shared::prefill_pipeline::{run_prefill_units_async, AsyncPrefillHooks, FfnRoute, Unit};
use cuteafd_transport::ExpertV2SourceKind;
use std::cell::RefCell;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnitConfig {
    pub waves: usize,
    pub span_rows: usize,
}
impl UnitConfig {
    pub fn new(unit_rows: usize, waves: usize, span_rows: usize) -> Result<Self> {
        ensure!((256..=4096).contains(&unit_rows) && unit_rows % 256 == 0,
            "V4.1 unit rows must be a 256-row multiple in 256..4096");
        ensure!((1..=2).contains(&waves), "V4.1 prefill waves must be 1 or 2");
        ensure!((unit_rows..=1048576).contains(&span_rows) && span_rows % 256 == 0,
            "V4.1 span rows must be a 256-row multiple between unit rows and 1048576");
        Ok(Self { waves, span_rows })
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

struct UnitSlot<'s, 'w, 'a> {
    pass: &'s mut TargetPass<'w, 'a>,
    transport: &'s mut NativeTp4Wave<'a>,
    batch: Option<RequestBatch>,
    prepared: Option<UnitPreparedLayer>,
    started: Instant,
}
struct UnitHooks<'s, 'r, 'w, 'a> {
    slots: Vec<UnitSlot<'s, 'w, 'a>>,
    requests: RefCell<&'r mut Requests<'a>>,
    lease: CacheLease,
    chunks: &'s [&'s [u32]],
    suffix: &'s mut EncoderSuffix<'a>,
    keep_running: &'s dyn Fn() -> bool,
    before_unit: &'s dyn Fn() -> Result<()>,
}
impl Drop for UnitHooks<'_, '_, '_, '_> {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if let Some(mut batch) = slot.batch.take() {
                self.requests.borrow_mut().revoke_batch(&mut batch);
                if let Err(error) = slot.pass.discard(&mut batch) {
                    tracing::error!(%error, "discarding cancelled encoder unit");
                }
            }
        }
    }
}
impl<'a> AsyncPrefillHooks for UnitHooks<'_, '_, '_, 'a> {
    type Ticket = UnitPendingLayer;
    fn route(&self, layer: usize) -> FfnRoute {
        if self.slots[0].transport.has_local_layer(layer) {
            FfnRoute::Local
        } else {
            FfnRoute::Spark
        }
    }
    async fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        ensure!(
            slot.batch.is_none() && slot.prepared.is_none(),
            "encoder unit slot occupied"
        );
        ensure!((self.keep_running)(), "client disconnected");
        (self.before_unit)()?;
        slot.started = Instant::now();
        let chunk = self.chunks[unit.index];
        slot.batch = Some(
            self.requests
                .borrow_mut()
                .reserve_encoder(&[RequestTokens {
                    lease: self.lease,
                    tokens: chunk,
                    image_mask: None,
                    kind: ExpertV2SourceKind::Prefill,
                }])?,
        );
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        slot.pass.state.begin()?;
        slot.pass
            .resources
            .execution
            .restart_for(CacheStage::Encoder);
        slot.pass.resources.lane.restart()?;
        slot.pass.resources.index.restart()?;
        slot.pass.resources.taps.reset();
        // SAFETY: the unit exclusively owns its batch and the pass's input storage.
        unsafe {
            self.requests.borrow_mut().begin_input(
                batch,
                &mut slot.pass.resources.embedding,
                &mut slot.pass.resources.lane,
            )?;
        }
        Ok(())
    }
    async fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        ensure!((self.keep_running)(), "client disconnected");
        ensure!(slot.prepared.is_none(), "encoder attention still pending");
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        // SAFETY: the preceding hook has consumed the previous layer's output.
        unsafe {
            slot.pass
                .prepare_encoder_query(&self.requests, batch, layer)
                .await?;
        }
        // SAFETY: attention completes before its cache/index/lane owners can change.
        let prepared = unsafe {
            self.requests.borrow_mut().prepare_encoder_layer(
                batch,
                &mut slot.pass.resources.execution,
                &mut slot.pass.resources.lane,
                &mut slot.pass.resources.index,
            )?
        };
        slot.prepared = Some(prepared.into_unit().await?);
        Ok(())
    }
    async fn local_ffn(&mut self, _layer: usize, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let prepared = slot.prepared.take().context("encoder attention absent")?;
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: the original bound input remains in the exclusive unit slot.
        let completed = unsafe {
            prepared
                .execute(
                    &mut slot.pass.resources.lane,
                    slot.transport,
                    0,
                    batch.image_mask(),
                )
                .await?
        };
        // SAFETY: the transport result and original unit binding remain live.
        unsafe {
            slot.pass.resources.execution.complete_layer(
                batch.cache()?,
                &mut slot.pass.resources.lane,
                completed,
            )?;
        }
        Ok(())
    }
    async fn send_experts(&mut self, _layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        let slot = &mut self.slots[unit.slot];
        let prepared = slot.prepared.take().context("encoder attention absent")?;
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: the slot is retained until the matching FIFO landing hook.
        unsafe {
            prepared
                .send(
                    &mut slot.pass.resources.lane,
                    slot.transport,
                    0,
                    batch.image_mask(),
                )
                .await
        }
    }
    async fn land_experts(
        &mut self,
        _layer: usize,
        unit: &Unit,
        ticket: Self::Ticket,
    ) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_ref().context("encoder unit batch absent")?;
        // SAFETY: no other unit can overwrite this original slot's shared values.
        let completed = unsafe {
            ticket
                .land(&mut slot.pass.resources.lane, slot.transport)
                .await?
        };
        // SAFETY: result and batch belong to the original prepared layer.
        unsafe {
            slot.pass.resources.execution.complete_layer(
                batch.cache()?,
                &mut slot.pass.resources.lane,
                completed,
            )?;
        }
        Ok(())
    }
    async fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        ensure!((self.keep_running)(), "client disconnected");
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        self.suffix.capture(&slot.pass.resources.lane.output()?)?;
        slot.pass.resources.lane.advance()?;
        // SAFETY: encoder output is complete and this batch owns its source-20 proposal.
        unsafe {
            slot.pass.resources.lane.begin_prepared()?;
            self.requests.borrow_mut().publish_encoder_boundary(
                batch,
                &mut slot.pass.resources.execution,
                &slot.pass.resources.lane,
            )?;
        }
        slot.pass.state = State::Encoded(batch.cache()?.identity());
        Ok(())
    }
    async fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        let slot = &mut self.slots[unit.slot];
        let batch = slot.batch.as_mut().context("encoder unit batch absent")?;
        slot.pass.commit(
            &mut self.requests.borrow_mut(),
            batch,
            &[unit.rows.len() as u32],
        )?;
        slot.batch = None;
        crate::families::deepseek_v41::v41_native_serve::console::totals::prefill(unit.rows.len());
        crate::families::deepseek_v41::v41_native_serve::console::Prefill::done(
            crate::families::deepseek_v41::v41_native_serve::console::PrefillKind::Chunk,
            unit.slot,
            unit.index,
            self.chunks.len(),
            unit.rows.len(),
            slot.started,
        );
        Ok(())
    }
}
impl<'w, 'a> TargetPass<'w, 'a> {
    /// # Safety
    /// Both slots retain their independent lane/TP2/shared/output storage until
    /// completion or drained cancellation. Decode is idle during this span.
    pub(super) async unsafe fn execute_encoder_units(
        &mut self, other: &mut Self, requests: &mut Requests<'a>, lease: CacheLease,
        chunks: &[&[u32]], transports: [&mut NativeTp4Wave<'a>; 2],
        suffix: &mut EncoderSuffix<'a>, keep_running: &dyn Fn() -> bool,
        before_unit: &dyn Fn() -> Result<()>, waves: usize,
    ) -> Result<()> {
        ensure!((1..=2).contains(&waves), "encoder unit owner count differs");
        self.set_attention_decode(false);
        other.set_attention_decode(false);
        let mut units = encoder_units(chunks)?;
        for unit in &mut units { unit.slot = unit.index % waves; }
        let [first, second] = transports;
        for layer in 0..20 {
            ensure!(first.has_local_layer(layer) == second.has_local_layer(layer),
                "encoder slot expert placement differs");
        }
        let mut hooks = UnitHooks {
            slots: vec![
                UnitSlot { pass: self, transport: first, batch: None, prepared: None, started: Instant::now() },
                UnitSlot { pass: other, transport: second, batch: None, prepared: None, started: Instant::now() },
            ],
            requests: RefCell::new(requests), lease, chunks, suffix, keep_running, before_unit,
        };
        run_prefill_units_async(&mut hooks, &units, waves).await
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
    fn typed_unit_configuration_bounds_rows_waves_and_spans() -> Result<()> {
        for rows in [256, 512, 1024, 2048, 4096] {
            for waves in [1, 2] {
                for span in [rows, 16384, 1048576] {
                    assert_eq!(UnitConfig::new(rows, waves, span)?, UnitConfig { waves, span_rows: span });
                }
            }
        }
        for rows in [0, 80, 255, 257, 4097] {
            assert!(UnitConfig::new(rows, 1, 16384).is_err());
        }
        for waves in [0, 3, usize::MAX] {
            assert!(UnitConfig::new(4096, waves, 16384).is_err());
        }
        for span in [0, 512, 4095, 16385, 1048577] {
            assert!(UnitConfig::new(4096, 2, span).is_err());
        }
        Ok(())
    }

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
