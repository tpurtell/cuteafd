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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_pipeline::{prefill_schedule, Step};

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
