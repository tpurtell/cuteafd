//! MiMo's decode views on the shared lane owners (`shared::serve::decode_lanes`).
//! Concurrent callers get an operation-only handle, never a raw engine view.
use super::{MimoEngine, MimoPlacement};
use crate::shared::serve::decode_lanes::{Aggregate, LaneFlags};
use crate::shared::serve::lanes::LaneSet;
use crate::shared::token_io::{DeviceLogits, RowResult, SelectBatch, TokenSelector};
use anyhow::{Context, Result};
use std::future::Future;

pub(in super::super) use crate::shared::serve::decode_lanes::RoundBank;

/// Own lane one until aggregate retirement proves both views releasable.
/// No Deref or accessor exposes either underlying view to concurrent callers.
pub(in super::super) struct DecodeLanes<'e, 'a> {
    first: &'e MimoEngine<'a>,
    second: Option<MimoEngine<'a>>,
    flags: LaneFlags,
}
impl<'e, 'a> DecodeLanes<'e, 'a> {
    pub(in super::super) fn new(first: &'e MimoEngine<'a>, second: Option<MimoEngine<'a>>) -> Self {
        Self { first, second, flags: LaneFlags::default() }
    }

    /// Capture every admitted view before readiness, including a lane that is
    /// idle at runtime C1. No serving operation can capture a new shape.
    pub(in super::super) fn prepare(&self, probe: bool) -> Result<usize> {
        self.flags.live()?;
        let mut graphs = 0;
        graphs += self.first.capture_decode_graphs(super::DECODE_ROWS)?;
        self.first.check_tp2_routers()?;
        if probe { self.first.arm_draft_probe()?; }
        if let Some(second) = &self.second {
            graphs += second.capture_decode_graphs(super::DECODE_ROWS)?;
            second.check_tp2_decode_routers(self.first)?;
            if probe { second.arm_draft_probe()?; }
            anyhow::ensure!(second.workspace.borrow().is_none()
                && second.prefill_kv_wide.borrow().is_none()
                && second.peer.as_ref().is_none_or(|peer| peer.workspace.borrow().is_none()
                    && peer.prefill_kv_wide.borrow().is_none()),
                "decode-only view acquired an unadmitted prefill allocation");
        }
        Ok(graphs)
    }

    pub(in super::super) fn install_second(&mut self, second: Option<MimoEngine<'a>>) {
        assert!(self.second.is_none(), "second decode view already installed");
        self.second = second;
    }

    pub(in super::super) fn has_second(&self) -> bool { self.second.is_some() }

    pub(in super::super) fn retire(&self) -> Result<()> {
        self.first.terminal_shutdown_views(self.second.as_ref())
    }

    pub(in super::super) fn retain(&self) {
        self.first.retain_serving_storage();
        if let Some(second) = &self.second { second.retain_serving_storage(); }
    }

    /// Publish both views' aborts (device flags only; nothing drains here).
    fn abort(&self) -> Result<()> {
        use crate::shared::{experts::rtx::pair::SubmissionOwners, peer_split::TerminalState};
        let mut errors = Vec::new();
        for view in std::iter::once(self.first).chain(self.second.as_ref()) {
            view.submission_failed.set(true);
            if view.terminal.get() == TerminalState::Active { view.terminal.set(TerminalState::Aborting); }
            if let Err(error) = view.terminal_owners().abort_all() { errors.push(format!("{error:#}")); }
        }
        if !errors.is_empty() {
            self.retain();
            anyhow::bail!("{}", errors.join("; "));
        }
        Ok(())
    }

    /// Run the lane loops (one when no second view is admitted). A failure
    /// publishes both aborts at once; both futures still run to their
    /// boundary, then both views retire together.
    pub(in super::super) async fn run<'s, F, G>(&'s self,
        first: impl FnOnce(DecodeLane<'s, 'e, 'a>) -> F,
        second: impl FnOnce(DecodeLane<'s, 'e, 'a>) -> G) -> Result<()>
    where F: Future<Output = Result<()>>, G: Future<Output = Result<()>> {
        let aggregate = Aggregate::new(&self.flags, || self.abort());
        let result = if self.second.is_some() {
            LaneSet::default().run(|_| aggregate.lane(|| first(DecodeLane { owner: self, index: 0 })),
                |_| aggregate.lane(|| second(DecodeLane { owner: self, index: 1 }))).await
        } else {
            aggregate.lane(|| first(DecodeLane { owner: self, index: 0 })).await
        };
        if result.is_err() {
            aggregate.fail();
            if let Err(error) = self.first.terminal_shutdown_views(self.second.as_ref()) {
                tracing::error!(%error, "failed decode aggregate retained after both lane futures returned");
            }
        }
        result
    }
}
impl Drop for DecodeLanes<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.first.terminal_shutdown_views(self.second.as_ref()) {
            self.first.retain_serving_storage();
            if let Some(second) = self.second.take() {
                second.retain_serving_storage();
                std::mem::forget(second);
            }
            tracing::error!(%error, "retaining both decode views after failed aggregate retirement");
        }
    }
}

pub(in super::super) struct DecodeLane<'s, 'e, 'a> {
    owner: &'s DecodeLanes<'e, 'a>,
    index: usize,
}
impl<'a> DecodeLane<'_, '_, 'a> {
    pub(in super::super) fn stopping(&self) -> bool { self.owner.flags.stopping() }
    pub(in super::super) fn stop(&self) { self.owner.flags.stop(); }
    fn view(&self) -> &MimoEngine<'a> {
        if self.index == 0 { self.owner.first } else { self.owner.second.as_ref().expect("admitted lane one") }
    }

    /// Only the whole-future boundary publishes failure, after its owned
    /// round has restored media and grammar owners to the serving bank.
    fn operation<T>(&self, body: impl FnOnce(&MimoEngine<'a>) -> Result<T>) -> Result<T> {
        self.owner.flags.live()?;
        let view = self.view();
        view.require_live()?;
        body(view)
    }

    pub(in super::super) fn index(&self) -> usize { self.index }
    pub(in super::super) fn library(&self) -> &'a cuteafd_ffi::NativeLibrary { self.view().library }
    pub(in super::super) fn profile(&self) -> [f64; 2] { *self.view().profile.borrow() }
    pub(in super::super) fn probe_begin(&self, rows: usize) { self.view().probe_begin(rows); }
    pub(in super::super) fn probe_end(&self) { self.view().probe_end(); }
    pub(in super::super) fn probe_finish<T>(&self, observe: impl FnOnce(
        &crate::shared::draft::routes::RoundRoutes, &[Option<f64>]) -> T) -> Option<T> {
        self.view().probe_finish(observe)
    }

    pub(in super::super) fn draft(&self, sequences: &[super::super::dflash::DraftSeq])
        -> Result<Vec<super::super::dflash::Draft>> {
        self.operation(|view| view.drafter.as_ref().context("lane has no DFlash drafter")?
            .draft(sequences, &view.embedding, view.head()))
    }

    pub(in super::super) fn sample(&self, selector: &mut TokenSelector<'a>, logits: &DeviceLogits,
        batch: &SelectBatch) -> Result<Vec<RowResult>> {
        self.operation(|_| selector.select(logits, batch))
    }

    pub(in super::super) fn update_context(&self, rows: &[super::super::dflash::ContextRow]) -> Result<()> {
        self.operation(|view| view.drafter.as_ref().context("lane has no DFlash drafter")?.update(rows))
    }

    pub(in super::super) async fn verify(&self, sequences: &mut [(&mut MimoPlacement, usize)], tokens: &[u32])
        -> Result<DeviceLogits> {
        self.owner.flags.live()?;
        self.view().verify_device_async(sequences, tokens).await?
            .context("decode needs every layer")
    }
}

#[cfg(test)]
mod tests {
    use super::RoundBank;

    /// Real request media survives a round and its restore in place (the
    /// generic guard is exercised in `shared::serve::decode_lanes`).
    #[test]
    fn request_media_owners_survive_extraction_and_restore() {
        let mut bank = RoundBank::default();
        bank.install((0..2).map(|_| cuteafd_engine::media::RequestMedia::new(Vec::new(), 1, 0).unwrap()).collect());
        {
            let round = bank.extract(|_| true);
            let mut round = round;
            assert_eq!(round.iter_mut().count(), 2);
        }
        let restored = bank.take_restored();
        assert!(restored.iter().all(|media| media.spans().is_empty()));
    }
}
