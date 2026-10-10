//! Native V4.1 expert family: coordinator exchange, local and TP2 waves, paired
//! EXL3 ownership and the dSpark drafter. The layer, resident-weight, execution,
//! host-exchange and EXL3 types live in `shared::experts`; the re-exports below
//! keep the family's paths for one step.
use crate::shared::memory::{DeviceAllocation, LoadStream};
pub(crate) mod coordinator;
pub(crate) mod paired;
pub(crate) mod local;
pub(crate) mod tp2;
pub(crate) mod tp2_ffn;
pub(crate) mod dspark;
pub(crate) use crate::shared::experts::exl3;
pub(crate) use crate::shared::experts::execution::ExpertExecution;
pub(crate) use crate::shared::experts::layer::{ExpertFormat, ExpertLayer, ExpertLoadBudget, ExpertWeights};

use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

/// The dSpark stage FFN on a native expert wave: the drafter's router and
/// shared expert are family code, the routed experts the shared execution.
impl ExpertExecution<'_, '_> {
    /// Run routing, shared FFN and the three routed experts on one wave stream.
    /// The existing reducer adds shared BF16 output after expert accumulation.
    /// # Safety
    /// Hidden states must be finite and initialized with producer writes ordered
    /// on this stream; serialize wave use and finish external readers before reuse.
    pub unsafe fn ffn_draft(
        &mut self,
        router: &mut dspark::DsparkRouter<'_, '_>,
        shared: &mut dspark::DsparkSharedFfn<'_, '_>,
        rows: u32,
    ) -> Result<()> {
        let launched = unsafe { self.enqueue_draft_ffn(router, shared, rows) };
        let drained = self.synchronize();
        launched.and(drained)
    }

    /// Caller must drain this wave's stream before releasing router/shared scratch.
    pub(super) unsafe fn enqueue_draft_ffn(
        &mut self,
        router: &mut dspark::DsparkRouter<'_, '_>,
        shared: &mut dspark::DsparkSharedFfn<'_, '_>,
        rows: u32,
    ) -> Result<()> {
        unsafe { self.enqueue_draft_ffn_on(router, shared, rows, self.stream.raw) }
    }
    /// Containing owner must drain the supplied stream before releasing scratch.
    pub(super) unsafe fn enqueue_draft_ffn_on(
        &mut self,
        router: &mut dspark::DsparkRouter<'_, '_>,
        shared: &mut dspark::DsparkSharedFfn<'_, '_>,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            router.matches(self._weights) && shared.matches(self._weights),
            "dSpark FFN stage owners differ"
        );
        ensure!(
            rows > 0 && rows <= self.kernel.info().capacity_rows,
            "invalid dSpark FFN rows"
        );
        let output = self
            .shared
            .as_ref()
            .context("dSpark FFN requires coordinator output")?
            .buffer;
        unsafe {
            router.enqueue(self.inputs(), rows as usize, stream)?;
            shared.enqueue(self.hidden.buffer, output, rows, stream)?;
            self.launch_on(rows, true, stream)
        }
    }

    /// Compute the dSpark shared expert directly into this wave's shared output.
    /// # Safety
    /// Hidden states must be finite and initialized, with producer writes ordered
    /// on this stream; external consumers must finish before output reuse.
    pub unsafe fn shared_draft(
        &mut self,
        shared: &mut dspark::DsparkSharedFfn<'_, '_>,
        rows: u32,
    ) -> Result<()> {
        ensure!(
            shared.matches(self._weights),
            "shared FFN and expert stage weights differ"
        );
        ensure!(
            rows > 0 && rows <= self.kernel.info().capacity_rows,
            "invalid shared FFN rows"
        );
        let output = self
            .shared
            .as_ref()
            .context("shared FFN requires coordinator output")?
            .buffer;
        let launched = unsafe { shared.enqueue(self.hidden.buffer, output, rows, self.stream.raw) };
        let drained = self.synchronize();
        launched.and(drained)
    }

    /// Route the current hidden states through this exact dSpark stage's gate.
    /// Drains the stream before returning so the router scratch can be reused.
    /// # Safety
    /// Hidden states must be initialized with finite router logits on this device;
    /// external producers/readers must finish or be ordered on this stream.
    pub unsafe fn route_draft(
        &mut self,
        router: &mut dspark::DsparkRouter<'_, '_>,
        rows: u32,
    ) -> Result<()> {
        ensure!(
            router.matches(self._weights),
            "router and expert stage weights differ"
        );
        ensure!(
            rows > 0 && rows <= self.kernel.info().capacity_rows,
            "invalid router rows"
        );
        let launched = unsafe { router.enqueue(self.inputs(), rows as usize, self.stream.raw) };
        let drained = self.synchronize();
        launched.and(drained)
    }

}
