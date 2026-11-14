//! One coordinator wave owns TP route planes through final native reduction.
use crate::shared::memory::Tag;
use super::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::expert::{SparkExperts, V41_PARTIAL_ROW_BYTES};

/// A Spark link with V4.1's request contract. Decode lanes each own one;
/// prefill units share one link whose return slots hold the in-flight units
/// (PLAN "Prefill contract": one transport, N waves).
pub(crate) struct V41Link<'a> {
    link: crate::shared::spark_intake::SparkLink<'a>,
}
impl<'a> V41Link<'a> {
    pub(crate) fn new(link: crate::shared::spark_intake::SparkLink<'a>) -> Self {
        Self { link }
    }
    /// The one prefill link: `slots` return slots of `unit_rows` rows.
    pub(crate) fn prefill(library: &'a NativeLibrary, transport: SparkExperts, slots: usize, unit_rows: u32)
        -> Result<Self> {
        let ranks = transport.world_size();
        library.v41_compact_reducer()?.require_rank_count(ranks as u32)?;
        let timing = crate::families::deepseek_v41::v41_native_serve::protocol_v2_timing();
        Ok(Self::new(crate::shared::spark_intake::SparkLink::from_transport_waves(library, transport,
            V41_PARTIAL_ROW_BYTES as usize, crate::shared::spark_intake::IntakePreference::Probed, slots, unit_rows,
            timing)?))
    }
    /// Device bytes of the prefill link's intake: every rank's partial-sum
    /// plane over its `reserve_rows` (slots x unit) rows.
    pub(crate) fn prefill_device_bytes(reserve_rows: usize, ranks: usize,
        format: cuteafd_core::SparkReplyFormat) -> Result<usize> {
        ensure!((1..=8).contains(&ranks) && reserve_rows > 0, "invalid V4.1 prefill link geometry");
        usize::try_from(cuteafd_loader::serving_capacity::prefill_return_bytes_with_format(
            1, reserve_rows as u64, ranks as u64, 5120, format)?).context("prefill link budget overflow")
    }
    pub(crate) fn link(&mut self) -> &mut crate::shared::spark_intake::SparkLink<'a> { &mut self.link }
    pub(crate) fn capacity(&self) -> u32 { self.link.capacity() }
    pub(crate) fn world_size(&self) -> usize { self.link.world_size() }
    pub(crate) fn reset_connections(&mut self) -> Result<()> { self.link.reset_connections() }
    /// Opens reply planes at capacity without naming or executing an expert layer.
    pub(crate) fn bootstrap(&mut self, nvfp4: bool) -> Result<()> {
        let dtype = if nvfp4 { cuteafd_transport::ExpertV2Dtype::Bf16 }
            else { cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32 };
        let mut shape = cuteafd_transport::SparkSessionShape::new(self.link.capacity(), 5120, 6, dtype)?;
        shape.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        self.link.bootstrap(&shape)
    }

}

pub(crate) struct NativeTp4Wave<'a> {
    /// The lane's own link (decode lanes); `None` for a prefill-only slot,
    /// whose units send on the shared prefill link.
    link: Option<V41Link<'a>>,
    // Drop drains the stream before fields release any GPU allocations.
    stream: LoadStream<'a>,
    output: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    ready_rows: Option<u32>,
    local: Option<super::local::LocalExpertWave<'a>>,
    tp2: Option<Box<super::tp2_ffn::Wave<'a>>>,
}

impl<'a> NativeTp4Wave<'a> {
    fn own_link(&mut self) -> Result<&mut V41Link<'a>> {
        self.link.as_mut().context("prefill-only slot has no Spark link of its own")
    }
    pub(crate) fn spark_world(&self) -> usize { self.link.as_ref().map_or(0, V41Link::world_size) }

    pub fn install_local(&mut self, wave: super::local::LocalExpertWave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.local = Some(wave);
        Ok(())
    }
    pub fn has_local_layer(&self, layer: usize) -> bool {
        self.local.as_ref().is_some_and(|wave| wave.contains(layer)) || self.has_tp2_layer(layer)
    }
    pub fn install_tp2(&mut self, wave: super::tp2_ffn::Wave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.tp2 = Some(Box::new(wave));
        Ok(())
    }
    /// Selects a prefill slot's rows in its TP2 FFN's shared planes, at rest
    /// (`shared::memory::row_window`); the reduce output is the slot's own.
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        if let Some(tp2) = self.tp2.as_mut() { tp2.select_window(window)?; }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn tp2_window(&self) -> Option<&crate::shared::memory::row_window::RowWindow> {
        self.tp2.as_ref().map(|tp2| tp2.window())
    }
    pub fn has_tp2_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains(layer))
    }
    pub fn has_tp2_shared_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains_shared(layer))
    }
    /// # Safety
    /// Input and router producers have completed. Both borrowed outputs remain
    /// immutable through this operation, including cancellation draining.
    pub async unsafe fn execute_tp2_ffn(&mut self, input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        ensure!(binding == input.binding() && input.layer == routed.layer
            && input.tokens.len() == routed.rows as usize && input.tokens == routed.tokens,
            "TP2 FFN input/router identity mismatch");
        let values = unsafe { self.tp2.as_mut().context("TP2 expert lane missing")?
            .execute(input.layer, routed.rows, input.values, routed.expert_input, routed.ids, routed.routing).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared buffers stay live and unmodified through drain.
    pub unsafe fn execute_local_ffn(&mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?.execute(routed, shared)? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding: routed.binding()?, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared inputs remain immutable until completion or drain.
    pub async unsafe fn execute_local_ffn_cooperative(&mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?
            .execute_cooperative(routed, shared).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// Admission invalidates prior output; completed RoCE sessions remain reusable.
    pub fn begin_request(&mut self) {
        self.ready_rows = None;
    }
    /// After a cancelled or failed pass whose streams drained: stops the
    /// exchange, resets the connections and starts it again.
    pub fn reset_connections(&mut self) -> Result<()> {
        self.ready_rows = None;
        match self.link.as_mut() { Some(link) => link.reset_connections(), None => Ok(()) }
    }
    /// Legacy TP4 upper-bound reservation (four rank planes), also sufficient
    /// for a legacy TP2 transport.
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        Self::device_bytes_for(capacity, 4)
    }
    /// Reservation for exactly `ranks` physical rank planes plus the final
    /// output plane. Six-rank layouts need their own, larger reservation.
    pub fn device_bytes_for(capacity: u32, ranks: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP wave capacity"
        );
        ensure!(
            (1..=8).contains(&ranks),
            "native TP wave requires one through eight ranks"
        );
        usize::try_from(cuteafd_loader::placement::families::deepseek_v41::owners::transport_bytes(
            u64::from(capacity), ranks)?).context("native TP wave budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        transport: SparkExperts,
        available_bytes: usize,
    ) -> Result<Self> {
        let capacity = transport.capacity();
        let world_size = transport.world_size();
        ensure!(
            (1..=8).contains(&world_size),
            "native TP wave requires one through eight ranks"
        );
        ensure!(
            Self::device_bytes_for(capacity, world_size)? <= available_bytes,
            "native TP wave exceeds device budget"
        );
        let reducer = library.v41_compact_reducer()?;
        // Prove the exact physical-rank reduction is available before any plane
        // or output allocation. Legacy two-rank transports also fail here rather
        // than at their first request.
        reducer.require_rank_count(world_size as u32)?;
        let timing = crate::families::deepseek_v41::v41_native_serve::protocol_v2_timing();
        let link = V41Link::new(crate::shared::spark_intake::SparkLink::from_transport(library, transport,
            V41_PARTIAL_ROW_BYTES as usize, crate::shared::spark_intake::IntakePreference::Probed, timing)?);
        let mut wave = Self::unlinked(library, capacity)?;
        wave.link = Some(link);
        Ok(wave)
    }
    /// A prefill-only slot's FFN owner: its reduce output, stream and local or
    /// TP2 expert workspace, without a Spark link (its units send on the
    /// shared prefill link).
    pub fn unlinked(library: &'a NativeLibrary, capacity: u32) -> Result<Self> {
        ensure!(capacity > 0 && capacity <= 4096, "invalid native TP wave capacity");
        Ok(Self {
            link: None,
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            output: DeviceAllocation::new(library, capacity as usize * 5120 * 2, Tag::workspace("v41/expert-wave"))?,
            library,
            ready_rows: None,
            local: None,
            tp2: None,
        })
    }

    /// Opens the lane's own link ([`V41Link::bootstrap`]).
    pub(crate) fn bootstrap(&mut self, nvfp4: bool) -> Result<()> {
        self.own_link()?.bootstrap(nvfp4)
    }
    /// Prefill slot 0 owns the one prefill link (its non-unit work sends on
    /// it). A units call borrows it for every slot's waves
    /// ([`Self::dispatch_unit`]) and returns it ([`Self::restore_link`]).
    pub(crate) fn take_link(&mut self) -> Result<V41Link<'a>> {
        self.ready_rows = None;
        self.link.take().context("prefill slot 0 has no link to lend")
    }
    pub(crate) fn restore_link(&mut self, link: V41Link<'a>) -> Result<()> {
        ensure!(self.link.is_none(), "prefill slot 0 already holds a link");
        self.link = Some(link);
        Ok(())
    }
    /// A prefill slot 0's FFN owner around the one prefill `link`.
    pub(crate) fn with_link(library: &'a NativeLibrary, link: V41Link<'a>) -> Result<Self> {
        let mut wave = Self::unlinked(library, link.capacity())?;
        wave.link = Some(link);
        Ok(wave)
    }

    /// Sends a remote layer's wave on the link's exchange: the router's route
    /// ids, gate weights and expert input rows copy into the wave's staging on
    /// the router stream (joined to the chain), and the exchange worker builds,
    /// posts and lands it. Nothing waits on the host. Finish with
    /// [`NativePendingFfn::finish`] (or its TP2 form).
    ///
    /// # Safety
    /// The router output was produced in stream order on its own stream (the
    /// chain head or `router_stream`) and stays unchanged until the wave's
    /// staging copies ran, which later work on `router_stream` orders.
    pub(crate) unsafe fn dispatch_ffn<'w>(&'w mut self,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        router_stream: *mut std::ffi::c_void, placement: u64,
        rows: &[crate::families::deepseek_v41::v41_backbone_router::ExpertRow]) -> Result<NativePendingFfn<'w, 'a>> {
        self.ready_rows = None;
        let link = self.link.as_mut().context("prefill-only slot has no Spark link of its own")?;
        let ticket = unsafe { send_on(self.library, &self.stream, link, routed, router_stream, placement, rows)? };
        Ok(NativePendingFfn { ticket: ticket.0, binding: ticket.1, layer: routed.layer, rows: routed.rows,
            library: self.library, stream: &self.stream, link: &mut link.link, output: self.output.buffer,
            ready_rows: &mut self.ready_rows, tp2: self.tp2.as_deref_mut() })
    }
    /// A prefill unit's wave on the shared prefill `link` (its return slots
    /// hold the in-flight units): the returned ticket owns nothing borrowed, so
    /// the unit's later land hook collects it after other units' hooks ran
    /// ([`Self::collect_unit`]); this slot's reduce output and stream stay its own.
    ///
    /// # Safety
    /// As [`Self::dispatch_ffn`].
    pub(crate) unsafe fn dispatch_unit(&mut self, link: &mut V41Link<'a>,
        routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
        router_stream: *mut std::ffi::c_void, placement: u64,
        rows: &[crate::families::deepseek_v41::v41_backbone_router::ExpertRow]) -> Result<NativeUnitFfn> {
        self.ready_rows = None;
        let (ticket, binding) = unsafe { send_on(self.library, &self.stream, link, routed, router_stream, placement, rows)? };
        Ok(NativeUnitFfn { ticket, binding, layer: routed.layer, rows: routed.rows,
            owner: self.output.buffer.ptr as usize })
    }
    /// Lands a unit's wave from the shared prefill `link` (waves land in
    /// dispatch order: the executor's FIFO) and reduces it with its `shared`
    /// values into this slot's output.
    ///
    /// # Safety
    /// `shared` holds the unit's completed shared contribution for exactly this
    /// wave's rows and stays unchanged until the reduce completes.
    pub(crate) async unsafe fn collect_unit<'w>(&'w mut self, link: &'w mut V41Link<'a>, unit: NativeUnitFfn,
        shared: CuteafdDeviceBuffer, destination: i32) -> Result<NativeFfnOutput<'w>> {
        ensure!(unit.owner == self.output.buffer.ptr as usize, "prefill unit ticket belongs to another slot");
        ensure!(shared.device_id == self.output.buffer.device_id && shared.bytes >= unit.rows as usize * 10240,
            "prefill unit shared contribution differs from its wave");
        let mut values = shared;
        values.bytes = unit.rows as usize * 10240;
        let pending = NativePendingFfn { ticket: unit.ticket, binding: unit.binding, layer: unit.layer, rows: unit.rows,
            library: self.library, stream: &self.stream, link: &mut link.link, output: self.output.buffer,
            ready_rows: &mut self.ready_rows, tp2: self.tp2.as_deref_mut() };
        let destination = Some(destination).filter(|&device| device != self.output.buffer.device_id);
        if destination.is_some() {
            let device = crate::shared::memory::device::Device { library: self.library, id: self.output.buffer.device_id };
            device.future(unsafe { pending.finish_values(values, destination) }).await
        } else {
            unsafe { pending.finish_values(values, None).await }
        }
    }
    /// The TP2 shared contribution of a prefill unit, into this transport's
    /// workspace (one transport per in-flight slot), on the reduce GPU.
    /// # Safety
    /// The unit's normalized input stays unchanged until it completes; the
    /// returned values stay unchanged until the unit lands.
    pub(crate) async unsafe fn unit_shared(&mut self,
        input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>) -> Result<CuteafdDeviceBuffer> {
        ensure!(matches!(input.values.device_id, 0 | 1), "TP2 shared input device differs");
        unsafe { self.tp2.as_mut().context("TP2 shared workspace missing")?
            .execute_shared_on(input.layer, input.tokens.len() as u32, input.values,
                self.output.buffer.device_id as usize).await }
    }
    /// Borrowed device view; never free it or retain it across wave reuse/drop.
    pub fn output(&self) -> Result<CuteafdDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("native TP wave output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * 5120 * 2;
        Ok(output)
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }
}
/// Sends a remote layer's wave on `link`'s exchange (see
/// [`NativeTp4Wave::dispatch_ffn`]), staging on `stream` when the router shares
/// the link's GPU.
///
/// # Safety
/// As [`NativeTp4Wave::dispatch_ffn`].
unsafe fn send_on<'a>(library: &'a NativeLibrary, stream: &LoadStream<'a>, link: &mut V41Link<'a>,
    routed: &crate::families::deepseek_v41::v41_backbone_router::RouterOutput<'_>,
    router_stream: *mut std::ffi::c_void, placement: u64,
    rows: &[crate::families::deepseek_v41::v41_backbone_router::ExpertRow])
    -> Result<(crate::shared::spark_waves::WaveTicket, crate::families::deepseek_v41::v41_attention_binding::QueryBinding)> {
    let capacity = link.link.capacity();
    ensure!(routed.rows > 0 && routed.rows <= capacity, "native dispatched FFN rows exceed capacity");
    let (binding, build) = routed.request_builder(placement, rows)?;
    let (layout, sources) = routed.wave_sources()?;
    // Stage on this wave's own stream when the router shares its GPU: the
    // lane's owners (the router among them) then carry nothing past their
    // per-layer host waits, which a host-ordered pass relies on when it
    // rebinds them (`LoadStream::require_complete`). A router on the other
    // GPU (TP2 shared, device-ordered passes) stages from its own stream.
    let stage = if sources[0].device_id == link.link.device() { stream.raw } else { router_stream };
    let exchange = link.link.exchange()?;
    // SAFETY: the sources are the router's live device outputs, complete in
    // chain order: `stage` joins the chain head, which the router's stage
    // finished (caller contract).
    let ticket = unsafe {
        crate::shared::memory::chain::join(library, stage)?;
        exchange.send(routed.rows as usize, exchange.compact_payload(routed.rows as usize), layout, sources, stage,
            crate::shared::spark_waves::combined_reply_eligible(rows.iter().map(|row|
                (row.request_id, row.kind == cuteafd_transport::ExpertV2SourceKind::Prefill))), build)?
    };
    Ok((ticket, binding))
}
impl Drop for NativeTp4Wave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native coordinator TP wave");
        }
    }
}

/// Complete ordered TP reduction plus the shared expert, borrowed until consumed.
pub(crate) struct NativeFfnOutput<'a> {
    pub values: CuteafdDeviceBuffer,
    binding: crate::families::deepseek_v41::v41_attention_binding::QueryBinding,
    _owner: std::marker::PhantomData<&'a ()>,
}
impl NativeFfnOutput<'_> {
    pub fn binding(&self) -> crate::families::deepseek_v41::v41_attention_binding::QueryBinding {
        self.binding
    }
}

/// A prefill unit's wave between its send and land hooks: owned, so no
/// transport borrow survives a hook. Dropping it unlanded leaves the wave
/// outstanding; the pass fails and resets the link after its streams drain.
pub(crate) struct NativeUnitFfn {
    ticket: crate::shared::spark_waves::WaveTicket,
    binding: crate::families::deepseek_v41::v41_attention_binding::QueryBinding,
    layer: usize,
    rows: u32,
    owner: usize,
}

/// A remote layer's wave on the exchange, borrowed until its reduction is
/// queued. Dropping it unfinished leaves the wave unlanded: the owner's pass
/// fails and resets the link (stopping the exchange) after its streams drain.
pub(crate) struct NativePendingFfn<'w, 'a> {
    ticket: crate::shared::spark_waves::WaveTicket,
    binding: crate::families::deepseek_v41::v41_attention_binding::QueryBinding,
    layer: usize,
    rows: u32,
    library: &'a NativeLibrary,
    stream: &'w LoadStream<'a>,
    link: &'w mut crate::shared::spark_intake::SparkLink<'a>,
    output: CuteafdDeviceBuffer,
    ready_rows: &'w mut Option<u32>,
    tp2: Option<&'w mut super::tp2_ffn::Wave<'a>>,
}
impl<'w> NativePendingFfn<'w, '_> {
    /// # Safety
    /// Shared values hold the completed contribution for this exact request and
    /// remain immutable until the final reduction completes on the stream.
    /// Every wait yields to the executor; cancellation drains before releasing frames.
    pub async unsafe fn finish(self,
        shared: &crate::families::deepseek_v41::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'w>> {
        ensure!(self.binding == shared.binding()? && self.layer == shared.layer && self.rows == shared.rows
            && shared.values.bytes == self.rows as usize * 10240 && shared.values.device_id == self.output.device_id,
            "native TP shared contribution differs from routed request");
        unsafe { self.finish_values(shared.values, None).await }
    }
    /// Run shared TP2 after the Spark dispatch, reduce on the transport GPU, and
    /// return the result to the block's GPU. The pending owner retains every lane workspace.
    /// # Safety
    /// Input is the completed normalized FFN input for the dispatched request;
    /// its storage remains immutable through completion or cancellation drain.
    pub async unsafe fn finish_tp2(mut self, input: &crate::families::deepseek_v41::v41_block::FfnInput<'_>) -> Result<NativeFfnOutput<'w>> {
        ensure!(self.binding == input.binding() && self.layer == input.layer
            && self.rows as usize == input.tokens.len() && matches!(input.values.device_id, 0 | 1),
            "TP2 shared input/request/device differs");
        let values = unsafe { self.tp2.as_mut().context("TP2 shared workspace missing")?
            .execute_shared_on(input.layer, self.rows, input.values, self.output.device_id as usize).await? };
        let destination = input.values.device_id;
        if destination != self.output.device_id {
            let device = crate::shared::memory::device::Device { library: self.library, id: self.output.device_id };
            device.future(unsafe { self.finish_values(values, Some(destination)) }).await
        } else {
            unsafe { self.finish_values(values, Some(destination)).await }
        }
    }
    async unsafe fn finish_values(mut self, values: CuteafdDeviceBuffer, destination: Option<i32>) -> Result<NativeFfnOutput<'w>> {
        ensure!(values.device_id == self.output.device_id && values.bytes == self.rows as usize * 10240,
            "shared reduction device/extent differs");
        let timing = std::time::Instant::now();
        let rows = self.rows;
        let slot = self.ticket.slot();
        let exchange = self.link.exchange()?;
        // Both execution modes queue the shared exchange's device wait. Its
        // per-rank pulls preserve early PCIe overlap without a host poll loop.
        // Non-deferred passes still drain the final reduction below before
        // reusing pinned staging or rebinding per-layer owners.
        let host_ordered = !crate::shared::memory::chain::deferred();
        // SAFETY: the reduce stream joins the chain (the shared contribution's
        // producers), then waits on the device for this wave's planes; the
        // intake planes, shared values and output stay live until the stream
        // completes (the pass drains before releasing them).
        unsafe {
            crate::shared::memory::chain::join(self.library, self.stream.raw)?;
            exchange.land(self.ticket, self.stream.raw)?;
            exchange.reduce(slot, values.ptr.cast(), self.output.ptr.cast(), rows as usize, self.stream.raw)?;
        }
        // Inside a chain this records the head; unchained (prefill) it yields
        // until the reduce completes instead of synchronizing the thread.
        // SAFETY: the pending owner retains every input until completion or cancellation.
        unsafe { crate::shared::memory::chain::finish_cooperative(self.stream).await? };
        // The reduce read the TP2 shared output (when this lane computes it):
        // release it now that the reduce is on the chain head.
        if let Some(tp2) = self.tp2.as_deref() {
            tp2.release_shared_output(values.device_id as usize)?;
        }
        if host_ordered && crate::shared::memory::chain::active() {
            // The layer's one host wait for its streams: the reduce joined the
            // chain head (attention, router, shared FFN), so its completion is
            // what the next layer's rebinds check (`require_complete`).
            // SAFETY: this pending owner retains the reduction's queued storage.
            unsafe { crate::shared::memory::chain::finish_or_wait(self.stream, false).await?; }
        }
        tracing::debug!(target: "cuteafd::timing", layer = self.layer, rows,
            collect_us = timing.elapsed().as_micros() as u64, "target collection");
        *self.ready_rows = Some(rows);
        let mut values = self.output;
        values.bytes = rows as usize * 10240;
        if let Some(device) = destination.filter(|&device| device != values.device_id) {
            values = unsafe { self.tp2.as_mut().context("TP2 return workspace missing")?
                .return_result(values, device as usize, rows).await? };
        }
        Ok(NativeFfnOutput {
            values,
            binding: self.binding,
            _owner: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod upload_tests {
    use super::*;

    #[test]
    fn tp_wave_budget_remains_tp4_upper_bound() -> Result<()> {
        assert!(NativeTp4Wave::device_bytes(0).is_err());
        assert!(NativeTp4Wave::device_bytes(4097).is_err());
        for capacity in [1, 16, 80, 4096] {
            let reserved = NativeTp4Wave::device_bytes(capacity)?;
            assert_eq!(reserved as u64, 4 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(capacity as u64 * 10240)?
                + capacity as u64 * 2 * 10240);
            assert!(reserved >= capacity as usize * (2 * V41_PARTIAL_ROW_BYTES as usize + 2 * 10240));
        }
        Ok(())
    }

}
