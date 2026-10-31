//! Paired routed execution, canonical routes and fused payloads on serialized rank streams.
use crate::shared::experts::exl3::execution::Exl3Modules;
use super::{
    ExchangeDtype, ExpertInput, FusedCombine, PartialDtype, RouteCheck, RouteIdentity, RouteSource,
    Routes, RtxExpertLayer, RtxShard, routes::SlotRouteChecks,
};
use crate::shared::{
    memory::HostAllocation,
    peer_split::{PeerExchange, RankDevice},
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    ffi::c_void,
};

/// Full submission boundary, including captured graph replays and caller-side
/// exchanges. Shutdown must publish aborts before draining and retire the engine;
/// A failed drain may return only if the caller retains every queued owner;
/// callers without quarantine ownership must terminate in shutdown instead.
#[derive(Debug, thiserror::Error)]
#[error("submission failed permanently: {cause:#}; {drain}")]
pub(crate) struct TerminalStepError {
    #[source]
    cause: anyhow::Error,
    drain: String,
}

/// Families supply owners; this sequence is also exercised by CPU fault tests.
pub(crate) trait SubmissionOwners {
    fn abort_all(&mut self) -> Result<()>;
    fn drain_ranks(&mut self) -> Result<()>;
    fn quiesce_links(&mut self) -> Result<()>;
    fn drain_copies(&mut self) -> Result<()>;
    fn retire_graphs(&mut self) -> Result<()>;
    fn finish(&mut self, drained: bool);
}

pub(crate) struct Teardown<A, D, Q, C, R, F> {
    pub abort: A,
    pub ranks: D,
    pub links: Q,
    pub copies: C,
    pub graphs: R,
    pub finish: F,
}
impl<A, D, Q, C, R, F> SubmissionOwners for Teardown<A, D, Q, C, R, F>
where
    A: FnMut() -> Result<()>,
    D: FnMut() -> Result<()>,
    Q: FnMut() -> Result<()>,
    C: FnMut() -> Result<()>,
    R: FnMut() -> Result<()>,
    F: FnMut(bool),
{
    fn abort_all(&mut self) -> Result<()> {
        (self.abort)()
    }
    fn drain_ranks(&mut self) -> Result<()> {
        (self.ranks)()
    }
    fn quiesce_links(&mut self) -> Result<()> {
        (self.links)()
    }
    fn drain_copies(&mut self) -> Result<()> {
        (self.copies)()
    }
    fn retire_graphs(&mut self) -> Result<()> {
        (self.graphs)()
    }
    fn finish(&mut self, drained: bool) {
        (self.finish)(drained)
    }
}

pub(crate) fn teardown(owners: &mut impl SubmissionOwners) -> Result<()> {
    let mut failures = Vec::new();
    let aborted = match owners.abort_all() {
        Ok(()) => true,
        Err(error) => {
            failures.push(format!("abort publication: {error:#}"));
            false
        }
    };
    // Failed abort publication cannot safely unblock a missing producer.
    let compute = if aborted {
        match owners.drain_ranks() {
            Ok(()) => true,
            Err(error) => {
                failures.push(format!("rank drainage: {error:#}"));
                false
            }
        }
    } else {
        false
    };
    let quiesced = match owners.quiesce_links() {
        Ok(()) => true,
        Err(error) => {
            failures.push(format!("link quiescence: {error:#}"));
            false
        }
    };
    // Copy streams can depend on compute events; never drain an unproved dependency.
    if compute && quiesced {
        if let Err(error) = owners.drain_copies() {
            failures.push(format!("copy drainage: {error:#}"));
        }
    }
    if failures.is_empty() {
        if let Err(error) = owners.retire_graphs() {
            failures.push(format!("graph retirement: {error:#}"));
        }
    }
    owners.finish(failures.is_empty());
    ensure!(
        failures.is_empty(),
        "queued owners retained: {}",
        failures.join("; ")
    );
    Ok(())
}

struct SubmissionGuard<S: FnOnce() -> Result<()>>(Option<S>);
impl<S: FnOnce() -> Result<()>> SubmissionGuard<S> {
    fn shutdown(&mut self) -> Result<()> {
        self.0.take().expect("submission already retired")()
    }
}
impl<S: FnOnce() -> Result<()>> Drop for SubmissionGuard<S> {
    fn drop(&mut self) {
        if let Some(shutdown) = self.0.take() {
            // Cancellation/unwinding must obey the same owner contract as enqueue errors.
            if let Err(error) = shutdown() {
                tracing::error!(%error, "cancelled submission retains queued owners");
            }
        }
    }
}

/// Unforgeable, borrowed proof that a whole-submission teardown guard is live.
/// Executors cannot construct or retain it beyond the guarded body.
pub(crate) struct SubmissionScope<'a> {
    _guard: std::marker::PhantomData<&'a ()>,
}
impl<S: FnOnce() -> Result<()>> SubmissionGuard<S> {
    fn scope(&self) -> SubmissionScope<'_> {
        SubmissionScope {
            _guard: std::marker::PhantomData,
        }
    }
}

pub(crate) fn submit<T>(
    require_live: impl FnOnce() -> Result<()>,
    shutdown: impl FnOnce() -> Result<()>,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    submit_scoped(require_live, shutdown, |_| body())
}

pub(crate) fn submit_scoped<T>(
    require_live: impl FnOnce() -> Result<()>,
    shutdown: impl FnOnce() -> Result<()>,
    body: impl FnOnce(&SubmissionScope<'_>) -> Result<T>,
) -> Result<T> {
    require_live()?;
    let mut guard = SubmissionGuard(Some(shutdown));
    match body(&guard.scope()) {
        Ok(value) => {
            guard.0.take();
            Ok(value)
        }
        Err(cause) => {
            // Teardown may abort to retain unsafe-to-release owners. Preserve
            // the submission error before that fail-safe can hide its cause.
            tracing::error!(error = %format!("{cause:#}"), "submission failed; draining queued owners");
            let drain = match guard.shutdown() {
                Ok(()) => "queued streams drained; engine cannot be reused".to_string(),
                Err(error) => format!("queued owners retained: {error:#}"),
            };
            Err(TerminalStepError { cause, drain }.into())
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedRoutes {
    pub section: usize,
    pub weights: usize,
    pub bytes: usize,
}
impl PackedRoutes {
    pub(crate) fn new(rows: usize, topk: usize) -> Self {
        let section = rows * topk * 4;
        let weights = section.next_multiple_of(16);
        Self {
            section,
            weights,
            bytes: (weights + section).next_multiple_of(16),
        }
    }
}

/// Who publishes a layer's canonical routes (and, for a source-only router,
/// its FP8 K32 wire rows) to the peer.
#[derive(Debug, Clone, Copy)]
struct Broadcast {
    source: usize,
    wire: bool,
}

pub(crate) struct Pair<'a> {
    pub layers: Vec<usize>,
    /// FP8 K32 wire bytes per row a source broadcast carries; 0 when routes only.
    wire_row_bytes: usize,
    ranks: [RefCell<Box<dyn RtxExpertLayer + 'a>>; 2],
    pub combine: FusedCombine<'a>,
    pub routes: PeerExchange<'a>,
    source: Cell<RouteSource>,
    /// Per-slot max-row pinned route copies per layer, allocated before a
    /// call's first peer wait and retained (never replaced) across units.
    checks: RefCell<Option<SlotRouteChecks<BTreeMap<usize, [HostAllocation<'a>; 2]>>>>,
    /// The unit each slot's live diagnostics belong to (`enqueue` knows only the slot).
    active: RefCell<Vec<Option<usize>>>,
    checking: Cell<bool>,
    lanes: usize,
    max_rows: usize,
    interval: usize,
    steps: Cell<usize>,
    library: &'a NativeLibrary,
    devices: [RankDevice; 2],
    topk: usize,
    hidden: usize,
}

impl<'a> Pair<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        library: &'a NativeLibrary,
        devices: [RankDevice; 2],
        ranks: [Box<dyn RtxExpertLayer + 'a>; 2],
        hidden: usize,
        topk: usize,
        max_rows: usize,
        lanes: usize,
        exchange: ExchangeDtype,
    ) -> Result<Self> {
        Self::new_with_wire(library, devices, ranks, hidden, topk, max_rows, lanes, exchange, false)
    }

    /// As [`Self::new`]; with `wire`, each route slot also holds `max_rows`
    /// FP8 K32 wire rows, so a layer whose router lives on one rank can
    /// publish its routes and input to the peer ([`Self::enqueue_from`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_wire(
        library: &'a NativeLibrary,
        devices: [RankDevice; 2],
        ranks: [Box<dyn RtxExpertLayer + 'a>; 2],
        hidden: usize,
        topk: usize,
        max_rows: usize,
        lanes: usize,
        exchange: ExchangeDtype,
        wire: bool,
    ) -> Result<Self> {
        let wire_row_bytes = if wire { hidden + hidden / 32 } else { 0 };
        ensure!(!wire || ranks.iter().all(|backend| backend.uses_fp8_input()),
            "a wire broadcast pair runs FP8 K32 input backends");
        let layers = ranks[0].layers().to_vec();
        ensure!(
            !layers.is_empty() && layers.windows(2).all(|ids| ids[0] < ids[1])
                && layers == ranks[1].layers(),
            "TP2 ordered layer inventories differ"
        );
        for (rank, backend) in ranks.iter().enumerate() {
            ensure!(
                backend.shard() == RtxShard::Tp2 { rank: rank as u8 }
                    && backend.device() == devices[rank].device
                    && backend.partial() == ranks[0].partial()
                    && backend.uses_fp8_input() == ranks[0].uses_fp8_input()
                    && (1..=max_rows).all(|rows| backend
                        .partial_layout(rows)
                        .is_ok_and(|layout| layout == super::PartialLayout::Tokens)),
                "TP2 rank {rank} shard/device/partial mismatch"
            );
        }
        ensure!(
            exchange == ExchangeDtype::Bf16 || ranks[0].partial() == PartialDtype::F32,
            "FP32 exchange requires FP32 routed partials"
        );
        let routes = PeerExchange::new_abortable(
            library,
            devices,
            2 * lanes,
            PackedRoutes::new(max_rows, topk).bytes + max_rows * wire_row_bytes,
        )?;
        for (rank, producer) in devices.into_iter().enumerate() {
            routes.require_cooperative_producer_stream(rank, producer)?;
        }
        let interval = std::env::var("CUTEAFD_ROUTE_CHECK")
            .ok()
            .map(|s| s.parse::<usize>())
            .transpose()
            .context("CUTEAFD_ROUTE_CHECK must be a nonnegative step interval")?
            .unwrap_or(0);
        Ok(Self {
            layers,
            wire_row_bytes,
            ranks: ranks.map(RefCell::new),
            combine: FusedCombine::new(library, exchange)?,
            routes,
            source: Cell::new(RouteSource::Replicated),
            checks: RefCell::new(None),
            active: RefCell::new(vec![None; lanes]),
            checking: Cell::new(false),
            lanes,
            max_rows,
            interval,
            steps: Cell::new(0),
            library,
            devices,
            topk,
            hidden,
        })
    }

    pub(crate) fn partial(&self) -> PartialDtype {
        self.ranks[0].borrow().partial()
    }

    pub(crate) fn uses_fp8_input(&self) -> bool {
        self.ranks[0].borrow().uses_fp8_input()
    }

    pub(crate) fn group_rows(&self, rows: usize) -> Result<usize> {
        let group = self.ranks[0].borrow().group_rows(rows)?;
        ensure!(
            self.ranks[1].borrow().group_rows(rows)? == group,
            "installed TP2 ranks disagree on expert grouping"
        );
        Ok(group)
    }

    pub(crate) fn weight_read_bytes(&self) -> Result<usize> {
        let bytes = self
            .ranks
            .iter()
            .map(|rank| rank.borrow().weight_read_bytes())
            .collect::<Result<Vec<_>>>()?;
        Ok(*bytes.iter().max().context("installed TP2 ranks missing")?)
    }

    /// Starts a call's route diagnostics: whether this call checks (`force` or
    /// the CUTEAFD_ROUTE_CHECK interval). When it does, every slot's max-row
    /// pinned storage exists before the caller queues any peer wait; strict
    /// (forced) checks fail on a mismatch, interval checks fall back to broadcast.
    pub(crate) fn start_checks(&self, force: bool) -> Result<bool> {
        let steps = self.steps.get() + 1;
        self.steps.set(steps);
        let check = force || (self.interval > 0 && steps % self.interval == 0);
        self.checking.set(check);
        if !check { return Ok(false); }
        let mut checks = self.checks.borrow_mut();
        // The previous span finished (drained) or failed terminally (the
        // submission teardown drained it), so its storage is released here,
        // before this call queues anything.
        ensure!(self.active.borrow().iter().all(Option::is_none), "undrained TP2 route diagnostics");
        let storage = (0..self.lanes).map(|_| self.layers.iter().map(|&layer| {
            let hosts = (0..2).map(|rank| self.routes.on(rank, ||
                HostAllocation::new(self.library, self.max_rows * self.topk * 8))).collect::<Result<Vec<_>>>()?;
            Ok((layer, hosts.try_into().map_err(|_| anyhow::anyhow!("two ranks"))?))
        }).collect::<Result<BTreeMap<_, _>>>()).collect::<Result<Vec<_>>>()?;
        *checks = Some(SlotRouteChecks::new(storage, force)?);
        Ok(true)
    }

    /// One unit's diagnostics on `slot` (its rows of every TP2 layer).
    pub(crate) fn begin_unit(&self, slot: usize, unit: usize, rows: usize) -> Result<()> {
        if !self.checking.get() { return Ok(()); }
        ensure!(rows <= self.max_rows, "route diagnostic rows {rows} exceed {}", self.max_rows);
        let mut checks = self.checks.borrow_mut();
        checks.as_mut().context("route diagnostics not started")?.begin(slot, unit, rows, true)?;
        *self.active.borrow_mut().get_mut(slot).context("route diagnostic slot")? = Some(unit);
        Ok(())
    }

    /// Drains both producer streams, then compares the unit's routes trimmed
    /// to its rows. A mismatch switches to broadcast (strict: an error).
    pub(crate) fn finish_unit(&self, slot: usize, unit: usize) -> Result<()> {
        if !self.checking.get() { return Ok(()); }
        let mut checks = self.checks.borrow_mut();
        let checks = checks.as_mut().context("route diagnostics not started")?;
        let topk = self.topk;
        let identity = checks.finish(slot, unit, |storage, rows| {
            for rank in 0..2 {
                // SAFETY: every producer of this unit's copies is already queued.
                self.routes.on(rank, || unsafe {
                    self.library.cuda_stream_synchronize(self.devices[rank].stream)
                })?;
            }
            let section = rows * topk * 4;
            let full = self.max_rows * topk * 4;
            Ok(storage.iter().map(|(&layer, hosts)| RouteCheck { layer, rows, topk,
                ranks: hosts.each_ref().map(|h| {
                    let bytes = h.bytes();
                    [&bytes[..section], &bytes[full..full + section]].concat()
                }) }).collect())
        })?;
        *self.active.borrow_mut().get_mut(slot).context("route diagnostic slot")? = None;
        if let Some(identity) = identity { self.publish_identity(&identity); }
        Ok(())
    }

    /// Ends a call's diagnostics; always called, also before propagating an
    /// executor error (strict checks reject a unit that never finished).
    pub(crate) fn finish_span(&self) -> Result<()> {
        if !self.checking.replace(false) { return Ok(()); }
        let checks = self.checks.borrow();
        checks.as_ref().context("route diagnostics not started")?.finish_span()
    }

    /// A single-pass call: each lane is one unit on its own slot.
    pub(crate) fn begin_checks(&self, rows: &[usize], force: bool) -> Result<()> {
        if !self.start_checks(force)? { return Ok(()); }
        for (lane, &rows) in rows.iter().enumerate() { self.begin_unit(lane, lane, rows)?; }
        Ok(())
    }

    pub(crate) fn checking(&self) -> bool {
        self.checking.get()
    }

    /// Completes a single-pass call's lanes, then its span.
    pub(crate) fn finish_checks(&self) -> Result<()> {
        if !self.checking.get() { return Ok(()); }
        let live = self.active.borrow().iter().enumerate()
            .filter_map(|(slot, unit)| unit.map(|unit| (slot, unit))).collect::<Vec<_>>();
        let result = live.into_iter().try_for_each(|(slot, unit)| self.finish_unit(slot, unit));
        let span = self.finish_span();
        result.and(span)
    }

    fn publish_identity(&self, identity: &RouteIdentity) {
        if identity.source() == RouteSource::Broadcast {
            self.source.set(RouteSource::Broadcast);
        }
        tracing::info!(layers_checked = identity.layers_checked, rows_checked = identity.rows_checked,
            first_mismatch = ?identity.first_mismatch, source = ?self.source.get(), "RTX TP2 route identity");
        cuteafd_bench::context::set_resolved(
            "route-source",
            if self.source.get() == RouteSource::Broadcast {
                "broadcast"
            } else {
                "replicated"
            },
        );
    }

    /// Resolve all lazy backend capacities before either rank queues a peer wait.
    /// # Safety
    /// Initialized maximum-capacity input/routes are live on rank's drained startup stream.
    pub(crate) unsafe fn prime(
        &self,
        rank: usize,
        input: ExpertInput,
        routes: Routes,
        rows: usize,
        shared: *const u16,
        out: *mut c_void,
    ) -> Result<()> {
        self.routes.on(rank, || {
            let mut backend = self.ranks[rank].borrow_mut();
            let stream = self.devices[rank].stream;
            // SAFETY: caller owns initialized startup buffers; no peer waits exist.
            unsafe {
                backend.prime(input, routes, stream)?;
                for capacity in [1, 16, 80, 256, 1024, 4096] {
                    if capacity <= rows || capacity == 4096 {
                        backend.enqueue(
                            self.layers[0],
                            capacity.min(rows),
                            input,
                            routes,
                            stream,
                            &mut || {},
                        )?;
                    }
                }
                self.combine_output(&**backend, shared, out, rows, stream)?;
                self.library.cuda_stream_synchronize(stream)
            }
        })
    }

    unsafe fn combine_output(
        &self,
        backend: &dyn RtxExpertLayer,
        shared: *const u16,
        out: *mut c_void,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        // SAFETY: backend output and caller's shared/output buffers are disjoint and stream-ordered.
        unsafe {
            match backend.partial() {
                PartialDtype::F32 => self.combine.partial(
                    backend.output().cast(),
                    shared,
                    out,
                    rows,
                    self.hidden,
                    stream,
                ),
                PartialDtype::Bf16 => self.library.peer_add_bf16(
                    backend.output(),
                    shared.cast(),
                    out,
                    rows * self.hidden,
                    stream,
                ),
            }
        }
    }

    /// Enqueue -> combine consumes shared backend scratch before another lane can reuse it.
    /// # Safety
    /// Inputs/routes/shared/payload are live on rank's stream, sized for rows, with disjoint output.
    /// abort publishes all family exchanges; the caller drains on failure before releasing owners.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue(
        &self,
        rank: usize,
        layer: usize,
        lane: usize,
        rows: usize,
        input: ExpertInput,
        routes: Routes,
        scratch: CuteafdDeviceBuffer,
        shared: *const u16,
        out: *mut c_void,
        abort: &mut dyn FnMut(),
    ) -> Result<()> {
        let broadcast = (self.source.get() == RouteSource::Broadcast).then_some(Broadcast { source: 0, wire: false });
        // SAFETY: forwarded unchanged; the caller upholds this function's contract.
        unsafe { self.enqueue_with(rank, layer, lane, rows, input, routes, scratch, shared, out, broadcast, abort) }
    }

    /// As [`Self::enqueue`], for a layer whose router, routes and FP8 K32 wire
    /// rows exist only on `source`: the source packs routes and wire into its
    /// `scratch` and pushes them over the peer exchange (device-driven, no host
    /// hop); the peer waits, then runs its half on the received rows.
    /// # Safety
    /// As [`Self::enqueue`]. On the peer, `input` and `routes` are ignored; on
    /// the source, `input` is FP8 K32 wire rows and `scratch` holds
    /// `PackedRoutes` plus the wire rows. Requires a pair built with
    /// [`Self::new_with_wire`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn enqueue_from(
        &self,
        rank: usize,
        source: usize,
        layer: usize,
        lane: usize,
        rows: usize,
        input: ExpertInput,
        routes: Routes,
        scratch: CuteafdDeviceBuffer,
        shared: *const u16,
        out: *mut c_void,
        abort: &mut dyn FnMut(),
    ) -> Result<()> {
        ensure!(source < 2 && self.wire_row_bytes > 0, "source broadcast needs a wire-capable pair");
        // SAFETY: forwarded unchanged; the caller upholds this function's contract.
        unsafe { self.enqueue_with(rank, layer, lane, rows, input, routes, scratch, shared, out,
            Some(Broadcast { source, wire: true }), abort) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn enqueue_with(
        &self,
        rank: usize,
        layer: usize,
        lane: usize,
        rows: usize,
        mut input: ExpertInput,
        mut routes: Routes,
        scratch: CuteafdDeviceBuffer,
        shared: *const u16,
        out: *mut c_void,
        broadcast: Option<Broadcast>,
        abort: &mut dyn FnMut(),
    ) -> Result<()> {
        let mut aborted = false;
        let mut abort_once = || {
            if !aborted {
                aborted = true;
                abort();
            }
        };
        let result = self.routes.on(rank, || {
            let stream = self.devices[rank].stream;
            let packed = PackedRoutes::new(rows, self.topk);
            // A source broadcast's peer has no routes of its own to check.
            let receives = broadcast.is_some_and(|b| b.wire && b.source != rank);
            let unit = if self.checking.get() && !receives { self.active.borrow().get(lane).copied().flatten() } else { None };
            if let Some(unit) = unit {
                let mut checks = self.checks.borrow_mut();
                let storage = checks.as_mut().context("route diagnostics not started")?
                    .storage(lane, unit)?.context("route diagnostics disabled for a live unit")?;
                let host = storage.get(&layer).context("route diagnostics omit a TP2 layer")?[rank].buffer;
                ensure!(rows <= self.max_rows, "route diagnostic row count exceeds its storage");
                let full = self.max_rows * self.topk * 4;
                // SAFETY: the slot's pinned storage is retained (never replaced)
                // until finish_unit drains both producer streams.
                unsafe {
                    self.library.copy_d2h_host_buffer_async(
                        host,
                        CuteafdDeviceBuffer {
                            ptr: routes.ids,
                            bytes: packed.section,
                            ..scratch
                        },
                        packed.section,
                        stream,
                    )?;
                    // Weights at the fixed max-row offset; finish trims both sections.
                    self.library.copy_d2h_host_buffer_async(
                        CuteafdHostBuffer {
                            ptr: host.ptr.cast::<u8>().add(full).cast(),
                            bytes: packed.section,
                            ..host
                        },
                        CuteafdDeviceBuffer {
                            ptr: routes.weights,
                            bytes: packed.section,
                            ..scratch
                        },
                        packed.section,
                        stream,
                    )?;
                }
            }
            if let Some(Broadcast { source, wire }) = broadcast {
                let at = 2 * lane + layer % 2;
                let wire_bytes = if wire { rows * self.wire_row_bytes } else { 0 };
                let extent = packed.bytes + wire_bytes;
                if rank == source {
                    ensure!(
                        extent <= scratch.bytes,
                        "router scratch cannot pack canonical routes"
                    );
                    // SAFETY: logits are consumed; aligned canonical sections fit this rank's scratch.
                    unsafe {
                        self.library
                            .cuda_zero_bytes_async(scratch, packed.bytes, stream)?;
                        self.library.copy_d2d_async(
                            scratch,
                            CuteafdDeviceBuffer {
                                ptr: routes.ids,
                                bytes: packed.section,
                                ..scratch
                            },
                            packed.section,
                            stream,
                        )?;
                        self.library.copy_d2d_async(
                            CuteafdDeviceBuffer {
                                ptr: scratch.ptr.cast::<u8>().add(packed.weights).cast(),
                                bytes: packed.section,
                                ..scratch
                            },
                            CuteafdDeviceBuffer {
                                ptr: routes.weights,
                                bytes: packed.section,
                                ..scratch
                            },
                            packed.section,
                            stream,
                        )?;
                        if wire {
                            let ExpertInput::Fp8K32(rows_ptr) = input else {
                                anyhow::bail!("source broadcast carries FP8 K32 wire rows");
                            };
                            self.library.copy_d2d_async(
                                CuteafdDeviceBuffer {
                                    ptr: scratch.ptr.cast::<u8>().add(packed.bytes).cast(),
                                    bytes: wire_bytes,
                                    ..scratch
                                },
                                CuteafdDeviceBuffer { ptr: rows_ptr, bytes: wire_bytes, ..scratch },
                                wire_bytes,
                                stream,
                            )?;
                        }
                    }
                    self.routes.push(source, at, scratch.ptr, extent)?;
                } else {
                    self.routes.wait(rank, at)?;
                    let received = self.routes.recv(rank, at)?;
                    // SAFETY: the source published the packed canonical sections (and
                    // wire rows) before this stream's wait.
                    routes = Routes {
                        ids: received,
                        weights: unsafe { received.cast::<u8>().add(packed.weights).cast() },
                    };
                    if wire {
                        // SAFETY: as above; the wire rows follow the packed routes.
                        input = ExpertInput::Fp8K32(unsafe { received.cast::<u8>().add(packed.bytes).cast() });
                    }
                }
            }
            let mut backend = self.ranks[rank].borrow_mut();
            // SAFETY: caller owns stream-ordered input/routes/shared/payload; combine consumes scratch now.
            unsafe {
                backend.enqueue(layer, rows, input, routes, stream, &mut abort_once)?;
                self.combine_output(&**backend, shared, out, rows, stream)?;
            }
            if let Some(Broadcast { source, .. }) = broadcast {
                let at = 2 * lane + layer % 2;
                // A publication is not consumption: the source must not overwrite
                // canonical rows until the peer's backend has finished reading them.
                if rank == source {
                    self.routes.wait(source, at)?;
                } else {
                    self.routes.signal(rank, at)?;
                }
            }
            Ok(())
        });
        if result.is_err() {
            abort_once();
        }
        result
    }
}

/// Device-scoped immutable routed weights. Cloning this factory never clones
/// scratch, output storage, events or native launch slots.
#[derive(Clone)]
pub(crate) enum OwnerWeights<'a> {
    Native(super::native::NativeTp2Weights<'a>),
    Exl3(super::exl3::Exl3Tp2Weights<'a>),
}
impl<'a> OwnerWeights<'a> {
    pub(crate) fn device(&self) -> crate::shared::memory::device::Device<'a> {
        match self {
            Self::Native(w) => w.device(),
            Self::Exl3(w) => w.device(),
        }
    }
    pub(crate) fn layers(&self) -> &[usize] {
        match self {
            Self::Native(w) => w.layers(),
            Self::Exl3(w) => w.layers(),
        }
    }
    pub(crate) fn uses_bf16_input(&self) -> bool {
        matches!(self, Self::Native(w) if w.uses_bf16_input())
    }
    pub(crate) fn max_rows(&self) -> usize {
        match self {
            Self::Native(w) => w.max_rows(),
            Self::Exl3(w) => w.max_rows(),
        }
    }
    /// The same resident weights admitted at `max_rows` (one load serves the
    /// prefill arena at the unit cap and the decode lanes at their width).
    pub(crate) fn with_max_rows(&self, max_rows: usize) -> Result<Self> {
        Ok(match self {
            Self::Native(w) => Self::Native(w.with_max_rows(max_rows)?),
            Self::Exl3(w) => Self::Exl3(w.with_max_rows(max_rows)?),
        })
    }
    fn execution(&self, modules: Option<Vec<Exl3Modules<'a>>>) -> Result<OwnerExecution<'a>> {
        Ok(match self {
            Self::Native(w) => {
                ensure!(modules.is_none(), "native owner received EXL3 modules");
                OwnerExecution::Native(w.execution()?)
            }
            Self::Exl3(w) => OwnerExecution::Exl3(w.execution_with_modules(modules)?),
        })
    }
}

enum OwnerExecution<'a> {
    Native(super::native::NativeTp2<'a>),
    Exl3(super::exl3::Exl3Tp2<'a>),
}
impl<'a> OwnerExecution<'a> {
    fn backend(&self) -> &dyn RtxExpertLayer {
        match self {
            Self::Native(e) => e,
            Self::Exl3(e) => e,
        }
    }
    fn backend_mut(&mut self) -> &mut dyn RtxExpertLayer {
        match self {
            Self::Native(e) => e,
            Self::Exl3(e) => e,
        }
    }
    fn bytes_per_expert(&self, layer: usize) -> Result<usize> {
        match self {
            Self::Native(e) => e.bytes_per_expert(layer),
            Self::Exl3(e) => e.bytes_per_expert(layer),
        }
    }
    fn allocation(&self) -> &crate::shared::memory::device::Allocation<'a> {
        match self {
            Self::Native(e) => e.allocation(),
            Self::Exl3(e) => e.allocation(),
        }
    }
}

pub(crate) struct OwnerInputs<'s, 'a> {
    pub input: CuteafdDeviceBuffer,
    pub ids: CuteafdDeviceBuffer,
    pub weights: CuteafdDeviceBuffer,
    pub producer: &'s crate::shared::memory::device::Stream<'a>,
}

struct OwnerRank<'a> {
    // Streams drain before launch owners and immutable weights are released.
    stream: crate::shared::memory::device::Stream<'a>,
    ready: crate::shared::memory::device::Event<'a>,
    timing: Option<[crate::shared::memory::device::Event<'a>; 2]>,
    execution: OwnerExecution<'a>,
}

/// The loader and admission share the same arithmetic/output selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OwnerPolicy<'p> {
    Native {
        selection: super::RtxSelectionPolicy,
        output: super::native::NativeOutputPolicy,
        nvfp4: bool,
    },
    Exl3 {
        package: &'p std::path::Path,
        selection: super::RtxSelectionPolicy,
    },
}
impl<'p> OwnerPolicy<'p> {
    pub(crate) fn native(library: &NativeLibrary, nvfp4: bool) -> Result<Self> {
        Ok(Self::native_for(nvfp4, library.v41_tp2_expert_reducer()?.supports_route_sums()))
    }
    fn native_for(nvfp4: bool, route_sums: bool) -> Self {
        Self::Native {
            selection: if nvfp4 { super::RtxSelectionPolicy::FirstCapacity }
                else { super::RtxSelectionPolicy::CanonicalDecode },
            output: if nvfp4 || !route_sums { super::native::NativeOutputPolicy::KernelPlanes }
                else { super::native::NativeOutputPolicy::TokenSums },
            nvfp4,
        }
    }
    pub(crate) fn exl3(package: &'p std::path::Path) -> Self {
        Self::Exl3 { package, selection: super::RtxSelectionPolicy::CanonicalDecode }
    }
}

pub(crate) struct OwnerPair<'a> {
    reductions: [super::combine::OwnerReduce<'a>; 2],
    ranks: [OwnerRank<'a>; 2],
    bf16_input: bool,
    max_rows: u32,
    geometry: cuteafd_core::ExpertGeometry,
}
impl<'a> OwnerPair<'a> {
    /// Exact per-GPU arena requests for the policy used to load the weights.
    pub(crate) fn device_bytes_for(library: &NativeLibrary, capacity: u32,
        geometry: cuteafd_core::ExpertGeometry, policy: OwnerPolicy<'_>) -> Result<usize> {
        let backend = match policy {
            OwnerPolicy::Native { selection, output, nvfp4 } =>
                super::native::NativeTp2::allocation_sizes_for_format(library,
                    capacity as usize, selection, output, nvfp4)?,
            OwnerPolicy::Exl3 { package, selection } =>
                super::exl3::Exl3Tp2::allocation_sizes_for_selection(package,
                    geometry.hidden as usize, capacity as usize,
                    cuteafd_core::Exl3RowPolicy::for_geometry(geometry), selection)?,
        };
        Self::allocation_sizes(backend, capacity, geometry)?.into_iter()
            .try_fold(0usize, |sum, n| sum.checked_add(n).context("owner arena overflow"))
    }
    fn allocation_sizes(backend: [usize; 2], capacity: u32,
        geometry: cuteafd_core::ExpertGeometry) -> Result<[usize; 4]> {
        let reduce = super::combine::OwnerReduce::allocation_sizes(capacity, geometry)?;
        Ok([backend[0], backend[1], reduce[0], reduce[1]])
    }

    pub(crate) fn new(
        weights: [OwnerWeights<'a>; 2],
        max_rows: u32,
        geometry: cuteafd_core::ExpertGeometry,
        timing: bool,
    ) -> Result<Self> {
        Self::with_modules(weights, max_rows, geometry, timing, [None, None])
    }

    pub(crate) fn with_modules(
        weights: [OwnerWeights<'a>; 2], max_rows: u32,
        geometry: cuteafd_core::ExpertGeometry, timing: bool,
        mut modules: [Option<Vec<Exl3Modules<'a>>>; 2],
    ) -> Result<Self> {
        use crate::shared::memory::device::{Event, Stream};
        let devices = [weights[0].device(), weights[1].device()];
        cuteafd_ffi::V41Tp2ExpertReducer::require_owner_geometry(geometry)?;
        ensure!(
            geometry == cuteafd_core::expert_geometry(),
            "owner geometry differs from installed expert geometry"
        );
        ensure!(
            weights
                .iter()
                .all(|weight| weight.max_rows() == max_rows as usize),
            "owner pair must equal admitted rank capacity"
        );
        ensure!(
            (1..=4096).contains(&max_rows)
                && weights[0].layers() == weights[1].layers()
                && weights[0].uses_bf16_input() == weights[1].uses_bf16_input(),
            "owner pair layer/input contract differs"
        );
        let mut make_rank = |rank: usize| -> Result<OwnerRank<'a>> {
            let execution = weights[rank].execution(modules[rank].take())?;
            ensure!(
                execution.backend().shard() == RtxShard::Tp2 { rank: rank as u8 },
                "owner pair rank identity differs"
            );
            Ok(OwnerRank {
                stream: Stream::new(devices[rank])?,
                ready: Event::new(devices[rank])?,
                timing: timing
                    .then(|| {
                        Ok::<_, anyhow::Error>([
                            Event::new(devices[rank])?,
                            Event::new(devices[rank])?,
                        ])
                    })
                    .transpose()?,
                execution,
            })
        };
        Ok(Self {
            reductions: [
                super::combine::OwnerReduce::new(
                    devices,
                    super::Combine::OwnerReduce { owner: 0 },
                    max_rows,
                    geometry,
                )?,
                super::combine::OwnerReduce::new(
                    devices,
                    super::Combine::OwnerReduce { owner: 1 },
                    max_rows,
                    geometry,
                )?,
            ],
            ranks: [make_rank(0)?, make_rank(1)?],
            bf16_input: weights[0].uses_bf16_input(),
            max_rows,
            geometry,
        })
    }

    pub(crate) fn uses_bf16_input(&self) -> bool {
        self.bf16_input
    }

    #[cfg(test)]
    pub(crate) fn rank_output(&self, rank: usize) -> CuteafdDeviceBuffer {
        self.ranks[rank].execution.allocation().buffer
    }
    #[cfg(test)]
    pub(crate) fn rank_stream(&self, rank: usize) -> &crate::shared::memory::device::Stream<'a> {
        &self.ranks[rank].stream
    }

    /// # Safety
    /// Each rank input remains complete and immutable through completion or
    /// cancellation drainage. Deferred callers retain all buffers until their
    /// enclosing chain completes; lanes never share execution state.
    pub(crate) async unsafe fn execute(
        &mut self,
        layer: usize,
        rows: u32,
        owner: usize,
        inputs: [OwnerInputs<'_, 'a>; 2],
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            owner < 2 && (1..=self.max_rows).contains(&rows),
            "invalid owner pair rows/owner"
        );
        struct Drain<'s, 'a> {
            ranks: &'s mut [OwnerRank<'a>; 2],
            complete: bool,
        }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    for rank in self.ranks.iter() {
                        rank.stream.drain_or_abort("cancelled owner expert rank");
                    }
                }
            }
        }
        let mut drain = Drain {
            ranks: &mut self.ranks,
            complete: false,
        };
        for (rank, input) in drain.ranks.iter_mut().zip(inputs) {
            ensure!(
                rank.stream.device.id == input.producer.device.id
                    && std::ptr::eq(rank.stream.device.library, input.producer.device.library),
                "owner pair input producer differs"
            );
            let geometry = self.geometry;
            let input_width = geometry.hidden as usize * if self.bf16_input { 2 } else { 33 };
            let input_width = if self.bf16_input {
                input_width
            } else {
                input_width / 32
            };
            for (buffer, width) in [
                (input.input, input_width),
                (input.ids, geometry.topk as usize * 4),
                (input.weights, geometry.topk as usize * 4),
            ] {
                ensure!(
                    !buffer.ptr.is_null()
                        && buffer.device_id == rank.stream.device.id
                        && buffer.bytes >= rows as usize * width,
                    "owner pair input device/extent differs"
                );
            }
            rank.ready.record(input.producer)?;
            let device = rank.stream.device;
            // SAFETY: both event and serialized stream are owned by this lane;
            // the outer guard drains already enqueued ranks on partial failure.
            device.run(|| unsafe {
                device
                    .library
                    .cuda_stream_wait_event(rank.stream.raw, rank.ready.raw)
            })?;
            // No peer waits in these rank launch streams; the reduction joins them later.
            if let Some([start, _]) = &mut rank.timing {
                start.record(&rank.stream)?;
            }
            let values = if self.bf16_input {
                ExpertInput::Bf16(input.input.ptr)
            } else {
                ExpertInput::Fp8K32(input.input.ptr)
            };
            // SAFETY: extents/devices were checked and the caller retains inputs;
            // the guard drains this lane's launch streams on partial failure.
            unsafe {
                rank.execution.backend_mut().enqueue(
                    layer,
                    rows as usize,
                    values,
                    Routes {
                        ids: input.ids.ptr,
                        weights: input.weights.ptr,
                    },
                    rank.stream.raw,
                    &mut || {},
                )?;
            }
            if let Some([_, end]) = &mut rank.timing {
                end.record(&rank.stream)?;
            }
        }
        let local = &drain.ranks[owner];
        let remote = &drain.ranks[1 - owner];
        let dtype = local.execution.backend().partial();
        let layout = local.execution.backend().partial_layout(rows as usize)?;
        ensure!(
            dtype == remote.execution.backend().partial()
                && layout == remote.execution.backend().partial_layout(rows as usize)?,
            "owner pair partial contracts differ"
        );
        // SAFETY: rank outputs and streams remain borrowed through reduction drainage.
        let output = unsafe {
            self.reductions[owner]
                .reduce(
                    local.execution.allocation(),
                    remote.execution.allocation(),
                    &local.stream,
                    &remote.stream,
                    rows,
                    dtype,
                    layout,
                )
                .await?
        };
        for rank in drain.ranks.iter() {
            if let Some([start, end]) = &rank.timing {
                // Deferred reduction only orders submission. Diagnostic reads
                // wait for the end event, which also completes the preceding start.
                // SAFETY: both events are retained on this rank's owning device.
                let gpu_ms = rank.stream.device.run(|| unsafe {
                    let library = rank.stream.device.library;
                    library.cuda_event_synchronize(end.raw)?;
                    library.cuda_event_elapsed_ms(start.raw, end.raw)
                })?;
                tracing::debug!(target: "cuteafd::timing", layer, rows, gpu=rank.stream.device.id,
                    gpu_us=gpu_ms * 1000.0, bytes_per_expert=rank.execution.bytes_per_expert(layer)?,
                    "TP2 routed rank GPU execution");
            }
        }
        drain.complete = true;
        Ok(output)
    }
}

#[cfg(test)]
pub(crate) struct TestSubmission<'a> {
    pub events: Vec<&'static str>,
    pub live: bool,
    pub skip: Option<&'static str>,
    schedule: Option<&'a crate::shared::peer_split::order::Schedule>,
}
#[cfg(test)]
impl<'a> TestSubmission<'a> {
    pub fn new(schedule: Option<&'a crate::shared::peer_split::order::Schedule>) -> Self {
        Self {
            events: Vec::new(),
            live: true,
            skip: None,
            schedule,
        }
    }
    fn phase(&mut self, name: &'static str, previous: Option<&str>) -> Result<()> {
        if self.skip == Some(name) {
            return Ok(());
        }
        ensure!(
            previous.is_none() || self.events.last().copied() == previous,
            "{name} ran without its required predecessor {previous:?}"
        );
        self.events.push(name);
        Ok(())
    }
    pub fn require_live(&self) -> Result<()> {
        ensure!(self.live, "retired engine");
        Ok(())
    }
    pub fn verify_retired(&self) -> Result<()> {
        ensure!(
            !self.live
                && self.events == ["abort", "ranks", "quiesce", "copies", "graphs", "finish"],
            "incomplete production teardown: {:?}",
            self.events
        );
        Ok(())
    }
}
#[cfg(test)]
impl SubmissionOwners for TestSubmission<'_> {
    fn abort_all(&mut self) -> Result<()> {
        self.live = false;
        self.phase("abort", None)
    }
    fn drain_ranks(&mut self) -> Result<()> {
        self.phase("ranks", Some("abort"))?;
        if let Some(schedule) = self.schedule {
            crate::shared::peer_split::order::check_terminal(
                schedule,
                self.events.contains(&"abort"),
            )
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        }
        Ok(())
    }
    fn quiesce_links(&mut self) -> Result<()> {
        self.phase("quiesce", Some("ranks"))
    }
    fn drain_copies(&mut self) -> Result<()> {
        self.phase("copies", Some("quiesce"))
    }
    fn retire_graphs(&mut self) -> Result<()> {
        self.phase("graphs", Some("copies"))
    }
    fn finish(&mut self, drained: bool) {
        if drained {
            self.events.push("finish");
        } else {
            self.events.push("retain");
        }
    }
}

#[cfg(test)]
mod tests {
    /// GLM's legacy pair keeps its exact behavior: rank 0 publishes routes only
    /// when the route check selected broadcast, slots carry no wire, and the
    /// generalized source broadcast reduces to the same push, wait and signal.
    #[test]
    fn legacy_enqueue_is_the_rank0_routes_only_source_broadcast() {
        let source = include_str!("pair.rs");
        let body = |name: &str| source.split(name).nth(1).unwrap().split("\n    }\n").next().unwrap().to_string();
        let enqueue = body("    pub(crate) unsafe fn enqueue(\n");
        assert!(enqueue.contains("(self.source.get() == RouteSource::Broadcast).then_some(Broadcast { source: 0, wire: false })"));
        let new = body("    pub(crate) fn new(\n");
        assert!(new.contains("Self::new_with_wire(library, devices, ranks, hidden, topk, max_rows, lanes, exchange, false)"));
        let with = source.split("    pub(crate) fn new_with_wire(").nth(1).unwrap();
        assert!(with.contains("let wire_row_bytes = if wire { hidden + hidden / 32 } else { 0 };"));
        assert!(with.contains("PackedRoutes::new(max_rows, topk).bytes + max_rows * wire_row_bytes"));
        let core = source.split("    unsafe fn enqueue_with(").nth(1).unwrap();
        assert!(core.contains("let wire_bytes = if wire { rows * self.wire_row_bytes } else { 0 };"));
        assert!(core.contains("self.routes.push(source, at, scratch.ptr, extent)?;"));
        assert!(core.contains("self.routes.wait(rank, at)?;"));
        assert!(core.contains("self.routes.wait(source, at)?;") && core.contains("self.routes.signal(rank, at)?;"));
    }

    #[test]
    fn owner_policy_bytes_equal_native_and_reducer_allocation_requests() -> anyhow::Result<()> {
        use super::{OwnerPair, OwnerPolicy};
        use crate::shared::experts::rtx::{native::{NativeTp2, NativeOutputPolicy}, RtxSelectionPolicy};
        let geometry = cuteafd_core::ExpertGeometry::DEEPSEEK_V41;
        for unit in [4096, 2752, 2048] {
            for (nvfp4, sums) in [(false, false), (false, true), (true, false), (true, true)] {
                let OwnerPolicy::Native { selection, output, .. } = OwnerPolicy::native_for(nvfp4, sums)
                    else { unreachable!() };
                assert_eq!(selection, if nvfp4 { RtxSelectionPolicy::FirstCapacity }
                    else { RtxSelectionPolicy::CanonicalDecode });
                assert_eq!(output, if nvfp4 || !sums { NativeOutputPolicy::KernelPlanes }
                    else { NativeOutputPolicy::TokenSums });
                let partial = unit as usize * 5120 * if nvfp4 { 12 } else if sums { 4 } else { 24 };
                let native = NativeTp2::allocation_sizes(12345, geometry, unit as usize, output, nvfp4)?;
                let requests = OwnerPair::allocation_sizes(native, unit, geometry)?;
                assert_eq!(requests, [12345, partial, unit as usize * 5120 * 6 * 4, unit as usize * 5120 * 2]);
                assert_eq!(requests.into_iter().sum::<usize>(), 12345 + partial + unit as usize * 133120);
            }
            let exl3 = OwnerPolicy::exl3(std::path::Path::new("fixture"));
            assert!(matches!(exl3, OwnerPolicy::Exl3 { selection: RtxSelectionPolicy::CanonicalDecode, .. }));
            let requests = OwnerPair::allocation_sizes([67890, unit as usize * 5120 * 4], unit, geometry)?;
            assert_eq!(requests.into_iter().sum::<usize>(), 67890 + unit as usize * 153600);
        }
        Ok(())
    }
    use super::*;
    #[test]
    fn failed_lane_quiescence_retains_endpoint_mr_and_intake_before_copies() {
        struct Owner(std::rc::Rc<Cell<usize>>);
        impl Drop for Owner { fn drop(&mut self) { self.0.set(self.0.get() + 1); } }
        let drops = std::rc::Rc::new(Cell::new(0));
        let owners = RefCell::new(Some([Owner(drops.clone()), Owner(drops.clone()), Owner(drops.clone())]));
        let events = RefCell::new(Vec::new());
        let result = teardown(&mut Teardown {
            abort: || { events.borrow_mut().push("abort"); Ok(()) },
            ranks: || { events.borrow_mut().push("ranks"); Ok(()) },
            links: || { events.borrow_mut().push("quiesce"); anyhow::bail!("injected lane QP destroy failure") },
            copies: || { events.borrow_mut().push("copies"); Ok(()) },
            graphs: || { events.borrow_mut().push("graphs"); Ok(()) },
            finish: |drained: bool| {
                assert!(!drained);
                events.borrow_mut().push("retain");
                assert!(owners.borrow().is_some());
            },
        });
        assert!(result.unwrap_err().to_string().contains("link quiescence"));
        assert_eq!(*events.borrow(), ["abort", "ranks", "quiesce", "retain"]);
        assert_eq!(drops.get(), 0);
        assert!(owners.borrow().is_some());
        drop(owners);
        assert_eq!(drops.get(), 3);
    }

    #[test]
    fn submission_faults_abort_then_drain_before_return_and_reject_replay() {
        for failure in 0..12 {
            let owners = RefCell::new(TestSubmission::new(None));
            let result: Result<()> = submit(
                || owners.borrow().require_live(),
                || teardown(&mut *owners.borrow_mut()),
                || {
                    for step in 0..12 {
                        ensure!(step != failure, "injected step {step}");
                    }
                    Ok(())
                },
            );
            assert!(result
                .unwrap_err()
                .downcast_ref::<TerminalStepError>()
                .is_some());
            owners.borrow().verify_retired().unwrap();
            let replayed = Cell::new(false);
            assert!(submit(
                || owners.borrow().require_live(),
                || panic!("already retired"),
                || {
                    replayed.set(true);
                    Ok(())
                }
            )
            .is_err());
            assert!(!replayed.get());
        }
    }

    #[test]
    fn original_submission_error_is_logged_before_shutdown() {
        #[derive(Clone)]
        struct Writer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        let bytes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let writer = Writer(bytes.clone());
        let subscriber = tracing_subscriber::fmt().without_time().with_ansi(false)
            .with_writer(move || writer.clone()).finish();
        tracing::subscriber::with_default(subscriber, || {
            let result: Result<()> = submit_scoped(|| Ok(()), || {
                let logged = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
                assert!(logged.contains("submission failed; draining queued owners"), "{logged}");
                assert!(logged.contains("short-prompt prefill: injected original cause"), "{logged}");
                Ok(())
            }, |_| Err(anyhow::anyhow!("injected original cause").context("short-prompt prefill")));
            assert!(result.unwrap_err().downcast_ref::<TerminalStepError>().is_some());
        });
    }

    #[test]
    fn dropped_submission_body_runs_the_real_teardown() {
        let owners = RefCell::new(TestSubmission::new(None));
        let cancelled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<()> = submit(
                || owners.borrow().require_live(),
                || teardown(&mut *owners.borrow_mut()),
                || panic!("dropped body"),
            );
        }));
        assert!(cancelled.is_err());
        owners.borrow().verify_retired().unwrap();
    }

    #[test]
    fn teardown_mutations_cannot_pass_the_retirement_gate() {
        for skip in ["abort", "ranks", "quiesce", "copies", "graphs"] {
            let mut owners = TestSubmission::new(None);
            owners.skip = Some(skip);
            let result = teardown(&mut owners);
            assert!(
                result.is_err() || owners.verify_retired().is_err(),
                "missing {skip} passed"
            );
        }
    }

    #[test]
    fn v4_exchange_waits_are_released_before_rank_drainage() {
        // Rank streams may hold queued Spark exchange waits: the abort stops
        // every link's exchange (releasing them) before the ranks drain.
        let source = include_str!("../../../families/deepseek_v4/engine/tp2.rs");
        let start = source.find("pub(crate) fn shutdown_submission(").unwrap();
        let source = &source[start..];
        let abort = source.find("abort: ||").unwrap();
        let publication = source
            .find("home.run(|| self.publish_tp2_abort())?;")
            .unwrap();
        let release = source.find("link.terminal_quiesce()?; }").unwrap();
        let ranks = source.find("ranks: ||").unwrap();
        assert!(abort < publication && publication < release && release < ranks);
    }

    #[test]
    fn submission_preserves_primary_cause_and_reports_retained_owners() {
        let error = submit(
            || Ok(()),
            || anyhow::bail!("copy drain failed; owners quarantined"),
            || -> Result<()> { anyhow::bail!("projection enqueue failed") },
        )
        .unwrap_err();
        assert!(error.downcast_ref::<TerminalStepError>().is_some());
        assert!(format!("{error:#}").contains("copy drain failed; owners quarantined"));
        assert!(error
            .chain()
            .any(|cause| cause.to_string() == "projection enqueue failed"));
    }

    fn shared_submission_entrypoint(source: &str) -> bool {
        [
            "crate::shared::experts::rtx::pair::submit(",
            "crate::shared::experts::rtx::pair::submit_scoped(",
        ]
        .iter()
        .any(|call| {
            source.match_indices(call).any(|(offset, _)| {
                offset == 0
                    || !source[..offset]
                        .ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == ':')
            })
        })
    }

    #[test]
    fn submission_entrypoint_audit_rejects_lookalikes_and_direct_backends() {
        for call in ["submit", "submit_scoped"] {
            let qualified = format!("crate::shared::experts::rtx::pair::{call}(");
            assert!(shared_submission_entrypoint(&qualified));
            for replacement in [
                format!("{call}("),
                format!("other::pair::{call}("),
                format!("crate::shared::experts::rtx::native::{call}("),
                format!("not_{qualified}"),
                format!("other::{qualified}"),
                "backend.enqueue(".into(),
                "backend.run(".into(),
            ] {
                assert!(
                    !shared_submission_entrypoint(&replacement),
                    "accepted {replacement}"
                );
            }
        }
    }

    #[test]
    fn family_submission_and_graph_entrypoints_use_the_shared_boundary() {
        for (source, step, graph) in [
            (
                include_str!("../../../families/deepseek_v4/engine.rs"),
                "step",
                "replay_on",
            ),
            (
                include_str!("../../../families/glm5_flash/engine.rs"),
                "decode_primitive",
                "replay_on",
            ),
            (
                include_str!("../../../families/glm5_flash/engine.rs"),
                "step_units_with",
                "replay_on",
            ),
            (
                include_str!("../../../families/glm5/engine.rs"),
                "step",
                "replay_on",
            ),
            (
                include_str!("../../../families/glm5/engine.rs"),
                "step_lanes",
                "replay_on",
            ),
            (
                include_str!("../../../families/qwen4/engine.rs"),
                "step",
                "replay",
            ),
        ] {
            let start = source.find(&format!("    fn {step}(")).unwrap();
            // The Flash executor body lives in units.rs, not beside its wrapper.
            let end = source[start..].find("\n    }").unwrap() + start;
            let wrapper = &source[start..end];
            if step == "step_units_with" {
                let units = include_str!("../../../families/glm5_flash/engine/units.rs");
                assert!(wrapper.contains("self.step_units_inner(scope,"));
                assert!(units.contains("pub(super) fn step_units_inner(&self, _scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>"));
            }
            assert!(
                shared_submission_entrypoint(wrapper),
                "{step} bypasses the shared boundary"
            );
            // Replacing either shared boundary with a backend call must fail this audit.
            let bypass = wrapper
                .replace(
                    "crate::shared::experts::rtx::pair::submit(",
                    "backend.enqueue(",
                )
                .replace(
                    "crate::shared::experts::rtx::pair::submit_scoped(",
                    "backend.enqueue(",
                );
            assert!(
                !shared_submission_entrypoint(&bypass),
                "{step} accepts a direct backend"
            );
            let start = source.find(&format!("    fn {graph}(")).unwrap();
            let body = &source[start
                ..source[start..]
                    .find(if graph == "replay" {
                        "if let Some(graph)"
                    } else {
                        "let graphs"
                    })
                    .unwrap()
                    + start];
            assert!(
                body.contains("self.require_live()?;"),
                "{graph} bypasses liveness"
            );
        }
        let mimo = include_str!("../../../families/mimo_v2/engine.rs");
        assert!(shared_submission_entrypoint(mimo));
        assert!(!mimo.contains("pub(crate) struct TerminalStepError"));
    }

    #[test]
    fn packed_routes_fit_aligned_exchange_for_every_short_shape() {
        for topk in [6, 8, 10] {
            for rows in 1..=4096 {
                let p = PackedRoutes::new(rows, topk);
                assert_eq!(p.section, rows * topk * 4);
                assert_eq!(p.weights % 16, 0);
                assert_eq!(p.bytes % 16, 0);
                assert!(p.bytes >= p.weights + p.section);
                assert!(p.bytes <= (rows * topk * 8).next_multiple_of(16) + 16);
            }
        }
    }
}
