//! Paired routed execution, canonical routes and fused payloads on serialized rank streams.
use super::{
    ExchangeDtype, ExpertInput, FusedCombine, PartialDtype, RouteCheck, RouteIdentity, RouteSource,
    Routes, RtxExpertLayer, RtxShard,
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
    ops::Range,
};

/// Full submission boundary, including captured graph replays and caller-side
/// exchanges. Shutdown must publish aborts before draining and retire the engine;
/// A failed drain may return only if the caller retains every queued owner;
/// callers without quarantine ownership must terminate in shutdown instead.
#[derive(Debug, thiserror::Error)]
#[error("submission failed permanently: {cause}; {drain}")]
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
where A: FnMut() -> Result<()>, D: FnMut() -> Result<()>, Q: FnMut() -> Result<()>,
    C: FnMut() -> Result<()>, R: FnMut() -> Result<()>, F: FnMut(bool),
{
    fn abort_all(&mut self) -> Result<()> { (self.abort)() }
    fn drain_ranks(&mut self) -> Result<()> { (self.ranks)() }
    fn quiesce_links(&mut self) -> Result<()> { (self.links)() }
    fn drain_copies(&mut self) -> Result<()> { (self.copies)() }
    fn retire_graphs(&mut self) -> Result<()> { (self.graphs)() }
    fn finish(&mut self, drained: bool) { (self.finish)(drained) }
}

pub(crate) fn teardown(owners: &mut impl SubmissionOwners) -> Result<()> {
    let mut failures = Vec::new();
    let aborted = match owners.abort_all() {
        Ok(()) => true,
        Err(error) => { failures.push(format!("abort publication: {error:#}")); false }
    };
    // Failed abort publication cannot safely unblock a missing producer.
    let compute = if aborted {
        match owners.drain_ranks() {
            Ok(()) => true,
            Err(error) => { failures.push(format!("rank drainage: {error:#}")); false }
        }
    } else { false };
    let quiesced = match owners.quiesce_links() {
        Ok(()) => true,
        Err(error) => { failures.push(format!("link quiescence: {error:#}")); false }
    };
    // Copy streams can depend on compute events; never drain an unproved dependency.
    if compute && quiesced {
        if let Err(error) = owners.drain_copies() { failures.push(format!("copy drainage: {error:#}")); }
    }
    if failures.is_empty() {
        if let Err(error) = owners.retire_graphs() { failures.push(format!("graph retirement: {error:#}")); }
    }
    owners.finish(failures.is_empty());
    ensure!(failures.is_empty(), "queued owners retained: {}", failures.join("; "));
    Ok(())
}

struct SubmissionGuard<S: FnOnce() -> Result<()>>(Option<S>);
impl<S: FnOnce() -> Result<()>> SubmissionGuard<S> {
    fn shutdown(&mut self) -> Result<()> { self.0.take().expect("submission already retired")() }
}
impl<S: FnOnce() -> Result<()>> Drop for SubmissionGuard<S> {
    fn drop(&mut self) {
        if let Some(shutdown) = self.0.take() {
            // Cancellation/unwinding must obey the same owner contract as enqueue errors.
            if let Err(error) = shutdown() { tracing::error!(%error, "cancelled submission retains queued owners"); }
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
        SubmissionScope { _guard: std::marker::PhantomData }
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
        Ok(value) => { guard.0.take(); Ok(value) }
        Err(cause) => {
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

pub(crate) struct Pair<'a> {
    pub layers: Range<usize>,
    ranks: [RefCell<Box<dyn RtxExpertLayer + 'a>>; 2],
    pub combine: FusedCombine<'a>,
    pub routes: PeerExchange<'a>,
    source: Cell<RouteSource>,
    checks: RefCell<BTreeMap<(usize, usize), (usize, [HostAllocation<'a>; 2])>>,
    checking: Cell<bool>,
    interval: usize,
    steps: Cell<usize>,
    library: &'a NativeLibrary,
    devices: [RankDevice; 2],
    topk: usize,
    hidden: usize,
}

impl<'a> Pair<'a> {
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
        let layers = ranks[0].layers();
        ensure!(
            !layers.is_empty() && layers == ranks[1].layers(),
            "TP2 layer ranges differ"
        );
        for (rank, backend) in ranks.iter().enumerate() {
            ensure!(
                backend.shard() == RtxShard::Tp2 { rank: rank as u8 }
                    && backend.device() == devices[rank].device
                    && backend.partial() == ranks[0].partial(),
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
            PackedRoutes::new(max_rows, topk).bytes,
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
            ranks: ranks.map(RefCell::new),
            combine: FusedCombine::new(library, exchange)?,
            routes,
            source: Cell::new(RouteSource::Replicated),
            checks: RefCell::new(BTreeMap::new()),
            checking: Cell::new(false),
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

    pub(crate) fn group_rows(&self, rows: usize) -> Result<usize> {
        let group = self.ranks[0].borrow().group_rows(rows)?;
        ensure!(self.ranks[1].borrow().group_rows(rows)? == group,
            "installed TP2 ranks disagree on expert grouping");
        Ok(group)
    }

    pub(crate) fn weight_read_bytes(&self) -> Result<usize> {
        let bytes = self.ranks.iter().map(|rank| rank.borrow().weight_read_bytes())
            .collect::<Result<Vec<_>>>()?;
        Ok(*bytes.iter().max().context("installed TP2 ranks missing")?)
    }

    pub(crate) fn begin_checks(&self, rows: &[usize], force: bool) -> Result<()> {
        let steps = self.steps.get() + 1;
        self.steps.set(steps);
        let check = force || (self.interval > 0 && steps % self.interval == 0);
        self.checking.set(check);
        if !check {
            return Ok(());
        }
        let mut checks = self.checks.borrow_mut();
        ensure!(checks.is_empty(), "undrained TP2 route diagnostics");
        for layer in self.layers.clone() {
            for (lane, &rows) in rows.iter().enumerate() {
                let hosts = (0..2)
                    .map(|rank| {
                        self.routes.on(rank, || {
                            HostAllocation::new(self.library, rows * self.topk * 8)
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                checks.insert(
                    (layer, lane),
                    (
                        rows,
                        hosts.try_into().map_err(|_| anyhow::anyhow!("two ranks"))?,
                    ),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn checking(&self) -> bool {
        self.checking.get()
    }

    pub(crate) fn finish_checks(&self) -> Result<()> {
        if !self.checking.replace(false) {
            return Ok(());
        }
        for rank in 0..2 {
            // SAFETY: every producer/wait for the completed step is already queued.
            self.routes.on(rank, || unsafe {
                self.library
                    .cuda_stream_synchronize(self.devices[rank].stream)
            })?;
        }
        let checks = std::mem::take(&mut *self.checks.borrow_mut())
            .into_iter()
            .map(|((layer, _), (rows, hosts))| RouteCheck {
                layer,
                rows,
                topk: self.topk,
                ranks: hosts.map(|h| h.bytes().to_vec()),
            })
            .collect::<Vec<_>>();
        let identity = RouteIdentity::check(&checks);
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
        Ok(())
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
                            self.layers.start,
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
        mut routes: Routes,
        scratch: CuteafdDeviceBuffer,
        shared: *const u16,
        out: *mut c_void,
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
            if self.checking.get() {
                let checks = self.checks.borrow();
                let (expected, hosts) = checks
                    .get(&(layer, lane))
                    .context("route diagnostics not prepared")?;
                ensure!(*expected == rows, "route diagnostic row count changed");
                let host = hosts[rank].buffer;
                // SAFETY: prepared pinned storage stays live through finish_checks' stream drains.
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
                    self.library.copy_d2h_host_buffer_async(
                        CuteafdHostBuffer {
                            ptr: host.ptr.cast::<u8>().add(packed.section).cast(),
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
            if self.source.get() == RouteSource::Broadcast {
                let at = 2 * lane + layer % 2;
                if rank == 0 {
                    ensure!(
                        packed.bytes <= scratch.bytes,
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
                    }
                    self.routes.push(0, at, scratch.ptr, packed.bytes)?;
                } else {
                    self.routes.wait(1, at)?;
                    let received = self.routes.recv(1, at)?;
                    // SAFETY: rank0 published the packed canonical sections before this stream's wait.
                    routes = Routes {
                        ids: received,
                        weights: unsafe { received.cast::<u8>().add(packed.weights).cast() },
                    };
                }
            }
            let mut backend = self.ranks[rank].borrow_mut();
            // SAFETY: caller owns stream-ordered input/routes/shared/payload; combine consumes scratch now.
            unsafe {
                backend.enqueue(layer, rows, input, routes, stream, &mut abort_once)?;
                self.combine_output(&**backend, shared, out, rows, stream)?;
            }
            if self.source.get() == RouteSource::Broadcast {
                let at = 2 * lane + layer % 2;
                // A publication is not consumption: rank0 must not overwrite
                // canonical rows until rank1's backend has finished reading them.
                if rank == 0 {
                    self.routes.wait(0, at)?;
                } else {
                    self.routes.signal(1, at)?;
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
        Self { events: Vec::new(), live: true, skip: None, schedule }
    }
    fn phase(&mut self, name: &'static str, previous: Option<&str>) -> Result<()> {
        if self.skip == Some(name) { return Ok(()); }
        ensure!(previous.is_none() || self.events.last().copied() == previous,
            "{name} ran without its required predecessor {previous:?}");
        self.events.push(name); Ok(())
    }
    pub fn require_live(&self) -> Result<()> { ensure!(self.live, "retired engine"); Ok(()) }
    pub fn verify_retired(&self) -> Result<()> {
        ensure!(!self.live && self.events == ["abort", "ranks", "quiesce", "copies", "graphs", "finish"],
            "incomplete production teardown: {:?}", self.events);
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
            crate::shared::peer_split::order::check_terminal(schedule, self.events.contains(&"abort"))
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        }
        Ok(())
    }
    fn quiesce_links(&mut self) -> Result<()> { self.phase("quiesce", Some("ranks")) }
    fn drain_copies(&mut self) -> Result<()> { self.phase("copies", Some("quiesce")) }
    fn retire_graphs(&mut self) -> Result<()> { self.phase("graphs", Some("copies")) }
    fn finish(&mut self, drained: bool) {
        if drained { self.events.push("finish"); }
        else { self.events.push("retain"); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn submission_faults_abort_then_drain_before_return_and_reject_replay() {
        for failure in 0..12 {
            let owners = RefCell::new(TestSubmission::new(None));
            let result: Result<()> = submit(
                || owners.borrow().require_live(),
                || teardown(&mut *owners.borrow_mut()),
                || {
                    for step in 0..12 { ensure!(step != failure, "injected step {step}"); }
                    Ok(())
                },
            );
            assert!(result.unwrap_err().downcast_ref::<TerminalStepError>().is_some());
            owners.borrow().verify_retired().unwrap();
            let replayed = Cell::new(false);
            assert!(submit(|| owners.borrow().require_live(), || panic!("already retired"),
                || { replayed.set(true); Ok(()) }).is_err());
            assert!(!replayed.get());
        }
    }

    #[test]
    fn dropped_submission_body_runs_the_real_teardown() {
        let owners = RefCell::new(TestSubmission::new(None));
        let cancelled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<()> = submit(|| owners.borrow().require_live(),
                || teardown(&mut *owners.borrow_mut()), || panic!("dropped body"));
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
            assert!(result.is_err() || owners.verify_retired().is_err(), "missing {skip} passed");
        }
    }

    #[test]
    fn v4_unabortable_device_proxy_is_refused_before_rank_drainage() {
        let source = include_str!("../../../families/deepseek_v4/engine/tp2.rs");
        let start = source.find("pub(crate) fn shutdown_submission(").unwrap();
        let source = &source[start..];
        let abort = source.find("abort: ||").unwrap();
        let publication = source.find("home.run(|| self.publish_tp2_abort())?;").unwrap();
        let refusal = source.find("ensure!(self.device_link.is_none()").unwrap();
        let ranks = source.find("ranks: ||").unwrap();
        assert!(abort < publication && publication < refusal && refusal < ranks);
    }

    #[test]
    fn submission_preserves_primary_cause_and_reports_retained_owners() {
        let error = submit(
            || Ok(()), || anyhow::bail!("copy drain failed; owners quarantined"),
            || -> Result<()> { anyhow::bail!("projection enqueue failed") },
        ).unwrap_err();
        assert!(error.downcast_ref::<TerminalStepError>().is_some());
        assert!(format!("{error:#}").contains("copy drain failed; owners quarantined"));
        assert!(error.chain().any(|cause| cause.to_string() == "projection enqueue failed"));
    }

    fn shared_submission_entrypoint(source: &str) -> bool {
        ["crate::shared::experts::rtx::pair::submit(",
            "crate::shared::experts::rtx::pair::submit_scoped("].iter().any(|call|
            source.match_indices(call).any(|(offset, _)| offset == 0
                || !source[..offset].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == ':')))
    }

    #[test]
    fn submission_entrypoint_audit_rejects_lookalikes_and_direct_backends() {
        for call in ["submit", "submit_scoped"] {
            let qualified = format!("crate::shared::experts::rtx::pair::{call}(");
            assert!(shared_submission_entrypoint(&qualified));
            for replacement in [format!("{call}("), format!("other::pair::{call}("),
                format!("crate::shared::experts::rtx::native::{call}("),
                format!("not_{qualified}"), format!("other::{qualified}"),
                "backend.enqueue(".into(), "backend.run(".into()] {
                assert!(!shared_submission_entrypoint(&replacement), "accepted {replacement}");
            }
        }
    }

    #[test]
    fn family_submission_and_graph_entrypoints_use_the_shared_boundary() {
        for (source, step, graph) in [
            (include_str!("../../../families/deepseek_v4/engine.rs"), "step", "replay_on"),
            (include_str!("../../../families/glm5_flash/engine.rs"), "step", "replay_on"),
            (include_str!("../../../families/glm5_flash/engine.rs"), "step_lanes", "replay_on"),
            (include_str!("../../../families/glm5/engine.rs"), "step", "replay_on"),
            (include_str!("../../../families/glm5/engine.rs"), "step_lanes", "replay_on"),
            (include_str!("../../../families/qwen4/engine.rs"), "step", "replay"),
        ] {
            let start = source.find(&format!("    fn {step}(")).unwrap();
            let end = source[start..].find(&format!("    fn {step}_inner(")).unwrap() + start;
            let wrapper = &source[start..end];
            assert!(shared_submission_entrypoint(wrapper), "{step} bypasses the shared boundary");
            // Replacing either shared boundary with a backend call must fail this audit.
            let bypass = wrapper.replace("crate::shared::experts::rtx::pair::submit(", "backend.enqueue(")
                .replace("crate::shared::experts::rtx::pair::submit_scoped(", "backend.enqueue(");
            assert!(!shared_submission_entrypoint(&bypass), "{step} accepts a direct backend");
            let start = source.find(&format!("    fn {graph}(")).unwrap();
            let body = &source[start..source[start..].find(if graph == "replay" { "if let Some(graph)" } else { "let graphs" }).unwrap() + start];
            assert!(body.contains("self.require_live()?;"), "{graph} bypasses liveness");
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
