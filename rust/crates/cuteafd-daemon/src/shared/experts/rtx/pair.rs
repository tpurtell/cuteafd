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

pub(crate) fn submit<T>(
    require_live: impl FnOnce() -> Result<()>,
    shutdown: impl FnOnce() -> Result<()>,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    require_live()?;
    match body() {
        Ok(value) => Ok(value),
        Err(cause) => {
            let drain = match shutdown() {
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
mod tests {
    use super::*;
    #[test]
    fn submission_faults_abort_then_drain_before_return_and_reject_replay() {
        // Includes the caller's payload push after rank0 queued its route ack,
        // later attention/copy/graph operations, and cancellation at each prefix.
        for failure in 0..12 {
            for cancellation in [false, true] {
                let live = Cell::new(true);
                let events = RefCell::new(Vec::new());
                let result: Result<()> = submit(
                    || { ensure!(live.get(), "closed"); Ok(()) },
                    || {
                        live.set(false);
                        events.borrow_mut().push("abort both exchanges");
                        events.borrow_mut().push("drain both ranks");
                        Ok(())
                    },
                    || {
                        for step in 0..12 {
                            events.borrow_mut().push("enqueue");
                            ensure!(step != failure, "injected step {step}, cancellation={cancellation}");
                        }
                        Ok(())
                    },
                );
                assert!(result.is_err());
                let recorded = events.borrow();
                assert_eq!(&recorded[recorded.len() - 2..], &["abort both exchanges", "drain both ranks"]);
                drop(recorded);
                let replayed = Cell::new(false);
                assert!(submit(
                    || { ensure!(live.get(), "closed"); Ok(()) },
                    || panic!("already retired"),
                    || { replayed.set(true); Ok(()) },
                ).is_err());
                assert!(!replayed.get());
            }
        }
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

    #[test]
    fn family_submission_and_graph_entrypoints_use_the_shared_boundary() {
        for (source, step, graph) in [
            (include_str!("../../../families/deepseek_v4/engine.rs"), "step", "replay_on"),
            (include_str!("../../../families/glm5_flash/engine.rs"), "step", "replay_on"),
            (include_str!("../../../families/glm5_flash/engine.rs"), "step_lanes", "replay_on"),
            (include_str!("../../../families/glm5/engine.rs"), "step", "replay_on"),
            (include_str!("../../../families/glm5/engine.rs"), "step_lanes", "replay_on"),
        ] {
            let start = source.find(&format!("    fn {step}(")).unwrap();
            let end = source[start..].find(&format!("    fn {step}_inner(")).unwrap() + start;
            assert!(source[start..end].contains("crate::shared::experts::rtx::pair::submit("));
            let start = source.find(&format!("    fn {graph}(")).unwrap();
            let body = &source[start..source[start..].find("let graphs").unwrap() + start];
            assert!(body.contains("self.require_live()?;"), "{graph} bypasses liveness");
        }
        let mimo = include_str!("../../../families/mimo_v2/engine.rs");
        assert!(mimo.contains("crate::shared::experts::rtx::pair::submit("));
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
