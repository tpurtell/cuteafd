//! One rank's native FP8-K32 TP2 experts, using V4.1's rank launch sequence.
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::experts::layer::{ExpertFormat, ExpertLayer, ExpertLoadBudget, ExpertWeights};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event, Stream};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::{expert_geometry, ExpertGeometry};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs, V41ExpertOutputKind,
    V41Tp2ExpertReducer,
};
use std::{ffi::c_void, ops::Range};

pub(super) fn kernel_capacities(max_rows: usize) -> Result<Vec<u32>> {
    ensure!((1..=4096).contains(&max_rows), "invalid TP2 rank capacity");
    let capacities = [1, 16, 80, 256, 1024, 4096];
    let compiled = capacities
        .into_iter()
        .find(|&c| c as usize >= max_rows)
        .unwrap();
    Ok(capacities.into_iter().filter(|&c| c <= compiled).collect())
}

pub(super) fn output_bytes(geometry: ExpertGeometry, rows: usize) -> Result<usize> {
    rows.checked_mul(geometry.hidden as usize)
        .and_then(|n| n.checked_mul(4))
        .context("TP2 output extent overflow")
}

pub(super) fn validate_layers(layers: &Range<usize>, rank: u8) -> Result<()> {
    ensure!(
        rank < 2 && !layers.is_empty() && layers.end <= expert_geometry().layers as usize,
        "invalid TP2 rank/layer range"
    );
    Ok(())
}

struct RankState<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; 44],
}

pub(super) trait CompletionEvent {
    fn record(&self, stream: *mut c_void) -> Result<()>;
    fn drain(&self, stream: *mut c_void) -> Result<()>;
    fn wait(&self) -> Result<()>;
}

impl CompletionEvent for Event<'_> {
    fn record(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: enqueue's caller retains its rank-local stream through recording.
        self.device
            .run(|| unsafe { self.device.library.cuda_event_record(self.raw, stream) })
    }
    fn drain(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: partial launches retain their caller's stream until this drain returns.
        self.device
            .run(|| unsafe { self.device.library.cuda_stream_synchronize(stream) })
    }
    fn wait(&self) -> Result<()> {
        // SAFETY: the retained event outlives every queued operation using this arena.
        self.device
            .run(|| unsafe { self.device.library.cuda_event_synchronize(self.raw) })
    }
}

/// Must be the first owner field: drains before weights/scratch/output release.
pub(super) struct RankCompletion<E: CompletionEvent> {
    event: E,
    queued: bool,
}

impl<E: CompletionEvent> RankCompletion<E> {
    pub(super) fn new(event: E) -> Self {
        Self {
            event,
            queued: false,
        }
    }
    pub(super) fn finish(&mut self, result: Result<()>, stream: *mut c_void) -> Result<()> {
        let recorded = self.event.record(stream);
        self.queued |= recorded.is_ok();
        if result.is_err() || recorded.is_err() {
            self.event.drain(stream)?;
        }
        result.and(recorded)
    }
}

impl<E: CompletionEvent> Drop for RankCompletion<E> {
    fn drop(&mut self) {
        if self.queued {
            if let Err(error) = self.event.wait() {
                tracing::error!(%error, "draining TP2 rank owner");
            }
        }
    }
}

/// Exclusive to one rank's serialized stream; destroy graphs before this owner.
pub(crate) struct NativeTp2<'a> {
    completion: RankCompletion<Event<'a>>,
    device: Device<'a>,
    rank: u8,
    layers: Range<usize>,
    geometry: ExpertGeometry,
    max_rows: usize,
    workspace_bytes: usize,
    states: DeviceOwner<'a, Vec<RankState<'a>>>,
    weights: DeviceOwner<'a, Vec<ExpertWeights<'a>>>,
    _scratch: Allocation<'a>,
    output: Allocation<'a>,
    reducer: V41Tp2ExpertReducer<'a>,
}

impl<'a> NativeTp2<'a> {
    /// CPU-only load plan. Solver half cost is resident + staging from this
    /// plan; workspace is charged once per rank/lane, not once per layer.
    pub(crate) fn plan_layer(
        library: &NativeLibrary,
        catalog: &cuteafd_loader::OfficialV41Catalog,
        layer: usize,
        rank: u8,
    ) -> Result<ExpertLoadBudget> {
        validate_layers(
            &(layer..layer.checked_add(1).context("TP2 layer overflow")?),
            rank,
        )?;
        ensure!(
            !ExpertFormat::of(catalog).is_nvfp4() && !ExpertFormat::of(catalog).is_exl3(),
            "NativeTp2 requires native MXFP4; use the checkpoint's native-format backend"
        );
        ExpertWeights::plan(
            library,
            catalog,
            ExpertLayer::BackboneTp2 {
                layer,
                rank: rank as usize,
            },
        )
    }

    /// Max kernel scratch across unique compiled capacities plus one FP32
    /// `[max_rows, hidden]` output. Weight and module bytes are separate.
    pub(crate) fn workspace_bytes_for(library: &NativeLibrary, max_rows: usize) -> Result<usize> {
        let scratch = kernel_capacities(max_rows)?
            .into_iter()
            .map(|capacity| {
                usize::try_from(library.v41_tp2_expert_info(capacity)?.scratch_bytes)
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap();
        scratch
            .checked_add(output_bytes(expert_geometry(), max_rows)?)
            .context("TP2 workspace overflow")
    }

    pub(crate) fn load(
        device: Device<'a>,
        catalog: &cuteafd_loader::OfficialV41Catalog,
        rank: u8,
        layers: Range<usize>,
        max_rows: usize,
        budget: usize,
    ) -> Result<Self> {
        validate_layers(&layers, rank)?;
        let geometry = expert_geometry();
        let workspace_bytes = Self::workspace_bytes_for(device.library, max_rows)?;
        let mut resident = workspace_bytes;
        for layer in layers.clone() {
            let plan = Self::plan_layer(device.library, catalog, layer, rank)?;
            resident = resident
                .checked_add(plan.resident_bytes)
                .context("TP2 resident overflow")?;
            ensure!(
                resident
                    .checked_add(plan.device_staging_bytes)
                    .context("TP2 peak overflow")?
                    <= budget,
                "TP2 weights + workspace + staging exceed budget {budget}"
            );
        }
        let weights = device.own(|| {
            let mut loaded = Vec::with_capacity(layers.len());
            let mut remaining = budget - workspace_bytes;
            for layer in layers.clone() {
                let weight = ExpertWeights::load(
                    device.library,
                    catalog,
                    ExpertLayer::BackboneTp2 {
                        layer,
                        rank: rank as usize,
                    },
                    remaining,
                )?;
                remaining -= weight.budget().resident_bytes;
                loaded.push(weight);
            }
            Ok(loaded)
        })?;
        let kernels = device.run(|| {
            kernel_capacities(max_rows)?
                .into_iter()
                .map(|c| device.library.v41_tp2_expert_kernel(c))
                .collect::<Result<Vec<_>>>()
        })?;
        for kernel in &kernels {
            let info = kernel.info();
            ensure!(
                info.hidden_size == geometry.hidden
                    && info.topk == geometry.topk
                    && info.experts == geometry.experts
                    && info.logical_intermediate == geometry.intermediate / 2
                    && info.input_dtype == 7
                    && kernel.output_kind() != V41ExpertOutputKind::Bf16Routes,
                "TP2 native geometry/format mismatch"
            );
        }
        let scratch_bytes = kernels
            .iter()
            .map(|k| k.info().scratch_bytes as usize)
            .max()
            .unwrap();
        let scratch = Allocation::new(device, scratch_bytes)?;
        let output = Allocation::new(device, output_bytes(geometry, max_rows)?)?;
        let stream = Stream::new(device)?;
        let states = device.own(|| {
            let mut states = Vec::with_capacity(kernels.len());
            for kernel in kernels {
                let mut slots = [std::ptr::null_mut(); 44];
                // SAFETY: one startup stream initializes the arena, drained before publication.
                let initialized = unsafe {
                    kernel.bind_scratch(scratch.buffer.ptr, scratch_bytes as u64, &mut slots)?;
                    kernel.initialize_scratch(scratch.buffer.ptr, scratch_bytes as u64, stream.raw)
                };
                let drained = stream.drain();
                initialized.and(drained)?;
                states.push(RankState { kernel, slots });
            }
            Ok(states)
        })?;
        let reducer = device.run(|| device.library.v41_tp2_expert_reducer())?;
        ensure!(
            if geometry.same_shape(&ExpertGeometry::DEEPSEEK_V41) {
                reducer.supports_route_sums()
            } else {
                reducer.supports_geometry_route_sums()
            },
            "TP2 ordered FP32 route sums unavailable"
        );
        Ok(Self {
            completion: RankCompletion::new(Event::new(device)?),
            device,
            rank,
            layers,
            geometry,
            max_rows,
            workspace_bytes,
            states,
            weights,
            _scratch: scratch,
            output,
            reducer,
        })
    }
}

impl RtxExpertLayer for NativeTp2<'_> {
    fn shard(&self) -> RtxShard {
        RtxShard::Tp2 { rank: self.rank }
    }
    fn partial(&self) -> PartialDtype {
        PartialDtype::F32
    }
    fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }
    fn device(&self) -> i32 {
        self.device.id
    }
    fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    fn output(&self) -> *mut c_void {
        self.output.buffer.ptr
    }
    unsafe fn enqueue(
        &mut self,
        layer: usize,
        rows: usize,
        input: ExpertInput,
        routes: Routes,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            self.layers.contains(&layer) && (1..=self.max_rows).contains(&rows),
            "TP2 layer/rows not resident"
        );
        let ExpertInput::Fp8K32(wire) = input else {
            bail!("NativeTp2 requires FP8 K32 input")
        };
        ensure!(
            !wire.is_null() && !routes.ids.is_null() && !routes.weights.is_null(),
            "null TP2 input"
        );
        let result = self.device.run(|| {
            let state = self
                .states
                .iter_mut()
                .find(|s| s.kernel.info().capacity_rows as usize >= rows)
                .unwrap();
            // SAFETY: caller retains complete rank-local inputs and owns exclusive stream/output use.
            unsafe {
                self.weights[layer - self.layers.start].bind(&state.kernel, &mut state.slots)?;
                state.slots[0] = wire;
                state.slots[1] = routes.ids;
                state.slots[2] = routes.weights;
                let info = state.kernel.info();
                state.kernel.launch(&V41ExpertLaunchArgs {
                    tensors: state.slots,
                    num_tokens: rows as i32,
                    max_rows: info.max_rows,
                    scatter_rows: rows as i32 * self.geometry.topk as i32,
                    rows_padded: info.rows_padded,
                    max_tasks: info.max_tasks,
                    max_phys_tiles: info.max_phys_tiles,
                    max_active_clusters: info.max_active_clusters,
                    stream,
                })?;
                let mut source = CuteafdDeviceBuffer {
                    ptr: state.slots[41],
                    bytes: output_bytes(self.geometry, rows)?,
                    device_id: self.device.id,
                    flags: 0,
                };
                match state.kernel.output_kind() {
                    V41ExpertOutputKind::Fp32Routes => {
                        source.bytes *= self.geometry.topk as usize;
                        if self.geometry.same_shape(&ExpertGeometry::DEEPSEEK_V41) {
                            self.reducer
                                .sum_routes(source, self.output.buffer, rows as u32, stream)
                        } else {
                            self.reducer.sum_routes_geometry(
                                source,
                                self.output.buffer,
                                rows as u32,
                                self.geometry,
                                stream,
                            )
                        }
                    }
                    V41ExpertOutputKind::Fp32Tokens => self.device.library.copy_d2d_async(
                        self.output.buffer,
                        source,
                        source.bytes,
                        stream,
                    ),
                    V41ExpertOutputKind::Bf16Routes => {
                        bail!("NativeTp2 excludes NVFP4 BF16 routes")
                    }
                }
            }
        });
        self.completion.finish(result, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn queued_drop_waits_before_releasing_storage() -> Result<()> {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Arc, Condvar, Mutex,
        };
        struct PendingEvent {
            gate: Arc<(Mutex<bool>, Condvar)>,
            waiting: mpsc::Sender<()>,
        }
        impl CompletionEvent for PendingEvent {
            fn record(&self, _: *mut c_void) -> Result<()> {
                Ok(())
            }
            fn drain(&self, _: *mut c_void) -> Result<()> {
                self.wait()
            }
            fn wait(&self) -> Result<()> {
                self.waiting.send(())?;
                let (lock, wake) = &*self.gate;
                let _complete = wake
                    .wait_while(lock.lock().unwrap(), |done| !*done)
                    .unwrap();
                Ok(())
            }
        }
        struct Storage(Arc<AtomicUsize>);
        impl Drop for Storage {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct Executor {
            completion: RankCompletion<PendingEvent>,
            _storage: Storage,
        }
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let freed = Arc::new(AtomicUsize::new(0));
        let (started, waiting) = mpsc::channel();
        let mut executor = Executor {
            completion: RankCompletion::new(PendingEvent {
                gate: gate.clone(),
                waiting: started,
            }),
            _storage: Storage(freed.clone()),
        };
        executor.completion.finish(Ok(()), std::ptr::null_mut())?;
        let (returned, done) = mpsc::channel();
        let dropper = std::thread::spawn(move || {
            drop(executor);
            returned.send(()).unwrap();
        });
        waiting.recv_timeout(std::time::Duration::from_secs(5))?;
        let released_early = freed.load(Ordering::SeqCst);
        let returned_early = done.try_recv().is_ok();
        // Release before asserting, so a regression never leaves a blocked thread.
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        dropper.join().unwrap();
        assert_eq!(released_early, 0);
        assert!(!returned_early);
        assert_eq!(freed.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn failed_launch_or_event_record_drains_the_caller_stream() {
        use std::{cell::RefCell, rc::Rc};
        struct ProbeEvent {
            fail_record: bool,
            calls: Rc<RefCell<Vec<&'static str>>>,
        }
        impl CompletionEvent for ProbeEvent {
            fn record(&self, _: *mut c_void) -> Result<()> {
                self.calls.borrow_mut().push("record");
                ensure!(!self.fail_record, "injected record failure");
                Ok(())
            }
            fn drain(&self, _: *mut c_void) -> Result<()> {
                self.calls.borrow_mut().push("drain");
                Ok(())
            }
            fn wait(&self) -> Result<()> {
                self.calls.borrow_mut().push("wait");
                Ok(())
            }
        }
        for fail_record in [false, true] {
            let calls = Rc::new(RefCell::new(Vec::new()));
            let mut completion = RankCompletion::new(ProbeEvent {
                fail_record,
                calls: calls.clone(),
            });
            let launch = if fail_record {
                Ok(())
            } else {
                Err(anyhow::anyhow!("injected partial launch failure"))
            };
            assert!(completion.finish(launch, std::ptr::null_mut()).is_err());
            assert_eq!(*calls.borrow(), ["record", "drain"]);
            drop(completion);
            assert_eq!(
                *calls.borrow(),
                if fail_record {
                    vec!["record", "drain"]
                } else {
                    vec!["record", "drain", "wait"]
                }
            );
        }
    }

    #[test]
    fn capacities_round_up_once() {
        for (rows, expected) in [
            (1, vec![1]),
            (3, vec![1, 16]),
            (80, vec![1, 16, 80]),
            (255, vec![1, 16, 80, 256]),
            (4096, vec![1, 16, 80, 256, 1024, 4096]),
        ] {
            assert_eq!(kernel_capacities(rows).unwrap(), expected);
        }
        assert!(kernel_capacities(0).is_err());
        assert!(kernel_capacities(4097).is_err());
    }
    #[test]
    fn output_extent_uses_each_geometry() {
        for g in [
            ExpertGeometry::DEEPSEEK_V41,
            ExpertGeometry::DEEPSEEK_V4_FLASH,
            ExpertGeometry::DEEPSEEK_V4_PRO,
        ] {
            for rows in [1, 3, 16, 80, 255, 4096] {
                assert_eq!(output_bytes(g, rows).unwrap(), rows * g.hidden as usize * 4);
            }
        }
        assert!(output_bytes(ExpertGeometry::DEEPSEEK_V4_PRO, usize::MAX).is_err());
    }
}
