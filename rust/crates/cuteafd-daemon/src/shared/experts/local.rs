//! One lane's local routed-expert scratch; both lanes share immutable weights.
use super::exl3::{
    execution::{Exl3Modules, Exl3Execution, Exl3InputFormat, Exl3Workspace},
    Exl3Weights,
};
use super::layer::{ExpertLayer, ExpertWeights};
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::ExpertGeometry;
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs, V41LocalExpertReducer,
};
use std::{ffi::c_void, path::Path, rc::Rc};

/// Activation contract of the resident kernel, independent of remote wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalInputFormat {
    Fp8K32,
    Bf16,
}
impl LocalInputFormat {
    fn for_nvfp4(nvfp4: bool) -> Self {
        if nvfp4 {
            Self::Bf16
        } else {
            Self::Fp8K32
        }
    }
    fn row_bytes(self, geometry: ExpertGeometry) -> usize {
        match self {
            Self::Fp8K32 => geometry.hidden as usize + geometry.hidden as usize / 32,
            Self::Bf16 => geometry.row_bytes() as usize,
        }
    }
    fn select(
        self,
        rows: u32,
        bf16: CuteafdDeviceBuffer,
        fp8: CuteafdDeviceBuffer,
        geometry: ExpertGeometry,
    ) -> Result<CuteafdDeviceBuffer> {
        let input = match self {
            Self::Bf16 => bf16,
            Self::Fp8K32 => fp8,
        };
        ensure!(
            input.bytes == rows as usize * self.row_bytes(geometry),
            "local expert activation extent differs from {self:?} kernel input"
        );
        Ok(input)
    }
}

struct State<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; 44],
}
enum Backend<'a> {
    Full {
        states: Vec<State<'a>>,
        _scratch: DeviceAllocation<'a>,
        weights: Rc<Vec<ExpertWeights<'a>>>,
    },
    Exl3 {
        states: Vec<Exl3Execution<'a>>,
        layers: usize,
    },
}
pub(crate) struct LocalExpertWave<'a> {
    // Drain before scratch, weights or output can be released on any exit.
    stream: LoadStream<'a>,
    backend: Backend<'a>,
    output: DeviceAllocation<'a>,
    reducer: V41LocalExpertReducer<'a>,
    capacity: u32,
    input_format: LocalInputFormat,
    geometry: ExpertGeometry,
}
impl<'a> LocalExpertWave<'a> {
    fn require_geometry(geometry: ExpertGeometry) -> Result<()> {
        ensure!(
            geometry.same_shape(&ExpertGeometry::DEEPSEEK_V41) && geometry.layers > 0,
            "native local expert ABI does not support geometry {}",
            geometry.key()
        );
        ensure!(
            geometry.same_shape(&cuteafd_core::expert_geometry()),
            "local expert geometry differs from installed kernels"
        );
        Ok(())
    }
    fn capacities(capacity: u32) -> Result<Vec<u32>> {
        ensure!(
            matches!(capacity, 1 | 16 | 80 | 256 | 1024 | 4096),
            "invalid local expert capacity"
        );
        let mut capacities: Vec<_> = [1, 16, 80].into_iter().filter(|&c| c < capacity).collect();
        capacities.push(capacity);
        Ok(capacities)
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        capacity: u32,
        geometry: ExpertGeometry,
    ) -> Result<usize> {
        Self::device_bytes_for(library, capacity, false, geometry)
    }
    /// The W4A4 family reports its own scratch plan and uses its own kernels.
    pub fn device_bytes_for(
        library: &NativeLibrary,
        capacity: u32,
        nvfp4: bool,
        geometry: ExpertGeometry,
    ) -> Result<usize> {
        Self::require_geometry(geometry)?;
        let arena = Self::capacities(capacity)?
            .into_iter()
            .try_fold(0usize, |largest, c| {
                let info = if nvfp4 {
                    library.v41_nvfp4_local_expert_info(c)?
                } else {
                    library.v41_local_expert_info(c)?
                };
                Ok::<_, anyhow::Error>(largest.max(usize::try_from(info.scratch_bytes)?))
            })?;
        Ok(usize::try_from(cuteafd_loader::placement::families::deepseek_v41::NativeWorkspace {
            scratch_bytes: arena as u64, kernel_intermediate: 0,
        }.lane_bytes(capacity as u64, geometry.hidden as u64, geometry.topk as u64, false, nvfp4)?)?)
    }
    pub fn new(
        library: &'a NativeLibrary,
        weights: Rc<Vec<ExpertWeights<'a>>>,
        capacity: u32,
        budget: usize,
        geometry: ExpertGeometry,
    ) -> Result<Self> {
        Self::require_geometry(geometry)?;
        ensure!(
            !weights.is_empty() && weights.len() <= geometry.layers as usize,
            "local experts require complete resident layers"
        );
        for (layer, weight) in weights.iter().enumerate() {
            ensure!(
                weight.layer == ExpertLayer::BackboneFull { layer }
                    && std::ptr::eq(weight.buffers[0].library, library),
                "local layer ownership differs"
            );
        }
        let nvfp4 = weights.first().is_some_and(|weight| weight.is_nvfp4());
        ensure!(
            weights.iter().all(|weight| weight.is_nvfp4() == nvfp4),
            "local expert layers mix activation formats"
        );
        ensure!(
            Self::device_bytes_for(library, capacity, nvfp4, geometry)? <= budget,
            "local expert workspace exceeds budget"
        );
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let scratch = DeviceAllocation::new(
            library,
            Self::device_bytes_for(library, capacity, nvfp4, geometry)?
                - capacity as usize * geometry.row_bytes() as usize,
        )?;
        let mut states = Vec::new();
        for c in Self::capacities(capacity)? {
            let kernel = if nvfp4 {
                library.v41_nvfp4_local_expert_kernel(c)?
            } else {
                library.v41_local_expert_kernel(c)?
            };
            ensure!(
                kernel.info().input_row_bytes()?
                    == LocalInputFormat::for_nvfp4(nvfp4).row_bytes(geometry),
                "local expert kernel activation format differs from resident weights"
            );
            let mut slots = [std::ptr::null_mut(); 44];
            unsafe {
                kernel.bind_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, &mut slots)?;
            }
            let initialized = unsafe {
                kernel.initialize_scratch(
                    scratch.buffer.ptr,
                    scratch.buffer.bytes as u64,
                    stream.raw,
                )
            };
            let drained = unsafe { library.cuda_stream_synchronize(stream.raw) };
            initialized.and(drained)?;
            states.push(State { kernel, slots });
        }
        Ok(Self {
            stream,
            backend: Backend::Full {
                states,
                _scratch: scratch,
                weights,
            },
            input_format: LocalInputFormat::for_nvfp4(nvfp4),
            output: DeviceAllocation::new(
                library,
                capacity as usize * geometry.row_bytes() as usize,
            )?,
            reducer: library.v41_local_expert_reducer()?,
            capacity,
            geometry,
        })
    }
    pub fn exl3_device_bytes(
        directory: &Path,
        capacity: u32,
        geometry: ExpertGeometry,
    ) -> Result<usize> {
        Self::require_geometry(geometry)?;
        let directories: Vec<_> = Self::capacities(capacity)?
            .into_iter()
            .map(|c| directory.join(format!("m{c}")))
            .collect();
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)?
            .checked_add(capacity as usize * geometry.row_bytes() as usize)
            .context("local EXL3 workspace overflow")
    }
    /// Trusted native package; all modules are bound before request processing.
    pub unsafe fn new_exl3(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
        capacity: u32,
        budget: usize,
        geometry: ExpertGeometry,
    ) -> Result<Self> {
        // SAFETY: forwards the trusted native package and lane lifetime contract.
        unsafe { Self::new_exl3_with_modules(library, weights, directory, capacity, budget, geometry, None) }
    }

    /// # Safety
    /// Trusted package; preloaded modules are exclusive to this serialized lane.
    pub(crate) unsafe fn new_exl3_with_modules(
        library: &'a NativeLibrary, weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path, capacity: u32, budget: usize, geometry: ExpertGeometry,
        modules: Option<Vec<Exl3Modules<'a>>>,
    ) -> Result<Self> {
        Self::require_geometry(geometry)?;
        ensure!(
            !weights.is_empty() && weights.len() <= geometry.layers as usize,
            "local EXL3 requires resident layers"
        );
        for (layer, weight) in weights.iter().enumerate() {
            ensure!(
                matches!(weight.layout.layer, cuteafd_loader::V41Exl3Layer::Backbone(index) if index == layer)
                    && weight.layout.world == 1
                    && weight.layout.rank == 0,
                "local EXL3 layer ownership differs"
            );
        }
        ensure!(
            Self::exl3_device_bytes(directory, capacity, geometry)? <= budget,
            "local EXL3 workspace exceeds budget"
        );
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let mut states = Vec::new();
        let capacities = Self::capacities(capacity)?;
        let directories: Vec<_> = capacities
            .iter()
            .map(|c| directory.join(format!("m{c}")))
            .collect();
        let arena = Exl3Workspace::new(library, &directories)?;
        let modules = match modules {
            Some(modules) => modules,
            None => directories.iter().map(|path| {
                // SAFETY: the caller trusts the native package.
                unsafe { Exl3Modules::load(library, path, Exl3InputFormat::Fp8K32) }
            }).collect::<Result<Vec<_>>>()?,
        };
        ensure!(modules.len() == capacities.len(), "local EXL3 preload capacity count differs");
        for (c, modules) in capacities.into_iter().zip(modules) {
            // SAFETY: each state owns its modules and serializes on the lane stream.
            let state = unsafe {
                Exl3Execution::with_modules(library, weights.clone(), modules, Some(arena.clone()))?
            };
            ensure!(
                state.capacity() == c as usize && state.output_element_bytes() == 4,
                "local EXL3 requires matching capacity and FP32 output"
            );
            states.push(state);
        }
        ensure!(
            arena.bytes()
                + states
                    .iter()
                    .map(|state| state.workspace_bytes())
                    .sum::<usize>()
                + capacity as usize * geometry.row_bytes() as usize
                == Self::exl3_device_bytes(directory, capacity, geometry)?,
            "local EXL3 shared workspace plan mismatch"
        );
        Ok(Self {
            stream,
            backend: Backend::Exl3 {
                states,
                layers: weights.len(),
            },
            input_format: LocalInputFormat::Fp8K32,
            output: DeviceAllocation::new(
                library,
                capacity as usize * geometry.row_bytes() as usize,
            )?,
            reducer: library.v41_local_expert_reducer()?,
            capacity,
            geometry,
        })
    }
    pub fn contains(&self, layer: usize) -> bool {
        layer
            < match &self.backend {
                Backend::Full { weights, .. } => weights.len(),
                Backend::Exl3 { layers, .. } => *layers,
            }
    }

    pub(crate) fn select_input(
        &self,
        rows: u32,
        bf16: CuteafdDeviceBuffer,
        fp8: CuteafdDeviceBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        self.input_format.select(rows, bf16, fp8, self.geometry)
    }

    /// # Safety
    /// Complete rank-local inputs and shared output remain immutable through
    /// completion or failure drainage. The caller checks family provenance.
    pub(crate) unsafe fn execute_buffers(
        &mut self,
        layer: usize,
        rows: u32,
        inputs: [CuteafdDeviceBuffer; 3],
        shared: CuteafdDeviceBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        // SAFETY: the caller retains complete buffers until finish drains them.
        unsafe {
            self.enqueue_buffers(layer, rows, inputs, shared)?;
            crate::shared::memory::chain::finish(self.stream.library, self.stream.raw)?;
        }
        Ok(self.output_rows(rows))
    }

    /// # Safety
    /// Same retained input contract as execute_buffers; cancellation drains.
    pub(crate) async unsafe fn execute_buffers_cooperative(
        &mut self,
        layer: usize,
        rows: u32,
        inputs: [CuteafdDeviceBuffer; 3],
        shared: CuteafdDeviceBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        // SAFETY: the caller retains complete buffers until finish or cancellation drainage.
        unsafe {
            self.enqueue_buffers(layer, rows, inputs, shared)?;
            crate::shared::memory::chain::finish_cooperative(&self.stream).await?;
        }
        Ok(self.output_rows(rows))
    }

    fn output_rows(&self, rows: u32) -> CuteafdDeviceBuffer {
        let mut output = self.output.buffer;
        output.bytes = rows as usize * self.geometry.row_bytes() as usize;
        output
    }

    // Caller checks block provenance and retains input owners through the drain.
    unsafe fn enqueue_buffers(
        &mut self,
        layer: usize,
        rows: u32,
        inputs: [CuteafdDeviceBuffer; 3],
        shared: CuteafdDeviceBuffer,
    ) -> Result<()> {
        ensure!(
            self.contains(layer) && rows > 0 && rows <= self.capacity,
            "local expert layer/rows are not resident or exceed capacity"
        );
        for (buffer, bytes) in [
            (
                inputs[0],
                rows as usize * self.input_format.row_bytes(self.geometry),
            ),
            (inputs[1], rows as usize * self.geometry.topk as usize * 4),
            (inputs[2], rows as usize * self.geometry.topk as usize * 4),
            (shared, rows as usize * self.geometry.row_bytes() as usize),
        ] {
            ensure!(
                buffer.bytes == bytes && buffer.device_id == self.output.buffer.device_id,
                "local expert input extent or device differs"
            );
        }
        // Router outputs were drained on the host before routing, so the routed
        // kernels start immediately and overlap the chained shared expert; only
        // the final reduction, which adds the shared contribution, joins it.
        let launched = (|| -> Result<()> {
            if crate::shared::memory::chain::deferred() {
                // Device-ordered passes do not drain the router on the host:
                // the routed kernels follow it (and the shared expert) in the chain.
                unsafe {
                    crate::shared::memory::chain::join(self.stream.library, self.stream.raw)?;
                }
            }
            match &mut self.backend {
                Backend::Exl3 { states, .. } => {
                    let state = states
                        .iter_mut()
                        .find(|s| s.capacity() >= rows as usize)
                        .context("local EXL3 capacity state missing")?;
                    let values = unsafe {
                        state.launch_layer(
                            layer,
                            [inputs[0], inputs[1], inputs[2]],
                            rows as usize,
                            self.stream.raw,
                        )?
                    };
                    unsafe {
                        crate::shared::memory::chain::join(self.stream.library, self.stream.raw)?;
                    }
                    unsafe {
                        self.reducer.finish(
                            values.ptr.cast(),
                            shared.ptr.cast(),
                            self.output.buffer.ptr.cast(),
                            rows,
                            true,
                            self.stream.raw,
                        )
                    }
                }
                Backend::Full {
                    states, weights, ..
                } => {
                    let state = states
                        .iter_mut()
                        .find(|s| s.kernel.info().capacity_rows >= rows)
                        .context("local expert capacity state missing")?;
                    weights[layer].bind(&state.kernel, &mut state.slots)?;
                    state.slots[0] = inputs[0].ptr;
                    state.slots[1] = inputs[1].ptr;
                    state.slots[2] = inputs[2].ptr;
                    let info = state.kernel.info();
                    let args = V41ExpertLaunchArgs {
                        tensors: state.slots,
                        num_tokens: rows as i32,
                        max_rows: info.max_rows,
                        scatter_rows: rows as i32 * self.geometry.topk as i32,
                        rows_padded: info.rows_padded,
                        max_tasks: info.max_tasks,
                        max_phys_tiles: info.max_phys_tiles,
                        max_active_clusters: info.max_active_clusters,
                        stream: self.stream.raw,
                    };
                    // Shared arena remains exclusive through this drain, including failure.

                    unsafe {
                        state.kernel.launch(&args)?;
                    }
                    unsafe {
                        crate::shared::memory::chain::join(self.stream.library, self.stream.raw)?;
                    }
                    unsafe {
                        match state.kernel.output_kind() {
                            cuteafd_ffi::V41ExpertOutputKind::Bf16Routes => {
                                self.reducer.finish_bf16_routes(
                                    state.slots[41].cast(),
                                    shared.ptr.cast(),
                                    self.output.buffer.ptr.cast(),
                                    rows,
                                    self.stream.raw,
                                )
                            }
                            cuteafd_ffi::V41ExpertOutputKind::Fp32Routes
                            | cuteafd_ffi::V41ExpertOutputKind::Fp32Tokens => self.reducer.finish(
                                state.slots[41].cast(),
                                shared.ptr.cast(),
                                self.output.buffer.ptr.cast(),
                                rows,
                                state.kernel.accumulates_tokens(),
                                self.stream.raw,
                            ),
                        }
                    }
                }
            }
        })();
        if let Err(error) = launched {
            unsafe {
                self.stream
                    .library
                    .cuda_stream_synchronize(self.stream.raw)?;
            }
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../families/deepseek_v41/v41_experts/local/tests.rs"]
mod tests;
