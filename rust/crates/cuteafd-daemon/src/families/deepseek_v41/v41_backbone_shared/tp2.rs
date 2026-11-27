//! Device-owned TP2 shared-expert weights and one rank's lane workspace.
use crate::shared::memory::Tag;
use crate::shared::memory::device::{Allocation, Device, Event, PeerTransfer, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, Bf16Add, V41Fp8Plan, V41SharedSwiGlu};
use cuteafd_loader::OfficialV41Catalog;
use std::rc::Rc;
use cuteafd_core::memory_layout::ArenaTag;

/// The packed weight and scales; the raw scales live until their pack ran
/// (a push lands them and packs at Done), then [`Weights::release_raw_scales`].
struct Matrix<'a> { weight: Allocation<'a>, scales: Allocation<'a>, source_scales: Option<Allocation<'a>> }

/// Shared TP2 storage. Final outputs are reserve planes read by a later
/// Spark reduction hook. Rank GEMM arenas and peer staging are unit scratch;
/// rank_released orders next writers after both rank readers join. The 1E
/// per-destination released guards remain on the output-add transfer. Launch
/// bookkeeping is shared under the submission mutex and GPU guarded, never
/// presumed idle merely because its Rust future returned under device ordering.
#[cfg(test)]
thread_local! { static SKIP_RANK_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub(crate) struct Storage<'a> {
    execution: tokio::sync::Mutex<Execution<'a>>,
    rows: u32,
    unit: u32,
    layers: usize,
}
impl<'a> Storage<'a> {
    pub(crate) fn storage_bytes(library: &cuteafd_ffi::NativeLibrary, rows: u32, unit: u32) -> Result<usize> {
        ensure!(unit > 0 && unit <= 4096 && unit <= rows
            && rows as usize <= crate::shared::memory::row_window::MAX_RESERVE_ROWS, "invalid shared TP2 storage geometry");
        Ok(RankWave::device_bytes(library, unit)? + unit as usize * 10240 + rows as usize * 10240)
    }
    pub(crate) fn new(weights: [Rc<Vec<Weights<'a>>>; 2], rows: u32, unit: u32) -> Result<Rc<Self>> {
        let first = weights[0].first().context("shared TP2 weights absent")?;
        Self::storage_bytes(first.device.library, rows, unit)?;
        let layers = weights[0].len();
        Ok(Rc::new(Self { execution: tokio::sync::Mutex::new(Execution::new(weights, rows, unit)?), rows, unit, layers }))
    }
}
pub(crate) struct Wave<'a> {
    storage: Rc<Storage<'a>>,
    window: crate::shared::memory::row_window::RowWindow,
}
impl<'a> Wave<'a> {
    pub(crate) fn device_bytes(library: &cuteafd_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        Storage::storage_bytes(library, capacity, capacity)
    }
    pub(crate) fn new(weights: [Rc<Vec<Weights<'a>>>; 2], capacity: u32) -> Result<Self> {
        Ok(Self::on_storage(Storage::new(weights, capacity, capacity)?))
    }
    pub(crate) fn on_storage(storage: Rc<Storage<'a>>) -> Self {
        Self { storage, window: crate::shared::memory::row_window::RowWindow::NONE }
    }
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows as usize
                && range.len() <= self.storage.unit as usize, "shared TP2 window exceeds reserve/unit");
        }
        self.window = window; Ok(())
    }
    pub(crate) fn contains(&self, layer: usize) -> bool { layer < self.storage.layers }
    pub(crate) fn release_output(&self, destination: usize) -> Result<()> {
        self.storage.execution.try_lock().context("shared TP2 submission still borrowed")?.release_output(destination)
    }
    pub(crate) fn release_scratch(&self) -> Result<()> {
        self.storage.execution.try_lock().context("shared TP2 submission still borrowed")?.rank_released.release()
    }
    /// # Safety
    /// Inputs remain live and immutable until the enclosing chain drains; the
    /// caller consumes the returned plane before reusing this slot's window.
    pub(crate) async unsafe fn execute(&mut self, layer: usize, rows: u32, destination: usize,
        inputs: [CuteafdDeviceBuffer; 2], producers: [&Stream<'a>; 2]) -> Result<CuteafdDeviceBuffer> {
        ensure!(self.window.admits(rows as usize), "shared TP2 rows differ from window");
        let mut execution = self.storage.execution.lock().await;
        execution.window = self.window.clone();
        unsafe { execution.execute(layer, rows, destination, inputs, producers).await }
    }
}

/// Complete shared TP2 operation for one lane, with either GPU as destination.
struct Execution<'a> {
    transfers: [PeerTransfer<'a>; 2],
    ready: [Event<'a>; 2],
    staging: [Allocation<'a>; 2],
    output: [Allocation<'a>; 2],
    /// Per destination: the release of `output`'s last reader (the Spark
    /// reduce on another stream), which the next `add` acquires instead of
    /// joining the whole chain head (hunt A 1E).
    released: [crate::shared::memory::row_window::ScratchGuard<'a>; 2],
    ranks: [RankWave<'a>; 2],
    add: Bf16Add<'a>,
    capacity: u32,
    window: crate::shared::memory::row_window::RowWindow,
    rank_released: crate::shared::memory::row_window::ScratchGuard<'a>,
}
impl<'a> Execution<'a> {
    /// The reader of `destination`'s output (the reduce) finished on the
    /// unit's chain: the next `add` there may overwrite it.
    pub(crate) fn release_output(&self, destination: usize) -> Result<()> {
        ensure!(destination < 2, "invalid shared TP2 destination");
        self.released[destination].release()
    }
    pub fn contains(&self, layer: usize) -> bool {
        self.ranks.iter().all(|rank| rank.weights.get(layer).is_some_and(|w| w.layer == layer))
    }
    fn new(weights: [Rc<Vec<Weights<'a>>>; 2], rows: u32, capacity: u32) -> Result<Self> {
        ensure!(!weights[0].is_empty() && weights[0].len() == weights[1].len(), "shared TP2 rank layers differ");
        let d0 = weights[0][0].device;
        let d1 = weights[1][0].device;
        ensure!(d0.id == 0 && d1.id == 1 && std::ptr::eq(d0.library, d1.library), "shared TP2 rank devices differ");
        let ranks = [RankWave::new(weights[0].clone(), capacity)?, RankWave::new(weights[1].clone(), capacity)?];
        let bytes = capacity as usize * 5120 * 2;
        Ok(Self {
            transfers: [PeerTransfer::new(d1,d0)?, PeerTransfer::new(d0,d1)?],
            ready: [Event::new(d0)?, Event::new(d1)?],
            staging: [Allocation::new(d0,bytes,Tag::staging("v41/shared-tp2"))?, Allocation::new(d1,bytes,Tag::staging("v41/shared-tp2"))?],
            output: [Allocation::new(d0,rows as usize*10240,Tag::workspace("v41/shared-tp2"))?, Allocation::new(d1,rows as usize*10240,Tag::workspace("v41/shared-tp2"))?],
            released: [crate::shared::memory::row_window::ScratchGuard::new(d0.library, &[0, 1])?,
                crate::shared::memory::row_window::ScratchGuard::new(d0.library, &[0, 1])?],
            ranks, add: d0.library.bf16_add()?, capacity,
            window: crate::shared::memory::row_window::RowWindow::NONE,
            rank_released: crate::shared::memory::row_window::ScratchGuard::new(d0.library, &[0, 1])?,
        })
    }
    /// # Safety
    /// Inputs are produced on the supplied streams and remain unchanged through
    /// completion. Each lane owns a separate Wave. Returned output remains valid
    /// only until the next execute or destruction of this owner.
    pub async unsafe fn execute(&mut self, layer: usize, rows: u32, destination: usize,
        inputs: [CuteafdDeviceBuffer; 2], producers: [&Stream<'a>; 2]) -> Result<CuteafdDeviceBuffer> {
        ensure!(destination < 2 && rows > 0 && rows <= self.capacity, "invalid shared TP2 destination/rows");
        struct Drain<'s, 'a> { ranks: &'s mut [RankWave<'a>; 2], complete: bool }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    for rank in self.ranks.iter() {
                        rank.stream.drain_or_abort("shared TP2 rank");
                    }
                }
            }
        }
        #[cfg(test)]
        let skip = SKIP_RANK_ACQUIRE.with(|s|s.get());
        #[cfg(not(test))]
        let skip = false;
        if !skip { for rank in &self.ranks {
            rank.stream.device.run(|| unsafe { self.rank_released.acquire(rank.stream.raw) })?;
        }}
        let mut drain = Drain { ranks: &mut self.ranks, complete: false };
        for rank in 0..2 {
            unsafe { drain.ranks[rank].enqueue(layer, rows, inputs[rank], producers[rank])?; }
        }
        let local = &drain.ranks[destination];
        let remote = &drain.ranks[1-destination];
        if !crate::shared::memory::chain::deferred() {
            remote.stream.wait().await?;
        }
        self.ready[destination].record(&local.stream)?;
        let ready = &self.ready[destination];
        let output = &self.output[destination];
        let output_buffer = self.window.apply(output.buffer, 10240);
        let add = &self.add;
        let count = rows as usize * 5120;
        unsafe { self.transfers[destination].copy_then_ordered(&remote.output, &mut self.staging[destination],
            &remote.stream, count*2, Some(&self.released[destination]), |peer, stream| {
                output.device.library.cuda_stream_wait_event(stream, ready.raw)?;
                add.launch(local.output.buffer, peer, output_buffer, count, stream)
            }).await?; }
        drain.complete = true;
        self.rank_released.release()?;
        let mut result = output_buffer;
        result.bytes = count*2;
        Ok(result)
    }
}

pub(crate) struct Weights<'a> {
    device: Device<'a>,
    layer: usize,
    matrices: [Matrix<'a>; 3],
}
impl<'a> Weights<'a> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        for (stem, matrix) in ["w1", "w3", "w2"].into_iter().zip(&self.matrices) {
            let name = format!("gpu{}.layers.{}.ffn.shared_experts.{stem}.weight", self.device.id, self.layer);
            matrix.weight.inventory(inventory, &name)?;
            matrix.scales.inventory(inventory,
                &crate::families::deepseek_v41::weight_layout::packed_scale_name(&name))?;
            if let Some(raw) = &matrix.source_scales { raw.inventory(inventory, &format!("{name}.transient_scales"))?; }
        }
        Ok(())
    }
    pub fn device(&self) -> Device<'a> { self.device }
    /// Frees raw scales kept for a recorded scale pack (after it ran).
    pub fn release_raw_scales(&mut self) { for m in &mut self.matrices { drop(m.source_scales.take()); } }
    pub fn device_bytes() -> usize { 3 * (5120 * 1152 + 5120 * 1152 / 32) }
    /// Including every matrix's raw scales (resident until their pack ran).
    pub fn load_peak_device_bytes() -> usize { Self::device_bytes() + 3 * (5120 * 1152 / 1024) }
    #[cfg(test)]
    pub fn load(device: Device<'a>, catalog: &OfficialV41Catalog, layer: usize, budget: usize) -> Result<Self> {
        Self::load_in(device, catalog, layer, budget, None)
    }
    pub fn load_in(device: Device<'a>, catalog: &OfficialV41Catalog, layer: usize, budget: usize,
        arena: Option<&crate::shared::memory::weight_arena::DeviceWeightArena<'a>>) -> Result<Self> {
        ensure!(layer < 40 && matches!(device.id, 0 | 1) && budget >= Self::load_peak_device_bytes(),
            "invalid TP2 shared weight layer, rank, or budget");
        let stream = Stream::new(device)?;
        let dense = crate::shared::experts::rtx::dense::DenseLoad::mount(device.library);
        let mut matrices = Vec::with_capacity(3);
        for name in ["w1", "w3", "w2"] {
            let (axis, k, n) = if name == "w2" { (1, 1152, 5120) } else { (0, 5120, 1152) };
            let kernel = device.run(|| device.library.v41_fp8_matrix_kernel(1, k, n))?;
            let prefix = format!("layers.{layer}.ffn.shared_experts.{name}");
            let weight_name = format!("{prefix}.weight");
            let weight = crate::shared::memory::weight_arena::weight_allocation(
                device, arena, &weight_name, (k*n) as usize, ArenaTag::weights("v41/target"))?;
            let scales = crate::shared::memory::weight_arena::weight_allocation(
                device, arena, &crate::families::deepseek_v41::weight_layout::packed_scale_name(&weight_name),
                (k*n/32) as usize, ArenaTag::weights("v41/target"))?;
            let source_scales = Allocation::new(device, (k*n/1024) as usize, Tag::staging("v41/shared-tp2-load"))?;
            for (suffix, target) in [("weight", &weight), ("scale", &source_scales)] {
                let rect = super::super::v41_tensors::half(catalog, &format!("{prefix}.{suffix}"), axis, device.id as usize)?;
                ensure!(rect.bytes() == target.buffer.bytes, "shared TP2 tensor has unexpected size");
                dense.operand(crate::shared::experts::rtx::dense::Operand { device: device.id, buffer: target.buffer,
                    lands: vec![(rect, 0)], bytes: target.buffer.bytes })?;
            }
            let packed = device.run(|| unsafe { crate::shared::experts::rtx::dense::v41_pack_scales(&kernel,
                source_scales.buffer, scales.buffer, stream.raw) });
            let drained = stream.drain();
            packed.and(drained)?;
            // Through the mount the pack already drained: release the raw scales now.
            let source_scales = crate::shared::experts::rtx::dense::is_recording().then_some(source_scales);
            matrices.push(Matrix { weight, scales, source_scales });
        }
        Ok(Self { device, layer, matrices: matrices.try_into().ok().unwrap() })
    }
}

pub(crate) struct RankWave<'a> {
    pub stream: Stream<'a>,
    ready: Event<'a>,
    weights: Rc<Vec<Weights<'a>>>,
    up_kernel: V41Fp8Plan<'a>,
    down_kernel: V41Fp8Plan<'a>,
    activation: V41SharedSwiGlu<'a>,
    up_scratch: Allocation<'a>,
    down_scratch: Allocation<'a>,
    alpha: Allocation<'a>,
    gate: Allocation<'a>,
    up: Allocation<'a>,
    intermediate: Allocation<'a>,
    pub output: Allocation<'a>,
    capacity: u32,
}
impl<'a> RankWave<'a> {
    fn allocation_sizes(up: usize, down: usize, unit: u32) -> [usize;7] {
        [up, down, 16, unit as usize*2304, unit as usize*2304, unit as usize*2304, unit as usize*10240]
    }
    pub fn device_bytes(library: &cuteafd_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        ensure!((1..=4096).contains(&capacity), "invalid shared TP2 capacity");
        let up = library.v41_fp8_matrix_plan_info(capacity,5120,1152)?;
        let down = library.v41_fp8_matrix_plan_info(capacity,1152,5120)?;
        Ok(Self::allocation_sizes(up.scratch_bytes as usize, down.scratch_bytes as usize, capacity).into_iter().sum())
    }
    pub fn new(weights: Rc<Vec<Weights<'a>>>, capacity: u32) -> Result<Self> {
        ensure!(!weights.is_empty() && weights.len() <= 40, "shared TP2 needs resident layers");
        let device = weights[0].device;
        ensure!(weights.iter().enumerate().all(|(layer,w)| w.layer == layer && w.device.id == device.id
            && std::ptr::eq(w.device.library, device.library)), "shared TP2 weight owners differ");
        let up_kernel = device.run(|| device.library.v41_fp8_matrix_plan(capacity, 5120, 1152))?;
        let down_kernel = device.run(|| device.library.v41_fp8_matrix_plan(capacity, 1152, 5120))?;
        let stream = Stream::new(device)?;
        let [up,down,a,gate,u,intermediate,output] = Self::allocation_sizes(
            up_kernel.info().scratch_bytes as usize, down_kernel.info().scratch_bytes as usize, capacity);
        let up_scratch = Allocation::new(device, up, Tag::workspace("v41/shared-tp2"))?;
        let down_scratch = Allocation::new(device, down, Tag::workspace("v41/shared-tp2"))?;
        let alpha = Allocation::new(device, a, Tag::workspace("v41/shared-tp2"))?;
        let initialized = device.run(|| unsafe {
            up_kernel.initialize_scratch(up_scratch.buffer, alpha.buffer, stream.raw)?;
            down_kernel.initialize_scratch(down_scratch.buffer, alpha.buffer, stream.raw)
        });
        let drained = stream.drain();
        initialized.and(drained)?;
        Ok(Self { stream, ready: Event::new(device)?, weights, up_kernel, down_kernel,
            activation: device.library.v41_shared_tp2_swiglu()?, up_scratch, down_scratch, alpha,
            gate: Allocation::new(device, gate, Tag::workspace("v41/shared-tp2"))?,
            up: Allocation::new(device, u, Tag::workspace("v41/shared-tp2"))?,
            intermediate: Allocation::new(device, intermediate, Tag::workspace("v41/shared-tp2"))?,
            output: Allocation::new(device, output, Tag::workspace("v41/shared-tp2"))?, capacity })
    }
    /// # Safety
    /// Previous use has completed. Input writes are ordered on producer, and
    /// input storage stays live/unchanged until this stream finishes.
    pub unsafe fn enqueue(&mut self, layer: usize, rows: u32, input: CuteafdDeviceBuffer,
        producer: &Stream<'a>) -> Result<()> {
        let device = self.output.device;
        ensure!(rows > 0 && rows <= self.capacity && layer < self.weights.len()
            && input.device_id == device.id && input.bytes >= rows as usize * 5120 * 2,
            "invalid shared TP2 input or layer");
        self.ready.record(producer)?;
        let matrices = &self.weights[layer].matrices;
        let queued = device.run(|| unsafe {
            device.library.cuda_stream_wait_event(self.stream.raw, self.ready.raw)?;
            for (matrix, output) in [(&matrices[0], &self.gate), (&matrices[1], &self.up)] {
                self.up_kernel.launch(input, matrix.weight.buffer, matrix.scales.buffer,
                    self.up_scratch.buffer, self.alpha.buffer, output.buffer, rows, self.stream.raw)
                    .with_context(|| format!("shared TP2 up/gate on GPU {}", device.id))?;
            }
            self.activation.launch(self.gate.buffer, self.up.buffer, self.intermediate.buffer, rows, self.stream.raw)?;
            self.down_kernel.launch(self.intermediate.buffer, matrices[2].weight.buffer, matrices[2].scales.buffer,
                self.down_scratch.buffer, self.alpha.buffer, self.output.buffer, rows, self.stream.raw)
                .with_context(|| format!("shared TP2 down on GPU {}", device.id))
        });
        if let Err(error) = queued { self.stream.drain()?; return Err(error); }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires shared TP2 CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT, two GPUs"]
    fn real_shared_tp2_complete_independent_lanes() -> Result<()> {
        let capacity: u32 = std::env::var("CUTEAFD_TP2_TEST_CAPACITY").unwrap_or_else(|_| "16".into()).parse()?;
        ensure!((1..=4096).contains(&capacity), "invalid fixture capacity");
        let lib = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        let weights = [
            Rc::new(vec![Weights::load(devices[0], &catalog, 0, Weights::load_peak_device_bytes())?]),
            Rc::new(vec![Weights::load(devices[1], &catalog, 0, Weights::load_peak_device_bytes())?]),
        ];
        let mut lanes = [Wave::new(weights.clone(), capacity)?, Wave::new(weights, capacity)?];
        let inputs = [Allocation::new(devices[0], capacity as usize*5120*2, crate::shared::memory::TEST)?, Allocation::new(devices[1], capacity as usize*5120*2, crate::shared::memory::TEST)?];
        let producers = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for (rows, nonzero) in [(1, false), (capacity, true), (1, true), (capacity, false)] {
            let host: Vec<u8> = (0..capacity as usize*5120).flat_map(|i| {
                let value = if nonzero { (i % 31) as f32 / 32.0 - 0.5 } else { 0.0 };
                ((value.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            for input in &inputs { input.device.run(|| lib.copy_h2d(input.buffer, &host))?; }
            let [left, right] = &mut lanes;
            let (a,b) = runtime.block_on(async { tokio::join!(
                unsafe { left.execute(0, rows, 0, [inputs[0].buffer, inputs[1].buffer], [&producers[0], &producers[1]]) },
                unsafe { right.execute(0, rows, 1, [inputs[0].buffer, inputs[1].buffer], [&producers[0], &producers[1]]) }
            ) });
            let outputs = [a?,b?];
            let mut actual = [vec![0; rows as usize*5120*2], vec![0; rows as usize*5120*2]];
            for rank in 0..2 { devices[rank].run(|| lib.copy_d2h_quiescent(&mut actual[rank], outputs[rank]))?; }
            assert_eq!(actual[0], actual[1], "destination changed shared output");
            let mut partials = [vec![0; actual[0].len()], vec![0; actual[0].len()]];
            for rank in 0..2 {
                let mut buffer = lanes[0].storage.execution.try_lock()?.ranks[rank].output.buffer;
                buffer.bytes = partials[rank].len();
                devices[rank].run(|| lib.copy_d2h_quiescent(&mut partials[rank], buffer))?;
            }
            for ((a,b),actual) in partials[0].chunks_exact(2).zip(partials[1].chunks_exact(2)).zip(actual[0].chunks_exact(2)) {
                let decode = |v: &[u8]| f32::from_bits((u16::from_ne_bytes([v[0],v[1]]) as u32) << 16);
                let sum = decode(a) + decode(b);
                assert!(sum.is_finite());
                let bits = sum.to_bits();
                let rounded = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
                assert_eq!(u16::from_ne_bytes([actual[0],actual[1]]), rounded);
            }
            assert_eq!(actual[0].chunks_exact(2).any(|v| u16::from_ne_bytes([v[0],v[1]]) & 0x7fff != 0), nonzero);
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires shared TP2 CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT, two GPUs"]
    fn real_shared_tp2_rank_loading_and_execution() -> Result<()> {
        let lib = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let mut ranks = Vec::new();
        for id in 0..2 {
            let device = Device { library: &lib, id };
            let weights = Rc::new(vec![Weights::load(device, &catalog, 0, Weights::load_peak_device_bytes())?]);
            ranks.push((RankWave::new(weights.clone(), 16)?, RankWave::new(weights, 16)?,
                Allocation::new(device, 16*5120*2, crate::shared::memory::TEST)?, Stream::new(device)?));
        }
        for nonzero in [false, true] {
            for (first, second, input, producer) in &mut ranks {
                let value = if nonzero { 0x3f00u16 } else { 0 };
                let host: Vec<u8> = value.to_ne_bytes().into_iter().cycle().take(input.buffer.bytes).collect();
                input.device.run(|| lib.copy_h2d(input.buffer, &host))?;
                unsafe {
                    first.enqueue(0, 16, input.buffer, producer)?;
                    second.enqueue(0, 16, input.buffer, producer)?;
                }
            }
            for (first, second, input, _) in &ranks {
                first.stream.drain()?;
                second.stream.drain()?;
                let mut a = vec![0; first.output.buffer.bytes];
                let mut b = vec![0; second.output.buffer.bytes];
                input.device.run(|| {
                    lib.copy_d2h_quiescent(&mut a, first.output.buffer)?;
                    lib.copy_d2h_quiescent(&mut b, second.output.buffer)
                })?;
                assert_eq!(a,b);
                assert!(a.chunks_exact(2).all(|v| u16::from_ne_bytes([v[0],v[1]]) & 0x7f80 != 0x7f80));
                assert_eq!(a.chunks_exact(2).any(|v| u16::from_ne_bytes([v[0],v[1]]) & 0x7fff != 0), nonzero);
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
}

#[cfg(test)]
impl Weights<'_> {
    pub(crate) fn visit(&self, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        for (name, m) in ["w1", "w3", "w2"].iter().zip(&self.matrices) {
            let p = format!("gpu{}.shared_tp2.layers.{}.{name}.", self.device.id, self.layer);
            f(format!("{p}weight"), m.weight.buffer);
            f(format!("{p}packed_scale"), m.scales.buffer);
        }
    }
}

#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use crate::shared::memory::{chain::StageChain,row_window::RowWindow};
    fn run(guarded: bool)->Result<()> {
        let lib=unsafe {cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)?};
        lib.cuda_set_device(0)?;
        let catalog=cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let devices=[Device {library:&lib,id:0},Device {library:&lib,id:1}];
        let weights=[Rc::new(vec![Weights::load(devices[0],&catalog,0,Weights::load_peak_device_bytes())?]),
            Rc::new(vec![Weights::load(devices[1],&catalog,0,Weights::load_peak_device_bytes())?])];
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        let inputs=[ [Allocation::new(devices[0],16*10240, crate::shared::memory::TEST)?,Allocation::new(devices[1],16*10240, crate::shared::memory::TEST)?],
            [Allocation::new(devices[0],16*10240, crate::shared::memory::TEST)?,Allocation::new(devices[1],16*10240, crate::shared::memory::TEST)?] ];
        for (seed,pair) in inputs.iter().enumerate() {for input in pair {
            input.device.run(||lib.copy_h2d(input.buffer,&(if seed==0 {0x3e80u16}else {0xbe80u16}).to_ne_bytes().repeat(16*5120)))?;
        }}
        let producers=[Stream::new(devices[0])?,Stream::new(devices[1])?];
        let mut private=Wave::new(weights.clone(),16)?;
        runtime.block_on(unsafe {private.execute(0,16,0,inputs[0].each_ref().map(|a|a.buffer),[&producers[0],&producers[1]])})?;
        let private_rank=private.storage.execution.try_lock().unwrap().ranks[0].output.buffer;
        let mut want=vec![0;16*10240];lib.copy_d2h_quiescent(&mut want,private_rank)?;
        let storage=Storage::new(weights,48,16)?;
        let mut slots=[Wave::on_storage(storage.clone()),Wave::on_storage(storage.clone())];
        for (i,slot) in slots.iter_mut().enumerate() {
            slot.select_window(RowWindow::within(i*16..i*16+16,48)?)?;
            runtime.block_on(unsafe {slot.execute(0,16,i,inputs[i].each_ref().map(|a|a.buffer),[&producers[0],&producers[1]])})?;
        }
        let rank=storage.execution.try_lock().unwrap().ranks[0].output.buffer;
        let reader=Stream::new(devices[1])?;let saved=Allocation::new(devices[1],want.len(), crate::shared::memory::TEST)?;
        devices[1].run(||unsafe {lib.copy_peer_async(saved.buffer,rank,want.len(),reader.raw)})?;reader.drain()?;
        let chains=[StageChain::on_devices(&lib,&[0,1])?,StageChain::on_devices(&lib,&[0,1])?];
        let gate=devices[1].run(||unsafe {cuteafd_ffi::test_support::CudaStreamGate::new(&lib,reader.raw)})?;
        runtime.block_on(chains[0].handle().scope(async {
            unsafe {slots[0].execute(0,16,0,inputs[0].each_ref().map(|a|a.buffer),[&producers[0],&producers[1]]).await?;}
            devices[1].run(||unsafe {crate::shared::memory::chain::join(&lib,reader.raw)?;
                lib.copy_peer_async(saved.buffer,rank,want.len(),reader.raw)?;
                crate::shared::memory::chain::finish(&lib,reader.raw)})?;
            slots[0].release_scratch()
        }))?;
        SKIP_RANK_ACQUIRE.with(|s|s.set(!guarded));
        let queued=runtime.block_on(chains[1].handle().scope(async {
            unsafe {slots[1].execute(0,16,1,inputs[1].each_ref().map(|a|a.buffer),[&producers[0],&producers[1]]).await?;}
            anyhow::Ok(())
        }));SKIP_RANK_ACQUIRE.with(|s|s.set(false));queued?;
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(if guarded {2}else {10});let mut done=false;
        while std::time::Instant::now()<deadline {
            let execution=storage.execution.try_lock().unwrap();let stream=&execution.ranks[0].stream;
            if stream.device.run(||unsafe {lib.cuda_stream_query(stream.raw)})? {done=true;break;}
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        devices[1].run(|| {drop(gate);Ok(())})?;for chain in &chains {chain.drain()?;}reader.drain()?;
        assert_eq!(done,!guarded,"shared TP2 rank B completion before reader release");
        let mut got=vec![0;want.len()];devices[1].run(||lib.copy_d2h_quiescent(&mut got,saved.buffer))?;
        if guarded {assert_eq!(got,want);}else {assert_ne!(got,want,"shared rank negative did not clobber");}
        Ok(())
    }
    #[test]
    #[ignore = "requires shared TP2 native kernels, snapshot and two GPUs"]
    fn shared_tp2_rank_scratch_preserves_gated_peer_reader()->Result<()> {for _ in 0..3 {run(true)?;} Ok(())}
    #[test]
    #[ignore = "requires shared TP2 native kernels, snapshot and two GPUs"]
    fn shared_tp2_without_rank_guard_clobbers_gated_peer_reader()->Result<()> {for _ in 0..3 {run(false)?;} Ok(())}
    #[test]
    fn shared_tp2_storage_bytes_equal_planned_allocations() {
        for unit in [4096,2752,2048] {
            let rank=RankWave::allocation_sizes(123,456,unit).into_iter().sum::<usize>();
            assert_eq!(rank+unit as usize*10240+8192*10240,123+456+16+unit as usize*27392+8192*10240);
        }
    }
}
