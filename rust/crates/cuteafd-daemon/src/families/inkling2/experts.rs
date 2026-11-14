//! Native coordinator experts; format-independent boundary retained by the forward.
use crate::shared::{experts::rtx::dense::{DenseLoad, Operand, Rect},
    memory::{device::{Allocation, Device, Stream}, receipts::Inventory, weight_arena::DeviceWeightArena}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, memory_layout::ArenaTag};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, fp8_moe::{Fp8MoeMetadata, Fp8MoeModule, Fp8MoeWeights}};
use cuteafd_loader::{families::inkling2::Inkling2Config, plan::Checkpoint,
    serving_capacity::WeightArenaLayout, shared::dense_read::{MountLoad, ReadBatch, STEP_BYTES}};
use std::{path::Path, rc::Rc};

/// The forward knows only normalized inputs/routes and the stream-ordered
/// output. EXL3 K4 can supply the same boundary without editing layer math.
pub(crate) trait Dispatcher {
    fn enqueue(&self, input: Buffer, ids: Buffer, weights: Buffer, output: Buffer,
        rows: usize, stream: &Stream<'_>) -> Result<()>;
}

pub(crate) struct Mxfp8Layout {
    pub arena: WeightArenaLayout,
    pub mount: MountLoad,
    lands: Vec<Vec<(Rect, usize)>>,
}
impl Mxfp8Layout {
    pub fn plan(checkpoint: &Checkpoint, cfg: &Inkling2Config, layer: usize) -> Result<Self> {
        ensure!(layer < cfg.layers && cfg.intermediate % 128 == 0 && cfg.hidden % 128 == 0,
            "Inkling MXFP8 layer geometry unsupported");
        let (e, h, i) = (cfg.experts, cfg.hidden, cfg.intermediate);
        let root = format!("model.llm.layers.{layer}.mlp.experts");
        let mut lands = Vec::new();
        let mut tensors = Vec::new();
        let mut mount = MountLoad::default();
        for (plane, projection, scale, part) in [(0, "w13", false, 0), (1, "w13", true, 0),
            (2, "w13", false, 1), (3, "w13", true, 1), (4, "w2", false, 0), (5, "w2", true, 0)] {
            let name = format!("{root}.{projection}_weight{}", if scale { "_scale" } else { "" });
            let header = checkpoint.header(&name).with_context(|| format!("Inkling expert requires {name}"))?;
            let dtype = if scale { DType::U8 } else { DType::F8E4M3 };
            let (rows, cols) = if projection == "w13" { (2*i, if scale { h/32 } else { h }) }
                else { (h, if scale { i/32 } else { i }) };
            ensure!(header.dtype == dtype && header.shape == [e, rows, cols]
                && header.byte_length == (e*rows*cols) as u64,
                "Inkling expert {name}: native MXFP8 header differs");
            let tensor = checkpoint.tensors.iter().find(|tensor| tensor.meta.name == name).context("Inkling tensor path")?;
            let (pitch, width, n) = if projection == "w13" { (2*cols, cols, i) } else { (cols, cols, h) };
            let bytes = e*n*width;
            tensors.push((plane.to_string(), bytes));
            let mut source = Vec::new();
            for expert in 0..e {
                source.push((Rect { name: name.clone(), path: checkpoint.snapshot.join(&tensor.shard),
                    base: header.byte_offset+(expert*rows*cols) as u64, row_bytes: pitch,
                    rows: 0..n, cols: part*cols..part*cols+width }, expert*n*width));
            }
            mount.merge(MountLoad::read(ReadBatch::new(n, pitch, width, STEP_BYTES)?));
            lands.push(source);
        }
        Ok(Self { arena: WeightArenaLayout::new("inkling2/golden/experts", tensors)?, mount, lands })
    }
}

pub(crate) struct Mxfp8<'a> {
    stream: Rc<Stream<'a>>,
    module: Fp8MoeModule,
    planes: Vec<Allocation<'a>>,
    scratch: Allocation<'a>,
    rows: usize,
}
impl<'a> Mxfp8<'a> {
    /// Metadata and all arena/workspace requests are admitted by the caller.
    pub fn load(stream: Rc<Stream<'a>>, layout: &Mxfp8Layout, metadata: &Fp8MoeMetadata,
        directory: &Path, cfg: &Inkling2Config, rows: usize) -> Result<Self> {
        let info = &metadata.info;
        ensure!(info.weights == Fp8MoeWeights::Mxfp8 && info.wire_input && info.tp == 1
            && (info.hidden, info.experts, info.topk, info.intermediate, info.slice)
                == (cfg.hidden, cfg.experts, cfg.topk, cfg.intermediate, cfg.intermediate),
            "Inkling native expert package differs from checkpoint");
        info.validate_activation(cuteafd_core::FfnActivation::Situ { beta: cfg.gate_softcap as f32,
            linear_beta: Some(cfg.up_softcap as f32) }, cuteafd_ffi::fp8_moe::Fp8MoeProjection::Bf16)?;
        let scratch_bytes = metadata.scratch_for(rows)?;
        let device: Device<'a> = stream.device;
        let arena = DeviceWeightArena::new(device, layout.arena.clone(), ArenaTag::weights("inkling2/experts"))?;
        let dense = DenseLoad::mount(device.library);
        dense.admit_mount(layout.mount)?;
        let mut planes = Vec::new();
        for (plane, lands) in layout.lands.iter().enumerate() {
            let view = layout.arena.view(&plane.to_string()).context("Inkling expert plane missing")?;
            let owner = arena.view(&plane.to_string(), view.bytes())?;
            dense.operand(Operand { device: device.id, buffer: owner.buffer, lands: lands.clone(), bytes: view.bytes() })?;
            planes.push(owner);
        }
        dense.release();
        let scratch = Allocation::new(device, scratch_bytes, cuteafd_core::memory_layout::Tag::workspace("inkling2/experts"))?;
        // SAFETY: trusted package, current device scoped by Device::run. Self
        // retains the package and drains its owning stream before unloading.
        let module = device.run(|| unsafe { Fp8MoeModule::load(directory) })?;
        ensure!(module.info() == info, "Inkling expert package changed since admission");
        Ok(Self { stream, module, planes, scratch, rows })
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (index, owner) in self.planes.iter().enumerate() { owner.inventory(inventory, &format!("{prefix}.{index}"))?; }
        self.scratch.inventory(inventory, "expert.scratch")
    }
}
impl Dispatcher for Mxfp8<'_> {
    fn enqueue(&self, input: Buffer, ids: Buffer, weights: Buffer, output: Buffer,
        rows: usize, stream: &Stream<'_>) -> Result<()> {
        ensure!(rows > 0 && rows <= self.rows && stream.raw == self.stream.raw
            && stream.device.id == self.stream.device.id, "Inkling expert stream/capacity differs");
        let info = self.module.info();
        for (buffer, bytes) in [(input, rows*info.wire_row_bytes()), (ids, rows*info.topk*4),
            (weights, rows*info.topk*4), (output, rows*info.hidden*2)] {
            ensure!(buffer.device_id == stream.device.id && buffer.bytes >= bytes && !buffer.ptr.is_null()
                && buffer.ptr as usize % 16 == 0 && buffer.flags == 0, "Inkling expert operand extent/device differs");
        }
        let pointers = [input.ptr, ids.ptr, weights.ptr, self.planes[0].buffer.ptr,
            self.planes[1].buffer.ptr, self.planes[2].buffer.ptr, self.planes[3].buffer.ptr,
            self.planes[4].buffer.ptr, self.planes[5].buffer.ptr, output.ptr, self.scratch.buffer.ptr];
        // SAFETY: admitted native planes, checked I/O extents and retained
        // stream/workspace/module owners. Caller orders route/input producers.
        unsafe { self.module.launch(&pointers, rows, stream.raw) }
    }
}
impl Drop for Mxfp8<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Inkling native expert module and weight arena"); }
}
