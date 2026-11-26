//! Serialized native ViT/aligner owner. No request KV or text execution state.
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, NativeLibrary, V41VisionOps};
use cuteafd_loader::{OfficialV41Catalog, V41Image, V41ImageGrid};
use cuteafd_core::memory_layout::ArenaTag;

#[derive(Clone, Copy)]
#[repr(usize)]
enum Slot {
    Patches,
    Hidden,
    Norm,
    Qkv,
    Q,
    K,
    V,
    Attention,
    Gates,
    Activated,
    Linear,
    Scores,
    ValueOrProbability,
    Merged,
    AlignedA,
    AlignedB,
    Span,
    Frequencies,
}
pub(crate) struct VisionRuntime<'a> {
    // Drain the stream before dropping the BLAS handle and any buffer/weights.
    stream: LoadStream<'a>,
    ops: V41VisionOps<'a>,
    _workspace: DeviceAllocation<'a>,
    weights: NativeRtxTensors<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    capacity: usize,
    fp32_attention: bool,
    ready: Option<V41ImageGrid>,
}
impl<'a> VisionRuntime<'a> {
    pub(super) fn names() -> Vec<String> {
        cuteafd_loader::placement::families::deepseek_v41::weight_layout::vision_names()
    }
    fn buffer_sizes(capacity: usize) -> Result<Vec<usize>> {
        ensure!(
            (1..=V41VisionOps::MAX_PATCHES).contains(&capacity),
            "invalid vision patch capacity"
        );
        Ok(vec![
            capacity * 588 * 2,
            capacity * 2048,
            capacity * 2048,
            capacity * 6144,
            capacity * 2048,
            capacity * 2048,
            capacity * 2048,
            capacity * 2048,
            capacity * 5632 * 2,
            capacity * 2816 * 2,
            (capacity * 3072).max(1024 * 5120) * 4,
            capacity * 16 * 128 * 4,
            capacity * 16 * 128 * 2,
            1024 * 9216 * 2,
            1024 * 5120 * 2,
            1024 * 5120 * 2,
            1024 * 5120 * 2,
            16 * 4,
        ])
    }
    pub fn weight_bytes(catalog: &OfficialV41Catalog) -> Result<usize> {
        let names = Self::names();
        for name in &names {
            ensure!(
                catalog.tensor(name)?.metadata.dtype == cuteafd_core::DType::Bf16,
                "vision tensor must retain BF16: {name}"
            );
        }
        Ok(Self::weight_layout(catalog)?.bytes())
    }
    fn weight_layout(catalog: &OfficialV41Catalog) -> Result<cuteafd_loader::serving_capacity::WeightArenaLayout> {
        // One header recipe for the tower's arena and for admission.
        cuteafd_loader::placement::families::deepseek_v41::weight_layout::vision_layout_from_headers(
            catalog.tensors().iter().map(|tensor| &tensor.metadata))
    }
    pub fn device_bytes(catalog: &OfficialV41Catalog, capacity: usize) -> Result<usize> {
        Ok(Self::weight_bytes(catalog)?
            + Self::buffer_sizes(capacity)?.iter().sum::<usize>()
            + V41VisionOps::WORKSPACE_BYTES)
    }
    pub fn new(
        lib: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        capacity: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(catalog, capacity)? <= budget,
            "vision owner exceeds device budget"
        );
        let names = Self::names();
        let layout = Self::weight_layout(catalog)?;
        // VisionRuntime is constructed inside its owning encoder thread/device.
        let arena = crate::shared::memory::weight_arena::DeviceWeightArena::new(
            crate::shared::memory::device::Device { library: lib, id: lib.cuda_get_device()? }, layout, ArenaTag::weights("v41/vision"))?;
        let weights = NativeRtxTensors::load_in(
            lib, catalog, &names, NativeRtxTensors::plan(catalog, &names)?, 16 << 20, Some(&arena),
        )?;
        super::weight_layout::probe_weights(lib, &[lib.cuda_get_device()?], "v41/vision-weights-loaded",
            |inventory| weights.inventory(inventory))?;
        let workspace = DeviceAllocation::new(lib, V41VisionOps::WORKSPACE_BYTES, Tag::workspace("v41/vision"))?;
        let ops = unsafe { lib.v41_vision_ops(workspace.buffer)? };
        let buffers = Self::buffer_sizes(capacity)?
            .into_iter()
            .map(|n| DeviceAllocation::new(lib, n, Tag::workspace("v41/vision")))
            .collect::<Result<Vec<_>>>()?;
        let stream = LoadStream {
            library: lib,
            raw: lib.cuda_stream_create()?,
        };
        let value = Self {
            stream,
            ops,
            _workspace: workspace,
            weights,
            buffers,
            capacity,
            fp32_attention: false,
            ready: None,
        };
        let frequencies: Vec<u8> = (0..16)
            .flat_map(|i| (1.0f32 / 10000.0f32.powf(i as f32 / 16.0)).to_le_bytes())
            .collect();
        lib.copy_h2d(value.buffer(Slot::Frequencies), &frequencies)?;
        Ok(value)
    }
    fn buffer(&self, slot: Slot) -> Buffer {
        self.buffers[slot as usize].buffer
    }
    fn linear(
        &self,
        input: Slot,
        name: &str,
        bias: bool,
        output: Slot,
        rows: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        unsafe {
            self.ops.linear(
                self.buffer(input),
                self.weights.get(&format!("{name}.weight"))?,
                if bias {
                    Some(self.weights.get(&format!("{name}.bias"))?)
                } else {
                    None
                },
                self.buffer(Slot::Linear),
                self.buffer(output),
                rows,
                k,
                n,
                self.stream.raw,
            )
        }
    }
    fn norm(&self, input: Slot, name: &str, output: Slot, rows: usize) -> Result<()> {
        unsafe {
            self.ops.norm(
                self.buffer(input),
                self.weights.get(name)?,
                self.buffer(output),
                rows,
                self.stream.raw,
            )
        }
    }
    fn element(
        &self,
        input: Slot,
        other: Option<Slot>,
        output: Slot,
        rows: usize,
        mode: i32,
    ) -> Result<()> {
        unsafe {
            self.ops.element(
                self.buffer(input),
                other.map(|s| self.buffer(s)),
                self.buffer(output),
                rows,
                mode,
                self.stream.raw,
            )
        }
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    /// Returns a completed span borrowed from this owner's reusable device storage.
    /// Copy or consume it before encoding another image or dropping this owner.
    pub fn encode(&mut self, image: &V41Image) -> Result<Buffer> {
        self.encode_patches(image.patches(), image.grid(), None)
    }
    pub(super) fn stream(&self) -> &LoadStream<'a> {
        &self.stream
    }

    // Optional observer is only for numerical qualification, and sees completed
    // outputs. Normal execution queues the entire image before one final drain.
    pub(super) fn encode_patches(
        &mut self,
        patches: &[u8],
        grid: V41ImageGrid,
        mut observer: Option<&mut dyn FnMut(&str, Buffer) -> Result<()>>,
    ) -> Result<Buffer> {
        self.ready = None;
        let rows = grid
            .vit_height
            .checked_mul(grid.vit_width)
            .ok_or_else(|| anyhow::anyhow!("vision grid overflow"))?;
        ensure!(
            grid.vit_height > 0
                && grid.vit_width > 0
                && rows <= self.capacity
                && grid.llm_height == grid.vit_height.div_ceil(3)
                && grid.llm_width == grid.vit_width.div_ceil(3)
                && grid.tokens() <= 1024
                && patches.len() == rows * 588 * 2,
            "invalid vision patches or grid"
        );
        let mut destination = self.buffer(Slot::Patches);
        destination.bytes = patches.len();
        self.stream.library.copy_h2d(destination, patches)?;
        let run = (|| -> Result<()> {
            let observe = |name: &str,
                           slot: Slot,
                           count: usize,
                           observer: &mut Option<&mut dyn FnMut(&str, Buffer) -> Result<()>>|
             -> Result<()> {
                if let Some(observer) = observer.as_deref_mut() {
                    self.synchronize()?;
                    let mut buffer = self.buffer(slot);
                    buffer.bytes = count;
                    observer(name, buffer)?;
                }
                Ok(())
            };
            self.linear(
                Slot::Patches,
                "vision.patch_embed.proj",
                true,
                Slot::Hidden,
                rows,
                588,
                1024,
            )?;
            observe("patch", Slot::Hidden, rows * 2048, &mut observer)?;
            for layer in 0..32 {
                let prefix = format!("vision.blocks.{layer}");
                self.norm(
                    Slot::Hidden,
                    &format!("{prefix}.norm1.weight"),
                    Slot::Norm,
                    rows,
                )?;
                self.linear(
                    Slot::Norm,
                    &format!("{prefix}.attn.wqkv"),
                    true,
                    Slot::Qkv,
                    rows,
                    1024,
                    3072,
                )?;
                unsafe {
                    self.ops.rope(
                        self.buffer(Slot::Qkv),
                        self.buffer(Slot::Frequencies),
                        self.buffer(Slot::Q),
                        self.buffer(Slot::K),
                        self.buffer(Slot::V),
                        grid.vit_height,
                        grid.vit_width,
                        self.stream.raw,
                    )?;
                }
                unsafe {
                    self.ops.attention(
                        self.buffer(Slot::Q),
                        self.buffer(Slot::K),
                        self.buffer(Slot::V),
                        self.buffer(Slot::Scores),
                        self.buffer(Slot::ValueOrProbability),
                        self.buffer(Slot::Linear),
                        self.buffer(Slot::Attention),
                        rows,
                        self.fp32_attention,
                        self.stream.raw,
                    )?;
                }
                self.linear(
                    Slot::Attention,
                    &format!("{prefix}.attn.wo"),
                    true,
                    Slot::Norm,
                    rows,
                    1024,
                    1024,
                )?;
                self.element(Slot::Hidden, Some(Slot::Norm), Slot::Hidden, rows, 0)?;
                self.norm(
                    Slot::Hidden,
                    &format!("{prefix}.norm2.weight"),
                    Slot::Norm,
                    rows,
                )?;
                self.linear(
                    Slot::Norm,
                    &format!("{prefix}.mlp.w1"),
                    false,
                    Slot::Gates,
                    rows,
                    1024,
                    5632,
                )?;
                self.element(Slot::Gates, None, Slot::Activated, rows, 1)?;
                self.linear(
                    Slot::Activated,
                    &format!("{prefix}.mlp.w2"),
                    false,
                    Slot::Norm,
                    rows,
                    2816,
                    1024,
                )?;
                self.element(Slot::Hidden, Some(Slot::Norm), Slot::Hidden, rows, 0)?;
                observe(
                    &format!("block-{layer}"),
                    Slot::Hidden,
                    rows * 2048,
                    &mut observer,
                )?;
            }
            self.norm(Slot::Hidden, "vision.norm.weight", Slot::Norm, rows)?;
            observe("norm", Slot::Norm, rows * 2048, &mut observer)?;
            let aligned = grid.llm_height * grid.llm_width;
            unsafe {
                self.ops.merge(
                    self.buffer(Slot::Norm),
                    self.buffer(Slot::Merged),
                    grid.vit_height,
                    grid.vit_width,
                    self.stream.raw,
                )?;
            }
            observe("merged", Slot::Merged, aligned * 18432, &mut observer)?;
            self.linear(
                Slot::Merged,
                "aligner.w1",
                true,
                Slot::AlignedA,
                aligned,
                9216,
                5120,
            )?;
            self.element(Slot::AlignedA, None, Slot::AlignedA, aligned, 2)?;
            self.linear(
                Slot::AlignedA,
                "aligner.w2",
                true,
                Slot::AlignedB,
                aligned,
                5120,
                5120,
            )?;
            observe("aligned", Slot::AlignedB, aligned * 10240, &mut observer)?;
            unsafe {
                self.ops.span(
                    self.buffer(Slot::AlignedB),
                    self.weights.get("image_start")?,
                    self.weights.get("image_newline")?,
                    self.weights.get("image_end")?,
                    self.buffer(Slot::Span),
                    grid.llm_height,
                    grid.llm_width,
                    self.stream.raw,
                )?;
            }
            Ok(())
        })();
        // Drain even on a failed launch before allowing buffers to be reused.
        run.and(self.synchronize())?;
        self.ready = Some(grid);
        let mut output = self.buffer(Slot::Span);
        output.bytes = grid.tokens() * 10240;
        Ok(output)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod push_tests {
    use super::*;
    use anyhow::Context;

    /// The coordinator-local tower's serving constructor recorded (as its owner
    /// thread does under push) and replayed through one committed coordinator
    /// rank must land byte-identical owners to the mount, without a payload read.
    #[test]
    #[ignore = "requires CUTEAFD_PUSH_V41 (SparkNest V4.1 snapshot), CUTEAFD_NATIVE_LIB and an RTX GPU"]
    fn installed_vision_tower_push_matches_mount_byte_for_byte() -> Result<()> {
        use crate::shared::{experts::rtx::dense, memory::device::Device};
        let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_V41").map(std::path::PathBuf::from) else { return Ok(()) };
        // SAFETY: the slot's native library, loaded once for this test process.
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let names: std::collections::HashSet<String> = VisionRuntime::names().into_iter().collect();
        let catalog = cuteafd_loader::read_official_v41_role_catalog(&snapshot, "vision", |name| names.contains(name))?;
        let (root, _) = cuteafd_loader::push_load::namespace(&snapshot).context("V4.1 outside SparkNest")?;
        let capacity = 1024;
        let bytes = VisionRuntime::device_bytes(&catalog, capacity)?;
        let mounted = VisionRuntime::new(&lib, &catalog, capacity, bytes)?;
        let (pushed, records) = dense::recording(|| VisionRuntime::new(&lib, &catalog, capacity, bytes))?;
        anyhow::ensure!(records.steps.is_empty() && records.finishes.is_empty() && !records.operands.is_empty(),
            "the tower records only direct BF16 operands");
        let staging = dense::RankLayout::new(records.operands.clone(), Vec::new(), 0)?.host_staging;
        anyhow::ensure!(staging.bytes()? == 0, "the tower adds no dense pinned staging");
        // Through the same address-only hand-off the owner thread uses.
        let sent = crate::shared::vision::push::Contribution::send_records(records.clone())?;
        let (contribution, _owner) = crate::shared::vision::push::Contribution::recorded(sent, std::sync::mpsc::channel().0,
            std::sync::mpsc::sync_channel(1).1, std::thread::spawn(|| ()))?;
        let replayed = contribution.records();
        anyhow::ensure!(replayed.operands.len() == records.operands.len(), "hand-off lost operands");
        drop(contribution);
        let sockets = tempfile::tempdir()?;
        crate::shared::experts::rtx::push::tests::receive_pushed(Device { library: &lib, id: 0 }, None, &root,
            sockets.path(), &replayed, None, 0, 0)?;
        let digest = |runtime: &VisionRuntime<'_>| -> Result<Vec<(String, Vec<u8>)>> {
            let mut out = Vec::new();
            let mut error = None;
            runtime.weights.visit("", &mut |name, buffer| {
                let mut bytes = vec![0u8; buffer.bytes];
                match lib.copy_d2h_quiescent(&mut bytes, buffer) { Ok(()) => out.push((name, bytes)), Err(e) => error = Some(e) }
            });
            error.map_or(Ok(out), Err)
        };
        let (expected, actual) = (digest(&mounted)?, digest(&pushed)?);
        anyhow::ensure!(expected.len() == actual.len() && expected.len() == names.len(), "tower owner inventories differ");
        let mut identical = 0usize;
        for ((name, mount), (other, push)) in expected.iter().zip(&actual) {
            anyhow::ensure!(name == other && mount == push, "pushed {name} differs from the mount");
            identical += mount.len();
        }
        eprintln!("V4.1 vision: {} operands, {} owners, {identical} B byte-identical; dense staging 0 B",
            records.operands.len(), expected.len());
        Ok(())
    }
}
