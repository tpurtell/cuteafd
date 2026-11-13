//! Native EXL3 dense projections for Step-5 (K6 target core, K4 MTP): each
//! replaces one BF16 `(x, w, out)` projection launch with the shared trellis
//! input/core/output programs. Weights stay compressed; scratch is one admitted
//! serialized owner per device and lane.
use crate::shared::dense_trellis::{CoreWeightLayout, CoreWeights, Projection, ProjectionLayout, ProjectionScratch};
use crate::shared::memory::device::Device;
use crate::shared::memory::{receipts::Inventory, weight_arena::{DeviceWeightArena, weight_allocation}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{dense_trellis::TrellisDType, programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::formats::{dense_trellis::TrellisSide, trellis_core::ProjectionScratchLayout};
use cuteafd_loader::plan::{checkpoint::Checkpoint, families::step5::coordinator::RankWeightLayout};
use std::{collections::BTreeMap, ffi::c_void, rc::Rc};
use cuteafd_core::memory_layout::ArenaTag;

/// One resident native projection and its resolved programs.
pub(super) struct NativeLinear<'a> {
    weights: CoreWeights<'a>,
    projection: Projection<'a>,
}

impl<'a> NativeLinear<'a> {
    pub(super) fn shape(&self) -> [usize; 2] { self.weights.shape }
    pub(super) fn side(&self) -> TrellisSide { self.weights.side }

    /// The three retained trellis owners, named by their arena keys.
    pub(super) fn inventory(&self, inventory: &mut Inventory<'_>, name: &str) -> Result<()> {
        for (part, owner) in [("trellis", &self.weights.payload), ("suh", &self.weights.suh), ("svh", &self.weights.svh)] {
            owner.inventory(inventory, &format!("{name}.{part}"))?;
        }
        Ok(())
    }

    /// BF16 [rows, N] from BF16 [rows, K]: the exact BF16-program contract.
    pub(super) fn launch(&self, rows: usize, input: Buffer, output: Buffer, scratch: &ProjectionScratch<'_>,
        stream: *mut c_void) -> Result<()> {
        ensure!(self.weights.side == TrellisSide::Output, "K-split native projection needs a peer reduction");
        self.projection.local(rows, input, &self.weights, scratch, output, stream)
    }

    /// The K-split (head-split o_proj) F32 core partial, before svh.
    pub(super) fn partial(&self, rows: usize, input: Buffer, output: Buffer, scratch: &ProjectionScratch<'_>,
        stream: *mut c_void) -> Result<()> {
        ensure!(self.weights.side == TrellisSide::Input && output.bytes >= rows * self.weights.shape[0] * 4,
            "native F32 partial requires an input-split owner and F32 rows");
        self.projection.core_into(rows, input, &self.weights, scratch, output, stream)
    }

    /// One svh egress of the ordered F32 sum of both rank partials.
    pub(super) fn egress(&self, rows: usize, sum: Buffer, output: Buffer, stream: *mut c_void) -> Result<()> {
        self.projection.program.output(rows, sum, self.weights.svh.buffer, output, stream)?;
        Ok(())
    }
}

/// A layer's native operands by checkpoint `.weight` name.
#[derive(Default)]
pub(super) struct NativeLayer<'a> {
    pub(super) linears: BTreeMap<String, Rc<NativeLinear<'a>>>,
}

impl<'a> NativeLayer<'a> {
    pub(super) fn get(&self, name: &str) -> Option<&NativeLinear<'a>> {
        self.linears.get(name).map(Rc::as_ref)
    }
    pub(super) fn is_empty(&self) -> bool { self.linears.is_empty() }

    /// Loads exactly the layout's native owners whose names match `select`.
    pub(super) fn load(device: Device<'a>, programs: &'a Programs<'a>, checkpoint: &Checkpoint,
        layout: &RankWeightLayout, rows: usize, select: impl Fn(&str) -> bool, arena: Option<&DeviceWeightArena<'a>>) -> Result<Self> {
        let mut linears = BTreeMap::new();
        for (name, core) in layout.core.iter().filter(|(name, _)| select(name)) {
            let core = CoreWeightLayout::from_shared(core.clone());
            let weights = read_core(device, checkpoint, &core, arena)?;
            let side = if core.side == TrellisSide::Input { 2 } else { 1 };
            let output = if core.shape[0] == 128896 { TrellisDType::F32 } else { TrellisDType::Bf16 };
            let projection = device.run(|| Projection::load_geometry_bits(programs, core.shape[1], core.shape[0],
                side, core.bits, crate::shared::dense_trellis::codebook(&weights), rows, output))?;
            linears.insert(name.clone(), Rc::new(NativeLinear { weights, projection }));
        }
        Ok(Self { linears })
    }

    /// The serialized scratch these projections need at `rows` live rows.
    pub(super) fn scratch_layout(&self, rows: usize) -> Result<Option<ProjectionScratchLayout>> {
        let mut merged: Option<ProjectionScratchLayout> = None;
        for linear in self.linears.values() {
            let [n, k] = linear.weights.shape;
            let next = ProjectionScratchLayout::new(k, n, rows)?;
            match &mut merged { Some(layout) => layout.merge(&next)?, None => merged = Some(next) }
        }
        Ok(merged)
    }
}

fn read_core<'a>(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoreWeightLayout, arena: Option<&DeviceWeightArena<'a>>) -> Result<CoreWeights<'a>> {
    let dense = crate::shared::experts::rtx::dense::DenseLoad::mount(device.library);
    device.run(|| layout.load_recorded_with(device, checkpoint, &dense, std::ptr::null_mut(),
        |name, bytes| weight_allocation(device, arena, name, bytes, ArenaTag::weights("step5/target"))))
}

/// Per-device lane scratch shared by every native projection on that lane.
pub(super) struct NativeScratch<'a> {
    pub(super) scratch: ProjectionScratch<'a>,
    pub(super) layout: ProjectionScratchLayout,
    tag: crate::shared::memory::Tag,
}
impl<'a> NativeScratch<'a> {
    pub(super) fn allocate(device: Device<'a>, layout: &ProjectionScratchLayout, tag: crate::shared::memory::Tag) -> Result<Self> {
        let owned = ProjectionLayout { extents: layout.extents.clone(), rows: layout.rows };
        Ok(Self { scratch: owned.allocate_packed(device, tag)?, layout: layout.clone(), tag })
    }
    /// A private copy for another concurrently running lane.
    pub(super) fn sibling(&self, device: Device<'a>) -> Result<std::rc::Rc<Self>> {
        Ok(std::rc::Rc::new(Self::allocate(device, &self.layout, self.tag)?))
    }
}

/// Native q/k/v/o for one attention owner; `o` is input-split on head-split ranks.
pub(super) struct NativeAttention<'a> {
    pub(super) qkv: Option<[NativeLinear<'a>; 3]>,
    pub(super) o: Option<NativeLinear<'a>>,
}

/// Operand slots (input norm, q, k, v, q_norm, k_norm, g, o) the native core replaces.
pub(super) const ATTENTION_SLOTS: [usize; 4] = [1, 2, 3, 7];

/// Retained operand slots and native linears of one owner, in load order.
pub(super) fn inventory_owner<'w, 'a: 'w>(inventory: &mut Inventory<'_>, prefix: &str,
    weights: impl IntoIterator<Item = Option<&'w crate::shared::memory::device::Allocation<'a>>>,
    native: &[&NativeLinear<'_>]) -> Result<()> {
    for (slot, weight) in weights.into_iter().enumerate() {
        if let Some(weight) = weight { weight.inventory(inventory, &format!("{prefix}.operand{slot}"))?; }
    }
    for (index, linear) in native.iter().enumerate() {
        linear.inventory(inventory, &format!("{prefix}.native{index}"))?;
    }
    Ok(())
}

impl<'a> NativeAttention<'a> {
    pub(super) fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (linear, p) in self.qkv.iter().flatten().zip(["q", "k", "v"]) {
            linear.inventory(inventory, &format!("{prefix}.{p}_proj"))?;
        }
        if let Some(o) = &self.o { o.inventory(inventory, &format!("{prefix}.o_proj"))?; }
        Ok(())
    }

    /// Q/K/V share one route; the output projection may retain native BF16.
    pub(super) fn load(device: Device<'a>, programs: &'a Programs<'a>, checkpoint: &Checkpoint,
        layout: &RankWeightLayout, layer: usize, rows: usize, q: usize, kv: usize, hidden: usize, arena: Option<&DeviceWeightArena<'a>>)
        -> Result<Option<Self>> {
        let names = ["q", "k", "v", "o"].map(|p| format!("model.layers.{layer}.self_attn.{p}_proj.weight"));
        let present = names.iter().filter(|name| layout.core.contains_key(*name)).count();
        if present == 0 { return Ok(None); }
        let qkv = names[..3].iter().filter(|name| layout.core.contains_key(*name)).count();
        ensure!(matches!(qkv, 0 | 3), "Step-5 layer {layer} attention requires complete EXL3 q/k/v");
        let mut core = NativeLayer::load(device, programs, checkpoint, layout, rows, |name| names.iter().any(|n| n == name), arena)?;
        let mut take = |name: &str| -> Result<NativeLinear<'a>> {
            Rc::try_unwrap(core.linears.remove(name).context("native attention operand")?)
                .map_err(|_| anyhow::anyhow!("native attention operand aliased"))
        };
        let has_o = layout.core.contains_key(&names[3]);
        let attention = Self { qkv: if qkv == 3 { Some([take(&names[0])?, take(&names[1])?, take(&names[2])?]) } else { None },
            o: if has_o { Some(take(&names[3])?) } else { None } };
        let split = attention.o.as_ref().is_some_and(|o| o.side() == TrellisSide::Input);
        ensure!(attention.qkv.as_ref().is_none_or(|linears| linears.iter().zip([q, kv, kv])
                .all(|(linear, n)| linear.shape() == [n, hidden] && linear.side() == TrellisSide::Output))
            && attention.o.as_ref().is_none_or(|o| o.shape() == [hidden, q]),
            "Step-5 layer {layer} native attention geometry (split o={split})");
        Ok(Some(attention))
    }

    pub(super) fn scratch_layout(&self, rows: usize) -> Result<ProjectionScratchLayout> {
        let mut merged = None::<ProjectionScratchLayout>;
        for linear in self.qkv.iter().flatten().chain(self.o.iter()) {
            let [n, k] = linear.shape();
            let next = ProjectionScratchLayout::new(k, n, rows)?;
            match &mut merged { Some(layout) => layout.merge(&next)?, None => merged = Some(next) }
        }
        merged.context("native attention scratch")
    }
}

/// Mixed-attention O keeps the selected storage and typed output together.
pub(super) enum OutputProjection<'a> {
    Bf16(cuteafd_ffi::draft_projection::DraftProjections<'a>),
    Fp8 { program: cuteafd_ffi::native_attention::Fp8Projections<'a>,
        output: cuteafd_ffi::draft_projection::ProjectionOutput },
}
impl<'a> OutputProjection<'a> {
    pub(super) fn load(programs: &'a Programs<'a>, layout: &RankWeightLayout, layer: usize,
        rows: usize, k: usize, n: usize, output: cuteafd_ffi::draft_projection::ProjectionOutput,
        scratch_bytes: usize) -> Result<Self> {
        let name = format!("model.layers.{layer}.self_attn.o_proj.weight");
        let operand = layout.operands.iter().find(|w| w.name == name).context("selected native O")?;
        if operand.dtype != cuteafd_core::DType::F8E4M3 {
            return Ok(Self::Bf16(cuteafd_ffi::draft_projection::DraftProjections::load_with_output(
                programs, &[(k, n)], rows, output)?));
        }
        let program = cuteafd_ffi::native_attention::Fp8Projections::load(programs, &[(k,n)], rows, output)?;
        ensure!(program.scratch_bytes(k,n)? <= scratch_bytes, "FP8 O manifest scratch exceeds named output scratch plane");
        Ok(Self::Fp8 { program, output })
    }

    pub(super) fn output(&self) -> cuteafd_ffi::draft_projection::ProjectionOutput {
        match self { Self::Bf16(program) => program.output(), Self::Fp8 { output, .. } => *output }
    }

    pub(super) fn launch(&self, input: *const c_void, weight: Buffer, out: Buffer,
        scratch: Buffer, rows: usize, k: usize, n: usize, stream: *mut c_void) -> Result<()> {
        match self {
            Self::Bf16(program) => {
                // SAFETY: caller retains admitted BF16 input, weight and typed output until stream drain.
                unsafe { program.launch_checked(input, weight.ptr, out, rows, k, n, stream)?; }
            }
            Self::Fp8 { program, .. } => {
                let (values, scales) = super::weights::fp8_buffers(weight, &[n,k])?;
                // Dedicated lane-local scratch stays disjoint from every attention plane.
                // SAFETY: disjoint admitted input/output/final weights and lane-local scratch,
                // retained by the lane through stream drainage; no hot-path allocation.
                unsafe { program.launch(input, values.ptr, scales.ptr, out, scratch, rows, k, n, stream)?; }
            }
        }
        Ok(())
    }
}
