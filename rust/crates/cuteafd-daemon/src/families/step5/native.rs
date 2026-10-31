//! Native EXL3 dense projections for Step-5 (K6 target core, K4 MTP): each
//! replaces one BF16 `(x, w, out)` projection launch with the shared trellis
//! input/core/output programs. Weights stay compressed; scratch is one admitted
//! serialized owner per device and lane.
use crate::shared::dense_trellis::{CoreWeightLayout, CoreWeights, Projection, ProjectionLayout, ProjectionScratch};
use crate::shared::memory::device::Device;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{dense_trellis::TrellisDType, programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::formats::{dense_trellis::TrellisSide, trellis_core::ProjectionScratchLayout};
use cuteafd_loader::plan::{checkpoint::Checkpoint, families::step5::coordinator::RankWeightLayout};
use std::{collections::BTreeMap, ffi::c_void, rc::Rc};

/// One resident native projection and its resolved programs.
pub(super) struct NativeLinear<'a> {
    weights: CoreWeights<'a>,
    projection: Projection<'a>,
}

impl<'a> NativeLinear<'a> {
    pub(super) fn shape(&self) -> [usize; 2] { self.weights.shape }
    pub(super) fn side(&self) -> TrellisSide { self.weights.side }

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
        layout: &RankWeightLayout, rows: usize, select: impl Fn(&str) -> bool) -> Result<Self> {
        let mut linears = BTreeMap::new();
        for (name, core) in layout.core.iter().filter(|(name, _)| select(name)) {
            let core = CoreWeightLayout::from_shared(core.clone());
            let weights = read_core(device, checkpoint, &core)?;
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

fn read_core<'a>(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoreWeightLayout) -> Result<CoreWeights<'a>> {
    let loaded = layout.read(|name| {
        let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("Step-5 native core header {name} not loaded"))?;
        let tensor = &checkpoint.tensors[at];
        let path = checkpoint.snapshot.join(&tensor.shard);
        let mut bytes = vec![0u8; usize::try_from(tensor.meta.byte_length)?];
        crate::shared::memory::load_timing::read(bytes.len(), || {
            use std::os::unix::fs::FileExt;
            std::fs::File::open(&path)?.read_exact_at(&mut bytes, tensor.meta.byte_offset)
        }).with_context(|| format!("reading {name}"))?;
        Ok(cuteafd_loader::LoadedTensor {
            info: cuteafd_core::TensorInfo { name: name.into(), file: tensor.shard.clone(), dtype: tensor.meta.dtype.clone(),
                shape: tensor.meta.shape.clone(), byte_offset: tensor.meta.byte_offset,
                byte_length: tensor.meta.byte_length, role: cuteafd_core::TensorRole::Other, layer_id: None,
                expert_id: None, is_quantization_metadata: false },
            source_path: path, bytes, elapsed_micros: 0, sha256: String::new(),
        })
    })?;
    device.run(|| layout.upload(device, loaded))
}

/// Per-device lane scratch shared by every native projection on that lane.
pub(super) struct NativeScratch<'a> {
    pub(super) scratch: ProjectionScratch<'a>,
    pub(super) layout: ProjectionScratchLayout,
}
impl<'a> NativeScratch<'a> {
    pub(super) fn allocate(device: Device<'a>, layout: &ProjectionScratchLayout) -> Result<Self> {
        let owned = ProjectionLayout { extents: layout.extents.clone(), rows: layout.rows };
        Ok(Self { scratch: owned.allocate(device)?, layout: layout.clone() })
    }
    /// A private copy for another concurrently running lane.
    pub(super) fn sibling(&self, device: Device<'a>) -> Result<std::rc::Rc<Self>> {
        Ok(std::rc::Rc::new(Self::allocate(device, &self.layout)?))
    }
}

/// Native q/k/v/o for one attention owner; `o` is input-split on head-split ranks.
pub(super) struct NativeAttention<'a> {
    pub(super) q: NativeLinear<'a>,
    pub(super) k: NativeLinear<'a>,
    pub(super) v: NativeLinear<'a>,
    pub(super) o: NativeLinear<'a>,
}

/// Operand slots (input norm, q, k, v, q_norm, k_norm, g, o) the native core replaces.
pub(super) const ATTENTION_SLOTS: [usize; 4] = [1, 2, 3, 7];

impl<'a> NativeAttention<'a> {
    /// `None` for a BF16 layer; mixed native/BF16 attention is refused.
    pub(super) fn load(device: Device<'a>, programs: &'a Programs<'a>, checkpoint: &Checkpoint,
        layout: &RankWeightLayout, layer: usize, rows: usize, q: usize, kv: usize, hidden: usize)
        -> Result<Option<Self>> {
        let names = ["q", "k", "v", "o"].map(|p| format!("model.layers.{layer}.self_attn.{p}_proj.weight"));
        let present = names.iter().filter(|name| layout.core.contains_key(*name)).count();
        if present == 0 { return Ok(None); }
        ensure!(present == 4, "Step-5 layer {layer} attention mixes native and BF16 operands");
        let mut core = NativeLayer::load(device, programs, checkpoint, layout, rows, |name| names.iter().any(|n| n == name))?;
        let mut take = |name: &str| -> Result<NativeLinear<'a>> {
            Rc::try_unwrap(core.linears.remove(name).context("native attention operand")?)
                .map_err(|_| anyhow::anyhow!("native attention operand aliased"))
        };
        let attention = Self { q: take(&names[0])?, k: take(&names[1])?, v: take(&names[2])?, o: take(&names[3])? };
        let split = attention.o.side() == TrellisSide::Input;
        ensure!(attention.q.shape() == [q, hidden] && attention.k.shape() == [kv, hidden]
            && attention.v.shape() == [kv, hidden] && attention.o.shape() == [hidden, q]
            && [&attention.q, &attention.k, &attention.v].iter().all(|l| l.side() == TrellisSide::Output),
            "Step-5 layer {layer} native attention geometry (split o={split})");
        Ok(Some(attention))
    }

    pub(super) fn scratch_layout(&self, rows: usize) -> Result<ProjectionScratchLayout> {
        let mut merged = None::<ProjectionScratchLayout>;
        for linear in [&self.q, &self.k, &self.v, &self.o] {
            let [n, k] = linear.shape();
            let next = ProjectionScratchLayout::new(k, n, rows)?;
            match &mut merged { Some(layout) => layout.merge(&next)?, None => merged = Some(next) }
        }
        merged.context("native attention scratch")
    }
}
