//! Family arithmetic only; the shared prefill executor owns unit/layer order.
use super::{attention, dense, weights::TargetWeights};
use crate::shared::memory::{device::{Allocation, Device, Stream}, receipts::Inventory};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, minimax_m3::{forward, producers}, programs::Programs};
use cuteafd_loader::families::minimax_m3::{MiniMaxM3Config,
    execution::{ForwardPrograms, ProducerPrograms, UnitLayout},
    weight_layout::{CoordinatorWeights, Mxfp8Projection}};
use std::{collections::BTreeMap, rc::Rc};

pub(crate) struct Kernels<'a> {
    device: Device<'a>,
    pub dense: dense::NativePrograms<'a>,
    pub attention: attention::Kernels<'a>,
    norm: forward::ForwardProducer<'a>,
    router: forward::ForwardProducer<'a>,
    oai_dense: producers::Producer<'a>,
    oai_shared: producers::Producer<'a>,
    add: cuteafd_ffi::Bf16Add<'a>,
}
impl<'a> Kernels<'a> {
    pub fn bind(device: Device<'a>, programs: &'a Programs<'a>, manifest: &serde_json::Value,
        cfg: &MiniMaxM3Config, weights: &CoordinatorWeights, rows: u32, context: u32) -> Result<Self> {
        UnitLayout::new(cfg, rows)?;
        let (_, forward) = ForwardPrograms::resolve(manifest, rows)?;
        let (_, producers) = ProducerPrograms::resolve(manifest, rows)?;
        for expected in forward.iter().chain(&producers) {
            let actual = programs.spec(&expected.name)?;
            ensure!(actual.pointers == expected.pointers && actual.scalar_kinds == expected.scalar_kinds
                && actual.scratch == expected.scratch && actual.capacity_rows == expected.capacity_rows
                && actual.geometry == expected.geometry,
                "MiniMax layer program {} differs from admission", expected.name);
        }
        let dense = dense::NativePrograms::bind(device, programs, weights, manifest, rows)?;
        let attention = attention::Kernels::bind(device, programs, manifest, cfg, rows, context)?;
        device.run(|| Ok(Self { device, dense, attention,
            norm: forward::ForwardProducer::bind(programs, forward::Kind::ResidualNorm)?,
            router: forward::ForwardProducer::bind(programs, forward::Kind::RouterScores)?,
            oai_dense: producers::Producer::bind(programs, producers::Kind::Oai { width: 12288 })?,
            oai_shared: producers::Producer::bind(programs, producers::Kind::Oai { width: 3072 })?,
            add: device.library.bf16_add()?,
        }))
    }
}

/// Independent residual/branches for one in-flight slot. The retained stream
/// drains before any storage retires, including a partially submitted chain.
pub(crate) struct UnitOwner<'a> {
    pub stream: Rc<Stream<'a>>,
    pub dense: dense::ExecutionOwner<'a>,
    attention: attention::Workspace<'a>,
    layout: UnitLayout,
    planes: BTreeMap<String, Allocation<'a>>,
}
impl Drop for UnitOwner<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("MiniMax unit arithmetic owners"); }
}
impl<'a> UnitOwner<'a> {
    pub fn new(kernels: &Kernels<'a>, stream: Rc<Stream<'a>>, cfg: &MiniMaxM3Config,
        weights: &CoordinatorWeights, rows: u32) -> Result<Self> {
        ensure!(stream.device.id == kernels.device.id
            && std::ptr::eq(stream.device.library, kernels.device.library), "MiniMax unit stream differs");
        let layout = UnitLayout::new(cfg, rows)?;
        let planes = layout.buffers().into_iter().map(|(name, bytes)| {
            Ok((name.into(), Allocation::new(kernels.device, usize::try_from(bytes)?,
                Tag::workspace("minimax/unit"))?))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self { dense: dense::ExecutionOwner::new(kernels.device, weights, rows)?,
            attention: attention::Workspace::new(&kernels.attention)?, stream, layout, planes })
    }
    fn owner(&self, name: &str) -> Result<&Allocation<'a>> {
        self.planes.get(name).with_context(|| format!("MiniMax unit lacks admitted {name}"))
    }
    pub fn plane(&self, name: &str) -> Result<Buffer> { Ok(self.owner(name)?.buffer) }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        self.dense.inventory(inventory, &format!("{prefix}.dense"))?;
        self.attention.inventory(inventory, &format!("{prefix}.attention"))?;
        for (name, owner) in &self.planes { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
    fn check(&self, kernels: &Kernels<'_>, rows: usize) -> Result<()> {
        ensure!(rows > 0 && rows <= self.layout.rows as usize && self.stream.device.id == kernels.device.id
            && std::ptr::eq(self.stream.device.library, kernels.device.library),
            "MiniMax live rows/device differ from unit owner");
        Ok(())
    }
    fn projection<'w>(weights: &'w CoordinatorWeights, layer: usize, suffix: &str) -> Result<&'w Mxfp8Projection> {
        let stem = format!("language_model.model.layers.{layer}.{suffix}");
        weights.projections.iter().find(|p| p.stem == stem)
            .with_context(|| format!("MiniMax unadmitted projection {stem}"))
    }
    fn weight(weights: &TargetWeights<'_>, layer: usize, suffix: &str) -> Result<Buffer> {
        Ok(weights.operand(&format!("language_model.model.layers.{layer}.{suffix}"))?.buffer)
    }
    fn normalize(&self, kernels: &Kernels<'_>, weights: &TargetWeights<'_>, layer: usize,
        suffix: &str, rows: usize) -> Result<()> {
        self.check(kernels, rows)?;
        let weight = Self::weight(weights, layer, suffix)?;
        self.stream.device.run(|| {
            // SAFETY: immutable arena view and unit-owned input/output survive
            // the retained stream; native binding checks all extents/aliases.
            unsafe { kernels.norm.norm(forward::NormBatch { rows, input: self.plane("residual")?,
                weight, output: self.dense.producer_destination() }, self.stream.raw)?; }
            Ok(())
        })
    }
    /// # Safety
    /// Request metadata/cache satisfy attention::Workspace::enqueue. Weights,
    /// kernels, cache and all request buffers survive the stream's drainage.
    pub unsafe fn attention(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>,
        admitted: &CoordinatorWeights, layer: usize, request: attention::Request) -> Result<()> {
        self.normalize(kernels, weights, layer, "input_layernorm.weight", request.rows)?;
        let names = [("self_attn.q_proj", "query"), ("self_attn.k_proj", "key"),
            ("self_attn.v_proj", "value"), ("self_attn.index_q_proj", "index_query"),
            ("self_attn.index_k_proj", "index_key")];
        let first = Self::projection(admitted, layer, names[0].0)?;
        let mut prepared = self.dense.prepare(&kernels.dense, first, request.rows as u32, self.stream.raw)?;
        for (suffix, output) in names {
            let destination = self.planes.get(output).context("MiniMax projection output missing")?;
            prepared.project(weights, Self::projection(admitted, layer, suffix)?)?.narrow(destination)?;
        }
        drop(prepared);
        let p = attention::Projections { query: self.plane("query")?, key: self.plane("key")?,
            value: self.plane("value")?, index_query: self.plane("index_query")?, index_key: self.plane("index_key")?,
            query_norm: Self::weight(weights, layer, "self_attn.q_norm.weight")?,
            key_norm: Self::weight(weights, layer, "self_attn.k_norm.weight")?,
            index_query_norm: Self::weight(weights, layer, "self_attn.index_q_norm.weight")?,
            index_key_norm: Self::weight(weights, layer, "self_attn.index_k_norm.weight")? };
        // SAFETY: the caller retains and validates request metadata/cache, and
        // all produced branch allocations are retained by this stream owner.
        let output = unsafe { self.attention.enqueue(&kernels.attention, p, request, &self.stream)? };
        let device = self.stream.device;
        device.run(|| {
            // SAFETY: copy is stream ordered after attention and before dense's
            // quantization; both owners remain live through drain.
            unsafe { device.library.copy_d2d_async(self.dense.producer_destination(), output,
                request.rows * 8192 * 2, self.stream.raw) }
        })?;
        let projection = Self::projection(admitted, layer, "self_attn.o_proj")?;
        self.dense.prepare(&kernels.dense, projection, request.rows as u32, self.stream.raw)?
            .project(weights, projection)?.narrow(self.planes.get("ffn").context("MiniMax ffn plane")?)?;
        self.accumulate(request.rows)
    }
    /// Router and FFN share the exact same normalized BF16 activation.
    pub fn router(&self, kernels: &Kernels<'_>, weights: &TargetWeights<'_>, layer: usize, rows: usize) -> Result<()> {
        self.normalize(kernels, weights, layer, "post_attention_layernorm.weight", rows)?;
        let device = self.stream.device;
        let weight = Self::weight(weights, layer, "block_sparse_moe.gate.weight")?;
        let bias = Self::weight(weights, layer, "block_sparse_moe.e_score_correction_bias")?;
        device.run(|| {
            // SAFETY: all disjoint input/logit/route planes and arena operands
            // remain live; selector receives E128/K4 and sigmoid/noaux_tc math.
            unsafe {
                kernels.router.router(forward::RouterBatch { rows, input: self.dense.producer_destination(),
                    weight, output: self.plane("logits")? }, self.stream.raw)?;
                device.library.router_select(self.plane("logits")?.ptr, bias.ptr, std::ptr::null(),
                    std::ptr::null(), self.plane("route_ids")?.ptr, self.plane("route_weights")?.ptr,
                    rows, 128, 4, 2.0, true, self.stream.raw)
            }
        })
    }
    /// Caller dispatches routed input before this replaces the normalized plane.
    /// Local dense layers normalize here; shared layers reuse router's norm.
    pub fn ffn(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>, admitted: &CoordinatorWeights,
        layer: usize, rows: usize, shared: bool) -> Result<()> {
        self.check(kernels, rows)?;
        if !shared { self.normalize(kernels, weights, layer, "post_attention_layernorm.weight", rows)?; }
        let stem = if shared { "block_sparse_moe.shared_experts" } else { "mlp" };
        let gate = Self::projection(admitted, layer, &format!("{stem}.gate_proj"))?;
        let up = Self::projection(admitted, layer, &format!("{stem}.up_proj"))?;
        let mut prepared = self.dense.prepare(&kernels.dense, gate, rows as u32, self.stream.raw)?;
        prepared.project(weights, gate)?.narrow(self.planes.get("gate").context("MiniMax gate plane")?)?;
        prepared.project(weights, up)?.narrow(self.planes.get("up").context("MiniMax up plane")?)?;
        drop(prepared);
        self.stream.device.run(|| {
            // SAFETY: gate/up have distinct retained BF16 destinations; the
            // former normalized activation may now be reused for OAI output.
            unsafe { (if shared { &kernels.oai_shared } else { &kernels.oai_dense }).oai(producers::OaiBatch {
                rows, gate: self.plane("gate")?, up: self.plane("up")?,
                output: self.dense.producer_destination() }, self.stream.raw)?; }
            Ok(())
        })?;
        let down = Self::projection(admitted, layer, &format!("{stem}.down_proj"))?;
        self.dense.prepare(&kernels.dense, down, rows as u32, self.stream.raw)?
            .project(weights, down)?.narrow(self.planes.get("ffn").context("MiniMax ffn plane")?)?;
        if !shared { self.accumulate(rows)?; }
        Ok(())
    }
    /// Official served-cache math rounds routed reduction to BF16 before adding
    /// the shared expert, then rounds that sum before the FP32 residual update.
    pub fn join_routed(&self, kernels: &Kernels<'_>, rows: usize) -> Result<()> {
        self.check(kernels, rows)?;
        self.stream.device.run(|| {
            // SAFETY: distinct retained planes, same stream as the landed wave.
            unsafe { kernels.add.launch(self.plane("routed")?, self.plane("ffn")?,
                self.plane("combined")?, rows * 6144, self.stream.raw) }
        })?;
        self.accumulate_from("combined", rows)
    }
    fn accumulate(&self, rows: usize) -> Result<()> { self.accumulate_from("ffn", rows) }
    fn accumulate_from(&self, source: &str, rows: usize) -> Result<()> {
        let device = self.stream.device;
        device.run(|| {
            // SAFETY: BF16 output and FP32 residual are separate retained owners;
            // every preceding producer and residual update uses this stream.
            unsafe { device.library.cuda_accumulate_bf16_to_f32_async(self.plane(source)?,
                self.plane("residual")?, rows * 6144, self.stream.raw) }
        })
    }
}
