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

fn segment_rows(segments: &[attention::Segment], capacity: usize) -> Result<usize> {
    packed_rows(segments.iter().map(|segment| (segment.first_row, segment.request.rows)), capacity)
}

fn packed_rows(spans: impl IntoIterator<Item = (usize, usize)>, capacity: usize) -> Result<usize> {
    let mut next = 0usize;
    for (first, rows) in spans {
        ensure!(first == next && rows > 0, "MiniMax attention segments must partition live rows");
        next = next.checked_add(rows).context("MiniMax packed row overflow")?;
        ensure!(next <= capacity, "MiniMax attention segments exceed the admitted owner");
    }
    ensure!(next > 0, "MiniMax attention segments are empty");
    Ok(next)
}

/// Independent residual/branches for one in-flight slot. The retained stream
/// drains before any storage retires, including a partially submitted chain.
pub(crate) struct UnitOwner<'a> {
    pub stream: Rc<Stream<'a>>,
    pub dense: dense::ExecutionOwner<'a>,
    attention: attention::Workspace<'a>,
    layout: UnitLayout,
    planes: BTreeMap<String, Rc<Allocation<'a>>>,
}
impl Drop for UnitOwner<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("MiniMax unit arithmetic owners"); }
}
impl<'a> UnitOwner<'a> {
    pub fn new(kernels: &Kernels<'a>, stream: Rc<Stream<'a>>, cfg: &MiniMaxM3Config,
        weights: &CoordinatorWeights, rows: u32) -> Result<Self> {
        Self::for_mode(kernels, stream, cfg, weights, rows, true)
    }
    pub fn new_decode(kernels: &Kernels<'a>, stream: Rc<Stream<'a>>, cfg: &MiniMaxM3Config,
        weights: &CoordinatorWeights, rows: u32) -> Result<Self> {
        ensure!(rows <= 64, "MiniMax decode owner exceeds native row capacity");
        Self::for_mode(kernels, stream, cfg, weights, rows, false)
    }
    fn for_mode(kernels: &Kernels<'a>, stream: Rc<Stream<'a>>, cfg: &MiniMaxM3Config,
        weights: &CoordinatorWeights, rows: u32, prefill: bool) -> Result<Self> {
        ensure!(stream.device.id == kernels.device.id
            && std::ptr::eq(stream.device.library, kernels.device.library), "MiniMax unit stream differs");
        let layout = UnitLayout::new(cfg, rows)?;
        let planes = layout.buffers().into_iter().map(|(name, bytes)| {
            Ok((name.into(), Rc::new(Allocation::new(kernels.device, usize::try_from(bytes)?,
                Tag::workspace("minimax/unit"))?)))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self { dense: dense::ExecutionOwner::new(kernels.device, weights, rows)?,
            attention: if prefill { attention::Workspace::new(&kernels.attention)? }
                else { attention::Workspace::new_decode(&kernels.attention)? }, stream, layout, planes })
    }
    fn owner(&self, name: &str) -> Result<&Allocation<'a>> {
        self.planes.get(name).map(Rc::as_ref).with_context(|| format!("MiniMax unit lacks admitted {name}"))
    }
    /// Prefix tails may use this scratch: begin_unit overwrites it before any model read.
    pub fn prefix_scratch(&self) -> Rc<Allocation<'a>> { Rc::clone(&self.planes["residual"]) }
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
        // SAFETY: the single segment forwards the caller's request contract.
        unsafe { self.attention_segments(kernels, weights, admitted, layer,
            &[attention::Segment { first_row: 0, request }]) }
    }
    /// # Safety
    /// Segments partition the live rows, and each request's local metadata and
    /// cache satisfy Workspace::enqueue. All owners survive stream drainage.
    pub unsafe fn attention_segments(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>,
        admitted: &CoordinatorWeights, layer: usize, segments: &[attention::Segment]) -> Result<()> {
        let rows = segment_rows(segments, self.layout.rows as usize)?;
        self.project_attention(kernels, weights, admitted, layer, rows)?;
        for segment in segments {
            let p = self.projections(weights, layer, segment.first_row, segment.request.rows)?;
            // SAFETY: each segment owns its metadata/cache, and the stream orders
            // scratch reuse after copying the preceding output into its row span.
            let output = unsafe { self.attention.enqueue(&kernels.attention, p, segment.request, &self.stream)? };
            self.copy_attention(output, segment.first_row, segment.request.rows)?;
        }
        self.finish_attention(kernels, weights, admitted, layer, rows)
    }
    /// # Safety
    /// Every decode row owns one request's tables and distinct append slot;
    /// request/cache operands remain valid until the retained stream drains.
    pub unsafe fn decode(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>,
        admitted: &CoordinatorWeights, layer: usize, request: attention::DecodeRequest) -> Result<()> {
        ensure!(request.batch == request.request.rows, "MiniMax decode rows must equal batch");
        let rows = request.request.rows;
        self.project_attention(kernels, weights, admitted, layer, rows)?;
        let p = self.projections(weights, layer, 0, rows)?;
        // SAFETY: caller supplies batched metadata; producer planes are retained.
        let output = unsafe { self.attention.decode(&kernels.attention, p, request, &self.stream)? };
        self.copy_attention(output, 0, rows)?;
        self.finish_attention(kernels, weights, admitted, layer, rows)
    }
    fn project_attention(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>,
        admitted: &CoordinatorWeights, layer: usize, rows: usize) -> Result<()> {
        self.normalize(kernels, weights, layer, "input_layernorm.weight", rows)?;
        let names = [("self_attn.q_proj", "query"), ("self_attn.k_proj", "key"),
            ("self_attn.v_proj", "value"), ("self_attn.index_q_proj", "index_query"),
            ("self_attn.index_k_proj", "index_key")];
        let first = Self::projection(admitted, layer, names[0].0)?;
        let mut prepared = self.dense.prepare(&kernels.dense, first, rows as u32, self.stream.raw)?;
        for (suffix, output) in names {
            let destination = self.planes.get(output).context("MiniMax projection output missing")?;
            prepared.project(weights, Self::projection(admitted, layer, suffix)?)?.narrow(destination)?;
        }
        Ok(())
    }
    fn projections(&self, weights: &TargetWeights<'_>, layer: usize, first: usize,
        rows: usize) -> Result<attention::Projections> {
        let view = |name, stride| attention::row_view(self.plane(name)?, first, rows, stride);
        Ok(attention::Projections { query: view("query", 8192 * 2)?, key: view("key", 512 * 2)?,
            value: view("value", 512 * 2)?, index_query: view("index_query", 512 * 2)?,
            index_key: view("index_key", 128 * 2)?,
            query_norm: Self::weight(weights, layer, "self_attn.q_norm.weight")?,
            key_norm: Self::weight(weights, layer, "self_attn.k_norm.weight")?,
            index_query_norm: Self::weight(weights, layer, "self_attn.index_q_norm.weight")?,
            index_key_norm: Self::weight(weights, layer, "self_attn.index_k_norm.weight")? })
    }
    fn copy_attention(&self, output: Buffer, first: usize, rows: usize) -> Result<()> {
        let destination = attention::row_view(self.dense.producer_destination(), first, rows, 8192 * 2)?;
        let device = self.stream.device;
        device.run(|| {
            // SAFETY: disjoint retained buffers, ordered before scratch reuse and
            // the batched O projection's activation quantization.
            unsafe { device.library.copy_d2d_async(destination, output, destination.bytes, self.stream.raw) }
        })
    }
    fn finish_attention(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>,
        admitted: &CoordinatorWeights, layer: usize, rows: usize) -> Result<()> {
        let projection = Self::projection(admitted, layer, "self_attn.o_proj")?;
        self.dense.prepare(&kernels.dense, projection, rows as u32, self.stream.raw)?
            .project(weights, projection)?.narrow(self.planes.get("ffn").context("MiniMax ffn plane")?)?;
        self.accumulate(rows)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_segments_partition_the_live_rows_without_gaps_or_overlap() {
        for spans in [vec![(0, 1)], vec![(0, 1), (1, 3), (4, 13)], vec![(0, 17)]] {
            let rows = spans.iter().map(|(_, rows)| rows).sum::<usize>();
            assert_eq!(packed_rows(spans, 17).unwrap(), rows);
        }
        for spans in [vec![], vec![(0, 0)], vec![(1, 3)], vec![(0, 2), (1, 1)],
            vec![(0, 2), (3, 1)], vec![(0, 18)]] {
            assert!(packed_rows(spans, 17).is_err());
        }
        assert!(packed_rows([(0, usize::MAX), (usize::MAX, 1)], usize::MAX).is_err());
    }
}
