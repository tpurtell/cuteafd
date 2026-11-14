//! Family arithmetic only; the shared prefill executor owns unit/layer order.
use super::{attention, dense, weights::TargetWeights};
use crate::shared::memory::{device::{Allocation, Device, Stream}, receipts::Inventory};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, minimax_m3::{forward, producers}, programs::Programs};
use cuteafd_loader::families::minimax_m3::{MiniMaxM3Config,
    execution::{AttentionLayout, ForwardPrograms, HeadsLayout, PartialFfnLayout, ProducerPrograms, RouterLayout, RouteReceiveLayout, SplitLayout, UnitLayout, WirePrecision},
    weight_layout::{CoordinatorWeights, Mxfp8Projection}};
use std::{collections::BTreeMap, rc::Rc};

pub(crate) struct Kernels<'a> {
    device: Device<'a>,
    pub dense: dense::NativePrograms<'a>,
    pub attention: Option<attention::Kernels<'a>>,
    norm: forward::ForwardProducer<'a>,
    router: Option<forward::ForwardProducer<'a>>,
    oai_dense: producers::Producer<'a>,
    oai_shared: producers::Producer<'a>,
    add: cuteafd_ffi::Bf16Add<'a>,
    split_add: Option<cuteafd_ffi::Bf16F32Add<'a>>,
    narrow: Option<cuteafd_ffi::F32Bf16Narrow<'a>>,
    heads: bool,
    f32_add: Option<cuteafd_ffi::F32Add<'a>>,
}
impl<'a> Kernels<'a> {
    pub fn bind(device: Device<'a>, programs: &'a Programs<'a>, manifest: &serde_json::Value,
        cfg: &MiniMaxM3Config, weights: &CoordinatorWeights, rows: u32, context: u32) -> Result<Self> {
        Self::bind_role(device, programs, manifest, cfg, weights, rows, context, false, false)
    }
    pub fn bind_split(device: Device<'a>, programs: &'a Programs<'a>, manifest: &serde_json::Value,
        cfg: &MiniMaxM3Config, weights: &CoordinatorWeights, rows: u32, context: u32) -> Result<Self> {
        Self::bind_role(device, programs, manifest, cfg, weights, rows, context, true, false)
    }
    pub fn bind_heads(device: Device<'a>, programs: &'a Programs<'a>, manifest: &serde_json::Value,
        cfg: &MiniMaxM3Config, weights: &CoordinatorWeights, rows: u32, context: u32) -> Result<Self> {
        Self::bind_role(device, programs, manifest, cfg, weights, rows, context, true, true)
    }
    pub fn has_router(&self) -> bool { self.router.is_some() }
    fn bind_role(device: Device<'a>, programs: &'a Programs<'a>, manifest: &serde_json::Value,
        cfg: &MiniMaxM3Config, weights: &CoordinatorWeights, rows: u32, context: u32, split: bool, heads: bool) -> Result<Self> {
        UnitLayout::new(cfg, rows)?;
        let (_, forward) = if heads { ForwardPrograms::resolve_heads(manifest, rows)? }
            else { ForwardPrograms::resolve(manifest, rows)? };
        let (_, producers) = if heads { ProducerPrograms::resolve_heads(manifest, rows)? }
            else if split { ProducerPrograms::resolve_split(manifest, rows)? }
            else { ProducerPrograms::resolve(manifest, rows)? };
        for expected in forward.iter().chain(&producers) {
            let actual = programs.spec(&expected.name)?;
            ensure!(actual.pointers == expected.pointers && actual.scalar_kinds == expected.scalar_kinds
                && actual.scratch == expected.scratch && actual.capacity_rows == expected.capacity_rows
                && actual.geometry == expected.geometry,
                "MiniMax layer program {} differs from admission", expected.name);
        }
        let dense = dense::NativePrograms::bind(device, programs, weights, manifest, rows)?;
        let attention = (!weights.layers.is_empty()).then(||
            if heads { attention::Kernels::bind_heads(device, programs, manifest, cfg, rows, context, attention::HeadLayout::Narrow) }
            else { attention::Kernels::bind(device, programs, manifest, cfg, rows, context) }).transpose()?;
        device.run(|| Ok(Self { device, dense, attention,
            norm: forward::ForwardProducer::bind(programs, forward::Kind::ResidualNorm)?,
            router: (!split || weights.rank == 0).then(||
                forward::ForwardProducer::bind(programs, forward::Kind::RouterScores)).transpose()?,
            oai_dense: producers::Producer::bind(programs, producers::Kind::Oai { width: if split { 6144 } else { 12288 } })?,
            oai_shared: producers::Producer::bind(programs, producers::Kind::Oai { width: if split { 1536 } else { 3072 } })?,
            add: device.library.bf16_add()?,
            split_add: split.then(|| device.library.bf16_f32_add()).transpose()?,
            narrow: split.then(|| device.library.f32_bf16_narrow()).transpose()?,
            heads, f32_add: heads.then(|| device.library.f32_add()).transpose()?,
        }))
    }
}

/// The topology chooses transport and fold ownership; both topologies use the
/// same rank-ordered arithmetic and the same whole-down BF16 boundary.
pub(crate) struct PartialFold<'a> {
    stream: Rc<Stream<'a>>,
    rank0: Rc<Allocation<'a>>,
    rank1: Rc<Allocation<'a>>,
    meeting: Rc<Allocation<'a>>,
    precision: dense::PartialPrecision,
    capacity: usize,
    f32_add: cuteafd_ffi::F32Add<'a>,
    bf16_add: cuteafd_ffi::Bf16F32Add<'a>,
    narrow: cuteafd_ffi::F32Bf16Narrow<'a>,
}
impl<'a> PartialFold<'a> {
    /// All three backings are caller-admitted. Rank order is physical rank order,
    /// regardless of which GPU owns this layer's residual.
    pub fn new(stream: Rc<Stream<'a>>, rank0: Rc<Allocation<'a>>, rank1: Rc<Allocation<'a>>,
        meeting: Rc<Allocation<'a>>, precision: dense::PartialPrecision, capacity: usize) -> Result<Self> {
        let device = stream.device;
        ensure!(capacity > 0 && capacity <= 4096, "MiniMax partial fold capacity differs");
        let count = capacity.checked_mul(6144).context("MiniMax partial fold overflow")?;
        for owner in [&rank0, &rank1, &meeting] {
            ensure!(owner.device.id == device.id && std::ptr::eq(owner.device.library, device.library),
                "MiniMax partial fold owner device/library differs");
        }
        validate_fold([rank0.buffer, rank1.buffer, meeting.buffer], None, count, precision)?;
        let (f32_add, bf16_add, narrow) = device.run(|| Ok((device.library.f32_add()?,
            device.library.bf16_f32_add()?, device.library.f32_bf16_narrow()?)))?;
        Ok(Self { stream, rank0, rank1, meeting, precision, capacity, f32_add, bf16_add, narrow })
    }
}
impl Drop for PartialFold<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("MiniMax partial fold owners"); }
}

fn validate_fold(buffers: [Buffer; 3], whole: Option<Buffer>, count: usize,
    precision: dense::PartialPrecision) -> Result<()> {
    let partial_bytes = count.checked_mul(precision.bytes()).filter(|&bytes| bytes > 0)
        .context("MiniMax fold partial overflow")?;
    let meeting_bytes = count.checked_mul(4).context("MiniMax fold meeting overflow")?;
    let mut ranges = [(0usize, 0usize); 4];
    for (index, (buffer, bytes, alignment)) in [(buffers[0], partial_bytes, precision.bytes()),
        (buffers[1], partial_bytes, precision.bytes()), (buffers[2], meeting_bytes, 4)]
        .into_iter().chain(whole.map(|buffer| (buffer, count * 2, 2))).enumerate() {
        ensure!(buffer.device_id >= 0 && buffer.device_id == buffers[2].device_id && buffer.flags == 0
            && !buffer.ptr.is_null() && buffer.ptr as usize % alignment == 0 && buffer.bytes >= bytes,
            "MiniMax fold extent/alignment/device differs");
        let start = buffer.ptr as usize;
        let end = start.checked_add(bytes).context("MiniMax fold pointer overflow")?;
        ensure!(ranges[..index].iter().all(|&(a, b)| end <= a || b <= start), "MiniMax fold owners overlap");
        ranges[index] = (start, end);
    }
    Ok(())
}

fn validate_fold_exclusions(buffers: [Buffer; 3], protected: impl IntoIterator<Item = Buffer>) -> Result<()> {
    let mut ranges = [(0usize, 0usize); 3];
    for (range, buffer) in ranges.iter_mut().zip(buffers) {
        let start = buffer.ptr as usize;
        *range = (start, start.checked_add(buffer.bytes).context("MiniMax fold backing overflow")?);
    }
    for buffer in protected {
        let start = buffer.ptr as usize;
        let end = start.checked_add(buffer.bytes).context("MiniMax protected backing overflow")?;
        ensure!(ranges.iter().all(|&(a, b)| end <= a || b <= start),
            "MiniMax fold owner overlaps unit storage");
    }
    Ok(())
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

pub(crate) struct SplitOwner<'a> {
    pub layout: SplitLayout,
    planes: BTreeMap<String, Rc<Allocation<'a>>>,
    pub partial: dense::PartialDestination<'a>,
    fold: PartialFold<'a>,
}
impl<'a> SplitOwner<'a> {
    fn new(stream: Rc<Stream<'a>>, layout: SplitLayout, rank: usize) -> Result<Self> {
        let planes = layout.buffers().into_iter().map(|(name, bytes)| Ok((name.into(),
            Rc::new(Allocation::new(stream.device, usize::try_from(bytes)?, Tag::workspace("minimax/split"))?))))
            .collect::<Result<BTreeMap<String, Rc<Allocation<'a>>>>>()?;
        let partial = dense::PartialDestination::new(Rc::clone(&planes["local"]),
            Rc::clone(&stream), Rc::clone(&stream), layout.wire)?;
        let (rank0, rank1) = if rank == 0 { ("local", "receive") } else { ("receive", "local") };
        let fold = PartialFold::new(stream, Rc::clone(&planes[rank0]), Rc::clone(&planes[rank1]),
            Rc::clone(&planes["meeting"]), layout.wire, layout.rows as usize)?;
        Ok(Self { layout, planes, partial, fold })
    }
    fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.planes { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}

/// Independent residual/branches for one in-flight slot. The retained stream
/// drains before any storage retires, including a partially submitted chain.
pub(crate) struct UnitOwner<'a> {
    pub stream: Rc<Stream<'a>>,
    pub dense: dense::ExecutionOwner<'a>,
    attention: Option<attention::Workspace<'a>>,
    split: Option<SplitOwner<'a>>,
    rows: u32,
    planes: BTreeMap<String, Rc<Allocation<'a>>>,
    routed_boundary: Rc<super::experts::Boundary<'a>>,
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
            routed_boundary: Rc::new(super::experts::Boundary::new(kernels.device)?),
            attention: kernels.attention.as_ref().map(|attention| if prefill { attention::Workspace::new(attention) }
                else { attention::Workspace::new_decode(attention) }).transpose()?,
            split: None, stream, rows: layout.rows, planes })
    }
    pub fn new_split(kernels: &Kernels<'a>, stream: Rc<Stream<'a>>, cfg: &MiniMaxM3Config,
        weights: &CoordinatorWeights, rows: u32, wire: WirePrecision, prefill: bool,
        receive_routes: bool) -> Result<Self> {
        ensure!(stream.device.id == kernels.device.id
            && std::ptr::eq(stream.device.library, kernels.device.library), "MiniMax split stream differs");
        ensure!(prefill || rows <= 64, "MiniMax split decode rows exceed native capacity");
        let partial = PartialFfnLayout::new(cfg, weights.rank as u8, rows)?;
        let mut buffers = partial.buffers();
        if !weights.layers.is_empty() { buffers.extend(AttentionLayout::new(cfg, rows)?.buffers()); }
        if weights.rank == 0 { buffers.extend(RouterLayout::new(rows)?.buffers()); }
        else if receive_routes { buffers.extend(RouteReceiveLayout::new(rows)?.buffers()); }
        let mut planes = BTreeMap::new();
        for (name, bytes) in buffers {
            ensure!(!planes.contains_key(name), "duplicate MiniMax split owner {name}");
            planes.insert(name.into(), Rc::new(Allocation::new(kernels.device, usize::try_from(bytes)?,
                Tag::workspace("minimax/unit"))?));
        }
        let split = SplitOwner::new(Rc::clone(&stream), SplitLayout::new(rows, wire)?, weights.rank)?;
        stream.device.run(|| {
            // SAFETY: fresh admitted flags are initialized on the retained unit
            // stream before any device-side control consumer can observe them.
            unsafe {
                stream.device.library.cuda_zero_bytes_async(split.planes["control"].buffer, 256, stream.raw)?;
                stream.device.library.cuda_zero_bytes_async(split.planes["abort"].buffer, 256, stream.raw)
            }
        })?;
        Ok(Self { dense: dense::ExecutionOwner::new(kernels.device, weights, rows)?,
            routed_boundary: Rc::new(super::experts::Boundary::new(kernels.device)?),
            attention: kernels.attention.as_ref().map(|attention| if prefill { attention::Workspace::new(attention) }
                else { attention::Workspace::new_decode(attention) }).transpose()?,
            split: Some(split), stream, rows, planes })
    }
    pub fn new_heads(kernels: &Kernels<'a>, stream: Rc<Stream<'a>>, cfg: &MiniMaxM3Config,
        weights: &CoordinatorWeights, rows: u32, prefill: bool) -> Result<Self> {
        ensure!(kernels.heads && stream.device.id == kernels.device.id
            && std::ptr::eq(stream.device.library, kernels.device.library), "MiniMax Heads stream/role differs");
        ensure!(prefill || rows <= 64, "MiniMax Heads decode exceeds capacity");
        let layout = HeadsLayout::new(cfg, weights.rank as u8, rows)?;
        let planes = layout.buffers().into_iter().map(|(name, bytes)| Ok((name.into(),
            Rc::new(Allocation::new(kernels.device, usize::try_from(bytes)?, Tag::workspace("minimax/unit"))?))))
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self { dense: dense::ExecutionOwner::new(kernels.device, weights, rows)?,
            routed_boundary: Rc::new(super::experts::Boundary::new(kernels.device)?),
            attention: kernels.attention.as_ref().map(|attention| if prefill { attention::Workspace::new(attention) }
                else { attention::Workspace::new_decode(attention) }).transpose()?,
            split: None, stream, rows, planes })
    }
    fn heads(&self) -> bool { self.planes.contains_key("reduction") }
    pub fn is_split(&self) -> bool { self.split.is_some() }
    fn owner(&self, name: &str) -> Result<&Allocation<'a>> {
        self.planes.get(name).map(Rc::as_ref).with_context(|| format!("MiniMax unit lacks admitted {name}"))
    }
    /// Prefix tails may use this scratch: begin_unit overwrites it before any model read.
    pub fn prefix_scratch(&self) -> Rc<Allocation<'a>> { Rc::clone(&self.planes["residual"]) }
    pub fn plane(&self, name: &str) -> Result<Buffer> { Ok(self.owner(name)?.buffer) }
    pub(crate) fn routed_rows(&self, rows: usize) -> Result<super::experts::RoutedRows<'a>> {
        ensure!(rows <= self.rows as usize, "MiniMax routed rows exceed unit");
        let retained = |name| self.planes.get(name).cloned().with_context(|| format!("MiniMax rank has no {name} role"));
        super::experts::RoutedRows::new(self.dense.activation.clone(), retained("route_ids")?,
            retained("route_weights")?, retained("routed")?, self.stream.clone(),
            self.routed_boundary.clone(), rows)
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        self.dense.inventory(inventory, &format!("{prefix}.dense"))?;
        if let Some(attention) = &self.attention { attention.inventory(inventory, &format!("{prefix}.attention"))?; }
        if let Some(split) = &self.split { split.inventory(inventory, &format!("{prefix}.split"))?; }
        for (name, owner) in &self.planes {
            owner.inventory(inventory, &format!("{prefix}.{name}"))?;
            if self.split.is_some() && !self.planes.contains_key("logits")
                && matches!(name.as_str(), "route_ids" | "route_weights") {
                inventory.receipts.last_mut().context("MiniMax route receipt")?.allocation.category =
                    cuteafd_core::memory_layout::Category::Transport;
            }
        }
        Ok(())
    }
    pub fn rows(&self) -> u32 { self.rows }
    /// CUDA-only target-residual tap seam. The optional drafter owns and admits
    /// its retained writer; ordinary serving allocates no tap buffers here.
    pub(crate) fn post_fold_tap_enqueue(&self, _layer: usize, _rows: usize) -> Result<()> { Ok(()) }
    #[cfg(test)]
    pub fn validate_split(&self, inputs: &cuteafd_loader::placement::families::minimax_m3::MiniMaxInputs,
        gpu: usize) -> Result<()> {
        let split = self.split.as_ref().context("MiniMax split owner missing")?;
        ensure!(split.layout.wire == inputs.split_wire.context("MiniMax selected wire")?
            && split.layout.events() == 2, "MiniMax split descriptor differs");
        for (name, bytes) in split.layout.buffers() {
            ensure!(split.planes[name].buffer.bytes == bytes as usize, "MiniMax split plane {name} differs");
        }
        for layer in 0..inputs.cfg.layers {
            let roles = inputs.roles(layer, gpu, self.rows)?;
            ensure!(roles.iter().any(|r| r.role == cuteafd_loader::families::minimax_m3::execution::ExecutionRole::WholeAttention)
                == (inputs.owners[layer] as usize == gpu), "MiniMax attention role differs");
        }
        Ok(())
    }
    fn check(&self, kernels: &Kernels<'_>, rows: usize) -> Result<()> {
        ensure!(rows > 0 && rows <= self.rows as usize && self.stream.device.id == kernels.device.id
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
        let rows = segment_rows(segments, self.rows as usize)?;
        self.project_attention(kernels, weights, admitted, layer, rows)?;
        for segment in segments {
            let p = self.projections(weights, layer, segment.first_row, segment.request.rows)?;
            // SAFETY: each segment owns its metadata/cache, and the stream orders
            // scratch reuse after copying the preceding output into its row span.
            let output = unsafe { self.attention.as_mut().context("MiniMax rank has no attention role")?.enqueue(
                kernels.attention.as_ref().context("MiniMax missing attention kernels")?, p, segment.request, &self.stream)? };
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
        let output = unsafe { self.attention.as_mut().context("MiniMax rank has no attention role")?.decode(
            kernels.attention.as_ref().context("MiniMax missing attention kernels")?, p, request, &self.stream)? };
        self.copy_attention(output, 0, rows)?;
        self.finish_attention(kernels, weights, admitted, layer, rows)
    }
    fn project_attention(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>,
        admitted: &CoordinatorWeights, layer: usize, rows: usize) -> Result<()> {
        self.routed_boundary.idle()?;
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
        let divisor = if self.heads() { 2 } else { 1 };
        Ok(attention::Projections { query: view("query", 8192 * 2 / divisor)?, key: view("key", 512 * 2 / divisor)?,
            value: view("value", 512 * 2 / divisor)?, index_query: view("index_query", 512 * 2 / divisor)?,
            index_key: view("index_key", 128 * 2)?,
            query_norm: Self::weight(weights, layer, "self_attn.q_norm.weight")?,
            key_norm: Self::weight(weights, layer, "self_attn.k_norm.weight")?,
            index_query_norm: Self::weight(weights, layer, "self_attn.index_q_norm.weight")?,
            index_key_norm: Self::weight(weights, layer, "self_attn.index_k_norm.weight")? })
    }
    fn copy_attention(&self, output: Buffer, first: usize, rows: usize) -> Result<()> {
        let destination = attention::row_view(self.dense.producer_destination(), first, rows,
            if self.heads() { 4096 * 2 } else { 8192 * 2 })?;
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
        let destination = self.planes.get(if self.heads() { "reduction" } else { "ffn" })
            .context("MiniMax attention projection destination")?;
        let mut prepared = self.dense.prepare(&kernels.dense, projection, rows as u32, self.stream.raw)?;
        let projected = prepared.project(weights, projection)?;
        if kernels.heads { projected.copy_partial(destination)?; } else { projected.narrow(destination)?; }
        drop(prepared);
        if kernels.heads { Ok(()) } else { self.accumulate(rows) }
    }
    /// Router and FFN share the exact same normalized BF16 activation.
    pub fn router(&self, kernels: &Kernels<'_>, weights: &TargetWeights<'_>, layer: usize, rows: usize) -> Result<()> {
        self.router_boundary()?;
        self.router_enqueue(kernels, weights, layer, rows)
    }
    pub fn router_boundary(&self) -> Result<()> { self.routed_boundary.idle() }
    /// CUDA-only body; graph callers perform router_boundary on every replay.
    pub fn router_enqueue(&self, kernels: &Kernels<'_>, weights: &TargetWeights<'_>, layer: usize, rows: usize) -> Result<()> {
        ensure!(kernels.router.is_some(), "MiniMax rank has no router role");
        self.normalize(kernels, weights, layer, "post_attention_layernorm.weight", rows)?;
        let device = self.stream.device;
        let weight = Self::weight(weights, layer, "block_sparse_moe.gate.weight")?;
        let bias = Self::weight(weights, layer, "block_sparse_moe.e_score_correction_bias")?;
        device.run(|| {
            // SAFETY: all disjoint input/logit/route planes and arena operands
            // remain live; selector receives E128/K4 and sigmoid/noaux_tc math.
            unsafe {
                kernels.router.as_ref().context("MiniMax rank has no router role")?.router(forward::RouterBatch { rows, input: self.dense.producer_destination(),
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
        ensure!(self.split.is_none(), "MiniMax split FFN must stage a partial before meeting");
        self.ffn_projection(kernels, weights, admitted, layer, rows, shared)
    }
    /// Stages a native half-FFN result; the caller must exchange and fold it
    /// before touching .ffn or advancing either rank's FP32 residual.
    pub fn partial_ffn(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>,
        admitted: &CoordinatorWeights, layer: usize, rows: usize, shared: bool) -> Result<()> {
        ensure!(self.split.is_some() || self.heads(), "MiniMax partial FFN requires split owners");
        self.ffn_projection(kernels, weights, admitted, layer, rows, shared)
    }
    fn ffn_projection(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>,
        admitted: &CoordinatorWeights, layer: usize, rows: usize, shared: bool) -> Result<()> {
        self.check(kernels, rows)?;
        self.ffn_boundary(shared)?;
        self.ffn_enqueue(kernels, weights, admitted, layer, rows, shared)
    }
    pub fn ffn_boundary(&self, shared: bool) -> Result<()> {
        if shared { self.routed_boundary.before_ffn(self.stream.raw) }
        else { self.routed_boundary.idle() }
    }
    /// CUDA-only body shared by eager and captured whole/partial FFN paths.
    /// The caller executes ffn_boundary before every invocation or replay.
    pub fn ffn_enqueue(&mut self, kernels: &Kernels<'a>, weights: &TargetWeights<'_>,
        admitted: &CoordinatorWeights, layer: usize, rows: usize, shared: bool) -> Result<()> {
        self.check(kernels, rows)?;
        if !shared || ((self.split.is_some() || self.heads()) && admitted.rank != 0) {
            self.normalize(kernels, weights, layer, "post_attention_layernorm.weight", rows)?;
        }
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
        let mut prepared = self.dense.prepare(&kernels.dense, down, rows as u32, self.stream.raw)?;
        let output = prepared.project(weights, down)?;
        if let Some(split) = &mut self.split {
            output.stage_partial(&mut split.partial)?;
        } else if kernels.heads {
            output.copy_partial(self.planes.get("reduction").context("MiniMax Heads partial plane")?)?;
        } else {
            output.narrow(self.planes.get("ffn").context("MiniMax ffn plane")?)?;
        }
        drop(prepared);
        if !shared && self.split.is_none() && !kernels.heads { self.accumulate(rows)?; }
        Ok(())
    }
    /// # Safety
    /// The peer partial has landed in this unit's receive plane, and both
    /// slots remain leased until the ordered fold and residual update finish.
    pub unsafe fn meet_ffn(&self, kernels: &Kernels<'_>, rows: usize, shared: bool) -> Result<()> {
        let split = self.split.as_ref().context("MiniMax meet requires split owners")?;
        // SAFETY: the caller supplies the same ordering/lease contract.
        unsafe { self.fold_partials(kernels, &split.fold, rows, shared) }
    }
    pub fn split_plane(&self, name: &str) -> Result<Buffer> {
        self.split.as_ref().and_then(|split| split.planes.get(name)).map(|owner| owner.buffer)
            .with_context(|| format!("MiniMax split plane missing: {name}"))
    }
    /// # Safety
    /// Both rank slots have landed and their producers precede this unit stream.
    /// No transport or producer may replace either slot until the fold completes.
    pub unsafe fn fold_partials(&self, kernels: &Kernels<'_>, fold: &PartialFold<'_>,
        rows: usize, shared: bool) -> Result<()> {
        self.fold_boundary(shared)?;
        // SAFETY: the caller retains both landed partials through this stream.
        unsafe { self.fold_partials_enqueue(kernels, fold, rows, shared)?; }
        if shared { self.joined(); }
        Ok(())
    }
    pub(crate) fn fold_boundary(&self, shared: bool) -> Result<()> {
        if shared { self.join_boundary() } else { self.routed_boundary.idle() }
    }
    /// CUDA-only fold, narrowing and routed/residual addition. Host ticket
    /// transitions run before/after this body on every captured replay.
    /// # Safety
    /// Both retained partials are landed and stay leased through the stream.
    unsafe fn fold_partials_enqueue(&self, kernels: &Kernels<'_>, fold: &PartialFold<'_>,
        rows: usize, shared: bool) -> Result<()> {
        self.check(kernels, rows)?;
        ensure!(rows <= fold.capacity && std::ptr::eq(self.stream.as_ref(), fold.stream.as_ref()),
            "MiniMax partial fold stream/capacity differs from unit");
        let whole = self.plane("ffn")?;
        let count = rows.checked_mul(6144).context("MiniMax partial fold count overflow")?;
        let buffers = [fold.rank0.buffer, fold.rank1.buffer, fold.meeting.buffer];
        validate_fold(buffers, Some(whole), count, fold.precision)?;
        validate_fold_exclusions(buffers, self.planes.values().map(|owner| owner.buffer).chain([
            self.dense.activation.buffer, self.dense.workspace.inputs.buffer,
            self.dense.workspace.records.buffer, self.dense.workspace.output.buffer]))?;
        self.stream.device.run(|| {
            // SAFETY: caller orders landed rank slots; checked typed disjoint
            // retained planes survive this stream and every captured replay.
            unsafe {
                match fold.precision {
                    dense::PartialPrecision::F32 => fold.f32_add.launch(fold.rank0.buffer,
                        fold.rank1.buffer, fold.meeting.buffer, count, self.stream.raw)?,
                    dense::PartialPrecision::Bf16 => fold.bf16_add.launch(fold.rank0.buffer,
                        fold.rank1.buffer, fold.meeting.buffer, count, self.stream.raw)?,
                }
                fold.narrow.launch(fold.meeting.buffer.ptr, whole.ptr, count, self.stream.raw)?;
            }
            Ok(())
        })?;
        // Preserve the whole-down narrowing before routed addition. The existing
        // routed boundary owns exactly-once combination, not this transport fold.
        if shared { self.join_routed_enqueue(kernels, rows) } else { self.accumulate(rows) }
    }
    /// # Safety
    /// Both retained peer partial planes are landed and remain leased.
    pub(crate) unsafe fn meet_ffn_enqueue(&self, kernels: &Kernels<'_>, rows: usize, shared: bool) -> Result<()> {
        let split = self.split.as_ref().context("MiniMax meet requires split owners")?;
        // SAFETY: forward the caller's landed partial and lease contract.
        unsafe { self.fold_partials_enqueue(kernels, &split.fold, rows, shared) }
    }
    /// Official served-cache math rounds routed reduction to BF16 before adding
    /// the shared expert, then rounds that sum before the FP32 residual update.
    pub fn join_routed(&self, kernels: &Kernels<'_>, rows: usize) -> Result<()> {
        self.check(kernels, rows)?;
        self.join_boundary()?;
        self.join_routed_enqueue(kernels, rows)?;
        self.joined();
        Ok(())
    }
    pub fn join_boundary(&self) -> Result<()> { self.routed_boundary.before_join(self.stream.raw) }
    pub fn joined(&self) { self.routed_boundary.joined(); }
    /// CUDA-only routed addition; the host boundary remains outside captures.
    pub fn join_routed_enqueue(&self, kernels: &Kernels<'_>, rows: usize) -> Result<()> {
        self.check(kernels, rows)?;
        self.stream.device.run(|| {
            // SAFETY: all planes are distinct retained owners on this stream.
            // The split meeting is free after fold narrowing; narrowing the
            // exactly representable BF16 sum preserves the whole join boundary.
            unsafe {
                if let Some(split) = &self.split {
                    let meeting = split.planes["meeting"].buffer;
                    kernels.split_add.as_ref().context("MiniMax split join kernel missing")?.launch(self.plane("routed")?, self.plane("ffn")?,
                        meeting, rows * 6144, self.stream.raw)?;
                    kernels.narrow.as_ref().context("MiniMax split narrow kernel missing")?.launch(
                        meeting.ptr, self.plane("ffn")?.ptr, rows * 6144, self.stream.raw)
                } else if kernels.heads {
                    let meeting = self.plane("reduction")?;
                    kernels.split_add.as_ref().context("MiniMax Heads join kernel missing")?.launch(
                        self.plane("routed")?, self.plane("ffn")?, meeting, rows * 6144, self.stream.raw)?;
                    kernels.narrow.as_ref().context("MiniMax Heads narrow kernel missing")?.launch(
                        meeting.ptr, self.plane("ffn")?.ptr, rows * 6144, self.stream.raw)
                } else { kernels.add.launch(self.plane("routed")?, self.plane("ffn")?,
                    self.plane("combined")?, rows * 6144, self.stream.raw) }
            }
        })?;
        self.accumulate_from(if self.split.is_some() || kernels.heads { "ffn" } else { "combined" }, rows)
    }
    /// # Safety
    /// Both ordered FP32 partials are local peer-exchange views, completed before
    /// this stream consumes them; their lease survives narrowing/residual update.
    pub(crate) unsafe fn merge_heads(&self, kernels: &Kernels<'_>, values: [*const std::ffi::c_void; 2],
        rows: usize, attention: bool, shared: bool) -> Result<()> {
        self.check(kernels, rows)?;
        ensure!(kernels.heads && self.heads(), "MiniMax head merge requires narrow owners");
        let count = rows.checked_mul(6144).context("MiniMax head merge extent overflow")?;
        let buffer = |ptr: *const std::ffi::c_void| Buffer { ptr: ptr.cast_mut(), bytes: count * 4,
            device_id: self.stream.device.id, flags: 0 };
        let meeting = self.dense.workspace.output.buffer;
        self.stream.device.run(|| {
            // SAFETY: ordered live partial views and disjoint serial temporary;
            // no subsequent projection reuses it before this narrowing completes.
            unsafe {
                kernels.f32_add.as_ref().context("MiniMax Heads add missing")?.launch(
                    buffer(values[0]), buffer(values[1]), meeting, count, self.stream.raw)?;
                kernels.narrow.as_ref().context("MiniMax Heads narrow missing")?.launch(
                    meeting.ptr, self.plane("ffn")?.ptr, count, self.stream.raw)
            }
        })?;
        if attention || !shared { self.accumulate(rows) } else { self.join_routed_enqueue(kernels, rows) }
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
    fn partial_join_rounds_shared_before_routed_and_preserves_fp32_residual_bits() {
        fn narrow(value: f32) -> f32 {
            let bits = value.to_bits();
            f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
        }
        // Independent nearest-even conversion by choosing adjacent BF16 values.
        fn reference(value: f32) -> f32 {
            let bits = value.to_bits();
            let lower = bits & 0xffff0000;
            let tail = bits & 0xffff;
            let rounded = lower + if tail > 0x8000 || (tail == 0x8000 && lower & 0x10000 != 0) {
                0x10000
            } else { 0 };
            f32::from_bits(rounded)
        }
        let cases = [
            (1.0, 0.00390625, -1.0, 0.25),
            (1.0, 0.01171875, -1.0, -0.25),
            (65536.0, 0.5, -65536.0, 0.125),
            (1.0e20, 1.0e-20, -1.0e20, -1.0e-10),
            (-65536.0, -0.5, 65536.0, -0.125),
            (0.0, -0.0, -0.0, 0.0),
        ];
        for wire in [dense::PartialPrecision::Bf16, dense::PartialPrecision::F32] {
            for (p0, p1, routed, residual) in cases {
                let stage = |value| match wire { dense::PartialPrecision::Bf16 => narrow(value),
                    dense::PartialPrecision::F32 => value };
                let stage_ref = |value| match wire { dense::PartialPrecision::Bf16 => reference(value),
                    dense::PartialPrecision::F32 => value };
                let shared = narrow(stage(p0) + stage(p1));
                let meeting = narrow(routed) + shared;
                let ffn = narrow(meeting);
                let expected_shared = reference(stage_ref(p0) + stage_ref(p1));
                let expected_ffn = reference(reference(routed) + expected_shared);
                assert_eq!(ffn.to_bits(), expected_ffn.to_bits(), "{wire:?}");
                assert_eq!((residual + ffn).to_bits(), (residual + expected_ffn).to_bits(), "{wire:?}");
            }
        }
        // Moving the shared narrowing after the routed add changes this tie.
        assert_ne!(narrow(narrow(1.0 + 0.00390625) - 1.0).to_bits(),
            narrow(1.0 + 0.00390625 - 1.0).to_bits());
    }

    #[test]
    fn both_wire_precisions_require_disjoint_rank_meeting_and_whole_planes() -> Result<()> {
        let buffer = |pointer, bytes| Buffer { ptr: pointer as *mut std::ffi::c_void,
            bytes, device_id: 0, flags: 0 };
        for precision in [dense::PartialPrecision::Bf16, dense::PartialPrecision::F32] {
            let count = 6144;
            let rank0 = buffer(0x10000usize, count * precision.bytes());
            let rank1 = buffer(0x20000usize, count * precision.bytes());
            let meeting = buffer(0x30000usize, count * 4);
            let whole = buffer(0x40000usize, count * 2);
            validate_fold([rank0, rank1, meeting], Some(whole), count, precision)?;
            for index in 0..3 {
                let mut invalid = [rank0, rank1, meeting];
                invalid[index].bytes -= 1;
                assert!(validate_fold(invalid, Some(whole), count, precision).is_err());
                invalid = [rank0, rank1, meeting];
                invalid[index].device_id = 1;
                assert!(validate_fold(invalid, Some(whole), count, precision).is_err());
            }
            for invalid in [rank0, rank1, meeting, Buffer { bytes: whole.bytes - 1, ..whole },
                Buffer { flags: 1, ..whole }, Buffer { ptr: std::ptr::null_mut(), ..whole }] {
                assert!(validate_fold([rank0, rank1, meeting], Some(invalid), count, precision).is_err());
            }
            assert!(validate_fold([rank0, rank0, meeting], Some(whole), count, precision).is_err());
            assert!(validate_fold([rank0, rank1, meeting], Some(whole), 0, precision).is_err());
            assert!(validate_fold([rank0, rank1, meeting], Some(whole), usize::MAX, precision).is_err());
        }
        Ok(())
    }

    #[test]
    fn partial_fold_cannot_alias_any_unit_plane_or_its_unused_tail() -> Result<()> {
        let buffer = |pointer, bytes| Buffer { ptr: pointer as *mut std::ffi::c_void,
            bytes, device_id: 0, flags: 0 };
        let buffers = [buffer(0x10000usize, 0x1000), buffer(0x20000, 0x1000), buffer(0x30000, 0x2000)];
        validate_fold_exclusions(buffers, [buffer(0x40000, 0x1000)])?;
        validate_fold_exclusions(buffers, [buffer(0x11000, 0x1000)])?;
        for slot in buffers {
            for protected in [slot, buffer(slot.ptr as usize + slot.bytes - 1, 4),
                buffer(slot.ptr as usize - 1, 2)] {
                assert!(validate_fold_exclusions(buffers, [protected]).is_err());
            }
        }
        assert!(validate_fold_exclusions(buffers, [buffer(usize::MAX, 2)]).is_err());
        assert!(validate_fold_exclusions([buffer(usize::MAX, 2), buffers[1], buffers[2]], []).is_err());
        Ok(())
    }

    fn residual_bytes(sequence: usize, positions: std::ops::Range<usize>) -> Vec<u8> {
        positions.flat_map(|position| (0..6144).flat_map(move |column| {
            let value = ((sequence * 101 + position * 37 + column * 17) % 257) as f32 / 128.0 - 1.0;
            value.to_le_bytes()
        })).collect()
    }

    fn read_residual(owner: &UnitOwner<'_>, rows: usize) -> Result<Vec<f32>> {
        owner.stream.drain()?;
        let mut bytes = vec![0; rows * 6144 * 4];
        owner.stream.device.run(|| owner.stream.device.library.copy_d2h(&mut bytes, owner.plane("residual")?))?;
        Ok(bytes.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().unwrap())).collect())
    }

    fn cache_bytes(library: &cuteafd_ffi::NativeLibrary, request: attention::Request) -> Result<Vec<Vec<u8>>> {
        [request.key_cache, request.value_cache, request.index_cache].into_iter().map(|buffer| {
            let mut bytes = vec![0; buffer.bytes];
            library.copy_d2h(&mut bytes, buffer)?;
            Ok(bytes)
        }).collect()
    }

    fn read_buffer(owner: &UnitOwner<'_>, buffer: Buffer) -> Result<Vec<u8>> {
        owner.stream.drain()?;
        let mut bytes = vec![0; buffer.bytes];
        owner.stream.device.run(|| owner.stream.device.library.copy_d2h(&mut bytes, buffer))?;
        Ok(bytes)
    }

    fn bf16_values(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(2).map(|v| f32::from_bits(u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16)).collect()
    }

    fn e4m3(value: u8) -> f32 {
        let exponent = (value >> 3) & 15;
        let fraction = f32::from(value & 7);
        let magnitude = if exponent == 0 { fraction * 2.0f32.powi(-9) }
            else { (1.0 + fraction / 8.0) * 2.0f32.powi(i32::from(exponent) - 7) };
        if value & 128 == 0 { magnitude } else { -magnitude }
    }

    // CPU replica of TrellisMxfp8Quant: K32 amax/448 rounded up to UE8M0,
    // then saturating E4M3 round-to-nearest-even payload conversion.
    fn mxfp8_qdq(input: &[f32]) -> (Vec<f32>, Vec<u8>, Vec<u8>) {
        assert_eq!(input.len() % 32, 0);
        assert!(input.iter().all(|value| value.is_finite()));
        let mut output = Vec::with_capacity(input.len());
        let mut payload = Vec::with_capacity(input.len());
        let mut scales = Vec::with_capacity(input.len() / 32);
        for block in input.chunks_exact(32) {
            let amax = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            let mut bits = (amax * (1.0f32 / 448.0)).to_bits();
            if bits & 0x007f_ffff != 0 { bits = (bits + 0x0080_0000) & 0x7f80_0000; }
            let code = ((bits >> 23) & 255) as u8;
            let scale = 2.0f32.powi(i32::from(code) - 127);
            scales.push(code);
            for value in block {
                let scaled = value.abs() / scale;
                let mut nearest = 0u8;
                let mut error = f32::INFINITY;
                for candidate in 0..=126u8 {
                    let candidate_error = (e4m3(candidate) - scaled).abs();
                    if candidate_error < error || (candidate_error == error && candidate & 1 == 0) {
                        nearest = candidate;
                        error = candidate_error;
                    }
                }
                let byte = nearest | if value.is_sign_negative() { 128 } else { 0 };
                payload.push(byte);
                output.push(e4m3(byte) * scale);
            }
        }
        (output, payload, scales)
    }

    #[test]
    fn mxfp8_diagnostic_matches_k32_scale_and_payload_rounding() {
        let mut input = [0.0f32; 32];
        input[..5].copy_from_slice(&[448.0, 1.0625, 1.1875, -1.0625, -0.0]);
        let (output, payload, scales) = mxfp8_qdq(&input);
        assert_eq!(scales, [127]);
        assert_eq!(&output[..4], &[448.0, 1.0, 1.25, -1.0]);
        assert_eq!(&payload[..5], &[0x7e, 0x38, 0x3a, 0xb8, 0x80]);
        input[0] = 449.0;
        assert_eq!(mxfp8_qdq(&input).2, [128]);
    }

    struct AttentionDiagnostic {
        query: Vec<f32>,
        output: Vec<f32>,
        selections: Vec<Vec<i32>>,
    }

    fn attention_diagnostic(owner: &UnitOwner<'_>) -> Result<AttentionDiagnostic> {
        let plane = |name| owner.attention.as_ref().context("MiniMax test attention role")?.diagnostic_plane(name);
        let query = bf16_values(&read_buffer(owner, attention::row_view(plane("attention.query")?, 0, 1, 16384)?)?);
        let output = bf16_values(&read_buffer(owner, attention::row_view(plane("attention.output")?, 0, 1, 16384)?)?);
        let bytes = read_buffer(owner, plane("selections")?)?;
        let stride = bytes.len() / 4;
        let selections = (0..4).map(|head| bytes[head * stride..head * stride + 64].chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap())).collect()).collect();
        Ok(AttentionDiagnostic { query, output, selections })
    }

    fn distance(actual: &[f32], reference: &[f32]) -> (f64, f32) {
        let squared = actual.iter().zip(reference).map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum::<f64>();
        let norm = reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
        let peak = reference.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max = actual.iter().zip(reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        ((squared / norm).sqrt(), max / peak)
    }

    fn diagnose_attention(owner: &UnitOwner<'_>, request: attention::Request,
        extend: &AttentionDiagnostic, decode: &AttentionDiagnostic) -> Result<()> {
        let integers = |bytes: Vec<u8>| bytes.chunks_exact(4).map(|v| i32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
        let floats = |bytes: Vec<u8>| bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
        let pages = integers(read_buffer(owner, request.pages)?);
        let kd = floats(read_buffer(owner, request.key_descale)?);
        let vd = floats(read_buffer(owner, request.value_descale)?);
        let key = read_buffer(owner, request.key_cache)?;
        let value = read_buffer(owner, request.value_cache)?;
        let mut outputs = Vec::new();
        for (route, diagnostic) in [("extend", extend), ("decode", decode)] {
            let mut reference = vec![0.0f32; 8192];
            for head in 0..64 {
                let kv_head = head / 16;
                let mut scores = vec![f32::NEG_INFINITY; request.context];
                let mut maximum = f32::NEG_INFINITY;
                for (token, score) in scores.iter_mut().enumerate() {
                    if diagnostic.selections[kv_head].contains(&((token / 128) as i32)) {
                        let base = (usize::try_from(pages[token / 64])? * 64 + token % 64) * 512 + kv_head * 128;
                        let mut dot = 0.0f32;
                        for column in 0..128 {
                            dot += diagnostic.query[head * 128 + column] * (e4m3(key[base + column]) * kd[kv_head]);
                        }
                        *score = dot * 128.0f32.sqrt().recip();
                        maximum = maximum.max(*score);
                    }
                }
                ensure!(maximum.is_finite(), "MiniMax diagnostic has no visible keys");
                let mut denominator = 0.0f32;
                for score in &mut scores { *score = (*score - maximum).exp(); denominator += *score; }
                for (token, score) in scores.into_iter().enumerate() {
                    let probability = score / denominator;
                    let base = (usize::try_from(pages[token / 64])? * 64 + token % 64) * 512 + kv_head * 128;
                    for column in 0..128 {
                        reference[head * 128 + column] += probability * (e4m3(value[base + column]) * vd[kv_head]);
                    }
                }
            }
            let (relative, max) = distance(&diagnostic.output, &reference);
            eprintln!("MiniMax context={} route={route} attention_vs_fp32 rel_l2={relative} max_abs_over_peak={max}", request.context);
            ensure!(relative <= 0.002, "MiniMax context={} route={route}: attention differs from FP32 cached-operand oracle: rel_l2={relative}", request.context);
            let heads = (0..64).map(|head| {
                let span = head * 128..(head + 1) * 128;
                let (relative, max) = distance(&diagnostic.output[span.clone()], &reference[span.clone()]);
                eprintln!("MiniMax context={} route={route} head={head} attention_vs_fp32 rel_l2={relative} max_abs_over_peak={max}", request.context);
                serde_json::json!({ "head": head, "relative_l2": relative, "max_abs_over_peak": max,
                    "actual": diagnostic.output[span.clone()], "reference": reference[span] })
            }).collect::<Vec<_>>();
            outputs.push(serde_json::json!({ "route": route, "relative_l2": relative,
                "max_abs_over_peak": max, "query": diagnostic.query,
                "selections": diagnostic.selections, "heads": heads }));
        }
        let (extend_qdq, extend_payload, extend_scales) = mxfp8_qdq(&extend.output);
        let (decode_qdq, decode_payload, decode_scales) = mxfp8_qdq(&decode.output);
        let payload_flips = extend_payload.iter().zip(&decode_payload).filter(|(a, b)| a != b).count();
        let scale_flips = extend_scales.iter().zip(&decode_scales).filter(|(a, b)| a != b).count();
        let (raw_relative, _) = distance(&decode.output, &extend.output);
        let (qdq_relative, _) = distance(&decode_qdq, &extend_qdq);
        eprintln!("MiniMax context={} attention_path rel_l2={raw_relative} first_activation_qdq_rel_l2={qdq_relative} payload_flips={payload_flips}/{} scale_flips={scale_flips}/{}",
            request.context, extend_payload.len(), extend_scales.len());
        if let Ok(directory) = std::env::var("CUTEAFD_MINIMAX_DIAGNOSTICS") {
            std::fs::create_dir_all(&directory)?;
            std::fs::write(std::path::Path::new(&directory).join(format!("attention-{}.json", request.context)),
                serde_json::to_vec(&serde_json::json!({ "context": request.context, "routes": outputs,
                    "first_activation_qdq": { "raw_relative_l2": raw_relative, "relative_l2": qdq_relative,
                        "payload_flips": payload_flips, "scale_flips": scale_flips } }))?)?;
        }
        ensure!(extend.query == decode.query, "MiniMax decode query differs from extend");
        ensure!(extend.selections == decode.selections, "MiniMax decode selection differs from extend");
        if request.context == 64 {
            let expected = std::iter::once(0).chain(std::iter::repeat(-1).take(15)).collect::<Vec<_>>();
            ensure!(extend.selections.iter().all(|head| *head == expected), "MiniMax context64 must select only forced block0");
        }
        Ok(())
    }

    #[test]
    fn context64_reference_selection_has_only_forced_local_block() {
        let visible = (0..64).map(|token| token / 128).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(visible, [0].into_iter().collect());
        assert_eq!((64 - 1) / 128, 0);
        assert_eq!([e4m3(0), e4m3(0x38), e4m3(0x7e), e4m3(0xfe)], [0.0, 1.0, 448.0, -448.0]);
    }

    fn compare_residual(actual: &[f32], reference: &[f32], label: &str, limits: (f64, f32)) -> Result<()> {
        ensure!(!actual.is_empty() && actual.len() == reference.len(), "{label}: residual extent");
        ensure!(actual.iter().chain(reference).all(|value| value.is_finite()), "{label}: nonfinite residual");
        let squared = actual.iter().zip(reference).map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum::<f64>();
        let norm = reference.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>();
        let peak = reference.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let max = actual.iter().zip(reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        ensure!(norm > 0.0 && peak > 0.0, "{label}: empty residual signal");
        let relative = (squared / norm).sqrt();
        eprintln!("MiniMax {label}: rows={} rel_l2={relative} max_abs_over_peak={}", actual.len() / 6144, max / peak);
        ensure!(relative <= limits.0 && max / peak <= limits.1, "{label}: residual exceeds regression tripwire");
        Ok(())
    }

    #[test]
    #[ignore = "requires MiniMax checkpoint and two native SM120 devices through golden-run --native-test"]
    fn native_split_ffn_bucket_padding_preserves_real_rows() -> Result<()> {
        use cuteafd_loader::{families::minimax_m3::{execution::WirePrecision,
            weight_layout::SplitFfnWeights}, plan::Checkpoint, serving_capacity::WeightArenaLayout};
        use std::path::Path;
        let snapshot = std::env::var("CUTEAFD_MINIMAX_SNAPSHOT")?;
        let checkpoint = Checkpoint::coordinator(Path::new(&snapshot), false, false)?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        // SAFETY: runner supplies the native library; all owners retire first.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        ensure!(crate::shared::peer_split::probed_device(&library, 0, Some(1))? == Some(1),
            "MiniMax padding oracle requires a CUDA peer pair");
        let manifest_path = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let programs = library.programs()?.with_manifest(Path::new(&manifest_path))?;
        let mut layouts = SplitFfnWeights::plan(&checkpoint, &cfg, &vec![0; cfg.layers],
            cuteafd_loader::placement::planner::EmbeddingHome::Device(0))?;
        for layout in &mut layouts {
            let selected = |name: &str| name.starts_with("language_model.model.layers.0.")
                || name.starts_with("language_model.model.layers.3.");
            layout.target.arena = WeightArenaLayout::new(format!("minimax/padding/{}", layout.target.rank),
                layout.target.arena.views().iter().filter(|(name, _)| selected(name))
                    .map(|(name, view)| (name.clone(), view.bytes())))?;
            layout.target.projections.retain(|projection| selected(&projection.stem));
            layout.target.layers.clear();
            layout.slices.retain(|name, _| selected(name));
        }
        let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
        let weights = layouts.iter().zip(devices).map(|(layout, device)|
            TargetWeights::load_split_ffn(device, &checkpoint, layout)).collect::<Result<Vec<_>>>()?;
        let kernels = layouts.iter().zip(devices).map(|(layout, device)|
            Kernels::bind_split(device, &programs, &manifest, &cfg, &layout.target, 64, 128))
            .collect::<Result<Vec<_>>>()?;
        let streams = devices.map(|device| Stream::new(device).map(Rc::new));
        let [left, right] = streams;
        let streams = [left?, right?];
        for wire in [WirePrecision::Bf16, WirePrecision::F32] {
            let mut units = (0..2).map(|gpu| UnitOwner::new_split(&kernels[gpu], Rc::clone(&streams[gpu]),
                &cfg, &layouts[gpu].target, 64, wire, false, false)).collect::<Result<Vec<_>>>()?;
            for (layer, shared) in [(0, false), (3, true)] {
                for live in [1, 2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33, 63, 64] {
                    let bucket = crate::shared::decode_graph::row_bucket(live);
                    let mut reference = None;
                    for rows in [live, bucket] {
                        for gpu in 0..2 {
                            let unit = &units[gpu];
                            let residual = residual_bytes(7, 0..64);
                            devices[gpu].run(|| {
                                library.copy_h2d(unit.plane("residual")?, &residual)?;
                                library.cuda_zero_bytes(unit.plane("routed")?, 64 * 6144 * 2)
                            })?;
                        }
                        if shared { units[0].router(&kernels[0], &weights[0], layer, rows)?; }
                        for gpu in 0..2 {
                            units[gpu].partial_ffn(&kernels[gpu], &weights[gpu], &layouts[gpu].target,
                                layer, rows, shared)?;
                        }
                        for stream in &streams { stream.drain()?; }
                        for gpu in 0..2 {
                            let source = units[1 - gpu].split_plane("local")?;
                            let receive = units[gpu].split_plane("receive")?;
                            let bytes = rows * 6144 * wire.bytes();
                            devices[1 - gpu].run(|| {
                                // SAFETY: both producers drained; retained P2P
                                // planes stay live through each fold below.
                                unsafe { library.copy_d2d_async(receive, source, bytes, streams[1 - gpu].raw) }
                            })?;
                        }
                        for stream in &streams { stream.drain()?; }
                        for gpu in 0..2 {
                            // SAFETY: both peer partial copies have drained.
                            unsafe { units[gpu].meet_ffn(&kernels[gpu], rows, shared)?; }
                        }
                        let actual = units.iter().map(|unit| read_residual(unit, live)
                            .map(|values| values.into_iter().map(f32::to_bits).collect::<Vec<_>>()))
                            .collect::<Result<Vec<_>>>()?;
                        ensure!(actual[0] == actual[1], "MiniMax rank fold differs");
                        if let Some(expected) = &reference {
                            ensure!(&actual == expected, "MiniMax live={live} bucket={bucket} layer={layer} wire={wire:?}: padding changed real rows");
                        } else { reference = Some(actual); }
                    }
                }
            }
        }
        eprintln!("MiniMax split padding oracle PASS: dense/shared, BF16/FP32 wire, live1..64 boundaries");
        Ok(())
    }

    #[test]
    #[ignore = "requires MiniMax checkpoint and native SM120 exports through golden-run --native-test"]
    fn native_layer_packed_segments_and_decode_match_independent_extend() -> Result<()> {
        use super::super::cache::Cache;
        use cuteafd_loader::{plan::Checkpoint, serving_capacity::WeightArenaLayout};
        use std::path::Path;
        let snapshot = std::env::var("CUTEAFD_MINIMAX_SNAPSHOT").context("set CUTEAFD_MINIMAX_SNAPSHOT")?;
        let checkpoint = Checkpoint::coordinator(Path::new(&snapshot), false, false)?;
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let mut weights = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &vec![0; cfg.layers])?
            .into_iter().next().context("MiniMax native layer layout")?;
        // Keep the real layer's native operands, without loading the other 59 layers or vocabulary head.
        let prefix = "language_model.model.layers.0.";
        weights.arena = WeightArenaLayout::new("minimax/native-layer-oracle", weights.arena.views().iter()
            .filter(|(name, _)| name.starts_with(prefix)).map(|(name, view)| (name.clone(), view.bytes())))?;
        weights.layers = vec![0];
        weights.projections.retain(|projection| projection.stem.starts_with(prefix));
        // SAFETY: the golden runner supplies the sealed native library; all owners retire before it.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = Device { library: &library, id: std::env::var("CUTEAFD_CACHE_TEST_DEVICE")
            .unwrap_or_else(|_| "0".into()).parse()? };
        let path = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let programs = library.programs()?.with_manifest(Path::new(&path))?;
        let target = TargetWeights::mount(device, &checkpoint, &weights)?;
        let kernels = Kernels::bind(device, &programs, &manifest, &cfg, &weights, 257, 270)?;
        let stream = Rc::new(Stream::new(device)?);
        let mut single = UnitOwner::new(&kernels, Rc::clone(&stream), &cfg, &weights, 257)?;
        let mut packed = UnitOwner::new(&kernels, Rc::clone(&stream), &cfg, &weights, 17)?;
        let mut decode = UnitOwner::new_decode(&kernels, Rc::clone(&stream), &cfg, &weights, 1)?;
        let mut cache_cfg = cfg.clone();
        cache_cfg.layers = 1;
        let mut reference = Vec::new();
        let mut segments = Vec::new();
        let mut inputs = Vec::new();
        let mut caches = Vec::new();
        let mut metadata = Vec::new();
        for (sequence, (begin, rows)) in [(63, 1), (127, 3), (257, 13)].into_iter().enumerate() {
            for _ in 0..2 {
                let cache = Cache::new(Rc::clone(&stream), &cache_cfg, begin + rows)?;
                let mut meta = cache.metadata(257)?;
                cache.prepare(&mut meta, 0..begin)?;
                let request = cache.request(&meta, 0)?;
                device.run(|| {
                    for buffer in [request.key_cache, request.value_cache, request.index_cache] {
                        library.cuda_zero_bytes(buffer, buffer.bytes)?;
                    }
                    Ok(())
                })?;
                device.run(|| library.copy_h2d(single.plane("residual")?, &residual_bytes(sequence, 0..begin)))?;
                // SAFETY: cold Cache owns unique pages; metadata and every arithmetic owner retain the stream.
                unsafe { single.attention(&kernels, &target, &weights, 0, cache.request(&meta, 0)?)?; }
                stream.drain()?;
                caches.push(cache);
                metadata.push(meta);
            }
            let at = sequence * 2;
            device.run(|| {
                assert_eq!(cache_bytes(&library, caches[at].request(&metadata[at], 0)?)?,
                    cache_bytes(&library, caches[at + 1].request(&metadata[at + 1], 0)?)?,
                    "identical cold prefixes must write byte-identical cache state");
                Ok(())
            })?;
            caches[at].prepare(&mut metadata[at], begin..begin + rows)?;
            let input = residual_bytes(sequence, begin..begin + rows);
            device.run(|| library.copy_h2d(single.plane("residual")?, &input))?;
            // SAFETY: this sequence's prefix has completed, and tail metadata references only its initialized pages.
            unsafe { single.attention(&kernels, &target, &weights, 0, caches[at].request(&metadata[at], 0)?)?; }
            single.ffn(&kernels, &target, &weights, 0, rows, false)?;
            reference.extend(read_residual(&single, rows)?);
            caches[at + 1].prepare(&mut metadata[at + 1], begin..begin + rows)?;
            segments.push(attention::Segment { first_row: inputs.len() / (6144 * 4),
                request: caches[at + 1].request(&metadata[at + 1], 0)? });
            inputs.extend(input);
        }
        device.run(|| library.copy_h2d(packed.plane("residual")?, &inputs))?;
        // SAFETY: the three disjoint cold-cache requests partition packed residual rows; no host packer is used.
        unsafe { packed.attention_segments(&kernels, &target, &weights, 0, &segments)?; }
        packed.ffn(&kernels, &target, &weights, 0, 17, false)?;
        let actual = read_residual(&packed, 17)?;
        let mut first = 0;
        for (sequence, rows) in [1, 3, 13].into_iter().enumerate() {
            let end = first + rows * 6144;
            compare_residual(&actual[first..end], &reference[first..end], &format!("packed tail sequence={sequence} rows={rows}"), (0.0, 0.0))?;
            assert!(actual[first..end].iter().zip(&reference[first..end])
                .all(|(a, b)| a.to_bits() == b.to_bits()), "packed tail residual must be byte-exact");
            first = end;
        }
        // Rowwise cache producers are shared, so batching cannot alter persistent state.
        device.run(|| {
            for sequence in 0..3 {
                let at = sequence * 2;
                assert_eq!(cache_bytes(&library, caches[at].request(&metadata[at], 0)?)?,
                    cache_bytes(&library, caches[at + 1].request(&metadata[at + 1], 0)?)?,
                    "packed sequence {sequence} cache differs from independent extend");
            }
            Ok(())
        })?;
        // Each attention route is gated above against FP32 with its actual FP8 cache.
        // First O-proj activation QDQ amplifies their ~0.001 path gap 4.0-5.3x:
        // contexts 64/130/270 flip 201/145/108 of 8192 E4M3 payloads, but 0/256
        // scales. O-proj/FFN then produce residual gaps 0.01330/0.01661/0.01542.
        // The 0.020 rel-L2 and 0.003 max/peak bounds are path-regression tripwires
        // (20%/45% over measured maxima), NOT full-layer FP32 fidelity claims.
        // End-to-end golden/fidelity qualification remains a separate gate.
        let mut decode_failures = Vec::new();
        // Rewriting the same final append with identical values lets decode and extend consume identical cache state.
        for (sequence, context) in [64, 130, 270].into_iter().enumerate() {
            let at = sequence * 2;
            caches[at].prepare(&mut metadata[at], context - 1..context)?;
            let request = caches[at].request(&metadata[at], 0)?;
            let input = residual_bytes(sequence, context - 1..context);
            device.run(|| library.copy_h2d(single.plane("residual")?, &input))?;
            // SAFETY: the cache is fully initialized and the one-row append targets its own last slot.
            unsafe { single.attention(&kernels, &target, &weights, 0, request)?; }
            let extend_attention = attention_diagnostic(&single)?;
            single.ffn(&kernels, &target, &weights, 0, 1, false)?;
            let expected = read_residual(&single, 1)?;
            let expected_cache = device.run(|| cache_bytes(&library, request))?;
            device.run(|| library.copy_h2d(decode.plane("residual")?, &input))?;
            // SAFETY: C1 cold metadata has batch id zero, [0,1] offsets and one padded page-table row.
            unsafe { decode.decode(&kernels, &target, &weights, 0, attention::DecodeRequest { request, batch: 1 })?; }
            let decode_attention = attention_diagnostic(&decode)?;
            diagnose_attention(&decode, request, &extend_attention, &decode_attention)?;
            decode.ffn(&kernels, &target, &weights, 0, 1, false)?;
            if let Err(error) = compare_residual(&read_residual(&decode, 1)?, &expected, &format!("native decode context={context}"), (0.020, 0.003)) {
                decode_failures.push(error.to_string());
            }
            assert_eq!(device.run(|| cache_bytes(&library, request))?, expected_cache,
                "native decode must not change cache bytes relative to identical extend");
        }
        ensure!(decode_failures.is_empty(), "{}", decode_failures.join("; "));
        Ok(())
    }

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
