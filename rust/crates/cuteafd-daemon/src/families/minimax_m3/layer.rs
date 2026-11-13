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
        let plane = |name| owner.attention.diagnostic_plane(name);
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
