//! Native router/shared expert bindings with transport-neutral wave boundaries.
use super::scratch::{self, Arena, Plane};
use super::weights::load_operand;
use crate::shared::{
    memory::device::{Allocation, Device, Stream},
    launch_grid::Fp8QuantizeGrid,
    spark_intake::SparkLink,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, FfnActivation};
use cuteafd_ffi::{
    programs::{Program, Programs, Scalar},
    CuteafdDeviceBuffer,
};
use cuteafd_loader::plan::{
    checkpoint::Checkpoint,
    families::step5::coordinator::{CoordinatorContract, RankWeightLayout, WeightOperand},
};
use std::ffi::c_void;

/// Builds a gate's Spark request for (layer, rows) from its staged routes and
/// wire rows, on the link's exchange worker.
pub(crate) type GateRequest = fn(usize, usize, cuteafd_transport::ExpertV2Dtype, Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, bytes::Bytes)
    -> Result<cuteafd_transport::ExpertProtocolV2Request>;

const H: usize = 4096;
const E: usize = 352;
const K: usize = 8;
const I: usize = 1536;
const INPUT: usize = 0;
const NORM: usize = 1;
const LOGITS: usize = 2;
const SELECT_LOGITS: usize = 3;
const IDS: usize = 4;
const ROUTES: usize = 5;
const GATE_UP: usize = 6;
const ACTIVATION: usize = 7;
const SHARED: usize = 8;
const ROUTED: usize = 9;
const ZERO: usize = 10;
const WIDE: usize = 11;
const COMBINED: usize = 12;
const OUTPUT: usize = 13;
const WIRE: usize = 14;

#[derive(Debug, Clone)]
struct Admission {
    rows: usize,
    /// Rows of every in-flight unit held by the live planes (`span >= rows`).
    span: usize,
    planes: [usize; 15],
    bytes: u64,
    /// A whole-owned (CSA) layer takes the second replicated-reply plane.
    full: bool,
}
impl Admission {
    fn new(
        rows: usize,
        weights: &[WeightOperand],
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&rows),
            "Step-5 routed rows must be 1..=4096"
        );
        let planes = cuteafd_loader::plan::families::step5::native::routed_planes(rows)?;
        let span = rows;
        let bytes = planes
            .iter()
            .map(|&n| n as u64)
            .chain(weights.iter().map(|w| w.allocation_bytes))
            .try_fold(0u64, |a, b| a.checked_add(b))
            .context("routed admission overflow")?;
        let total = reservations
            .iter()
            .try_fold(bytes, |a, b| a.checked_add(*b))
            .context("routed reservation overflow")?;
        ensure!(
            total <= available,
            "Step-5 routed chain needs {total} bytes, available {available}"
        );
        Ok(Self {
            rows,
            span,
            planes,
            bytes,
            full: false,
        })
    }
}

pub(super) struct RoutedPlan {
    layer: usize,
    layout: RankWeightLayout,
    weights: Vec<WeightOperand>,
    admission: Admission,
    activation: String,
    epsilon: f32,
    bf16_input: bool,
    /// Native shared gate/up/down `.weight` names (EXL3 checkpoints).
    native: Option<[String; 3]>,
}
impl RoutedPlan {
    pub(super) fn with_expert_format(mut self, format: cuteafd_loader::formats::fp8_experts::ExpertFormat) -> Self {
        self.bf16_input = format.bf16_input();
        self
    }

    pub fn bytes(&self) -> u64 {
        self.admission.bytes
    }

    /// Live planes hold `span` rows (every in-flight unit); scratch holds one.
    pub(super) fn with_span(mut self, span: usize) -> Result<Self> {
        ensure!(span >= self.admission.rows, "Step-5 routed span smaller than unit capacity");
        let live = cuteafd_loader::plan::families::step5::native::routed_planes(span)?;
        for plane in 0..15 {
            if cuteafd_loader::plan::families::step5::native::routed_live(plane) {
                self.admission.planes[plane] = live[plane];
            }
        }
        self.admission.span = span;
        Ok(self)
    }
    pub fn workspace_bytes(&self) -> u64 {
        cuteafd_loader::plan::families::step5::native::plane_bytes(&self.admission.planes).expect("admitted planes")
    }

    pub fn prepare(
        programs: &Programs<'_>,
        checkpoint: &Checkpoint,
        layer: usize,
        rows: usize,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        Self::prepare_layout(programs, checkpoint, layer, rows, available, reservations,
            RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?)
    }

    pub fn for_execution(
        programs: &Programs<'_>,
        checkpoint: &Checkpoint,
        layer: usize,
        rows: usize,
        available: u64,
        reservations: &[u64],
        rank: usize,
        ownership: &cuteafd_loader::plan::families::step5::ownership::ExecutionOwnership,
    ) -> Result<Self> {
        ensure!(ownership.layer(layer, rank)?.ffn, "Step-5 routed FFN is not owned by rank {rank}");
        Self::prepare_layout(programs, checkpoint, layer, rows, available, reservations,
            RankWeightLayout::for_execution(checkpoint, rank, ownership)?)
    }

    fn prepare_layout(
        programs: &Programs<'_>,
        checkpoint: &Checkpoint,
        layer: usize,
        rows: usize,
        available: u64,
        reservations: &[u64],
        layout: RankWeightLayout,
    ) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        ensure!(
            contract
                .layers
                .get(layer)
                .is_some_and(|l| l.dense_activation.is_none()),
            "Step-5 routed owner requires a routed target layer"
        );
        let Some(FfnActivation::SiluPostClampFp32Product { limit }) =
            contract.layers[layer].shared_activation
        else {
            anyhow::bail!("Step-5 unsupported shared activation");
        };
        ensure!(
            limit == 7.0 || limit == 10.0,
            "Step-5 unsupported shared clamp"
        );
        let activation = format!("step5_post_silu_w1536_c{}", limit as u32);
        let native_names = ["gate", "up", "down"].map(|p| format!("model.layers.{layer}.share_expert.{p}_proj.weight"));
        let native = native_names.iter().all(|name| layout.core.contains_key(name));
        ensure!(native || native_names.iter().all(|name| !layout.core.contains_key(name)),
            "Step-5 layer {layer} shared FFN mixes native and BF16 operands");
        let mut weights = Vec::new();
        let bf16_shared: &[(&str, DType, Vec<usize>)] = &[
            ("share_expert.gate_up_proj.weight", DType::Bf16, vec![2 * I, H]),
            ("share_expert.down_proj.weight", DType::Bf16, vec![H, I]),
        ];
        let shared: &[(&str, DType, Vec<usize>)] = if native { &[] } else { bf16_shared };
        for (suffix, dtype, shape) in [
            ("post_attention_layernorm.weight", DType::F32, vec![H]),
            ("moe.gate.weight", DType::Bf16, vec![E, H]),
            ("moe.router_bias", DType::F32, vec![E]),
        ].into_iter().chain(shared.iter().cloned()) {
            let name = format!("model.layers.{layer}.{suffix}");
            let weight = layout
                .operands
                .iter()
                .find(|w| w.name == name)
                .context("missing routed operand")?;
            ensure!(
                weight.dtype == dtype && weight.shape == shape,
                "Step-5 routed shape/dtype: {name}"
            );
            weights.push(weight.clone());
        }
        if native {
            for (name, shape) in native_names.iter().zip([[I, H], [I, H], [H, I]]) {
                let core = &layout.core[name];
                ensure!(core.shape == shape && core.ranks == 1, "{name}: native shared FFN must be whole {shape:?}");
            }
            let split = format!("step5_post_silu_split_w1536_c{}", limit as u32);
            let spec = programs.spec(&split)?;
            ensure!(spec.pointers == ["gate", "up", "out"] && spec.scalar_kinds == b"i"
                && spec.capacity_rows as usize >= rows && spec.geometry["width"] == I
                && spec.geometry["limit"].as_f64() == Some(limit as f64), "{split}: split post-SiLU ABI");
        }
        for (name, pointers, scalars) in definitions(&activation).into_iter()
            .filter(|(name, _, _)| !native || !matches!(*name, "step5_shared_gate_up" | "step5_shared_down")) {
            let spec = programs.spec(name)?;
            ensure!(
                spec.pointers == pointers
                    && spec.scalar_kinds == scalars
                    && spec.capacity_rows as usize >= rows
                    && spec.scratch.is_empty(),
                "Step-5 routed ABI/capacity: {name}"
            );
        }
        let norm = &programs.spec("step5_hidden_norm")?.geometry;
        ensure!(
            norm["width"] == H
                && norm["heads"] == 1
                && norm["zero_centered"] == true
                && norm["rotary"] == 0
                && norm["bias"] == false
                && norm["subtract_mean"] == false
                && norm["epsilon"] == "runtime_float32",
            "Step-5 routed norm geometry"
        );
        for (name, width, outputs, dtype) in [
            ("step5_router", H, E, "torch.float32"),
            ("step5_shared_gate_up", H, 2 * I, "torch.bfloat16"),
            ("step5_shared_down", I, H, "torch.bfloat16"),
        ].into_iter().filter(|(name, ..)| !native || *name == "step5_router") {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["width"] == width
                    && g["outputs"] == outputs
                    && g["output_dtype"] == dtype
                    && g["skinny_rows"] == 0,
                "Step-5 routed projection geometry: {name}"
            );
        }
        let quant = &programs.spec("step5_expert_input_quant")?.geometry;
        ensure!(quant["hidden"] == H && quant["row_bytes"] == H + H / 32
            && quant["expected_m"] == 80 && quant["amax_floor"].as_f64() == Some(1e-4),
            "Step-5 routed input quantizer geometry");
        let route = &programs.spec("step5_route_weights")?.geometry;
        let post = &programs.spec(&activation)?.geometry;
        ensure!(
            route["experts"] == E
                && route["topk"] == K
                && route["epsilon"].as_f64() == Some(1e-20)
                && route["factor"] == 3.0
                && post["width"] == I
                && post["limit"].as_f64() == Some(limit as f64)
                && programs.spec("step5_residual_bf16")?.geometry["width"] == H
                && programs.spec("step5_residual_f32")?.geometry["width"] == H,
            "Step-5 routed pointwise geometry"
        );
        let mut admission = Admission::new(rows, &weights, available, reservations)?;
        admission.full = contract.layers[layer].sliding.is_none();
        if native {
            admission.bytes = native_names.iter().try_fold(admission.bytes, |n, name|
                n.checked_add(layout.core[name].bytes as u64)).context("routed native admission overflow")?;
        }
        Ok(Self {
            layer,
            layout,
            weights,
            admission,
            activation,
            epsilon: contract.norms.post_attention,
            bf16_input: true,
            native: native.then_some(native_names),
        })
    }
}

fn definitions(activation: &str) -> [(&str, &[&str], &[u8]); 9] {
    [
        (
            "step5_hidden_norm",
            &["x", "weight", "bias", "cos", "sin", "positions", "out"],
            b"if",
        ),
        ("step5_router", &["x", "w", "out"], b"i"),
        (
            "step5_route_weights",
            &["logits", "bias", "ids", "out"],
            b"i",
        ),
        ("step5_shared_gate_up", &["x", "w", "out"], b"i"),
        (activation, &["gate_up", "out"], b"i"),
        ("step5_shared_down", &["x", "w", "out"], b"i"),
        ("step5_residual_bf16", &["residual", "delta", "out"], b"i"),
        ("step5_residual_f32", &["residual", "delta", "out"], b"i"),
        ("step5_expert_input_quant", &["source_ptr", "values_ptr", "scale_rows_ptr", "scale_mma_ptr"], b"ii"),
    ]
}

struct RoutedWeights<'a> {
    weights: Vec<Allocation<'a>>,
    programs: Vec<Program<'a>>,
    epsilon: f32,
    layer: usize,
    bf16_input: bool,
    quantize_grid: Fp8QuantizeGrid,
    /// EXL3 shared gate, up, down and the split activation program.
    native: Option<([super::native::NativeLinear<'a>; 3], Program<'a>)>,
}

pub(super) struct RoutedFfn<'a> {
    stream: Stream<'a>,
    planes: Vec<Plane<'a>>,
    native_scratch: Option<std::rc::Rc<super::native::NativeScratch<'a>>>,
    admission: Admission,
    immutable: std::rc::Rc<RoutedWeights<'a>>,
    /// Row offset of the selected prefill unit inside the live planes.
    unit: std::cell::Cell<usize>,
}
impl<'a> RoutedFfn<'a> {
    pub(super) fn workspace_bytes(&self) -> u64 {
        cuteafd_loader::plan::families::step5::native::plane_bytes(&self.admission.planes).expect("admitted planes")
    }


    pub fn load(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        layer: usize,
        plan: RoutedPlan,
    ) -> Result<Self> {
        Self::load_in(device, programs, checkpoint, layer, plan, None)
    }

    pub(super) fn load_in(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        layer: usize,
        plan: RoutedPlan,
        arena: Option<(&Arena<'a>, usize)>,
    ) -> Result<Self> {
        Self::load_shared(device, programs, checkpoint, layer, plan, arena, None)
    }

    /// `native_scratch` is the rank's shared projection scratch (native shared FFNs).
    pub(super) fn load_shared(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        layer: usize,
        plan: RoutedPlan,
        arena: Option<(&Arena<'a>, usize)>,
        native_scratch: Option<std::rc::Rc<super::native::NativeScratch<'a>>>,
    ) -> Result<Self> {
        ensure!(layer == plan.layer, "Step-5 routed plan layer mismatch");
        let native_set = plan.native.is_some();
        // Native layers resolve placeholders for the unused BF16 shared programs:
        // the definitions keep their indices; those slots are never launched.
        let resolved = definitions(&plan.activation)
            .iter()
            .map(|(name, pointers, _)| {
                let name = if native_set && matches!(*name, "step5_shared_gate_up" | "step5_shared_down") {
                    "step5_router" } else { name };
                device.run(|| programs.program(name, if name == "step5_router" { &["x", "w", "out"] } else { pointers }))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scratch_out = None;
        let native = match &plan.native {
            None => None,
            Some(names) => {
                let mut core = super::native::NativeLayer::load(device, programs, checkpoint, &plan.layout,
                    plan.admission.rows, |name| names.iter().any(|n| n == name))?;
                scratch_out = Some(match native_scratch {
                    Some(scratch) => scratch,
                    None => std::rc::Rc::new(super::native::NativeScratch::allocate(device,
                        &core.scratch_layout(plan.admission.rows)?.context("native shared scratch")?)?),
                });
                let mut take = |name: &str| -> Result<super::native::NativeLinear<'a>> {
                    std::rc::Rc::try_unwrap(core.linears.remove(name).context("native shared operand")?)
                        .map_err(|_| anyhow::anyhow!("native shared operand aliased"))
                };
                let linears = [take(&names[0])?, take(&names[1])?, take(&names[2])?];
                let split = format!("step5_post_silu_split_w1536_c{}",
                    plan.activation.rsplit('c').next().context("activation limit")?);
                Some((linears, device.run(|| programs.program(&split, &["gate", "up", "out"]))?))
            }
        };
        let weights = plan
            .weights
            .iter()
            .map(|w| load_operand(device, checkpoint, &plan.layout, w))
            .collect::<Result<Vec<_>>>()?;
        Self::workspace(device, plan.admission, std::rc::Rc::new(RoutedWeights {
            weights, programs: resolved, epsilon: plan.epsilon, layer,
            bf16_input: plan.bf16_input,
            quantize_grid: Fp8QuantizeGrid::new(device.library.sm_count()?, None)?,
            native,
        }), arena).map(|mut owner| { owner.native_scratch = scratch_out; owner })
    }

    /// Share only native weights/programs; route, reduction and zero planes are private.
    pub(super) fn new_lane(&self) -> Result<Self> {
        let mut lane = Self::workspace(self.stream.device, self.admission.clone(), self.immutable.clone(), None)?;
        if let Some(shared) = &self.native_scratch { lane.native_scratch = Some(shared.sibling(self.stream.device)?); }
        Ok(lane)
    }

    fn workspace(device: Device<'a>, admission: Admission,
        immutable: std::rc::Rc<RoutedWeights<'a>>, arena: Option<(&Arena<'a>, usize)>) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("step5_routed_workspace");
        let spans = cuteafd_loader::plan::families::step5::native::routed_layout(admission.rows,
            admission.span, admission.full)?;
        let planes = scratch::region_planes(device, &spans, arena)?;
        let stream = Stream::new(device)?;
        Ok(Self { stream, planes, native_scratch: None, admission, immutable, unit: std::cell::Cell::new(0) })
    }
    pub fn validate_live_weights(&self, device: Device<'_>) -> Result<()> {
        super::weights::validate_live_weights(device.id, &self.immutable.weights)
    }

    /// Live planes (OUTPUT, INPUT, SHARED, ROUTED) are viewed at the selected
    /// unit's rows; hook-local scratch always starts at row zero.
    fn plane(&self, index: usize) -> CuteafdDeviceBuffer {
        let buffer = self.planes[index].buffer;
        if !cuteafd_loader::plan::families::step5::native::routed_live(index) { return buffer; }
        let stride = buffer.bytes / self.admission.span;
        let offset = self.unit.get() * stride;
        CuteafdDeviceBuffer { ptr: buffer.ptr.wrapping_byte_add(offset), bytes: buffer.bytes - offset, ..buffer }
    }

    /// Selects a prefill unit's rows of the live planes; whole steps use row 0.
    pub(super) fn select_unit(&self, start: usize, rows: usize) -> Result<()> {
        ensure!(start.checked_add(rows).is_some_and(|end| end <= self.admission.span) && rows <= self.admission.rows,
            "Step-5 routed unit rows outside admitted span");
        self.unit.set(start);
        Ok(())
    }

    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.plane(INPUT)
    }
    pub fn output(&self) -> CuteafdDeviceBuffer {
        self.plane(OUTPUT)
    }
    pub fn bytes(&self) -> u64 {
        self.admission.bytes
    }
    #[cfg(test)]
    pub fn route_ids(&self) -> CuteafdDeviceBuffer {
        self.plane(IDS)
    }

    fn launch(
        &self,
        index: usize,
        pointers: &[*mut c_void],
        rows: usize,
        stream: &Stream<'a>,
    ) -> Result<()> {
        let mut scalars = vec![Scalar::I32(rows as i32)];
        if index == 0 {
            scalars.push(Scalar::F32(self.immutable.epsilon));
        }
        stream.device.run(|| {
            // SAFETY: admitted live-row views are retained by the transaction until this stream drains.
            unsafe { self.immutable.programs[index].launch(pointers, &scalars, stream.raw) }
        })
    }

    /// One routed layer through `link`'s exchange on `stream`: the front end,
    /// the wave out (built by `request` on the link's worker), the shared
    /// expert, the device wait for the partials and their reduce.
    pub fn enqueue_on(
        &self,
        rows: usize,
        link: &std::cell::RefCell<SparkLink<'a>>,
        request: GateRequest,
        stream: &Stream<'a>,
    ) -> Result<()> {
        ensure!(
            (1..=self.admission.rows).contains(&rows),
            "Step-5 routed live rows exceed admission"
        );
        ensure!(
            stream.device.id == self.stream.device.id
                && std::ptr::eq(stream.device.library, self.stream.device.library),
            "Step-5 routed stream mismatch"
        );
        self.frontend_on(rows, stream)?;
        let layer = self.immutable.layer;
        let layout = crate::shared::experts::staging::RouteStaging::new(rows, K, self.wire_row_bytes())?;
        let dtype = self.wire_dtype();
        let mut link = link.borrow_mut();
        let exchange = link.exchange()?;
        let wave = stream.device.run(|| {
            // SAFETY: the diagnostic owner retains normalized rows and routes
            // through drainage; the copies read them in stream order.
            unsafe {
                exchange.send(rows, layout, self.wave_inputs(), stream.raw,
                    Box::new(move |routes, wire| request(layer, rows, dtype, routes, wire)))
            }
        })?;
        self.shared_on(rows, stream)?;
        let slot = wave.slot();
        stream.device.run(|| {
            exchange.land(wave, stream.raw)?;
            // SAFETY: `land` orders the slot's planes on this stream before
            // the reduce; the routed plane is this owner's disjoint output.
            unsafe { exchange.reduce(slot, std::ptr::null(), self.plane(ROUTED).ptr.cast(), rows, stream.raw) }
        })?;
        drop(link);
        self.finish_on(rows, stream)
    }

    pub fn wire_dtype(&self) -> cuteafd_transport::ExpertV2Dtype {
        if self.immutable.bf16_input { cuteafd_transport::ExpertV2Dtype::Bf16 }
        else { cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32 }
    }

    pub fn wire_row_bytes(&self) -> usize {
        if self.immutable.bf16_input { H * 2 } else { H + H / 32 }
    }

    pub fn wave_inputs(&self) -> [CuteafdDeviceBuffer; 3] {
        [self.plane(IDS), self.plane(ROUTES), self.plane(if self.immutable.bf16_input { NORM } else { WIRE })]
    }

    pub fn routed_output(&self) -> CuteafdDeviceBuffer {
        self.plane(ROUTED)
    }

    fn check_stream(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        ensure!((1..=self.admission.rows).contains(&rows)
            && stream.device.id == self.stream.device.id
            && std::ptr::eq(stream.device.library, self.stream.device.library),
            "Step-5 routed live rows/stream mismatch");
        Ok(())
    }

    /// A receiving replica needs the native shared expert, not another router/wave.
    pub fn normalize_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.check_stream(rows, stream)?;
        let p = |i: usize| self.plane(i).ptr;
        let weight = self.immutable.weights[0].buffer.ptr;
        self.launch(0, &[p(INPUT), weight, weight, weight, weight, weight, p(NORM)], rows, stream)
    }

    /// Produce native wire rows and biased top-8 routes without choosing a transport.
    pub fn frontend_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.normalize_on(rows, stream)?;
        let p = |i: usize| self.plane(i).ptr;
        let w = |i: usize| self.immutable.weights[i].buffer.ptr;
        self.launch(1, &[p(NORM), w(1), p(LOGITS)], rows, stream)?;
        stream.device.run(|| {
            // SAFETY: disjoint admitted logits planes and routes remain live through the stream drain.
            unsafe {
                stream.device.library.copy_d2d_async(
                    self.plane(SELECT_LOGITS),
                    self.plane(LOGITS),
                    rows * E * 4,
                    stream.raw,
                )?;
                stream.device.library.router_select(
                    p(SELECT_LOGITS),
                    w(2),
                    std::ptr::null(),
                    std::ptr::null(),
                    p(IDS),
                    p(ROUTES),
                    rows,
                    E,
                    K,
                    3.0,
                    true,
                    stream.raw,
                )
            }
        })?;
        // The selector mutates only its copy. Recompute official (sigmoid+bias)-bias weights from raw logits.
        self.launch(2, &[p(LOGITS), w(2), p(IDS), p(ROUTES)], rows, stream)?;
        if !self.immutable.bf16_input {
            let grid = self.immutable.quantize_grid.blocks(rows, H);
            stream.device.run(|| {
                // SAFETY: the admitted wire plane holds packed values and K32 scales;
                // its source and outputs stay live until stream-ordered staging drains.
                unsafe { self.immutable.programs[8].launch(
                    &[p(NORM), p(WIRE), p(WIRE).wrapping_byte_add(H), p(ZERO)],
                    &[Scalar::I32(rows as i32), Scalar::I32(grid as i32)], stream.raw) }
            })?;
        }
        Ok(())
    }

    pub fn shared_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.check_stream(rows, stream)?;
        let p = |i: usize| self.plane(i).ptr;
        if let Some((native, split)) = &self.immutable.native {
            let scratch = &self.native_scratch.as_ref().context("native shared scratch")?.scratch;
            let view = |i: usize, bytes: usize| CuteafdDeviceBuffer { bytes, ..self.plane(i) };
            let gate = view(GATE_UP, rows * I * 2);
            let up = CuteafdDeviceBuffer { ptr: gate.ptr.wrapping_byte_add(rows * I * 2), ..gate };
            native[0].launch(rows, view(NORM, rows * H * 2), gate, scratch, stream.raw)?;
            native[1].launch(rows, view(NORM, rows * H * 2), up, scratch, stream.raw)?;
            stream.device.run(|| {
                // SAFETY: gate/up are disjoint row blocks of the admitted GATE_UP plane,
                // retained with ACTIVATION until this stream drains.
                unsafe { split.launch(&[gate.ptr, up.ptr, p(ACTIVATION)], &[Scalar::I32(rows as i32)], stream.raw) }
            })?;
            return native[2].launch(rows, view(ACTIVATION, rows * I * 2), view(SHARED, rows * H * 2), scratch, stream.raw);
        }
        let w = |i: usize| self.immutable.weights[i].buffer.ptr;
        self.launch(3, &[p(NORM), w(3), p(GATE_UP)], rows, stream)?;
        self.launch(4, &[p(GATE_UP), p(ACTIVATION)], rows, stream)?;
        self.launch(5, &[p(ACTIVATION), w(4), p(SHARED)], rows, stream)
    }

    /// The caller orders a completed BF16 routed reduction before this native join.
    pub fn finish_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.check_stream(rows, stream)?;
        let p = |i: usize| self.plane(i).ptr;
        stream.device.run(|| {
            // SAFETY: the reused zero plane is disjoint and ordered before its consumer.
            unsafe { stream.device.library.cuda_zero_bytes_async(self.plane(ZERO),
                rows * H * 4, stream.raw) }
        })?;
        // Match residual + (float(routed) + float(shared)), not (residual + routed) + shared.
        self.launch(6, &[p(ZERO), p(ROUTED), p(WIDE)], rows, stream)?;
        self.launch(6, &[p(WIDE), p(SHARED), p(COMBINED)], rows, stream)?;
        self.launch(7, &[p(INPUT), p(COMBINED), p(OUTPUT)], rows, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires sealed native library, pinned Step-5 snapshot and CUDA"]
    fn native_lane_shares_weights_and_isolates_workspace() -> Result<()> {
        // SAFETY: the operator supplies the matching sealed native library.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let programs = library.programs()?.with_manifest(std::path::Path::new(
            &std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
        let checkpoint = Checkpoint::open(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let plan = RoutedPlan::prepare(&programs, &checkpoint, 3, 1, u64::MAX, &[])?;
        let owner = RoutedFfn::load(device, &programs, &checkpoint, 3, plan)?;
        let lane = owner.new_lane()?;
        ensure!(std::rc::Rc::ptr_eq(&owner.immutable, &lane.immutable)
            && owner.stream.raw != lane.stream.raw
            && owner.planes.iter().zip(&lane.planes).all(|(a, b)|
                a.buffer.ptr != b.buffer.ptr && a.buffer.bytes == b.buffer.bytes),
            "Step-5 lane immutable/private storage identities");
        let immutable = lane.immutable.clone();
        drop(owner);
        ensure!(std::rc::Rc::strong_count(&immutable) == 2,
            "Step-5 weights must survive the original lane");
        lane.stream.drain()?;
        drop(lane);
        ensure!(std::rc::Rc::strong_count(&immutable) == 1,
            "Step-5 lane must release its immutable reference after drainage");
        Ok(())
    }

    #[test]
    fn routed_admission_charges_logits_copy_and_separate_fp32_branches() -> Result<()> {
        let one = Admission::new(1, &[], u64::MAX, &[])?;
        assert_eq!(one.planes[LOGITS], E * 4);
        assert_eq!(one.planes[SELECT_LOGITS], E * 4);
        assert_eq!(one.planes[IDS], 256);
        assert_eq!(one.planes[ZERO], H * 4);
        assert_eq!(one.planes[COMBINED], H * 4);
        assert_eq!(one.planes[WIRE], H + H / 32);
        assert_eq!(one.bytes, one.planes.iter().sum::<usize>() as u64);
        assert!(Admission::new(0, &[], u64::MAX, &[]).is_err());
        assert!(Admission::new(4097, &[], u64::MAX, &[]).is_err());
        assert!(Admission::new(1, &[], one.bytes - 1, &[]).is_err());
        assert!(Admission::new(1, &[], one.bytes, &[1]).is_err());
        assert!(Admission::new(1, &[], u64::MAX, &[u64::MAX]).is_err());
        Ok(())
    }
}
