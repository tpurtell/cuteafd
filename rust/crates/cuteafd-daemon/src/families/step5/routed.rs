//! Native router/shared expert bindings with transport-neutral wave boundaries.
use super::weights::load_operand;
use crate::shared::{
    memory::device::{Allocation, Device, Stream},
    spark_intake::SparkDeviceLink,
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

#[derive(Debug)]
struct Admission {
    rows: usize,
    planes: [usize; 14],
    bytes: u64,
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
        let planes = [
            (H, 4),
            (H, 2),
            (E, 4),
            (E, 4),
            (K, 4),
            (K, 4),
            (2 * I, 2),
            (I, 2),
            (H, 2),
            (H, 2),
            (H, 4),
            (H, 4),
            (H, 4),
            (H, 4),
        ]
        .map(|(width, element)| (rows * width * element).max(256));
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
            planes,
            bytes,
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
}
impl RoutedPlan {
    pub fn bytes(&self) -> u64 {
        self.admission.bytes
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.admission.planes.iter().map(|&n| n as u64).sum()
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
        let mut weights = Vec::new();
        for (suffix, dtype, shape) in [
            ("post_attention_layernorm.weight", DType::F32, vec![H]),
            ("moe.gate.weight", DType::Bf16, vec![E, H]),
            ("moe.router_bias", DType::F32, vec![E]),
            (
                "share_expert.gate_up_proj.weight",
                DType::Bf16,
                vec![2 * I, H],
            ),
            ("share_expert.down_proj.weight", DType::Bf16, vec![H, I]),
        ] {
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
        for (name, pointers, scalars) in definitions(&activation) {
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
        ] {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["width"] == width
                    && g["outputs"] == outputs
                    && g["output_dtype"] == dtype
                    && g["skinny_rows"] == 0,
                "Step-5 routed projection geometry: {name}"
            );
        }
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
        let admission = Admission::new(rows, &weights, available, reservations)?;
        Ok(Self {
            layer,
            layout,
            weights,
            admission,
            activation,
            epsilon: contract.norms.post_attention,
        })
    }
}

fn definitions(activation: &str) -> [(&str, &[&str], &[u8]); 8] {
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
    ]
}

pub(super) struct RoutedFfn<'a> {
    stream: Stream<'a>,
    weights: Vec<Allocation<'a>>,
    planes: Vec<Allocation<'a>>,
    programs: Vec<Program<'a>>,
    admission: Admission,
    epsilon: f32,
    layer: usize,
}
impl<'a> RoutedFfn<'a> {
    pub fn load(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        layer: usize,
        plan: RoutedPlan,
    ) -> Result<Self> {
        ensure!(layer == plan.layer, "Step-5 routed plan layer mismatch");
        let resolved = definitions(&plan.activation)
            .iter()
            .map(|(name, pointers, _)| device.run(|| programs.program(name, pointers)))
            .collect::<Result<Vec<_>>>()?;
        let weights = plan
            .weights
            .iter()
            .map(|w| load_operand(device, checkpoint, &plan.layout, w))
            .collect::<Result<Vec<_>>>()?;
        let _scope = cuteafd_ffi::memory_ledger::scope("step5_routed_workspace");
        let planes = plan
            .admission
            .planes
            .iter()
            .map(|&n| Allocation::new(device, n))
            .collect::<Result<Vec<_>>>()?;
        let stream = Stream::new(device)?;
        device.run(|| {
            // SAFETY: private zero plane is retained until stream drainage and never overwritten.
            unsafe {
                device.library.cuda_zero_bytes_async(
                    planes[ZERO].buffer,
                    planes[ZERO].buffer.bytes,
                    stream.raw,
                )
            }
        })?;
        stream.drain()?;
        Ok(Self {
            stream,
            weights,
            planes,
            programs: resolved,
            admission: plan.admission,
            epsilon: plan.epsilon,
            layer,
        })
    }
    pub fn validate_live_weights(&self, device: Device<'_>) -> Result<()> {
        super::weights::validate_live_weights(device.id, &self.weights)
    }

    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.planes[INPUT].buffer
    }
    pub fn output(&self) -> CuteafdDeviceBuffer {
        self.planes[OUTPUT].buffer
    }
    pub fn bytes(&self) -> u64 {
        self.admission.bytes
    }
    #[cfg(test)]
    pub fn route_ids(&self) -> CuteafdDeviceBuffer {
        self.planes[IDS].buffer
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
            scalars.push(Scalar::F32(self.epsilon));
        }
        stream.device.run(|| {
            // SAFETY: admitted live-row views are retained by the transaction until this stream drains.
            unsafe { self.programs[index].launch(pointers, &scalars, stream.raw) }
        })
    }

    pub fn enqueue_on(
        &self,
        rows: usize,
        kind: u32,
        link: &SparkDeviceLink<'a>,
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
        stream.device.run(|| {
            // SAFETY: the diagnostic owner retains normalized rows and routes through drainage.
            unsafe {
                link.dispatch(self.layer, rows, kind, self.planes[IDS].buffer,
                    self.planes[ROUTES].buffer, self.planes[NORM].buffer, stream.raw)
            }
        })?;
        self.shared_on(rows, stream)?;
        stream.device.run(|| {
            // SAFETY: the retained diagnostic link orders collection before reduction.
            unsafe {
                link.collect(stream.raw)?;
                link.reduce(std::ptr::null(), self.planes[ROUTED].buffer.ptr.cast(), rows, stream.raw)
            }
        })?;
        self.finish_on(rows, stream)
    }

    pub fn wave_inputs(&self) -> [CuteafdDeviceBuffer; 3] {
        [self.planes[IDS].buffer, self.planes[ROUTES].buffer, self.planes[NORM].buffer]
    }

    pub fn routed_output(&self) -> CuteafdDeviceBuffer {
        self.planes[ROUTED].buffer
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
        let p = |i: usize| self.planes[i].buffer.ptr;
        let weight = self.weights[0].buffer.ptr;
        self.launch(0, &[p(INPUT), weight, weight, weight, weight, weight, p(NORM)], rows, stream)
    }

    /// Produce native BF16 wire rows and biased top-8 routes without choosing a transport.
    pub fn frontend_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.normalize_on(rows, stream)?;
        let p = |i: usize| self.planes[i].buffer.ptr;
        let w = |i: usize| self.weights[i].buffer.ptr;
        self.launch(1, &[p(NORM), w(1), p(LOGITS)], rows, stream)?;
        stream.device.run(|| {
            // SAFETY: disjoint admitted logits planes and routes remain live through the stream drain.
            unsafe {
                stream.device.library.copy_d2d_async(
                    self.planes[SELECT_LOGITS].buffer,
                    self.planes[LOGITS].buffer,
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
        Ok(())
    }

    pub fn shared_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.check_stream(rows, stream)?;
        let p = |i: usize| self.planes[i].buffer.ptr;
        let w = |i: usize| self.weights[i].buffer.ptr;
        self.launch(3, &[p(NORM), w(3), p(GATE_UP)], rows, stream)?;
        self.launch(4, &[p(GATE_UP), p(ACTIVATION)], rows, stream)?;
        self.launch(5, &[p(ACTIVATION), w(4), p(SHARED)], rows, stream)
    }

    /// The caller orders a completed BF16 routed reduction before this native join.
    pub fn finish_on(&self, rows: usize, stream: &Stream<'a>) -> Result<()> {
        self.check_stream(rows, stream)?;
        let p = |i: usize| self.planes[i].buffer.ptr;
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
    fn routed_admission_charges_logits_copy_and_separate_fp32_branches() -> Result<()> {
        let one = Admission::new(1, &[], u64::MAX, &[])?;
        assert_eq!(one.planes[LOGITS], E * 4);
        assert_eq!(one.planes[SELECT_LOGITS], E * 4);
        assert_eq!(one.planes[IDS], 256);
        assert_eq!(one.planes[ZERO], H * 4);
        assert_eq!(one.planes[COMBINED], H * 4);
        assert_eq!(one.bytes, one.planes.iter().sum::<usize>() as u64);
        assert!(Admission::new(0, &[], u64::MAX, &[]).is_err());
        assert!(Admission::new(4097, &[], u64::MAX, &[]).is_err());
        assert!(Admission::new(1, &[], one.bytes - 1, &[]).is_err());
        assert!(Admission::new(1, &[], one.bytes, &[1]).is_err());
        assert!(Admission::new(1, &[], u64::MAX, &[u64::MAX]).is_err());
        Ok(())
    }
}
