//! Admitted final BF16 narrowing, zero-centered norm and shared vocabulary head.
use super::weights::load_operand;
use crate::shared::{
    memory::device::{Allocation, Device, Stream},
    target_head::{self, TargetHead},
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{
    programs::{Program, Programs, Scalar, VocabularyHead},
    CuteafdDeviceBuffer,
};
use cuteafd_loader::plan::{
    checkpoint::Checkpoint,
    families::step5::{Config, coordinator::{RankWeightLayout, WeightOperand}},
};

const H: usize = 4096;
const V: usize = 128896;
const POINTERS: &[&str] = &["x", "weight", "bias", "cos", "sin", "positions", "out"];

pub(super) struct HeadPlan {
    layout: RankWeightLayout,
    weights: Vec<WeightOperand>,
    planes: [usize; 4],
    rows: usize,
    epsilon: f32,
    bytes: u64,
    predictor: bool,
}

impl HeadPlan {
    fn plane_bytes(rows: usize) -> Result<[usize; 4]> {
        ensure!((1..=4096).contains(&rows), "Step-5 head row capacity must be 1..4096");
        cuteafd_loader::plan::families::step5::native::head_planes(rows)
    }

    pub fn prepare(programs: &Programs<'_>, checkpoint: &Checkpoint, rows: usize) -> Result<Self> {
        Self::prepare_layout(programs, checkpoint, rows, RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?, None)
    }

    pub fn for_execution(programs: &Programs<'_>, checkpoint: &Checkpoint, rows: usize,
        ownership: &cuteafd_loader::plan::families::step5::ownership::ExecutionOwnership) -> Result<Self> {
        Self::prepare_layout(programs, checkpoint, rows, RankWeightLayout::for_execution(checkpoint,
            ownership.execution().spec().head_gpu as usize, ownership)?, None)
    }

    pub(super) fn for_mtp_stage(programs: &Programs<'_>, checkpoint: &Checkpoint,
        stage: usize, rows: usize) -> Result<Self> {
        Self::prepare_layout(programs, checkpoint, rows,
            RankWeightLayout::for_mtp_stage(checkpoint, stage)?, Some(stage))
    }

    fn prepare_layout(programs: &Programs<'_>, checkpoint: &Checkpoint, rows: usize,
        layout: RankWeightLayout, stage: Option<usize>) -> Result<Self> {
        let planes = Self::plane_bytes(rows)?;
        let cfg = Config::parse(checkpoint)?;
        let names = if let Some(stage) = stage {
            let layer = cfg.layers + stage;
            [format!("model.layers.{layer}.transformer.shared_head.norm.weight"),
                format!("model.layers.{layer}.transformer.shared_head.output.weight")]
        } else { ["model.norm.weight".into(), "lm_head.weight".into()] };
        let weights = [(names[0].as_str(), DType::F32, vec![H]),
            (names[1].as_str(), DType::Bf16, vec![V, H])]
            .into_iter().map(|(name, dtype, shape)| {
                let weight = layout.operands.iter().find(|w| w.name == name)
                    .context("Step-5 final operand missing")?;
                ensure!(weight.dtype == dtype && weight.shape == shape,
                    "Step-5 final operand shape/dtype mismatch: {name}");
                Ok(weight.clone())
            }).collect::<Result<Vec<_>>>()?;
        let spec = programs.spec("step5_final_norm")?;
        let g = &spec.geometry;
        ensure!(spec.pointers == POINTERS && spec.scalar_kinds == b"if"
            && spec.capacity_rows as usize >= rows && spec.scratch.is_empty()
            && g["width"] == H && g["heads"] == 1 && g["zero_centered"] == true
            && g["subtract_mean"] == false && g["bias"] == false && g["rotary"] == 0
            && g["epsilon"] == "runtime_float32", "Step-5 final norm ABI/geometry mismatch");
        let bytes = weights.iter().map(|w| w.allocation_bytes)
            .chain(planes.iter().map(|&n| n as u64))
            .try_fold(0u64, |n, v| n.checked_add(v)).context("Step-5 head byte overflow")?;
        Ok(Self { layout, weights, planes, rows, epsilon: cfg.norm_eps, bytes,
            predictor: stage.is_some() })
    }

    pub fn bytes(&self) -> u64 { self.bytes }

    pub(super) fn workspace_bytes(&self) -> u64 {
        self.planes.iter().map(|&n| n as u64).sum()
    }

    pub(super) fn immutable_bytes(&self) -> u64 {
        self.bytes - self.workspace_bytes()
    }
}

struct HeadWeights<'a> {
    norm: Program<'a>,
    weights: Vec<Allocation<'a>>,
    epsilon: f32,
    predictor: bool,
}

pub(super) struct TargetHeadOwner<'a> {
    // Drain before programs, library handles, weights and workspace retire.
    stream: Stream<'a>,
    head: VocabularyHead<'a>,
    planes: Vec<Allocation<'a>>,
    rows: usize,
    immutable: std::rc::Rc<HeadWeights<'a>>,
}

impl<'a> TargetHeadOwner<'a> {
    pub fn load(device: Device<'a>, programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        plan: HeadPlan, available: u64, reservations: &[u64]) -> Result<Self> {
        let total = reservations.iter().try_fold(plan.bytes, |n, v| n.checked_add(*v))
            .context("Step-5 final head reservation overflow")?;
        ensure!(total <= available, "Step-5 final head and reservations exceed admission");
        device.run(|| {
            let norm = programs.program("step5_final_norm", POINTERS)?;
            let weights = plan.weights.iter().map(|w| load_operand(device, checkpoint, &plan.layout, w))
                .collect::<Result<Vec<_>>>()?;
            Self::workspace(device, plan.rows, std::rc::Rc::new(HeadWeights {
                norm, weights, epsilon: plan.epsilon, predictor: plan.predictor,
            }))
        })
    }

    /// Every lane has its own BLAS handle, stream, norm/logit planes and scratch.
    pub(super) fn new_lane(&self) -> Result<Self> {
        Self::workspace(self.stream.device, self.rows, self.immutable.clone())
    }

    fn workspace(device: Device<'a>, rows: usize,
        immutable: std::rc::Rc<HeadWeights<'a>>) -> Result<Self> {
        device.run(|| {
            let planes = HeadPlan::plane_bytes(rows)?.iter().map(|&n| Allocation::new(device, n))
                .collect::<Result<Vec<_>>>()?;
            // SAFETY: private workspace survives the private handle on this device.
            let head = unsafe { device.library.vocabulary_head_rows(planes[3].buffer.ptr,
                H as u32, rows as u32, V as u32)? };
            let stream = Stream::new(device)?;
            Ok(Self { stream, head, planes, rows, immutable })
        })
    }

    /// Inspect the already-drained norm plane without introducing another target launch.
    pub fn normalized(&self) -> Result<Vec<f32>> {
        let mut bytes = vec![0; self.rows * H * 2];
        self.stream.device.run(|| self.stream.device.library.copy_d2h(&mut bytes, self.planes[1].buffer))?;
        Ok(bytes.chunks_exact(2).map(|b|
            f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16)).collect())
    }

    /// The transaction retains this owner and input through the lane stream's drain.
    pub(super) fn device_logits_on(&self, input: CuteafdDeviceBuffer, rows: usize,
        stream: &Stream<'a>) -> Result<crate::shared::token_io::DeviceLogits> {
        super::transaction::validate_active_rows(rows, self.rows)?;
        ensure!(!input.ptr.is_null() && input.bytes >= rows * H * 4
            && input.device_id == self.stream.device.id && stream.device.id == input.device_id
            && std::ptr::eq(stream.device.library, self.stream.device.library),
            "Step-5 final hidden extent/device/stream");
        let mut guard = stream.cancellation_guard("Step-5 final head input borrow");
        stream.device.run(|| {
            let library = stream.device.library;
            let p = self.planes.iter().map(|p| p.buffer.ptr).collect::<Vec<_>>();
            let weight = self.immutable.weights[0].buffer.ptr;
            // SAFETY: admitted lane-private planes, head handle and input stay live
            // through the transaction drain; runtime rows never exceed capacity.
            unsafe {
                library.cuda_f32_to_bf16_async(input, self.planes[0].buffer, rows * H, stream.raw)?;
                self.immutable.norm.launch(&[p[0], weight, weight, weight, weight, weight, p[1]],
                    &[Scalar::I32(rows as i32), Scalar::F32(self.immutable.epsilon)], stream.raw)?;
                if self.immutable.predictor {
                    crate::shared::draft::launch_bf16_head(&self.head, p[1].cast(),
                        self.immutable.weights[1].buffer.ptr.cast(), p[2].cast(), rows, stream.raw)?;
                } else {
                    target_head::launch(TargetHead::configured(), &self.head, p[1],
                        self.immutable.weights[1].buffer.ptr, p[2].cast(), rows, stream.raw)?;
                }
            }
            guard.complete = true;
            Ok(crate::shared::token_io::DeviceLogits { ptr: p[2].cast_const(), rows, vocab: V,
                stride: V, stream: stream.raw, greedy: None })
        })
    }

    /// Caller drains the target chain first; this diagnostic downloads all rows.
    pub fn logits(&self, input: CuteafdDeviceBuffer) -> Result<Vec<f32>> {
        let mut guard = self.stream.cancellation_guard("Step-5 diagnostic head input borrow");
        let logits = self.device_logits_on(input, self.rows, &self.stream)?;
        let values = self.stream.device.run(|| logits.to_host(self.stream.device.library))?;
        guard.complete = true;
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite Step-5 logits");
        Ok(values)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    pub(in super::super) fn admission_plan(rows: usize) -> HeadPlan {
        let planes = HeadPlan::plane_bytes(rows).unwrap();
        HeadPlan {
            layout: RankWeightLayout { rank: 0, ranks: 1, operands: vec![], resident_bytes: 0 },
            weights: vec![], planes, rows, epsilon: 1e-5, predictor: false,
            bytes: ((H * 4 + V * H * 2) as u64) + planes.iter().map(|&n| n as u64).sum::<u64>(),
        }
    }

    #[test]
    #[ignore = "requires sealed native library, pinned Step-5 snapshot and CUDA"]
    fn native_lane_shares_weights_and_isolates_workspace() -> Result<()> {
        // SAFETY: the operator supplies the matching sealed native library.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let programs = library.programs()?.with_manifest(std::path::Path::new(
            &std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
        let checkpoint = Checkpoint::open(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let plan = HeadPlan::prepare(&programs, &checkpoint, 1)?;
        let owner = TargetHeadOwner::load(device, &programs, &checkpoint, plan, u64::MAX, &[])?;
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
    fn final_planes_charge_both_bf16_boundaries_logits_and_head_workspace() {
        for rows in [1, 64, 640, 4096] {
            let planes = HeadPlan::plane_bytes(rows).unwrap();
            assert_eq!(planes, [rows * H * 2, rows * H * 2, rows * V * 4, 4 << 20]);
        }
        for rows in [0, 4097, usize::MAX] { assert!(HeadPlan::plane_bytes(rows).is_err()); }
    }
}
