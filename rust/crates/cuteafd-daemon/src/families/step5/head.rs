//! Admitted final BF16 narrowing, zero-centered norm and shared vocabulary head.
use super::weights::load_operand;
use crate::shared::{
    memory::device::{Allocation, Device, Stream},
    target_head::{self, TargetHead},
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{
    programs::{Program, Programs, Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE},
    CuteafdDeviceBuffer,
};
use cuteafd_loader::plan::{
    checkpoint::Checkpoint,
    families::step5::coordinator::{CoordinatorContract, RankWeightLayout, WeightOperand},
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
}

impl HeadPlan {
    fn plane_bytes(rows: usize) -> Result<[usize; 4]> {
        ensure!((1..=4096).contains(&rows), "Step-5 head row capacity must be 1..4096");
        Ok([rows * H * 2, rows * H * 2, rows * V * 4, VOCABULARY_HEAD_WORKSPACE])
    }

    pub fn prepare(programs: &Programs<'_>, checkpoint: &Checkpoint, rows: usize) -> Result<Self> {
        Self::prepare_layout(programs, checkpoint, rows, RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?)
    }

    pub fn for_execution(programs: &Programs<'_>, checkpoint: &Checkpoint, rows: usize,
        ownership: &cuteafd_loader::plan::families::step5::ownership::ExecutionOwnership) -> Result<Self> {
        Self::prepare_layout(programs, checkpoint, rows, RankWeightLayout::for_execution(checkpoint,
            ownership.execution().spec().head_gpu as usize, ownership)?)
    }

    fn prepare_layout(programs: &Programs<'_>, checkpoint: &Checkpoint, rows: usize, layout: RankWeightLayout) -> Result<Self> {
        let planes = Self::plane_bytes(rows)?;
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let weights = [("model.norm.weight", DType::F32, vec![H]),
            ("lm_head.weight", DType::Bf16, vec![V, H])]
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
        Ok(Self { layout, weights, planes, rows, epsilon: contract.norms.final_norm, bytes })
    }

    pub fn bytes(&self) -> u64 { self.bytes }
}

pub(super) struct TargetHeadOwner<'a> {
    // Drain before programs, library handles, weights and workspace retire.
    stream: Stream<'a>,
    head: VocabularyHead<'a>,
    norm: Program<'a>,
    weights: Vec<Allocation<'a>>,
    planes: Vec<Allocation<'a>>,
    rows: usize,
    epsilon: f32,
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
            let planes = plan.planes.iter().map(|&n| Allocation::new(device, n))
                .collect::<Result<Vec<_>>>()?;
            // SAFETY: admitted workspace is retained after the head handle, on this device.
            let head = unsafe { device.library.vocabulary_head_rows(planes[3].buffer.ptr,
                H as u32, plan.rows as u32, V as u32)? };
            let stream = Stream::new(device)?;
            Ok(Self { stream, head, norm, weights, planes, rows: plan.rows, epsilon: plan.epsilon })
        })
    }

    /// Inspect the already-drained norm plane without introducing another target launch.
    pub fn normalized(&self) -> Result<Vec<f32>> {
        let mut bytes = vec![0; self.rows * H * 2];
        self.stream.device.run(|| self.stream.device.library.copy_d2h(&mut bytes, self.planes[1].buffer))?;
        Ok(bytes.chunks_exact(2).map(|b|
            f32::from_bits(u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16)).collect())
    }

    /// Caller drains the target chain first; this method drains before releasing its input borrow.
    pub fn logits(&self, input: CuteafdDeviceBuffer) -> Result<Vec<f32>> {
        ensure!(input.bytes >= self.rows * H * 4, "Step-5 final hidden extent");
        let mut guard = self.stream.cancellation_guard("Step-5 final head input borrow");
        self.stream.device.run(|| {
            let library = self.stream.device.library;
            let p = self.planes.iter().map(|p| p.buffer.ptr).collect::<Vec<_>>();
            let weight = self.weights[0].buffer.ptr;
            // SAFETY: all extents were admitted; disjoint owners and input remain live until drain.
            unsafe {
                library.cuda_f32_to_bf16_async(input, self.planes[0].buffer,
                    self.rows * H, self.stream.raw)?;
                self.norm.launch(&[p[0], weight, weight, weight, weight, weight, p[1]],
                    &[Scalar::I32(self.rows as i32), Scalar::F32(self.epsilon)], self.stream.raw)?;
                target_head::launch(TargetHead::configured(), &self.head, p[1],
                    self.weights[1].buffer.ptr, p[2].cast(), self.rows, self.stream.raw)?;
            }
            self.stream.drain()?;
            guard.complete = true;
            let mut bytes = vec![0; self.rows * V * 4];
            library.copy_d2h(&mut bytes, self.planes[2].buffer)?;
            let values = bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
            ensure!(values.iter().all(|v| v.is_finite()), "nonfinite Step-5 logits");
            Ok(values)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_planes_charge_both_bf16_boundaries_logits_and_head_workspace() {
        for rows in [1, 64, 640, 4096] {
            let planes = HeadPlan::plane_bytes(rows).unwrap();
            assert_eq!(planes, [rows * H * 2, rows * H * 2, rows * V * 4, 4 << 20]);
        }
        for rows in [0, 4097, usize::MAX] { assert!(HeadPlan::plane_bytes(rows).is_err()); }
    }
}
