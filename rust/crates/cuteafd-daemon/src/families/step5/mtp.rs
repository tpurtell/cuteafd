//! Predictor input boundary only; no scheduler or cache frontier is advanced here.
use super::weights::load_operand;
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer};
use cuteafd_loader::plan::{checkpoint::Checkpoint,
    families::step5::{Config, coordinator::{RankWeightLayout, WeightOperand}}};
use std::rc::Rc;

const H: usize = 4096;
const NORM_POINTERS: &[&str] = &["x", "weight", "bias", "cos", "sin", "positions", "out"];
const EH_POINTERS: &[&str] = &["x", "w", "out"];

fn planes(rows: usize) -> Result<[usize; 6]> {
    ensure!((1..=4096).contains(&rows), "Step-5 predictor input row capacity");
    // BF16 hidden, normalized embedding/hidden, concat, EH, F32 residual.
    Ok([2, 2, 2, 4, 2, 4].map(|element| rows * H * element))
}

fn validate_programs(norm: &ProgramSpec, eh: &ProgramSpec, rows: usize) -> Result<()> {
    planes(rows)?;
    let g = &norm.geometry;
    ensure!(norm.pointers == NORM_POINTERS && norm.scalar_kinds == b"if"
        && norm.capacity_rows as usize >= rows && norm.scratch.is_empty()
        && g["width"] == H && g["heads"] == 1 && g["zero_centered"] == true
        && g["subtract_mean"] == false && g["bias"] == false && g["rotary"] == 0
        && g["epsilon"] == "runtime_float32", "Step-5 predictor input norm ABI/geometry");
    let g = &eh.geometry;
    ensure!(eh.pointers == EH_POINTERS && eh.scalar_kinds == b"i"
        && eh.capacity_rows as usize >= rows && eh.scratch.is_empty()
        && g["width"] == 2 * H && g["outputs"] == H
        && g["output_dtype"] == "torch.bfloat16" && g["skinny_rows"] == 0,
        "Step-5 predictor EH ABI/geometry");
    Ok(())
}

pub(super) struct InputPlan {
    layout: RankWeightLayout,
    weights: Vec<WeightOperand>,
    rows: usize,
    epsilon: f32,
    bytes: u64,
}
impl InputPlan {
    pub(super) fn prepare(programs: &Programs<'_>, checkpoint: &Checkpoint,
        stage: usize, rows: usize) -> Result<Self> {
        let layout = RankWeightLayout::for_mtp_stage(checkpoint, stage)?;
        let cfg = Config::parse(checkpoint)?;
        let prefix = format!("model.layers.{}.", cfg.layers + stage);
        let weights = [("enorm.weight", DType::F32, vec![H]),
            ("hnorm.weight", DType::F32, vec![H]),
            ("eh_proj.weight", DType::Bf16, vec![H, 2 * H])].into_iter()
            .map(|(suffix, dtype, shape)| {
                let name = format!("{prefix}{suffix}");
                let weight = layout.operands.iter().find(|w| w.name == name)
                    .context("Step-5 predictor input operand missing")?;
                ensure!(weight.dtype == dtype && weight.shape == shape,
                    "Step-5 predictor input operand shape/dtype: {name}");
                Ok(weight.clone())
            }).collect::<Result<Vec<_>>>()?;
        validate_programs(programs.spec("step5_final_norm")?,
            programs.spec("step5_mtp_eh")?, rows)?;
        let bytes = weights.iter().map(|w| w.allocation_bytes)
            .chain(planes(rows)?.into_iter().map(|n| n as u64))
            .try_fold(0u64, u64::checked_add).context("Step-5 predictor input byte overflow")?;
        Ok(Self { layout, weights, rows, epsilon: cfg.norm_eps, bytes })
    }

    pub(super) fn bytes(&self) -> u64 { self.bytes }
}

struct InputWeights<'a> {
    norm: Program<'a>,
    eh: Program<'a>,
    weights: Vec<Allocation<'a>>,
    epsilon: f32,
}

pub(super) struct InputOwner<'a> {
    // This lane stream drains before any borrowed programs or storage retire.
    stream: Rc<Stream<'a>>,
    immutable: Rc<InputWeights<'a>>,
    planes: Vec<Allocation<'a>>,
    rows: usize,
}
impl<'a> InputOwner<'a> {
    pub(super) fn load(device: Device<'a>, programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint, plan: InputPlan, stream: Rc<Stream<'a>>,
        available: u64, reservations: &[u64]) -> Result<Self> {
        let bytes = reservations.iter().try_fold(plan.bytes, |n, v| n.checked_add(*v))
            .context("Step-5 predictor input reservation overflow")?;
        ensure!(bytes <= available && stream.device.id == device.id
            && std::ptr::eq(stream.device.library, device.library),
            "Step-5 predictor input admission/device mismatch");
        let immutable = device.run(|| {
            let norm = programs.program("step5_final_norm", NORM_POINTERS)?;
            let eh = programs.program("step5_mtp_eh", EH_POINTERS)?;
            let weights = plan.weights.iter()
                .map(|w| load_operand(device, checkpoint, &plan.layout, w))
                .collect::<Result<Vec<_>>>()?;
            Ok(Rc::new(InputWeights { norm, eh, weights, epsilon: plan.epsilon }))
        })?;
        Self::workspace(stream, immutable, plan.rows)
    }

    fn workspace(stream: Rc<Stream<'a>>, immutable: Rc<InputWeights<'a>>,
        rows: usize) -> Result<Self> {
        let planes = planes(rows)?.into_iter().map(|n| Allocation::new(stream.device, n))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stream, immutable, planes, rows })
    }

    pub(super) fn new_lane(&self, stream: Rc<Stream<'a>>) -> Result<Self> {
        ensure!(stream.device.id == self.stream.device.id
            && std::ptr::eq(stream.device.library, self.stream.device.library)
            && stream.raw != self.stream.raw, "Step-5 predictor input lane device/identity");
        Self::workspace(stream, self.immutable.clone(), self.rows)
    }

    /// Embedding BF16 and the previous predictor's own F32 residual are retained
    /// by the enclosing transaction through this stream's drain. Stage zero uses
    /// target hidden, not a different predictor's cache or normalized head input.
    pub(super) fn enqueue(&self, embedding: CuteafdDeviceBuffer,
        previous_hidden: CuteafdDeviceBuffer, rows: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!((1..=self.rows).contains(&rows), "Step-5 predictor input active rows");
        for (buffer, element) in [(embedding, 2), (previous_hidden, 4)] {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= rows * H * element
                && buffer.device_id == self.stream.device.id,
                "Step-5 predictor input buffer extent/device");
        }
        let mut guard = self.stream.cancellation_guard("Step-5 predictor input borrow");
        self.stream.device.run(|| {
            let library = self.stream.device.library;
            let p = |n: usize| self.planes[n].buffer.ptr;
            let w = |n: usize| self.immutable.weights[n].buffer.ptr;
            let scalars = [Scalar::I32(rows as i32), Scalar::F32(self.immutable.epsilon)];
            // SAFETY: checked external inputs and all private, disjoint planes
            // stay live until this stream drains. Norm's unused operands are ignored.
            unsafe {
                library.cuda_f32_to_bf16_async(previous_hidden, self.planes[0].buffer,
                    rows * H, self.stream.raw)?;
                self.immutable.norm.launch(&[embedding.ptr, w(0), w(0), w(0), w(0), w(0), p(1)],
                    &scalars, self.stream.raw)?;
                self.immutable.norm.launch(&[p(0), w(1), w(1), w(1), w(1), w(1), p(2)],
                    &scalars, self.stream.raw)?;
                library.glm_dflash_tap(p(1), p(3), rows, H, 2 * H, 0, self.stream.raw)?;
                library.glm_dflash_tap(p(2), p(3), rows, H, 2 * H, H, self.stream.raw)?;
                self.immutable.eh.launch(&[p(3), w(2), p(4)],
                    &[Scalar::I32(rows as i32)], self.stream.raw)?;
                library.cuda_zero_bytes_async(self.planes[5].buffer, rows * H * 4, self.stream.raw)?;
                library.cuda_accumulate_bf16_to_f32_async(self.planes[4].buffer,
                    self.planes[5].buffer, rows * H, self.stream.raw)?;
            }
            Ok(())
        })?;
        guard.complete = true;
        Ok(self.planes[5].buffer)
    }
}
impl Drop for InputOwner<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Step-5 predictor input owner"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn specs() -> (ProgramSpec, ProgramSpec) {
        let spec = |name: &str, pointers: &[&str], scalars: &[u8], geometry| ProgramSpec {
            name: name.into(), pointers: pointers.iter().map(|s| (*s).into()).collect(),
            scalar_kinds: scalars.to_vec(), scratch: Default::default(),
            capacity_rows: 4096, geometry,
        };
        (spec("step5_final_norm", NORM_POINTERS, b"if", serde_json::json!({
            "width": H, "heads": 1, "zero_centered": true, "subtract_mean": false,
            "bias": false, "rotary": 0, "epsilon": "runtime_float32"})),
        spec("step5_mtp_eh", EH_POINTERS, b"i", serde_json::json!({
            "width": 2 * H, "outputs": H, "output_dtype": "torch.bfloat16", "skinny_rows": 0})))
    }

    #[test]
    fn native_input_workspace_fits_the_conservative_forecast() -> Result<()> {
        for rows in [1, 64, 256, 4096] {
            let private = planes(rows)?.into_iter().sum::<usize>() as u64;
            assert_eq!(private, (rows * H * 16) as u64);
            let blocks = cuteafd_loader::plan::families::step5::native::swa_planes(rows, 64)?
                .into_iter().chain(cuteafd_loader::plan::families::step5::native::dense_planes(rows)?)
                .chain(cuteafd_loader::plan::families::step5::native::head_planes(rows)?)
                .sum::<usize>() as u64;
            assert!(private + blocks <=
                cuteafd_loader::plan::families::step5::native::mtp_forecast_workspace(rows, 1)?);
        }
        for rows in [0, 4097, usize::MAX] { assert!(planes(rows).is_err()); }
        Ok(())
    }

    #[test]
    fn native_input_programs_require_bf16_eh_and_whole_concatenation() -> Result<()> {
        let (norm, eh) = specs();
        validate_programs(&norm, &eh, 256)?;
        let mut bad = eh.clone();
        bad.geometry["width"] = H.into();
        assert!(validate_programs(&norm, &bad, 256).is_err());
        bad = eh.clone();
        bad.geometry["output_dtype"] = "torch.float32".into();
        assert!(validate_programs(&norm, &bad, 256).is_err());
        bad = eh.clone();
        bad.capacity_rows = 255;
        assert!(validate_programs(&norm, &bad, 256).is_err());
        bad = eh.clone();
        bad.scratch.insert("undeclared".into(), 256);
        assert!(validate_programs(&norm, &bad, 256).is_err());
        let mut bad_norm = norm.clone();
        bad_norm.geometry["zero_centered"] = false.into();
        assert!(validate_programs(&bad_norm, &eh, 256).is_err());
        bad_norm = norm.clone();
        bad_norm.scalar_kinds.pop();
        assert!(validate_programs(&bad_norm, &eh, 256).is_err());
        Ok(())
    }
}
