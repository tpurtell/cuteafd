//! Geometry-generic FP8/MXFP4/NVFP4 RTX TP2, with BF16 input and rank partials.
use super::native::RankCompletion;
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::experts::fp8::{exact_layout, Fp8Experts, Fp8Layer};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_ffi::fp8_moe::{Fp8MoeInfo, Fp8MoeMetadata, Fp8MoeWeights};
use cuteafd_loader::formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Slicing};
use std::{ffi::c_void, ops::Range, path::{Path, PathBuf}};

#[derive(Debug, Clone)]
pub(crate) struct Fp8MoeTp2Plan {
    pub directory: PathBuf,
    pub slicing: Slicing,
    pub resident_layer_bytes: usize,
    /// Uploads use host staging only; no temporary device allocation.
    pub staging_bytes: usize,
    pub scratch_bytes: usize,
    pub output_bytes: usize,
    pub workspace_bytes: usize,
}

fn validate_info(info: &Fp8MoeInfo, tensors: &Fp8ExpertTensors, rank: usize, slicing: Slicing,
    max_rows: usize) -> Result<()> {
    ensure!(rank < 2 && max_rows > 0, "invalid FP8 MoE TP2 rank/capacity");
    let shape = tensors.shape();
    let weights_match = matches!((tensors.format(), info.weights),
        (ExpertFormat::Fp8Block128, Fp8MoeWeights::Fp8)
        | (ExpertFormat::Mxfp4, Fp8MoeWeights::Mxfp4)
        | (ExpertFormat::Nvfp4, Fp8MoeWeights::Nvfp4 { .. }));
    ensure!(!info.wire_input && info.tp == 2 && info.hidden == shape.hidden
        && info.intermediate == shape.intermediate && info.experts == shape.experts
        && info.topk == shape.topk && info.slice == tensors.rank_width(2, rank, slicing)? && weights_match,
        "FP8 MoE TP2 package does not serve this checkpoint with BF16 input: {info:?}");
    ensure!(info.capacity_for(max_rows).is_some(), "FP8 MoE TP2 package has no capacity for {max_rows} rows");
    Ok(())
}

fn workspace(scratch: usize, hidden: usize, rows: usize) -> Result<(usize, usize, usize)> {
    ensure!(rows > 0 && hidden > 0, "invalid FP8 MoE TP2 output geometry");
    let output = rows.checked_mul(hidden).and_then(|n| n.checked_mul(2))
        .context("FP8 MoE TP2 output size overflow")?;
    let scratch = scratch.max(256);
    let total = scratch.checked_add(output).context("FP8 MoE TP2 workspace size overflow")?;
    Ok((scratch, output, total))
}

/// One serialized rank stream shares the package scratch and BF16 output.
/// Graph owners must be destroyed before this owner, as with NativeTp2.
pub(crate) struct Fp8MoeTp2<'a> {
    completion: RankCompletion<Event<'a>>,
    device: Device<'a>,
    rank: u8,
    layers: Range<usize>,
    max_rows: usize,
    workspace_bytes: usize,
    experts: DeviceOwner<'a, Fp8Experts<'a>>,
    output: Allocation<'a>,
}

impl<'a> Fp8MoeTp2<'a> {
    /// Static package ABI only: no CUDA initialization or device allocation.
    /// The directory is the padded tp2 layout; exact_layout selects its sibling.
    pub(crate) fn plan(tensors: &Fp8ExpertTensors, directory: &Path, rank: usize,
        max_rows: usize) -> Result<Fp8MoeTp2Plan> {
        ensure!(rank < 2, "invalid FP8 MoE TP2 rank");
        let (directory, slicing) = exact_layout(directory, tensors, 2, rank);
        // SAFETY: the engine selects a trusted, generated package; the read
        // calls only its static geometry and scratch ABI, not CUDA creation.
        let metadata = unsafe { Fp8MoeMetadata::read(&directory) }?;
        validate_info(&metadata.info, tensors, rank, slicing, max_rows)?;
        let resident_layer_bytes = Fp8Layer::bytes_for(tensors, 2, rank, slicing)?;
        let (scratch_bytes, output_bytes, workspace_bytes) =
            workspace(metadata.scratch_for(max_rows)?, tensors.shape().hidden, max_rows)?;
        Ok(Fp8MoeTp2Plan { directory, slicing, resident_layer_bytes, staging_bytes: 0,
            scratch_bytes, output_bytes, workspace_bytes })
    }

    pub(crate) fn load(device: Device<'a>, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: Range<usize>, rank: usize, max_rows: usize, budget: usize) -> Result<Self> {
        ensure!(rank < 2 && !layers.is_empty() && layers.start >= tensors.shape().first_layer
            && layers.end <= tensors.shape().layers, "invalid FP8 MoE TP2 rank/layer range");
        let plan = Self::plan(tensors, directory, rank, max_rows)?;
        let resident = plan.resident_layer_bytes.checked_mul(layers.len())
            .context("FP8 MoE TP2 resident size overflow")?;
        ensure!(resident.checked_add(plan.workspace_bytes).context("FP8 MoE TP2 allocation size overflow")? <= budget,
            "FP8 MoE TP2 rank {rank} needs {resident} resident + {} workspace bytes; budget {budget}",
            plan.workspace_bytes);
        let _scope = cuteafd_ffi::memory_ledger::scope("experts/TP2 FP8 expert layer halves");
        let completion = RankCompletion::new(Event::new(device)?);
        let experts = device.own(|| Fp8Experts::load(device.library, tensors, &plan.directory, layers.clone(),
            2, rank, max_rows, budget - plan.output_bytes))?;
        ensure!(experts.resident_bytes() == resident && experts.scratch_bytes() == plan.scratch_bytes,
            "FP8 MoE TP2 static allocation plan disagrees with runtime");
        let output = Allocation::new(device, plan.output_bytes)?;
        Ok(Self { completion, device, rank: rank as u8, layers, max_rows,
            workspace_bytes: plan.workspace_bytes, experts, output })
    }

    /// Both halves are admitted before either allocates. Unlike Native/EXL3,
    /// Fp8ExpertTensors has no shared-read projection bank yet; each rank loads
    /// only its own slice through the existing bounded readers.
    pub(crate) fn load_pair(devices: [Device<'a>; 2], tensors: &Fp8ExpertTensors, directory: &Path,
        layers: Range<usize>, max_rows: usize, budgets: [usize; 2]) -> Result<[Self; 2]> {
        ensure!(devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid FP8 MoE TP2 device pair");
        for rank in 0..2 {
            let plan = Self::plan(tensors, directory, rank, max_rows)?;
            let required = plan.resident_layer_bytes.checked_mul(layers.len())
                .and_then(|n| n.checked_add(plan.workspace_bytes)).context("FP8 MoE TP2 allocation overflow")?;
            ensure!(required <= budgets[rank], "FP8 MoE TP2 rank {rank} needs {required} bytes; budget {}", budgets[rank]);
        }
        Ok([
            Self::load(devices[0], tensors, directory, layers.clone(), 0, max_rows, budgets[0])?,
            Self::load(devices[1], tensors, directory, layers, 1, max_rows, budgets[1])?,
        ])
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.experts.resident_bytes()
    }
}

impl RtxExpertLayer for Fp8MoeTp2<'_> {
    fn shard(&self) -> RtxShard { RtxShard::Tp2 { rank: self.rank } }
    fn partial(&self) -> PartialDtype { PartialDtype::Bf16 }
    fn layers(&self) -> Range<usize> { self.layers.clone() }
    fn device(&self) -> i32 { self.device.id }
    fn workspace_bytes(&self) -> usize { self.workspace_bytes }
    fn output(&self) -> *mut c_void { self.output.buffer.ptr }

    unsafe fn enqueue(&mut self, layer: usize, rows: usize, input: ExpertInput, routes: Routes,
        stream: *mut c_void) -> Result<()> {
        ensure!(self.layers.contains(&layer) && (1..=self.max_rows).contains(&rows),
            "FP8 MoE TP2 layer/row range exceeded");
        let ExpertInput::Bf16(input) = input else { bail!("FP8 MoE TP2 requires BF16 input rows") };
        let index = self.experts.index_of(layer)?;
        let result = self.device.run(|| {
            // SAFETY: caller retains rank-local input, routes and stream through
            // completion, and serializes reuse of this owner's scratch/output.
            unsafe { self.experts.run(index, rows, input, routes.ids, routes.weights, self.output(), stream) }
        });
        self.completion.finish(result, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_contract_is_geometry_generic_and_rejects_wire_or_wrong_rank() -> Result<()> {
        use cuteafd_loader::plan::testing::{fp8, mxfp4, nvfp4, write_snapshot};
        use cuteafd_loader::RoutedExpertShape;
        for (format, hidden, intermediate, topk) in [
            (ExpertFormat::Fp8Block128, 4096, 2048, 8),
            (ExpertFormat::Mxfp4, 4096, 1280, 8),
            (ExpertFormat::Mxfp4, 7168, 2048, 10),
            (ExpertFormat::Nvfp4, 2560, 640, 8),
        ] {
            let root = tempfile::tempdir()?;
            let mut weights = Vec::new();
            for (projection, n, k) in [("gate", intermediate, hidden), ("up", intermediate, hidden),
                ("down", hidden, intermediate)] {
                let name = format!("model.layers.0.mlp.experts.0.{projection}_proj");
                weights.extend(match format {
                    ExpertFormat::Fp8Block128 => fp8(&name, n, k, None),
                    ExpertFormat::Mxfp4 => mxfp4(&name, n, k),
                    ExpertFormat::Nvfp4 => nvfp4(&name, n, k),
                });
            }
            write_snapshot(root.path(), &serde_json::json!({}), &weights, None);
            let tensors = Fp8ExpertTensors::read(root.path(), RoutedExpertShape { layers: 1, first_layer: 0,
                experts: 1, topk, hidden, intermediate, draft_stages: 0, draft_experts: 0 })?;
            for rank in 0..2 {
                let mut info = Fp8MoeInfo { hidden, intermediate, slice: tensors.rank_width(2, rank, Slicing::Blocks(128))?,
                    experts: 1, topk, tp: 2, wire_input: false, swiglu_limit: 0.0, capacities: vec![1, 16, 4096],
                    weights: match format { ExpertFormat::Fp8Block128 => Fp8MoeWeights::Fp8,
                        ExpertFormat::Mxfp4 => Fp8MoeWeights::Mxfp4,
                        ExpertFormat::Nvfp4 => Fp8MoeWeights::Nvfp4 { w4a4: true } } };
                validate_info(&info, &tensors, rank, Slicing::Blocks(128), 4096)?;
                assert!(validate_info(&info, &tensors, 2, Slicing::Blocks(128), 16).is_err());
                assert!(validate_info(&info, &tensors, rank, Slicing::Blocks(128), 4097).is_err());
                info.wire_input = true;
                assert!(validate_info(&info, &tensors, rank, Slicing::Blocks(128), 16).is_err());
                info.wire_input = false;
                info.tp = 1;
                assert!(validate_info(&info, &tensors, rank, Slicing::Blocks(128), 16).is_err());
                info.tp = 2;
                info.slice += 128;
                assert!(validate_info(&info, &tensors, rank, Slicing::Blocks(128), 16).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn workspace_includes_one_bf16_partial_and_package_scratch() {
        assert_eq!(workspace(1024, 4096, 16).unwrap(), (1024, 131072, 132096));
        assert_eq!(workspace(0, 7168, 1).unwrap(), (256, 14336, 14592));
        assert!(workspace(usize::MAX, 4096, 16).is_err());
        assert!(workspace(1, usize::MAX, 16).is_err());
        assert!(workspace(1, 4096, 0).is_err());
    }
}
