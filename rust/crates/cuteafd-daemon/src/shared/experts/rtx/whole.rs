//! Geometry-generic whole routed layers over checkpoint-native FP8-family packages.
use super::native::RankCompletion;
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::{experts::fp8::{self, Fp8Experts, Fp8Layer},
    memory::{device::{Allocation, Device, DeviceOwner, Event}, receipts::Inventory, Tag}};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_ffi::fp8_moe::Fp8MoeInfo;
use cuteafd_loader::{formats::fp8_experts::{Fp8ExpertTensors, Slicing},
    serving_capacity::{fp8moe_tp2_workspace, WeightArenaLayout}};
use std::{ffi::c_void, path::{Path, PathBuf}};

#[derive(Debug, Clone)]
pub(crate) struct WholePlan {
    pub directory: PathBuf,
    pub arena: WeightArenaLayout,
    pub scratch_bytes: usize,
    pub output_bytes: usize,
    pub workspace_bytes: usize,
}

fn validate_layers(layers: &[usize], shape: &cuteafd_loader::RoutedExpertShape) -> Result<()> {
    ensure!(!layers.is_empty() && layers.windows(2).all(|ids| ids[0] < ids[1])
        && layers.iter().all(|layer| (shape.first_layer..shape.layers).contains(layer)),
        "invalid whole routed expert backbone inventory {layers:?}");
    Ok(())
}

fn validate_info(info: &Fp8MoeInfo, tensors: &Fp8ExpertTensors, layers: &[usize],
    slicing: Slicing, max_rows: usize) -> Result<()> {
    validate_layers(layers, tensors.shape())?;
    ensure!(max_rows > 0 && !info.wire_input, "whole FP8-family experts require BF16 input and nonzero capacity");
    fp8::validate_checkpoint_contract(info, tensors, layers, 1, 0, slicing)?;
    ensure!(info.capacity_for(max_rows).is_some(), "whole expert package cannot hold {max_rows} rows");
    Ok(())
}

/// One serialized stream's scratch/output. Completion must drain before any
/// package, arena alias or writable output can retire.
pub(crate) struct Whole<'a> {
    completion: RankCompletion<Event<'a>>,
    device: Device<'a>,
    layers: Vec<usize>,
    max_rows: usize,
    workspace_bytes: usize,
    experts: DeviceOwner<'a, Fp8Experts<'a>>,
    output: Allocation<'a>,
}

impl<'a> Whole<'a> {
    /// Static package and canonical final backing only; no CUDA allocations.
    pub(crate) fn plan(device: i32, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: &[usize], max_rows: usize) -> Result<WholePlan> {
        validate_layers(layers, tensors.shape())?;
        let tensors = tensors.for_coordinator().for_layer_metadata(layers[0])?;
        let (directory, slicing) = fp8::exact_layout(directory, &tensors, 1, 0);
        let metadata = fp8::package_metadata(&directory, &tensors, layers, 1, 0, slicing)?;
        validate_info(&metadata.info, &tensors, layers, slicing, max_rows)?;
        let arena = crate::shared::experts::weight_layout::layout(device, "fp8",
            fp8::regions(&tensors, layers, 1, 0, slicing)?)?;
        let (scratch_bytes, output_bytes, workspace_bytes) = fp8moe_tp2_workspace(
            metadata.scratch_for(max_rows)?, tensors.shape().hidden, max_rows)?;
        Ok(WholePlan { directory, arena, scratch_bytes, output_bytes, workspace_bytes })
    }

    pub(crate) fn load(device: Device<'a>, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: Vec<usize>, max_rows: usize, budget: usize) -> Result<Self> {
        Self::bind(device, tensors, directory, layers, max_rows, budget, None)
    }

    pub(crate) fn from_pushed(device: Device<'a>, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: Vec<usize>, max_rows: usize, budget: usize, pushed: Vec<Fp8Layer<'a>>) -> Result<Self> {
        Self::bind(device, tensors, directory, layers, max_rows, budget, Some(pushed))
    }

    #[allow(clippy::too_many_arguments)]
    fn bind(device: Device<'a>, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: Vec<usize>, max_rows: usize, budget: usize, pushed: Option<Vec<Fp8Layer<'a>>>) -> Result<Self> {
        let plan = Self::plan(device.id, tensors, directory, &layers, max_rows)?;
        crate::shared::experts::weight_layout::admit(&plan.arena, 0, plan.workspace_bytes, budget)?;
        if let Some(pushed) = &pushed {
            ensure!(pushed.iter().map(|layer| layer.layer).eq(layers.iter().copied()),
                "whole pushed expert inventory differs from admission");
        }
        let tensors = tensors.for_coordinator().for_layer_metadata(layers[0])?;
        let completion = RankCompletion::new(Event::new(device)?);
        let experts = device.own(|| match pushed {
            Some(pushed) => Fp8Experts::from_pushed(device.library, &tensors, &plan.directory,
                &layers, 1, 0, max_rows, budget - plan.output_bytes, pushed),
            None => Fp8Experts::load(device.library, &tensors, &plan.directory,
                &layers, 1, 0, max_rows, budget - plan.output_bytes),
        })?;
        ensure!(experts.layers.iter().map(|layer| layer.layer).eq(layers.iter().copied())
            && experts.resident_bytes() == plan.arena.payload_bytes()
            && experts.scratch_bytes() == plan.scratch_bytes,
            "whole expert runtime owners disagree with admission");
        let output = Allocation::new(device, plan.output_bytes, Tag::workspace("experts/whole-output"))?;
        Ok(Self { completion, device, layers, max_rows, workspace_bytes: plan.workspace_bytes,
            experts, output })
    }
}

impl RtxExpertLayer for Whole<'_> {
    fn allocation_receipts(&self, inventory: &mut Inventory<'_>, name: &str) -> Result<()> {
        self.experts.allocation_receipts(inventory, name)?;
        self.output.inventory(inventory, &format!("{name}.output"))
    }
    fn shard(&self) -> RtxShard { RtxShard::Whole }
    fn partial(&self) -> PartialDtype { PartialDtype::Bf16 }
    fn uses_fp8_input(&self) -> bool { false }
    fn layers(&self) -> &[usize] { &self.layers }
    fn device(&self) -> i32 { self.device.id }
    fn workspace_bytes(&self) -> usize { self.workspace_bytes }
    fn output(&self) -> *mut c_void { self.output.buffer.ptr }
    fn group_rows(&self, rows: usize) -> Result<usize> { Ok(self.experts.weight_read_group(rows, true)?.0) }
    fn weight_read_bytes(&self) -> Result<usize> { Ok(self.experts.weight_read_bytes()) }

    unsafe fn prime(&mut self, input: ExpertInput, routes: Routes, stream: *mut c_void) -> Result<()> {
        let top = self.experts.module.info().capacity_for(self.max_rows).context("whole priming capacity")?;
        let capacities = self.experts.module.info().capacities.iter().copied()
            .filter(|capacity| *capacity <= top).collect::<Vec<_>>();
        for capacity in capacities {
            // SAFETY: caller initializes max_rows input/routes on a stream without peer waits.
            unsafe { self.enqueue(self.layers[0], capacity.min(self.max_rows), input, routes, stream, &mut || {})?; }
            self.device.run(|| {
                // SAFETY: startup stream is retained and has no pending peer waits.
                unsafe { self.device.library.cuda_stream_synchronize(stream) }
            })?;
        }
        Ok(())
    }

    unsafe fn enqueue(&mut self, layer: usize, rows: usize, input: ExpertInput, routes: Routes,
        stream: *mut c_void, abort: &mut dyn FnMut()) -> Result<()> {
        let result = (|| {
            ensure!((1..=self.max_rows).contains(&rows), "whole expert rows exceed admitted capacity");
            let index = super::resident_index(&self.layers, layer)?;
            let ExpertInput::Bf16(input) = input else { bail!("whole FP8-family experts require BF16 input") };
            self.device.run(|| {
                // SAFETY: caller retains rank-local input/routes and serializes scratch/output reuse.
                unsafe { self.experts.run_bf16(index, rows, input, routes.ids, routes.weights, self.output(), stream) }
            })
        })();
        self.completion.finish(result, stream, abort)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_ffi::fp8_moe::{Fp8MoeActivation, Fp8MoeProjection, Fp8MoeWeights};
    use cuteafd_loader::{formats::fp8_experts::ExpertFormat, plan::testing::{fp8, mxfp4, nvfp4, write_snapshot},
        RoutedExpertShape};

    #[test]
    fn whole_package_uses_catalog_geometry_and_checkpoint_native_arithmetic() -> Result<()> {
        for (format, hidden, intermediate, topk) in [
            (ExpertFormat::Fp8Block128, 4096, 2048, 8),
            (ExpertFormat::Mxfp4, 7168, 2048, 10),
            (ExpertFormat::Nvfp4, 6144, 3072, 4),
        ] {
            let root = tempfile::tempdir()?;
            let mut weights = Vec::new();
            for (projection, n, k) in [("gate", intermediate, hidden), ("up", intermediate, hidden),
                ("down", hidden, intermediate)] {
                let name = format!("model.layers.3.mlp.experts.0.{projection}_proj");
                weights.extend(match format {
                    ExpertFormat::Fp8Block128 => fp8(&name, n, k, None),
                    ExpertFormat::Mxfp4 => mxfp4(&name, n, k),
                    ExpertFormat::Nvfp4 => nvfp4(&name, n, k),
                    ExpertFormat::Bf16 => unreachable!(),
                });
            }
            write_snapshot(root.path(), &serde_json::json!({}), &weights, None);
            let tensors = Fp8ExpertTensors::read(root.path(), RoutedExpertShape { layers: 10, first_layer: 3,
                experts: 1, topk, hidden, intermediate, draft_stages: 0, draft_experts: 0 })?;
            let mut info = Fp8MoeInfo { hidden, intermediate, slice: intermediate, experts: 1, topk, tp: 1,
                wire_input: false, swiglu_limit: tensors.shape().geometry()?.swiglu_limit().unwrap_or(0.0),
                swiglu_alpha: 1.0, swiglu_beta: 0.0, activation: Fp8MoeActivation::SiluBf16Product,
                projection: Fp8MoeProjection::Bf16, capacities: vec![1, 16, 4096], weights: match format {
                    ExpertFormat::Fp8Block128 => Fp8MoeWeights::Fp8,
                    ExpertFormat::Mxfp4 => Fp8MoeWeights::Mxfp4,
                    ExpertFormat::Nvfp4 => Fp8MoeWeights::Nvfp4,
                    ExpertFormat::Bf16 => unreachable!(),
                } };
            validate_info(&info, &tensors, &[3], Slicing::Blocks(128), 4096)?;
            for layers in [vec![], vec![2], vec![3, 3], vec![4, 3], vec![10]] {
                assert!(validate_layers(&layers, tensors.shape()).is_err());
            }
            assert!(validate_info(&info, &tensors, &[3], Slicing::Blocks(128), 0).is_err());
            assert!(validate_info(&info, &tensors, &[3], Slicing::Blocks(128), 4097).is_err());
            info.wire_input = true;
            assert!(validate_info(&info, &tensors, &[3], Slicing::Blocks(128), 16).is_err());
            info.wire_input = false;
            info.tp = 2;
            assert!(validate_info(&info, &tensors, &[3], Slicing::Blocks(128), 16).is_err());
            info.tp = 1;
            info.topk += 1;
            assert!(validate_info(&info, &tensors, &[3], Slicing::Blocks(128), 16).is_err());
        }
        Ok(())
    }

    #[test]
    fn whole_workspace_has_only_named_package_scratch_and_one_bf16_output() -> Result<()> {
        for hidden in [2560, 6144, 7168] {
            for rows in [1, 3, 64, 4096] {
                for package in [0, 256, 163_779_584] {
                    let (scratch, output, total) = fp8moe_tp2_workspace(package, hidden, rows)?;
                    assert_eq!(scratch, package.max(256));
                    assert_eq!(output, rows * hidden * 2);
                    assert_eq!(total, scratch + output);
                }
            }
        }
        Ok(())
    }
}
