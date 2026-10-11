//! EXL3 TP2 rank execution: raw FP32 partials, without finish or re-quantization.
use super::native::{kernel_capacities, output_bytes, validate_layers, RankCompletion};
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::experts::{
    exl3::{
        execution::{Exl3Execution, Exl3InputFormat, Exl3Workspace},
        Exl3Weights,
    },
    layer::ExpertLayer,
};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::{expert_geometry, ExpertGeometry};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::{
    ffi::c_void,
    ops::Range,
    path::{Path, PathBuf},
    rc::Rc,
};

/// Exclusive to one rank's serialized stream; all capacities share one scratch arena.
pub(crate) struct Exl3Tp2<'a> {
    completion: RankCompletion<Event<'a>>,
    device: Device<'a>,
    rank: u8,
    layers: Range<usize>,
    geometry: ExpertGeometry,
    max_rows: usize,
    workspace_bytes: usize,
    executions: Option<DeviceOwner<'a, Vec<Exl3Execution<'a>>>>,
    weights: DeviceOwner<'a, Rc<Vec<Exl3Weights<'a>>>>,
    output: Option<Allocation<'a>>,
}

fn load_layer_pairs<T, W>(
    mut ranks: [T; 2],
    layers: Range<usize>,
    mut load: impl FnMut(usize) -> Result<[W; 2]>,
    mut publish: impl FnMut(&mut T, W, usize) -> Result<()>,
) -> Result<[T; 2]> {
    for layer in layers {
        let pair = load(layer)?;
        for (rank, weight) in pair.into_iter().enumerate() {
            publish(&mut ranks[rank], weight, rank)?;
        }
    }
    Ok(ranks)
}

fn directories(package: &Path, max_rows: usize) -> Result<Vec<PathBuf>> {
    Ok(kernel_capacities(max_rows)?
        .into_iter()
        .map(|c| package.join(format!("m{c}")))
        .collect())
}

impl<'a> Exl3Tp2<'a> {
    /// Preserve the native/V4.1 input quantization; missing manifest input_format
    /// means the package reconstructs BF16 from these FP8 K32 wire rows.
    pub(crate) const INPUT_FORMAT: Exl3InputFormat = Exl3InputFormat::Fp8K32;
    /// Shared max-sized scratch plus each unique capacity's private state/LUT
    /// and wire reconstruction, then one FP32 `[max_rows, hidden]` output.
    pub(crate) fn workspace_bytes_for(
        package: &Path,
        hidden: usize,
        max_rows: usize,
    ) -> Result<usize> {
        let paths = directories(package, max_rows)?;
        for path in &paths {
            let manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path.join("v41_exl3.json"))?)?;
            ensure!(
                manifest["hidden"].as_u64() == Some(hidden as u64)
                    && manifest["capacity"].as_u64()
                        == path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .and_then(|s| s.strip_prefix('m'))
                            .and_then(|s| s.parse().ok()),
                "EXL3 TP2 manifest geometry/capacity mismatch"
            );
        }
        Exl3Workspace::plan(&paths, Self::INPUT_FORMAT)?
            .checked_add(
                max_rows
                    .checked_mul(hidden)
                    .and_then(|n| n.checked_mul(4))
                    .context("EXL3 TP2 output overflow")?,
            )
            .context("EXL3 TP2 workspace overflow")
    }

    /// Loads each layer's two slices concurrently before advancing to the next
    /// layer. Published Rc/executor owners never leave the parent thread.
    pub(crate) fn load_pair(
        devices: [Device<'a>; 2],
        catalog: &cuteafd_loader::OfficialV41Catalog,
        package: &Path,
        layers: Range<usize>,
        max_rows: usize,
        budgets: [usize; 2],
    ) -> Result<[Self; 2]> {
        Self::load_pair_with_packages(devices, catalog, expert_geometry(), [package; 2],
            layers, max_rows, budgets)
    }

    /// Rank-specific exports support unequal complete-H128 partitions (Qwen 384/256).
    pub(crate) fn load_pair_with_packages(
        devices: [Device<'a>; 2],
        catalog: &cuteafd_loader::OfficialV41Catalog,
        geometry: ExpertGeometry,
        packages: [&Path; 2],
        layers: Range<usize>,
        max_rows: usize,
        budgets: [usize; 2],
    ) -> Result<[Self; 2]> {
        let label = format!("experts/TP2 expert layer halves {}..{}", layers.start, layers.end);
        let _memory_scope = cuteafd_ffi::memory_ledger::scope_owned(&label);
        validate_layers(&layers, 0)?;
        ensure!(geometry == expert_geometry(), "EXL3 TP2 geometry differs from native process geometry");
        ensure!(
            devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid EXL3 TP2 device pair"
        );
        let workspace_bytes = [
            Self::workspace_bytes_for(packages[0], geometry.hidden as usize, max_rows)?,
            Self::workspace_bytes_for(packages[1], geometry.hidden as usize, max_rows)?,
        ];
        let mut remaining = [0; 2];
        for rank in 0..2 {
            remaining[rank] = budgets[rank]
                .checked_sub(workspace_bytes[rank])
                .context("EXL3 TP2 workspace exceeds budget")?;
            let mut used = workspace_bytes[rank];
            for layer in layers.clone() {
                let plan = Exl3Weights::plan(catalog, ExpertLayer::BackboneTp2 { layer, rank })?;
                used = used
                    .checked_add(plan.resident_bytes)
                    .context("EXL3 TP2 resident overflow")?;
                ensure!(
                    used.checked_add(plan.device_staging_bytes)
                        .context("EXL3 TP2 peak overflow")?
                        <= budgets[rank],
                    "EXL3 TP2 rank {rank} weights + workspace + staging exceed budget"
                );
            }
        }
        let ranks = [
            Self {
                completion: RankCompletion::new(Event::new(devices[0])?),
                device: devices[0],
                rank: 0,
                layers: layers.clone(),
                geometry,
                max_rows,
                workspace_bytes: workspace_bytes[0],
                executions: None,
                weights: devices[0].own(|| Ok(Rc::new(Vec::with_capacity(layers.len()))))?,
                output: None,
            },
            Self {
                completion: RankCompletion::new(Event::new(devices[1])?),
                device: devices[1],
                rank: 1,
                layers: layers.clone(),
                geometry,
                max_rows,
                workspace_bytes: workspace_bytes[1],
                executions: None,
                weights: devices[1].own(|| Ok(Rc::new(Vec::with_capacity(layers.len()))))?,
                output: None,
            },
        ];
        let mut ranks = if std::env::var("CUTEAFD_TP2_SHARED_READ").as_deref() == Ok("0") {
            let remaining = std::cell::RefCell::new(remaining);
            load_layer_pairs(ranks, layers.clone(), |layer| {
                let available = *remaining.borrow();
                // SAFETY: fresh GPU owners only; load drains all uploads and
                // DeviceOwner drops on the owning GPU, including partial errors.
                unsafe { cuteafd_ffi::synchronized_load::load_pair(|rank| devices[rank].own(|| {
                    let _memory_scope = cuteafd_ffi::memory_ledger::scope_owned(&label);
                    Ok(vec![Exl3Weights::load(devices[rank].library, catalog,
                        ExpertLayer::BackboneTp2 { layer, rank }, available[rank])?])
                })) }
            }, |owner, mut loaded, rank| {
                remaining.borrow_mut()[rank] -= loaded[0].budget.resident_bytes;
                devices[rank].run(|| {
                    Rc::get_mut(owner.weights.get_mut()).expect("unpublished EXL3 rank")
                        .append(loaded.get_mut());
                    Ok(())
                })
            })?
        } else {
            let pair = Exl3Weights::load_tp2_pair(devices, catalog, layers.clone(), remaining)?;
            let mut ranks = ranks;
            for (rank, mut loaded) in pair.into_iter().enumerate() {
                devices[rank].run(|| {
                    Rc::get_mut(ranks[rank].weights.get_mut())
                        .expect("unpublished EXL3 rank").append(loaded.get_mut());
                    Ok(())
                })?;
            }
            ranks
        };
        for (index, rank) in ranks.iter_mut().enumerate() {
            let paths = directories(packages[index], max_rows)?;
            rank.executions = Some(rank.device.own(|| {
                let arena = Exl3Workspace::new(rank.device.library, &paths)?;
                paths
                    .iter()
                    .zip(kernel_capacities(max_rows)?)
                    .map(|(path, capacity)| {
                        // SAFETY: trusted pinned packages; capacities serialize on the rank stream.
                        let execution = unsafe {
                            Exl3Execution::with_shared_workspace(
                                rank.device.library,
                                rank.weights.get().clone(),
                                path,
                                Self::INPUT_FORMAT,
                                Some(arena.clone()),
                            )?
                        };
                        ensure!(
                            execution.capacity() == capacity as usize
                                && execution.output_element_bytes() == 4,
                            "EXL3 TP2 requires matching capacity and FP32 token sums"
                        );
                        Ok(execution)
                    })
                    .collect::<Result<Vec<_>>>()
            })?);
            rank.output = Some(Allocation::new(
                rank.device,
                output_bytes(geometry, max_rows)?,
            )?);
        }
        Ok(ranks)
    }
}

impl RtxExpertLayer for Exl3Tp2<'_> {
    fn shard(&self) -> RtxShard {
        RtxShard::Tp2 { rank: self.rank }
    }
    fn partial(&self) -> PartialDtype {
        PartialDtype::F32
    }
    fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }
    fn device(&self) -> i32 {
        self.device.id
    }
    fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    fn output(&self) -> *mut c_void {
        self.output
            .as_ref()
            .expect("initialized EXL3 rank")
            .buffer
            .ptr
    }
    unsafe fn enqueue(
        &mut self,
        layer: usize,
        rows: usize,
        input: ExpertInput,
        routes: Routes,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            self.layers.contains(&layer) && (1..=self.max_rows).contains(&rows),
            "EXL3 TP2 layer/rows not resident"
        );
        let ExpertInput::Fp8K32(wire) = input else {
            bail!("Exl3Tp2 requires FP8 K32 input")
        };
        ensure!(
            !wire.is_null() && !routes.ids.is_null() && !routes.weights.is_null(),
            "null EXL3 TP2 input"
        );
        let buffer = |ptr, width| CuteafdDeviceBuffer {
            ptr,
            bytes: rows * width,
            device_id: self.device.id,
            flags: 0,
        };
        let inputs = [
            buffer(wire, self.geometry.hidden as usize * 33 / 32),
            buffer(routes.ids, self.geometry.topk as usize * 4),
            buffer(routes.weights, self.geometry.topk as usize * 4),
        ];
        let result = self.device.run(|| {
            let execution = self
                .executions
                .as_mut()
                .expect("initialized EXL3 rank")
                .iter_mut()
                .find(|e| e.capacity() >= rows)
                .unwrap();
            // SAFETY: caller owns rank-local inputs and exclusive stream/arena/output use through completion.
            unsafe {
                execution.launch_layer_into(
                    layer - self.layers.start,
                    inputs,
                    rows,
                    stream,
                    self.output
                        .as_ref()
                        .expect("initialized EXL3 output")
                        .buffer,
                )
            }
        });
        self.completion.finish(result.map(|_| ()), stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank1_mid_load_failure_releases_both_rank_owners() {
        use std::{
            cell::{Cell, RefCell},
            rc::Rc,
        };
        struct ProbeAllocation {
            rank: usize,
            live: Rc<Cell<usize>>,
            released: Rc<RefCell<Vec<usize>>>,
        }
        impl Drop for ProbeAllocation {
            fn drop(&mut self) {
                self.live.set(self.live.get() - 1);
                self.released.borrow_mut().push(self.rank);
            }
        }
        let live = Rc::new(Cell::new(0));
        let released = Rc::new(RefCell::new(Vec::new()));
        let mut order = Vec::new();
        let ranks = [Vec::new(), Vec::new()];
        let result = load_layer_pairs(ranks, 0..3, |layer| {
            let pair = std::array::from_fn(|rank| {
                order.push((layer, rank));
                live.set(live.get() + 1);
                ProbeAllocation { rank, live: live.clone(), released: released.clone() }
            });
            ensure!(layer != 1, "injected rank1 mid-load failure");
            Ok(pair)
        }, |owner, weight, _| { owner.push(weight); Ok(()) });
        assert!(result.is_err());
        assert_eq!(order, [(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(live.get(), 0);
        let mut freed = released.borrow().clone();
        freed.sort();
        assert_eq!(freed, [0, 0, 1, 1]);
    }

    #[test]
    fn workspace_counts_shared_max_and_private_capacity_state_once() -> Result<()> {
        let root = tempfile::tempdir()?;
        for (capacity, scratch) in [(1, 64), (16, 1024)] {
            let directory = root.path().join(format!("m{capacity}"));
            std::fs::create_dir_all(&directory)?;
            std::fs::write(
                directory.join("v41_exl3.json"),
                serde_json::to_vec(&serde_json::json!({
                    "capacity": capacity, "hidden": 8, "trellis_lut": {"bytes": 32}, "buffers": {
                        "scratch": {"allocation": "scratch", "dtype": "f32", "bytes": scratch, "zero_on_create": false},
                        "alias": {"allocation": "scratch", "dtype": "f32", "bytes": 99999, "zero_on_create": false},
                        "state": {"allocation": "state", "dtype": "i32", "bytes": 0, "zero_on_create": true}
                    }
                }))?,
            )?;
        }
        let private = 2 * (32 + 16);
        let wire_decode = (1 + 16) * 8 * 2;
        assert_eq!(
            Exl3Tp2::workspace_bytes_for(root.path(), 8, 16)?,
            1024 + private + wire_decode + 16 * 8 * 4
        );
        assert_eq!(
            Exl3Tp2::workspace_bytes_for(root.path(), 8, 3)?,
            1024 + private + wire_decode + 3 * 8 * 4
        );
        assert!(Exl3Tp2::workspace_bytes_for(root.path(), 16, 16).is_err());
        Ok(())
    }

    #[test]
    fn capacity_paths_are_unique_at_exact_boundaries() {
        for rows in [1, 16, 80, 256, 1024, 4096] {
            let paths = directories(Path::new("rtx-tp2"), rows).unwrap();
            assert_eq!(
                paths
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                paths.len()
            );
            assert_eq!(
                paths.last().unwrap(),
                &Path::new("rtx-tp2").join(format!("m{rows}"))
            );
        }
    }
}
