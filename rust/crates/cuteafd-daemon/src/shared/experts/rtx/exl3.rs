//! EXL3 TP2 rank execution: raw FP32 partials, without finish or re-quantization.
use super::native::{output_bytes, validate_layers, RankCompletion};
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxSelectionPolicy, RtxShard};
use crate::shared::experts::{
    exl3::{
        execution::{
            Exl3Modules, Exl3Execution, Exl3InputFormat, Exl3ReductionPlan, Exl3RowPolicy, Exl3Workspace,
        },
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
    row_policy: Exl3RowPolicy,
    selection: RtxSelectionPolicy,
    reduction_plan: Option<Exl3ReductionPlan>,
    package: PathBuf,
    max_rows: usize,
    workspace_bytes: usize,
    executions: Option<DeviceOwner<'a, Vec<Exl3Execution<'a>>>>,
    weights: Rc<DeviceOwner<'a, Rc<Vec<Exl3Weights<'a>>>>>,
    output: Option<Allocation<'a>>,
}

#[derive(Clone)]
pub(crate) struct Exl3Tp2Weights<'a> {
    device: Device<'a>,
    rank: u8,
    layers: Range<usize>,
    geometry: ExpertGeometry,
    row_policy: Exl3RowPolicy,
    selection: RtxSelectionPolicy,
    package: PathBuf,
    max_rows: usize,
    workspace_bytes: usize,
    weights: Rc<DeviceOwner<'a, Rc<Vec<Exl3Weights<'a>>>>>,
}

impl<'a> Exl3Tp2Weights<'a> {
    pub(crate) fn device(&self) -> Device<'a> {
        self.device
    }
    pub(crate) fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }
    pub(crate) fn max_rows(&self) -> usize {
        self.max_rows
    }
    pub(crate) fn execution(&self) -> Result<Exl3Tp2<'a>> {
        self.execution_with_modules(None)
    }
    pub(crate) fn execution_with_modules(&self, modules: Option<Vec<Exl3Modules<'a>>>) -> Result<Exl3Tp2<'a>> {
        let reduction_plan = match self.selection {
            RtxSelectionPolicy::FirstCapacity => None,
            RtxSelectionPolicy::CanonicalDecode => Some(Exl3ReductionPlan::new(&directories(
                &self.package,
                self.max_rows,
                self.row_policy,
            )?)?),
        };
        let mut lane = Exl3Tp2 {
            completion: RankCompletion::new(Event::new(self.device)?),
            device: self.device,
            rank: self.rank,
            layers: self.layers.clone(),
            geometry: self.geometry,
            row_policy: self.row_policy,
            selection: self.selection,
            reduction_plan,
            package: self.package.clone(),
            max_rows: self.max_rows,
            workspace_bytes: self.workspace_bytes,
            executions: None,
            weights: self.weights.clone(),
            output: None,
        };
        lane.initialize_lane(modules)?;
        Ok(lane)
    }
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

fn directories(package: &Path, max_rows: usize, row_policy: Exl3RowPolicy) -> Result<Vec<PathBuf>> {
    selected_directories(
        package,
        max_rows,
        row_policy,
        RtxSelectionPolicy::CanonicalDecode,
    )
}

fn selected_capacities(
    max_rows: usize,
    row_policy: Exl3RowPolicy,
    selection: RtxSelectionPolicy,
) -> Result<Vec<u32>> {
    if selection == RtxSelectionPolicy::FirstCapacity {
        super::native::kernel_capacities(row_policy.required_capacity(max_rows))
    } else {
        Ok(row_policy.capacities(max_rows)?)
    }
}

fn selected_directories(
    package: &Path,
    max_rows: usize,
    row_policy: Exl3RowPolicy,
    selection: RtxSelectionPolicy,
) -> Result<Vec<PathBuf>> {
    Ok(selected_capacities(max_rows, row_policy, selection)?
        .into_iter()
        .map(|capacity| package.join(format!("m{capacity}")))
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
        Self::workspace_bytes_for_policy(package, hidden, max_rows, Exl3RowPolicy::active())
    }

    fn workspace_bytes_for_policy(
        package: &Path,
        hidden: usize,
        max_rows: usize,
        row_policy: Exl3RowPolicy,
    ) -> Result<usize> {
        Self::workspace_bytes_for_selection(
            package,
            hidden,
            max_rows,
            row_policy,
            RtxSelectionPolicy::CanonicalDecode,
        )
    }

    pub(crate) fn workspace_bytes_for_selection(
        package: &Path,
        hidden: usize,
        max_rows: usize,
        row_policy: Exl3RowPolicy,
        selection: RtxSelectionPolicy,
    ) -> Result<usize> {
        let paths = selected_directories(package, max_rows, row_policy, selection)?;
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
        if selection == RtxSelectionPolicy::CanonicalDecode {
            Exl3ReductionPlan::new(&paths)?;
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
        Self::load_pair_with_packages(
            devices,
            catalog,
            expert_geometry(),
            [package; 2],
            layers,
            max_rows,
            budgets,
        )
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
        Self::load_pair_with_selection(
            devices,
            catalog,
            geometry,
            packages,
            layers,
            max_rows,
            budgets,
            RtxSelectionPolicy::CanonicalDecode,
        )
    }

    pub(crate) fn load_pair_with_selection(
        devices: [Device<'a>; 2],
        catalog: &cuteafd_loader::OfficialV41Catalog,
        geometry: ExpertGeometry,
        packages: [&Path; 2],
        layers: Range<usize>,
        max_rows: usize,
        budgets: [usize; 2],
        selection: RtxSelectionPolicy,
    ) -> Result<[Self; 2]> {
        let owners = Self::load_pair_weights(
            devices, catalog, geometry, packages, layers, max_rows, budgets, selection,
        )?;
        Ok([owners[0].execution()?, owners[1].execution()?])
    }

    pub(crate) fn load_pair_weights(
        devices: [Device<'a>; 2],
        catalog: &cuteafd_loader::OfficialV41Catalog,
        geometry: ExpertGeometry,
        packages: [&Path; 2],
        layers: Range<usize>,
        max_rows: usize,
        budgets: [usize; 2],
        selection: RtxSelectionPolicy,
    ) -> Result<[Exl3Tp2Weights<'a>; 2]> {
        let started = std::time::Instant::now();
        let label = format!(
            "experts/TP2 expert layer halves {}..{}",
            layers.start, layers.end
        );
        let _memory_scope = cuteafd_ffi::memory_ledger::scope_owned(&label);
        validate_layers(&layers, 0)?;
        ensure!(
            geometry == expert_geometry(),
            "EXL3 TP2 geometry differs from native process geometry"
        );
        let row_policy = Exl3RowPolicy::for_geometry(geometry);
        ensure!(
            devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid EXL3 TP2 device pair"
        );
        let workspace_bytes = [
            Self::workspace_bytes_for_selection(
                packages[0],
                geometry.hidden as usize,
                max_rows,
                row_policy,
                selection,
            )?,
            Self::workspace_bytes_for_selection(
                packages[1],
                geometry.hidden as usize,
                max_rows,
                row_policy,
                selection,
            )?,
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
        let unloaded = [
            devices[0].own(|| Ok(Rc::new(Vec::with_capacity(layers.len()))))?,
            devices[1].own(|| Ok(Rc::new(Vec::with_capacity(layers.len()))))?,
        ];
        let [left, right] = unloaded;
        let make = |rank: usize, weights| Exl3Tp2Weights {
            device: devices[rank],
            rank: rank as u8,
            layers: layers.clone(),
            geometry,
            row_policy,
            selection,
            package: packages[rank].to_path_buf(),
            max_rows,
            workspace_bytes: workspace_bytes[rank],
            weights: Rc::new(weights),
        };
        let ranks = [make(0, left), make(1, right)];
        let ranks = if std::env::var("CUTEAFD_TP2_SHARED_READ").as_deref() == Ok("0") {
            let remaining = std::cell::RefCell::new(remaining);
            load_layer_pairs(
                ranks,
                layers.clone(),
                |layer| {
                    let available = *remaining.borrow();
                    // SAFETY: fresh GPU owners only; load drains all uploads and
                    // DeviceOwner drops on the owning GPU, including partial errors.
                    unsafe {
                        cuteafd_ffi::synchronized_load::load_pair(|rank| {
                            devices[rank].own(|| {
                                let _memory_scope = cuteafd_ffi::memory_ledger::scope_owned(&label);
                                Ok(vec![Exl3Weights::load(
                                    devices[rank].library,
                                    catalog,
                                    ExpertLayer::BackboneTp2 { layer, rank },
                                    available[rank],
                                )?])
                            })
                        })
                    }
                },
                |owner, mut loaded, rank| {
                    remaining.borrow_mut()[rank] -= loaded[0].budget.resident_bytes;
                    devices[rank].run(|| {
                        Rc::get_mut(
                            Rc::get_mut(&mut owner.weights)
                                .expect("unpublished device owner")
                                .get_mut(),
                        )
                        .expect("unpublished EXL3 rank")
                        .append(loaded.get_mut());
                        Ok(())
                    })
                },
            )?
        } else {
            let pair = Exl3Weights::load_tp2_pair(devices, catalog, layers.clone(), remaining)?;
            let mut ranks = ranks;
            for (rank, mut loaded) in pair.into_iter().enumerate() {
                devices[rank].run(|| {
                    Rc::get_mut(
                        Rc::get_mut(&mut ranks[rank].weights)
                            .expect("unpublished device owner")
                            .get_mut(),
                    )
                    .expect("unpublished EXL3 rank")
                    .append(loaded.get_mut());
                    Ok(())
                })?;
            }
            ranks
        };
        for rank in &ranks {
            tracing::info!(
                rank = rank.rank,
                device = rank.device.id,
                layers = rank.layers.len(),
                resident_bytes = rank
                    .weights
                    .iter()
                    .map(|weight| weight.budget.resident_bytes)
                    .sum::<usize>(),
                workspace_bytes = rank.workspace_bytes,
                elapsed_seconds = started.elapsed().as_secs_f64(),
                "EXL3 TP2 rank load complete"
            );
        }
        Ok(ranks)
    }

    fn initialize_lane(&mut self, modules: Option<Vec<Exl3Modules<'a>>>) -> Result<()> {
        let paths = selected_directories(
            &self.package,
            self.max_rows,
            self.row_policy,
            self.selection,
        )?;
        self.executions = Some(self.device.own(|| {
            let arena = Exl3Workspace::new(self.device.library, &paths)?;
            let modules = match modules {
                Some(modules) => modules,
                None => paths.iter().map(|path| {
                    // SAFETY: these are the same trusted pinned packages used for binding.
                    unsafe { Exl3Modules::load(self.device.library, path, Self::INPUT_FORMAT) }
                }).collect::<Result<Vec<_>>>()?,
            };
            ensure!(modules.len() == paths.len(), "EXL3 TP2 preload capacity count differs");
            modules.into_iter()
                .zip(selected_capacities(
                    self.max_rows,
                    self.row_policy,
                    self.selection,
                )?)
                .map(|(modules, capacity)| {
                    // SAFETY: trusted pinned packages; capacities serialize on the rank stream.
                    let execution = unsafe {
                        Exl3Execution::with_modules(
                            self.device.library,
                            self.weights.get().clone(),
                            modules,
                            Some(arena.clone()),
                        )?
                    };
                    crate::shared::inventory::startup_probe(
                        self.device.library,
                        &[self.device.id],
                        &format!("exl3-package-m{capacity}"),
                    )?;
                    ensure!(
                        execution.capacity() == capacity as usize
                            && execution.output_element_bytes() == 4,
                        "EXL3 TP2 requires matching capacity and FP32 token sums"
                    );
                    Ok(execution)
                })
                .collect::<Result<Vec<_>>>()
        })?);
        self.output = Some(Allocation::new(
            self.device,
            output_bytes(self.geometry, self.max_rows)?,
        )?);
        Ok(())
    }

    pub(crate) fn immutable_weights(&self) -> Exl3Tp2Weights<'a> {
        Exl3Tp2Weights {
            device: self.device,
            rank: self.rank,
            layers: self.layers.clone(),
            geometry: self.geometry,
            row_policy: self.row_policy,
            selection: self.selection,
            package: self.package.clone(),
            max_rows: self.max_rows,
            workspace_bytes: self.workspace_bytes,
            weights: self.weights.clone(),
        }
    }

    pub(crate) fn fork_lane(&self) -> Result<Self> {
        self.immutable_weights().execution()
    }

    pub(super) fn allocation(&self) -> &Allocation<'a> {
        self.output.as_ref().expect("initialized EXL3 output")
    }
    pub(super) fn execution_device(&self) -> Device<'a> {
        self.device
    }
    pub(super) fn bytes_per_expert(&self, layer: usize) -> Result<usize> {
        ensure!(
            self.layers.contains(&layer),
            "EXL3 TP2 weight metadata layer unavailable"
        );
        let weight = &self.weights[layer - self.layers.start];
        Ok(weight.budget.resident_bytes / weight.layout.experts)
    }

    fn execution_index(&self, rows: usize) -> Result<usize> {
        let required = self.row_policy.required_capacity(rows);
        let executions = self
            .executions
            .as_ref()
            .context("uninitialized EXL3 rank")?;
        self.selection.execution_index(
            rows,
            required,
            executions.iter().map(|execution| execution.capacity()),
            || {
                self.reduction_plan
                    .as_ref()
                    .context("missing canonical EXL3 plan")?
                    .execution_index(
                        rows,
                        required,
                        executions.iter().map(|execution| execution.capacity()),
                    )
            },
        )
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.weights
            .iter()
            .map(|weight| weight.budget.resident_bytes)
            .sum()
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
    fn group_rows(&self, rows: usize) -> Result<usize> {
        ensure!(rows <= self.max_rows, "invalid EXL3 TP2 grouping row count");
        let executions = self
            .executions
            .as_ref()
            .context("uninitialized EXL3 rank")?;
        let selected = if rows == 0 {
            executions.last()
        } else {
            executions
                .iter()
                .find(|e| e.capacity() >= self.row_policy.required_capacity(rows))
        };
        Ok(selected
            .context("no EXL3 TP2 capacity for this many rows")?
            .group_rows())
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
        abort: &mut dyn FnMut(),
    ) -> Result<()> {
        let result = (|| {
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
            let selected = self.execution_index(rows)?;
            self.device.run(|| {
                let executions = self.executions.as_mut().expect("initialized EXL3 rank");
                let execution = &mut executions[selected];
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
            })
        })();
        self.completion.finish(result.map(|_| ()), stream, abort)
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
        let result = load_layer_pairs(
            ranks,
            0..3,
            |layer| {
                let pair = std::array::from_fn(|rank| {
                    order.push((layer, rank));
                    live.set(live.get() + 1);
                    ProbeAllocation {
                        rank,
                        live: live.clone(),
                        released: released.clone(),
                    }
                });
                ensure!(layer != 1, "injected rank1 mid-load failure");
                Ok(pair)
            },
            |owner, weight, _| {
                owner.push(weight);
                Ok(())
            },
        );
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
                    "capacity": capacity, "hidden": 8, "tile": [64, 128, 64, 128], "trellis_lut": {"bytes": 32}, "buffers": {
                        "scratch": {"allocation": "scratch", "dtype": "f32", "bytes": scratch, "zero_on_create": false},
                        "alias": {"allocation": "scratch", "dtype": "f32", "bytes": 99999, "zero_on_create": false},
                        "state": {"allocation": "state", "dtype": "i32", "bytes": 0, "zero_on_create": true}
                    }
                }))?,
            )?;
        }
        assert!(
            Exl3Tp2::workspace_bytes_for_policy(root.path(), 8, 16, Exl3RowPolicy::Nearest)
                .is_err()
        );
        assert!(
            Exl3Tp2::workspace_bytes_for_policy(root.path(), 8, 3, Exl3RowPolicy::Nearest).is_err()
        );
        assert!(Exl3Tp2::workspace_bytes_for_policy(
            root.path(),
            8,
            16,
            Exl3RowPolicy::GlmFlashK64
        )
        .is_err());
        let directory = root.path().join("m80");
        std::fs::create_dir_all(&directory)?;
        std::fs::write(
            directory.join("v41_exl3.json"),
            serde_json::to_vec(&serde_json::json!({
                "capacity": 80, "hidden": 8, "tile": [64, 128, 64, 128], "trellis_lut": {"bytes": 32}, "buffers": {
                    "scratch": {"allocation": "scratch", "dtype": "f32", "bytes": 2048, "zero_on_create": false},
                    "state": {"allocation": "state", "dtype": "i32", "bytes": 0, "zero_on_create": true}
                }
            }))?,
        )?;
        for rows in [2, 3, 8, 16] {
            assert!(Exl3Tp2::workspace_bytes_for_policy(
                root.path(),
                16,
                rows,
                Exl3RowPolicy::Nearest
            )
            .is_err());
            assert_eq!(
                Exl3Tp2::workspace_bytes_for_policy(root.path(), 8, rows, Exl3RowPolicy::Nearest)?,
                2048 + 3 * (32 + 16) + (1 + 16 + 80) * 8 * 2 + rows * 8 * 4
            );
            assert_eq!(
                Exl3Tp2::workspace_bytes_for_policy(
                    root.path(),
                    8,
                    rows,
                    Exl3RowPolicy::GlmFlashK64
                )?,
                2048 + 3 * (32 + 16) + (1 + 16 + 80) * 8 * 2 + rows * 8 * 4
            );
        }
        Ok(())
    }

    #[test]
    fn shared_policy_selects_preloaded_tp2_capacities() -> Result<()> {
        for policy in [Exl3RowPolicy::Nearest, Exl3RowPolicy::GlmFlashK64] {
            for maximum in [1, 2, 8, 16, 17, 80, 256, 4096] {
                let paths = directories(Path::new("rtx-tp2"), maximum, policy)?;
                let capacities = policy.capacities(maximum)?;
                assert_eq!(paths.len(), capacities.len());
                for live in 1..=maximum {
                    let selected = capacities
                        .iter()
                        .find(|&&c| c as usize >= policy.required_capacity(live))
                        .unwrap();
                    if policy == Exl3RowPolicy::GlmFlashK64 && (2..=16).contains(&live) {
                        assert_eq!(*selected, 80);
                    }
                    assert!(paths.contains(&Path::new("rtx-tp2").join(format!("m{selected}"))));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn capacity_paths_are_unique_at_exact_boundaries() {
        for rows in [1, 16, 80, 256, 1024, 4096] {
            let paths = directories(Path::new("rtx-tp2"), rows, Exl3RowPolicy::Nearest).unwrap();
            assert_eq!(
                paths
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                paths.len()
            );
            assert_eq!(
                paths.last().unwrap(),
                &Path::new("rtx-tp2").join(format!("m{}", if rows == 16 { 80 } else { rows }))
            );
        }
    }
}
