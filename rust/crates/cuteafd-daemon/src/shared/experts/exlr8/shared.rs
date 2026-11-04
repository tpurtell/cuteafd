//! Native shared-expert residency and bounded lane execution for EXLR8 families.
use super::{Exlr8Execution, Exlr8Weights, view};
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, exlr8::KernelFamily, programs::Programs};
use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};
use std::ffi::c_void;

const ROWS: usize = cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS;

pub(crate) struct NativePlan {
    plans: Vec<Residency>,
    family: KernelFamily,
    pub(crate) resident: u64,
    pub(crate) workspace: usize,
    host_peak: usize,
}

impl NativePlan {
    #[cfg(test)]
    pub(crate) fn allocation_fixture(family: KernelFamily, plans: Vec<Residency>, workspace: usize) -> Self {
        Self { family, resident: plans.iter().map(|plan| plan.resident_bytes as u64).sum(),
            plans, workspace, host_peak: 0 }
    }

    pub(crate) fn new(
        snapshot: &std::path::Path,
        manifest: &str,
        first: usize,
        layers: usize,
        programs: &Programs<'_>,
        family: KernelFamily,
        world: usize,
        rank: usize,
    ) -> Result<Self> {
        let plans = (first..layers)
            .map(|layer| {
                Residency::plan(snapshot, manifest, layer, world, rank, ExpertSet::Shared)
                    .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(!plans.is_empty(), "EXLR8 requires a routed shared layer");
        let budgets = plans
            .iter()
            .map(Exlr8Weights::plan)
            .collect::<Result<Vec<_>>>()?;
        let sum = |bytes: Vec<usize>| {
            bytes.into_iter().try_fold(0usize, |total, bytes| {
                total
                    .checked_add(bytes)
                    .context("EXLR8 coordinator admission overflow")
            })
        };
        let resident = u64::try_from(sum(plans.iter().map(|plan| plan.resident_bytes).collect())?)?;
        let retained_host = sum(plans.iter().map(Residency::host_bytes).collect())?;
        let pinned = budgets
            .iter()
            .map(|budget| budget.pinned_host_bytes)
            .max()
            .unwrap_or(0);
        let scratch = budgets
            .iter()
            .zip(&plans)
            .map(|(budget, plan)| budget.read_scratch_bytes - plan.host_bytes())
            .max()
            .unwrap_or(0);
        let host_peak = retained_host
            .checked_add(pinned)
            .and_then(|n| n.checked_add(scratch))
            .context("EXLR8 coordinator host peak overflow")?;
        let workspace = Exlr8Execution::shared_workspace_bytes(programs, family, ROWS)?;
        Ok(Self {
            plans,
            family,
            resident,
            workspace,
            host_peak,
        })
    }

    pub(crate) fn admit_host(&self) -> Result<()> {
        let (_, available) = cuteafd_loader::formats::mapped_preload::host_memory()?;
        let available = usize::try_from(available)?;
        ensure!(
            self.host_peak <= available,
            "EXLR8 coordinator host peak {} exceeds available {available}",
            self.host_peak
        );
        Ok(())
    }
}

pub(crate) struct NativeExperts<'a> {
    family: KernelFamily,
    workspace: usize,
    weights: Vec<Exlr8Weights<'a>>,
    first: usize,
}

impl<'a> NativeExperts<'a> {
    pub(crate) fn load(library: &'a NativeLibrary, plan: NativePlan) -> Result<Self> {
        plan.admit_host()?;
        let NativePlan {
            plans,
            family,
            workspace,
            ..
        } = plan;
        let first = plans
            .first()
            .context("EXLR8 shared residency is empty")?
            .layer;
        ensure!(
            plans
                .iter()
                .enumerate()
                .all(|(i, plan)| plan.layer == first + i),
            "EXLR8 shared residency must cover consecutive routed layers"
        );
        let weights = plans
            .into_iter()
            .map(|plan| {
                let bytes = plan.resident_bytes;
                Exlr8Weights::load(library, plan, bytes)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            family,
            workspace,
            weights,
            first,
        })
    }

    pub(crate) fn execution(
        &self,
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
    ) -> Result<Exlr8Execution<'a>> {
        let mut execution =
            Exlr8Execution::new_shared(library, programs, self.family, ROWS, self.workspace)?;
        library.copy_h2d(execution.ids()?, &vec![0; ROWS * 4])?;
        let route_weights: Vec<u8> = (0..ROWS).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        library.copy_h2d(execution.route_weights()?, &route_weights)?;
        execution.seal_graph_storage()?;
        Ok(execution)
    }

    pub(crate) fn layers(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.weights.iter().map(|weights| weights.layout.layer)
    }

    /// Retained shared weights only; hook executions and loading history compose separately.
    pub(crate) fn weight_receipts(
        &self,
        prefix: &str,
        devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>,
        lifetime: cuteafd_loader::placement::planner::Lifetime,
    ) -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        let first = self.weights.first().context("EXLR8 shared census is empty")?;
        let physical = first.buffer.device_id;
        let rank = devices.logical_rank(physical)?;
        let world = first.layout.world;
        ensure!((1..=2).contains(&world) && devices.physical_device((world - 1) as u8).is_ok()
            && devices.physical_device(world as u8).is_err(),
            "EXLR8 shared census differs from coordinator world");
        let geometry = self.family.geometry();
        for (index, weights) in self.weights.iter().enumerate() {
            let layout = &weights.layout;
            ensure!(self.first.checked_add(index) == Some(layout.layer)
                && layout.set == ExpertSet::Shared && layout.world == world && layout.rank == usize::from(rank)
                && layout.hidden == geometry.hidden && layout.units == geometry.units && layout.experts == 1
                && weights.buffer.device_id == physical,
                "EXLR8 shared census inventory, geometry or logical rank differs from retained owners");
        }
        Exlr8Weights::weight_receipts(&self.weights, prefix, devices.location(physical)?, physical, lifetime)
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.weights
            .iter()
            .map(|weights| weights.budget.resident_bytes)
            .sum()
    }

    fn weights(&self, layer: usize) -> Result<&Exlr8Weights<'a>> {
        self.weights
            .get(layer.checked_sub(self.first).context("EXLR8 dense layer")?)
            .context("EXLR8 shared layer is not resident")
    }

    pub(crate) fn shared(
        &self,
        library: &NativeLibrary,
        layer: usize,
        execution: &Exlr8Execution<'_>,
        input: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let weights = self.weights(layer)?;
        let stride = weights.layout.hidden * 2;
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let trace = *TRACE
            .get_or_init(|| std::env::var("CUTEAFD_EXLR8_TILE_TRACE").is_ok_and(|v| v == "1"));
        let issued = |first, operation, returned| {
            if trace {
                eprintln!(
                    "EXLR8 shared enqueue rank={} layer={layer} rows={rows} first={first} stream={stream:p} operation={operation} returned={returned}",
                    weights.layout.rank
                );
            }
        };
        for first in (0..rows).step_by(ROWS) {
            let count = (rows - first).min(ROWS);
            let input = view(input, first * stride, count * stride)?;
            let output = view(output, first * stride, count * stride)?;
            // SAFETY: one stream orders copies, K1 and copies out before the next
            // chunk reuses the bounded arena; no host synchronization is needed.
            unsafe {
                issued(first, "input_copy", false);
                library.copy_d2d_async(execution.input()?, input, input.bytes, stream)?;
                issued(first, "input_copy", true);
                issued(first, "execute", false);
                execution.run(weights, count, stream)?;
                issued(first, "execute", true);
                issued(first, "output_copy", false);
                library.copy_d2d_async(output, execution.output()?, output.bytes, stream)?;
                issued(first, "output_copy", true);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod owner_tests {
    use super::*;
    use cuteafd_ffi::{memory_ledger, native_library_lifetime_fixture::Fixture};
    use cuteafd_loader::{placement::{families::glm5::hooks::HookBuffers, planner::{self, *}},
        plan::exlr8::{residency::WeightLayout, shared_workspace_layout}};
    use std::collections::BTreeMap;

    pub(crate) fn shared_residency(family: KernelFamily, layer: usize, world: usize, rank: usize) -> Residency {
        let geometry = family.geometry();
        let mut bytes = 0usize;
        let mut reserve = |extent: usize| {
            let offset = bytes;
            bytes = (bytes + extent).next_multiple_of(256);
            offset
        };
        let signs_offsets = [reserve(geometry.hidden), reserve(geometry.hidden)];
        let pointers_offset = reserve(geometry.units * 3 * 3 * 8);
        let bits_offset = reserve(geometry.units * 3 * 4);
        Residency { layer, world, rank, hidden: geometry.hidden, experts: 1, units: geometry.units,
            set: ExpertSet::Shared, set_id: "allocation-only".into(), manifest: "allocation-only".into(),
            loads: vec![], signs: [vec![1; geometry.hidden], vec![255; geometry.hidden]],
            signs_offsets, pointers_offset, bits_offset, compact_offsets: [0; 2],
            resident_bytes: bytes, payload_bytes: 0, staging_bytes: 0 }
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn retained_shared_experts_and_hooks_match_independent_contract() -> Result<()> {
        let fixture = Fixture::build()?;
        for family in [KernelFamily::Glm, KernelFamily::GlmFlash] {
            let library = fixture.load()?;
            fixture.quiet_events(&library, true)?;
            fixture.immediate_copies(&library)?;
            fixture.exlr8_programs(&library, family == KernelFamily::GlmFlash)?;
            let manifest = super::super::owner_manifest(family);
            let path = fixture.directory().join("PROGRAMS.json");
            std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
            let programs = library.programs()?.with_manifest(&path)?;
            let workspace = shared_workspace_layout(&manifest, family)?;
            for physical in [1, 0] { fixture.configure_memory(&library, physical, 128 << 20)?; }
            for physical in [vec![1], vec![1, 0]] {
                let devices = CoordinatorDevices::new(&physical)?;
                for (rank, &device) in physical.iter().enumerate() {
                    library.cuda_set_device(device)?;
                    let tag = format!("shared/{family:?}/{}/{rank}", physical.len());
                    let _owner = memory_ledger::owner(&tag);
                    let layouts = (3..5).map(|layer| shared_residency(family, layer, physical.len(), rank))
                        .collect::<Vec<_>>();
                    let mut native = NativeExperts::load(&library,
                        NativePlan::allocation_fixture(family, layouts.clone(), workspace.bytes))?;
                    let executions = [native.execution(&library, &programs)?, native.execution(&library, &programs)?];
                    fixture.finish_ordered_uploads(&library)?;
                    assert!(library.release_sync_h2d_staging()? > 0);
                    let location = Location::coordinator(rank as u8);
                    let mut allocations = WeightLayout::from_residencies("test.shared", &layouts.iter().collect::<Vec<_>>(),
                        WeightArenaTopology::PerLayer, location, Lifetime::RESIDENT)?.allocations;
                    let hooks = HookBuffers { device: rank as u8, maximum_pages: 1, native: None, core: None,
                        shared_workspace: Some(workspace.clone()) };
                    let mut contract = Contract::fixed(64, 1, 1, Facts { resident_weights: vec![],phase_order: cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT, allocations: vec![], deferred: vec![],
                        sequential: vec![], reservations: vec![], allowances: vec![] });
                    contract.variables = vec![Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]),
                        Variable::Lanes(vec![1])];
                    contract.components = hooks.components()?;
                    allocations.extend(contract.facts(&[0, 0], 1)?.allocations);
                    let facts = Facts { resident_weights: vec![],phase_order: cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT, allocations, deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![] };
                    let budgets = vec![Budget { id: "CPU shared fixture".into(), locations: vec![location],
                        physical_bytes: 128 << 20, logical_bytes: 128 << 20, headroom: 0,
                        baseline: Baseline::Forecast { occupied: 0, source: "CPU fixture".into(), assumed: false } }];
                    let phases = planner::evaluate(&facts, &budgets)?;
                    let plan = ResolvedPlan { request: Request { context: 64, pool: Pin::Exact(64), onboard: Pin::Auto,
                            prefill: Pin::Auto, spark_reply_format: None, operation: Operation::Serving , resident_weight_limit: None,},
                        decisions: Decisions { spark_reply_format: None, attention: vec![], onboard_layers: vec![], expert_homes: Default::default(),
                            embedding: None, prefill: PrefillShape { rows: 64, waves: 1 }, lanes: 1 },
                        pool_tokens: 64, facts, budgets, phases };
                    let mut receipts = native.weight_receipts("test.shared", devices, Lifetime::RESIDENT)?;
                    for (kind, execution) in ["decode", "prefill"].into_iter().zip(&executions) {
                        receipts.extend(execution.workspace_receipts(&format!("glm.target.gpu.{rank}.{kind}.native.shared"),
                            location, device, Lifetime { first: Phase::Warmup, last: Phase::Growth })?);
                    }
                    planner::reconcile(&plan, Phase::Ready, &receipts)?;
                    assert!(native.weight_receipts("", devices, Lifetime::RESIDENT).is_err());
                    assert!(native.weight_receipts("test.shared", CoordinatorDevices::new(&[7])?, Lifetime::RESIDENT).is_err());
                    if physical.len() == 2 {
                        assert!(native.weight_receipts("test.shared", CoordinatorDevices::new(&[0, 1])?, Lifetime::RESIDENT).is_err());
                    }
                    for defect in 0..8 {
                        let layout = native.weights[0].layout.clone();
                        let original_first = native.first;
                        let original_device = native.weights[0].buffer.device_id;
                        match defect {
                            0 => native.first += 1,
                            1 => native.weights[0].layout.layer += 1,
                            2 => native.weights[0].layout.rank += 1,
                            3 => native.weights[0].layout.world = 3,
                            4 => native.weights[0].layout.set = ExpertSet::Routed,
                            5 => native.weights[0].layout.hidden += 1,
                            6 => native.weights[0].buffer.device_id = 7,
                            7 => native.weights[0].layout.experts = 2,
                            _ => unreachable!(),
                        }
                        assert!(native.weight_receipts("test.shared", devices, Lifetime::RESIDENT).is_err(), "defect {defect}");
                        native.first = original_first;
                        native.weights[0].buffer.device_id = original_device;
                        native.weights[0].layout = layout;
                    }
                    let expected = receipts.iter().filter(|r| r.allocation.view.is_none())
                        .map(|r| (r.address as usize, r.allocation.bytes as usize)).collect::<BTreeMap<_, _>>();
                    let actual = memory_ledger::allocations().into_iter().filter(|r| r.key.owner == Some(tag.as_str()))
                        .map(|r| (r.address, r.bytes)).collect::<BTreeMap<_, _>>();
                    assert_eq!(actual, expected, "every shared weight/hook allocation must appear once");
                    drop(executions);
                    drop(native);
                    assert!(memory_ledger::allocations().iter().all(|r| r.key.owner != Some(tag.as_str())));
                    assert_eq!(library.cuda_memory_info()?, (128 << 20, 128 << 20));
                }
            }
        }
        Ok(())
    }
}
