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

    pub(crate) fn residency(&self) -> &[Residency] { &self.plans }

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
    pub(crate) fn allocation_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        for weight in &self.weights {
            weight.allocation_receipts(inventory, &format!("{name}.layer.{}", weight.layout.layer))?;
        }
        Ok(())
    }

    pub(crate) fn load(library: &'a NativeLibrary, plan: NativePlan) -> Result<Self> {
        plan.admit_host()?;
        Self::load_inner(library, plan)
    }

    pub(crate) fn load_admitted(library: &'a NativeLibrary, plan: NativePlan,
        admission: &cuteafd_loader::placement::families::glm5::admission::NativeLoad) -> Result<Self> {
        Self::load_validated(library, plan, |plans| admission.validate_shared(plans))
    }

    /// The caller consumes its pre-solve receipt; this path never samples capacity.
    /// Validation completes before any weights are allocated or published.
    pub(crate) fn load_validated(library: &'a NativeLibrary, plan: NativePlan,
        validate: impl FnOnce(&[Residency]) -> Result<()>) -> Result<Self> {
        validate(plan.residency())?;
        Self::load_inner(library, plan)
    }

    fn load_inner(library: &'a NativeLibrary, plan: NativePlan) -> Result<Self> {
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
        let layout = cuteafd_loader::plan::exlr8::residency::shared_arena(&plans)?;
        let device = crate::shared::memory::device::Device { library, id: library.cuda_get_device()? };
        let arena = crate::shared::memory::weight_arena::DeviceWeightArena::new(device, layout,
            cuteafd_core::memory_layout::ArenaTag::weights("glm/shared"))?;
        let weights = plans
            .into_iter()
            .map(|plan| {
                let bytes = plan.resident_bytes;
                let name = Exlr8Weights::shared_request(&plan)?.id;
                let allocation = crate::shared::memory::DeviceAllocation::from_allocation(arena.view(&name, bytes)?);
                Exlr8Weights::load_into(library, plan, bytes, allocation)
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
mod receipt_tests {
    use super::*;

    #[test]
    fn shared_layers_retain_one_rank_backing_and_rebase_descriptors() -> Result<()> {
        use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};
        use cuteafd_loader::placement::planner::{ComponentKind, CoordinatorDevices, Lifetime};
        use cuteafd_core::memory_layout::Category;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::exlr8::fixture(root.path(), false);
        for world in [1, 2] {
            for rank in 0..world {
                library.cuda_set_device(rank as i32)?;
                let first = Residency::plan(root.path(), "mixed", 3, world, rank, ExpertSet::Shared)?;
                let mut second = first.clone();
                second.layer += 1;
                let plans = vec![first, second];
                let layout = cuteafd_loader::plan::exlr8::residency::shared_arena(&plans)?;
                let before = fixture.events()?.matches('D').count();
                let shared = NativeExperts::load_validated(&library, NativePlan {
                    plans: plans.clone(), family: KernelFamily::Glm, resident: layout.payload_bytes() as u64,
                    workspace: 0, host_peak: 0,
                }, |_| Ok(()))?;
                assert_eq!(fixture.events()?.matches('D').count() - before, 1);
                assert_eq!(shared.resident_bytes(), layout.payload_bytes());
                let mut inventory = crate::shared::memory::receipts::Inventory::new(
                    CoordinatorDevices::new(&[0, 1])?, "shared", Lifetime::RESIDENT,
                    Category::Weights, "shared weights");
                shared.allocation_receipts(&mut inventory, "weights")?;
                let roots = inventory.receipts.iter().filter(|r| r.allocation.view.is_none()).collect::<Vec<_>>();
                assert_eq!(roots.len(), 1);
                assert_eq!((roots[0].allocation.id.as_str(), roots[0].allocation.bytes),
                    (layout.id(), layout.bytes() as u64));
                for weights in &shared.weights {
                    for unit in &weights.layout.loads {
                        for (source, &offset) in unit.source.iter().zip(&unit.destination) {
                            let mut expected = vec![0; source.bytes];
                            use std::os::unix::fs::FileExt;
                            std::fs::File::open(&source.path)?.read_exact_at(&mut expected, source.offset)?;
                            let mut actual = vec![0; source.bytes];
                            library.copy_d2h(&mut actual, view(weights.buffer, offset, source.bytes)?)?;
                            assert_eq!(actual, expected);
                        }
                    }
                    let (pointers, _) = weights.layout.descriptors(weights.buffer.ptr as u64)?;
                    let mut actual = vec![0; pointers.len() * 8];
                    library.copy_d2h(&mut actual, weights.pointers()?)?;
                    assert_eq!(actual, pointers.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<_>>());
                }
                let mut weights = cuteafd_loader::placement::families::glm5::weights::WeightInventory {
                    ranks: world, owners: vec![], core_loading: Default::default(), kv_b_loading: Default::default(),
                    row_loading: Default::default(), attention_mount: Default::default(),
                };
                let all = (0..world).map(|r| {
                    let mut rank_plans = plans.clone();
                    for plan in &mut rank_plans { plan.rank = r; }
                    rank_plans
                }).collect::<Vec<_>>();
                weights.extend_shared(&all)?;
                let components = weights.components(1)?;
                let tails = components.iter().filter_map(|component| match &component.kind {
                    ComponentKind::AllocationGranularity(tail) if tail.location
                        == cuteafd_loader::placement::planner::Location::coordinator(rank as u8) => Some(tail),
                    _ => None,
                }).collect::<Vec<_>>();
                assert_eq!(tails.len(), 1);
                assert_eq!(tails[0].backings.len(), 1);
                assert_eq!(tails[0].backings[0].requests, [layout.bytes() as u64]);
            }
        }
        Ok(())
    }

    #[test]
    fn recorded_shared_arena_payload_matches_mount_through_committed_ring() -> Result<()> {
        use crate::shared::experts::rtx::{dense, push};
        use crate::shared::memory::device::Device;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::exlr8::fixture(root.path(), false);
        let sockets = tempfile::tempdir()?;
        for world in [1, 2] {
            for rank in 0..world {
                let device = Device { library: &library, id: rank as i32 };
                device.run(|| -> Result<()> {
                    let first = Residency::plan(root.path(), "mixed", 3, world, rank, ExpertSet::Shared)?;
                    let mut second = first.clone();
                    second.layer += 1;
                    let plans = vec![first, second];
                    let plan = || NativePlan { plans: plans.clone(), family: KernelFamily::Glm,
                        resident: plans.iter().map(|p| p.resident_bytes as u64).sum(), workspace: 0, host_peak: 0 };
                    let mounted = NativeExperts::load_validated(&library, plan(), |_| Ok(()))?;
                    let before = fixture.events()?.matches('D').count();
                    let timing = crate::shared::memory::load_timing::Split::start();
                    let (pushed, records) = dense::recording(||
                        NativeExperts::load_validated(&library, plan(), |_| Ok(())))?;
                    assert_eq!(fixture.events()?.matches('D').count() - before, 1);
                    assert_eq!(timing.read_bytes(), 0, "recording must not read shared payloads");
                    assert!(!records.operands.is_empty());
                    push::tests::receive_pushed(device, None, root.path(), sockets.path(), &records, None, 0, rank)?;
                    for (mount, push) in mounted.weights.iter().zip(&pushed.weights) {
                        for unit in &mount.layout.loads {
                            for (source, &offset) in unit.source.iter().zip(&unit.destination) {
                                let mut expected = vec![0; source.bytes];
                                let mut actual = expected.clone();
                                library.copy_d2h(&mut expected, view(mount.buffer, offset, source.bytes)?)?;
                                library.copy_d2h(&mut actual, view(push.buffer, offset, source.bytes)?)?;
                                assert_eq!(actual, expected);
                            }
                        }
                        let (pointers, _) = push.layout.descriptors(push.buffer.ptr as u64)?;
                        let mut actual = vec![0; pointers.len() * 8];
                        library.copy_d2h(&mut actual, push.pointers()?)?;
                        assert_eq!(actual, pointers.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<_>>());
                    }
                    Ok(())
                })?;
            }
        }
        assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
        Ok(())
    }

    #[test]
    fn validated_shared_load_checks_receipt_before_allocating_without_resampling() {
        let source = include_str!("shared.rs").split("\n#[cfg(test)]\nmod receipt_tests").next().unwrap();
        let validated = source.split("pub(crate) fn load_validated").nth(1).unwrap()
            .split("fn load_inner").next().unwrap();
        assert!(validated.find("validate(plan.residency())?").unwrap()
            < validated.find("Self::load_inner").unwrap());
        assert!(!validated.contains("admit_host"));
        assert!(!validated.contains("host_memory"));
    }
}
