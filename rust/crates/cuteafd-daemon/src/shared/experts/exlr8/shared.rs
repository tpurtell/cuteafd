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
