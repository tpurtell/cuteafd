//! Native shared expert owner used by the ordinary GLM MoE execution path.
use super::*;
use crate::shared::experts::exlr8::{Exlr8Execution, Exlr8Weights, view};
use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};

const ROWS: usize = cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS;

#[cfg(test)]
mod golden;

pub(crate) struct NativePlan {
    plans: Vec<Residency>,
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
        let workspace = Exlr8Execution::shared_workspace_bytes(programs, ROWS)?;
        Ok(Self {
            plans,
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
    workspace: usize,
    weights: Vec<Exlr8Weights<'a>>,
    first: usize,
}

impl<'a> NativeExperts<'a> {
    pub(crate) fn load(
        library: &'a NativeLibrary,
        plan: NativePlan,
    ) -> Result<Self> {
        plan.admit_host()?;
        let NativePlan {
            plans, workspace, ..
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
            workspace,
            weights,
            first,
        })
    }

    pub(super) fn execution(
        &self,
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
    ) -> Result<Exlr8Execution<'a>> {
        let mut execution =
            Exlr8Execution::new_shared(library, programs, ROWS, self.workspace)?;
        library.copy_h2d(execution.ids()?, &vec![0; ROWS * 4])?;
        library.copy_h2d(execution.route_weights()?, bytes_of(&vec![1.0f32; ROWS]))?;
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
        w: &Workspace<'_>,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let weights = self.weights(layer)?;
        let execution = w.native_shared.as_ref().context("EXLR8 lane shared workspace")?;
        let stride = weights.layout.hidden * 2;
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let trace = *TRACE.get_or_init(|| std::env::var("CUTEAFD_EXLR8_TILE_TRACE").is_ok_and(|v| v == "1"));
        let issued = |first, operation, returned| {
            if trace {
                eprintln!("EXLR8 shared enqueue rank={} layer={layer} rows={rows} first={first} stream={stream:p} operation={operation} returned={returned}",
                    weights.layout.rank);
            }
        };
        for first in (0..rows).step_by(ROWS) {
            let count = (rows - first).min(ROWS);
            let input = view(w.x.buffer, first * stride, count * stride)?;
            let output = view(w.shared.buffer, first * stride, count * stride)?;
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
