//! Native shared expert owner used by the ordinary GLM MoE execution path.
use super::*;
use crate::shared::experts::exlr8::{Exlr8Execution, Exlr8Transforms, Exlr8Weights, view};
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
    ) -> Result<Self> {
        let plans = (first..layers)
            .map(|layer| {
                Residency::plan(snapshot, manifest, layer, 1, 0, ExpertSet::Shared)
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
        let workspace = Exlr8Execution::workspace_bytes(programs, ROWS, ExpertSet::Shared)?;
        Ok(Self {
            plans,
            resident,
            workspace,
            host_peak,
        })
    }

    pub(crate) fn admit_host(&self) -> Result<()> {
        let text = std::fs::read_to_string("/proc/meminfo")?;
        let available = text
            .lines()
            .find_map(|line| line.strip_prefix("MemAvailable:"))
            .context("EXLR8 host admission needs MemAvailable")?
            .split_whitespace()
            .next()
            .context("EXLR8 host memory value")?
            .parse::<usize>()?
            .checked_mul(1024)
            .context("EXLR8 host memory overflow")?;
        let available = cuteafd_ffi::cgroup_memory_headroom(available)?;
        ensure!(
            self.host_peak <= available,
            "EXLR8 coordinator host peak {} exceeds available {available}",
            self.host_peak
        );
        Ok(())
    }
}

pub(crate) struct NativeExperts<'a> {
    execution: Exlr8Execution<'a>,
    transforms: Exlr8Transforms<'a>,
    weights: Vec<Exlr8Weights<'a>>,
    first: usize,
}

impl<'a> NativeExperts<'a> {
    pub(crate) fn load(
        library: &'a NativeLibrary,
        programs: &'a Programs<'a>,
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
        let transforms = Exlr8Transforms::new(library, programs)?;
        let mut execution =
            Exlr8Execution::new(library, programs, ROWS, workspace, ExpertSet::Shared)?;
        library.copy_h2d(execution.ids()?, &vec![0; ROWS * 4])?;
        library.copy_h2d(execution.route_weights()?, bytes_of(&vec![1.0f32; ROWS]))?;
        execution.seal_graph_storage()?;
        let weights = plans
            .into_iter()
            .map(|plan| {
                let bytes = plan.resident_bytes;
                Exlr8Weights::load(library, plan, bytes)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            execution,
            transforms,
            weights,
            first,
        })
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

    pub(crate) fn rotate(
        &self,
        layer: usize,
        w: &Workspace<'_>,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        // SAFETY: workspace and native weights are engine-owned through stream drain.
        unsafe {
            self.transforms.rotate(
                self.weights(layer)?,
                rows,
                w.x.buffer,
                w.wire.buffer,
                stream,
            )
        }
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
        let stride = weights.layout.hidden * 2;
        for first in (0..rows).step_by(ROWS) {
            let count = (rows - first).min(ROWS);
            let input = view(w.wire.buffer, first * stride, count * stride)?;
            let output = view(w.shared.buffer, first * stride * 2, count * stride * 2)?;
            // SAFETY: one stream orders copies, K1 and copies out before the next
            // chunk reuses the bounded arena; no host synchronization is needed.
            unsafe {
                library.copy_d2d_async(self.execution.input()?, input, input.bytes, stream)?;
                self.execution.run(weights, count, stream)?;
                library.copy_d2d_async(output, self.execution.output()?, output.bytes, stream)?;
            }
        }
        Ok(())
    }

    pub(crate) fn bind_planes(
        &self,
        library: &NativeLibrary,
        w: &Workspace<'_>,
        planes: [*const u16; MAX_RANKS],
        ranks: usize,
    ) -> Result<()> {
        ensure!((1..=6).contains(&ranks), "EXLR8 requires 1..=6 F32 rank planes");
        ensure!(planes[..ranks].iter().all(|ptr| !ptr.is_null()), "EXLR8 null rank plane");
        let addresses = planes.map(|ptr| ptr as usize);
        match *w.native_plane_identity.borrow() {
            Some(bound) => ensure!(bound == (addresses, ranks), "EXLR8 intake pointers changed while retained"),
            None => {
                let table = w.native_planes.as_ref().context("EXLR8 rank pointer storage")?;
                let pointers: Vec<u64> = addresses[..ranks].iter().map(|&ptr| ptr as u64).collect();
                library.copy_h2d(table.buffer, bytes_of(&pointers))?;
            }
        }
        *w.native_plane_identity.borrow_mut() = Some((addresses, ranks));
        Ok(())
    }

    pub(crate) fn reduce(
        &self,
        library: &NativeLibrary,
        layer: usize,
        w: &Workspace<'_>,
        planes: [*const u16; MAX_RANKS],
        ranks: usize,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        self.bind_planes(library, w, planes, ranks)?;
        let table = w.native_planes.as_ref().context("EXLR8 rank pointer storage")?;
        // SAFETY: intake completion precedes this call. Each thread reads its own
        // residual before writing the same element, so in-place residual is safe.
        unsafe {
            self.transforms.reduce(
                self.weights(layer)?,
                rows,
                ranks,
                table.buffer,
                w.shared.buffer,
                w.h.buffer,
                w.h.buffer,
                stream,
            )
        }
    }
}
