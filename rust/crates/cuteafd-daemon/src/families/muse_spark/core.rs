//! Muse Spark's K6 projection owners on one coordinator rank: every program
//! the admitted layouts name, loaded by package geometry (no Muse stems), and
//! the one lane-private scratch set they share. The scratch the programs
//! require must equal the one admission charged
//! (`weight_layout::projections`), or startup refuses before allocating it.
use crate::shared::{dense_trellis::{CoreWeights, Projection, ProjectionLayout, ProjectionScratch},
    memory::device::Device};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{dense_trellis::TrellisDType, programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::muse_spark::weight_layout::{projection_key, ProjectionKey, ProjectionOutput,
    RankProjections};
use std::{collections::BTreeMap, ffi::c_void};

pub(crate) struct CorePrograms<'a> {
    projections: BTreeMap<(ProjectionKey, ProjectionOutput), Projection<'a>>,
    pub(crate) scratch: Option<ProjectionScratch<'a>>,
}

fn dtype(output: ProjectionOutput) -> TrellisDType {
    match output { ProjectionOutput::Bf16 => TrellisDType::Bf16, ProjectionOutput::F32 => TrellisDType::F32 }
}

impl<'a> CorePrograms<'a> {
    pub(crate) fn load(programs: &'a Programs<'a>, device: Device<'a>, admitted: &RankProjections) -> Result<Self> {
        let mut projections = BTreeMap::new();
        let mut required: Option<ProjectionLayout> = None;
        for (&(key, output), &rows) in &admitted.programs {
            let (k, n, side, book) = key;
            let projection = device.run(|| Projection::load_geometry(programs, k, n, side, book, rows, dtype(output)))
                .with_context(|| format!("Muse K6 program ({k}, {n}, side {side}, {book}) at {rows} rows"))?;
            let mut layout = ProjectionLayout::new(&projection.program, rows)?;
            if let Some(scratch) = &admitted.scratch { layout.rows = scratch.rows; }
            match &mut required { Some(r) => r.merge(&layout)?, None => required = Some(layout) }
            projections.insert((key, output), projection);
        }
        let scratch = match (&required, &admitted.scratch) {
            (None, None) => None,
            (Some(required), Some(admitted)) => {
                ensure!(required.extents == admitted.extents && required.rows == admitted.rows,
                    "Muse K6 program scratch differs from admission");
                Some(required.allocate(device)?)
            }
            _ => anyhow::bail!("Muse K6 programs and admitted scratch disagree on presence"),
        };
        Ok(Self { projections, scratch })
    }

    fn projection(&self, weights: &CoreWeights<'_>, output: ProjectionOutput) -> Result<&Projection<'a>> {
        let key = projection_key(&cuteafd_loader::formats::trellis_core::CoreWeightLayout {
            stem: String::new(), side: weights.side, rank: 0, ranks: 1, shape: weights.shape, bytes: 0,
            codebook: weights.codebook });
        self.projections.get(&(key, output)).with_context(|| format!("Muse K6 program {key:?} {output:?} not admitted"))
    }

    fn scratch(&self) -> Result<&ProjectionScratch<'a>> {
        self.scratch.as_ref().context("Muse rank has no K6 scratch")
    }

    /// A whole or output-split projection with BF16 rows out.
    pub(crate) fn local(&self, rows: usize, input: Buffer, weights: &CoreWeights<'_>, output: Buffer,
        stream: *mut c_void) -> Result<()> {
        self.projection(weights, ProjectionOutput::Bf16)?.local(rows, input, weights, self.scratch()?, output, stream)
    }

    /// The F32 core result (head; the TP2 o_proj partial before its peer reduction).
    pub(crate) fn core(&self, rows: usize, input: Buffer, weights: &CoreWeights<'_>, stream: *mut c_void)
        -> Result<Buffer> {
        self.projection(weights, ProjectionOutput::F32)?.core(rows, input, weights, self.scratch()?, stream)
    }

    /// The K-split egress (shared rule): `rank0 + rank1` BF16 cores widened and
    /// added in rank order in F32 (`reduced`), then the one H128/svh egress.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduced(&self, add: &cuteafd_ffi::Bf16F32Add<'_>, rows: usize, rank0: Buffer, rank1: Buffer,
        reduced: Buffer, weights: &CoreWeights<'_>, output: Buffer, stream: *mut c_void) -> Result<()> {
        self.projection(weights, ProjectionOutput::F32)?.reduced(add, rows, rank0, rank1, reduced, weights, output, stream)
    }

    pub(crate) fn program_count(&self) -> usize { self.projections.len() }
}
