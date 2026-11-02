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

/// Visit exact BF16-input/F32-output row views before reusing the head scratch.
fn head_chunks(rows: usize, input: Buffer, output: Buffer, k: usize, n: usize,
    mut launch: impl FnMut(usize, Buffer, Buffer) -> Result<()>) -> Result<()> {
    let extent = |width: usize, element: usize| -> Result<usize> {
        rows.checked_mul(width).and_then(|v| v.checked_mul(element)).filter(|&v| v > 0)
            .context("Muse head typed extent overflow or empty")
    };
    let (input_bytes, output_bytes) = (extent(k, 2)?, extent(n, 4)?);
    ensure!(input.bytes >= input_bytes && output.bytes >= output_bytes,
        "Muse head input/output exceeds admitted extent");
    let start = input.ptr as usize;
    let end = start.checked_add(input_bytes).context("Muse head input address overflow")?;
    let out = output.ptr as usize;
    let out_end = out.checked_add(output_bytes).context("Muse head output address overflow")?;
    ensure!(start != 0 && out != 0 && (end <= out || out_end <= start), "Muse head input/output aliases or is null");
    for first in (0..rows).step_by(cuteafd_loader::families::muse_spark::lane::HEAD_ROWS) {
        let count = (rows - first).min(cuteafd_loader::families::muse_spark::lane::HEAD_ROWS);
        let view = |buffer: Buffer, stride: usize| Buffer {
            ptr: buffer.ptr.cast::<u8>().wrapping_add(first * stride).cast(), bytes: count * stride, ..buffer
        };
        launch(count, view(input, k * 2), view(output, n * 4))?;
    }
    Ok(())
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

    /// The whole K6 head with its F32 H128/svh egress, including drafter batches
    /// wider than the admitted 64-row head scratch. The head stream serializes
    /// each core/egress pair before its scratch can be reused by the next chunk.
    pub(crate) fn head_f32(&self, rows: usize, input: Buffer, weights: &CoreWeights<'_>, output: Buffer,
        stream: *mut c_void) -> Result<()> {
        let projection = self.projection(weights, ProjectionOutput::F32)?;
        let scratch = self.scratch()?;
        head_chunks(rows, input, output, weights.shape[1], weights.shape[0], |count, input, output| {
            projection.local(count, input, weights, scratch, output, stream)
        })
    }

    /// The F32 core result before a K-split projection's peer reduction.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_head_chunks_cover_target_and_drafter_rows_without_reusing_output() -> Result<()> {
        let (k, n) = (8192, 202112);
        for rows in [1, 16, 63, 64, 65, 80, 128, 129] {
            let input = Buffer { ptr: 0x10000usize as *mut c_void, bytes: rows * k * 2, device_id: 0, flags: 0 };
            let output = Buffer { ptr: 0x10000000usize as *mut c_void, bytes: rows * n * 4, ..input };
            let mut first = 0;
            let mut batches = Vec::new();
            head_chunks(rows, input, output, k, n, |count, x, y| {
                assert!((1..=64).contains(&count));
                assert_eq!((x.ptr as usize, x.bytes), (input.ptr as usize + first * k * 2, count * k * 2));
                assert_eq!((y.ptr as usize, y.bytes), (output.ptr as usize + first * n * 4, count * n * 4));
                batches.push(count);
                first += count;
                Ok(())
            })?;
            assert_eq!(first, rows);
            if rows == 128 { assert_eq!(batches, [64, 64]); }
            for (bad_rows, bad_input, bad_output) in [(0, input, output), (rows + 1, input, output),
                (rows, input, Buffer { bytes: output.bytes - 1, ..output }),
                (rows, input, Buffer { ptr: input.ptr, ..output }), (usize::MAX, input, output)] {
                let mut called = false;
                assert!(head_chunks(bad_rows, bad_input, bad_output, k, n, |_, _, _| {
                    called = true;
                    Ok(())
                }).is_err());
                assert!(!called, "validate the whole batch before any device launch");
            }
        }
        let source = include_str!("core.rs").split("#[cfg(test)]").next().unwrap();
        let head = source.split("pub(crate) fn head_f32(").nth(1).unwrap().split("pub(crate) fn core(").next().unwrap();
        assert!(head.contains("ProjectionOutput::F32") && head.contains("projection.local("));
        assert!(!head.contains("projection.core("), "the head needs its H128/svh egress, not just an F32 accumulator");
        Ok(())
    }
}
