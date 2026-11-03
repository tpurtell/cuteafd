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
            bits: weights.bits, codebook: weights.codebook });
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

    /// The K-split egress (shared rule): `rank0 + rank1` F32 cores
    /// added in rank order (`reduced`), then the one H128/svh egress.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduced(&self, add: &cuteafd_ffi::F32Add<'_>, rows: usize, rank0: Buffer, rank1: Buffer,
        reduced: Buffer, weights: &CoreWeights<'_>, output: Buffer, stream: *mut c_void) -> Result<()> {
        self.projection(weights, ProjectionOutput::F32)?.reduced_f32(add, rows, rank0, rank1, reduced, weights, output, stream)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tp2_package_admission_preserves_f32_until_post_egress_narrowing() -> Result<()> {
        use cuteafd_ffi::dense_trellis::{validate_output_buffers, validate_package_geometry, DenseTrellisError};
        use cuteafd_loader::families::muse_spark::{core::CoreInventory, lane::LanePlanes, tp2::{slot_bytes, Channel},
            weight_layout::projections, MuseSparkConfig};
        let snapshot = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_muse_spark_snapshot(snapshot.path(), &[]);
        let checkpoint = cuteafd_loader::plan::Checkpoint::inventory(snapshot.path())?;
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let inventory = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        // Exact relevant program/policy records extracted from the coherent be1726 seal.
        let manifest = serde_json::from_str(include_str!("fixtures/tp2-dense-programs.json"))?;
        let admitted = projections(&cfg, &inventory, 2, 2752)?;
        let planes = LanePlanes::new(&cfg, 2, 2752)?;
        for (rank, programs) in admitted.iter().enumerate() {
            let base = Buffer { ptr: (0x1000_0000usize + rank * 0x1_0000_0000) as *mut c_void,
                bytes: planes.bytes()?, device_id: rank as i32, flags: 0 };
            let lane = super::super::lane::LaneViews::carve(&planes, base)?;
            for rows in [1, 8, 16, 64, 129, 130, 2752] {
                let reduced = lane.rows("o_reduced", rows)?;
                let svh = Buffer { ptr: 0x10_0000_0000usize as *mut c_void, bytes: cfg.hidden * 2, ..base };
                for (k, channel, plane) in [(4096, Channel::Attention, "attended"), (2048, Channel::Moe, "moe")] {
                    let key = (k, cfg.hidden, 2, "mcg");
                    let output_dtype = *programs.programs.keys().find(|(geometry, _)| *geometry == key).unwrap();
                    let names = validate_package_geometry(&manifest, k, cfg.hidden, 2, "mcg", 2752, dtype(output_dtype.1))?;
                    assert_eq!(names[0], format!("dense_trellis_k6_trellis_input_k{k}_bf16"));
                    assert_eq!(names[1], format!("dense_trellis_k6_trellis_muse_{}_tp2_mcg",
                        if k == 4096 { "o" } else { "post_expert" }));
                    assert_eq!(names[2], "dense_trellis_k6_trellis_output_n8192_f32");
                    assert!(validate_package_geometry(&manifest, k, cfg.hidden, 2, "mcg", 2752, TrellisDType::Bf16).is_err());
                    assert_eq!(lane.rows("o_core", rows)?.bytes, rows * cfg.hidden * 4);
                    assert_eq!(slot_bytes(channel, cfg.hidden, cfg.expert_hidden, cfg.topk, rows)?, rows * cfg.hidden * 4);
                    let output = lane.rows(plane, rows)?;
                    // Reproduce the original rejected operand using the real TP2 lane layout.
                    let error = validate_output_buffers(rows, cfg.hidden, TrellisDType::F32, reduced, svh, output).unwrap_err();
                    assert!(matches!(error, DenseTrellisError::Buffer { stage: "output", operand: "output",
                        reason: "short extent", actual, required, device, .. }
                        if actual == rows * cfg.hidden * 2 && required == rows * cfg.hidden * 4 && device == rank as i32));
                    // o_core has been consumed by the sum and is now free for F32 H128 egress.
                    let staged = lane.rows("o_core", rows)?;
                    validate_output_buffers(rows, cfg.hidden, dtype(output_dtype.1), reduced, svh, staged)?;
                    assert_eq!(staged.bytes, 2 * output.bytes);
                }
            }
        }
        Ok(())
    }

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
        Ok(())
    }
}
