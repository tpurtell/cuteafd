//! Muse Spark's K6 projection owners on one coordinator rank: every program
//! the admitted layouts name, loaded by package geometry (no Muse stems), and
//! the one lane-private scratch set they share. The scratch the programs
//! require must equal the one admission charged
//! (`weight_layout::projections`), or startup refuses before allocating it.
use crate::shared::{dense_trellis::{CoreWeights, Projection, ProjectionLayout, ProjectionScratch},
    memory::{device::{Allocation, Device}, Tag}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{dense_trellis::TrellisDType, programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::muse_spark::weight_layout::{projection_key, ProjectionKey, ProjectionOutput,
    RankProjections};
use std::{collections::BTreeMap, ffi::c_void};

pub(crate) struct CorePrograms<'a> {
    exact_cap: usize,
    projections: BTreeMap<(ProjectionKey, ProjectionOutput), Projection<'a>>,
    pub(crate) scratch: Option<ProjectionScratch<'a>>,
    native: NativePrograms<'a>,
}

struct NativePrograms<'a> {
    bf16: cuteafd_ffi::draft_projection::DraftProjections<'a>,
    bf16_f32: cuteafd_ffi::draft_projection::DraftProjections<'a>,
    fp8: Option<cuteafd_ffi::native_attention::Fp8Projections<'a>>,
    fp8_f32: Option<cuteafd_ffi::native_attention::Fp8Projections<'a>>,
    scratch: Option<Allocation<'a>>,
    rows: usize,
}

impl<'a> NativePrograms<'a> {
    /// The program bindings `load` needs and the scratch bytes they require.
    /// The native loader caches modules per device, so a dropped binding stays
    /// loaded for the process and `load` costs no further module memory.
    fn bind(programs: &'a Programs<'a>, device: Device<'a>, admitted: &RankProjections) -> Result<(Self, usize)> {
        use cuteafd_ffi::draft_projection::{DraftProjections, ProjectionOutput as Output};
        use cuteafd_ffi::native_attention::Fp8Projections;
        use cuteafd_loader::placement::planner::native_attention::NativeResolution;
        ensure!(admitted.native.is_empty() || admitted.native_programs.resolution == NativeResolution::Exact,
            "Muse runtime requires exact native export admission");
        let shapes = |fp8, output| admitted.native.iter().filter(|request|
            request.layout.fp8 == fp8 && request.output == output)
            .map(|request| (request.layout.shape[1], request.layout.shape[0]))
            .collect::<std::collections::BTreeSet<_>>().into_iter().collect::<Vec<_>>();
        let rows = admitted.native_rows;
        let bf16 = device.run(|| Ok(DraftProjections::load_with_output(programs,
            &shapes(false, ProjectionOutput::Bf16), rows, Output::Bf16)?))?;
        let bf16_f32 = device.run(|| Ok(DraftProjections::load_with_output(programs,
            &shapes(false, ProjectionOutput::F32), rows, Output::F32)?))?;
        let mut required = 0usize;
        let mut load_fp8 = |output, dtype| -> Result<Option<Fp8Projections<'a>>> {
            let shapes = shapes(true, output);
            if shapes.is_empty() { return Ok(None); }
            let registry = device.run(|| Ok(Fp8Projections::load(programs, &shapes, rows, dtype)?))?;
            for (k, n) in shapes { required = required.max(registry.scratch_bytes(k, n)?); }
            Ok(Some(registry))
        };
        let fp8 = load_fp8(ProjectionOutput::Bf16, Output::Bf16)?;
        let fp8_f32 = load_fp8(ProjectionOutput::F32, Output::F32)?;
        ensure!(u64::try_from(required)? == admitted.native_programs.fp8_scratch_bytes,
            "Muse native scratch differs from exact admission");
        Ok((Self { bf16, bf16_f32, fp8, fp8_f32, scratch: None, rows }, required))
    }

    fn launch(&self, rows: usize, input: Buffer, weight: &super::weights::NativeWeights<'_>,
        output: Buffer, stream: *mut c_void) -> Result<()> {
        use cuteafd_ffi::draft_projection::ProjectionOutput as Output;
        use cuteafd_loader::formats::trellis_core::CoreSplit;
        let dtype = if weight.layout.ranks > 1 && weight.layout.split == CoreSplit::Input {
            Output::F32
        } else { Output::Bf16 };
        let [n, k] = weight.layout.shape;
        let bytes = rows.checked_mul(k).and_then(|bytes| bytes.checked_mul(2))
            .context("Muse native input extent overflow")?;
        ensure!(rows > 0 && rows <= self.rows && input.bytes >= bytes
            && input.device_id == output.device_id && weight.values.buffer.device_id == output.device_id,
            "Muse native projection exceeds admitted rows/buffers or device differs");
        if weight.layout.fp8 {
            let registry = match dtype { Output::Bf16 => &self.fp8, Output::F32 => &self.fp8_f32 }
                .as_ref().context("Muse native FP8 output registry absent")?;
            let scales = weight.scales.as_ref().context("Muse native FP8 scales absent")?.buffer;
            let scratch = self.scratch.as_ref().context("Muse native FP8 scratch absent")?.buffer;
            ensure!(scales.device_id == output.device_id, "Muse native scales device differs");
            // SAFETY: admitted resident weights and lane-private buffers survive queued stream work.
            unsafe { registry.launch(input.ptr, weight.values.buffer.ptr, scales.ptr,
                output, scratch, rows, k, n, stream)?; }
        } else {
            ensure!(weight.scales.is_none(), "Muse native BF16 projection has unexpected scales");
            let registry = match dtype { Output::Bf16 => &self.bf16, Output::F32 => &self.bf16_f32 };
            // SAFETY: admitted resident weights and lane buffers survive queued stream work; output is checked.
            unsafe { registry.launch_checked(input.ptr, weight.values.buffer.ptr, output, rows, k, n, stream)?; }
        }
        Ok(())
    }
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
    /// Every admitted K6 and native projection program, bound without the lane
    /// scratch. Startup runs this before the measured pre-weight sample so the
    /// module footprint is part of the resident process, not late growth (#1554).
    pub(crate) fn preload(programs: &'a Programs<'a>, device: Device<'a>, admitted: &RankProjections) -> Result<()> {
        Self::bind(programs, device, admitted).map(|_| ())
    }

    /// The program bindings `load` needs, their scratch layouts and the native
    /// scratch bytes. Dropping a binding keeps its modules loaded for the process.
    fn bind(programs: &'a Programs<'a>, device: Device<'a>, admitted: &RankProjections)
        -> Result<(BTreeMap<(ProjectionKey, ProjectionOutput), Projection<'a>>, Option<ProjectionLayout>,
            usize, NativePrograms<'a>, usize)> {
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
        ensure!(required.is_some() == admitted.scratch.is_some(),
            "Muse K6 programs and admitted scratch disagree on presence");
        if let (Some(required), Some(admitted)) = (&required, &admitted.scratch) {
            ensure!(required.extents == admitted.extents && required.rows == admitted.rows,
                "Muse K6 program scratch differs from admission");
        }
        let exact_cap = crate::shared::prefill_packing::coordinator_row_cap(programs,
            projections.values().flat_map(|projection| {
                let policy = projection.program.policy();
                [&policy.native_input, &policy.native, &policy.output_program].into_iter()
                    .map(|name| (name.as_str(), crate::shared::prefill_packing::ProgramRows::PackedRows))
            }).chain(admitted.native_programs.programs.iter().map(|name|
                (name.as_str(), crate::shared::prefill_packing::ProgramRows::PackedRows))),
            admitted.native_rows)?;
        let (native, native_scratch) = NativePrograms::bind(programs, device, admitted)?;
        Ok((projections, required, exact_cap, native, native_scratch))
    }

    pub(crate) fn load(programs: &'a Programs<'a>, device: Device<'a>, admitted: &RankProjections) -> Result<Self> {
        let (projections, required, exact_cap, mut native, native_scratch) =
            Self::bind(programs, device, admitted)?;
        let scratch = match required {
            Some(required) => Some(required.allocate(device,
                crate::shared::memory::Tag::workspace("muse/native-projection"))?),
            None => None,
        };
        native.scratch = if native_scratch == 0 { None }
            else { Some(Allocation::new(device, native_scratch, Tag::workspace("muse/core-scratch"))?) };
        Ok(Self { projections, scratch, exact_cap, native })
    }

    pub(crate) fn packed_cap(&self) -> usize { self.exact_cap }

    pub(crate) fn native(&self, rows: usize, input: Buffer, weight: &super::weights::NativeWeights<'_>,
        output: Buffer, stream: *mut c_void) -> Result<()> {
        self.native.launch(rows, input, weight, output, stream)
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
    fn native_registries_preserve_output_stage_rows_and_exact_scratch() -> Result<()> {
        use cuteafd_ffi::{draft_projection::ProjectionOutput as Output, native_library_lifetime_fixture::Fixture};
        use cuteafd_loader::{families::muse_spark::{core::NativeLayout, weight_layout::NativeRequest},
            formats::trellis_core::CoreSplit, placement::inventory::ProgramsManifest};
        use cuteafd_loader::placement::planner::native_attention::{NativeProgram, NativePrograms as Resolved};
        use serde_json::json;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        let mut entries = Vec::new();
        for output in [Output::Bf16, Output::F32] {
            entries.push(json!({"name":output.program_name(128,128),"capacity_rows":4096,
                "pointers":[{"name":"x"},{"name":"w"},{"name":"out"}],"scalars":[{"type":"int32"}],
                "scratch_bytes_at_capacity":{},"geometry":{"width":128,"outputs":128,
                    "output_dtype":output.torch_dtype(),"reduction":"ordered_k16","tiles_by_rows":[[1,16,32,2]],
                    "exact_row_cap":4096}}));
            for stage in [64, 4096] {
                let suffix = if output == Output::F32 { "_f32" } else { "" };
                entries.push(json!({"name":format!("native_attention_n128_k128_m{stage}{suffix}"),
                    "capacity_rows":stage,"pointers":[{"name":"x"},{"name":"weight"},{"name":"scales"},
                        {"name":"out"},{"name":"scratch"}],"scalars":[{"type":"int32"}],
                    "scratch_bytes_at_capacity":{"scratch":if output == Output::F32 {2048} else {512}},
                    "geometry":{"input_features":128,"output_features":128,"output_dtype":output.torch_dtype(),
                        "max_rows":stage,"decode_rows":64,"scale_block":128,"storage":"row_major_e4m3",
                        "reduction":"canonical_fp8_gemv64_w8a16","exact_row_cap":stage}}));
            }
        }
        fixture.register_programs(&library, &entries)?;
        for index in 0..entries.len() { fixture.register_reply(&library, index as u32, 7, b'P')?; }
        let manifest = json!({"programs":entries});
        let path = tempfile::tempdir()?;
        let file = path.path().join("PROGRAMS.json");
        std::fs::write(&file, serde_json::to_vec(&manifest)?)?;
        let programs = library.programs()?.with_manifest(&file)?;
        let typed = ProgramsManifest::from_manifest(&manifest)?;
        let device = Device { library: &library, id: 1 };
        for ceiling in [64, 2752] {
            let owners = [false, true].into_iter().flat_map(|fp8| [ProjectionOutput::Bf16, ProjectionOutput::F32]
                .into_iter().map(move |output| NativeRequest {
                    layout: NativeLayout { stem: format!("{fp8}-{output:?}"), shape: [128,128], fp8,
                        rank: 1, ranks: 2, split: if output == ProjectionOutput::F32 { CoreSplit::Input }
                            else { CoreSplit::Output } }, output })).collect::<Vec<_>>();
            let requests = owners.iter().map(|owner| NativeProgram { layout: &owner.layout,
                output: if owner.output == ProjectionOutput::F32 { Output::F32 } else { Output::Bf16 } })
                .collect::<Vec<_>>();
            let mut admitted = RankProjections { programs: Default::default(), scratch: None,
                native_rows: ceiling, native_programs: Resolved::resolve(&requests, ceiling, Some(&typed))?,
                native: owners.clone() };
            let (mut registry, scratch_bytes) = NativePrograms::bind(&programs, device, &admitted)?;
            registry.scratch = Some(Allocation::new(device, scratch_bytes, crate::shared::memory::TEST)?);
            assert_eq!(registry.scratch.as_ref().unwrap().buffer.bytes, 2048);
            for owner in &owners {
                let weight = device.own(|| {
                    use crate::shared::{memory::DeviceAllocation, native_attention::NativeAttentionWeights};
                    Ok(NativeAttentionWeights { layout: owner.layout.clone(),
                        values: DeviceAllocation::new(&library, 128 * 128 * if owner.layout.fp8 {1} else {2}, crate::shared::memory::TEST)?,
                        scales: owner.layout.fp8.then(|| DeviceAllocation::new(&library, 4, crate::shared::memory::TEST)).transpose()? })
                })?;
                let input = Allocation::new(device, (ceiling + 1) * 128 * 2, crate::shared::memory::TEST)?;
                device.run(|| library.copy_h2d(input.buffer, &7u32.to_ne_bytes()))?;
                let output_bytes = if owner.output == ProjectionOutput::F32 {4} else {2};
                let output = Allocation::new(device, (ceiling + 1) * 128 * output_bytes, crate::shared::memory::TEST)?;
                let before = fixture.events()?.bytes().filter(|&event| event == b'P').count();
                for rows in [1, 4, 64, ceiling] {
                    device.run(|| registry.launch(rows, input.buffer, &weight, output.buffer, std::ptr::null_mut()))?;
                }
                assert_eq!(fixture.events()?.bytes().filter(|&event| event == b'P').count(), before + 4);
                assert!(registry.launch(ceiling + 1, input.buffer, &weight, output.buffer, std::ptr::null_mut()).is_err());
                assert!(registry.launch(1, input.buffer, &weight, Buffer { bytes: 128 * output_bytes - 1,
                    ..output.buffer }, std::ptr::null_mut()).is_err());
                assert!(registry.launch(1, input.buffer, &weight, Buffer { device_id: 0,
                    ..output.buffer }, std::ptr::null_mut()).is_err());
                assert_eq!(fixture.events()?.bytes().filter(|&event| event == b'P').count(), before + 4);
            }
            library.cuda_set_device(0)?;
            drop(registry);
            admitted.native_programs.fp8_scratch_bytes += 256;
            assert!(NativePrograms::bind(&programs, device, &admitted).is_err());
            admitted.native_programs = Resolved::resolve(&requests, ceiling, None)?;
            assert!(NativePrograms::bind(&programs, device, &admitted).is_err());
        }
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

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
