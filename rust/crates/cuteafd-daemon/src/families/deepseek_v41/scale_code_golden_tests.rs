//! Real checkpoint scale planes, independently packed on the CPU and compared
//! with production mount, Spark push, coordinator TP2 and recorded dSpark loads.
//! Run in the slot's architecture dev image.
use super::coordinator_golden_tests::{library, snapshot};
use crate::shared::{
    experts::layer::{ExpertLayer, ExpertWeights},
    memory::{device::Device, weight_arena::{ArenaLayout, ExpectedCoverage}},
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{formats::scale_codes::{plane_bytes, Rows}, OfficialV41Catalog, V41ExpertSelection};

fn expected(catalog: &OfficialV41Catalog, selection: V41ExpertSelection) -> Result<[Vec<u8>; 2]> {
    use crate::shared::experts::rtx::dense::Rect;
    let staging = catalog.expert_staging(selection)?;
    let logical = staging.intermediate_size();
    let stored = logical.div_ceil(128) * 128;
    let hidden = catalog.routed_experts().hidden;
    let rank = match selection {
        V41ExpertSelection::Backbone { rank, .. }
        | V41ExpertSelection::BackboneTp2 { rank, .. }
        | V41ExpertSelection::BackboneTp { rank, .. }
        | V41ExpertSelection::DsparkTp2 { rank, .. } => rank,
        _ => 0,
    };
    let mut planes = [vec![0; plane_bytes(2 * stored, hidden / 32).context("gate plane")?],
        vec![0; plane_bytes(hidden, logical / 32).context("down plane")?]];
    for slot in [4, 3, 5] {
        let down = slot == 5;
        let name = &staging.tensor_names()[slot];
        let tensor = catalog.tensor(name)?;
        let row_bytes = tensor.metadata.shape[1];
        let rows = if down { 0..hidden } else { rank * logical..(rank + 1) * logical };
        let cols = if down { rank * logical / 32..(rank + 1) * logical / 32 } else { 0..hidden / 32 };
        let rect = Rect { name: name.clone(), path: catalog.snapshot().join(&tensor.shard),
            base: tensor.metadata.byte_offset, row_bytes, rows, cols };
        let mut source = vec![0; rect.bytes()];
        rect.read(&mut source)?;
        Rows { tensor: name, bytes: &source, rows: rect.rows.len(), pitch: rect.width(),
            columns: rect.width(), first_row: rect.rows.start }.pack(&mut planes[usize::from(down)],
                if down { hidden } else { 2 * stored }, if slot == 3 { stored } else { 0 })?;
    }
    Ok(planes)
}

fn compare(library: &NativeLibrary, buffer: CuteafdDeviceBuffer, offset: usize, expected: &[u8], name: &str) -> Result<()> {
    ensure!(offset.checked_add(expected.len()).is_some_and(|end| end <= buffer.bytes), "short coded {name}");
    // SAFETY: the checked range belongs to the live mount or push owner.
    let selected = CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
        bytes: expected.len(), ..buffer };
    let mut actual = vec![0; expected.len()];
    Device { library, id: buffer.device_id }.run(|| library.copy_d2h(&mut actual, selected))?;
    assert_eq!(actual, expected, "{name}");
    Ok(())
}

fn compare_mount(library: &NativeLibrary, catalog: &OfficialV41Catalog, weights: &ExpertWeights<'_>,
    layer: ExpertLayer) -> Result<()> {
    let experts = layer.select_info(library, 16, false)?.experts as usize;
    for expert in [0, experts - 1] {
        let oracle = expected(catalog, layer.expert(expert))?;
        for (plane, bytes) in [1, 3].into_iter().zip(oracle) {
            let buffer = weights.buffers[plane].buffer;
            ensure!(buffer.bytes == bytes.len() * experts, "mount coded extent mismatch");
            compare(library, buffer, expert * bytes.len(), &bytes, &format!("{layer:?} expert{expert} plane{plane}"))?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_V41, CUTEAFD_NATIVE_LIB and a GPU with V4.1 expert packages"]
fn installed_mount_scale_codes_equal_cpu_pack() -> Result<()> {
    let snapshot = snapshot().context("set CUTEAFD_PUSH_V41")?;
    let library = library()?;
    library.cuda_set_device(0)?;
    let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
    ensure!(catalog.mxfp4_scale_codes() && catalog.nvfp4().is_none(), "requires native V4.1");
    for layer in [ExpertLayer::BackboneTp2 { layer: 3, rank: 0 },
        ExpertLayer::BackboneTp2 { layer: 3, rank: 1 },
        ExpertLayer::Dspark { stage: 0 }, ExpertLayer::Dspark { stage: 1 }, ExpertLayer::Dspark { stage: 2 }] {
        let budget = ExpertWeights::plan(&library, &catalog, layer)?;
        let weights = ExpertWeights::load(&library, &catalog, layer, budget.peak_device_bytes()?)?;
        compare_mount(&library, &catalog, &weights, layer)?;
        eprintln!("{layer:?}: mount scale planes byte-exact against CPU");
    }
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_V41, CUTEAFD_NATIVE_LIB and one GPU"]
fn installed_pushed_scale_codes_equal_cpu_pack() -> Result<()> {
    use crate::shared::{experts::rtx::dense::{testing::FakeRank},
        memory::{push_consumer, push_native::ScalePlan}};
    use cuteafd_loader::push_load::{builder::Builder, native::NativePlanes};
    let snapshot = snapshot().context("set CUTEAFD_PUSH_V41")?;
    let library = library()?;
    library.cuda_set_device(0)?;
    let device = Device { library: &library, id: 0 };
    let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
    ensure!(catalog.mxfp4_scale_codes() && catalog.nvfp4().is_none(), "requires native V4.1");
    for world in [1, 2, 3, 4, 6] {
        let rank = world - 1;
        let hidden = catalog.routed_experts().hidden;
        let logical = catalog.routed_experts().intermediate / world;
        // One real expert is sufficient to cover each source geometry without
        // staging a full checkpoint in this transform-focused ring test.
        let planes = NativePlanes::with_scale_codes(hidden, logical, logical.div_ceil(128) * 128, 1, false, true)?;
        let layout = ArenaLayout::new([planes.bytes], planes.bytes)?.with_generated(planes.generated_ranges())?;
        let scales = ScalePlan::new(&catalog, [(3, planes.clone(), 0)], world, rank)?;
        let mut builder = Builder::new(&snapshot, vec![scales.logical_bytes(&layout)]);
        scales.append(&mut builder, 0, scales.base(&layout))?;
        let checked = builder.finish()?;
        let expected_coverage = ExpectedCoverage::from_plan(&checked, 0)?;
        let arena = scales.destination(device, layout, expected_coverage)?;
        let ring = cuteafd_ffi::push_ring::RegisteredRing::new(&library, 8,
            cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES)?;
        let fake = FakeRank::new(&checked, &snapshot, 0, &ring)?;
        let delivered = push_consumer::consume(device, arena, ring, fake, 8)?;
        // This is the production post-Done padding pass; it must not erase
        // generated scales just because they were not source-delivered bytes.
        delivered.zero_padding()?;
        let ready = delivered.initialize(|_| Ok(()))?;
        let selection = if world == 1 { V41ExpertSelection::BackboneFull { layer: 3, expert: 0 } }
            else { V41ExpertSelection::BackboneTp { layer: 3, expert: 0, world, rank } };
        let oracle = expected(&catalog, selection)?;
        for (plane, bytes) in [1, 3].into_iter().zip(oracle) {
            compare(&library, ready.region(0)?.buffer(), planes.planes[plane].0, &bytes,
                &format!("push TP{world} rank{rank} plane{plane}"))?;
        }
        let peaks = crate::shared::experts::rtx::dense::testing::PEAKS.with(|p| p.take());
        for (staged, pinned) in peaks {
            ensure!(staged <= scales.host_staging().staging_bytes as u64
                && pinned <= scales.host_staging().staging_bytes as u64, "unbounded scale staging");
            eprintln!("push TP{world}: scale planes byte-exact; staged={staged} pinned={pinned}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_V41, CUTEAFD_NATIVE_LIB and V4.1 TP2 GPU packages"]
fn installed_coordinator_tp2_scale_codes_equal_cpu_pack() -> Result<()> {
    use crate::shared::{
        experts::rtx::{dense::{testing::PEAKS, Job, Operand, Rect, Records, Step}, push::tests::receive_pushed},
        memory::{push_coordinator::CoordinatorExpertPlan, DeviceAllocation},
    };
    let snapshot = snapshot().context("set CUTEAFD_PUSH_V41")?;
    let library = library()?;
    library.cuda_set_device(0)?;
    let device = Device { library: &library, id: 0 };
    let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
    ensure!(catalog.mxfp4_scale_codes() && catalog.nvfp4().is_none(), "requires native V4.1");
    let tensor = catalog.tensor("layers.3.ffn.experts.0.w1.scale")?;
    let rect = Rect { name: tensor.metadata.name.clone(), path: snapshot.join(&tensor.shard),
        base: tensor.metadata.byte_offset, row_bytes: tensor.metadata.shape[1], rows: 0..1,
        cols: 0..tensor.metadata.shape[1] };
    let mut dense_expected = vec![0; rect.bytes()];
    rect.read(&mut dense_expected)?;
    let allocations = [DeviceAllocation::new(&library, 256)?, DeviceAllocation::new(&library, 256)?];
    let mut records = Records::default();
    records.operands.push(Operand { device: 0, buffer: allocations[0].buffer,
        lands: vec![(rect.clone(), 0)], bytes: rect.bytes() });
    records.steps.push(Step { device: 0, rects: vec![rect.clone()],
        jobs: vec![Job::Upload { at: 0, bytes: rect.bytes(), dst: allocations[1].buffer }] });
    let socket_root = std::env::var_os("CUTEAFD_SCALE_CODES_SOCKET_DIR")
        .context("set CUTEAFD_SCALE_CODES_SOCKET_DIR to a writable task directory")?;
    let sockets = tempfile::tempdir_in(socket_root)?;
    // Receive each logical TP2 half on one physical GPU, with real dense and
    // staged bytes on either side of the TP2 arena's logical space. The
    // bounded fake source and receive_rank exercise the actual Combined path.
    for rank in 0..2 {
        PEAKS.with(|p| { p.take(); });
        let layer = ExpertLayer::BackboneTp2 { layer: 3, rank };
        let budget = ExpertWeights::plan(&library, &catalog, layer)?;
        let plan = CoordinatorExpertPlan::new(&catalog, &[3], rank, budget.resident_bytes)?;
        ensure!(plan.layout.bytes() == budget.resident_bytes, "TP2 admitted extent mismatch");
        let pushed = receive_pushed(device, Some(&catalog), &snapshot, sockets.path(), &records,
            Some(plan), 6, rank)?.native()?;
        ensure!(pushed.len() == 1, "expected one committed TP2 layer");
        compare_mount(&library, &catalog, &pushed[0], layer)?;
        for allocation in &allocations {
            compare(&library, allocation.buffer, 0, &dense_expected, "combined dense/staged bytes")?;
        }
        let peaks = PEAKS.with(|p| p.take());
        ensure!(peaks.len() == 2, "expected dense and coded TP2 staging ledgers");
        for (staged, pinned) in peaks {
            ensure!(staged <= 256u64 << 20 && pinned + (1 << 20) <= 257u64 << 20,
                "coordinator TP2 scale staging exceeds 257 MiB including job table");
            eprintln!("coordinator TP2 rank{rank}: staged={staged} pinned={pinned}");
        }
        eprintln!("{layer:?}: coordinator receive_rank scale planes byte-exact against CPU");
    }
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_V41, CUTEAFD_NATIVE_LIB and dSpark GPU packages"]
fn installed_recorded_dspark_scale_codes_equal_cpu_pack() -> Result<()> {
    use crate::shared::experts::rtx::dense::{self, RankLayout, Recorder};
    let snapshot = snapshot().context("set CUTEAFD_PUSH_V41")?;
    let library = library()?;
    library.cuda_set_device(0)?;
    let device = Device { library: &library, id: 0 };
    let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
    for stage in 0..3 {
        let layer = ExpertLayer::Dspark { stage };
        let budget = ExpertWeights::plan(&library, &catalog, layer)?;
        let recorder = Recorder::start(true)?.context("recorder")?;
        let weights = ExpertWeights::load(&library, &catalog, layer, budget.peak_device_bytes()?)?;
        let records = recorder.finish();
        let layout = RankLayout::new(records.operands, records.steps, 0)?.with_finishes(&records.finishes);
        dense::testing::replay(device, layout, &snapshot, 1, 0)?;
        compare_mount(&library, &catalog, &weights, layer)?;
        eprintln!("{layer:?}: recorded push scale planes byte-exact against CPU");
    }
    Ok(())
}
