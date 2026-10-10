//! `expert-probe --local`: the probe's wire rows and routes through the
//! coordinator's resident-layer path (`dsv4::local::LocalExperts`) on this GPU.
use crate::cli::ExpertProbeArgs;
use crate::families::deepseek_v4::local::{LocalExperts, LocalLayer};
use crate::shared::memory::DeviceAllocation;
use anyhow::{Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::OfficialV41Catalog;
use cuteafd_transport::ExpertProtocolV2RouteEntry;
use std::time::{Duration, Instant};

/// Loads only what the probe needs (dSpark stages `0..=stage`, or backbone
/// layer `layer` alone), runs one checked launch and, with `--repeat`, times
/// back-to-back launches. Returns routed experts as FP32 rows and the time
/// of the checked launch.
pub(super) fn run(
    args: &ExpertProbeArgs,
    catalog: &OfficialV41Catalog,
    wire: &[u8],
    routes: &[ExpertProtocolV2RouteEntry],
) -> Result<(Vec<f32>, Duration)> {
    let shape = *catalog.routed_experts();
    let (hidden, topk, rows) = (shape.hidden, shape.topk, args.rows as usize);
    let native_lib = args.native_lib.as_deref().context("--local needs --native-lib")?;
    // SAFETY: a trusted image library, loaded once for this process.
    let library = unsafe { NativeLibrary::load(native_lib) }?;
    library.cuda_set_device(0)?;
    let stream = library.cuda_stream_create()?;
    let (free, _) = library.cuda_memory_info()?;
    let (stages, backbone, layer) = match args.stage {
        Some(stage) => (stage + 1, 0..0, LocalLayer::Stage(stage)),
        None => (0, args.layer..args.layer + 1, LocalLayer::Backbone(args.layer)),
    };
    let layers = backbone.len();
    let started = Instant::now();
    let mut local = LocalExperts::load_range(&library, native_lib, catalog, stages, backbone, rows,
        free.saturating_sub(4 << 30), stream)?
        .context("no coordinator expert kernels or package for this checkpoint")?;
    anyhow::ensure!(local.stages() == stages && local.layers() == layers,
        "only {} stages and {} layers fit on this GPU", local.stages(), local.layers());
    eprintln!("resident in {:.1} s", started.elapsed().as_secs_f64());

    let upload = |bytes: &[u8]| -> Result<DeviceAllocation<'_>> {
        let allocation = DeviceAllocation::new(&library, bytes.len().max(16))?;
        library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    };
    let ids: Vec<u8> = routes.iter().flat_map(|r| r.expert_id.to_le_bytes()).collect();
    let weights: Vec<u8> = routes.iter().flat_map(|r| r.gate_weight.to_le_bytes()).collect();
    anyhow::ensure!(ids.len() == rows * topk * 4, "probe routes are not rows x top-k");
    let (wire, ids, weights) = (upload(wire)?, upload(&ids)?, upload(&weights)?);
    let shared = upload(&vec![0u8; rows * hidden * 2])?;
    let launch = |local: &mut LocalExperts<'_>| -> Result<()> {
        // SAFETY: every input is a live device allocation of the documented
        // extent, and the stream is drained before any of them is released.
        unsafe {
            local.run(layer, rows, wire.buffer.ptr, ids.buffer.ptr, weights.buffer.ptr, shared.buffer.ptr, stream)
        }
    };
    let sync = || unsafe { library.cuda_stream_synchronize(stream) };
    launch(&mut local)?;
    sync()?;
    let started = Instant::now();
    launch(&mut local)?;
    sync()?;
    let elapsed = started.elapsed();
    if args.repeat > 0 {
        let mut times = Vec::with_capacity(args.repeat);
        for _ in 0..args.repeat {
            let started = Instant::now();
            launch(&mut local)?;
            sync()?;
            times.push(started.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(f64::total_cmp);
        let started = Instant::now();
        for _ in 0..args.repeat {
            launch(&mut local)?;
        }
        sync()?;
        let queued = started.elapsed().as_secs_f64() * 1e6 / args.repeat as f64;
        println!("local {layer:?} rows {rows} over {} repeats: synchronized median {:.0} us, min {:.0} us; \
            back-to-back {queued:.0} us", args.repeat, times[times.len() / 2], times[0]);
    }
    let mut bytes = vec![0u8; rows * hidden * 2];
    let mut output = local.output.buffer;
    output.bytes = bytes.len();
    library.copy_d2h(&mut bytes, output)?;
    let actual = bytes.chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16)).collect();
    drop(local);
    // SAFETY: nothing is queued on the drained stream.
    unsafe { library.cuda_stream_destroy(stream)? };
    Ok((actual, elapsed))
}

/// `--local --local-tp N --exl3-package DIR`: every rank slice of an implicit
/// Spark TP-N EXL3 layout (`DIR/tp<N>-rank<R>`, the package the rank would
/// serve) runs on this GPU in turn on the probe's wire rows; BF16 rank
/// partials are summed in FP32 as the coordinator does. With `--repeat`, each
/// rank's back-to-back launch time is printed. Returns every row (FP32) and
/// the slowest rank's checked launch.
pub(super) fn run_exl3_ranks(
    args: &ExpertProbeArgs,
    catalog: &OfficialV41Catalog,
    wire: &[u8],
    routes: &[ExpertProtocolV2RouteEntry],
) -> Result<(Vec<f32>, Duration)> {
    use crate::shared::experts::exl3::execution::{Exl3Execution, Exl3InputFormat};
    use crate::shared::experts::{exl3::Exl3Weights, layer::ExpertLayer};
    let shape = *catalog.routed_experts();
    let (hidden, topk, rows, world) = (shape.hidden, shape.topk, args.rows as usize, args.local_tp);
    anyhow::ensure!(matches!(world, 2 | 3 | 4 | 6), "--local-tp {world}: EXL3 Spark worlds are 2, 3, 4 and 6");
    let root = args.exl3_package.as_deref().context("--exl3-package")?;
    let native_lib = args.native_lib.as_deref().context("--local needs --native-lib")?;
    // SAFETY: a trusted image library, loaded once for this process.
    let library = unsafe { NativeLibrary::load(native_lib) }?;
    library.cuda_set_device(0)?;
    let stream = library.cuda_stream_create()?;
    let capacity = [1usize, 16, 80, 256, 1024, 4096].into_iter().find(|&c| c >= rows)
        .context("rows exceed the largest EXL3 capacity")?;
    let upload = |bytes: &[u8]| -> Result<DeviceAllocation<'_>> {
        let allocation = DeviceAllocation::new(&library, bytes.len().max(16))?;
        library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    };
    let ids: Vec<u8> = routes.iter().flat_map(|r| (r.expert_id as i32).to_le_bytes()).collect();
    let weights: Vec<u8> = routes.iter().flat_map(|r| r.gate_weight.to_le_bytes()).collect();
    anyhow::ensure!(ids.len() == rows * topk * 4, "probe routes are not rows x top-k");
    let (wire, ids, weights) = (upload(wire)?, upload(&ids)?, upload(&weights)?);
    let sync = || unsafe { library.cuda_stream_synchronize(stream) };
    let mut total = vec![0f32; rows * hidden];
    let mut slowest = Duration::ZERO;
    for rank in 0..world {
        let layer = args.layer;
        let selection = match world {
            2 => ExpertLayer::BackboneTp2 { layer, rank },
            4 => ExpertLayer::Backbone { layer, rank },
            _ => ExpertLayer::BackboneExl3Tp { layer, rank, world },
        };
        let (free, _) = library.cuda_memory_info()?;
        let started = Instant::now();
        let resident = std::rc::Rc::new(vec![Exl3Weights::load(&library, catalog, selection,
            free.saturating_sub(4 << 30))?]);
        let directory = root.join(format!("tp{world}-rank{rank}")).join(format!("m{capacity}"));
        // SAFETY: a trusted package; the stream is drained before the
        // execution, its weights or any input is released.
        let mut execution = unsafe {
            Exl3Execution::with_input_format(&library, resident, &directory, Exl3InputFormat::Fp8K32)
        }.with_context(|| format!("EXL3 package {}", directory.display()))?;
        anyhow::ensure!(execution.output_element_bytes() == 2, "Spark EXL3 packages return BF16 partials");
        eprintln!("rank {rank}/{world} resident in {:.1} s", started.elapsed().as_secs_f64());
        let inputs = [wire.buffer, ids.buffer, weights.buffer];
        // SAFETY: inputs are live device allocations of the documented extents.
        let mut launch = || unsafe { execution.launch_layer(0, inputs, rows, stream) };
        launch()?;
        sync()?;
        let started = Instant::now();
        let output = launch()?;
        sync()?;
        slowest = slowest.max(started.elapsed());
        if args.repeat > 0 {
            let started = Instant::now();
            for _ in 0..args.repeat {
                launch()?;
            }
            sync()?;
            println!("local exl3 layer {layer} rows {rows} tp{world} rank {rank}: back-to-back {:.1} us over {} launches",
                started.elapsed().as_secs_f64() * 1e6 / args.repeat as f64, args.repeat);
        }
        let mut bytes = vec![0u8; rows * hidden * 2];
        let mut output = output;
        output.bytes = bytes.len();
        library.copy_d2h(&mut bytes, output)?;
        for (sum, pair) in total.iter_mut().zip(bytes.chunks_exact(2)) {
            *sum += f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
        }
    }
    // SAFETY: nothing is queued on the drained stream.
    unsafe { library.cuda_stream_destroy(stream)? };
    Ok((total, slowest))
}
