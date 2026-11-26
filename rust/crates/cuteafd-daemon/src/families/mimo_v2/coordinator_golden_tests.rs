//! MiMo coordinator operands from the installed checkpoint (CUTEAFD_PUSH_MIMO)
//! on real RTX GPUs: decoder layers 0 (dense MLP) and 1 (SWA + router), the
//! final norm and head on a two-GPU head split and on one GPU, and MTP layer 0
//! with its extras, under the checkpoint's resolved FP8 formats. Digests are
//! pinned to the mount loader as it was before push recipes, so the push
//! recipes (landings, staged row quantization, scale gathers at Done) must
//! reproduce it bit for bit. CUTEAFD_MIMO_GOLDEN_WRITE=1 records instead.
use super::weights::{MimoLayer, MimoLoader, MimoWeights};
use crate::shared::memory::device::Device;
use anyhow::{Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::mimo_v2::{projection::MimoProjectionRepresentation, MimoV2Config};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(crate) type Digests = BTreeMap<String, String>;

pub(crate) fn digest(library: &NativeLibrary, buffer: CuteafdDeviceBuffer) -> Result<String> {
    let mut bytes = vec![0u8; buffer.bytes];
    Device { library, id: buffer.device_id }.run(|| library.copy_d2h_quiescent(&mut bytes, buffer))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

pub(crate) fn layer_digests(library: &NativeLibrary, prefix: &str, layer: &MimoLayer<'_>, out: &mut Digests)
    -> Result<()> {
    for (name, mut buffer) in layer.buffers() {
        // Sinks below 256 B (a split rank's share) sit in a 256 B allocation
        // whose tail no loader writes: digest the operand's own bytes.
        if name == "sinks" { buffer.bytes = buffer.bytes.min(SINK_BYTES.with(|s| s.get()) / if layer.split { 2 } else { 1 }); }
        let key = format!("{prefix}.{name}");
        anyhow::ensure!(out.insert(key.clone(), digest(library, buffer)?).is_none(), "duplicate {key}");
    }
    Ok(())
}

pub(crate) fn weights_digests(library: &NativeLibrary, prefix: &str, weights: &MimoWeights<'_>, out: &mut Digests)
    -> Result<()> {
    for (layer, share) in weights.layers.iter().enumerate() {
        layer_digests(library, &format!("{prefix}.layers.{layer}"), share, out)?;
    }
    out.insert(format!("{prefix}.norm"), digest(library, weights.norm.buffer)?);
    for (i, allocation) in weights.head.allocations().into_iter().enumerate() {
        out.insert(format!("{prefix}.head.{i}"), digest(library, allocation.buffer)?);
    }
    Ok(())
}

thread_local! {
    /// The whole sink tensor's bytes (set by `setup`).
    static SINK_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

pub(crate) fn snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTEAFD_PUSH_MIMO").map(std::path::PathBuf::from)
}

pub(crate) fn library() -> Result<NativeLibrary> {
    let path = std::env::var("CUTEAFD_NATIVE_LIB").context("set CUTEAFD_NATIVE_LIB")?;
    // SAFETY: the slot's native library, loaded once for this test process.
    unsafe { NativeLibrary::load(path) }
}

/// The golden's layers, the formats serving resolves for them, and a loader.
pub(crate) struct Setup {
    pub cfg: MimoV2Config,
    pub checkpoint: cuteafd_loader::plan::checkpoint::Checkpoint,
    pub formats: BTreeMap<String, MimoProjectionRepresentation>,
    pub checkpoint_tp: usize,
}

pub(crate) const LAYERS: usize = 2;

pub(crate) fn setup(snapshot: &std::path::Path) -> Result<Setup> {
    let cfg = MimoV2Config::read(snapshot)?;
    let checkpoint = cuteafd_loader::plan::checkpoint::Checkpoint::coordinator(snapshot, false, true)?;
    let formats = (0..LAYERS).map(|l| format!("model.layers.{l}.self_attn.o_proj.weight"))
        .chain(std::iter::once("model.mtp.layers.0.self_attn.o_proj.weight".to_string()))
        .map(|name| (name, MimoProjectionRepresentation::Fp8)).collect();
    let sinks = checkpoint.tensors.iter().find(|t| t.meta.name == "model.layers.1.self_attn.attention_sink_bias")
        .context("layer 1 sinks")?.meta.byte_length as usize;
    SINK_BYTES.with(|s| s.set(sinks));
    Ok(Setup { cfg, checkpoint_tp: cuteafd_loader::families::mimo_v2::checkpoint_tp(snapshot)?, checkpoint, formats })
}

pub(crate) fn loader<'a>(library: &'a NativeLibrary, setup: &'a Setup, peers: usize) -> MimoLoader<'a> {
    MimoLoader { library, checkpoint: &setup.checkpoint, stream: std::ptr::null_mut(),
        checkpoint_tp: setup.checkpoint_tp, fp8_head: true, output_formats: &setup.formats,
        fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax, device: 0,
        peers: (1..=peers).map(|device| crate::shared::peer_split::RankDevice { device: device as i32,
            stream: std::ptr::null_mut() }).collect(),
        dense: std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::mount(library)), transient: Default::default(), arenas: Default::default() }
}

/// Every golden operand through `loader`s (two-GPU head split, then one GPU).
pub(crate) fn mount(library: &NativeLibrary, setup: &Setup, out: &mut Digests) -> Result<()> {
    let split = loader(library, setup, 1);
    let (weights, shares) = split.model(&setup.cfg, LAYERS)?;
    weights_digests(library, "split.rank0", &weights, out)?;
    for (rank, share) in shares.iter().enumerate() {
        for (layer, part) in share.iter().enumerate() {
            layer_digests(library, &format!("split.rank{}.layers.{layer}", rank + 1), part, out)?;
        }
    }
    drop((weights, shares));
    let mtp = split.mtp_layer(&setup.cfg, 0)?;
    layer_digests(library, "mtp0", &mtp, out)?;
    for (i, extra) in split.mtp_extras(0)?.iter().enumerate() { out.insert(format!("mtp0.extra{i}"), digest(library, extra.buffer)?); }
    drop(mtp);
    let single = loader(library, setup, 0);
    let (weights, _) = single.model(&setup.cfg, LAYERS)?;
    weights_digests(library, "single", &weights, out)?;
    Ok(())
}

pub(crate) fn path() -> std::path::PathBuf {
    std::env::var_os("CUTEAFD_MIMO_GOLDEN").map(std::path::PathBuf::from).unwrap_or_else(||
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/mimo_v2/coordinator_golden.json"))
}

pub(crate) fn check(actual: &Digests, subset: bool) -> Result<()> {
    if std::env::var_os("CUTEAFD_MIMO_GOLDEN_WRITE").is_some() {
        let out = std::env::var_os("CUTEAFD_MIMO_GOLDEN_OUT").map(std::path::PathBuf::from).unwrap_or_else(path);
        std::fs::write(&out, serde_json::to_vec_pretty(actual)?)?;
        eprintln!("wrote {} operand digests to {}", actual.len(), out.display());
        return Ok(());
    }
    let golden: Digests = serde_json::from_slice(&std::fs::read(path())?)?;
    if !subset {
        anyhow::ensure!(actual.keys().eq(golden.keys()), "operand inventory changed: {} actual, {} golden",
            actual.len(), golden.len());
    }
    let differ = actual.iter().filter(|(k, v)| golden.get(*k) != Some(v)).map(|(k, _)| k.clone()).collect::<Vec<_>>();
    anyhow::ensure!(differ.is_empty(), "{} operands differ from the golden, first {:?}", differ.len(),
        &differ[..differ.len().min(8)]);
    eprintln!("PASS {} operand digests match the golden", actual.len());
    Ok(())
}

#[test]
#[ignore = "requires CUTEAFD_PUSH_MIMO, CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let setup = setup(&snapshot)?;
    let cfg = &setup.cfg;
    let first_layer = cfg.dense.iter().position(|dense| !dense).unwrap_or(cfg.layers);
    let _ = cuteafd_core::set_expert_geometry(cuteafd_loader::RoutedExpertShape { layers: cfg.layers, first_layer,
        experts: cfg.experts, topk: cfg.topk, hidden: cfg.hidden, intermediate: cfg.moe_intermediate,
        draft_stages: 0, draft_experts: 0, activation: cuteafd_core::ExpertActivation::SiluUnclamped }.geometry()?);
    let library = library()?;
    library.cuda_set_device(0)?;
    for gpu in [0, 1] { Device { library: &library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?; }
    let mut actual = Digests::new();
    mount(&library, &setup, &mut actual)?;
    check(&actual, false)
}

/// The same split operands recorded (allocated as through the mount, bytes
/// deferred), then each GPU's records pushed through a ring into its rank in
/// file order (row-quant steps on the GPU, scale gathers at Done): every
/// operand matches the mount golden and each rank's staging stays one buffer.
#[test]
#[ignore = "requires CUTEAFD_PUSH_MIMO (a SparkNest namespace), CUTEAFD_NATIVE_LIB and two RTX GPUs"]
fn installed_pushed_coordinator_operands_match_the_mount_golden() -> Result<()> {
    use crate::shared::experts::rtx::dense::{DenseLoad, STEP_BYTES};
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let setup = setup(&snapshot)?;
    let cfg = &setup.cfg;
    let first_layer = cfg.dense.iter().position(|dense| !dense).unwrap_or(cfg.layers);
    let _ = cuteafd_core::set_expert_geometry(cuteafd_loader::RoutedExpertShape { layers: cfg.layers, first_layer,
        experts: cfg.experts, topk: cfg.topk, hidden: cfg.hidden, intermediate: cfg.moe_intermediate,
        draft_stages: 0, draft_experts: 0, activation: cuteafd_core::ExpertActivation::SiluUnclamped }.geometry()?);
    let library = library()?;
    library.cuda_set_device(0)?;
    for gpu in [0, 1] { Device { library: &library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?; }
    let (root, _) = cuteafd_loader::push_load::namespace(&snapshot)?;
    let load = std::rc::Rc::new(DenseLoad::push(&library));
    let split = MimoLoader { dense: load.clone(), ..loader(&library, &setup, 1) };
    let (weights, shares) = split.model(&setup.cfg, LAYERS)?;
    let records = load.sink.as_ref().context("push sink")?.take_all();
    let (bytes, operands, staged, steps, largest, finishes) = records.summary();
    eprintln!("records: class2 {bytes} B in {operands} operands; class3 {staged} B in {steps} steps (largest {largest} B); \
        {finishes} finishes");
    anyhow::ensure!(largest <= STEP_BYTES, "a step stages {largest} bytes");
    let sockets = tempfile::tempdir()?;
    for gpu in 0..2 {
        let started = std::time::Instant::now();
        crate::shared::experts::rtx::push::tests::receive_pushed(Device { library: &library, id: gpu as i32 },
            None, &root, sockets.path(), &records, None, 6, gpu)?;
        eprintln!("gpu{gpu} rank pushed in {:.2} s", started.elapsed().as_secs_f64());
    }
    split.release_transient();
    for (staged, pinned) in crate::shared::experts::rtx::dense::testing::PEAKS.with(|p| p.take()) {
        eprintln!("rank staging peak {staged} B in use, {pinned} B pinned");
        anyhow::ensure!(staged <= STEP_BYTES as u64 && pinned <= STEP_BYTES as u64, "staging beyond one step");
    }
    let mut actual = Digests::new();
    weights_digests(&library, "split.rank0", &weights, &mut actual)?;
    for (rank, share) in shares.iter().enumerate() {
        for (layer, part) in share.iter().enumerate() {
            layer_digests(&library, &format!("split.rank{}.layers.{layer}", rank + 1), part, &mut actual)?;
        }
    }
    check(&actual, true)
}

/// The scale gather the push recipes use reproduces the host expansion the
/// mount loader did before them (fused qkv per-row scales and their K-block
/// major copy, padding rows zero), on the installed checkpoint's real grids,
/// through the gather's CPU twin (the fake native DSO).
#[test]
#[ignore = "requires CUTEAFD_PUSH_MIMO, an allocated CPU build slot and explicit NVMe fixture directory"]
fn scale_gather_matches_the_host_qkv_expansion() -> Result<()> {
    use cuteafd_loader::families::mimo_v2::{FusedQkvLayout, MimoAttention};
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let setup = setup(&snapshot)?;
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
    let library = fixture.load()?;
    let cfg = &setup.cfg;
    for (layer, attention) in [(0usize, cfg.attention[0]), (1, cfg.attention[1])] {
        let name = format!("model.layers.{layer}.self_attn.qkv_proj.weight_scale_inv");
        let tensor = setup.checkpoint.tensors.iter().find(|t| t.meta.name == name).context("grid")?;
        let mut bytes = vec![0u8; tensor.meta.byte_length as usize];
        use std::os::unix::fs::FileExt;
        std::fs::File::open(snapshot.join(&tensor.shard))?.read_exact_at(&mut bytes, tensor.meta.byte_offset)?;
        let grid: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let k_blocks = tensor.meta.shape[1];
        for ranks in [1usize, 2] {
            let share = cfg.head_split(ranks)?;
            let layout = FusedQkvLayout::new(&share, attention, setup.checkpoint_tp / ranks)?;
            let (width, segments) = layout.program_segments(&share)?;
            let _ = MimoAttention::Full;
            for rank in 0..ranks {
                let scale0 = rank * layout.scale_rows();
                // The pre-push host expansion.
                let mut row_scales = vec![0f32; width * k_blocks];
                let mut index = vec![u32::MAX; width];
                for segment in &segments {
                    for r in 0..segment.rows {
                        let source = scale0 + segment.scale_row_of(r);
                        row_scales[(segment.dest_row + r) * k_blocks..][..k_blocks]
                            .copy_from_slice(&grid[source * k_blocks..][..k_blocks]);
                        index[segment.dest_row + r] = source as u32;
                    }
                }
                let mut host_k = vec![0f32; width * k_blocks];
                for b in 0..k_blocks { for r in 0..width { host_k[b * width + r] = row_scales[r * k_blocks + b]; } }
                let (mut row, mut kmaj) = (vec![0f32; width * k_blocks], vec![0f32; width * k_blocks]);
                // SAFETY: host buffers of the shapes the CPU twin reads and writes.
                unsafe { library.load_scale_gather(grid.as_ptr().cast(), index.as_ptr().cast(), row.as_mut_ptr().cast(),
                    kmaj.as_mut_ptr().cast(), width, k_blocks, k_blocks, 0, std::ptr::null_mut())?; }
                anyhow::ensure!(row.iter().map(|v| v.to_bits()).eq(row_scales.iter().map(|v| v.to_bits()))
                    && kmaj.iter().map(|v| v.to_bits()).eq(host_k.iter().map(|v| v.to_bits())),
                    "layer {layer} ranks {ranks} rank {rank}: gathered scales differ from the host expansion");
            }
        }
        eprintln!("layer {layer}: gathered qkv scales match the host expansion on one and two GPUs");
    }
    Ok(())
}

/// Exercise every target operand selected by the Flash max card before any
/// hardware turn. Extent-only allocations avoid reading or materializing weights.
#[test]
#[ignore = "requires CUTEAFD_PUSH_MIMO Flash snapshot and an NVMe fixture directory"]
fn installed_flash_checkpoint_formats_plan_all_dense_operands() -> Result<()> {
    use crate::shared::experts::rtx::dense::{self, DenseLoad};
    use crate::shared::token_io::{EmbedPlacement, EmbedSource};
    let snapshot = snapshot().context("set CUTEAFD_PUSH_MIMO to the Flash MOPD snapshot")?;
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    for device in 0..2 { fixture.configure_memory(&library, device, 512 << 30)?; }
    let cfg = MimoV2Config::read(&snapshot)?;
    let checkpoint = cuteafd_loader::plan::checkpoint::Checkpoint::coordinator(&snapshot, false, true)?;
    let formats = (0..cfg.layers).map(|layer|
        (format!("model.layers.{layer}.self_attn.o_proj.weight"), MimoProjectionRepresentation::Bf16)).collect();
    let setup = Setup { cfg, checkpoint, formats,
        checkpoint_tp: cuteafd_loader::families::mimo_v2::checkpoint_tp(&snapshot)? };
    let tensor = setup.checkpoint.tensor_names().find(|&name| name == "model.embed_tokens.weight")
        .and_then(|name| setup.checkpoint.tensors.iter().find(|tensor| tensor.meta.name == name))
        .context("Flash embedding headers")?;
    let source = EmbedSource::new(&snapshot, &tensor.shard, &tensor.meta, setup.cfg.hidden)?;
    let load = std::rc::Rc::new(DenseLoad::push(&library));
    let planning = MimoLoader { dense: load.clone(), fp8_head: false, ..loader(&library, &setup, 1) };
    let (_owners, mut records) = dense::recording(||
        planning.with_embedding(&setup.cfg, setup.cfg.layers, 0, source, EmbedPlacement::Gpu))?;
    records.extend(load.sink.as_ref().context("push sink")?.take_all());
    assert_eq!(records.operands.iter().filter(|operand| operand.lands.iter()
        .any(|(rect, _)| rect.name.ends_with(".self_attn.o_proj.weight"))).count(), setup.cfg.layers * 2);
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let (plans, _arenas) = dense::plan_records(&snapshot, records, &devices, 4)?;
    assert_eq!(plans.len(), 2);
    eprintln!("Flash max: all {} target layers, BF16 head and GPU embedding planned for 4 Sparks + 2 RTX", setup.cfg.layers);
    Ok(())
}

/// The whole MiMo V2.6 Pro max push group's plan from installed headers, for
/// offline replay of SparkNest's grant schedule (CUTEAFD_PUSH_PLAN_DUMP: the
/// JSON path, `{plan, file_sizes, arenas}`): six Spark ranks with their TP6
/// exact native FP8 slices of layers 12..=69 (the max card's admission),
/// then the two coordinator GPU ranks' dense records for all layers (FP8
/// head and o_proj, recorded through the native fixture, no GPU).
#[test]
#[ignore = "requires CUTEAFD_PUSH_MIMO, CUTEAFD_PUSH_PLAN_DUMP and an NVMe fixture directory"]
fn dump_max_group_plan_from_installed_headers() -> Result<()> {
    use crate::shared::experts::rtx::dense::{plan_records, DenseLoad};
    use crate::shared::memory::push_fp8::RankPlan;
    let Some(snapshot) = snapshot() else { return Ok(()) };
    let out = std::path::PathBuf::from(std::env::var_os("CUTEAFD_PUSH_PLAN_DUMP").context("set CUTEAFD_PUSH_PLAN_DUMP")?);
    let (sparks, world, spark_layers) = (6usize, 8usize, 12usize..=69);
    let (root, _) = cuteafd_loader::push_load::namespace(&snapshot)?;
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let tensors = catalog.fp8().context("native expert headers")?;
    // Serving's slicing: exact native-unit slices (the max card's residents,
    // 74,544,316,416 B on rank 0, match these and not the padded layout).
    let layers: Vec<_> = spark_layers.map(|layer| Ok((layer, tensors.for_layer_metadata(layer)?.exact_slicing(sparks))))
        .collect::<Result<_>>()?;
    let mut plans = Vec::new();
    for rank in 0..sparks {
        let admitted = layers.iter().try_fold(0usize, |total, &(layer, slicing)| -> Result<usize> {
            Ok(total + tensors.for_layer_metadata(layer)?.layer_load(layer, sparks, rank, slicing)?.resident)
        })?;
        let plan = RankPlan::new(&catalog, &layers, sparks, rank, admitted)?;
        eprintln!("Spark rank {rank}: {} resident bytes", plan.layout.bytes());
        plans.push(plan);
    }
    // The coordinator's dense records, as serving records them before membership.
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..2 { fixture.configure_memory(&library, device, 512 << 30)?; }
    let cfg = MimoV2Config::read(&snapshot)?;
    let checkpoint = cuteafd_loader::plan::checkpoint::Checkpoint::coordinator(&snapshot, false, true)?;
    let formats: BTreeMap<String, MimoProjectionRepresentation> = (0..cfg.layers)
        .map(|l| (format!("model.layers.{l}.self_attn.o_proj.weight"), MimoProjectionRepresentation::Fp8)).collect();
    let setup = Setup { cfg, checkpoint, formats, checkpoint_tp: cuteafd_loader::families::mimo_v2::checkpoint_tp(&snapshot)? };
    let load = std::rc::Rc::new(DenseLoad::push(&library));
    let planning = MimoLoader { dense: load.clone(), ..loader(&library, &setup, 1) };
    let model = planning.model(&setup.cfg, setup.cfg.layers)?;
    let records = load.sink.as_ref().context("push sink")?.take_all();
    let (landed, operands, staged, steps, largest, finishes) = records.summary();
    eprintln!("coordinator records: landed {landed} B in {operands} operands; staged {staged} B in {steps} steps \
        (largest {largest} B); {finishes} finishes");
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let (dense, _arenas) = plan_records(&snapshot, records, &devices, sparks)?;
    let mut arenas: Vec<u64> = plans.iter().map(|plan| plan.layout.logical_bytes() as u64).collect();
    arenas.extend(dense.iter().map(|plan| plan.base + plan.host_bytes));
    let mut builder = cuteafd_loader::push_load::builder::Builder::new(&root, arenas);
    for (rank, plan) in plans.iter().enumerate() {
        plan.append(&mut builder, &catalog, u32::try_from(rank)?, sparks)?;
    }
    // No coordinator TP2: each GPU's staged entries start at its device base.
    for (gpu, plan) in dense.iter().enumerate() {
        (plan.append)(&mut builder, u32::try_from(sparks + gpu)?, plan.base)?;
    }
    let checked = builder.finish()?;
    anyhow::ensure!(checked.arenas.len() == world, "eight ranks");
    for rank in 0..world {
        let entries = checked.plan.entries.iter().filter(|e| e.rank == rank as u32).count();
        eprintln!("rank {rank}: {} B in {entries} entries", checked.totals.rank_bytes[rank]);
    }
    let file = std::io::BufWriter::new(std::fs::File::create(&out)?);
    serde_json::to_writer(file, &serde_json::json!({"plan": checked.plan, "file_sizes": checked.file_sizes,
        "arenas": checked.arenas}))?;
    eprintln!("wrote {} ({} files, {} patterns, {} entries)", out.display(), checked.plan.files.len(),
        checked.plan.patterns.len(), checked.plan.entries.len());
    drop(model);
    Ok(())
}
