//! Cold coordinator-only native prefix golden, with bounded layer residency.
use super::{admission, experts::{Dispatcher, Mxfp8, Mxfp8Layout, Resident}, head::Head, layers::{self, Kernels},
    owners::UnitOwner, weights::TargetWeights};
use crate::shared::{experts::{self, exl3::Exl3Weights, layer::ExpertLayer,
    rtx::{exl3::Exl3Tp2, RtxSelectionPolicy}}, memory::device::{Device, Stream}};
use cuteafd_loader::{OfficialV41Catalog, serving_capacity::WeightArenaLayout, shared::dense_read::MountLoad};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::fp8_moe::Fp8MoeMetadata;
use cuteafd_loader::{families::inkling2::{execution::UnitLayout, weight_layout::CoordinatorWeights, Inkling2Config}, plan::Checkpoint};
use std::{path::PathBuf, rc::Rc};

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)] pub attention: crate::cli::CoreViewArgs,
    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long, env="CUTEAFD_NATIVE_LIB")] pub native_lib: PathBuf,
    #[arg(long, default_value="/opt/cuteafd/share/PROGRAMS.json")] pub manifest: PathBuf,
    #[arg(long="kv-cache", default_value="fp8")] pub kv_format: cuteafd_loader::families::inkling2::InklingKvCache,
    /// JSON array of token ids; no tokenizer or prompt preprocessing in golden.
    #[arg(long)] pub tokens: PathBuf,
    /// Prefix stops after this zero-based layer; no full-model claim if <47.
    #[arg(long, default_value_t=0)] last_layer: usize,
    #[arg(long, default_value_t=0)] pub device: i32,
    #[arg(long, requires="width_invariance")] pub split_device: Option<i32>,
    /// Existing parent; files are created below this new directory.
    #[arg(long)] pub save_logits: PathBuf,
    #[arg(long)] nll: bool,
    /// Optional raw FP32 matching partial-prefix reference logits.
    #[arg(long)] golden: Option<PathBuf>,
    #[arg(long)] check_headers: bool,
    /// Diagnostic native hidden/routes/logits at selected zero-based layers.
    #[arg(long, value_delimiter=',')] tap_layers: Vec<usize>,
    /// Compare the actual served target's teacher-forced decode/verify widths.
    #[arg(long, value_delimiter=',')] pub width_invariance: Vec<usize>,
    /// Compare every accepted prefix's live KV/FIR state and next-cycle logits.
    #[arg(long, requires="width_invariance")] pub commit_invariance: bool,
    /// Compare repeated exact-width graph replays from the same prompt state.
    #[arg(long, requires="width_invariance", conflicts_with="commit_invariance")] pub graph_parity: bool,
    /// Run a 16-sequence x 8-token (128-row) graphed verify, fidelity and FIR commit oracle.
    #[arg(long, requires="width_invariance")] pub wide_batch: bool,
    #[arg(long, default_value_t=64)] pub decode_rows: u32,
    #[arg(long, default_value_t=64)] pub prefill: usize,
    #[arg(long, value_delimiter=',')] pub peers: Vec<std::net::SocketAddr>,
    #[arg(long, default_value="auto")] pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    #[arg(long)] pub pool_tokens: Option<u64>,
    #[arg(long, default_value_t=16)] pub max_sequences: usize,
    #[arg(long, default_value_t=4096)] pub prefill_rows: usize,
    #[arg(long, default_value_t=3)] pub prefill_waves: u64,
    #[arg(long, default_value="fp8")] pub draft_kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    #[command(flatten)] pub prefix: crate::shared::prefix::PrefixArgs,
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || run(args)).await?
}

fn raw(buffer: cuteafd_ffi::CuteafdDeviceBuffer, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
    ensure!(bytes <= buffer.bytes, "Inkling golden view exceeds admitted owner");
    Ok(cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..buffer })
}

enum ColdExperts {
    Mxfp8 { layout: Mxfp8Layout, scratch: usize },
    Exl3 { arena: WeightArenaLayout, scratch: Vec<(String, usize)>, mount: MountLoad },
}

impl ColdExperts {
    fn facts(&self) -> (&WeightArenaLayout, Vec<(String, usize)>, MountLoad) {
        match self {
            Self::Mxfp8 { layout, scratch } => (&layout.arena, vec![("expert.scratch".into(), *scratch)], layout.mount),
            Self::Exl3 { arena, scratch, mount } => (arena, scratch.iter().map(|(name, bytes)|
                (format!("experts.{name}"), *bytes)).collect(), *mount),
        }
    }
    fn load<'a>(&self, stream: Rc<Stream<'a>>, catalog: &OfficialV41Catalog, metadata: Option<&Fp8MoeMetadata>,
        package: &std::path::Path, cfg: &Inkling2Config, layer: usize, rows: usize) -> Result<ColdDispatch<'a>> {
        let device = stream.device;
        match self {
            Self::Mxfp8 { layout, .. } => Ok(ColdDispatch::Mxfp8(device.run(|| Mxfp8::load(stream,
                layout, metadata.context("Inkling MXFP8 metadata missing")?, package, cfg, rows))?)),
            Self::Exl3 { arena, scratch, .. } => {
                let budget = arena.bytes() + scratch.iter().map(|(_, bytes)| bytes).sum::<usize>();
                let backend = Exl3Tp2::load_whole_weights(device, catalog, package, vec![layer], rows, budget,
                    RtxSelectionPolicy::CanonicalDecode)?.execution()?;
                Ok(ColdDispatch::Exl3(Resident::new(layer, Box::new(backend), stream)?))
            }
        }
    }
}

enum ColdDispatch<'a> { Mxfp8(Mxfp8<'a>), Exl3(Resident<'a>) }
impl<'a> ColdDispatch<'a> {
    fn dispatcher(&self) -> &dyn Dispatcher {
        match self { Self::Mxfp8(experts) => experts, Self::Exl3(experts) => experts }
    }
    fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        match self {
            Self::Mxfp8(experts) => experts.inventory(inventory, "experts"),
            Self::Exl3(experts) => {
                experts.inventory(inventory, "experts")?;
                // The shared backend returns both classes. Its arena already
                // carries its allocation tag, so retain only resident receipts.
                let arena = inventory.receipts.iter().find(|r| r.allocation.group == "weight arena")
                    .context("Inkling EXL3 arena receipt missing")?.allocation.id.clone();
                inventory.receipts.retain(|r| r.allocation.id == arena ||
                    r.allocation.view.as_ref().is_some_and(|view| view.backing == arena));
                Ok(())
            }
        }
    }
    fn workspace_inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        match self {
            Self::Mxfp8(experts) => experts.workspace_inventory(inventory),
            Self::Exl3(experts) => {
                let first = inventory.receipts.len();
                experts.inventory(inventory, "experts")?;
                let mut expert = inventory.receipts.split_off(first);
                expert.retain(|r| r.allocation.group != "weight arena" && r.allocation.view.is_none());
                inventory.receipts.extend(expert);
                Ok(())
            }
        }
    }
}

fn run(mut args: GoldenArgs) -> Result<()> {
    if !args.width_invariance.is_empty(){return super::width_invariance::run(&mut args);}
    ensure!(args.nll, "Inkling cold golden requires --nll");
    let view = args.attention.selected_view.clone().map(Ok).unwrap_or_else(|| args.attention.select(&args.snapshot))?;
    args.attention.validate_view(&view)?;
    let checkpoint = Checkpoint::open_view(&view)?;
    let cfg = Inkling2Config::from_hf(&checkpoint.config)?;
    ensure!(args.last_layer < cfg.layers, "Inkling --last-layer outside checkpoint");
    let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(&args.tokens)?)?;
    ensure!((2..=4096).contains(&tokens.len()) && tokens.iter().all(|&t| (t as usize)<cfg.unpadded_vocab_size),
        "Inkling golden needs2..4096 valid token ids");
    ensure!(args.tap_layers.iter().all(|&layer| layer<=args.last_layer), "Inkling tap layer exceeds run depth");
    let rows = tokens.len()-1;
    let unit_layout = UnitLayout::with_format(&cfg, rows, tokens.len().max(64), args.kv_format)?;
    let target_layout = CoordinatorWeights::golden(&checkpoint, &cfg, &(0..=args.last_layer).collect::<Vec<_>>())?;
    let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
    let geometry = catalog.routed_experts().geometry()?;
    ensure!(geometry == cuteafd_core::ExpertGeometry::INKLING2, "Inkling golden expert geometry differs from the released model");
    cuteafd_core::set_expert_geometry(geometry).map_err(|previous| anyhow::anyhow!("Inkling golden requires isolated expert geometry: {previous:?}"))?;
    let native_dir = args.native_lib.parent().context("Inkling native library directory")?;
    let package = if let Some(exl3) = catalog.exl3() {
        native_dir.join("exl3").join(exl3.package_name("inkling2")).join("rtx-tp1")
    } else { native_dir.join("fp8/fp8-inkling2-mxfp8/tp1") };
    // SAFETY: trusted generated package ABI metadata; no live buffer access.
    let metadata = if catalog.exl3().is_none() { Some(unsafe { Fp8MoeMetadata::read(&package)? }) } else { None };
    let selection = RtxSelectionPolicy::CanonicalDecode;
    let expert_layouts = (0..=args.last_layer).map(|index| {
        if let Some(metadata) = &metadata {
            let layout = Mxfp8Layout::plan(&checkpoint, &cfg, index)?;
            Ok(ColdExperts::Mxfp8 { layout, scratch: metadata.scratch_for(rows)? })
        } else {
            let layer = ExpertLayer::BackboneFull { layer: index };
            let arena = experts::weight_layout::layout(args.device, "exl3", experts::exl3::regions(&catalog, [layer])?)?;
            let load = Exl3Weights::plan(&catalog, layer)?;
            ensure!(load.device_staging_bytes == 0, "Inkling cold EXL3 requires zero device load staging");
            let scratch = Exl3Tp2::whole_allocation_layout(&package, rows, geometry, selection)?;
            Ok(ColdExperts::Exl3 { arena, scratch, mount: MountLoad { pinned_source: load.pinned_host_bytes as u64,
                full_row_cache: load.read_scratch_bytes as u64, pinned_tables: 0 } })
        }
    }).collect::<Result<Vec<_>>>()?;
    ensure!(!args.save_logits.exists(), "Inkling golden output must be new");
    if args.check_headers {
        println!("Inkling golden headers: layers={} rows={} core={} nativeMXFP8={} partial={}", args.last_layer+1, rows,
            view.mode(), metadata.is_some(), args.last_layer+1 != cfg.layers);
        return Ok(());
    }
    // SAFETY: library outlives every loaded program, stream and physical owner.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load_for_experts(&args.native_lib, geometry) }?;
    let device = Device { library: &library, id: args.device };
    let programs = device.run(|| Ok(library.programs()?.with_manifest(&args.manifest)?))?;
    let mut kernels = device.run(|| Kernels::bind(&programs, args.kv_format))?;
    // Load expert code before taking the pre-weight CUDA sample.
    let _expert_code = if metadata.is_some() {
        // SAFETY: trusted generated package on the retained current device.
        Some(device.run(|| unsafe { cuteafd_ffi::fp8_moe::Fp8MoeModule::load(&package) })?)
    } else { None };
    let _exl3_code = if metadata.is_none() {
        Some(Exl3Tp2::preload_whole_modules(device, &package, rows, geometry, selection)?)
    } else { None };
    let manifest = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
    let mut compact_programs = target_layout.trellis_programs(&manifest, rows, rows)?;
    let fp8_programs = target_layout.fp8_programs(&manifest, rows, rows)?;
    compact_programs.extend(fp8_programs.keys().cloned());
    let (free, total) = device.run(|| {
        let (loaded, _) = programs.load_matching(|name| compact_programs.contains(name))?;
        ensure!(loaded == compact_programs.len(), "Inkling golden compact program table differs from manifest");
        library.cuda_memory_info()
    })?;
    let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    let fp8_bytes = fp8_programs.into_values().max().unwrap_or(0);
    let (expert_arena, expert_scratch, expert_mount) = expert_layouts[0].facts();
    let plan = admission::admit(&checkpoint, &cfg, &target_layout, expert_arena, &expert_scratch, expert_mount,
        &unit_layout, rows, fp8_bytes, free as u64, total as u64, host)?;
    if !args.tap_layers.is_empty() { std::fs::create_dir(&args.save_logits)?; }
    let stream = Rc::new(Stream::new(device)?);
    let owner = UnitOwner::new(Rc::clone(&stream), unit_layout)?;
    let weights = TargetWeights::load(device, &checkpoint, &target_layout)?;
    kernels.bind_compact(&programs, Rc::clone(&stream), &target_layout, rows, rows, fp8_bytes)?;
    let head = Head::new(Rc::clone(&stream), &cfg, rows, target_layout.stored_head())?;
    let positions = (0..rows).flat_map(|n| (n as i64).to_le_bytes()).collect::<Vec<_>>();
    let rings = (0..rows).flat_map(|n| ((n%1088) as i64).to_le_bytes()).collect::<Vec<_>>();
    let pages = (0..owner.layout.buffers["page_table"]/4).flat_map(|n| (n as i32).to_le_bytes()).collect::<Vec<_>>();
    stream.drain()?;
    device.run(|| {
        let tile_rows = (0..rows).step_by(10)
            .flat_map(|row| (row as i32).to_le_bytes()).collect::<Vec<_>>();
        library.copy_h2d(owner.buffer("tile_rows")?, &tile_rows)?;
        library.copy_h2d(owner.buffer("positions")?, &positions)?;
        library.copy_h2d(owner.buffer("cache_slots")?, &positions)?;
        library.copy_h2d(owner.buffer("ring_slots")?, &rings)?;
        library.copy_h2d(owner.buffer("page_table")?, &pages)
    })?;
    head.embed(&kernels, &owner, &weights, &cfg, &tokens[..rows])?;
    for (index, layout) in expert_layouts.iter().enumerate() {
        // Reuse one independently retired expert arena: no one-card fullmodel
        // fit requirement, and native expert precision is never changed.
        // Every layer owns separate causal histories; the cold whole-sequence
        // unit reuses storage only after the preceding layer has drained.
        device.run(|| {
            for name in ["attn_conv_history", "mlp_conv_history", "k_conv_history", "v_conv_history", "ring", "global_cache"] {
                let buffer = owner.buffer(name)?;
                // SAFETY: no queued prior-layer reader remains after drain.
                unsafe { library.cuda_zero_bytes_async(buffer, buffer.bytes, stream.raw)?; }
            }
            Ok(())
        })?;
        if args.tap_layers.contains(&index) {
            stream.drain()?;
            let out = args.save_logits.join(format!("layer-{index}"));
            std::fs::create_dir(&out)?;
            for (name, source) in [("input_partial", "partial"), ("input_arena", "residual_arena")] {
                let buffer = owner.buffer(source)?;
                let mut bytes = vec![0u8; buffer.bytes];
                stream.copy_d2h(&mut bytes, buffer)?;
                std::fs::write(out.join(format!("{name}.bin")), bytes)?;
            }
        }
        let (arena, scratch, mount) = layout.facts();
        let layer_plan = if index == 0 { None } else { Some(admission::admit(&checkpoint, &cfg, &target_layout,
            arena, &scratch, mount, &owner.layout, rows, fp8_bytes, free as u64, total as u64, host)?) };
        let plan = layer_plan.as_ref().unwrap_or(&plan);
        let experts = layout.load(Rc::clone(&stream), &catalog, metadata.as_ref(), &package, &cfg, index, rows)?;
        device.run(|| super::prefill::layer(&kernels, &owner, &weights, experts.dispatcher(), &cfg, index, rows))?;
        stream.drain()?;
        if args.tap_layers.contains(&index) {
            let out = args.save_logits.join(format!("layer-{index}"));
            for name in ["partial", "mixed", "projection_input", "route_ids", "route_weights", "route_logits"] {
                let buffer = owner.buffer(name)?;
                let mut bytes = vec![0u8; buffer.bytes];
                stream.copy_d2h(&mut bytes, buffer)?;
                std::fs::write(out.join(format!("{name}.bin")), bytes)?;
            }
            std::fs::write(out.join("meta.json"), serde_json::to_vec_pretty(&serde_json::json!({
                "rows":rows,"arena_rows":owner.layout.rows,"arena_candidates":owner.layout.blocks,
                "completed_blocks":index.div_ceil(cfg.res_block),"hidden":cfg.hidden,"experts":cfg.experts,
                "topk":cfg.topk,"local":cfg.local[index],"provenance":"exact native F32 input partial and residual arena, copied before layer"}))?)?;
        }
        let physical = [args.device];
        let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical)?;
        let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "inkling2",
            cuteafd_loader::placement::planner::Lifetime::RESIDENT,
            cuteafd_core::memory_layout::Category::Weights, "cold native weights");
        weights.inventory(&mut inventory, "target")?;
        let mut expert_inventory = crate::shared::memory::receipts::Inventory::new(devices, "inkling2",
            cuteafd_loader::placement::planner::Lifetime::RESIDENT,
            cuteafd_core::memory_layout::Category::Experts, "cold native experts");
        experts.inventory(&mut expert_inventory)?;
        inventory.receipts.extend(expert_inventory.receipts);
        let mut working = crate::shared::memory::receipts::Inventory::new(devices, "inkling2",
            cuteafd_loader::placement::planner::Lifetime::TARGET,
            cuteafd_core::memory_layout::Category::Workspace, "cold native workspaces");
        owner.inventory(&mut working, "unit")?;
        head.inventory(&mut working, "head")?;
        kernels.compact_inventory(&mut working, "target")?;
        experts.workspace_inventory(&mut working)?;
        let measured = inventory.receipts.iter().chain(&working.receipts)
            .filter(|r| r.allocation.view.is_none() && r.allocation.location == cuteafd_loader::placement::planner::Location::coordinator(0))
            .map(|r| r.allocation.bytes).sum::<u64>();
        let admitted = plan.facts.allocations.iter().filter(|a| a.view.is_none()
            && a.location == cuteafd_loader::placement::planner::Location::coordinator(0)
            && a.lifetime.contains(cuteafd_loader::placement::planner::Phase::PostTargetKv, plan.facts.phase_order)).map(|a| a.bytes).sum::<u64>();
        let receipts = inventory.receipts.iter().chain(&working.receipts).cloned().collect::<Vec<_>>();
        admission::reconcile_native(plan, devices, &receipts)?;
        ensure!(measured == admitted, "Inkling measured device owners differ from cold admission: {measured}/{admitted}");
        tracing::info!(layer=index, measured_device_owner_bytes=measured, admitted_device_owner_bytes=admitted,
            native_weight_arena_aliases=inventory.receipts.len(), "Inkling native cold owner census");
        println!("Inkling native layer{index} complete rows={rows}");
    }
    let candidates = super::layers::residual_site(&cfg, args.last_layer)?.mlp_candidates;
    let logits = head.project(&kernels, &owner, &weights, &cfg, rows, candidates)?;
    stream.drain()?;
    let mut bytes = vec![0u8; rows*cfg.unpadded_vocab_size*4];
    let download_bytes = bytes.len();
    stream.copy_d2h(&mut bytes, raw(logits, download_bytes)?)?;
    let values = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "Inkling golden nonfinite logits");
    let mut nll = 0f64;
    for (row, &next) in values.chunks_exact(cfg.unpadded_vocab_size).zip(&tokens[1..]) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        nll += maximum + row.iter().map(|&v| (v as f64-maximum).exp()).sum::<f64>().ln() - row[next as usize] as f64;
    }
    nll /= rows as f64;
    let comparison = if let Some(path) = &args.golden {
        let expected = std::fs::read(path)?;
        ensure!(expected.len() == bytes.len(), "Inkling partial reference logits shape differs");
        let mut square = 0f64;
        let mut norm = 0f64;
        let mut dot = 0f64;
        let mut actual_norm = 0f64;
        for (&actual, expected) in values.iter().zip(expected.chunks_exact(4)) {
            let expected = f32::from_le_bytes(expected.try_into().unwrap()) as f64;
            ensure!(expected.is_finite(), "Inkling reference nonfinite logits");
            square += (actual as f64-expected).powi(2); norm += expected*expected;
            dot += actual as f64*expected; actual_norm += (actual as f64).powi(2);
        }
        let rms = (square/norm.max(1e-30)).sqrt();
        let cosine = dot/(norm*actual_norm).sqrt().max(1e-30);
        ensure!(rms<=0.02 && cosine>=0.999, "Inkling partial logits fail RMS/cosine bar: {rms}/{cosine}");
        let mut reference_nll = 0f64;
        for (row, &next) in expected.chunks_exact(cfg.unpadded_vocab_size*4).zip(&tokens[1..]) {
            let row = row.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64).collect::<Vec<_>>();
            let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            reference_nll += maximum + row.iter().map(|&v| (v-maximum).exp()).sum::<f64>().ln() - row[next as usize];
        }
        reference_nll /= rows as f64;
        ensure!((nll-reference_nll).abs() <= 0.03, "Inkling partial NLL exceeds0.03 bar: {nll}/{reference_nll}");
        serde_json::json!({"normalized_rms":rms,"cosine":cosine,"reference_nll":reference_nll,
            "nll_delta":nll-reference_nll})
    } else { serde_json::Value::Null };
    if args.tap_layers.is_empty() { std::fs::create_dir(&args.save_logits)?; }
    std::fs::write(args.save_logits.join("logits.bin"), bytes)?;
    std::fs::write(args.save_logits.join("tokens.bin"), tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let report = serde_json::json!({"family":"inkling2","native":true,"partial":args.last_layer+1 != cfg.layers,
        "last_layer":args.last_layer,"rows":rows,"vocab":cfg.unpadded_vocab_size,"mean_nll":nll,
        "reference_comparison":comparison, "core_view":view.mode().name()});
    std::fs::write(args.save_logits.join("result.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("Inkling native golden mean NLL={nll:.8} partial={} rows={rows}", args.last_layer+1 != cfg.layers);
    Ok(())
}
