//! Native TP1 startup for Inkling 2 Mini over the same shared solver as `cuteafd plan`.
//!
//! Reports per-phase readiness timing (`seconds` and `s / resident GB`), loads target +
//! NVFP4/MXFP8 routed experts + optional drafter + optional PE-Lang vision tower via one
//! standalone GPU-streaming push transaction (or mount fallback), reconciles every backing
//! against solved admission, and attaches `#2697` soft-cap exact-row decode graphs.
use super::layers::MiniKernels;
use crate::shared::{
    inventory::{InventoryPhase, RuntimeInventory},
    prefix::{budget::HostBudget, PrefixArgs},
};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{chat::inkling2::InklingEncoding, ModelEncoding};
use cuteafd_ffi::{programs::Programs, NativeLibrary};
use cuteafd_loader::{
    families::inkling2::draft::Selection,
    families::inkling2_mini::{
        admission::{gather_with_weights, Coordinator, FactsRequest, SelectedWeights},
        execution::DECODE_ROWS,
        Inkling2MiniConfig, InklingKvCache,
    },
    placement::{
        families::inkling2_mini::{self as shared, AdmittedMini, LocalPackage, MiniInputs},
        planner::PrefillShape,
        ResidentWeights,
    },
    plan::Checkpoint,
};
use std::{path::Path, sync::Arc};

#[derive(Clone, Copy)]
pub(crate) struct Options<'a> {
    pub device: i32,
    pub manifest: &'a Path,
    pub native_lib: &'a Path,
    pub prefill: PrefillShape,
    pub decode_rows: u32,
    pub sequences: u64,
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
    pub kv_format: InklingKvCache,
    pub drafter: Option<cuteafd_loader::families::qwen4::Qwen4KvCache>,
    pub dflash: Option<(&'a Path, cuteafd_loader::families::qwen4::Qwen4KvCache)>,
    pub prefix: &'a PrefixArgs,
    pub resident_weights: ResidentWeights,
    pub expert_usage: bool,
    pub prewarm_graphs: bool,
    pub vision: &'a crate::shared::vision::still::Options,
}

pub(crate) struct Prepared<'a> {
    pub kernels: MiniKernels<'a>,
    pub encoding: ModelEncoding,
    pub eos: Vec<u32>,
    pub inputs: MiniInputs,
    pub admitted: AdmittedMini,
    pub runtime: RuntimeInventory,
    pub module_pins: Vec<cuteafd_ffi::fp8_moe::Fp8MoeModule>,
}

fn validate(options: &Options<'_>, cfg: &Inkling2MiniConfig) -> Result<()> {
    ensure!(options.device >= 0, "Inkling 2 Mini requires a valid GPU device");
    cuteafd_core::concurrency::validate("inkling2_mini", options.sequences)?;
    let ring = cuteafd_loader::families::inkling2::execution::SequenceLayout::new(&cfg.text, true, 1)?.ring_rows;
    cuteafd_loader::families::inkling2::verify::verify_ring(cfg.text.window, ring, DECODE_ROWS)?;
    ensure!(
        options.context_tokens == 0 || options.context_tokens <= cfg.text.max_position as u64,
        "Inkling 2 Mini context exceeds checkpoint"
    );
    ensure!(
        options.prefill.rows > 0
            && options.prefill.rows <= 4096
            && options.prefill.rows % 64 == 0
            && (1..=3).contains(&options.prefill.waves),
        "Inkling 2 Mini prefill differs from shared export/wave capacity"
    );
    ensure!(
        (1..=shared::MAX_DECODE_ROWS).contains(&options.decode_rows),
        "Inkling 2 Mini decode_rows {} exceeds supported 1..={}",
        options.decode_rows,
        shared::MAX_DECODE_ROWS
    );
    ensure!(
        options.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
        "Inkling 2 Mini requires exact captured 48-slot SWA/conv frontiers"
    );
    Ok(())
}

pub(crate) fn prepare<'a>(
    library: &'a NativeLibrary,
    programs: &'a Programs<'a>,
    checkpoint: &Checkpoint,
    cfg: &Inkling2MiniConfig,
    options: Options<'_>,
) -> Result<Prepared<'a>> {
    validate(&options, cfg)?;
    cfg.release_geometry()?;
    let encoding = ModelEncoding::Inkling(Arc::new(InklingEncoding::from_snapshot(&checkpoint.snapshot)?));
    let eos = encoding.eos_token_ids();
    ensure!(eos == [200_006], "Inkling 2 Mini EOS differs from native marker");
    let catalog = match &checkpoint.view {
        Some(view) => cuteafd_loader::read_expert_catalog_view(view)?,
        None => cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?,
    };
    let manifest = Arc::new(serde_json::from_slice(&std::fs::read(options.manifest)?)?);
    let selected_weights = SelectedWeights::new(checkpoint, cfg, options.drafter.is_some())?;
    let mut kernels = None;
    let mut module_pins = Vec::new();
    let replay_cap = cuteafd_loader::families::inkling2_mini::execution::replay_capacity(options.decode_rows as usize);
    let fir_commit = if replay_cap == 64 { "fir_history_commit" } else { "fir_history_commit_m128" };
    let devices = [options.device];
    let runtime = RuntimeInventory::measure(library, &devices, InventoryPhase::PreWeight, |_| {
        programs.program("inkling2_padded_transpose_bf16_r16_c1024_p64", &["source", "output"])?;
        kernels = Some(MiniKernels::bind_rows(programs, options.kv_format, options.decode_rows as usize)?);
        if options.drafter.is_some() || options.dflash.is_some() {
            programs.load_matching(|name| {
                name.starts_with("inkling_draft_") || name.starts_with("draft_bf16_") || name == fir_commit
            })?;
        }
        if let Some(tensors) = catalog.fp8() {
            let layers: Vec<usize> = (0..cfg.text.layers).collect();
            for (view, _) in tensors.homogeneous_layers(&layers)? {
                let directory = crate::shared::experts::fp8::package_directory(options.native_lib, 1, view.format());
                // SAFETY: trusted installed static expert package; retained across sampling.
                module_pins.push(unsafe { cuteafd_ffi::fp8_moe::Fp8MoeModule::load(&directory)? });
            }
        }
        Ok(())
    })?;
    let kernels = kernels.context("Mini kernels not initialized")?;
    let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
    let sample = &runtime.gpus[0];
    let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    let host_cache_bytes = match options.prefix.host_cache_bytes {
        HostBudget::Auto => None,
        HostBudget::Bytes(bytes) => Some(bytes),
    };
    let request = FactsRequest {
        vision: options.vision.vision,
        vision_replicas: options.vision.vision_replicas,
        media_cache_bytes: options.vision.media_cache_bytes.map(|b| b.0 as u64),
        checkpoint,
        catalog: &catalog,
        cfg,
        kv_format: options.kv_format,
        drafter: options.drafter.map(|kv_cache| Selection {
            kv_cache,
            sms: library.sm_count().unwrap_or(1) as usize,
        }),
        dflash: options.dflash.map(|(snapshot, kv_cache)| (snapshot, Selection { kv_cache, sms: library.sm_count().unwrap_or(1) as usize })),
        manifest_path: Some(options.manifest),
        manifest: Some(manifest),
        coordinator: Coordinator::Measured {
            sample: (sample.total_bytes, policy.gpu_cap_bytes(sample.total_bytes), sample.free_bytes),
            loaded_code: 0,
        },
        prefill: options.prefill,
        decode_rows: options.decode_rows,
        context_tokens: options.context_tokens,
        pool_tokens: options.pool_tokens,
        target_pool_tokens: None,
        sequences: options.sequences,
        host,
        host_cache_bytes,
        prefix_entries: options.prefix.prefix_cache_entries as u64,
        prefix_mark_budget: options.prefix.prefix_cache_mark_mib as u64 * (1 << 20),
        head: crate::shared::target_head::TargetHead::configured().blas_head(true),
        expert_usage: options.expert_usage,
        resident_weight_limit: options.resident_weights,
    };
    let inputs = gather_with_weights(&request, selected_weights)?;
    let admitted = inputs.admit(shared::request(
        inputs.context,
        options.pool_tokens,
        inputs.routed.len(),
        options.prefill,
        cuteafd_loader::placement::planner::Operation::Serving,
        inputs.resident_weight_limit,
    ))?;
    let sampled_free = [sample.free_bytes];
    runtime.record("inkling2_mini", &sampled_free);
    cuteafd_bench::context::ResolvedSettings::placement(&admitted.placement).publish();
    cuteafd_bench::context::set_resolved("max-context-tokens", &admitted.solved.served_context().to_string());
    cuteafd_bench::context::set_resolved("decode-rows", &options.decode_rows.to_string());
    Ok(Prepared {
        kernels,
        encoding,
        eos,
        inputs,
        admitted,
        runtime,
        module_pins,
    })
}

pub(crate) fn whole<'a>(
    library: &'a Arc<NativeLibrary>,
    programs: &'a Programs<'a>,
    view: &cuteafd_loader::plan::checkpoint::CheckpointView,
    checkpoint: &Checkpoint,
    cfg: &Inkling2MiniConfig,
    options: Options<'_>,
) -> Result<(
    super::engine::Execution<'a>,
    MiniInputs,
    AdmittedMini,
    ModelEncoding,
    crate::shared::vision::still::Ready,
)> {
    use crate::shared::memory::device::{Device, Stream};
    let total_start = std::time::Instant::now();
    let phase = crate::shared::memory::load_timing::Split::start();
    let t_admit = std::time::Instant::now();
    let prepared = prepare(library, programs, checkpoint, cfg, options)?;
    let admit_secs = t_admit.elapsed().as_secs_f64();
    phase.log("inkling2_mini code and admission");
    cuteafd_bench::context::phase("Inkling 2 Mini code and admission");

    let phase = crate::shared::memory::load_timing::Split::start();
    let t_load = std::time::Instant::now();
    let physical = [options.device];
    let mut media = crate::shared::vision::still::Ready::load(
        options.vision,
        &checkpoint.snapshot,
        options.native_lib,
        library,
        &physical,
        prepared
            .inputs
            .vision
            .as_ref()
            .and_then(|v| v.admitted(&prepared.admitted.solved)),
        cuteafd_loader::media::ImageFamily::Inkling2,
        prepared.inputs.vision.as_ref().map_or(0, |v| v.media_cache),
    )?;
    let catalog = cuteafd_loader::read_expert_catalog_view(view)?;
    let device = Device {
        library,
        id: options.device,
    };
    let (target, pushed, draft_weights) = super::weights::deliver_alone(
        device,
        view,
        checkpoint,
        &catalog,
        &prepared.inputs,
        &prepared.admitted,
        std::mem::take(&mut media.push),
    )?;
    let load_secs = t_load.elapsed().as_secs_f64();
    phase.log("inkling2_mini weight delivery");
    cuteafd_bench::context::phase("Inkling 2 Mini weight delivery");

    let phase = crate::shared::memory::load_timing::Split::start();
    let t_bind = std::time::Instant::now();
    let Prepared {
        kernels,
        encoding,
        inputs,
        admitted,
        module_pins,
        ..
    } = prepared;
    let stream = std::rc::Rc::new(Stream::new(device)?);
    let requests = super::owners::RequestOwners::new(stream.clone(), &inputs, &admitted)?;
    let LocalPackage::Installed { backings } = &inputs.local else {
        anyhow::bail!("Inkling 2 Mini admitted resident expert backings absent");
    };
    let decode_rows = inputs.decode_rows as usize;
    let body_rows = usize::try_from(inputs.prefill.rows.max(u64::from(inputs.decode_rows)))?;
    let head = super::head::MiniHead::new(stream.clone(), cfg, decode_rows)?;
    let local = super::experts::serving_dispatchers(
        stream.clone(),
        &catalog,
        options.native_lib,
        backings,
        body_rows,
        pushed,
    )?;
    let mut execution = super::engine::Execution::new(requests, head, target, kernels, local, cfg.clone())?;
    if let Some(draft_weights) = draft_weights {
        let facts = inputs
            .drafter
            .as_ref()
            .context("delivered Mini drafter has no admission facts")?;
        let mut layout = facts.layout.clone();
        let rank = &mut layout.runtime[0];
        *rank = cuteafd_loader::formats::block_dspark::recurrent::runtime::RecurrentRuntimeLayout::new(
            &layout.config,
            rank.slots,
            rank.sequences,
            rank.context_rows,
            usize::try_from(inputs.context.maximum)?,
            usize::try_from(admitted.solved.pool_tokens)?,
            library.sm_count()? as usize,
            rank.kv_cache,
            0,
            1,
            0,
        )?;
        let report = crate::shared::draft::report::admitted(
            crate::shared::draft::report::Kind::Dspark,
            crate::shared::draft::report::Placement::Whole,
            0,
            &physical,
            &admitted.solved,
        )?;
        crate::shared::draft::report::publish(report);
        let unembed = execution.target.operand("model.llm.unembed.weight")?.buffer;
        execution.draft = Some(std::rc::Rc::new(super::draft::Binding::new(
            programs,
            stream.clone(),
            draft_weights,
            layout,
            &admitted.solved,
            unembed,
            usize::try_from(inputs.prefill.rows)?,
            usize::try_from(inputs.prefill.waves)?,
            decode_rows,
            inputs.prefix_mark_slots,
        )?));
    }

    if let Some(layout) = &inputs.dflash {
        let embed = execution.target.operand("model.llm.embed.weight")?.buffer;
        execution.draft = Some(std::rc::Rc::new(super::draft::Binding::new_dflash(
            programs, stream.clone(), layout, &admitted.solved, embed, cfg.text.unpadded_vocab_size, cfg.text.rms_norm_eps as f32,
            usize::try_from(inputs.prefill.rows)?, usize::try_from(inputs.prefill.waves)?, decode_rows, inputs.prefix_mark_slots)?));
        let report = crate::shared::draft::report::admitted(crate::shared::draft::report::Kind::Dflash2,
            crate::shared::draft::report::Placement::Whole, 0, &physical, &admitted.solved)?;
        crate::shared::draft::report::publish(report);
    }

    let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical)?;
    let mut weights_inv = crate::shared::memory::receipts::Inventory::new(
        devices,
        "inkling_mini_target",
        cuteafd_loader::placement::planner::Lifetime::RESIDENT,
        cuteafd_core::memory_layout::Category::Weights,
        "Inkling 2 Mini target",
    );
    execution.target.inventory(&mut weights_inv, "lead")?;
    for receipt in weights_inv.receipts.iter().filter(|r| r.allocation.view.is_none()) {
        let actual = &receipt.allocation;
        let planned = admitted
            .solved
            .facts
            .allocations
            .iter()
            .find(|owner| owner.id == actual.id)
            .context("Mini target arena backing not admitted")?;
        ensure!(
            actual.bytes == planned.bytes && actual.location == planned.location,
            "Mini target arena backing differs: {}",
            actual.id
        );
    }
    ensure!(
        weights_inv.receipts.iter().filter(|r| r.allocation.view.is_none()).count() == 1,
        "Inkling 2 Mini must own exactly one target arena"
    );

    let mut inventory = crate::shared::memory::receipts::Inventory::new(
        devices,
        "retained",
        cuteafd_loader::placement::planner::Lifetime::TARGET,
        cuteafd_core::memory_layout::Category::Workspace,
        "Inkling 2 Mini requests",
    );
    let mut census = crate::shared::memory::receipts::Census::default();
    execution.requests.allocation_receipts(&mut inventory, &mut census, devices)?;
    execution.head.inventory(&mut inventory, "inkling2_mini.head")?;
    super::admission::reconcile_requests(&admitted.solved, &inputs, &inventory.receipts)?;

    let graph_keys = cuteafd_loader::serving_capacity::inkling2_mini_graphs::KeySpace::for_config(
        inputs.sequences,
        decode_rows,
        inputs.drafter.is_some() || inputs.dflash.is_some(),
    );
    let graph_policy = cuteafd_loader::serving_capacity::inkling2_mini_graphs::policy(0, graph_keys)?;
    ensure!(
        inputs.graph_reserve == graph_policy.bytes,
        "Mini graph cache disagrees with admission"
    );
    execution.graphs = super::graphs::DecodeGraphs::new(&execution.requests.decode.stream, graph_policy)?;
    drop(module_pins);
    let bind_secs = t_bind.elapsed().as_secs_f64();
    phase.log("inkling2_mini request owners and graph attachment");
    cuteafd_bench::context::phase("Inkling 2 Mini request owners and graph attachment");

    let phase = crate::shared::memory::load_timing::Split::start();
    let t_prewarm = std::time::Instant::now();
    let (prewarm_captured, prewarm_wanted) = if options.prewarm_graphs {
        execution.prewarm_graphs(usize::try_from(options.sequences)?)?
    } else {
        (0, 0)
    };
    let prewarm_secs = t_prewarm.elapsed().as_secs_f64();
    if options.prewarm_graphs {
        phase.log("inkling2_mini graph prewarm");
        cuteafd_bench::context::phase("Inkling 2 Mini graph prewarm");
    }

    let total_secs = total_start.elapsed().as_secs_f64();
    let resident_bytes: u64 = inputs.target.execution_arena()?.bytes() as u64
        + backings.iter().map(|b| b.arena.bytes() as u64).sum::<u64>()
        + inputs.drafter.as_ref().map_or(0, |d| d.layout.weights[0].arena.bytes() as u64)
        + inputs.dflash.as_ref().map_or(0, |d| d.arena.bytes() as u64);
    let resident_gb = resident_bytes as f64 / 1e9;
    tracing::info!(
        admit_secs = format_args!("{admit_secs:.3}"),
        weight_delivery_secs = format_args!("{load_secs:.3}"),
        bind_secs = format_args!("{bind_secs:.3}"),
        prewarm_secs = format_args!("{prewarm_secs:.3}"),
        prewarm_captured,
        prewarm_wanted,
        total_ready_secs = format_args!("{total_secs:.3}"),
        resident_gb = format_args!("{resident_gb:.2}"),
        secs_per_resident_gb = format_args!("{:.4}", total_secs / resident_gb.max(1e-9)),
        load_gb_per_sec = format_args!("{:.2}", resident_gb / load_secs.max(1e-9)),
        "Inkling 2 Mini readiness breakdown"
    );
    Ok((execution, inputs, admitted, encoding, media))
}
