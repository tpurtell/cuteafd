//! Eager native target and default-on MTP over the shared scheduler.
use super::{admission::{self, RuntimeInputs}, backend::{Inputs, Step5Backend}, cache::CacheRank,
    execution::Execution, transaction::Staging};
use crate::shared::{inventory::{InventoryPhase, RuntimeInventory}, memory::device::Device,
    placement::{spark_rings, SparkAdmission}, prefix::PrefixArgs, qwen::scheduler::{self, Backend},
    spark_intake::SparkLink, token_io::{EmbedPlacement, EmbedSource, TokenEmbedding, TokenIoArgs}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::{placement::{Demand, GpuBudget, Inventory, ProgramSet, SparkWorkingSet},
    plan::{checkpoint::Checkpoint, families::step5::Config, spec::DraftOwnership}};
use cuteafd_transport::{expert::SparkWorld, worker_selection::ModelAdmission, TcpTransportConfig};
use std::{cell::RefCell, net::SocketAddr, path::PathBuf, sync::{Arc, Mutex}};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long)]
    split_device: Option<i32>,
    #[arg(long, value_delimiter = ',', required = true)]
    peers: Vec<SocketAddr>,
    #[arg(long)]
    pool_tokens: Option<u64>,
    /// Immutable target and native predictor KV storage.
    #[arg(long, default_value = "fp8")]
    kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    #[arg(long, default_value_t = 1_048_576)]
    max_context: u64,
    /// Prefill unit rows (PLAN "Prefill contract"); a prefill call holds
    /// `--prefill-waves` units; zero selects the shared topology shape.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=4096))]
    prefill_rows: u32,
    /// Prefill units in flight: Spark waves outstanding on the one transport.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=4))]
    prefill_waves: u8,
    /// Pack queued short prompts, optionally preserving singleton arithmetic routes.
    #[arg(long, value_enum, default_value = "unrestricted")]
    prefill_packing: crate::shared::prefill_packing::PackingMode,
    /// Routed layers whose EXL3 experts run whole on the entry RTX: auto lets the
    /// planner keep as many as fit beside the pool; FP8/NVFP4 releases onboard none.
    #[arg(long, default_value = "auto")]
    rtx_expert_layers: cuteafd_loader::placement::Onboard,
    /// Native predictor depth; 0 disables speculation.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(0..=3))]
    mtp: u8,
    #[arg(long, default_value = "0.0.0.0:8000")]
    listen: String,
    #[arg(long, default_value_t = 32768)]
    max_output: u32,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u8).range(1..=64))]
    max_sequences: u8,
    #[arg(long)]
    model_id: Option<String>,
    /// All prefill rows are always scored by this eager target.
    #[arg(long)]
    full_prefill_logits: bool,
    #[command(flatten)]
    prefix: PrefixArgs,
    #[command(flatten)]
    decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)]
    token_io: TokenIoArgs,
    #[command(flatten)]
    api: crate::shared::api::ApiArgs,
}

impl ServeArgs {
    fn endpoint(&self, bf16: bool) -> SparkEndpoint {
        SparkEndpoint { ring_depth: None, capacity: self.prefill_rows, ingress: if bf16 { SparkIngress::Bf16 } else { SparkIngress::Native } }
    }

    fn shape(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        Ok(prefill_shape(self.peers.len(), cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::step5::STEP5), PrefillOverrides {
            unit_rows: (self.prefill_rows != 0).then_some(u64::from(self.prefill_rows)),
            waves: (self.prefill_waves != 0).then_some(u64::from(self.prefill_waves)),
            span_rows: None, reserve_rows: None,
        })?)
    }

    fn validate(&self) -> Result<()> {
        let shape = self.shape()?;
        ensure!(self.device >= 0 && self.split_device.is_none_or(|d| d >= 0 && d != self.device),
            "Step-5 needs one GPU or two distinct actual GPUs");
        ensure!(matches!(self.peers.len(), 2 | 3 | 4 | 6), "Step-5 Spark experts run TP2, TP3, TP4 or TP6 (packages decide)");
        ensure!(self.peers.iter().enumerate().all(|(i, peer)| !self.peers[..i].contains(peer)),
            "Step-5 Spark peers must be distinct");
        ensure!((262_144..=1_048_576).contains(&self.max_context), "Step-5 request context must be 256K..=1M");
        ensure!(u64::from(self.max_sequences) <= shape.unit_rows,
            "Step-5 sequence capacity exceeds row capacity");
        ensure!(u64::from(self.mtp) < shape.unit_rows,
            "Step-5 predictor depth exceeds row capacity");
        ensure!(self.token_io.embed_placement == EmbedPlacement::Gpu,
            "Step-5 requires admitted GPU token embeddings");
        ensure!(self.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Step-5 supports exact prefix frontiers only");
        Ok(())
    }
}

/// The sealed package's exports, which the shared Spark admission covers. All
/// layouts must agree; worker admission independently validates loaded exports.
pub(crate) fn package_exports(manifest: &serde_json::Value) -> Result<Vec<u32>> {
    let layouts = manifest["layouts"].as_object().context("Step-5 package layouts missing")?;
    let mut exports = None;
    for layout in layouts.values() {
        ensure!(layout["hidden"] == 4096 && layout["experts"] == 352
            && layout["top_k"] == 8 && layout["intermediate"] == 1536,
            "Step-5 package geometry differs from checkpoint");
        let capacities = layout["capacities"].as_array().context("Step-5 package capacities missing")?
            .iter().map(|v| v["capacity"].as_u64().context("Step-5 package capacity must be integer")
                .and_then(|capacity| Ok(u32::try_from(capacity)?)))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!capacities.is_empty() && exports.as_ref().is_none_or(|known| known == &capacities),
            "Step-5 installed package extents disagree or are empty");
        exports = Some(capacities);
    }
    exports.context("Step-5 installed package has no layouts")
}

/// The one Step-5 Spark admission for serve and golden: the endpoint's unit,
/// typed owners over the sealed package's exports.
pub(crate) fn spark_model(exports: &[u32], endpoint: SparkEndpoint, waves: u64, clients: usize,
    max_frame_bytes: usize) -> Result<ModelAdmission> {
    let executions = cuteafd_loader::plan::spark_executions::admit_family("step5", exports, None,
        u64::from(endpoint.capacity), waves)?;
    ModelAdmission::for_clients(ExpertGeometry::STEP5, endpoint, spark_rings(max_frame_bytes)?, clients, executions)
}

// Startup loads the whole selected namespace, not just the current layer route.
// Shared SWA/CSA projections are packed even when CSA finishes per member.
const PACKED_PROGRAM_ROLES: &[(&str, crate::shared::prefill_packing::ProgramRows)] = {
    use crate::shared::prefill_packing::ProgramRows::{PackedRows, SequenceLocal};
    &[
        ("hidden_norm", PackedRows),
        ("attn_q", PackedRows), ("attn_kv", PackedRows), ("attn_gate", PackedRows),
        ("attn_out", PackedRows), ("attn_out_partial", PackedRows),
        ("head_gate", PackedRows),
        ("swa_q_norm_rope", PackedRows), ("swa_k_norm_rope", PackedRows),
        ("csa_q_norm_rope", PackedRows), ("csa_k_norm_rope", PackedRows),
        ("index_q", PackedRows), ("index_kz", PackedRows),
        ("index_q_norm", PackedRows), ("index_k_norm", PackedRows),
        ("index_weight", PackedRows),
        ("dense_gate_up", PackedRows), ("dense_down", PackedRows),
        ("post_silu_w13824_c7", PackedRows), ("post_silu_w13824_c10", PackedRows),
        ("router", PackedRows), ("route_weights", PackedRows),
        ("shared_gate_up", PackedRows), ("shared_down", PackedRows),
        ("post_silu_w1536_c7", PackedRows), ("post_silu_w1536_c10", PackedRows),
        ("residual_bf16", PackedRows), ("residual_f32", PackedRows),
        ("expert_input_quant", PackedRows), ("pack_fp8_kv_plane", PackedRows),
        ("sliding_positions", SequenceLocal), ("sliding_commit", SequenceLocal),
        ("swa_attention", SequenceLocal), ("swa_attention_fp8", SequenceLocal),
        ("csa_attention", SequenceLocal), ("csa_attention_fp8", SequenceLocal),
        ("region_compress", SequenceLocal), ("region_scores", SequenceLocal),
        ("region_selection", SequenceLocal), ("region_commit", SequenceLocal),
        // Predictor/head calls retain their member-local row count. RoPE runs at load.
        ("final_norm", SequenceLocal), ("mtp_eh", SequenceLocal),
        ("rope_s192_llama3", SequenceLocal), ("rope_s192_plain", SequenceLocal),
        ("rope_s32_llama3", SequenceLocal), ("rope_s32_plain", SequenceLocal),
        ("rope_s64_llama3", SequenceLocal), ("rope_s64_plain", SequenceLocal),
    ]
};

fn packed_program_inventory(split: bool) -> Vec<(String, crate::shared::prefill_packing::ProgramRows)> {
    std::iter::once("step5").chain(split.then_some("step52")).flat_map(|namespace|
        PACKED_PROGRAM_ROLES.iter().map(move |&(suffix, role)| (format!("{namespace}_{suffix}"), role)))
        .collect()
}

fn coordinator_packing_cap(programs: &cuteafd_ffi::programs::Programs<'_>, split: bool,
    mode: crate::shared::prefill_packing::PackingMode, rows: usize) -> Result<usize> {
    use crate::shared::prefill_packing::{coordinator_row_cap, validate_program_inventory, PackingMode};
    let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms {
        family: "step5", split_family: split.then_some("step52") };
    let owned = packed_program_inventory(split);
    let inventory = owned.iter().map(|(name, role)| (name.as_str(), *role)).collect::<Vec<_>>();
    validate_program_inventory(programs.names().filter(|name| selected.contains(name)), &inventory)?;
    if mode == PackingMode::Exact { coordinator_row_cap(programs, inventory, rows) } else { Ok(rows) }
}

fn text_media() -> cuteafd_engine::media::MediaAdmission<scheduler::Prompt, crate::shared::qwen::Encoder> {
    cuteafd_engine::media::MediaAdmission::new(cuteafd_engine::media::EmbeddingCache::new(0),
        crate::shared::qwen::Encoder::Off, 16)
}

fn with_backend<T>(args: &ServeArgs,
    run: impl FnOnce(&Step5Backend<'_, '_>, usize) -> Result<T>) -> Result<T> {
    let shape = args.shape()?;
    let mut resolved = args.clone();
    resolved.prefill_rows = u32::try_from(shape.unit_rows)?;
    resolved.prefill_waves = u8::try_from(shape.waves)?;
    let args = &resolved;
    args.validate()?;
    let checkpoint = Checkpoint::coordinator(&args.snapshot, false, args.mtp > 0)?;
    let cfg = Config::parse(&checkpoint)?;
    let mut catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    catalog.select_draft(DraftOwnership::None)?;
    ensure!(catalog.routed_experts().geometry()? == ExpertGeometry::STEP5, "Step-5 routed geometry mismatch");
    let format = super::experts::StepExperts::from_catalog(&catalog)?;
    cuteafd_core::set_expert_geometry(ExpertGeometry::STEP5)
        .map_err(|_| anyhow::anyhow!("Step-5 serving requires an isolated process"))?;
    let metadata = cuteafd_loader::plan::checkpoint::read_json(&format.package_manifest(&args.native_lib)?)?;
    let exports = format.exports(&metadata)?;
    // SAFETY: this owner outlives all programs, native lanes, transports and allocations.
    let library = Arc::new(unsafe { NativeLibrary::load(&args.native_lib)? });
    library.cuda_set_device(args.device)?;
    let ids = std::iter::once(args.device).chain(args.split_device).collect::<Vec<_>>();
    let devices = ids.iter().map(|&id| Device { library: &library, id }).collect::<Vec<_>>();
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let coordinator_cap = coordinator_packing_cap(&programs, args.split_device.is_some(),
        args.prefill_packing, args.prefill_rows as usize)?;
    let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms {
        family: "step5", split_family: args.split_device.map(|_| "step52") };
    let measured = RuntimeInventory::measure(&library, &ids, InventoryPhase::PreWeight, |id| {
        programs.load_matching(|name| selected.contains(name))?;
        if let Some(other) = ids.iter().copied().find(|&other| other != id) {
            library.cuda_enable_peer(other)?;
            library.tp2_native_initialize()?;
            library.peer_exchange_initialize()?;
            library.peer_abort_initialize()?;
            if std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").as_deref() == Ok("1") {
                library.peer_diagnostic_initialize()?;
            }
        }
        Ok(())
    })?;
    let selected_programs = ProgramSet::from_names(&["step5", "step52"], programs.names());
    let pending_code = ids.iter().enumerate().map(|(rank, &id)| crate::shared::inventory::pending_code(
        &library, id, rank, ids.len() == 2, "step5", &format.package_name(), &selected_programs))
        .collect::<Result<Vec<_>>>()?;
    let headroom = cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes;
    let mut inputs = RuntimeInputs { kv_format: args.kv_cache, inventory: Inventory { gpus: measured.baselines(&[]).into_iter()
        .map(|(capacity_bytes, baseline)| GpuBudget { capacity_bytes, baseline, headroom_bytes: headroom }).collect(),
        spark_ranks: args.peers.len(), peer_access: ids.len() == 2 }, rows: usize::try_from(shape.span_rows)?,
        full_prefill_logits: args.full_prefill_logits,
        waves: usize::from(args.prefill_waves), unit: args.prefill_rows as usize,
        sequences: usize::from(args.max_sequences), context: args.max_context,
        requested_pool: args.pool_tokens,
        marks: admission::prefix_marks(&args.prefix, usize::from(args.max_sequences)),
        peer_diagnostics: std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").as_deref() == Ok("1"), pending_code, startup: vec![],
        onboard: cuteafd_loader::plan::families::step5::admission::OnboardInputs::exl3(&catalog,
            Some(&args.manifest), args.prefill_rows as usize, args.rtx_expert_layers)? };
    let admission = admission::admit_mtp(&programs, &checkpoint, &inputs, usize::from(args.mtp))?;
    cuteafd_bench::context::set_rtx_expert_layers(admission.placement.onboard_layers);
    // Uploads convert on the host and copy synchronously. No transient device
    // conversion arena exists: each loading phase is bounded by retained owners.
    inputs.startup = admission.placement.items.iter().enumerate().map(|(rank, items)| {
        let bytes = items.iter().try_fold(0u64, |sum, item| sum.checked_add(item.bytes))
            .context("Step-5 loading phase overflow")?;
        Ok(Demand::new(rank as u8, cuteafd_core::memory_layout::Category::Workspace,
            "Step-5 retained loading peak", bytes, cuteafd_core::memory_layout::Basis::Exact))
    }).collect::<Result<Vec<_>>>()?;
    let admission = admission::admit_mtp(&programs, &checkpoint, &inputs, usize::from(args.mtp))?;
    measured.record("step5", &measured.gpus.iter().map(|g| g.free_bytes).collect::<Vec<_>>());
    let endpoint = args.endpoint(format.bf16_endpoint());
    let clients = admission.ownership.routed_endpoints()?;
    let config = TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(), ..Default::default() };
    // Spark admission is planned (headers only) here and started below, once the
    // coordinator's own allocations are admitted; the coordinator then loads
    // its weights while the Sparks load theirs (and, pushing, in one group).
    let spark_admission = SparkAdmission::plan(&args.snapshot, ExpertGeometry::STEP5, Some(&catalog), &SparkWorkingSet {
        rtx_layers: admission.ownership.onboarded_layers().to_vec(),
        spark_layers: (3..91).filter(|layer| !admission.ownership.onboarded(*layer)).collect(), spark_ranks: args.peers.len(),
        prefill_lanes: 1 }, args.peers.len(), None,
        spark_model(&exports, endpoint, u64::from(args.prefill_waves), clients, config.max_frame_bytes)?)?;
    let available = inputs.inventory.gpus.iter().map(GpuBudget::available).collect::<Vec<_>>();
    let reserves = (0..ids.len()).map(|rank| {
        let native = Execution::serving_admission(&programs, &checkpoint, &admission.ownership, rank, inputs.unit_rows(),
            inputs.rows, inputs.logit_rows())?;
        let mut native = native;
        if rank == usize::from(admission.ownership.execution().spec().head_gpu) {
            native.workspace = native.workspace.max(
                cuteafd_loader::plan::families::step5::native::mtp_forecast_workspace(inputs.unit_rows(), usize::from(args.mtp), inputs.kv_format)?);
        }
        let native = native.check(u64::MAX, &[])?;
        let total = admission.placement.items[rank].iter().try_fold(0u64, |sum, item| sum.checked_add(item.bytes))
            .context("Step-5 solved allocation overflow")?;
        total.checked_sub(native).context("Step-5 native allocation missing from solved ledger")
    }).collect::<Result<Vec<_>>>()?;
    let head_rank = usize::from(admission.ownership.execution().spec().head_gpu);
    let onboarded_budget = admission.placement.expert_ranges.get(usize::from(
        admission.ownership.execution().spec().entry_gpu)).filter(|range| range.layers > 0)
        .map(|range| usize::try_from(range.peak_bytes)).transpose()?;
    let predictor_budget = if args.mtp > 0 {
        let own = cuteafd_loader::plan::families::step5::coordinator::RankWeightLayout::for_mtp(
            &checkpoint, usize::from(args.mtp))?.resident_bytes;
        let total = admission.placement.items[head_rank].iter().try_fold(0u64, |sum, item| sum.checked_add(item.bytes))
            .context("Step-5 solved predictor allocation overflow")?;
        Some((available[head_rank], total.checked_sub(own).context("Step-5 predictor absent from ledger")?))
    } else { None };
    let caches = devices.iter().enumerate().map(|(rank, &device)| Ok(RefCell::new(CacheRank::new(device,
        admission.cache.clone(), rank, inputs.sequences as u64, admission.marks as u64,
        admission.placement.pool_tokens / 64, admission.reservations[rank].total)?)))
        .collect::<Result<Vec<_>>>()?;
    let stages = devices.iter().enumerate().map(|(rank, &device)| {
        let cache = caches[rank].borrow();
        Staging::new_for_units(device, &programs, &cache,
            Staging::bytes_for_sequences(&admission.cache, rank, inputs.sequences)?, inputs.sequences, inputs.unit_rows())
    }).collect::<Result<Vec<_>>>()?;
    let target_input = Inputs::allocate(devices[0], inputs.rows, Inputs::bytes(inputs.rows)?)?;
    let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == "model.embed_tokens.weight")
        .context("Step-5 embedding header missing")?;
    let source = EmbedSource::new(&args.snapshot, &tensor.shard, &tensor.meta, cfg.hidden)?;
    library.cuda_set_device(args.device)?;
    let ownership = admission.ownership;
    let caches_ref: Vec<_> = caches.iter().collect();
    let stages_ref: Vec<_> = stages.iter().collect();
    // Admission precedes allocation: `load_native` re-checks every rank's
    // admitted budget before allocating. With coordinator push, the native
    // owners are allocated and their records planned before membership (a
    // planning error offers no coordinator ranks: the Sparks still push and the
    // coordinator fills through the mount); otherwise they load through the
    // mount while the Sparks load.
    let push = spark_admission.coordinator_pushes();
    let mut native = None;
    let mut coordinator = None;
    let mut ranks = None;
    if push {
        let recorder = crate::shared::experts::rtx::dense::Recorder::start(true)?.context("dense recorder")?;
        let loaded = Execution::load_native(&programs, &checkpoint, &ownership, &caches_ref, &stages_ref,
            inputs.unit_rows(), inputs.rows, inputs.logit_rows(), &available, &reserves, 92);
        let records = recorder.finish();
        native = Some(loaded?);
        let planned = (|| -> Result<_> {
            let (landed, operands, staged, steps, largest, finishes) = records.summary();
            tracing::info!(landed, operands, staged, steps, largest_step_bytes = largest, finishes,
                "Step-5 coordinator dense records planned");
            let dense = crate::shared::experts::rtx::dense::plan_records(&args.snapshot, records.clone(), &devices,
                args.peers.len())?;
            let prepared = crate::shared::experts::rtx::push::CoordinatorPush::prepare(&args.snapshot, None,
                devices.len(), None)?;
            Ok((prepared, dense))
        })();
        match planned {
            Ok(((push, streams, offered), dense)) => {
                coordinator = Some((push, records, dense));
                ranks = Some((streams, offered));
            }
            Err(error) => {
                // No coordinator ranks: the Sparks still push; fill the recorded owners here.
                tracing::info!(%error, "Step-5 coordinator push unavailable; mount");
                crate::shared::experts::rtx::dense::fill_records_by_mount(&library, &records, |_| std::ptr::null_mut())?;
            }
        }
    }
    let pending = spark_admission.start_with(&args.peers, ranks)?;
    let (embedding, native) = crate::shared::placement::load_while_admitting(Some(pending), || {
        TokenEmbedding::load(&library, source, EmbedPlacement::Gpu, || -> Result<super::execution::LoadedNative<'_>> {
            if let Some((push, records, dense)) = coordinator {
                let split = crate::shared::memory::load_timing::Split::start();
                let (pushed, ()) = push.receive_with(devices.clone(), None, args.peers.len(), Some(Ok(dense)), || ());
                let pushed = pushed?;
                if pushed.is_none() {
                    crate::shared::experts::rtx::dense::fill_records_by_mount(&library, &records, |_| std::ptr::null_mut())?;
                }
                split.log(if pushed.is_some() { "step5 dense push" } else { "step5 dense mount fill" });
                tracing::info!(pushed = pushed.is_some(), operands = records.operands.len(), steps = records.steps.len(),
                    "Step-5 coordinator dense operands landed");
            }
            let native = match native {
                Some(native) => native,
                None => Execution::load_native(&programs, &checkpoint, &ownership, &caches_ref, &stages_ref,
                    inputs.unit_rows(), inputs.rows, inputs.logit_rows(),
                    &available, &reserves, 92)?,
            };
            // Onboarded EXL3 routed layers, whole on the entry GPU inside its solved expert arena.
            let entry = usize::from(ownership.execution().spec().entry_gpu);
            let onboarded = ownership.onboarded_layers();
            if onboarded.is_empty() { return Ok(native); }
            let peak = onboarded_budget.context("Step-5 onboarded layers without a solved expert arena")?;
            let package = format.package_manifest(&args.native_lib)?.parent().context("EXL3 package")?.join("rtx-tp1");
            native.with_onboarded(devices[entry], &catalog, &package, onboarded, inputs.unit_rows(), peak)
        })
    })?;
    cuteafd_transport::worker_selection::require_peer_clients(&args.peers, endpoint, clients)?;
    let world = SparkWorld::canonical(args.peers.len())?;
    let transport = devices[0].own(|| {
        // One return slot per prefill wave.
        let waves = usize::from(args.prefill_waves);
        let mut link = SparkLink::new_waves(&library, world, &args.peers, args.prefill_rows, config, 4096 * 2,
            waves, args.prefill_rows)?;
        link.enable_terminal_ownership(library.clone())?;
        link.bootstrap(&super::execution::bootstrap_request(inputs.unit_rows(), 1)?)?;
        Ok(link)
    })?;
    library.cuda_set_device(args.device)?;
    let execution = Execution::bind_native(&checkpoint, ownership, native, caches_ref, stages_ref, inputs.rows,
        Some(transport), 92, 1)?;
    let mtp = predictor_budget.map(|(available, reserved)| super::mtp::Runtime::load(
        &execution, &programs, &checkpoint, &caches[head_rank].borrow(),
        usize::from(args.mtp), inputs.unit_rows(), available, &[reserved])).transpose()?;
    let backend = Step5Backend::new(&execution, caches.iter().collect(), &embedding, &target_input)?
        .with_full_prefill_logits(args.full_prefill_logits)
        .with_prefill_depth(crate::shared::prefill_depth::PrefillDepth::new(usize::from(args.prefill_waves),
            args.prefill_rows as usize)?)?
        .configure_packing(args.prefill_packing, coordinator_cap)?;
    let backend = if let Some(mtp) = &mtp { backend.with_mtp(mtp) } else { backend };
    library.cuda_set_device(args.device)?;
    tracing::info!(rows=inputs.rows, sequences=inputs.sequences, mtp=args.mtp,
        context=backend.geometry().max_context, pool_tokens=admission.placement.pool_tokens,
        "Step-5 admitted eager target and native predictor depth");
    run(&backend, inputs.sequences)
}

fn worker(args: &ServeArgs, receive: &mut tokio::sync::mpsc::Receiver<cuteafd_api::openai::NativeRequest>,
    stats: &Arc<Mutex<serde_json::Value>>, eos: Vec<u32>, initialized: impl FnOnce(usize) -> Result<()>) -> Result<()> {
    with_backend(args, |backend, sequences| {
        let drafts = if args.mtp == 0 { scheduler::Drafts::None }
            else { scheduler::Drafts::Neural { depth: usize::from(args.mtp), fixed: false } };
        scheduler::schedule(backend, &args.snapshot, receive, stats, sequences, drafts,
            eos, args.decode_share, &args.prefix, args.token_io.token_select, &mut text_media(), None,
            || initialized(backend.geometry().max_context))
    })
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    use cuteafd_api::openai::{chat::step5::Step5Encoding, ModelEncoding, ModelProfile, NativeLimits};
    args.validate()?;
    let api = args.api.load()?;
    let snapshot = args.snapshot.clone();
    let encoding = Arc::new(Step5Encoding::from_snapshot(&snapshot)?);
    let eos = encoding.tokens().eos.clone();
    let model = args.model_id.clone().or_else(|| snapshot.ancestors().find_map(|path| {
        let name = path.file_name()?.to_str()?.strip_prefix("models--")?;
        let (organization, model) = name.split_once("--")?;
        Some(format!("{organization}/{model}"))
    })).context("Step-5 requires --model-id for a snapshot without HF identity")?;
    let profile = ModelProfile::new(model, ModelEncoding::Step5(encoding));
    let (queue, mut receive) = tokio::sync::mpsc::channel(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let engine_args = args.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let handle = tokio::task::spawn_blocking(move || {
        let mut ready = Some(ready_tx);
        let result = worker(&engine_args, &mut receive, &worker_stats, eos, |context| {
            if let Some(ready) = ready.take() { let _ = ready.send(Ok(context)); }
            Ok(())
        });
        if let Some(ready) = ready.take() {
            let _ = ready.send(Err(anyhow::anyhow!("Step-5 initialization failed: {:#}",
                result.as_ref().err().context("worker exited before readiness")?)));
        }
        result
    });
    let context = ready_rx.await.context("Step-5 worker exited before readiness")??;
    let limits = NativeLimits::new(u32::try_from(context)?, args.max_output)?;
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new(
        "step5", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, std::time::Duration::from_secs(25),
        hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(handle) => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args() -> ServeArgs {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-step5", "--snapshot", "/m",
            "--native-lib", "/lib.so", "--peers", "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4"]).unwrap();
        let crate::cli::Commands::ServeStep5(args) = cli.command else { panic!("Step-5 dispatch") };
        args
    }

    fn live_mtp_args() -> ServeArgs {
        let mut args = args();
        args.split_device = Some(1);
        args.prefill_rows = 64;
        args.prefill_waves = 1;
        args.max_sequences = 3;
        args.mtp = 3;
        // Smallest supported request context; the checkpoint's 1M compiled extent stays unchanged.
        args.max_context = 262_144;
        args.pool_tokens = Some(args.max_context);
        args.prefix.prefix_cache_entries = 1;
        args.prefix.prefix_cache_min_tokens = 1;
        args
    }

    #[test]
    fn live_mtp_fixture_context_and_explicit_pool_agree() -> Result<()> {
        let args = live_mtp_args();
        args.validate()?;
        assert_eq!(args.max_context, 262_144);
        assert_eq!(args.pool_tokens, Some(args.max_context));
        let policy = cuteafd_loader::placement::PoolPolicy::resolve(
            &[96 << 30; 2], args.max_context, args.pool_tokens, 64, false);
        policy.validate_explicit()?;
        assert_eq!(policy.compiled_context, args.max_context);
        assert!(args.max_context >= 640 + u64::from(args.mtp));
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only: requires real checkpoint headers and sealed native export metadata"]
    fn native_mtp_serve_startup_dry_run() -> Result<()> {
        use cuteafd_loader::{placement::{self, Baseline}, plan::{self, layout::LayoutOptions}};
        let mut args = args();
        args.snapshot = std::env::var("CUTEAFD_STEP5_SNAPSHOT")?.into();
        args.kv_cache = std::env::var("CUTEAFD_STEP5_KV_CACHE")
            .unwrap_or_else(|_| "bf16".into()).parse()?;
        args.native_lib = std::env::var("CUTEAFD_NATIVE_LIB")?.into();
        args.manifest = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?.into();
        args.split_device = Some(1);
        args.prefill_rows = 256;
        args.prefill_waves = 1;
        args.max_sequences = 16;
        args.prefix.host_cache_bytes = crate::shared::prefix::HostBudget::Auto;
        args.peers = (1..=6).map(|port| format!("127.0.0.1:{port}").parse()).collect::<std::result::Result<_, _>>()?;
        args.max_context = 1_048_576;
        args.pool_tokens = None;
        let checkpoint = Checkpoint::coordinator(&args.snapshot, false, true)?;
        let mut catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
        catalog.select_draft(DraftOwnership::None)?;
        ensure!(catalog.routed_experts().geometry()? == ExpertGeometry::STEP5, "routed geometry");
        let format = super::super::experts::StepExperts::from_catalog(&catalog)?;
        let metadata = plan::checkpoint::read_json(&format.package_manifest(&args.native_lib)?)?;
        let exports = format.exports(&metadata)?;
        let manifest = plan::checkpoint::read_json(&args.manifest)?;
        ensure!(manifest["programs"].as_array().context("native programs")?.iter()
            .all(|program| program.get("card_plans").is_none()),
            "CPU-only startup needs exports without runtime SM selection");
        // SAFETY: only export metadata is read; no CUDA init, device or allocation calls.
        let library = unsafe { NativeLibrary::load(&args.native_lib)? };
        let programs = library.programs()?.with_manifest(&args.manifest)?;
        let capacity = placement::inventory::PRO_TOTAL_BYTES;
        let arch = placement::inventory::ArchContext::coordinator(capacity, None);
        let selected = ProgramSet::from_names(&["step5", "step52"], programs.names());
        let mut inputs = RuntimeInputs {
                kv_format: args.kv_cache,
            inventory: Inventory { gpus: vec![GpuBudget { capacity_bytes: capacity,
                baseline: Baseline::Measured { free_bytes: capacity - arch.context_bytes },
                headroom_bytes: LayoutOptions::default().headroom_bytes }; 2], spark_ranks: 6, peer_access: true },
            rows: 256, waves: 1, unit: 0, full_prefill_logits: false, sequences: 16, context: args.max_context, requested_pool: None,
            marks: admission::prefix_marks(&args.prefix, 16), peer_diagnostics: false,
            pending_code: (0..2).map(|rank| placement::inventory::loaded_code_allowance(
                "step5", &format.package_name(), true, rank, &arch, Some(&selected)).bytes).collect(),
            startup: vec![],
            onboard: cuteafd_loader::plan::families::step5::admission::OnboardInputs::exl3(&catalog,
                Some(&args.manifest), 256, args.rtx_expert_layers)?,
        };
        for depth in 1..=3 {
            args.mtp = depth;
            args.validate()?;
            inputs.startup.clear();
            let admitted = admission::admit_mtp(&programs, &checkpoint, &inputs, usize::from(depth))?;
            inputs.startup = admitted.placement.items.iter().enumerate().map(|(rank, items)|
                Demand::new(rank as u8, cuteafd_core::memory_layout::Category::Workspace,
                    "Step-5 retained loading peak", items.iter().map(|item| item.bytes).sum(),
                    cuteafd_core::memory_layout::Basis::Exact)).collect();
            let admitted = admission::admit_mtp(&programs, &checkpoint, &inputs, usize::from(depth))?;
            let endpoint = args.endpoint(format.bf16_endpoint());
            let clients = admitted.ownership.routed_endpoints()?;
            SparkAdmission::plan(&args.snapshot, ExpertGeometry::STEP5, Some(&catalog), &SparkWorkingSet {
                rtx_layers: vec![], spark_layers: (3..91).collect(), spark_ranks: 6, prefill_lanes: 1 },
                6, None, spark_model(&exports, endpoint, 1, clients, TcpTransportConfig::default().max_frame_bytes)?)?;
            super::super::execution::bootstrap_request(inputs.rows, 1)?;
            let head_rank = usize::from(admitted.ownership.execution().spec().head_gpu);
            for rank in 0..2 {
                let mut native = Execution::serving_admission(&programs, &checkpoint,
                    &admitted.ownership, rank, inputs.unit_rows(), inputs.rows, inputs.logit_rows())?;
                if rank == head_rank {
                    native.workspace = native.workspace.max(
                        plan::families::step5::native::mtp_forecast_workspace(inputs.unit_rows(), usize::from(depth), inputs.kv_format)?);
                }
                let native = native.check(u64::MAX, &[])?;
                let total = admitted.placement.items[rank].iter().map(|item| item.bytes).sum::<u64>();
                ensure!(native <= total && total <= inputs.inventory.gpus[rank].available(),
                    "native allocation exceeds solved startup budget at rank {rank}");
            }
            let predictor = plan::families::step5::coordinator::RankWeightLayout::for_mtp(
                &checkpoint, usize::from(depth))?.resident_bytes;
            let total = admitted.placement.items[head_rank].iter().map(|item| item.bytes).sum::<u64>();
            ensure!(predictor <= total
                && admitted.cache.ranks[head_rank].mtp_lengths.len() == usize::from(depth),
                "predictor runtime exceeds admitted owners/cache");
            for stage in 0..usize::from(depth) {
                super::super::mtp::InputPlan::prepare(&programs, &checkpoint, stage, inputs.unit_rows())?;
                super::super::head::HeadPlan::for_mtp_stage(&programs, &checkpoint, stage,
                    plan::families::step5::native::head_rows(inputs.rows, false))?;
                super::super::ffn::DensePlan::for_mtp_stage(&programs, &checkpoint,
                    stage, inputs.rows, u64::MAX, &[])?;
            }
            let planned = plan::plan(&args.snapshot, &plan::PlanOptions {
                placement: plan::ExpertPlacement::from_spark_ranks(6),
                draft_ownership: Some(DraftOwnership::NativeMtp),
                layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![capacity; 2], step5_kv: args.kv_cache,
                    native_mtp_layers: usize::from(depth), workspace_manifest: Some(args.manifest.clone()),
                    ..Default::default() }), ..Default::default()
            })?;
            ensure!(planned.wip_launchable(), "MTP depth {depth}: {:?}", planned.hints);
            ensure!(planned.solver_placement.as_ref() == Some(&admitted.placement),
                "planned/measured MTP startup differs at depth {depth}");
            eprintln!("Step-5 CPU serve startup: depth={depth} real_headers=true installed_exports=true admission=true CUDA_initialized=false");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires sealed native exports, full target weights and real Spark workers"]
    fn live_mtp_target_abort_restore_and_fresh_marks_are_exact() -> Result<()> {
        use crate::shared::qwen::scheduler::DraftState as _;
        use cuteafd_engine::prefix::{After, PrefixFamily, SnapshotKind};
        use super::super::prefix::Placement;
        let mut args = live_mtp_args();
        args.kv_cache = std::env::var("CUTEAFD_STEP5_KV_CACHE")
            .unwrap_or_else(|_| "bf16".into()).parse()?;
        args.snapshot = std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?.into();
        args.native_lib = std::env::var("CUTEAFD_NATIVE_LIB")?.into();
        args.manifest = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?.into();
        args.peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?.split(',')
            .map(str::parse).collect::<std::result::Result<Vec<_>, _>>()?;
        let windows: serde_json::Value = serde_json::from_slice(&std::fs::read(
            std::path::Path::new(&std::env::var("CUTEAFD_GOLDEN_FIXTURES")?).join("windows.json"))?)?;
        let tokens = windows["windows"][0]["tokens"].as_array().context("live exactness tokens")?
            .iter().map(|v| v.as_u64().context("live token id").and_then(|n| Ok(u32::try_from(n)?)))
            .collect::<Result<Vec<_>>>()?;
        ensure!(tokens.len() > 577, "live exactness needs at least 578 real tokens");
        with_backend(&args, |backend, _| {
            backend.draft_geometry(&args.snapshot, 3)?;
            let (family, mut pool) = backend.prefix_cache(&args.prefix, 3)?;
            let prefill = |placement: &mut Placement, state: &mut super::super::mtp::DraftState,
                start: usize, end: usize, pool: &cuteafd_engine::prefix::RefPagePool| -> Result<()> {
                for at in (start..end).step_by(64) {
                    let last = (at + 64).min(end);
                    backend.prefill_device(placement, &tokens[at..last], 0, pool)?;
                    backend.prefill_draft(placement, at, &tokens[at..last], tokens.get(last).copied(), state)?;
                }
                Ok(())
            };
            for frontier in [63usize, 64, 65, 575, 576] {
                let mut source = pool.admit_cold(&family, frontier, frontier + 4,
                    |pages| Placement { pages, sequence: 0, committed: 0 })?.placement;
                let mut source_draft = super::super::mtp::DraftState::default();
                backend.bind_request(&mut source, (), cuteafd_engine::media::RequestMedia::new(Vec::new(), 4096, frontier)?,
                    &tokens, 0)?;
                prefill(&mut source, &mut source_draft, 0, frontier, pool.pool())?;
                source_draft.close(tokens[frontier]);
                let expected = backend.diagnostic_state(&source)?;
                ensure!(pool.capture(&family, SnapshotKind::Prompt, &tokens[..frontier], &source, After::default())?,
                    "live exactness snapshot not retained");
                let mut cold = pool.admit_cold(&family, frontier, frontier + 4,
                    |pages| Placement { pages, sequence: 1, committed: 0 })?.placement;
                let mut cold_draft = super::super::mtp::DraftState::default();
                backend.bind_request(&mut cold, (), cuteafd_engine::media::RequestMedia::new(Vec::new(), 4096, frontier)?,
                    &tokens, 0)?;
                prefill(&mut cold, &mut cold_draft, 0, frontier, pool.pool())?;
                ensure!(expected == backend.diagnostic_state(&cold)?, "live fresh mark differs at {frontier}");
                let compatible = |saved: &super::super::prefix::Frontier, end|
                    backend.snapshot_compatible(saved, end, true);
                let restored = pool.admit_native_compatible(&family, &tokens[..frontier + 1],
                    &tokens[..frontier + 1], &[], frontier + 4, false, Some(&compatible),
                    |pages| Placement { pages, sequence: 2, committed: 0 })?;
                ensure!(restored.resume == frontier, "live exact prefix miss at {frontier}");
                let mut restored = restored.placement;
                let mut restored_draft = super::super::mtp::DraftState::default();
                backend.bind_request(&mut restored, (), cuteafd_engine::media::RequestMedia::new(Vec::new(), 4096, frontier)?,
                    &tokens, frontier)?;
                ensure!(expected == backend.diagnostic_state(&restored)?, "live restore mark differs at {frontier}");
                // Each independent continuation gets a private append lease,
                // exactly as the scheduler's page ownership requires.
                for placement in [&mut source, &mut cold, &mut restored] {
                    let old = std::mem::take(&mut placement.pages);
                    placement.pages = pool.fork_step(&family, &old, frontier, frontier + 4)?;
                    pool.release(&family, &old)?;
                }
                let drafts = backend.draft(&mut [scheduler::DraftSeq { placement: &source,
                    seq: &mut source_draft, depth: 3 }], &mut scheduler::DraftTiming::default())?.remove(0);
                ensure!(drafts.len() == 3, "live MTP did not propose every admitted stage");
                ensure!(expected == backend.diagnostic_state(&source)?, "draft mutated committed mark at {frontier}");
                let rows = std::iter::once(tokens[frontier]).chain(drafts).collect::<Vec<_>>();
                backend.diagnostic_abort_once();
                ensure!(backend.verify_device(&mut [(&mut source, &rows)], true, pool.pool()).is_err(),
                    "live abort injection did not fire");
                ensure!(expected == backend.diagnostic_state(&source)?, "aborted target changed mark at {frontier}");
                backend.verify_device(&mut [(&mut source, &rows)], true, pool.pool())?;
                backend.accept(&mut [scheduler::Verified { placement: &mut source, seq: &mut source_draft,
                    start: frontier, history: (), rows: &rows, first_row: 0,
                    kept: Some((1, tokens[frontier + 1])), finished: true, neural: true }], true)?;
                for (placement, draft) in [(&mut cold, &mut cold_draft), (&mut restored, &mut restored_draft)] {
                    backend.verify_device(&mut [(placement, &rows)], true, pool.pool())?;
                    backend.accept(&mut [scheduler::Verified { placement, seq: draft, start: frontier,
                        history: (), rows: &rows, first_row: 0, kept: Some((1, tokens[frontier + 1])),
                        finished: true, neural: true }], true)?;
                }
                let continued = backend.diagnostic_state(&source)?;
                ensure!(continued == backend.diagnostic_state(&cold)?, "live replay/fresh continuation differs at {frontier}");
                ensure!(continued == backend.diagnostic_state(&restored)?, "live restored continuation differs at {frontier}");
                ensure!(pool.capture(&family, SnapshotKind::Turn, &tokens[..frontier + 1], &source,
                    After::default())?, "live drafting turn not retained");
                for placement in [&source, &cold, &restored] { pool.release(&family, &placement.pages)?; }
                let next = pool.admit_native_compatible(&family, &tokens[..frontier + 2],
                    &tokens[..frontier + 2], &[], frontier + 4, false, Some(&compatible),
                    |pages| Placement { pages, sequence: 0, committed: 0 })?;
                ensure!(next.resume == frontier + 1, "live drafting conversation missed its prior turn");
                let mut next = next.placement;
                backend.bind_request(&mut next, (), cuteafd_engine::media::RequestMedia::new(Vec::new(), 4096, frontier + 1)?,
                    &tokens, frontier + 1)?;
                ensure!(continued == backend.diagnostic_state(&next)?, "live drafting turn restore differs");
                let mut fresh = pool.admit_cold(&family, frontier + 1, frontier + 4,
                    |pages| Placement { pages, sequence: 1, committed: 0 })?.placement;
                let mut fresh_draft = super::super::mtp::DraftState::default();
                backend.bind_request(&mut fresh, (), cuteafd_engine::media::RequestMedia::new(Vec::new(), 4096, frontier + 1)?,
                    &tokens, 0)?;
                prefill(&mut fresh, &mut fresh_draft, 0, frontier, pool.pool())?;
                prefill(&mut fresh, &mut fresh_draft, frontier, frontier + 1, pool.pool())?;
                ensure!(continued == backend.diagnostic_state(&fresh)?, "live prior-turn hit differs from cold prefill");
                backend.require_live()?;
                println!("Step-5 live exactness frontier={frontier} target_kv_csa=real predictor_stages=3 abort_replay=true restore=true fresh_marks=true turn_hit=true byte_exact=true");
                for placement in [&next, &fresh] { pool.release(&family, &placement.pages)?; }
                pool.clear(&family)?;
            }
            family.drain().map_err(|e| anyhow::anyhow!("live exactness drain: {e}"))?;
            Ok(())
        })
    }

    fn packing_non_inferior(delta_kl: f64, delta_top1: f64) -> bool {
        delta_kl.is_finite() && delta_top1.is_finite()
            && delta_kl <= 0.005 + 1e-12 && delta_top1 >= -0.01 - 1e-12
    }

    #[test]
    fn packed_fidelity_non_inferiority_bounds_are_inclusive_and_finite() {
        assert!(packing_non_inferior(0.005, -0.01));
        assert!(packing_non_inferior(0.0, 0.0));
        assert!(!packing_non_inferior(0.0051, 0.0));
        assert!(!packing_non_inferior(0.0, -0.0101));
        assert!(!packing_non_inferior(f64::NAN, 0.0));
        assert!(!packing_non_inferior(0.0, f64::INFINITY));
    }

    #[test]
    #[ignore = "requires sealed native exports, target weights and real Spark workers"]
    fn live_packed_prefill_matches_singletons_with_private_mtp() -> Result<()> {
        use super::super::{mtp::DraftState, prefix::Placement};
        use cuteafd_engine::prefix::PrefixFamily;
        let mut args = live_mtp_args();
        args.split_device = std::env::var("CUTEAFD_STEP5_SPLIT_DEVICE").ok().map(|s| s.parse()).transpose()?;
        args.device = std::env::var("CUTEAFD_STEP5_DEVICE").unwrap_or_else(|_| "0".into()).parse()?;
        args.prefill_rows = 256;
        args.max_sequences = 4;
        args.prefill_packing = <crate::shared::prefill_packing::PackingMode as clap::ValueEnum>::from_str(
            &std::env::var("CUTEAFD_STEP5_PACKING").unwrap_or_else(|_| "exact".into()), false)
            .map_err(|e| anyhow::anyhow!(e))?;
        args.kv_cache = std::env::var("CUTEAFD_STEP5_KV_CACHE").unwrap_or_else(|_| "fp8".into()).parse()?;
        args.snapshot = std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?.into();
        args.native_lib = std::env::var("CUTEAFD_NATIVE_LIB")?.into();
        args.manifest = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?.into();
        args.peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?.split(',')
            .map(str::parse).collect::<std::result::Result<Vec<_>, _>>()?;
        let fixtures = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures.join("meta.json"))?)?;
        super::super::golden::reference_identity(&meta, false)?;
        use sha2::{Digest, Sha256};
        let tokenizer = format!("{:x}", Sha256::digest(std::fs::read(args.snapshot.join("tokenizer.json"))?));
        ensure!(meta["snapshot_identity"]["tokenizer_sha256"] == tokenizer,
            "packed reference tokenizer differs from target");
        let window = meta["windows"].as_array().context("packed reference windows")?.iter()
            .find(|w| w["id"] == "original-page-leases" && w["path"] == "windows/original-page-leases")
            .context("packed reference window absent")?;
        let positions = window["positions"].as_array().context("packed reference positions")?.iter()
            .map(|p| p.as_u64().map(|p| p as usize).context("invalid packed reference position"))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!positions.is_empty() && positions.windows(2).all(|p| p[0] < p[1]),
            "packed reference positions must increase");
        let folder = fixtures.join("windows/original-page-leases");
        let bytes = std::fs::read(folder.join("tokens.bin"))?;
        ensure!(bytes.len() % 4 == 0, "packed exactness token byte extent");
        let tokens = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        ensure!(tokens.len() >= 192, "packed exactness needs 192 real tokens");
        let reference_bytes = std::fs::read(folder.join("logits.bin"))?;
        let vocab = window["vocab"].as_u64().context("packed reference vocabulary")? as usize;
        ensure!(vocab > 0 && reference_bytes.len() == positions.len() * vocab * 4,
            "packed reference logit extent");
        let reference_logits = reference_bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        with_backend(&args, |backend, _| {
            let (family, mut pool) = backend.prefix_cache(&args.prefix, 4)?;
            let mut placements = (0..4).map(|sequence| pool.admit_cold(&family, 192, 256,
                |pages| Placement { pages, sequence, committed: 0 }).map(|a| a.placement)).collect::<std::result::Result<Vec<_>, _>>()?;
            let mut drafts = (0..4).map(|_| DraftState::default()).collect::<Vec<_>>();
            ensure!(backend.geometry().vocab_size == vocab, "packed reference vocabulary mismatch");
            let prompts = [&tokens[..128], &tokens[..159]];
            for (i, (p, draft)) in placements.iter_mut().zip(&mut drafts).enumerate() {
                let prompt = prompts[i % 2];
                backend.bind_request(p, (), cuteafd_engine::media::RequestMedia::new(Vec::new(), 4096, prompt.len())?, prompt, 0)?;
                if i % 2 == 1 {
                    backend.prefill_device(p, &prompt[..65], 0, pool.pool())?;
                    backend.prefill_draft(p, 0, &prompt[..65], prompt.get(65).copied(), draft)?;
                }
            }
            let chunks = [&prompts[0][..96], &prompts[1][65..105]];
            let limits = backend.packed_limits().context("Step-5 packing limits missing")?;
            let layout = [(0, chunks[0].len()), (65, chunks[1].len())];
            let groups = crate::shared::prefill_packing::groups(2, |range|
                crate::shared::prefill_packing::plan(&layout[range], &limits).is_ok());
            let mut packed_logits = vec![Vec::new(); 2];
            let mut packed_scores = vec![Vec::new(); 2];
            let mut packs = 0;
            for range in groups {
                if range.len() == 1 {
                    let i = range.start;
                    let start = placements[i].committed;
                    let logits = backend.prefill_device(&mut placements[i], chunks[i], 1, pool.pool())?.context("singleton head")?;
                    packed_logits[i] = logits.row_host(backend.library(), 0)?;
                    backend.prefill_draft(&placements[i], start, chunks[i], None, &mut drafts[i])?;
                } else {
                    packs += 1;
                    let mut sequences = placements[..2].iter_mut().zip(&mut drafts[..2]).enumerate().map(|(i, (p, draft))|
                        scheduler::PrefillSeq { start: p.committed, placement: p, tokens: chunks[i], next: None, draft, neural: true })
                        .collect::<Vec<_>>();
                    let outcomes = backend.prefill_packed(&mut sequences, pool.pool(), &mut |i, logits| {
                        packed_logits[i] = logits.row_host(backend.library(), 0)?;
                        Ok(Ok(()))
                    })?;
                    for outcome in outcomes { outcome?; }
                    // Each stored position p predicts token p from the true prefix [0, p).
                    // MTP/head scratch may change, but the packed residual has its own owner.
                    for i in 0..2 {
                        let (start, rows) = layout[i];
                        let selected = positions.iter().enumerate().filter(|(_, p)|
                            **p > start && **p <= start + rows).collect::<Vec<_>>();
                        ensure!(!selected.is_empty(), "packed member {i} has no official scored rows");
                        let offset = chunks[..i].iter().map(|c| c.len()).sum::<usize>();
                        let packed_rows = selected.iter().map(|(_, p)| offset + **p - start - 1).collect::<Vec<_>>();
                        let actual = backend.diagnostic_packed_logits(&packed_rows)?;
                        packed_scores[i] = actual.into_iter().flatten().collect();
                    }
                }
            }
            let exact = args.prefill_packing == crate::shared::prefill_packing::PackingMode::Exact;
            ensure!(exact || packs == 1, "unrestricted fixture did not exercise packed execution");
            let mut byte_exact = true;
            let mut pooled = [0.0; 5]; // packed KL, singleton KL, packed hits, singleton hits, rows
            let mut non_inferior = true;
            for i in 0..2 {
                let j = i + 2;
                let start = placements[j].committed;
                let logits = backend.prefill_device(&mut placements[j], chunks[i], 1, pool.pool())?.context("alone head")?;
                let alone = logits.row_host(backend.library(), 0)?;
                if !exact {
                    let selected = positions.iter().enumerate().filter(|(_, p)|
                        **p > start && **p <= start + chunks[i].len()).collect::<Vec<_>>();
                    let rows = selected.iter().map(|(_, p)| **p - start - 1).collect::<Vec<_>>();
                    let control = backend.diagnostic_packed_logits(&rows)?.into_iter().flatten().collect::<Vec<_>>();
                    let expected = selected.iter().flat_map(|(row, _)|
                        reference_logits[row * vocab..(row + 1) * vocab].iter().copied()).collect::<Vec<_>>();
                    let score = |actual: &[f32]| -> Result<(f64, usize)> {
                        let (kl, _) = crate::shared::golden_scores::scored_kl(&expected, actual, vocab)?;
                        let argmax = |row: &[f32]| row.iter().enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
                        let hits = expected.chunks_exact(vocab).zip(actual.chunks_exact(vocab))
                            .filter(|(a, b)| argmax(a) == argmax(b)).count();
                        Ok((kl, hits))
                    };
                    let (packed_kl, packed_hits) = score(&packed_scores[i])?;
                    let (control_kl, control_hits) = score(&control)?;
                    let n = rows.len() as f64;
                    let delta_kl = packed_kl - control_kl;
                    let delta_top1 = (packed_hits as f64 - control_hits as f64) / n;
                    non_inferior &= packing_non_inferior(delta_kl, delta_top1);
                    println!("Step-5 packed official member={i} rows={} packed_KL={packed_kl:.9e} singleton_KL={control_kl:.9e} packed_top1={:.6} singleton_top1={:.6} delta_KL={delta_kl:.9e} delta_top1={delta_top1:.6}",
                        rows.len(), packed_hits as f64 / n, control_hits as f64 / n);
                    pooled[0] += packed_kl * n;
                    pooled[1] += control_kl * n;
                    pooled[2] += packed_hits as f64;
                    pooled[3] += control_hits as f64;
                    pooled[4] += n;
                }
                backend.prefill_draft(&placements[j], start, chunks[i], None, &mut drafts[j])?;
                let packed = &packed_logits[i];
                let (kl, _) = crate::shared::golden_scores::scored_kl(&alone, packed, backend.geometry().vocab_size)?;
                let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
                let top1 = argmax(packed) == argmax(&alone);
                let logits_equal = packed.iter().map(|v| v.to_bits()).eq(alone.iter().map(|v| v.to_bits()));
                let state = backend.diagnostic_state(&placements[i])?;
                let reference = backend.diagnostic_state(&placements[j])?;
                ensure!(state.len() == reference.len() && state.iter().zip(&reference).all(|(a, b)| a.len() == b.len()),
                    "packed member {i} target/predictor state shape differs from singleton");
                let differing = state.iter().flatten().zip(reference.iter().flatten()).filter(|(a, b)| a != b).count();
                let bytes = state.iter().map(Vec::len).sum::<usize>();
                let frontier = placements[i].committed == placements[j].committed
                    && placements[i].committed == layout[i].0 + chunks[i].len();
                ensure!(frontier, "packed member {i} frontier differs from singleton");
                println!("Step-5 packed member={i} KL(alone||packed)={kl:.9e} top1_agreement={}/1 logits_byte_exact={logits_equal} state_differing_bytes={differing}/{bytes} frontier={frontier}", usize::from(top1));
                byte_exact &= logits_equal && differing == 0;
                if exact {
                    ensure!(logits_equal, "packed member {i} logits differ from singleton");
                    ensure!(differing == 0, "packed member {i} target/predictor state differs from singleton");
                }
            }
            if !exact {
                ensure!(pooled[4] > 0.0, "packed fidelity needs scored rows");
                let delta_kl = (pooled[0] - pooled[1]) / pooled[4];
                let delta_top1 = (pooled[2] - pooled[3]) / pooled[4];
                println!("Step-5 packed pooled rows={} packed_KL={:.9e} singleton_KL={:.9e} packed_top1={:.6} singleton_top1={:.6} delta_KL={delta_kl:.9e} delta_top1={delta_top1:.6}",
                    pooled[4], pooled[0] / pooled[4], pooled[1] / pooled[4], pooled[2] / pooled[4], pooled[3] / pooled[4]);
                ensure!(non_inferior && packing_non_inferior(delta_kl, delta_top1),
                    "packed member or pooled fidelity regresses its same-row singleton control");
            }
            println!("Step-5 packed check mode={:?} members=2 packs={packs} cap={} target_kv_csa=checked predictor_stages=3 logits=checked byte_exact={byte_exact}", args.prefill_packing, limits.rows);
            for p in &placements { pool.release(&family, &p.pages)?; }
            family.drain().map_err(|e| anyhow::anyhow!("packed exactness drain: {e}"))?;
            Ok(())
        })
    }

    #[test]
    fn packed_startup_catalogue_has_explicit_roles_and_rejects_new_programs() -> Result<()> {
        use crate::shared::prefill_packing::{validate_program_inventory, ProgramRows::*};
        for split in [false, true] {
            let owned = packed_program_inventory(split);
            let inventory = owned.iter().map(|(name, role)| (name.as_str(), *role)).collect::<Vec<_>>();
            assert_eq!(inventory.len(), if split { 96 } else { 48 });
            validate_program_inventory(inventory.iter().map(|(name, _)| *name), &inventory)?;
            let mut required = inventory.iter().map(|(name, _)| *name).collect::<Vec<_>>();
            required.push("step5_new_projection");
            assert!(validate_program_inventory(required, &inventory).is_err());
            assert!(validate_program_inventory(inventory.iter().skip(1).map(|(name, _)| *name), &inventory).is_err());
            for prefix in std::iter::once("step5").chain(split.then_some("step52")) {
                let role = |suffix| inventory.iter().find(|(name, _)| *name == format!("{prefix}_{suffix}"))
                    .map(|(_, role)| *role).unwrap();
                for suffix in ["router", "attn_q", "index_weight", "pack_fp8_kv_plane", "shared_gate_up"] {
                    assert_eq!(role(suffix), PackedRows);
                }
                for suffix in ["swa_attention", "csa_attention_fp8", "region_compress", "mtp_eh", "final_norm"] {
                    assert_eq!(role(suffix), SequenceLocal);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_exact_coordinator_cap_keeps_large_partner_singleton() -> Result<()> {
        use crate::shared::prefill_packing::{groups, plan, Limits};
        let chunks = [(0, 32), (65, 80), (0, 16), (511, 48)];
        let expert_cap = 256usize;
        let limits = Limits { rows: expert_cap.min(64), chunk_rows: 256, max_context: 4096, taps: None };
        assert_eq!(groups(chunks.len(), |range| plan(&chunks[range], &limits).is_ok()),
            [0..1, 1..2, 2..4]);
        assert!(plan(&chunks[..2], &Limits { rows: expert_cap, ..limits }).is_ok());
        assert_eq!(chunks[1].1, 80, "the large member retains its singleton arithmetic");
        let unknown = Limits { rows: 0, ..limits };
        assert_eq!(groups(chunks.len(), |range| plan(&chunks[range], &unknown).is_ok()),
            [0..1, 1..2, 2..3, 3..4]);
        Ok(())
    }

    #[test]
    fn target_serving_defaults_and_invalid_owners() -> Result<()> {
        let mut args = args();
        args.validate()?;
        assert_eq!((args.max_context, args.max_sequences, args.mtp), (1_048_576, 16, 3));
        assert_eq!(args.kv_cache, cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8);
        assert_eq!(cuteafd_loader::plan::layout::LayoutOptions::default().step5_kv, args.kv_cache);
        for format in ["bf16", "fp8"] {
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-step5", "--snapshot", "/m",
                "--native-lib", "/lib.so", "--peers", "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4",
                "--kv-cache", format])?;
            let crate::cli::Commands::ServeStep5(explicit) = cli.command else { panic!("Step-5 dispatch") };
            explicit.validate()?;
            assert_eq!(explicit.kv_cache.to_string(), format);
        }
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-step5", "--snapshot", "/m",
            "--native-lib", "/lib.so", "--peers", "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4",
            "--mtp", "0"])?;
        let crate::cli::Commands::ServeStep5(off) = cli.command else { panic!("Step-5 dispatch") };
        off.validate()?;
        assert_eq!(off.mtp, 0);
        args.split_device = Some(args.device);
        assert!(args.validate().is_err());
        args.split_device = Some(1);
        args.validate()?;
        args.max_context = 262_143;
        assert!(args.validate().is_err());
        args.max_context = 262_144;
        args.peers[1] = args.peers[0];
        assert!(args.validate().is_err());
        Ok(())
    }

    #[test]
    fn topology_shapes_and_explicit_units_match_admission() -> Result<()> {
        // The fixed three units of the reserve (TJ 2026-10-28).
        for (sparks, unit, waves, span) in [(4, 2752, 3, 8256), (6, 2752, 3, 8256)] {
            let mut args = args();
            args.peers = (1..=sparks).map(|port| format!("127.0.0.1:{port}").parse())
                .collect::<std::result::Result<_, _>>()?;
            args.validate()?;
            let shape = args.shape()?;
            assert_eq!((shape.unit_rows, shape.waves, shape.span_rows), (unit, waves, span));
            args.prefill_rows = unit as u32;
            let exports = [1, 16, 80, 256, 1024, 4096];
            let model = spark_model(&exports, args.endpoint(false), waves, 2, TcpTransportConfig::default().max_frame_bytes)?;
            // One prefill owner at the selected unit's covering export; the k units only buffer I/O.
            assert_eq!(model.executions.prefill, cuteafd_core::PrefillExecution { capacity: unit as u32,
                compiled_capacity: 4096, unit_rows: unit as u32, buffered_units: waves as u8 });
            assert_eq!(model.executions.decode, [cuteafd_core::DecodeExecution { capacity: 64, compiled_capacity: 80 }]);
            assert_eq!(model.endpoints[0].capacity, unit as u32);
            for (unit, waves) in [(4096, 2), (2752, 3), (2048, 4)] {
                args.prefill_rows = unit;
                args.prefill_waves = waves;
                let shape = args.shape()?;
                assert_eq!((shape.unit_rows, shape.waves), (u64::from(unit), u64::from(waves)));
                args.validate()?;
            }
            args.prefill_rows = 512;
            for waves in 1..=4 {
                args.prefill_waves = waves;
                args.validate()?;
                let shape = args.shape()?;
                assert_eq!((shape.unit_rows, shape.waves, shape.span_rows),
                    (512, u64::from(waves), 512 * u64::from(waves)));
            }
            args.prefill_rows = 4097;
            assert!(args.validate().is_err());
            args.prefill_rows = 2753;
            assert!(args.validate().is_err());
        }
        Ok(())
    }


    fn planned(family: &str, backend: cuteafd_loader::plan::spark_executions::SparkBackend, decode: Option<u32>,
        unit: u64, waves: u64) -> cuteafd_core::SparkExecutions {
        let options = cuteafd_loader::plan::layout::LayoutOptions { prefill_unit_rows: Some(unit),
            prefill_waves: waves, ..Default::default() };
        cuteafd_loader::plan::layout::planned_spark_executions(family, backend, decode, &options).unwrap()
    }

    #[test]
    fn planner_serve_and_golden_admit_identical_spark_owners() -> Result<()> {
        use cuteafd_loader::plan::spark_executions::SparkBackend;
        for (waves, unit) in [(1u64, 4096u32), (2, 4096), (3, 2752), (4, 2048), (3, 640)] {
            let endpoint = SparkEndpoint { ring_depth: None, capacity: unit, ingress: SparkIngress::Bf16 };
            // Serve and golden share `spark_model`; the planner charges the same owners.
            assert_eq!(spark_model(&SparkBackend::Fp8.exports(), endpoint, waves, 1, 64 << 20)?.executions,
                planned("step5", SparkBackend::Fp8, None, u64::from(unit), waves));
        }
        Ok(())
    }

    #[test]
    fn installed_capacity_is_not_live_rows_or_assumed_fp8_format() -> Result<()> {
        let mut metadata = serde_json::json!({"layouts":{"tp4":{"hidden":4096,"experts":352,
            "top_k":8,"intermediate":1536,"capacities":[{"capacity":1},{"capacity":1024}]}}});
        assert_eq!(package_exports(&metadata)?, [1, 1024]);
        // A fixed unit inside the package's largest export is admitted there;
        // one beyond it is refused, never stretched.
        let endpoint = |capacity| SparkEndpoint { ring_depth: None, capacity, ingress: SparkIngress::Bf16 };
        let admitted = spark_model(&package_exports(&metadata)?, endpoint(256), 1, 1, 64 << 20)?;
        assert_eq!(admitted.executions.prefill.compiled_capacity, 1024);
        let error = spark_model(&package_exports(&metadata)?, endpoint(2048), 1, 1, 64 << 20).unwrap_err();
        assert!(matches!(error.downcast_ref::<cuteafd_core::ExecutionAdmissionError>(),
            Some(cuteafd_core::ExecutionAdmissionError::Uncovered { owner: "prefill", rows: 2048, .. })), "{error:#}");
        metadata["layouts"]["tp6"] = metadata["layouts"]["tp4"].clone();
        metadata["layouts"]["tp6"]["capacities"][1]["capacity"] = 4096.into();
        assert!(package_exports(&metadata).is_err());
        assert!(package_exports(&serde_json::json!({"layouts":{}})).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod load_order_tests {
    /// Serving admits every coordinator owner, plans the Spark admission from
    /// headers, records (pushing) or starts first, loads the native owners
    /// while the Sparks load (`load_while_admitting`: a Spark failure returns
    /// its error after the coordinator owners drop), and connects the Spark
    /// transport only after admission completed.
    #[test]
    fn coordinator_loads_while_admitting_and_connects_after() {
        let source = include_str!("serve.rs");
        let start = source.find("fn with_backend<T>").unwrap();
        let body = &source[start..start + source[start..].find("\nfn worker(").unwrap()];
        let at = |needle: &str| body.find(needle).unwrap_or_else(|| panic!("missing {needle}"));
        assert!(at("admission::admit_mtp(") < at("let spark_admission = SparkAdmission::plan("));
        assert!(at("let spark_admission = SparkAdmission::plan(") < at("Execution::load_native("));
        assert!(at("Execution::load_native(") < at("spark_admission.start_with("));
        assert!(at("spark_admission.start_with(") < at("load_while_admitting(Some(pending)"));
        assert!(at("load_while_admitting(Some(pending)") < at("require_peer_clients("));
        assert!(at("require_peer_clients(") < at("SparkLink::new_waves("));
        assert!(at("SparkLink::new_waves(") < at("Execution::bind_native("));
        assert!(!body.contains(".select(&args.peers)"), "admission must not block before the coordinator load");
    }
}
