//! Hy4-flash HTTP/request lifecycle (`serve-hy4f`) over the shared scheduler.
use super::backend::Hy4Backend;
use super::execution::Hy4Engine;
use crate::shared::{
    experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy},
    inventory::{InventoryPhase, RuntimeInventory},
    memory::device::Device,
    prefix::{budget::HostBudget, PrefixArgs},
    qwen::scheduler::{self, Backend},
    token_io::{EmbedPlacement, SelectPlacement, TokenIoArgs},
};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{chat::hy4_flash::Hy4Encoding, ModelEncoding, ModelProfile, NativeRequest};
use cuteafd_engine::prefix::PrefixFamily;
use cuteafd_loader::{
    families::hy4_flash::{execution::PAGE_ROWS, Hy4FlashConfig, Hy4KvCache},
    plan::Checkpoint,
};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub attention: crate::cli::CoreViewArgs,
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long)]
    pub split_device: Option<i32>,
    #[arg(long, env = "CUTEAFD_ATTENTION_PLACEMENT", default_value = "auto",
          value_parser = ["auto", "heads", "layers", "context"])]
    pub attention_placement: String,
    #[arg(long, value_delimiter = ',')]
    pub peers: Vec<SocketAddr>,
    #[arg(long)]
    pub pool_tokens: Option<u64>,
    #[arg(long = "kv-cache", default_value = "fp8")]
    pub kv_format: Hy4KvCache,
    #[arg(long = "max-context-tokens", value_parser = crate::shared::context::parse_tokens::<usize>)]
    pub max_context: Option<usize>,
    #[arg(long, default_value = "auto")]
    pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    #[arg(long, default_value_t = 16)]
    pub max_sequences: usize,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long)]
    pub model_id: Option<String>,
    #[arg(long)]
    pub prefill_rows: Option<usize>,
    #[arg(long, default_value = "auto")]
    pub decode_rows: cuteafd_loader::plan::decode_rows::DecodeRowsSetting,
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=3))]
    pub prefill_waves: Option<u8>,
    /// `exact` needs an attested coordinator route cap, which Hy4 does not report yet.
    #[arg(long, value_enum, default_value = "unrestricted")]
    pub prefill_packing: crate::shared::prefill_packing::PackingMode,
    /// Optional explicit path to `exl3-hy4f-k45-mul1/rtx-tp{1,2}` package directory.
    #[arg(long)]
    pub expert_package: Option<PathBuf>,
    #[command(flatten)]
    pub prefix: PrefixArgs,
    #[command(flatten)]
    pub draft_length: crate::shared::draft::length::DraftLength,
    #[command(flatten)]
    pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)]
    pub loop_abort: crate::shared::loop_abort::LoopAbortArgs,
    #[command(flatten)]
    pub token_io: TokenIoArgs,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
}

impl ServeArgs {
    fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        Ok(prefill_shape(
            self.peers.len(),
            cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::hy4_flash::HY4_FLASH),
            PrefillOverrides {
                unit_rows: self.prefill_rows.map(|rows| rows as u64),
                waves: self.prefill_waves.map(u64::from),
                span_rows: None,
                reserve_rows: None,
            },
        )?)
    }
}

fn resolve_expert_package(native_lib: &Path, explicit: Option<&Path>, gpus: usize, catalog: &cuteafd_loader::OfficialV41Catalog) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    let exl3 = catalog.exl3().context("Hy4-flash checkpoint has no EXL3 routed expert catalog")?;
    let base = native_lib.parent().context("Hy4 native library directory absent")?;
    Ok(base.join("exl3").join(exl3.package_name("hy4f")).join(format!("rtx-tp{gpus}")))
}

fn resolve_attention_placement(gpus: usize, setting: &str) -> Result<&'static str> {
    match (gpus, setting) {
        (1, "auto" | "heads") => Ok("whole"),
        (1, other) => anyhow::bail!(
            "Hy4-flash single-GPU serving does not support ATTENTION_PLACEMENT={other}; use auto or heads"
        ),
        (2, "heads") => Ok("heads"),
        (2, "auto" | "context") => anyhow::bail!(
            "Hy4-flash 2-GPU default ContextSplit serving is not implemented yet: missing SparkInfer partial-attention LSE + lse_combine2 (hy4_mla.py), cross-GPU scored_index_topk (hy4_indexer.py), and 32-head ContextSplit execution in Hy4Engine (#2579). Use explicit ATTENTION_PLACEMENT=heads to opt into replicated-KV head-split, or serve on 1 GPU."
        ),
        (2, other) => anyhow::bail!(
            "Hy4-flash 2-GPU serving does not support ATTENTION_PLACEMENT={other}; ContextSplit is the default (pending #2579) and heads is explicit opt-in"
        ),
        _ => anyhow::bail!("Hy4-flash supports 1 or 2 coordinator GPUs, got {gpus}"),
    }
}

pub(crate) async fn run(mut args: ServeArgs) -> Result<()> {
    cuteafd_core::concurrency::validate("hy4_flash", args.max_sequences as u64)?;
    args.prefill_packing.require_unrestricted("Hy4-flash")?;
    ensure!(
        args.token_io.embed_placement == EmbedPlacement::Gpu && args.token_io.token_select == SelectPlacement::Device,
        "Hy4-flash serving requires GPU embedding and device token selection"
    );
    ensure!(
        args.peers.len() <= 6 && args.peers.iter().collect::<std::collections::BTreeSet<_>>().len() == args.peers.len(),
        "Hy4-flash supports 0..=6 distinct Spark peer endpoints"
    );
    let gpus = if args.split_device.is_some() { 2 } else { 1 };
    let decode_rows = cuteafd_loader::plan::resolve_decode_rows("hy4_flash",
        cuteafd_loader::plan::decode_rows::DecodeRowsRequest {
            requested: args.decode_rows.as_option(), concurrency: args.max_sequences as u64,
            decode_lanes: 1, rtx_gpus: gpus, draft_length: Some(0),
        })? as usize;
    let resolved_attention = resolve_attention_placement(gpus, args.attention_placement.as_str())?;
    args.prefix.resolve_entries(args.max_sequences);
    let prefill = args.prefill()?;
    cuteafd_bench::context::ResolvedSettings::prefill(prefill)
        .value("decode-rows", decode_rows)
        .value("attention-placement", resolved_attention)
        .publish();

    let encoding = ModelEncoding::Hy4Flash(Arc::new(Hy4Encoding::from_snapshot(&args.snapshot)?));
    let eos = encoding.eos_token_ids();
    ensure!(!eos.is_empty(), "Hy4-flash checkpoint resolved empty EOS set");
    let model = args
        .model_id
        .clone()
        .or_else(|| cuteafd_api::openai::snapshot_model_id(&args.snapshot))
        .context("Hy4-flash snapshot without HF identity requires --model-id")?;
    let profile = ModelProfile::new(model, encoding);
    let api = args.api.load()?;
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(
        cuteafd_core::concurrency::queue_depth(args.max_sequences as u64) as usize,
    );
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let worker_args = args.clone();

    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| {
            let view = worker_args
                .attention
                .selected_view
                .clone()
                .map(Ok)
                .unwrap_or_else(|| worker_args.attention.select(&worker_args.snapshot))?;
            worker_args.attention.validate_view(&view)?;
            let checkpoint = Checkpoint::open_view(&view)?;
            let cfg = Hy4FlashConfig::from_hf(&checkpoint.config)?;
            cfg.release_geometry()?;
            cuteafd_bench::context::set_family("hy4_flash");
            if let Err(prev) = cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::HY4_FLASH) {
                ensure!(
                    prev == cuteafd_core::ExpertGeometry::HY4_FLASH,
                    "Hy4-flash process geometry already fixed to {prev:?}"
                );
            }

            // SAFETY: every device owner retires inside this worker before NativeLibrary drops.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;

            let mut device_ids = vec![worker_args.device];
            if let Some(second) = worker_args.split_device {
                ensure!(second != worker_args.device, "--split-device must differ from --device");
                device_ids.push(second);
            }
            let gpus = device_ids.len();
            let devices: Vec<Device<'_>> = device_ids.iter().map(|&id| Device { library: &library, id }).collect();
            let unit_rows = usize::try_from(prefill.unit_rows)?;
            let capacity_rows = unit_rows.max(decode_rows);
            let max_context = match worker_args.max_context {
                Some(0) | None => cfg.max_position,
                Some(n) => n.min(cfg.max_position),
            };
            let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
            let routed_layers = cfg.routed_layers();
            let budget = usize::try_from(crate::shared::memory::budget::resident_weight_limit().0)
                .unwrap_or(80usize << 30);

            let pre = RuntimeInventory::measure(&library, &device_ids, InventoryPhase::PreWeight, |_| Ok(()))?;
            let min_free = pre.gpus.iter().map(|g| g.free_bytes).min().unwrap_or(0);
            let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
            let effective_cap = pre
                .gpus
                .iter()
                .map(|g| policy.gpu_cap_bytes(g.total_bytes).min(g.free_bytes))
                .min()
                .unwrap_or(min_free);

            let model = cuteafd_loader::plan::families::hy4_flash::Model::open(&checkpoint)?;
            let modes = cuteafd_loader::placement::families::hy4_flash::layer_modes(
                &cfg,
                gpus,
                cuteafd_loader::placement::AttentionPlacement::Heads,
            );
            let target_layouts = cuteafd_loader::families::hy4_flash::weight_layout::target_arenas(
                &checkpoint,
                &cfg,
                &model.core,
                &modes,
                gpus,
            )?;
            let target_bytes = target_layouts.iter().map(|l| l.arena.bytes() as u64).max().unwrap_or(0);

            // Resolve how many routed layers are onboarded on RTX vs dispatched to Spark peers.
            let onboard_count = if worker_args.peers.is_empty() {
                routed_layers.len()
            } else {
                match worker_args.rtx_expert_layers.resolve(routed_layers.len()) {
                    cuteafd_loader::placement::Onboard::Layers(n) => n.min(routed_layers.len()),
                    cuteafd_loader::placement::Onboard::Auto => {
                        let per_layer_bytes = catalog
                            .exl3()
                            .and_then(|e| e.residency(cuteafd_loader::V41Exl3Layer::Backbone(routed_layers[0]), gpus, 0).ok())
                            .and_then(|r| r.device_arena_layout().ok())
                            .map(|(_, b)| b as u64)
                            .unwrap_or(1_400_000_000 / gpus as u64);
                        // Leave headroom for target weights + KV + workspaces + Spark links.
                        let weight_room = (budget as u64)
                            .min(effective_cap.saturating_sub(14u64 << 30))
                            .saturating_sub(target_bytes);
                        let fit = (weight_room / per_layer_bytes.max(1)) as usize;
                        // Keep at least 1 layer on Sparks when Spark peers are configured.
                        fit.min(routed_layers.len().saturating_sub(1))
                    }
                }
            };
            let onboarded_layers = routed_layers[..onboard_count].to_vec();
            let spark_layers = routed_layers[onboard_count..].to_vec();

            // Admit and bootstrap Spark workers for non-onboarded routed layers before GPU allocation.
            let spark_link = if !worker_args.peers.is_empty() {
                use crate::shared::{
                    placement::{spark_rings, SparkAdmission},
                    spark_intake::SparkLink,
                };
                use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
                use cuteafd_loader::placement::{
                    families::hy4_flash as hy4_place,
                    planner::PrefillShape,
                    SparkWorkingSet,
                };
                use cuteafd_transport::{
                    expert::{SparkWorld, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
                    worker_selection::ModelAdmission,
                    ExpertV2Dtype, SparkSessionShape, TcpTransportConfig,
                };
                let working = SparkWorkingSet {
                    rtx_layers: onboarded_layers.clone(),
                    spark_layers: spark_layers.clone(),
                    spark_ranks: worker_args.peers.len(),
                    prefill_lanes: hy4_place::LANES,
                };
                let endpoint = SparkEndpoint {
                    ring_depth: None,
                    capacity: u32::try_from(capacity_rows)?,
                    ingress: SparkIngress::Fp8K32,
                    partial_reply_bytes: None,
                };
                let tcp = TcpTransportConfig::default();
                let backend_exec = cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?;
                let executions = cuteafd_loader::plan::spark_executions::admit_family(
                    "hy4_flash",
                    &backend_exec.exports(),
                    None,
                    prefill.unit_rows,
                    prefill.waves,
                )?;
                let model_adm = ModelAdmission::for_clients(
                    ExpertGeometry::HY4_FLASH,
                    endpoint,
                    spark_rings(tcp.max_frame_bytes)?,
                    1,
                    executions,
                )?;
                SparkAdmission::plan_view(
                    &view,
                    ExpertGeometry::HY4_FLASH,
                    Some(&catalog),
                    &working,
                    worker_args.peers.len(),
                    model_adm,
                    None,
                )?
                .select(&worker_args.peers)?;

                let shape = PrefillShape { rows: capacity_rows as u64, waves: prefill.waves };
                let links = hy4_place::spark_link(
                    0,
                    worker_args.peers.len(),
                    shape,
                    cuteafd_core::spark_reply::requested(),
                )?;
                let geometry = links.into_iter().next().context("Hy4 Spark link geometry absent")?;
                let mut link = devices[0].run(|| {
                    SparkLink::prepared(
                        &library,
                        SparkWorld::canonical(worker_args.peers.len())?,
                        &worker_args.peers,
                        tcp,
                        geometry,
                    )
                })?;
                link.enable_terminal_ownership(Arc::clone(&library))?;
                let mut session = SparkSessionShape::new(
                    u32::try_from(capacity_rows)?,
                    u32::try_from(cfg.routed_expert_hidden)?,
                    u32::try_from(cfg.topk)?,
                    ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                )?;
                session.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
                devices[0].run(|| link.bootstrap(&session))?;
                Some(link)
            } else {
                None
            };

            let onboarded: Vec<Option<Exl3Tp2<'_>>> = if onboarded_layers.is_empty() {
                (0..gpus).map(|_| None).collect()
            } else {
                let expert_pkg = resolve_expert_package(
                    &worker_args.native_lib,
                    worker_args.expert_package.as_deref(),
                    gpus,
                    &catalog,
                )?;
                if gpus == 1 {
                    let w = Exl3Tp2::load_whole_weights(
                        devices[0],
                        &catalog,
                        &expert_pkg,
                        onboarded_layers.clone(),
                        capacity_rows,
                        budget,
                        RtxSelectionPolicy::FirstCapacity,
                    )?;
                    vec![Some(w.execution()?)]
                } else {
                    let [w0, w1] = Exl3Tp2::load_pair_weights(
                        [devices[0], devices[1]],
                        &catalog,
                        cuteafd_core::ExpertGeometry::HY4_FLASH,
                        [expert_pkg.as_path(), expert_pkg.as_path()],
                        onboarded_layers.clone(),
                        capacity_rows,
                        [budget, budget],
                        RtxSelectionPolicy::FirstCapacity,
                    )?;
                    vec![Some(w0.execution()?), Some(w1.execution()?)]
                }
            };

            // Measure remaining GPU free memory after routed experts, then size DSA KV pool safely.
            let post_exp = RuntimeInventory::measure(&library, &device_ids, InventoryPhase::PreWeight, |_| Ok(()))?;
            let after_exp_free = post_exp
                .gpus
                .iter()
                .map(|g| g.free_bytes.min(effective_cap))
                .min()
                .unwrap_or(0);
            let swa_ring_pages = ((cfg.sliding_window + capacity_rows).div_ceil(PAGE_ROWS) + 1)
                .max(cuteafd_loader::families::hy4_flash::execution::SWA_RING_PAGES);
            let swa_seq_bytes = (cfg.swa_layers().len() * swa_ring_pages * PAGE_ROWS * worker_args.kv_format.record_bytes()) as u64;
            let cache_layout = cuteafd_loader::families::hy4_flash::execution::CacheLayout::new(
                &cfg,
                worker_args.kv_format,
                worker_args.max_sequences,
                cfg.sparse_layers().len(),
                cfg.swa_layers().len(),
            )?;
            let mark_bytes = cache_layout.mark_bytes_per_slot as u64;
            let mark_slots = if worker_args.prefix.prefix_cache_entries == 0 {
                0
            } else {
                cuteafd_core::prefix::mark_slots_with_host(
                    worker_args.max_sequences as u64,
                    worker_args.prefix.prefix_cache_entries as u64,
                    mark_bytes,
                    worker_args.prefix.prefix_cache_mark_mib as u64 * (1 << 20),
                    worker_args.prefix.host_cache_bytes != HostBudget::Bytes(0),
                ) as usize
            };
            let est_seq_pages = max_context.div_ceil(PAGE_ROWS).max(16);
            let rank_ws_bytes = Hy4Engine::estimated_rank_workspace_bytes(
                &cfg,
                gpus,
                capacity_rows,
                decode_rows,
                est_seq_pages,
                worker_args.max_sequences,
            );
            let reserve_and_ws: u64 = target_bytes
                + swa_seq_bytes * worker_args.max_sequences as u64
                + mark_bytes * mark_slots as u64
                + rank_ws_bytes
                + (1u64 << 30); // CUDA driver/runtime headroom + selector + fragmentation tail
            let kv_avail = after_exp_free.saturating_sub(reserve_and_ws);
            let page_bytes_per_rank = (cfg.sparse_layers().len()
                * (PAGE_ROWS * worker_args.kv_format.record_bytes()
                    + (PAGE_ROWS / cfg.indexer_compress_ratio) * cuteafd_loader::families::hy4_flash::execution::INDEX_GROUP_RECORD_BYTES)) as u64;
            let max_fit_pages = (kv_avail / page_bytes_per_rank.max(1)) as usize;
            let min_pages = max_context.div_ceil(PAGE_ROWS);
            let chosen_pages = match worker_args.pool_tokens {
                Some(tokens) if tokens > 0 => {
                    let req = usize::try_from(tokens.div_ceil(PAGE_ROWS as u64))?;
                    ensure!(
                        req <= max_fit_pages,
                        "Hy4 requested pool {tokens} tokens ({req} pages) exceeds fitted GPU capacity ({max_fit_pages} pages)"
                    );
                    req
                }
                _ => {
                    let cap_pages = policy
                        .kv_max
                        .map(|t| usize::try_from(t.div_ceil(PAGE_ROWS as u64)).unwrap_or(max_fit_pages))
                        .unwrap_or(max_fit_pages);
                    max_fit_pages.min(cap_pages)
                }
            };
            ensure!(
                chosen_pages >= min_pages.min(256),
                "Hy4 fitted KV pool ({chosen_pages} pages) is below minimum usable context"
            );
            let served_context = max_context.min(chosen_pages * PAGE_ROWS);
            let pool_tokens = (chosen_pages * PAGE_ROWS) as u64;

            let mut engine = Hy4Engine::new_serving(
                &devices,
                &programs,
                &checkpoint,
                cfg.clone(),
                worker_args.kv_format,
                unit_rows,
                decode_rows,
                usize::try_from(prefill.waves)?,
                chosen_pages,
                worker_args.max_sequences,
                served_context,
                mark_slots,
                false,
                onboarded,
            )?;
            if let Some(link) = spark_link {
                engine.attach_spark(link, onboarded_layers.clone());
            }
            let graph_layout = cuteafd_loader::serving_capacity::hy4_graphs::Hy4GraphLayout::resolve(
                gpus,
                worker_args.peers.len(),
            );
            engine.configure_graphs(graph_layout)?;

            let family = engine.prefix()?;
            let host_total = cuteafd_loader::formats::mapped_preload::host_memory_before_process()
                .map(|(total, _)| total)
                .unwrap_or(0);
            let host_quota = match worker_args.prefix.host_cache_bytes {
                HostBudget::Bytes(0) => 0,
                HostBudget::Bytes(b) => b,
                HostBudget::Auto => {
                    let layout = family.layout();
                    let geom = cuteafd_core::prefix::HostCacheGeometry {
                        unit_rows: PAGE_ROWS as u64,
                        page_bytes: layout.page_bytes as u64,
                        mark_bytes: layout.mark_bytes as u64,
                        draft_bytes: 0,
                    };
                    geom.resolve(None, served_context as u64, worker_args.max_sequences as u64, host_total, geom.chunk_bytes())
                        .unwrap_or(0)
                }
            };
            let host = (host_quota > 0 && worker_args.prefix.prefix_cache_entries > 0)
                .then(|| worker_args.prefix.host_config_for(family.layout(), served_context, host_quota))
                .transpose()?;

            cuteafd_bench::context::ResolvedSettings::default()
                .value("max-context-tokens", served_context)
                .value("pool-tokens", pool_tokens)
                .value("rtx-expert-layers", onboarded_layers.len())
                .value("prefill-rows", unit_rows)
                .value("prefill-waves", prefill.waves)
                .publish();

            let backend = Hy4Backend::new(engine, served_context, unit_rows, host)?;
            backend.warm_decode_graphs(graph_layout)?;
            let mut media_admission = cuteafd_engine::media::MediaAdmission::new(
                cuteafd_engine::media::EmbeddingCache::new(0),
                crate::shared::qwen::Encoder::Off,
                worker_args.max_sequences,
            );
            library.cuda_set_device(worker_args.device)?;
            scheduler::schedule(
                &backend,
                &worker_args.snapshot,
                &mut receive,
                &worker_stats,
                worker_args.max_sequences,
                scheduler::Drafts::None,
                worker_args.draft_length,
                eos,
                worker_args.decode_share,
                &worker_args.prefix,
                worker_args.loop_abort.config(),
                worker_args.token_io.token_select,
                &mut media_admission,
                None,
                || {
                    backend.require_live()?;
                    RuntimeInventory::measure(&library, &device_ids, InventoryPhase::PostLoad, |_| Ok(()))?
                        .record("hy4_flash-serving-loaded", &[]);
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok(served_context));
                    }
                    Ok(())
                },
            )
        })();
        crate::shared::serve::forward_initialization(
            result,
            ready.is_some(),
            "Hy4-flash initialization failed",
            |error| {
                ready
                    .take()
                    .expect("pending readiness")
                    .send(Err(error))
                    .map_err(|r| match r {
                        Err(e) => e,
                        Ok(_) => unreachable!("sent error"),
                    })
            },
        )
    });

    let context = ready_rx
        .await
        .context("Hy4-flash worker exited before readiness")??;
    let limits = args.prefix.reserve.limits(context)?;
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(false, || {
        Ok(crate::shared::console::Layout::new(
            "hy4_flash",
            profile.id.clone(),
            snapshot.clone(),
        ))
    });
    let router = cuteafd_api::openai::router_for_model(
        queue,
        limits,
        stats,
        std::time::Duration::from_secs(25),
        hub.clone(),
        api.serve(profile, &snapshot)?,
    );
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    let mut worker = Some(worker);
    tokio::select! {
        served = crate::shared::api::serve_http(listener, api.app(router, hub), "hy4_flash") => served?,
        _ = crate::shared::api::watch_scheduler(&mut worker) => unreachable!(),
    }
    crate::shared::api::finish_scheduler(&mut worker, "hy4_flash").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn serve_hy4f_cli_and_resolved_seam_match() -> anyhow::Result<()> {
        let argv = [
            "cuteafd",
            "--gpu-memory-cap",
            "97250MiB",
            "serve-hy4f",
            "--snapshot",
            "/model",
            "--native-lib",
            "/opt/cuteafd/lib/libcuteafd_native.so",
            "--device",
            "0",
            "--split-device",
            "1",
            "--listen",
            "0.0.0.0:8823",
            "--max-sequences",
            "16",
            "--max-context-tokens",
            "auto",
            "--core-view",
            "original",
            "--prefill-rows",
            "2048",
            "--prefill-waves",
            "1",
        ];
        let crate::cli::Commands::ServeHy4f(mut args) = crate::cli::Cli::try_parse_from(argv)?.command else {
            anyhow::bail!("Hy4-flash command not parsed");
        };
        assert_eq!((args.max_context, args.max_sequences, args.split_device), (Some(0), 16, Some(1)));
        crate::commands::bench::assert_resolved_family_seam("serve-hy4f", &[], || {
            args.prefix.resolve_entries(args.max_sequences);
            let prefill = args.prefill()?;
            cuteafd_bench::context::ResolvedSettings::default()
                .value("max-context-tokens", 262144)
                .value("pool-tokens", 262144)
                .value("rtx-expert-layers", 41)
                .value("attention-placement", "heads")
                .value("prefill-rows", prefill.unit_rows)
                .value("prefill-waves", prefill.waves)
                .publish();
            Ok(())
        })?;
        Ok(())
    }

    #[tokio::test]
    async fn serve_hy4f_refuses_wide_decode_before_opening_checkpoint() -> anyhow::Result<()> {
        let base = ["cuteafd", "serve-hy4f", "--snapshot", "/not-opened", "--native-lib", "/not-opened.so",
            "--decode-rows", "128", "--attention-placement", "heads"];
        for extra in [&[][..], &["--split-device", "1"][..]] {
            let crate::cli::Commands::ServeHy4f(args) = crate::cli::Cli::try_parse_from(base.iter().chain(extra))?.command
                else { anyhow::bail!("not serve-hy4f") };
            let error = super::run(args).await.unwrap_err();
            assert!(error.to_string().contains("Hy4-flash split-K decode exports"), "{error}");
        }
        Ok(())
    }

    #[test]
    fn serve_hy4f_accepts_unrestricted_packing_and_refuses_exact_by_name() -> anyhow::Result<()> {
        use crate::shared::prefill_packing::PackingMode;
        let base = ["cuteafd", "serve-hy4f", "--snapshot", "/model", "--native-lib", "/lib.so"];
        for (extra, mode) in [(&[][..], PackingMode::Unrestricted),
            (&["--prefill-packing", "unrestricted"][..], PackingMode::Unrestricted),
            (&["--prefill-packing", "exact"][..], PackingMode::Exact)] {
            let crate::cli::Commands::ServeHy4f(args) =
                crate::cli::Cli::try_parse_from(base.iter().chain(extra))?.command else { anyhow::bail!("not serve-hy4f") };
            assert_eq!(args.prefill_packing, mode);
            let checked = args.prefill_packing.require_unrestricted("Hy4-flash");
            assert_eq!(checked.is_ok(), mode == PackingMode::Unrestricted);
            if let Err(error) = checked { assert!(error.to_string().contains("PREFILL_PACKING=exact")); }
        }
        Ok(())
    }

    #[test]
    fn serve_hy4f_spark_peers_cli_parses() -> anyhow::Result<()> {
        for (peers_csv, expected) in [
            ("10.55.0.1:19441", 1usize),
            ("10.55.0.1:19441,10.55.0.2:19441,10.55.0.3:19441,10.55.0.4:19441", 4usize),
        ] {
            let argv = [
                "cuteafd",
                "serve-hy4f",
                "--snapshot",
                "/model",
                "--native-lib",
                "/opt/cuteafd/lib/libcuteafd_native.so",
                "--device",
                "0",
                "--peers",
                peers_csv,
                "--core-view",
                "k6-compact",
                "--rtx-expert-layers",
                "auto",
            ];
            let crate::cli::Commands::ServeHy4f(args) = crate::cli::Cli::try_parse_from(argv)?.command else {
                anyhow::bail!("Hy4-flash Spark CLI not parsed");
            };
            assert_eq!(args.peers.len(), expected);
            assert!(args.split_device.is_none());
        }
        Ok(())
    }

    #[test]
    fn attention_placement_refuses_2gpu_context_loudly_and_allows_explicit_heads() -> anyhow::Result<()> {
        assert_eq!(super::resolve_attention_placement(1, "auto")?, "whole");
        assert_eq!(super::resolve_attention_placement(1, "heads")?, "whole");
        assert!(super::resolve_attention_placement(1, "context").is_err());
        assert_eq!(super::resolve_attention_placement(2, "heads")?, "heads");
        let err_auto = super::resolve_attention_placement(2, "auto").unwrap_err().to_string();
        let err_ctx = super::resolve_attention_placement(2, "context").unwrap_err().to_string();
        assert!(err_auto.contains("ContextSplit") && err_auto.contains("ATTENTION_PLACEMENT=heads"), "{err_auto}");
        assert!(err_ctx.contains("ContextSplit") && err_ctx.contains("ATTENTION_PLACEMENT=heads"), "{err_ctx}");
        assert!(super::resolve_attention_placement(2, "layers").is_err());
        Ok(())
    }

    #[test]
    fn shared_scheduler_owns_lanes_and_typed_readiness() {
        let source = include_str!("serve.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("scheduler::schedule("));
        assert!(source.contains("Checkpoint::open_view(&view)?"));
        assert!(source.contains("context::set_family(\"hy4_flash\")"));
        assert!(source.contains("shared::serve::forward_initialization("));
        assert!(source.contains("finish_scheduler(&mut worker"));
        assert!(source.find("Hy4Encoding::from_snapshot").unwrap() < source.find("NativeLibrary::load").unwrap());
    }
}
