//! Hy4 (`hy_v4_vl`) HTTP/request lifecycle (`serve-hy4`) over the shared scheduler
//! and shared contract admission (`Hy4Inputs::admit`).
use super::backend::Hy4Backend;
use super::execution::Hy4Engine;
use super::programs::Hy4Programs;
use crate::shared::{
    experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy},
    inventory::{InventoryPhase, RuntimeInventory},
    memory::{device::Device, receipts::Inventory},
    prefix::{budget::HostBudget, PrefixArgs},
    qwen::scheduler::{self, Backend},
    token_io::{EmbedPlacement, SelectPlacement, TokenIoArgs},
};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{chat::hy4_flash::Hy4Encoding, ModelEncoding, ModelProfile, NativeRequest};
use cuteafd_core::memory_layout::Category;
use cuteafd_engine::prefix::PrefixFamily;
use cuteafd_loader::{
    families::hy4::{
        admission::{self, Coordinator, FactsRequest},
        graph_reserve, Hy4Config, Hy4KvCache,
    },
    placement::{
        families::hy4::{self as hy4_place, AdmittedHy4, Hy4Inputs},
        planner::{
            ComponentKind, CoordinatorDevices, Lifetime, Location, Operation, Phase, PrefillShape,
            Receipt, Recipe, ResolvedPlan,
        },
        ArchContext, AttentionPlacement, Onboard,
    },
    plan::Checkpoint,
};
use std::{
    collections::BTreeMap,
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
    #[arg(
        long,
        env = "CUTEAFD_ATTENTION_PLACEMENT",
        default_value = "auto",
        value_parser = ["auto", "heads", "layers", "context"]
    )]
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
    #[arg(long, value_enum, default_value = "unrestricted")]
    pub prefill_packing: crate::shared::prefill_packing::PackingMode,
    /// Optional explicit path to `exl3-hy4-k45-mul1/rtx-tp{1,2}` package directory.
    #[arg(long)]
    pub expert_package: Option<PathBuf>,
    #[command(flatten)]
    pub media: crate::shared::vision::still::Options,
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
    #[command(flatten)]
    pub console: crate::shared::console::ConsoleArgs,
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true
    )]
    pub expert_usage: bool,
    #[arg(skip)]
    pub expert_usage_reader: cuteafd_core::expert_usage::Reader,
}

impl ServeArgs {
    fn resolved_decode_rows(&self, gpus: usize) -> Result<usize> {
        Ok(cuteafd_loader::plan::resolve_decode_rows(
            "hy4",
            cuteafd_loader::plan::decode_rows::DecodeRowsRequest {
                requested: self.decode_rows.as_option(),
                concurrency: self.max_sequences as u64,
                decode_lanes: 1,
                rtx_gpus: gpus,
                draft_length: Some(0),
            },
        )? as usize)
    }

    fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        Ok(prefill_shape(
            self.peers.len(),
            cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::hy4::HY4),
            PrefillOverrides {
                unit_rows: self.prefill_rows.map(|rows| rows as u64),
                waves: self.prefill_waves.map(u64::from),
                span_rows: None,
                reserve_rows: None,
            },
        )?)
    }
}

pub(super) fn resolve_expert_package(
    native_lib: &Path,
    explicit: Option<&Path>,
    gpus: usize,
    catalog: &cuteafd_loader::OfficialV41Catalog,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    let exl3 = catalog
        .exl3()
        .context("Hy4 checkpoint has no EXL3 routed expert catalog")?;
    let base = native_lib
        .parent()
        .context("Hy4 native library directory absent")?;
    Ok(base
        .join("exl3")
        .join(exl3.package_name("hy4"))
        .join(format!("rtx-tp{gpus}")))
}

pub(super) fn resolve_attention_placement(
    gpus: usize,
    setting: &str,
) -> Result<(&'static str, AttentionPlacement)> {
    match (gpus, setting) {
        (1, "auto" | "heads") => Ok(("whole", AttentionPlacement::Heads)),
        (1, other) => anyhow::bail!(
            "Hy4 single-GPU serving does not support ATTENTION_PLACEMENT={other}; use auto or heads"
        ),
        (2, "auto" | "context") => Ok(("context", AttentionPlacement::Context)),
        (2, "heads") => Ok(("heads", AttentionPlacement::Heads)),
        (2, other) => anyhow::bail!(
            "Hy4 2-GPU serving does not support ATTENTION_PLACEMENT={other}; use auto/context (ContextSplit default) or explicit heads"
        ),
        _ => anyhow::bail!("Hy4 supports 1 or 2 coordinator GPUs, got {gpus}"),
    }
}

fn pending_code_allowance(
    sample: &cuteafd_loader::placement::RuntimeSample,
    rank: usize,
    split: bool,
) -> u64 {
    let arch = ArchContext::for_device(&sample.arch, sample.total_bytes);
    let code = cuteafd_loader::placement::inventory::loaded_code_allowance(
        "hy4",
        "exl3",
        split,
        rank as u8,
        arch,
        None,
    );
    let pending = code.pending(sample.context_bytes, arch.context_bytes);
    tracing::info!(
        device = sample.device,
        family = "hy4",
        loaded_code_bytes = code.bytes,
        untracked_bytes = sample.context_bytes,
        basis = ?code.basis,
        pending_bytes = pending,
        "Hy4 loaded code reserved beside admission"
    );
    pending
}

fn publish_provenance(
    args: &ServeArgs,
    runtime: &RuntimeInventory,
    admitted: &AdmittedHy4,
    attention: &str,
    graph_policies: &[cuteafd_loader::placement::graph_reserve::Resolved],
) -> Result<()> {
    use sha2::{Digest, Sha256};
    let hash = |path: &Path| -> Result<String> {
        let mut file = std::fs::File::open(path)?;
        let mut digest = Sha256::new();
        std::io::copy(&mut file, &mut digest)?;
        Ok(format!("{:x}", digest.finalize()))
    };
    cuteafd_ffi::memory_ledger::fault::set_inputs(serde_json::json!({
        "memory_reconciliation_provenance": {
            "schema_version": 1,
            "family": "hy4",
            "source_commit": env!("CUTEAFD_BUILD_COMMIT"),
            "source_dirty": env!("CUTEAFD_BUILD_DIRTY"),
            "toolchain": env!("CUTEAFD_BUILD_TOOLCHAIN"),
            "native_sha256": hash(&args.native_lib)?,
            "programs_sha256": hash(&args.manifest)?,
            "layout": format!(
                "{} RTX/{} Sparks; {attention}; context/{}; slots/{}",
                runtime.gpus.len(),
                args.peers.len(),
                admitted.solved.served_context(),
                args.max_sequences
            ),
            "devices": runtime.gpus,
            "spark_world": args.peers.len(),
            "onboard_layers": admitted.onboard,
            "prefill": admitted.solved.decisions.prefill,
            "kv_cache": format!("{:?}", args.kv_format),
            "graph_basis": {
                "basis": "bytes",
                "reserve_policy": graph_policies
            }
        }
    }));
    Ok(())
}

/// Reconcile every retained target weight arena and request/scratch/cache backing
/// recorded by `Hy4Engine::inventory` against the solved `Hy4Inputs` admission plan.
fn reconcile_runtime_inventory(
    plan: &ResolvedPlan,
    inputs: &Hy4Inputs,
    weight_receipts: &[Receipt],
    retained_receipts: &[Receipt],
) -> Result<()> {
    let mut target_backings = 0usize;
    for receipt in weight_receipts.iter().filter(|r| r.allocation.view.is_none()) {
        let actual = &receipt.allocation;
        if actual.id.starts_with("hy4/target/gpu") {
            let planned = plan
                .facts
                .allocations
                .iter()
                .find(|owner| owner.view.is_none() && owner.id == actual.id)
                .with_context(|| format!("Hy4 target arena backing not admitted: {}", actual.id))?;
            ensure!(
                actual.bytes == planned.bytes && actual.location == planned.location,
                "Hy4 target arena backing {} differs: actual {} != admitted {}",
                actual.id,
                actual.bytes,
                planned.bytes
            );
            target_backings += 1;
        }
    }
    ensure!(
        target_backings == inputs.gpus,
        "Hy4 target arena backing count {target_backings} != {} GPUs",
        inputs.gpus
    );

    let contract = inputs.prepare(&plan.request)?;
    let selected = |id: &str| {
        id.starts_with("hy4.cache.")
            || id.starts_with("hy4.scratch.")
            || id.starts_with("hy4.wave.")
            || id.starts_with("hy4.head.")
            || id.starts_with("hy4.target.")
            || id.starts_with("hy4.peer.")
            || id.starts_with("hy4.context_exchange.")
            || id.starts_with("expert-usage.gpu")
    };
    let planned: BTreeMap<&str, &cuteafd_loader::placement::planner::Allocation> = plan
        .facts
        .allocations
        .iter()
        .filter(|a| {
            a.view.is_none()
                && selected(&a.id)
                && a.lifetime.contains(Phase::Ready, contract.phase_order)
        })
        .map(|a| (a.id.as_str(), a))
        .collect();

    let mut actual: BTreeMap<String, &Receipt> = BTreeMap::new();
    for receipt in retained_receipts
        .iter()
        .filter(|r| r.allocation.view.is_none() && r.allocation.location != Location::HOST)
    {
        let stripped = receipt
            .allocation
            .id
            .strip_prefix("retained.")
            .unwrap_or(&receipt.allocation.id);
        let canonical = stripped
            .strip_suffix(".storage.projection_arena")
            .unwrap_or(stripped)
            .to_owned();
        ensure!(
            actual.insert(canonical.clone(), receipt).is_none(),
            "Hy4 duplicate retained backing receipt {canonical}"
        );
    }

    ensure!(
        actual.len() == planned.len(),
        "Hy4 retained owner count differs: missing {:?}, extra {:?}",
        planned.keys().filter(|id| !actual.contains_key(**id)).collect::<Vec<_>>(),
        actual.keys().filter(|id| !planned.contains_key(id.as_str())).collect::<Vec<_>>()
    );

    let pages = plan.pool_tokens.div_ceil(plan.pool_page_rows);
    for (id, receipt) in actual {
        let owner = planned
            .get(id.as_str())
            .with_context(|| format!("Hy4 unadmitted retained owner {id}"))?;
        let curves: Vec<_> = contract
            .components
            .iter()
            .filter_map(|c| match &c.kind {
                ComponentKind::Allocation { template, .. } if template.id == id => Some(&c.bytes),
                _ => None,
            })
            .collect();
        let curve = curves
            .first()
            .with_context(|| format!("Hy4 retained owner {id} missing contract curve"))?;
        ensure!(
            curves.iter().all(|other| *other == *curve),
            "Hy4 retained owner {id} has ambiguous curves"
        );
        let piece = curve
            .0
            .iter()
            .find(|piece| (piece.first..=piece.last).contains(&pages))
            .with_context(|| format!("Hy4 retained owner {id} curve does not cover {pages} pages"))?;
        let payload = pages
            .checked_sub(piece.first)
            .and_then(|p| piece.bytes_per_page.checked_mul(p))
            .and_then(|b| piece.bytes_at_first.checked_add(b))
            .context("Hy4 retained owner payload overflow")?;
        ensure!(
            piece.backings == 1
                && receipt.address != 0
                && receipt.allocation.bytes == payload
                && receipt.allocation.location == owner.location
                && curve.at(pages)? == owner.bytes,
            "Hy4 retained owner {id} differs from admission: actual {}, requested {payload}, charged {}",
            receipt.allocation.bytes,
            owner.bytes
        );
    }
    Ok(())
}

pub(crate) async fn run(mut args: ServeArgs) -> Result<()> {
    cuteafd_core::concurrency::validate("hy4", args.max_sequences as u64)?;
    args.prefill_packing.require_unrestricted("Hy4")?;
    ensure!(
        args.token_io.embed_placement == EmbedPlacement::Gpu
            && args.token_io.token_select == SelectPlacement::Device,
        "Hy4 serving requires GPU embedding and device token selection"
    );
    ensure!(
        args.peers.len() <= 8
            && args.peers.iter().collect::<std::collections::BTreeSet<_>>().len() == args.peers.len(),
        "Hy4 supports 0..=8 distinct Spark peer endpoints"
    );
    let gpus = if args.split_device.is_some() { 2 } else { 1 };
    let decode_rows = args.resolved_decode_rows(gpus)?;
    let (resolved_attention, attention_placement) =
        resolve_attention_placement(gpus, args.attention_placement.as_str())?;
    args.prefix.resolve_entries(args.max_sequences);
    let prefill = args.prefill()?;
    cuteafd_bench::context::ResolvedSettings::prefill(prefill)
        .value("decode-rows", decode_rows)
        .value("attention-placement", resolved_attention)
        .publish();

    let encoding = ModelEncoding::Hy4Flash(Arc::new(Hy4Encoding::from_snapshot(&args.snapshot)?));
    let eos = encoding.eos_token_ids();
    ensure!(!eos.is_empty(), "Hy4 checkpoint resolved empty EOS set");
    let model = args
        .model_id
        .clone()
        .or_else(|| cuteafd_api::openai::snapshot_model_id(&args.snapshot))
        .context("Hy4 snapshot without HF identity requires --model-id")?;
    let expert_usage = args
        .expert_usage_reader
        .configure(args.expert_usage, false)
        .map_err(anyhow::Error::msg)?;
    let mut profile = ModelProfile::new(model, encoding);
    profile.expert_usage = args.expert_usage_reader.clone();
    let api = args.api.load()?;
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(
        cuteafd_core::concurrency::queue_depth(args.max_sequences as u64) as usize,
    );
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    type Vision = (
        Option<Arc<cuteafd_api::openai::media::MediaPreparer>>,
        Option<Arc<std::sync::atomic::AtomicBool>>,
    );
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(usize, Vision)>>();
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
            crate::cli::publish_core_view(&view);
            let checkpoint = Checkpoint::open_view(&view)?;
            let cfg = Hy4Config::from_hf(&checkpoint.config)?;
            cfg.release_geometry()?;
            cuteafd_bench::context::set_family("hy4");
            if let Err(prev) = cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::HY4) {
                ensure!(
                    prev == cuteafd_core::ExpertGeometry::HY4,
                    "Hy4 process geometry already fixed to {prev:?}"
                );
            }

            let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
            let model = cuteafd_loader::plan::families::hy4::Model::open(&checkpoint)?;
            let routed_layers = cfg.routed_layers();
            let onboard_req = worker_args.rtx_expert_layers.resolve(routed_layers.len());
            if let Onboard::Layers(n) = onboard_req {
                ensure!(
                    n <= routed_layers.len(),
                    "Hy4 RTX_EXPERT_LAYERS={n} exceeds {} routed layers",
                    routed_layers.len()
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
            if gpus == 2 {
                let probed = crate::shared::peer_split::probed_device(
                    &library,
                    device_ids[0],
                    Some(device_ids[1]),
                )?;
                ensure!(
                    probed == Some(device_ids[1]),
                    "Hy4 TP2 requires bidirectional GPU-to-GPU peer access between devices {} and {}",
                    device_ids[0],
                    device_ids[1]
                );
            }
            let devices: Vec<Device<'_>> = device_ids
                .iter()
                .map(|&id| Device { library: &library, id })
                .collect();
            let unit_rows = usize::try_from(prefill.unit_rows)?;
            let capacity_rows = unit_rows.max(decode_rows);
            let prefill_waves = usize::try_from(prefill.waves)?;

            let context_split = attention_placement == AttentionPlacement::Context;
            let mut preloaded_programs: Vec<Hy4Programs<'_>> = Vec::with_capacity(gpus);
            let pre = RuntimeInventory::measure(&library, &device_ids, InventoryPhase::PreWeight, |_| {
                preloaded_programs.push(Hy4Programs::load(
                    &programs,
                    gpus,
                    worker_args.kv_format,
                    context_split,
                )?);
                Ok(())
            })?;

            let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
            let samples: Vec<(u64, u64, u64)> = pre
                .gpus
                .iter()
                .map(|g| (g.total_bytes, policy.gpu_cap_bytes(g.total_bytes), g.free_bytes))
                .collect();
            let loaded_code: Vec<u64> = pre
                .gpus
                .iter()
                .enumerate()
                .map(|(rank, sample)| pending_code_allowance(sample, rank, gpus == 2))
                .collect();
            let peer_strings: Vec<String> = worker_args.peers.iter().map(ToString::to_string).collect();
            let spark_bytes = if peer_strings.is_empty() {
                Vec::new()
            } else {
                cuteafd_loader::serving_capacity::spark_memory::runtime_bytes(&peer_strings)?
            };
            let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
            let host_cache_bytes = match worker_args.prefix.host_cache_bytes {
                HostBudget::Auto => None,
                HostBudget::Bytes(bytes) => Some(bytes),
            };
            let shape = PrefillShape {
                rows: prefill.unit_rows,
                waves: prefill.waves,
            };
            let replies = cuteafd_core::spark_reply::requested();
            let resident_weight_limit = Some(crate::shared::memory::budget::resident_weight_limit());

            let facts_req = FactsRequest {
                checkpoint: &checkpoint,
                catalog: &catalog,
                cfg: &cfg,
                model_core: &model.core,
                kv_format: worker_args.kv_format,
                manifest_path: Some(&worker_args.manifest),
                coordinator: Coordinator::Measured { samples, loaded_code },
                attention: attention_placement,
                spark_ranks: worker_args.peers.len(),
                spark_bytes,
                prefill: shape,
                decode_rows: u32::try_from(decode_rows)?,
                context_tokens: worker_args.max_context.unwrap_or(0) as u64,
                pool_tokens: worker_args.pool_tokens,
                target_pool_tokens: None,
                sequences: worker_args.max_sequences as u64,
                host,
                host_cache_bytes,
                prefix_entries: worker_args.prefix.prefix_cache_entries as u64,
                replies,
                expert_usage,
                resident_weight_limit,
                vision: worker_args.media.vision,
                vision_replicas: worker_args.media.vision_replicas,
                media_cache_bytes: worker_args.media.media_cache_bytes.map(|bytes| bytes.0 as u64),
            };
            let inputs = admission::gather(&facts_req)
                .map_err(|refused| anyhow::anyhow!("{}: {}", refused.what, refused.how))?;
            let solve_req = hy4_place::request(
                inputs.context,
                worker_args.pool_tokens,
                onboard_req,
                shape,
                Operation::Serving,
                inputs.resident_weight_limit,
                replies,
            );
            let admitted = inputs.admit(solve_req).map_err(|error| match onboard_req {
                Onboard::Layers(n) => anyhow::anyhow!(
                    "Hy4 RTX_EXPERT_LAYERS={n} refused by shared admission: {error}"
                ),
                Onboard::Auto => anyhow::anyhow!("Hy4 shared admission refused: {error}"),
            })?;

            cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "hy4");
            let graph_layout = graph_reserve::Layout::resolve(
                gpus, worker_args.peers.len(), attention_placement,
            );
            let graph_policies: Vec<_> = (0..gpus)
                .map(|rank| graph_reserve::reserve_policy(graph_layout, rank))
                .collect::<std::result::Result<_, _>>()?;
            ensure!(
                graph_policies.iter().map(|policy| policy.bytes).eq(inputs.graph_reserve.iter().copied()),
                "Hy4 graph reserve policies disagreed with admission"
            );
            publish_provenance(&worker_args, &pre, &admitted, resolved_attention, &graph_policies)?;
            let sampled_free: Vec<u64> = pre.gpus.iter().map(|g| g.free_bytes).collect();
            pre.record("hy4", &sampled_free);
            cuteafd_bench::context::ResolvedSettings::placement(&admitted.placement).publish();

            let onboard_count = admitted.onboard;
            let onboarded_layers = routed_layers[..onboard_count].to_vec();
            let spark_layers = routed_layers[onboard_count..].to_vec();
            let pool_tokens = admitted.solved.pool_tokens;
            let chosen_pages = usize::try_from(pool_tokens / admitted.solved.pool_page_rows)?;
            let served_context = usize::try_from(admitted.solved.served_context())?;
            let admitted_context = usize::try_from(inputs.context.maximum)?;
            let host_quota = inputs.host_prefix.quota_bytes;

            // Start the bundled 40-layer ViT owner on its admitted device before the
            // target/Spark push group so its deferred arena lands in the same group.
            let vision_placement = inputs
                .vision
                .as_ref()
                .and_then(|vision| vision.admitted(&admitted.solved));
            let mut media = crate::shared::vision::still::Ready::load(
                &worker_args.media,
                &worker_args.snapshot,
                &worker_args.native_lib,
                &library,
                &device_ids,
                vision_placement,
                cuteafd_loader::media::ImageFamily::Hy4,
                inputs.vision.as_ref().map_or(0, |vision| vision.media_cache),
            )?;
            let media_push = std::mem::take(&mut media.push);
            library.cuda_set_device(worker_args.device)?;

            let rtx_plan = if onboarded_layers.is_empty() {
                None
            } else {
                let expert_pkg = resolve_expert_package(
                    &worker_args.native_lib,
                    worker_args.expert_package.as_deref(),
                    gpus,
                    &catalog,
                )?;
                let peak_budgets: Vec<usize> = if gpus == 1 {
                    vec![usize::try_from(admitted.placement.expert_ranges[0].peak_bytes)?]
                } else {
                    let tp2 = admitted
                        .placement
                        .tp2
                        .as_ref()
                        .context("Hy4 missing admitted TP2 expert range")?;
                    vec![usize::try_from(tp2.peak_bytes[0])?, usize::try_from(tp2.peak_bytes[1])?]
                };
                let workspace = if gpus == 1 {
                    Exl3Tp2::whole_workspace_bytes(
                        &expert_pkg,
                        capacity_rows,
                        cuteafd_core::ExpertGeometry::HY4,
                        RtxSelectionPolicy::CanonicalDecode,
                    )?
                } else {
                    let row_policy =
                        crate::shared::experts::exl3::execution::Exl3RowPolicy::for_geometry(
                            cuteafd_core::ExpertGeometry::HY4,
                        );
                    Exl3Tp2::workspace_bytes_for_selection(
                        &expert_pkg,
                        cfg.hidden,
                        capacity_rows,
                        row_policy,
                        RtxSelectionPolicy::CanonicalDecode,
                    )?
                };
                let payload_budgets: Vec<usize> = peak_budgets
                    .iter()
                    .map(|&peak| {
                        peak.checked_sub(workspace)
                            .context("Hy4 onboarded expert workspace exceeds admitted peak")
                    })
                    .collect::<Result<_>>()?;
                Some((expert_pkg, peak_budgets, payload_budgets))
            };

            let head_rows = capacity_rows.min(hy4_place::DECODE_ROWS as usize);
            let (spark_link, preloaded_weights, onboarded): (
                Option<crate::shared::spark_intake::SparkLink<'_>>,
                Option<Vec<super::weights::Hy4RankWeights<'_>>>,
                Vec<Option<Exl3Tp2<'_>>>,
            ) = if !worker_args.peers.is_empty() && !spark_layers.is_empty() {
                use crate::shared::{
                    experts::{
                        layer::ExpertLayer,
                        rtx::{
                            dense,
                            push::{CoordinatorExperts, CoordinatorPush, PushPolicy, WholeOwner},
                        },
                    },
                    placement::{spark_rings, SparkAdmission},
                    spark_intake::SparkLink,
                };
                use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
                use cuteafd_loader::placement::SparkWorkingSet;
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
                let backend_exec =
                    cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?;
                let executions = cuteafd_loader::plan::spark_executions::admit_family(
                    "hy4",
                    &backend_exec.exports(),
                    Some(decode_rows as u32),
                    prefill.unit_rows,
                    prefill.waves,
                )?;
                let model_adm = ModelAdmission::for_clients(
                    ExpertGeometry::HY4,
                    endpoint,
                    spark_rings(tcp.max_frame_bytes)?,
                    1,
                    executions,
                )?;
                let spark_adm = SparkAdmission::plan_view(
                    &view,
                    ExpertGeometry::HY4,
                    Some(&catalog),
                    &working,
                    worker_args.peers.len(),
                    model_adm,
                    None,
                )?;

                // Unified SparkNest push group: Spark workers + coordinator dense target
                // weights + bundled ViT arena + onboarded RTX EXL3 experts in one group.
                let policy = PushPolicy::for_snapshot(&checkpoint.snapshot)?;
                let media_records = media_push.records();
                let planned = if spark_adm.coordinator_pushes() {
                    policy.plan(|| -> Result<_> {
                        let (weights, mut records) = dense::recording(|| {
                            super::weights::Hy4RankWeights::record_all(
                                &devices,
                                &programs,
                                &checkpoint,
                                &cfg,
                                capacity_rows,
                                head_rows,
                            )
                        })?;
                        records.extend(media_records);
                        let (operands, steps) = (records.operands.len(), records.steps.len());
                        let dense = dense::plan_records(
                            &checkpoint.snapshot,
                            records,
                            &devices,
                            worker_args.peers.len(),
                        )?;
                        let experts = rtx_plan.as_ref().map(|(_, _, payload_budgets)| {
                            if gpus == 1 {
                                CoordinatorExperts::Whole {
                                    owners: vec![WholeOwner {
                                        rank: 0,
                                        selections: onboarded_layers
                                            .iter()
                                            .map(|&layer| ExpertLayer::BackboneFull { layer })
                                            .collect(),
                                        budget: payload_budgets[0],
                                    }],
                                }
                            } else {
                                CoordinatorExperts::Tp2 {
                                    layers: onboarded_layers.clone(),
                                    budgets: payload_budgets.clone(),
                                }
                            }
                        });
                        let prepared = CoordinatorPush::prepare_view(
                            &view,
                            Some(&catalog),
                            gpus,
                            experts,
                        )?;
                        Ok((prepared, weights, dense, operands, steps))
                    })?
                } else {
                    None
                };
                let (planned, ranks) = match planned {
                    Some(((push, streams, ranks), weights, dense, operands, steps)) => {
                        (Some((push, weights, dense, operands, steps)), Some((streams, ranks)))
                    }
                    None => (None, None),
                };
                let pending = spark_adm.start_with(&worker_args.peers, ranks)?;
                let (weights, pushed) = match planned {
                    Some((push, weights, dense, operands, steps)) => {
                        let started = std::time::Instant::now();
                        let (pushed, ()) = push.receive_with(
                            devices.to_vec(),
                            Some(&catalog),
                            worker_args.peers.len(),
                            Some(Ok(dense)),
                            || (),
                        );
                        let pushed = policy.committed(pushed)?;
                        tracing::info!(
                            pushed = pushed.is_some(),
                            operands,
                            steps,
                            onboard_layers = onboarded_layers.len(),
                            elapsed_ms = started.elapsed().as_millis() as u64,
                            "Hy4 unified SparkNest push group committed"
                        );
                        media_push.commit()?;
                        (weights, pushed)
                    }
                    None => {
                        let weights = super::weights::Hy4RankWeights::load_all(
                            &devices,
                            &programs,
                            &checkpoint,
                            &cfg,
                            capacity_rows,
                            head_rows,
                            media_push,
                        )?;
                        (weights, None)
                    }
                };
                pending.wait()?;
                if pushed.is_some() {
                    super::weights::Hy4RankWeights::finish_kv_b(
                        &weights,
                        &programs,
                        &checkpoint,
                        &cfg,
                    )?;
                }

                let pushed_exl3 = pushed
                    .map(|ranks| {
                        ranks
                            .into_iter()
                            .map(|r| r.exl3())
                            .collect::<Result<Vec<_>>>()
                    })
                    .transpose()?;
                let onboarded: Vec<Option<Exl3Tp2<'_>>> = match &rtx_plan {
                    None => (0..gpus).map(|_| None).collect(),
                    Some((expert_pkg, peak_budgets, _)) => {
                        if gpus == 1 {
                            let w = match pushed_exl3 {
                                Some(mut ranks) => {
                                    ensure!(ranks.len() == 1, "Hy4 pushed whole expert rank count");
                                    Exl3Tp2::pushed_whole_weights(
                                        devices[0],
                                        &catalog,
                                        expert_pkg,
                                        onboarded_layers.clone(),
                                        capacity_rows,
                                        peak_budgets[0],
                                        RtxSelectionPolicy::CanonicalDecode,
                                        ranks.pop().unwrap(),
                                    )?
                                }
                                None => Exl3Tp2::load_whole_weights(
                                    devices[0],
                                    &catalog,
                                    expert_pkg,
                                    onboarded_layers.clone(),
                                    capacity_rows,
                                    peak_budgets[0],
                                    RtxSelectionPolicy::CanonicalDecode,
                                )?,
                            };
                            vec![Some(w.execution()?)]
                        } else {
                            let [w0, w1] = match pushed_exl3 {
                                Some(ranks) => {
                                    let pair = ranks.try_into().map_err(|_| {
                                        anyhow::anyhow!("Hy4 pushed TP2 expert rank count")
                                    })?;
                                    Exl3Tp2::pushed_pair_weights(
                                        [devices[0], devices[1]],
                                        &catalog,
                                        ExpertGeometry::HY4,
                                        [expert_pkg.as_path(), expert_pkg.as_path()],
                                        onboarded_layers.clone(),
                                        capacity_rows,
                                        [peak_budgets[0], peak_budgets[1]],
                                        RtxSelectionPolicy::CanonicalDecode,
                                        pair,
                                    )?
                                }
                                None => Exl3Tp2::load_pair_weights(
                                    [devices[0], devices[1]],
                                    &catalog,
                                    ExpertGeometry::HY4,
                                    [expert_pkg.as_path(), expert_pkg.as_path()],
                                    onboarded_layers.clone(),
                                    capacity_rows,
                                    [peak_budgets[0], peak_budgets[1]],
                                    RtxSelectionPolicy::CanonicalDecode,
                                )?,
                            };
                            vec![Some(w0.execution()?), Some(w1.execution()?)]
                        }
                    }
                };

                let geometry = inputs
                    .links
                    .first()
                    .context("Hy4 admitted Spark link geometry absent")?
                    .clone();
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
                    u32::try_from(cfg.hidden)?,
                    u32::try_from(cfg.topk)?,
                    ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                )?;
                session.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
                devices[0].run(|| link.bootstrap(&session))?;
                (Some(link), Some(weights), onboarded)
            } else {
                let weights = super::weights::Hy4RankWeights::load_all(
                    &devices,
                    &programs,
                    &checkpoint,
                    &cfg,
                    capacity_rows,
                    head_rows,
                    media_push,
                )?;
                let onboarded = match &rtx_plan {
                    None => (0..gpus).map(|_| None).collect(),
                    Some((expert_pkg, peak_budgets, _)) => {
                        if gpus == 1 {
                            let w = Exl3Tp2::load_whole_weights(
                                devices[0],
                                &catalog,
                                expert_pkg,
                                onboarded_layers.clone(),
                                capacity_rows,
                                peak_budgets[0],
                                RtxSelectionPolicy::CanonicalDecode,
                            )?;
                            vec![Some(w.execution()?)]
                        } else {
                            let [w0, w1] = Exl3Tp2::load_pair_weights(
                                [devices[0], devices[1]],
                                &catalog,
                                cuteafd_core::ExpertGeometry::HY4,
                                [expert_pkg.as_path(), expert_pkg.as_path()],
                                onboarded_layers.clone(),
                                capacity_rows,
                                [peak_budgets[0], peak_budgets[1]],
                                RtxSelectionPolicy::CanonicalDecode,
                            )?;
                            vec![Some(w0.execution()?), Some(w1.execution()?)]
                        }
                    }
                };
                (None, Some(weights), onboarded)
            };

            library.cuda_set_device(worker_args.device)?;

            let mut engine = Hy4Engine::new_serving(
                &devices,
                &programs,
                Some(preloaded_programs),
                preloaded_weights,
                Some(&worker_args.manifest),
                &checkpoint,
                cfg.clone(),
                worker_args.kv_format,
                attention_placement,
                unit_rows,
                decode_rows,
                prefill_waves,
                chosen_pages,
                worker_args.max_sequences,
                served_context,
                admitted_context,
                false,
                onboarded,
                Default::default(),
            )?;
            if let Some(link) = spark_link {
                engine.attach_spark(link, onboarded_layers.clone());
            }
            engine.configure_expert_usage(
                inputs.expert_usage.as_ref(),
                worker_args.expert_usage_reader.clone(),
            )?;
            engine.configure_graphs(&graph_policies)?;

            let coord_devices = CoordinatorDevices::new(&device_ids)?;
            let mut weights_inv = Inventory::new(
                coord_devices,
                "hy4_weights",
                Lifetime::RESIDENT,
                Category::Weights,
                "Hy4 target weights",
            );
            let mut retained_inv = Inventory::new(
                coord_devices,
                "retained",
                Lifetime::TARGET,
                Category::Workspace,
                "Hy4 retained owners",
            );
            engine.inventory(&mut weights_inv, &mut retained_inv)?;
            reconcile_runtime_inventory(
                &admitted.solved,
                &inputs,
                &weights_inv.receipts,
                &retained_inv.receipts,
            )?;
            crate::shared::memory_report::release_load_staging(&library)?;

            let family = engine.prefix()?;
            let host = (host_quota > 0 && worker_args.prefix.prefix_cache_entries > 0)
                .then(|| {
                    worker_args
                        .prefix
                        .host_config_for(family.layout(), served_context, host_quota)
                })
                .transpose()?;

            cuteafd_bench::context::ResolvedSettings::default()
                .value("expert-usage", expert_usage)
                .value("max-context-tokens", served_context)
                .value("pool-tokens", pool_tokens)
                .value("rtx-expert-layers", onboarded_layers.len())
                .value("prefill-rows", unit_rows)
                .value("prefill-waves", prefill.waves)
                .publish();

            let backend = Hy4Backend::new(
                engine,
                served_context,
                unit_rows,
                host,
                &worker_args.snapshot,
                media.preparer.is_some(),
            )?;
            let mut image = media.preparer.clone();
            let mut health = media.encoder.health_handle();
            backend.enable_expert_usage();
            let mut media_admission = cuteafd_engine::media::MediaAdmission::new(
                cuteafd_engine::media::EmbeddingCache::new(media.cache_bytes),
                media.encoder,
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
                    crate::shared::memory_report::reconcile::capture("ready");
                    RuntimeInventory::measure(&library, &device_ids, InventoryPhase::PostLoad, |_| {
                        Ok(())
                    })?
                    .record("hy4-serving-loaded", &[]);
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok((served_context, (image.take(), health.take()))));
                    }
                    Ok(())
                },
            )
        })();
        crate::shared::serve::forward_initialization(
            result,
            ready.is_some(),
            "Hy4 initialization failed",
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

    let (context, (image, health)) = ready_rx
        .await
        .context("Hy4 worker exited before readiness")??;
    // Vision is advertised only once the tower owner reported ready.
    let profile = match image {
        Some(image) => {
            let mut profile = profile.with_grouped_vision(image);
            profile.vision_health = health;
            profile
        }
        None => profile,
    };
    let limits = args.prefix.reserve.limits(context)?;
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(args.console.console_text, || {
        Ok(crate::shared::console::Layout::new(
            "hy4",
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
        served = crate::shared::api::serve_http(listener, api.app(router, hub), "hy4") => served?,
        _ = crate::shared::api::watch_scheduler(&mut worker) => unreachable!(),
    }
    crate::shared::api::finish_scheduler(&mut worker, "hy4").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn serve_hy4_cli_and_resolved_seam_match() -> anyhow::Result<()> {
        let argv = [
            "cuteafd",
            "--gpu-memory-cap",
            "97250MiB",
            "serve-hy4",
            "--snapshot",
            "/model",
            "--native-lib",
            "/opt/cuteafd/lib/libcuteafd_native.so",
            "--device",
            "0",
            "--split-device",
            "1",
            "--peers",
            "10.55.0.1:19441,10.55.0.2:19441,10.55.0.3:19441,10.55.0.4:19441",
            "--listen",
            "0.0.0.0:8824",
            "--max-sequences",
            "16",
            "--max-context-tokens",
            "auto",
            "--core-view",
            "k8-compact",
            "--prefill-rows",
            "2048",
            "--prefill-waves",
            "1",
            "--decode-rows",
            "32",
        ];
        let crate::cli::Commands::ServeHy4(mut args) = crate::cli::Cli::try_parse_from(argv)?.command
        else {
            anyhow::bail!("Hy4 command not parsed");
        };
        assert_eq!(
            (args.max_context, args.max_sequences, args.split_device, args.peers.len()),
            (Some(0), 16, Some(1), 4)
        );
        crate::commands::bench::assert_resolved_family_seam("serve-hy4", &[], || {
            args.prefix.resolve_entries(args.max_sequences);
            let prefill = args.prefill()?;
            let decode_rows = args.resolved_decode_rows(2)?;
            assert_eq!(decode_rows, 32);
            cuteafd_bench::context::ResolvedSettings::default()
                .value("decode-rows", decode_rows)
                .value("max-context-tokens", 1_048_576)
                .value("pool-tokens", 262_144)
                .value("rtx-expert-layers", 0)
                .value("attention-placement", "context")
                .value("prefill-rows", prefill.unit_rows)
                .value("prefill-waves", prefill.waves)
                .publish();
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn attention_placement_defaults_to_context_on_2gpu_and_allows_explicit_heads() -> anyhow::Result<()> {
        use cuteafd_loader::placement::AttentionPlacement;
        assert_eq!(
            super::resolve_attention_placement(1, "auto")?,
            ("whole", AttentionPlacement::Heads)
        );
        assert_eq!(
            super::resolve_attention_placement(1, "heads")?,
            ("whole", AttentionPlacement::Heads)
        );
        assert!(super::resolve_attention_placement(1, "context").is_err());
        assert_eq!(
            super::resolve_attention_placement(2, "auto")?,
            ("context", AttentionPlacement::Context)
        );
        assert_eq!(
            super::resolve_attention_placement(2, "context")?,
            ("context", AttentionPlacement::Context)
        );
        assert_eq!(
            super::resolve_attention_placement(2, "heads")?,
            ("heads", AttentionPlacement::Heads)
        );
        assert!(super::resolve_attention_placement(2, "layers").is_err());
        Ok(())
    }

    #[test]
    fn shared_scheduler_owns_lanes_and_typed_readiness() {
        let source = include_str!("serve.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("scheduler::schedule("));
        assert!(source.contains("Checkpoint::open_view(&view)?"));
        assert!(source.contains("context::set_family(\"hy4\")"));
        assert!(source.contains("shared::serve::forward_initialization("));
        assert!(source.contains("finish_scheduler(&mut worker"));
        assert!(source.contains("admission::gather(&facts_req)"));
        assert!(source.contains("inputs.admit(solve_req)"));
        assert!(source.contains("engine.inventory(&mut weights_inv, &mut retained_inv)"));
        assert!(source.contains("fault::set_inputs("));
        assert!(source.contains("reserve::log_plan("));
        assert!(
            source.find("Hy4Encoding::from_snapshot").unwrap()
                < source.find("NativeLibrary::load").unwrap()
        );
    }

    #[test]
    fn reconcile_runtime_inventory_accepts_exact_and_rejects_drift() -> anyhow::Result<()> {
        use cuteafd_loader::families::hy4::execution::IndexTopkScratch;
        use cuteafd_loader::families::hy4::weight_layout::CoordinatorWeights;
        use cuteafd_loader::families::hy4::{Hy4Config, Hy4IndexerKind, Hy4KvCache, IndexCacheSchedule};
        use cuteafd_loader::placement::families::hy4::{
            self as hy4_place, Hy4Inputs, LocalPackage, SparkCharge,
        };
        use cuteafd_loader::placement::planner::{
            Allocation, Baseline, Budget, ComponentKind, Context, Device, Domain, Location,
            Operation, PrefillShape, Receipt, Recipe,
        };
        use cuteafd_loader::placement::{AttentionPlacement, Onboard, PoolFacts, PoolMemory};
        use cuteafd_loader::serving_capacity::WeightArenaLayout;
        use std::collections::BTreeMap;

        let kinds: Vec<Hy4IndexerKind> = (0..78)
            .map(|l| {
                if l == 0 || (l >= 1 && (l - 1) % 4 == 0) {
                    Hy4IndexerKind::Full
                } else {
                    Hy4IndexerKind::Shared
                }
            })
            .collect();
        let cfg = Hy4Config {
            vocab_size: 120_832,
            hidden: 6144,
            layers: 78,
            intermediate: 18_432,
            max_position: 262_144,
            rms_norm_eps: 1e-5,
            qa_kva_eps: 1e-6,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            q_lora_rank: 2048,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            heads: 64,
            rope_theta: 10_000_000.0,
            indexer_heads: 32,
            indexer_head_dim: 128,
            indexer_topk: 2048,
            index_cache: IndexCacheSchedule::from_kinds(kinds)?,
            dense_layer: (0..78).map(|i| i == 0).collect(),
            experts: 256,
            topk: 8,
            moe_intermediate: 2048,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.827,
            swiglu_limit: 10.0,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        };
        let prefill = PrefillShape { rows: 2048, waves: 3 };
        let replies = cuteafd_core::spark_reply::SparkReplyFormats::default();
        let context = Context::resolve(262_144, None);
        let modes = hy4_place::layer_modes(&cfg, 1, AttentionPlacement::Heads);
        let routed = cfg.routed_layers();
        let mut facts = PoolFacts::new(context, None, 64);
        facts.memory = vec![PoolMemory { available_bytes: 96u64 << 30, bytes_per_unit: 48 * 64 * 1024 }];
        let pool = cuteafd_loader::placement::prepare_pool(facts)?;
        let target = CoordinatorWeights {
            rank: 0,
            layers: (0..78).collect(),
            arena: WeightArenaLayout::new(
                "hy4/target/gpu0",
                vec![("lm_head.weight".to_string(), 120_832 * 6144 * 2)],
            )?,
            core: BTreeMap::new(),
            fp8: BTreeMap::new(),
        };
        let inputs = Hy4Inputs {
            vision: None,
            cfg,
            kv_format: Hy4KvCache::Fp8,
            gpus: 1,
            spark_ranks: 4,
            context,
            pool,
            prefill,
            decode_rows: hy4_place::DECODE_ROWS,
            sequences: 16,
            targets: vec![target],
            target_mount_load: Default::default(),
            index_topk: IndexTopkScratch::from_manifest_value(None, 1, false, 262_144)?,
            placement: AttentionPlacement::Heads,
            modes,
            routed: routed.clone(),
            expert_usage: None,
            local: LocalPackage::Missing("synthetic".into()),
            spark: vec![vec![SparkCharge { experts: 1 << 30, owners: 1 << 28, runtime: 1 << 28 }; 4]; routed.len() + 1],
            links: hy4_place::spark_link(0, 4, prefill, replies)?,
            budgets: vec![
                hy4_place::planned_budget(0, 96u64 << 30, 96u64 << 30, 1u64 << 30),
                Budget {
                    id: "spark0".into(),
                    locations: vec![Location { device: Device::Spark(0), domain: Domain::Managed }],
                    physical_bytes: 120u64 << 30,
                    logical_bytes: 120u64 << 30,
                    baseline: Baseline::Forecast { occupied: 0, source: "spark".into(), assumed: false },
                    headroom: 4u64 << 30,
                },
                Budget {
                    id: "spark1".into(),
                    locations: vec![Location { device: Device::Spark(1), domain: Domain::Managed }],
                    physical_bytes: 120u64 << 30,
                    logical_bytes: 120u64 << 30,
                    baseline: Baseline::Forecast { occupied: 0, source: "spark".into(), assumed: false },
                    headroom: 4u64 << 30,
                },
                Budget {
                    id: "spark2".into(),
                    locations: vec![Location { device: Device::Spark(2), domain: Domain::Managed }],
                    physical_bytes: 120u64 << 30,
                    logical_bytes: 120u64 << 30,
                    baseline: Baseline::Forecast { occupied: 0, source: "spark".into(), assumed: false },
                    headroom: 4u64 << 30,
                },
                Budget {
                    id: "spark3".into(),
                    locations: vec![Location { device: Device::Spark(3), domain: Domain::Managed }],
                    physical_bytes: 120u64 << 30,
                    logical_bytes: 120u64 << 30,
                    baseline: Baseline::Forecast { occupied: 0, source: "spark".into(), assumed: false },
                    headroom: 4u64 << 30,
                },
                Budget {
                    id: "host".into(),
                    locations: vec![Location::HOST, Location::PINNED],
                    physical_bytes: 512u64 << 30,
                    logical_bytes: 512u64 << 30,
                    baseline: Baseline::Forecast { occupied: 0, source: "host".into(), assumed: false },
                    headroom: 16u64 << 30,
                },
            ],
            code: vec![256u64 << 20],
            graph_reserve: vec![0],
            host_prefix: cuteafd_loader::placement::planner::serving::HostPrefixOwners {
                quota_bytes: 0,
                chunk_bytes: 1,
            },
            resident_weight_limit: None,
        };
        let req = hy4_place::request(
            inputs.context,
            None,
            Onboard::Layers(0),
            prefill,
            Operation::Serving,
            None,
            replies,
        );
        let admitted = inputs.admit(req)?;
        let contract = inputs.prepare(&admitted.solved.request)?;
        let pages = admitted.solved.pool_tokens.div_ceil(admitted.solved.pool_page_rows);

        let mut weights_receipts = Vec::new();
        let mut retained_receipts = Vec::new();
        let mut addr = 0x1000_0000u64;
        for alloc in admitted.solved.facts.allocations.iter().filter(|a| a.view.is_none()) {
            if alloc.id.starts_with("hy4/target/gpu") {
                weights_receipts.push(Receipt { allocation: alloc.clone(), address: addr });
                addr += 0x1000;
            } else if alloc.id.starts_with("hy4.cache.")
                || alloc.id.starts_with("hy4.scratch.")
                || alloc.id.starts_with("hy4.wave.")
                || alloc.id.starts_with("hy4.head.")
                || alloc.id.starts_with("hy4.target.")
            {
                let curve = contract
                    .components
                    .iter()
                    .find_map(|c| match &c.kind {
                        ComponentKind::Allocation { template, .. } if template.id == alloc.id => Some(&c.bytes),
                        _ => None,
                    })
                    .expect("contract curve");
                let piece = curve.0.iter().find(|p| (p.first..=p.last).contains(&pages)).unwrap();
                let payload = piece.bytes_at_first + piece.bytes_per_page * (pages - piece.first);
                let mut rec = alloc.clone();
                rec.bytes = payload;
                if rec.id.starts_with("hy4.cache.") || rec.id.starts_with("hy4.head.") {
                    rec.id = format!("retained.{}", rec.id);
                }
                retained_receipts.push(Receipt { allocation: rec, address: addr });
                addr += 0x1000;
            }
        }
        super::reconcile_runtime_inventory(&admitted.solved, &inputs, &weights_receipts, &retained_receipts)?;

        let mut missing = retained_receipts.clone();
        missing.pop();
        assert!(super::reconcile_runtime_inventory(&admitted.solved, &inputs, &weights_receipts, &missing).is_err());

        let mut wrong_size = retained_receipts.clone();
        wrong_size[0].allocation.bytes += 4096;
        assert!(super::reconcile_runtime_inventory(&admitted.solved, &inputs, &weights_receipts, &wrong_size).is_err());

        let mut extra = retained_receipts;
        extra.push(Receipt {
            allocation: Allocation::buffer(
                "retained.hy4.rogue",
                Location::coordinator(0),
                4096,
                cuteafd_loader::placement::planner::Lifetime::TARGET,
                cuteafd_core::memory_layout::Category::Workspace,
                "rogue",
            ),
            address: addr,
        });
        assert!(super::reconcile_runtime_inventory(&admitted.solved, &inputs, &weights_receipts, &extra).is_err());
        Ok(())
    }
}
