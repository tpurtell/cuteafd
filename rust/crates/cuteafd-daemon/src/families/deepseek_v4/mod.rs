//! DeepSeek V4 (Flash / Pro) coordinator engine over the exported b12x programs.
pub(crate) mod engine;
pub(crate) mod admission;
mod draft_binding;
#[cfg(test)]
mod placement_tests;
pub(crate) mod local;
pub(crate) mod metadata;
pub(crate) mod pool;
pub(crate) mod prefix;
pub(crate) mod serve;
pub(crate) mod weights;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::deepseek_v4::DeepseekV4Config;
use crate::shared::spark_intake::SparkLink;
use cuteafd_transport::TcpTransportConfig;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// What every DeepSeek V4 command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    /// Checkpoint snapshot directory.
    #[arg(long)]
    pub snapshot: PathBuf,
    /// Spark expert ranks in TP order, comma-separated HOST:PORT.
    #[arg(long, default_value = "")]
    pub peers: String,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    /// dsv4_programs.json written by the exporter next to the library.
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    /// Longest sequence; 0 selects checkpoint full, bounded by compiled support.
    #[arg(skip)]
    pub max_context: usize,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Split every backbone layer's attention heads (w_q rows, sinks, wo
    /// groups) and shared expert over --device and this second GPU; mHC, the
    /// latent projection, compressors, indexer and caches are replicated, the
    /// partial sums meet over peer memory. Routers are replicated; head and
    /// drafter stay on --device. Routed-expert TP2 halves fill both GPUs after
    /// reserving their KV pools.
    #[arg(long)]
    pub split_device: Option<i32>,
    /// Optional expert-input quantizer SM ceiling (default: this device's SM count).
    #[arg(long)]
    pub sms: Option<usize>,
    /// Rows per prefill unit; defaults to the export's maximum capacity.
    #[arg(long)]
    pub prefill_rows: Option<usize>,
    /// Independently outstanding units on one Spark transport.
    #[arg(long, default_value_t = engine::PREFILL_WAVES)]
    pub prefill_waves: usize,
    /// Sequences that can be resident at once (compressor state slots).
    #[arg(long, default_value_t = 8)]
    pub max_sequences: usize,
    /// Admit every prefill row's logits at startup for fidelity probes.
    #[arg(long)]
    pub full_prefill_logits: bool,
    /// Total tokens the compressed-cache pools hold across sequences; 0 uses
    /// planner admission from measured free memory before cache allocation.
    #[arg(long, default_value_t = 0)]
    pub pool_tokens: usize,
    /// Routed-expert layers to keep on the coordinator GPUs (from layer 0):
    /// `max` places the most layers that still leave a 262K pool;
    /// `auto` (the default on every layout) reserves the target KV pool first and
    /// fills what is left; `N`, `N%` or `all` fix the RTX layers and the KV pool
    /// takes every remaining byte. 0 sends every layer to the Sparks.
    #[arg(long, value_parser = parse_onboard, conflicts_with = "local_expert_layers")]
    pub rtx_expert_layers: Option<cuteafd_loader::placement::Onboard>,
    /// Alias of `--rtx-expert-layers N`.
    #[arg(long)]
    pub local_expert_layers: Option<usize>,
    /// Removed GPU1 whole-layer placement flag.
    #[arg(long = "peer-expert-ranges", hide = true, value_parser = crate::cli::reject_peer_expert_ranges)]
    pub deprecated_peer_expert_ranges: bool,
    /// Keep the dSpark drafter's stage experts on the coordinator GPU (before
    /// backbone layers) so the engine can draft.
    #[arg(long)]
    pub dspark: bool,
    /// GiB kept free on the coordinator GPU for workspaces and headroom.
    #[arg(long, default_value_t = 10)]
    pub reserve_gib: usize,
    /// Benchmarks and token gates only: MoE layers run the shared expert
    /// alone (no Sparks, no local experts; outputs do not match the model).
    #[arg(long, hide = true)]
    pub skip_routed_experts: bool,
    #[command(flatten)]
    pub token_io: crate::shared::token_io::TokenIoArgs,
    /// Attention over the two GPUs: auto (the recorded default, heads), context (the C4
    /// layers' compressed records split by 256-token unit parity with a q/candidate/partial
    /// exchange per C4 layer; C128 and window state stay replicated), or heads.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLACEMENT", default_value = "auto")]
    pub attention_placement: String,
    /// Rows the context split's partial reduction plan is tuned for, fixed at startup: auto
    /// (dSpark's verify width, draft block + 1; one row without a drafter) or 1..=decode rows.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLAN_ROWS", default_value = "auto", value_parser = parse_plan_rows)]
    pub attention_plan_rows: PlanRows,
}

/// `--attention-plan-rows`: `auto` or an explicit row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PlanRows(pub Option<usize>);

fn parse_plan_rows(text: &str) -> std::result::Result<PlanRows, String> {
    match text {
        "auto" => Ok(PlanRows(None)),
        n => n.parse::<usize>().ok().filter(|&n| n > 0).map(|n| PlanRows(Some(n)))
            .ok_or_else(|| format!("attention plan rows must be auto or a positive integer (got {n:?})")),
    }
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Published shared fidelity directory, or tokens.bin/layerNN.bin/logits.bin from golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// Score every prefill row: mean next-token NLL, top-1 agreement and KL against golden.
    #[arg(long)]
    pub nll: bool,
    /// One window from the published sealed quick subset (default: all eight).
    #[arg(long)]
    pub window: Option<String>,
    /// Save tokens.bin and F32 logits.bin for a reference run: with --nll every prefill row,
    /// with --prefill N only the decode/verify rows (positions N.., tokens.bin from N), so two
    /// runs compare exactly the rows the decode path produced.
    #[arg(long)]
    pub save_logits: Option<PathBuf>,
    /// Compare only the first N layers' streams (all logits still compared).
    #[arg(long)]
    pub layers: Option<usize>,
    /// Prefill only the first N tokens and decode the rest one at a time
    /// (teacher-forced), comparing every decode row with the golden logits.
    #[arg(long)]
    pub prefill: Option<usize>,
    /// Before each decode step, draft with dSpark after the true next token
    /// and report how many drafts match the golden continuation.
    #[arg(long)]
    pub draft: bool,
    /// Teacher-force decode in verify steps of this many rows per step.
    #[arg(long)]
    pub verify_rows: Option<usize>,
    /// Prefill in chunks of this many tokens (continuation compressor).
    #[arg(long)]
    pub chunk: Option<usize>,
    /// Token I/O gate after the golden prompt (--prefill N truncates it),
    /// then stop: the resident embedding table against the shard, device
    /// against host greedy selection over this many decode steps, and device
    /// against host sampling.
    #[arg(long)]
    pub token_check: Option<usize>,
    /// Prefix-cache restore check at each token P (comma separated): prefill the first P tokens
    /// (of --prefill, default P + --resume-span), capture their snapshot (shared units, the
    /// copied tail unit, the window/compressor mark), restore it into a second sequence,
    /// continue both (prefill in --prefill-chunk rows, then --resume-decode greedy steps) and
    /// compare every layer's rows, the logits, the paged rows, the final positional state and
    /// the mark round trip byte for byte (a restore must be exact). Every P runs with every
    /// --prefill-chunk. Needs --max-sequences 3 or more.
    #[arg(long, value_delimiter = ',')]
    pub resume_at: Vec<usize>,
    /// Prefill chunk rows of --resume-at, comma separated (default the engine's prefill rows).
    #[arg(long, value_delimiter = ',')]
    pub prefill_chunk: Vec<usize>,
    /// Tokens prefilled past P by --resume-at without --prefill.
    #[arg(long, default_value_t = 1000)]
    pub resume_span: usize,
    #[arg(long, default_value_t = 4)]
    pub resume_decode: usize,
    /// With --resume-at: prefill the second sequence cold on its own units instead of
    /// restoring it (the floor: what page placement alone changes).
    #[arg(long, hide = true)]
    pub resume_cold: bool,
    /// --resume-at attempts on fresh sequences per case (all must be byte-identical).
    #[arg(long, default_value_t = 1)]
    pub resume_repeat: usize,
    /// --resume-at prefills as serving does (chunks of 512 rows or more split into two lanes,
    /// up to twice the prefill rows per chunk); layer streams are then not compared.
    #[arg(long)]
    pub resume_lanes: bool,
}

fn parse_onboard(text: &str) -> std::result::Result<cuteafd_loader::placement::Onboard, String> {
    text.parse()
}

impl EngineArgs {
    /// The resolved `--rtx-expert-layers` / `--local-expert-layers`.
    /// Pool first uniformly unless an explicit onboard policy is selected.
    pub(crate) fn onboard(&self) -> Result<cuteafd_loader::placement::Onboard> {
        use cuteafd_loader::placement::Onboard;
        Ok(match (self.rtx_expert_layers, self.local_expert_layers) {
            (Some(onboard), _) => onboard,
            (None, Some(layers)) => Onboard::Layers(layers),
            (None, None) => cuteafd_loader::placement::families::deepseek_v4::default_onboard(),
        })
    }

    pub(crate) fn fixed_onboard(&self) -> bool {
        self.rtx_expert_layers.is_some() || self.local_expert_layers.is_some()
    }
}

/// The engine runs three shapes: every layer head-split under `--split-device` (`heads`), the
/// same with every C4 layer token-split (`context`: all or none of them, C128 and window layers
/// head-split), or every layer whole on GPU0, and no hop but the step-start streams push.
/// Anything else the solver hands it is refused before any cache or expert is allocated.
/// Returns whether the C4 layers run `context`.
fn check_modes(placement: &cuteafd_loader::placement::Placement, split: bool, cfg: &DeepseekV4Config)
    -> Result<bool> {
    use cuteafd_loader::placement::{families::DEEPSEEK_V4, FfnMode, LayerMode};
    DEEPSEEK_V4.check(placement).map_err(|error| anyhow::anyhow!("DeepSeek V4 placement: {error}"))?;
    let context = split && placement.layers.iter().any(|a| a.mode == LayerMode::ContextSplit);
    ensure!(placement.layers.len() == cfg.n_layers, "DeepSeek V4 placement covers {} of {} layers",
        placement.layers.len(), cfg.n_layers);
    for (layer, (assignment, &ratio)) in placement.layers.iter().zip(&cfg.compress_ratios).enumerate() {
        let engine = match (split, context && ratio == 4) {
            (true, true) => LayerMode::ContextSplit,
            (true, false) => LayerMode::HeadSplit,
            (false, _) => LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
        };
        ensure!(assignment.mode == engine, "DeepSeek V4 runs layer {layer} (C{ratio}) as {engine:?}; the placement has {:?}",
            assignment.mode);
    }
    Ok(context)
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn similarity(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE), diff.sqrt() / nb.sqrt().max(f64::MIN_POSITIVE))
}

/// The checkpoint's `embed.weight` (BF16 [vocab, dim]).
fn embed_source(catalog: &cuteafd_loader::OfficialV41Catalog, dim: usize) -> Result<crate::shared::token_io::EmbedSource> {
    let tensor = catalog.tensor("embed.weight")?;
    crate::shared::token_io::EmbedSource::new(catalog.snapshot(), &tensor.shard, &tensor.metadata, dim)
}

pub(crate) async fn run_golden(mut args: GoldenArgs) -> Result<()> {
    args.engine.full_prefill_logits |= args.nll;
    tokio::task::spawn_blocking(move || golden(args)).await?
}

/// Everything the engine borrows, built on the calling (blocking) thread.
pub(crate) struct Loaded {
    pub snapshot: PathBuf,
    pub catalog: cuteafd_loader::OfficialV41Catalog,
    pub library: std::sync::Arc<NativeLibrary>,
    pub cfg: DeepseekV4Config,
    pub family: &'static str,
    pub manifest: serde_json::Value,
}

pub(crate) fn load(args: &EngineArgs) -> Result<Loaded> {
    let mut catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::selected("deepseek_v4",
        if args.dspark { "dspark" } else { "off" }))?;
    let geometry = catalog.routed_experts().geometry()?;
    cuteafd_core::set_expert_geometry(geometry).map_err(|g| anyhow::anyhow!("geometry already {g:?}"))?;
    let family = match geometry.family() {
        Some("dsv4f") => "dsv4f",
        Some("dsv4p") => "dsv4p",
        other => anyhow::bail!("not a DeepSeek V4 checkpoint (expert family {other:?})"),
    };
    let cfg = DeepseekV4Config::read(&args.snapshot, 1)?;
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&args.native_lib) }?);
    library.cuda_set_device(args.device)?;
    let mut manifest = serde_json::from_str(&std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("reading {}", args.manifest.display()))?)?;
    cuteafd_core::coordinator_programs::select_card_plans(&mut manifest,
        library.sm_count()?.try_into().context("runtime SM count exceeds u32")?)?;
    Ok(Loaded { snapshot: args.snapshot.clone(), catalog, library, cfg, family, manifest })
}

/// Builds the engine over `loaded` and hands it, with a Spark transport and a
/// runtime for it, to `body`. `held` is the device memory `body` allocates
/// besides the engine's workspaces (the prefix cache's mark arena): it is kept
/// free of local experts.
pub(crate) fn with_engine<T>(
    loaded: &Loaded,
    args: &EngineArgs,
    prefix: Option<&crate::shared::prefix::PrefixArgs>,
    held: impl FnOnce(&engine::Engine<'_>) -> Result<usize>,
    body: impl FnOnce(&engine::Engine<'_>, &mut [SparkLink<'_>], &tokio::runtime::Runtime,
        Option<cuteafd_hostcache::config::Config>) -> Result<T>,
) -> Result<T> {
    let programs = loaded.library.programs()?.with_manifest(&args.manifest)?;
    let caps = &loaded.manifest["capacities"];
    let stream = loaded.library.cuda_stream_create()?;
    // The head split's second GPU and its stream (load kernels, then the engine's).
    // A head split needs its share's programs (`dsv4f2` / `dsv4p2`) in this build.
    let share_program = format!("{}2_wo_m{}", loaded.family, caps["decode_rows"].as_u64().unwrap_or(64));
    let split_device = match args.split_device {
        Some(device) if programs.spec(&share_program).is_ok() => Some(device),
        Some(device) => {
            tracing::info!(device, "no head-split programs ({share_program}) in this build; serving from --device alone");
            None
        }
        None => None,
    };
    let split_device = crate::shared::peer_split::probed_device(&loaded.library, args.device, split_device)?;
    let split_family = format!("{}2", loaded.family);
    let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms {
        family: loaded.family, split_family: split_device.map(|_| split_family.as_str()),
    };
    selected.validate_v4(caps["prefill_rows"].as_u64().context("prefill_rows")?,
        caps["decode_rows"].as_u64().context("decode_rows")?, programs.names())?;
    let started = Instant::now();
    if let Some(device) = split_device {
        ensure!(device != args.device, "--split-device must differ from --device");
        loaded.library.cuda_enable_peer(device)?;
        crate::shared::peer_split::on_device(&loaded.library, device, args.device,
            || loaded.library.cuda_enable_peer(args.device))?;
    }
    // One inventory sample per GPU around the family's own program modules
    // (placement PR 2): the context and module bytes the planner charges as
    // `ArchContext` + `ProgramSet`.
    let devices: Vec<i32> = std::iter::once(args.device).chain(split_device).collect();
    let inventory = crate::shared::inventory::RuntimeInventory::measure(&loaded.library, &devices, crate::shared::inventory::InventoryPhase::PreWeight, |_| {
        let (loaded_count, skipped) = programs.load_matching(|name| selected.contains(name))?;
        tracing::info!(loaded = loaded_count, skipped, elapsed_ms = started.elapsed().as_millis() as u64,
            "DeepSeek V4 programs loaded");
        Ok(())
    })?;
    let peer_stream = match split_device {
        Some(device) => Some((device, crate::shared::peer_split::on_device(&loaded.library, device, args.device,
            || loaded.library.cuda_stream_create())?)),
        None => None,
    };
    let mut max_context = crate::shared::context::checkpoint_context(
        &args.snapshot, &args.manifest, "deepseek_v4", args.max_context)?;
    let prefill_kernel_rows = caps["prefill_rows"].as_u64().context("prefill_rows")? as usize;
    let prefill_rows = args.prefill_rows.unwrap_or(prefill_kernel_rows.min(4096));
    ensure!(matches!(prefill_rows, 512 | 1024 | 2048 | 4096)
        && prefill_rows <= caps["prefill_rows"].as_u64().context("prefill_rows")? as usize
        && (1..=3).contains(&args.prefill_waves), "V4 prefill needs page-aligned rows within the export capacity and 1..=3 waves");
    let decode_rows = caps["decode_rows"].as_u64().context("decode_rows")? as usize;
    let skip = args.skip_routed_experts;
    let peers = args.peers.split(',').filter(|p| !p.is_empty()).map(str::parse)
        .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
    if skip {
        tracing::warn!("--skip-routed-experts: MoE layers run the shared expert alone (outputs do not match the model)");
    } else {
        ensure!((1..=8).contains(&peers.len()), "DeepSeek V4 Spark transport requires 1..=8 peers, got {}", peers.len());
        loaded.library.v41_compact_reducer()?.require_rank_count(peers.len() as u32)?;
    }
    // Resolve once: the admitted endpoint set and the constructed links must agree.
    let device_exchange = !skip && crate::shared::spark_intake::device_exchange_enabled()
        && crate::shared::spark_intake::device_exchange_available(&loaded.library)?;
    // Select and freeze owners before the first weight upload. Header-derived
    // uploads are explicit demands, so the owner chooser sees their true costs.
    let placement = {
        use cuteafd_loader::placement::Baseline;
        ensure!(args.max_sequences > 0, "--max-sequences must be positive");
        let devices: Vec<_> = std::iter::once(args.device).chain(split_device).collect();
        // Each GPU's free bytes less the measured code still to load before ready (lazily loaded
        // functions, cuBLAS: `placement::inventory::LOADED_CODE`), which the planner charges in its
        // baseline.
        let embedding_bytes = if args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu {
            loaded.catalog.tensor("embed.weight")?.metadata.byte_length
        } else { 0 };
        let gpus = devices.iter().enumerate().map(|(rank, &device)| {
            let pending = crate::shared::inventory::pending_code(&loaded.library, device, rank, devices.len() == 2,
                loaded.family, cuteafd_loader::placement::families::deepseek_v4::code_experts(&loaded.catalog),
                &cuteafd_loader::placement::ProgramSet::from_manifest(&loaded.manifest,
                    &[loaded.family, &format!("{}2", loaded.family)]))?;
            crate::shared::peer_split::on_device(&loaded.library, device, args.device, || {
                let (free, total) = loaded.library.cuda_memory_info()?;
                Ok((total as u64, Baseline::Measured { free_bytes: (free as u64).saturating_sub(pending)
                    .saturating_sub(if rank == 0 { embedding_bytes } else { 0 }) }))
            })
        }).collect::<Result<Vec<_>>>()?;
        let cache_stages = if args.dspark && loaded.cfg.dspark_block_size > 0
            && loaded.catalog.tensor("mtp.0.main_proj.weight").is_ok() { 3 } else { 0 };
        let onboard = args.onboard()?;
        let stages = if args.dspark && !args.skip_routed_experts { cache_stages } else { 0 };
        let max_rows = prefill_rows.max(decode_rows);
        let expert_workspace = if args.skip_routed_experts
            || (stages == 0 && (split_device.is_some() || onboard.layers(loaded.cfg.n_layers) == Some(0))) { Some(0) }
            else { local::workspace_bytes(&loaded.library, &args.native_lib, &loaded.catalog,
                max_rows)?.map(|bytes| bytes as u64) };
        let tp2_workspace = if split_device.is_some() && !args.skip_routed_experts
            && onboard.layers(loaded.cfg.n_layers) != Some(0) {
            use crate::shared::experts::rtx::{native::NativeTp2, exl3::Exl3Tp2};
            let measured = if let Some(manifest) = loaded.catalog.exl3() {
                let package = crate::shared::experts::exl3::aot_layout_directory(&args.native_lib,
                    manifest.decoder_tiers(), "rtx-tp2");
                Exl3Tp2::workspace_bytes_for(&package, loaded.cfg.dim, max_rows)
            } else { NativeTp2::workspace_bytes_for(&loaded.library, max_rows) };
            match measured {
                Ok(bytes) => {
                    let planned = cuteafd_loader::serving_capacity::deepseek_v4_tp2_workspace(
                        &loaded.catalog, Some(&args.manifest), max_rows as u64)?;
                    ensure!(bytes as u64 == planned,
                        "V4 TP2 workspace differs from planner: runtime {bytes}, planned {planned}");
                    Some([bytes as u64; 2])
                }
                Err(error) => {
                    tracing::warn!(%error, "V4 TP2 expert package unavailable; auto may keep routed layers on Sparks");
                    None
                }
            }
        } else { None };
        let exchange_f32 = engine::ExchangePolicy::from_env()? == engine::ExchangePolicy::F32;
        let inputs = admission::Inputs { cfg: &loaded.cfg, catalog: &loaded.catalog, manifest: &loaded.manifest,
            family: loaded.family, gpus, loaded_modes: None, cache_stages,
            prefill_rows, decode_rows, max_context, prefix,
            expert_workspace, tp2_workspace, exchange_f32 };
        let request = admission::request(args, &inputs)?;
        let placement = cuteafd_loader::placement::solve(&request)
            .map_err(|error| anyhow::anyhow!("DeepSeek V4 admission: {error}"))?;
        check_modes(&placement, split_device.is_some(), &loaded.cfg)?;
        if !skip {
            use cuteafd_loader::placement::{ExpertHome, SparkWorkingSet};
            let working = SparkWorkingSet {
                spark_layers: placement.layers.iter().enumerate().filter_map(|(layer, home)|
                    (home.experts == ExpertHome::Spark).then_some(layer)).collect(),
                rtx_layers: placement.layers.iter().enumerate().filter_map(|(layer, home)|
                    matches!(home.experts, ExpertHome::RtxWhole { .. } | ExpertHome::RtxTp2).then_some(layer)).collect(),
                spark_ranks: peers.len(), prefill_lanes: 1,
            };
            crate::shared::placement::SparkAdmission::plan(&loaded.snapshot,
                loaded.catalog.routed_experts().geometry()?, Some(&loaded.catalog), &working,
                peers.len(), None, admission::spark_model(decode_rows, device_exchange)?)?.select(&peers)?;
        }
        let loaded_modes = placement.layers.iter().map(|layer| layer.mode).collect::<Vec<_>>();
        let started = Instant::now();
        let loader = weights::WeightLoader {
            library: &loaded.library, catalog: &loaded.catalog, programs: &programs, family: loaded.family, stream,
            device: args.device,
            peers: peer_stream.iter().map(|&(device, stream)| crate::shared::peer_split::RankDevice { device, stream })
                .collect(),
        };
        let (embedding, (model, shares)) = crate::shared::token_io::TokenEmbedding::load(&loaded.library,
            embed_source(&loaded.catalog, loaded.cfg.dim)?, args.token_io.embed_placement, || { let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights"); loader.model_modes(&loaded.cfg, args.dspark, &loaded_modes) })?;
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "DeepSeek V4 coordinator weights resident");

        // Measured cross-check uses the same frozen owner map and explicitly
        // admits the selected pool/expert count. Allocation slack may not
        // silently relocate a loaded layer or shrink the promised pool.
        let measured_gpus = devices.iter().enumerate().map(|(rank, &device)| {
            let pending = crate::shared::inventory::pending_code(&loaded.library, device, rank, devices.len() == 2,
                loaded.family, cuteafd_loader::placement::families::deepseek_v4::code_experts(&loaded.catalog),
                &cuteafd_loader::placement::ProgramSet::from_manifest(&loaded.manifest,
                    &[loaded.family, &format!("{}2", loaded.family)]))?;
            crate::shared::peer_split::on_device(&loaded.library, device, args.device, || {
                let (free, total) = loaded.library.cuda_memory_info()?;
                Ok((total as u64, Baseline::Measured { free_bytes: (free as u64).saturating_sub(pending) }))
            })
        }).collect::<Result<Vec<_>>>()?;
        inventory.record(loaded.family, &measured_gpus.iter().map(|(_, b)| match b {
            Baseline::Measured { free_bytes } => *free_bytes, Baseline::Planned { .. } => 0 }).collect::<Vec<_>>());
        let measured_inputs = admission::Inputs { gpus: measured_gpus, loaded_modes: Some(&loaded_modes), ..inputs };
        let mut measured_request = admission::request(args, &measured_inputs)?;
        measured_request.pool.pin_selected(placement.pool_tokens);
        measured_request.onboard = cuteafd_loader::placement::Onboard::Layers(placement.onboard_layers);
        let checked = cuteafd_loader::placement::solve_frozen(&measured_request, &placement)
            .map_err(|error| anyhow::anyhow!("DeepSeek V4 measured frozen admission: {error}"))?;
        ensure!(checked.layers == placement.layers && checked.pool_tokens == placement.pool_tokens
            && checked.movables == placement.movables && checked.tp2 == placement.tp2
            && checked.expert_ranges == placement.expert_ranges,
            "DeepSeek V4 measured admission disagrees with frozen pre-load placement");
        tracing::info!(pool_tokens = placement.pool_tokens, onboard = %onboard, onboard_layers = placement.onboard_layers,
            local_per_gpu = ?placement.expert_ranges, "DeepSeek V4 placement: {}", placement.summary());
        cuteafd_bench::context::set_resolved("rtx-expert-layers", &placement.onboard_layers.to_string());
        cuteafd_bench::context::set_resolved("rtx-expert-ranges", &placement.expert_ranges.iter()
            .map(|r| format!("{}..{}", r.first, r.first + r.layers)).collect::<Vec<_>>().join(","));
        cuteafd_bench::context::set_resolved("pool-tokens", &placement.pool_tokens.to_string());
        cuteafd_bench::context::set_resolved("attention-placement", &placement.attention_summary());
        cuteafd_bench::context::set_resolved("attention-by-kind", &serde_json::to_string(&placement.attention_by_kind)?);
        (placement, embedding, model, shares)
    };
    let (placement, embedding, model, mut shares) = placement;
    let context_mode = if check_modes(&placement, split_device.is_some(), &loaded.cfg)? {
        Some(cuteafd_loader::placement::families::deepseek_v4::manifest_context(&loaded.manifest, loaded.family,
            loaded.cfg.n_heads, caps["decode_rows"].as_u64().context("decode_rows")?)
            .context("context placement without the build's C4 context programs")?)
    } else { None };
    // The solver charged staging over the compiled extent it was given.
    let context_extent = max_context;
    max_context = crate::shared::context::pool_context("deepseek_v4", max_context, args.max_context == 0,
        usize::try_from(placement.pool_tokens)?, 256)?;
    let shape = pool::PoolShape::new(
        args.max_sequences,
        prefill_rows,
        pool::PoolShape::context_units(pool::PoolShape::units_for(usize::try_from(placement.pool_tokens)?,
            args.max_sequences), context_mode.is_some()),
    );
    let mut engine = engine::Engine::new(engine::EngineParts {
        library: &loaded.library,
        programs: &programs,
        cfg: loaded.cfg.clone(),
        weights: model,
        family: loaded.family,
        decode_rows: caps["decode_rows"].as_u64().context("decode_rows")? as usize,
        prefill_rows,
        prefill_kernel_rows,
        prefill_waves: args.prefill_waves,
        c128_width: usize::try_from(cuteafd_loader::serving_capacity::compiled_c128_width(&loaded.manifest, loaded.family)?)?,
        max_context,
        stream,
        sms: args.sms,
        shape,
        embedding,
        skip_routed: skip,
        context: context_mode,
    })?;
    // Startup and inference share retirement: an error must not drop a rank
    // owner while the other rank is waiting for its missing publication.
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let mut transports = Vec::new();
    let result = (|| -> Result<T> {
        if let Some((device, stream)) = peer_stream {
            engine.attach_peer(device, stream, shares.pop().context("head-split shares")?, engine::PeerParts { shape, tp2: placement.tp2.as_ref().is_some_and(|t| t.layers > 0) })?;
        }
        // Freeze the prefix quota and admit it together with every future table before
        // attach_context or workspace preparation can reserve ordered-upload staging.
        let host_config = admission::host_memory(&engine, args, prefix, context_extent)?;
        let held = held(&engine)?;
        let _ = held; // Prefix bytes were reserved before placement.
        let started = Instant::now();
        let stages = if args.dspark { engine.weights.dspark.as_ref().map_or(0, |d| d.stages.len()) } else { 0 };
        let max_rows = engine.decode_rows.max(engine.prefill_rows);
        let range = &placement.expert_ranges[0];
        ensure!(split_device.is_none() || range.layers == 0,
            "TP1 backbone layers are forbidden under a head split");
        let layers: Vec<_> = (range.first..range.first + range.layers).collect();
        let local = if skip { None } else { local::LocalExperts::load(&loaded.library, &args.native_lib,
            &loaded.catalog, stages, &layers,
            max_rows, usize::try_from(range.peak_bytes)?, stream)? };
        ensure!(local.as_ref().map_or(&[][..], |l| l.layers()) == layers,
            "DeepSeek V4 TP1 allocation differs from admitted placement");
        engine.install_local(local)?;
        if let Some(tp2) = placement.tp2.as_ref().filter(|t| t.layers > 0 && !skip) {
            use crate::shared::experts::rtx::{native::NativeTp2, exl3::Exl3Tp2, RtxExpertLayer};
            use crate::shared::memory::device::Device;
            let devices = [Device { library: &loaded.library, id: args.device },
                Device { library: &loaded.library, id: split_device.context("TP2 needs a head split")? }];
            let layers: Vec<_> = (tp2.first..tp2.first + tp2.layers).collect();
            let budgets = [usize::try_from(tp2.peak_bytes[0])?, usize::try_from(tp2.peak_bytes[1])?];
            let ranks: [Box<dyn RtxExpertLayer>; 2] = if let Some(manifest) = loaded.catalog.exl3() {
                let package = crate::shared::experts::exl3::aot_layout_directory(&args.native_lib,
                    manifest.decoder_tiers(), "rtx-tp2");
                Exl3Tp2::load_pair(devices, &loaded.catalog, &package, layers.clone(), max_rows, budgets)?
                    .map(|rank| Box::new(rank) as Box<dyn RtxExpertLayer>)
            } else {
                NativeTp2::load_pair(devices, &loaded.catalog, layers.clone(), max_rows, budgets)?
                    .map(|rank| Box::new(rank) as Box<dyn RtxExpertLayer>)
            };
            ensure!(ranks.iter().all(|r| r.layers() == layers), "TP2 loaded layers differ from admitted placement");
            engine.install_tp2(ranks)?;
        }
        // The auto reduction plan uses draft_block(), which is nonzero only after
        // the drafter's local stage experts are installed. Freeze it before capture.
        if context_mode.is_some() {
            engine.attach_context(context_extent, &loaded.manifest, args.attention_plan_rows.0)?;
        }
        tracing::info!(tp1 = ?placement.expert_ranges, tp2 = ?placement.tp2,
            elapsed_ms = started.elapsed().as_millis() as u64, "DeepSeek V4 expert layers resident on coordinator GPUs");
        // Declare rank ownership once for every transport and device lane.
        let spark_world = (!peers.is_empty()).then(|| cuteafd_transport::expert::SparkWorld::canonical(peers.len())).transpose()?;
        // Decode graphs retain fixed reply pointers; prefill alone rotates N
        // FIFO return slots on its separate connection set.
        if !skip {
            transports.push(SparkLink::new(&loaded.library,
                spark_world.context("Spark transport needs a declared world")?, &peers, engine.decode_rows as u32,
                TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(), timeout: Duration::from_secs(120), max_frame_bytes: 64 << 20 },
                loaded.cfg.dim * 2)?);
            transports.push(SparkLink::new_waves(&loaded.library,
                spark_world.context("Spark transport needs a declared world")?, &peers, 4096,
                TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(), timeout: Duration::from_secs(120), max_frame_bytes: 64 << 20 },
                loaded.cfg.dim * 2, engine.prefill_waves, engine.prefill_rows as u32)?);
        }
        for transport in &mut transports { transport.enable_terminal_ownership(loaded.library.clone())?; }
        // Every rank's persistent bootstrap runs before the warm-up wave does,
        // so the workers' installed grouping is readable at any later point.
        let warm_shape = |index| if index == 0 {
            (engine.decode_rows, cuteafd_transport::ExpertV2SourceKind::Decode)
        } else { (engine.prefill_rows, cuteafd_transport::ExpertV2SourceKind::Prefill) };
        for (index, transport) in transports.iter_mut().enumerate() {
            let (rows, kind) = warm_shape(index);
            transport.bootstrap(&engine.warm_request(rows, kind)?)?;
        }
        crate::shared::memory_report::release_load_staging(&loaded.library);
        let started = Instant::now();
        let identity = if skip { None } else { Some(cuteafd_transport::worker_selection::WorkerIdentity::from_snapshot(
            &loaded.snapshot, loaded.catalog.routed_experts().geometry()?, spark_world.context("worker identity needs a declared world")?)?) };
        let mut installed = None;
        for (index, transport) in transports.iter_mut().enumerate() {
            let (rows, kind) = warm_shape(index);
            engine.warm_transport(transport, &runtime, rows, kind)?;
            let reports = transport.transport.installed_grouping()?;
            for (rank, report) in reports.iter().enumerate() {
                report.validate_for(identity.as_ref().context("Spark worker identity")?, rank,
                    engine.decode_rows, engine.local_layers()..loaded.cfg.n_layers)?;
            }
            if let Some(previous) = &installed {
                ensure!(previous == &reports, "Spark transports report different installed expert packages");
            } else { installed = Some(reports); }
        }
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "Spark expert transport warm");
        if device_exchange {
            let started = Instant::now();
            engine.attach_device_link(spark_world.context("device exchange needs a declared world")?, &peers,
                TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(), timeout: Duration::from_secs(120), max_frame_bytes: 64 << 20 })?;
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "decode and verify waves use the device exchange");
        }
        engine.prepare_workspaces()?;
        if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
        engine.warm_local_graphs()?;
        engine.check_routes(&mut transports, &runtime)?;
        let (graphs0, graphs1, captures) = engine.graph_inventory();
        tracing::info!(graphs0, graphs1, captures, units = engine.shape.units,
            classes = metadata::c4_width_classes(engine.shape.units).len(), "V4 decode graph inventory");
        let _coverage_session = engine.graph_coverage.session();
        body(&engine, &mut transports, &runtime, host_config)
    })();
    crate::shared::decode_graph::fatal_drain(engine.shutdown_submission(&mut transports), "V4 RTX engine retirement");
    if let Err(error) = &result {
        // Teardown may fail after a device fault and would otherwise hide this.
        tracing::error!("{error:#}");
    }
    drop(engine);
    drop(transports);
    // SAFETY: the engine that used the streams is gone.
    unsafe {
        loaded.library.cuda_stream_destroy(stream)?;
        if let Some((device, stream)) = peer_stream {
            loaded.library.cuda_set_device(device)?;
            let destroyed = loaded.library.cuda_stream_destroy(stream);
            loaded.library.cuda_set_device(args.device)?;
            destroyed?;
        }
    }
    result
}

fn golden(args: GoldenArgs) -> Result<()> {
    let shared = args.golden.join("manifest.json").try_exists()?;
    if shared {
        ensure!(args.nll && args.save_logits.is_some(), "shared V4 fidelity needs --nll --save-logits DIR");
        ensure!(args.prefill.is_none() && !args.draft && args.verify_rows.is_none()
            && args.token_check.is_none() && args.resume_at.is_empty() && args.layers.is_none(),
            "shared V4 fidelity cannot be combined with legacy golden diagnostics");
        let output = args.save_logits.as_deref().unwrap();
        ensure!(!output.try_exists()? && output.parent().is_some_and(std::path::Path::is_dir),
            "V4 fidelity output needs an existing parent and a new directory");
    } else {
        ensure!(args.window.is_none(), "--window needs a published shared fidelity set");
    }
    let started = Instant::now();
    let loaded = load(&args.engine)?;
    let cfg = loaded.cfg.clone();
    let windows = if shared { Some(fidelity_windows(&args.golden, loaded.family, args.window.as_deref())?) } else { None };
    with_engine(&loaded, &args.engine, None, |_| Ok(0), |engine, transports, runtime, _host_config| {
        let before = engine.graph_capture_counts();
        println!("golden ready: {:.3} s | graph captures {} | TP2 expert graph captures {}",
            started.elapsed().as_secs_f64(), before.0, before.1);
        let result = if let Some(windows) = &windows {
            score_fidelity(&args, engine, transports, runtime, windows)
        } else { match args.token_check {
            Some(steps) => token_check(&args, &loaded, engine, transports, runtime, steps),
            None if !args.resume_at.is_empty() => resume(&args, engine, transports, runtime),
            None if args.nll => nll_run(&args, &cfg, engine, transports, runtime),
            None => golden_run(&args, &cfg, engine, transports, runtime),
        } };
        let after = engine.graph_capture_counts();
        println!("golden post-ready captures: all {} | TP2 experts {}", after.0 - before.0, after.1 - before.1);
        for setting in cuteafd_bench::context::get().settings.into_iter().filter(|s| s.name.starts_with("route-")) {
            println!("golden {}: {}", setting.name, setting.value.unwrap_or_default());
        }
        result
    })
}

fn fidelity_publication(family: &str) -> Result<(&'static str, &'static str, &'static str)> {
    use cuteafd_bench::fidelity_dataset;
    Ok(match family {
        "dsv4f" => (fidelity_dataset::V4FLASH_CONFIG, fidelity_dataset::V4FLASH_REVISION,
            "0566bf91385805a7ab98f850e066c173b467db8bdf574240edee854290d33100"),
        "dsv4p" => (fidelity_dataset::V4PRO_CONFIG, fidelity_dataset::V4PRO_REVISION,
            "7b4fd7b51670ff03fbfab6a21b72d999163ddee9bd3eec34c89f1d6c2f83488f"),
        _ => anyhow::bail!("no shared V4 fidelity publication for {family}"),
    })
}

fn fidelity_windows(root: &std::path::Path, family: &str, selected: Option<&str>)
    -> Result<Vec<cuteafd_bench::reference::Window>> {
    use cuteafd_bench::fidelity_dataset;
    use sha2::{Digest, Sha256};
    let (config, _, hash) = fidelity_publication(family)?;
    let bytes = std::fs::read(root.join("manifest.json"))?;
    ensure!(format!("{:x}", Sha256::digest(&bytes)) == hash,
        "V4 fidelity manifest differs from immutable publication");
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == config && manifest["vocab"] == 129280,
        "V4 fidelity config/vocabulary differs");
    let reference = fidelity_dataset::load(root, &manifest)?;
    let mut windows = fidelity_dataset::quick_windows(&reference)?;
    if let Some(id) = selected {
        windows.retain(|window| window.id == id);
        ensure!(windows.len() == 1, "V4 fidelity window must belong to the sealed quick subset");
    }
    ensure!(windows.iter().all(|window| window.media.is_empty() && window.tokens.len() <= 16384
        && window.score_from > 0 && window.score_from < window.tokens.len()),
        "V4 fidelity requires text-only bounded causal windows");
    Ok(windows)
}

fn fidelity_chunks(tokens: usize, capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    ensure!((2..=16384).contains(&tokens) && capacity > 0, "V4 fidelity causal window/chunk extent differs");
    let count = tokens - 1;
    Ok((0..count).step_by(capacity).map(|first| first..(first + capacity).min(count)).collect())
}

fn score_fidelity(args: &GoldenArgs, engine: &engine::Engine<'_>, transports: &mut [SparkLink<'_>],
    runtime: &tokio::runtime::Runtime, windows: &[cuteafd_bench::reference::Window]) -> Result<()> {
    use cuteafd_api::openai::probe::{Probe, ProbeSpec};
    use cuteafd_bench::{fidelity_rows, reference::Fidelity};
    let (config, revision, hash) = fidelity_publication(engine.family)?;
    ensure!(engine.weights.layers.len() == engine.cfg.n_layers && engine.cfg.vocab_size == 129280
        && !args.engine.skip_routed_experts, "V4 fidelity needs every target layer, routed experts and published vocabulary");
    let root = args.save_logits.as_deref().context("shared V4 fidelity requires --save-logits")?;
    std::fs::create_dir(root).context("V4 fidelity output must be a new directory with an existing parent")?;
    let chunk_rows = args.chunk.unwrap_or(engine.prefill_capacity()).min(engine.prefill_capacity());
    ensure!(chunk_rows > 0, "V4 fidelity chunk must be positive");
    let provenance = serde_json::json!({"prefill_rows":engine.prefill_rows,
        "prefill_waves":engine.prefill_waves,"prefill_span_rows":chunk_rows});
    let mut allocator = pool::PoolAllocator::for_engine(engine)?;
    let mut records = Vec::new();
    let vocab = engine.cfg.vocab_size;
    for window in windows {
        let mut placement = allocator.admit(window.tokens.len() - 1)?;
        let dump = root.join(&window.id);
        let probe = Probe::new(ProbeSpec { cold: true, no_speculation: true,
            prompt_ids: Some(window.tokens.clone()), score_from: Some(window.score_from),
            score_positions: Some(window.positions.iter().map(|p| p.pos).collect()), top_k: 32,
            want: window.want(), dump_rows: Some(dump.clone()), ..Default::default() });
        probe.admitted("deepseek_v4", &window.tokens, 0);
        probe.selected_score_path("prefill");
        probe.provenance(provenance.clone());
        for chunk in fidelity_chunks(window.tokens.len(), chunk_rows)? {
            let first_scored = chunk.start.max(window.score_from - 1).min(chunk.end);
            let wanted = chunk.end - first_scored;
            let logits = engine.prefill(&mut placement, &window.tokens[chunk.clone()], transports, runtime, wanted, None)?;
            ensure!(logits.len() == wanted * vocab, "V4 fidelity logits shape differs");
            for (row, logits) in logits.chunks_exact(vocab).enumerate() {
                probe.row(first_scored + row + 1, logits);
            }
            if wanted > 0 { probe.scored_prefill_chunk(first_scored + 1, wanted); }
        }
        let record = probe.record();
        ensure!(record.error.is_none(), "V4 fidelity row dump failed: {:?}", record.error);
        let mut score = window.score(&record.rows);
        ensure!(score.missing == 0 && score.non_finite == 0 && score.records.len() == window.positions.len(),
            "V4 fidelity did not cover every finite scored row");
        fidelity_rows::score_compact(vocab, window, &dump, &mut score)?;
        println!("V4 window {}: rows={} top-1={:.4}% mean NLL={:.8} mean KL(top1024+tail)={:.8}",
            window.id, score.positions, score.top1 * 100.0, score.nll, score.kl);
        let file = std::fs::OpenOptions::new().write(true).create_new(true).open(root.join(format!("{}.json", window.id)))?;
        serde_json::to_writer(file, &serde_json::json!({"probe":record,"score":score}))?;
        records.extend(score.records);
        engine.drain_state()?;
        allocator.release(placement);
    }
    let score = Fidelity::from_records(records);
    println!("V4 prefill logits: rows={} top-1 agreement {:.4}% | mean NLL engine {:.8} | mean KL(top1024+tail) {:.8}",
        score.positions, score.top1 * 100.0, score.nll, score.kl);
    let file = std::fs::OpenOptions::new().write(true).create_new(true).open(root.join("fidelity.json"))?;
    serde_json::to_writer(file, &serde_json::json!({"schema":"cuteafd.deepseek_v4.golden/1",
        "reference_config":config,"reference_revision":revision,"reference_manifest_sha256":hash,
        "path_shape":"prefill-shaped","kl_kind":"top1024+tail","provenance":provenance,"score":score}))?;
    Ok(())
}

/// `--nll` scores the complete prompt with prefill-shaped kernels, never decode fallbacks.
fn nll_run(args: &GoldenArgs, cfg: &DeepseekV4Config, engine: &engine::Engine<'_>,
    transports: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime) -> Result<()> {
    use std::hash::{Hash, Hasher};
    let bytes = std::fs::read(args.golden.join("tokens.bin"))?;
    ensure!(bytes.len() % 4 == 0, "tokens.bin must contain U32 tokens");
    let tokens: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let vocab = cfg.vocab_size;
    ensure!(tokens.len() >= 2 && tokens.iter().all(|&t| (t as usize) < vocab), "invalid golden tokens");
    let golden = match std::fs::read(args.golden.join("logits.bin")) {
        Ok(bytes) => {
            ensure!(bytes.len() == tokens.len() * vocab * 4, "golden logits row coverage differs");
            Some(f32s(&bytes))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let mut allocator = pool::PoolAllocator::for_engine(engine)?;
    let mut placement = allocator.admit(tokens.len())?;
    let chunk = args.chunk.unwrap_or(engine.prefill_capacity()).min(engine.prefill_capacity());
    ensure!(chunk > 0, "prefill chunk must be positive");
    let started = Instant::now();
    let mut logits = Vec::new();
    for tokens in tokens.chunks(chunk) {
        logits.extend(engine.prefill(&mut placement, tokens, transports, runtime, tokens.len(), None)?);
    }
    allocator.release(placement);
    let seconds = started.elapsed().as_secs_f64();
    ensure!(logits.len() == tokens.len() * vocab && logits.iter().all(|v| v.is_finite()),
        "missing or nonfinite engine logits");
    if let Some(dir) = &args.save_logits {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("tokens.bin"), bytes)?;
        std::fs::write(dir.join("logits.bin"), logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    }
    let lp = |row: &[f32]| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max + row.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln();
        row.iter().map(|&v| v as f64 - lse).collect::<Vec<_>>()
    };
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0;
    let (mut nll, mut reference_nll, mut kl, mut agree) = (0.0, 0.0, 0.0, 0usize);
    let mut digest = std::collections::hash_map::DefaultHasher::new();
    logits.iter().for_each(|v| v.to_bits().hash(&mut digest));
    for (r, ours) in logits.chunks_exact(vocab).enumerate() {
        let q = lp(ours);
        if let Some(&next) = tokens.get(r + 1) { nll -= q[next as usize]; }
        if let Some(golden) = &golden {
            let theirs = &golden[r * vocab..][..vocab];
            ensure!(theirs.iter().all(|v| v.is_finite()), "nonfinite golden row {r}");
            agree += usize::from(argmax(ours) == argmax(theirs));
            let p = lp(theirs);
            kl += p.iter().zip(&q).map(|(p,q)| p.exp() * (p-q)).sum::<f64>();
            if let Some(&next) = tokens.get(r + 1) { reference_nll -= p[next as usize]; }
        }
    }
    let rows = tokens.len();
    if golden.is_some() {
        println!("prefill logits: {rows} tokens in {seconds:.2} s | top-1 agreement {:.2}% | mean NLL engine {:.4} golden {:.4} | mean KL(golden||engine) {:.5}",
            100.0 * agree as f64 / rows as f64, nll / (rows-1) as f64, reference_nll / (rows-1) as f64, kl / rows as f64);
    } else {
        println!("prefill logits: {rows} tokens in {seconds:.2} s | mean NLL engine {:.4} | logits digest {:016x}",
            nll / (rows-1) as f64, digest.finish());
    }
    Ok(())
}

/// `--resume-at P,..`: [`prefix::resume_check`] for every P and every --prefill-chunk.
fn resume(args: &GoldenArgs, engine: &engine::Engine<'_>, transports: &mut [SparkLink<'_>],
    runtime: &tokio::runtime::Runtime) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let chunks = if args.prefill_chunk.is_empty() { vec![engine.prefill_rows] } else { args.prefill_chunk.clone() };
    let cases: Vec<prefix::ResumeCase> = chunks.iter().flat_map(|&chunk| args.resume_at.iter().map(move |&at| (at, chunk)))
        .map(|(at, chunk)| prefix::ResumeCase { at, n: args.prefill.unwrap_or(at + args.resume_span).min(tokens.len()),
            chunk })
        .collect();
    let mut runner = prefix::Runner { engine, transports, runtime, lanes: args.resume_lanes };
    let passed = prefix::resume_check(&mut runner, &tokens, &cases, args.resume_decode, args.resume_cold,
        args.resume_repeat)?;
    ensure!(passed, "a restored sequence differs from the straight one (see above)");
    println!("resume check: all {} cases byte-identical", cases.len());
    Ok(())
}

fn golden_run(
    args: &GoldenArgs,
    cfg: &DeepseekV4Config,
    engine: &engine::Engine<'_>,
    transports: &mut [SparkLink<'_>],
    runtime: &tokio::runtime::Runtime,
) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let started = Instant::now();
    let compare_layers = args.layers.unwrap_or(cfg.n_layers);
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    let mut allocator = pool::PoolAllocator::for_engine(engine)?;
    let mut placement = allocator.admit(tokens.len())?;
    let chunk = args.chunk.unwrap_or(engine.prefill_capacity()).min(engine.prefill_capacity());
    let mut logits = Vec::new();
    let mut offset = 0;
    while offset < prefill {
        let end = (offset + chunk).min(prefill);
        let mut compare = |layer: usize, stream: &[u8]| -> Result<()> {
            if layer < compare_layers {
                let golden = std::fs::read(args.golden.join(format!("layer{layer:02}.bin")))?;
                ensure!(golden.len() == stream.len(), "golden layer {layer} has {} bytes, engine {}", golden.len(), stream.len());
                let (cosine, rel) = similarity(&bf16s(stream), &bf16s(&golden));
                println!("layer {layer:2}: cosine {cosine:.6} rel_l2 {rel:.3e}");
            }
            Ok(())
        };
        // Per-layer streams are compared only for a whole-prompt prefill.
        let whole = offset == 0 && end == tokens.len() && compare_layers > 0;
        logits.extend(engine.prefill(&mut placement, &tokens[offset..end], transports, runtime,
            end - offset, if whole { Some(&mut compare) } else { None })?);
        offset = end;
    }
    let prefill_elapsed = started.elapsed();
    println!("prefill host phases: {}", engine.profile.borrow().report());
    *engine.profile.borrow_mut() = engine::Profile::default();
    let decode_started = Instant::now();
    let mut position = prefill;
    let (mut drafted, mut matched, mut draft_steps) = (0usize, vec![0usize; engine.draft_block() + 1], 0usize);
    let mut draft_seconds = 0f64;
    while position < tokens.len() {
        if args.draft {
            let block = engine.draft_block();
            let mut inputs = vec![tokens[position]];
            inputs.resize(block, cfg.dspark_noise_token_id as u32);
            let started = Instant::now();
            let drafts = engine.draft(&[engine::DraftRequest { placement: &placement, token: tokens[position] }], &inputs)?;
            draft_seconds += started.elapsed().as_secs_f64();
            let truth = &tokens[(position + 1).min(tokens.len())..(position + 1 + block).min(tokens.len())];
            let accepted = drafts[0].iter().zip(truth).take_while(|(d, t)| d == t).count();
            matched[accepted] += 1;
            drafted += truth.len();
            draft_steps += 1;
        }
        let end = (position + args.verify_rows.unwrap_or(1)).min(tokens.len());
        logits.extend(if args.verify_rows.is_some() {
            engine.verify(&mut [(&mut placement, &tokens[position..end])], transports, runtime)?
        } else {
            engine.decode(&mut [(&mut placement, tokens[position])], transports.first_mut(), runtime)?
        });
        position = end;
    }
    if args.draft && draft_steps > 0 {
        let accepted: usize = matched.iter().enumerate().map(|(n, count)| n * count).sum();
        println!("drafts: {draft_steps} steps, {accepted} of {drafted} drafted tokens accepted as a prefix \
            ({:.2} per step), histogram {matched:?}, {:.2} ms/draft", accepted as f64 / draft_steps as f64,
            draft_seconds * 1e3 / draft_steps as f64);
    }
    let decode_steps = tokens.len() - prefill;
    if decode_steps > 0 {
        println!("decode host phases: {}", engine.profile.borrow().report());
        println!("decode: {decode_steps} steps in {:.2} s ({:.1} ms/token)", decode_started.elapsed().as_secs_f64(),
            decode_started.elapsed().as_secs_f64() * 1e3 / decode_steps as f64);
    }
    println!("prefill: {prefill} tokens in {:.2} s", prefill_elapsed.as_secs_f64());
    let vocab = cfg.vocab_size;
    let golden = f32s(&std::fs::read(args.golden.join("logits.bin"))?);
    ensure!(golden.len() == logits.len(), "golden logits {} vs engine {}", golden.len(), logits.len());
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap();
    let (mut agree, mut next_ok, mut golden_next_ok) = (0usize, 0usize, 0usize);
    for (row, (ours, theirs)) in logits.chunks_exact(vocab).zip(golden.chunks_exact(vocab)).enumerate() {
        let (a, b) = (argmax(ours), argmax(theirs));
        agree += usize::from(a == b);
        if row + 1 < tokens.len() {
            next_ok += usize::from(a == tokens[row + 1] as usize);
            golden_next_ok += usize::from(b == tokens[row + 1] as usize);
        }
    }
    let t = tokens.len();
    let (cosine, rel) = similarity(&logits[(t - 1) * vocab..], &golden[(t - 1) * vocab..]);
    println!(
        "logits: top-1 agreement {:.1}% | next-token accuracy engine {:.1}% golden {:.1}% | last row cosine {cosine:.6} rel_l2 {rel:.3e}",
        100.0 * agree as f64 / t as f64,
        100.0 * next_ok as f64 / (t - 1) as f64,
        100.0 * golden_next_ok as f64 / (t - 1) as f64,
    );
    if decode_steps > 0 {
        let (mut decode_agree, mut worst) = (0usize, 1f64);
        for row in prefill..t {
            let (ours, theirs) = (&logits[row * vocab..][..vocab], &golden[row * vocab..][..vocab]);
            decode_agree += usize::from(argmax(ours) == argmax(theirs));
            if argmax(ours) != argmax(theirs) {
                let margin = |l: &[f32]| { let mut v = l.to_vec(); v.sort_by(|a, b| b.total_cmp(a)); v[0] - v[1] };
                println!("decode row {row}: engine {} golden {} (golden top-2 margin {:.3}, cosine {:.6})",
                    argmax(ours), argmax(theirs), margin(theirs), similarity(ours, theirs).0);
            }
            worst = worst.min(similarity(ours, theirs).0);
        }
        // FNV-1a over the decode rows' logits bits: equal digests are byte-identical runs.
        let digest = logits[prefill * vocab..].iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
            (h ^ u64::from(v.to_bits())).wrapping_mul(0x100_0000_01b3)
        });
        println!("decode rows: top-1 agreement {:.1}% | worst row cosine {worst:.6} | logits digest {digest:016x}",
            100.0 * decode_agree as f64 / decode_steps as f64);
    }
    if let Some(dir) = &args.save_logits {
        ensure!(decode_steps > 0, "--save-logits without --nll saves the decode rows: pass --prefill N");
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("tokens.bin"), tokens[prefill..].iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>())?;
        std::fs::write(dir.join("logits.bin"), logits[prefill * vocab..].iter().flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>())?;
        println!("saved {decode_steps} decode rows (positions {prefill}..{t}) to {}", dir.display());
    }
    Ok(())
}

/// `--token-check N`: prefill the golden prompt, then [`crate::shared::token_io::gate`]
/// over N greedy decode steps.
fn token_check(args: &GoldenArgs, loaded: &Loaded, engine: &engine::Engine<'_>, transports: &mut [SparkLink<'_>],
    runtime: &tokio::runtime::Runtime, steps: usize) -> Result<()> {
    let mut tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if let Some(p) = args.prefill {
        tokens.truncate(p);
    }
    let mut allocator = pool::PoolAllocator::for_engine(engine)?;
    let mut placement = allocator.admit(tokens.len() + steps + 1)?;
    let chunks: Vec<&[u32]> = tokens.chunks(engine.prefill_capacity()).collect();
    let mut last = None;
    for (i, chunk) in chunks.iter().enumerate() {
        last = engine.prefill_device(&mut placement, chunk, transports, runtime, usize::from(i + 1 == chunks.len()))?;
    }
    let last = last.context("prefill produced no logits")?.row_host(&loaded.library, 0)?;
    let first = cuteafd_core::TargetSamplingParams::greedy().select_token(&last, None, 0)? as u32;
    let result = crate::shared::token_io::gate(&loaded.library, &engine.embedding, first, steps, |token| {
        engine.decode_device(&mut [(&mut placement, token)], transports.first_mut(), runtime)
    });
    allocator.release(placement);
    result
}

#[cfg(test)]
mod fidelity_tests {
    use super::*;

    #[test]
    fn published_fidelity_identity_is_family_specific() {
        let (flash, flash_revision, flash_hash) = fidelity_publication("dsv4f").unwrap();
        let (pro, pro_revision, pro_hash) = fidelity_publication("dsv4p").unwrap();
        assert_ne!(flash, pro);
        assert_ne!(flash_revision, pro_revision);
        assert_ne!(flash_hash, pro_hash);
        assert_eq!(flash_hash.len(), 64);
        assert_eq!(pro_hash.len(), 64);
        assert!(fidelity_publication("dsv4f2").is_err());
    }

    #[test]
    fn fidelity_chunks_predict_each_next_token_once_without_final_token_input() -> Result<()> {
        for tokens in [2, 513, 576, 2011, 4610, 11284, 16384] {
            for capacity in [512, 4096, 8192, 16384] {
                let chunks = fidelity_chunks(tokens, capacity)?;
                assert_eq!(chunks.first().unwrap().start, 0);
                assert_eq!(chunks.last().unwrap().end, tokens - 1);
                assert_eq!(chunks.iter().map(|chunk| chunk.len()).sum::<usize>(), tokens - 1);
                assert!(chunks.iter().all(|chunk| !chunk.is_empty() && chunk.len() <= capacity));
                assert!(chunks.windows(2).all(|pair| pair[0].end == pair[1].start));
            }
        }
        assert!(fidelity_chunks(1, 4096).is_err());
        assert!(fidelity_chunks(16385, 4096).is_err());
        assert!(fidelity_chunks(513, 0).is_err());
        Ok(())
    }

    #[test]
    fn common_span_preserves_unit_cuts_at_one_and_two_waves() -> Result<()> {
        use crate::shared::prefill_pipeline::prefill_units;
        // Full spans have an even unit count; tiny tails cannot split below the floor.
        for rows in [4096, 2048, 1024, 512] {
            for span in [8192, 16384, 256] {
                assert_eq!(prefill_units(span, rows, 1, 256)?, prefill_units(span, rows, 2, 256)?);
            }
        }
        for (span, rows) in [(4609, 4096), (4613, 4096), (11283, 2048), (11136, 2048)] {
            assert_eq!(prefill_units(span, rows, 1, 256)?, prefill_units(span, rows, 2, 256)?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod program_selection_tests {
    use cuteafd_core::coordinator_programs::{AttentionReductionPlan, CoordinatorPrograms};

    #[test]
    fn context_plan_is_frozen_after_draft_experts_and_before_capture() {
        let source = include_str!("mod.rs");
        let startup = source.split("let result = (|| -> Result<T> {").nth(1).unwrap()
            .split("body(&engine, &mut transports, &runtime)").next().unwrap();
        let install = startup.find("engine.install_local(local)").unwrap();
        let attach = startup.find("engine.attach_context(").unwrap();
        let capture = startup.find("engine.warm_local_graphs()").unwrap();
        assert!(install < attach && attach < capture,
            "context auto plan must see resident draft experts and precede graph capture");
        assert_eq!(startup.matches("engine.attach_context(").count(), 1);
        let context = include_str!("engine/context.rs");
        assert!(context.contains("let block = self.draft_block();"));
        assert!(context.contains("(block > 0).then_some(block)"));
        let ready = include_str!("engine/dspark.rs");
        assert!(ready.contains("self.local.borrow().as_ref().is_some_and(|l| l.stages() == stages)"));
        assert_eq!(AttentionReductionPlan::new(64, Some(5), None).unwrap().rows(), 6);
        assert_eq!(AttentionReductionPlan::new(64, None, None).unwrap().rows(), 1);
    }

    #[test]
    fn every_v4_dispatch_template_is_selected_and_required_at_startup() {
        let sources = [include_str!("engine.rs"), include_str!("engine/dspark.rs"), include_str!("weights.rs")];
        for family in ["dsv4f", "dsv4p"] {
            let split_family = format!("{family}2");
            for split in [false, true] {
                let selected = CoordinatorPrograms { family, split_family: split.then_some(split_family.as_str()) };
                let required = selected.v4_required(4096, 64);
                let mut checked = 0;
                for source in sources {
                    for line in source.lines().filter(|line| line.contains("self.run")) {
                        // The only nonliteral route is run -> run_on forwarding `name`.
                        let args = line.split("self.run").nth(1).unwrap();
                        let Some(start) = args.find('"') else {
                            assert!(args.trim() == "_on(0, false, name, pointers, scalars)",
                                "unrecognized nonliteral dispatch: {line}");
                            continue;
                        };
                        let template = args[start + 1..].split('"').next().unwrap();
                        for (mode, cap) in [("decode", "64"), ("prefill", "4096")] {
                            if template.contains("_decode_") && mode != "decode" { continue; }
                            for attention in ["win", "c4", "c128"] {
                                for ratio in ["4", "128"] {
                                    for program in ["prefill", "continuation"] {
                                        if template.contains("{}") {
                                            assert!(line.contains("self.kernel_rows(cap)"),
                                                "unrecognized positional dispatch capacity: {line}");
                                        }
                                        let suffix = template.replace("{cap}", cap).replace("{}", cap).replace("{mode}", mode)
                                            .replace("{attention}", attention).replace("{ratio}", ratio)
                                            .replace("{program}", program);
                                        let name = format!("{family}_{suffix}");
                                        assert!(selected.contains(&name), "unselected dispatch {name}");
                                        assert!(required.contains(&name), "startup misses dispatch {name}");
                                        if split && (args.starts_with("_on(1, true,") || args.starts_with("_on(rank, split,")
                                            || args.starts_with("_on(rank, weights.attention_split,")
                                            || args.starts_with("_on(0, weights.ffn_split,") || args.starts_with("_on(1, weights.ffn_split,")) {
                                            let name = format!("{split_family}_{suffix}");
                                            assert!(selected.contains(&name) && required.contains(&name),
                                                "startup misses split dispatch {name}");
                                        }
                                    }
                                }
                            }
                        }
                        checked += 1;
                    }
                }
                assert!(checked >= 20, "dispatch parser must cover engine and drafter calls");
                for source in sources {
                    for call in source.split("self.programs.program(").skip(1) {
                        let template = call.split('"').nth(1).unwrap();
                        if template.contains("block_fp8_scale_prep") {
                            let name = format!("{family}_block_fp8_scale_prep");
                            assert!(selected.contains(&name) && required.contains(&name));
                        } else {
                            assert!(["{}_{name}", "{family}_{name}"].contains(&template),
                                "unrecognized direct program dispatch {template}");
                        }
                    }
                }
                selected.validate_v4(4096, 64, &required).unwrap();
                for missing in &required {
                    let incomplete = required.iter().filter(|name| *name != missing);
                    let error = selected.validate_v4(4096, 64, incomplete).unwrap_err();
                    assert_eq!(&error.0, missing);
                    assert!(error.to_string().contains(missing));
                }
            }
        }
    }
}
