//! DeepSeek V4 (Flash / Pro) coordinator engine over the exported b12x programs.
pub(crate) mod engine;
pub(crate) mod admission;
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
    /// `max` (the one-RTX default) places the most layers that still leave a 262K
    /// pool (v2's policy); `auto` (two-RTX default) reserves the 2M KV pool first and fills
    /// what is left; `N`, `N%` or `all` fix the RTX layers and the KV pool
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
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// Score every prefill row: mean next-token NLL, top-1 agreement and KL against golden.
    #[arg(long)]
    pub nll: bool,
    /// With --nll, also save tokens.bin and F32 logits.bin for a reference run.
    #[arg(long, requires = "nll")]
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
    pub(crate) fn onboard(&self) -> Result<cuteafd_loader::placement::Onboard> {
        use cuteafd_loader::placement::Onboard;
        Ok(match (self.rtx_expert_layers, self.local_expert_layers) {
            (Some(onboard), _) => onboard,
            (None, Some(layers)) => Onboard::Layers(layers),
            (None, None) => cuteafd_loader::placement::families::deepseek_v4::default_onboard(if self.split_device.is_some() { 2 } else { 1 }),
        })
    }

    pub(crate) fn fixed_onboard(&self) -> bool {
        self.rtx_expert_layers.is_some() || self.local_expert_layers.is_some()
    }
}

/// The engine runs one shape: every layer head-split under `--split-device`,
/// every layer whole on GPU0 otherwise, and no hop but the step-start
/// streams push. Anything else the solver hands it is refused before any
/// cache or expert is allocated.
fn check_modes(placement: &cuteafd_loader::placement::Placement, split: bool) -> Result<()> {
    use cuteafd_loader::placement::{families::DEEPSEEK_V4, FfnMode, LayerMode};
    DEEPSEEK_V4.check(placement).map_err(|error| anyhow::anyhow!("DeepSeek V4 placement: {error}"))?;
    let engine = if split { LayerMode::HeadSplit } else { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } };
    if let Some((layer, assignment)) = placement.layers.iter().enumerate().find(|(_, a)| a.mode != engine) {
        anyhow::bail!("DeepSeek V4 runs every layer as {engine:?}; the placement has layer {layer} as {:?}",
            assignment.mode);
    }
    Ok(())
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
    pub library: NativeLibrary,
    pub cfg: DeepseekV4Config,
    pub family: &'static str,
    pub manifest: serde_json::Value,
}

pub(crate) fn load(args: &EngineArgs) -> Result<Loaded> {
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    let geometry = catalog.routed_experts().geometry()?;
    cuteafd_core::set_expert_geometry(geometry).map_err(|g| anyhow::anyhow!("geometry already {g:?}"))?;
    let family = match geometry.family() {
        Some("dsv4f") => "dsv4f",
        Some("dsv4p") => "dsv4p",
        other => anyhow::bail!("not a DeepSeek V4 checkpoint (expert family {other:?})"),
    };
    let cfg = DeepseekV4Config::read(&args.snapshot, 1)?;
    let library = unsafe { NativeLibrary::load(&args.native_lib) }?;
    library.cuda_set_device(args.device)?;
    let manifest = serde_json::from_str(&std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("reading {}", args.manifest.display()))?)?;
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
    body: impl FnOnce(&engine::Engine<'_>, &mut [SparkLink<'_>], &tokio::runtime::Runtime) -> Result<T>,
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
    let (loaded_count, skipped) = programs.load_matching(|name| selected.contains(name))?;
    tracing::info!(loaded = loaded_count, skipped, elapsed_ms = started.elapsed().as_millis() as u64,
        "DeepSeek V4 programs loaded");
    let peer_stream = match split_device {
        Some(device) => {
            ensure!(device != args.device, "--split-device must differ from --device");
            loaded.library.cuda_enable_peer(device)?;
            loaded.library.cuda_set_device(device)?;
            let stream = loaded.library.cuda_enable_peer(args.device)
                .and_then(|()| programs.load_matching(|name| selected.contains(name)).map(|_| ()))
                .and_then(|()| loaded.library.cuda_stream_create());
            loaded.library.cuda_set_device(args.device)?;
            Some((device, stream?))
        }
        None => None,
    };
    let started = Instant::now();
    let loader = weights::WeightLoader {
        library: &loaded.library, catalog: &loaded.catalog, programs: &programs, family: loaded.family, stream,
        device: args.device,
        peers: peer_stream.iter().map(|&(device, stream)| crate::shared::peer_split::RankDevice { device, stream })
            .collect(),
    };
    let (embedding, (model, mut shares)) = crate::shared::token_io::TokenEmbedding::load(&loaded.library,
        embed_source(&loaded.catalog, loaded.cfg.dim)?, args.token_io.embed_placement, || { let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights"); loader.model(&loaded.cfg) })?;
    tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "DeepSeek V4 coordinator weights resident");
    let mut max_context = crate::shared::context::checkpoint_context(
        &args.snapshot, &args.manifest, "deepseek_v4", args.max_context)?;
    let prefill_rows = caps["prefill_rows"].as_u64().context("prefill_rows")? as usize;
    let decode_rows = caps["decode_rows"].as_u64().context("decode_rows")? as usize;
    // One shared admission solve (`cuteafd_loader::placement`), the same
    // request `cuteafd plan --layout` resolves, over one measured sample per
    // GPU taken after weights and modules and before any cache or expert.
    let placement = {
        use cuteafd_loader::placement::Baseline;
        ensure!(args.max_sequences > 0, "--max-sequences must be positive");
        let devices: Vec<_> = std::iter::once(args.device).chain(split_device).collect();
        let gpus = devices.iter().map(|&device| crate::shared::peer_split::on_device(
            &loaded.library, device, args.device, || {
                let (free, total) = loaded.library.cuda_memory_info()?;
                Ok((total as u64, Baseline::Measured { free_bytes: free as u64 }))
            })).collect::<Result<Vec<_>>>()?;
        let cache_stages = model.dspark.as_ref().map_or(0, |d| d.stages.len());
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
            family: loaded.family, gpus, cache_stages, prefill_rows, decode_rows, max_context, prefix,
            expert_workspace, tp2_workspace, exchange_f32 };
        let request = admission::request(args, &inputs)?;
        let placement = cuteafd_loader::placement::solve(&request)
            .map_err(|error| anyhow::anyhow!("DeepSeek V4 admission: {error}"))?;
        check_modes(&placement, split_device.is_some())?;
        tracing::info!(pool_tokens = placement.pool_tokens, onboard = %onboard, onboard_layers = placement.onboard_layers,
            local_per_gpu = ?placement.expert_ranges, "DeepSeek V4 placement: {}", placement.summary());
        cuteafd_bench::context::set_resolved("rtx-expert-layers", &placement.onboard_layers.to_string());
        cuteafd_bench::context::set_resolved("rtx-expert-ranges", &placement.expert_ranges.iter()
            .map(|r| format!("{}..{}", r.first, r.first + r.layers)).collect::<Vec<_>>().join(","));
        cuteafd_bench::context::set_resolved("pool-tokens", &placement.pool_tokens.to_string());
        placement
    };
    max_context = crate::shared::context::pool_context("deepseek_v4", max_context, args.max_context == 0,
        usize::try_from(placement.pool_tokens)?, 256)?;
    let shape = pool::PoolShape::new(
        args.max_sequences,
        caps["prefill_rows"].as_u64().context("prefill_rows")? as usize,
        pool::PoolShape::units_for(usize::try_from(placement.pool_tokens)?, args.max_sequences),
    );
    let skip = args.skip_routed_experts;
    let peers = args.peers.split(',').filter(|p| !p.is_empty()).map(str::parse)
        .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
    if skip {
        tracing::warn!("--skip-routed-experts: MoE layers run the shared expert alone (outputs do not match the model)");
    } else {
        ensure!(matches!(peers.len(), 2 | 3 | 4 | 6), "DeepSeek V4 serves 2, 3, 4 or 6 Spark ranks, got {}", peers.len());
        loaded.library.v41_compact_reducer()?.require_rank_count(peers.len() as u32)?;
    }
    let mut engine = engine::Engine::new(engine::EngineParts {
        library: &loaded.library,
        programs: &programs,
        cfg: loaded.cfg.clone(),
        weights: model,
        family: loaded.family,
        decode_rows: caps["decode_rows"].as_u64().context("decode_rows")? as usize,
        prefill_rows: caps["prefill_rows"].as_u64().context("prefill_rows")? as usize,
        c128_width: usize::try_from(cuteafd_loader::serving_capacity::compiled_c128_width(&loaded.manifest, loaded.family)?)?,
        max_context,
        stream,
        sms: args.sms,
        shape,
        embedding,
        skip_routed: skip,
    })?;
    if let Some((device, stream)) = peer_stream {
        engine.attach_peer(device, stream, shares.pop().context("head-split shares")?, engine::PeerParts { shape, tp2: placement.tp2.as_ref().is_some_and(|t| t.layers > 0) })?;
    }
    let held = held(&engine)?;
    let _ = held; // Prefix bytes were reserved before placement.
    let started = Instant::now();
    let stages = if args.dspark { engine.weights.dspark.as_ref().map_or(0, |d| d.stages.len()) } else { 0 };
    let max_rows = engine.decode_rows.max(engine.prefill_rows);
    let range = &placement.expert_ranges[0];
    ensure!(split_device.is_none() || range.layers == 0,
        "TP1 backbone layers are forbidden under a head split");
    let local = if skip { None } else { local::LocalExperts::load_range(&loaded.library, &args.native_lib,
        &loaded.catalog, stages, range.first..range.first + range.layers,
        max_rows, usize::try_from(range.peak_bytes)?, stream)? };
    ensure!(local.as_ref().map_or(0, |l| l.layers()) == range.layers,
        "DeepSeek V4 TP1 allocation differs from admitted placement");
    engine.install_local(local);
    if let Some(tp2) = placement.tp2.as_ref().filter(|t| t.layers > 0 && !skip) {
        use crate::shared::experts::rtx::{native::NativeTp2, exl3::Exl3Tp2, RtxExpertLayer};
        use crate::shared::memory::device::Device;
        let devices = [Device { library: &loaded.library, id: args.device },
            Device { library: &loaded.library, id: split_device.context("TP2 needs a head split")? }];
        let layers = tp2.first..tp2.first + tp2.layers;
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
    tracing::info!(tp1 = ?placement.expert_ranges, tp2 = ?placement.tp2,
        elapsed_ms = started.elapsed().as_millis() as u64, "DeepSeek V4 expert layers resident on coordinator GPUs");
    // Implicit Spark worlds: TP4 executors 1..=4, TP2 5..=6, TP3 7..=9, TP6 27..=32.
    let executors = (0..peers.len())
        .map(|rank| cuteafd_transport::expert::v41_spark_executor_id(peers.len(), rank))
        .collect::<Result<Vec<u64>>>()?;
    // One transport (connection set) per prefill lane, so each lane can keep
    // a Spark wave in flight; decode uses the first.
    let lanes = if skip { 0 } else { engine::PREFILL_LANES };
    let mut transports = (0..lanes).map(|_| SparkLink::new(&loaded.library, &peers, &executors, 4096,
        TcpTransportConfig { timing: false, timeout: Duration::from_secs(120), max_frame_bytes: 64 << 20 },
        loaded.cfg.dim * 2))
        .collect::<Result<Vec<_>>>()?;
    crate::shared::memory_report::release_load_staging(&loaded.library);
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let started = Instant::now();
    for transport in &mut transports {
        engine.warm_transport(transport, &runtime)?;
    }
    tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "Spark expert transport warm");
    if crate::shared::spark_intake::device_exchange_enabled() && !skip
        && crate::shared::spark_intake::device_exchange_available(&loaded.library)? {
        let started = Instant::now();
        engine.attach_device_link(&peers, &executors,
            TcpTransportConfig { timing: false, timeout: Duration::from_secs(120), max_frame_bytes: 64 << 20 })?;
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "decode and verify waves use the device exchange");
    }
    if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
    engine.warm_local_graphs()?;
    engine.check_routes(&mut transports, &runtime)?;
    let result = body(&engine, &mut transports, &runtime);
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
    let started = Instant::now();
    let loaded = load(&args.engine)?;
    let cfg = loaded.cfg.clone();
    with_engine(&loaded, &args.engine, None, |_| Ok(0), |engine, transports, runtime| {
        let before = engine.graph_capture_counts();
        println!("golden ready: {:.3} s | graph captures {} | TP2 expert graph captures {}",
            started.elapsed().as_secs_f64(), before.0, before.1);
        let result = match args.token_check {
            Some(steps) => token_check(&args, &loaded, engine, transports, runtime, steps),
            None if !args.resume_at.is_empty() => resume(&args, engine, transports, runtime),
            None if args.nll => nll_run(&args, &cfg, engine, transports, runtime),
            None => golden_run(&args, &cfg, engine, transports, runtime),
        };
        let after = engine.graph_capture_counts();
        println!("golden post-ready captures: all {} | TP2 experts {}", after.0 - before.0, after.1 - before.1);
        for setting in cuteafd_bench::context::get().settings.into_iter().filter(|s| s.name.starts_with("route-")) {
            println!("golden {}: {}", setting.name, setting.value.unwrap_or_default());
        }
        result
    })
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
    let mut allocator = pool::PoolAllocator::new(engine.shape);
    let mut placement = allocator.admit(tokens.len())?;
    let chunk = args.chunk.unwrap_or(engine.prefill_rows).min(engine.prefill_rows);
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
    let mut allocator = pool::PoolAllocator::new(engine.shape);
    let mut placement = allocator.admit(tokens.len())?;
    let chunk = args.chunk.unwrap_or(engine.prefill_rows).min(engine.prefill_rows);
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
    let mut allocator = pool::PoolAllocator::new(engine.shape);
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
mod program_selection_tests {
    use cuteafd_core::coordinator_programs::CoordinatorPrograms;

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
                                        let suffix = template.replace("{cap}", cap).replace("{mode}", mode)
                                            .replace("{attention}", attention).replace("{ratio}", ratio)
                                            .replace("{program}", program);
                                        let name = format!("{family}_{suffix}");
                                        assert!(selected.contains(&name), "unselected dispatch {name}");
                                        assert!(required.contains(&name), "startup misses dispatch {name}");
                                        if split && (args.starts_with("_on(1, true,") || args.starts_with("_on(rank, split,")
                                            || args.starts_with("_on(0, weights.split,")) {
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
