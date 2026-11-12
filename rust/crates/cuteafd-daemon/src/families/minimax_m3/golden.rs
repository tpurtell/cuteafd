//! Cold full-model qualification only. No serving entrypoint or reference bridge.
use super::{admission, cache::Cache, head::Head, layer::{Kernels, UnitOwner}, prefill::Prefill,
    weights::TargetWeights};
use crate::shared::{inventory::{InventoryPhase, RuntimeInventory},
    memory::{device::{Allocation, Device, Stream}, receipts::Inventory},
    placement::{spark_rings, SparkAdmission}, prefill_pipeline::sequence_units,
    spark_intake::SparkLink};
use anyhow::{ensure, Context, Result};
use cuteafd_bench::{fidelity_dataset, fidelity_rows, reference::{Fidelity, Window}};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress, SparkReplyFormat,
    memory_layout::{Category, Tag}};
use cuteafd_loader::{families::minimax_m3::{MiniMaxM3Config, execution::*, weight_layout::CoordinatorWeights},
    placement::{PreparedSparkLink, SparkEndpointOwners, SparkIntakeCandidates, SparkWorkingSet,
        planner::{CoordinatorDevices, Lifetime}}, plan::{Checkpoint, checkpoint::CheckpointView}};
use cuteafd_transport::{ExpertV2Dtype, SparkSessionShape, TcpTransportConfig,
    expert::{SparkWorld, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16}, worker_selection::ModelAdmission};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, net::SocketAddr, path::{Path, PathBuf}, rc::Rc, sync::Arc};

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)] pub(crate) attention: crate::cli::CoreViewArgs,
    #[arg(long)] pub(crate) snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] manifest: PathBuf,
    /// Immutable MiniMax top1024-plus-tail reference publication.
    #[arg(long)] golden: PathBuf,
    /// New directory for shared scoring probes and fidelity.json.
    #[arg(long)] save_logits: PathBuf,
    #[arg(long)] window: Option<String>,
    #[arg(long, value_delimiter = ',', num_args = 1..)] peers: Vec<SocketAddr>,
    #[arg(long, default_value_t = 0)] device: i32,
    #[arg(long, default_value_t = 1024)] prefill_rows: u32,
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=4))] prefill_waves: u8,
    /// Validate checkpoint/reference/export headers without starting CUDA or peers.
    #[arg(long)] check_headers: bool,
    /// Full-model scoring is the only supported MiniMax golden operation.
    #[arg(long, required = true)] nll: bool,
}

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn reference(root: &Path, id: Option<&str>, vocab: usize) -> Result<Vec<Window>> {
    let bytes = std::fs::read(root.join("manifest.json"))?;
    ensure!(digest(&bytes) == fidelity_dataset::MINIMAX_M3_MANIFEST_SHA256,
        "MiniMax reference manifest differs from immutable publication");
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == fidelity_dataset::MINIMAX_M3_CONFIG
        && manifest["vocab"].as_u64() == Some(vocab as u64), "MiniMax reference config/vocabulary differs");
    let panel = fidelity_dataset::load(root, &manifest)?;
    let mut windows = fidelity_dataset::quick_windows(&panel)?;
    if let Some(id) = id {
        windows.retain(|window| window.id == id);
        ensure!(windows.len() == 1, "MiniMax window must belong to the sealed quick subset");
    }
    ensure!(!windows.is_empty() && windows.iter().all(|window| window.media.is_empty()
        && (2..=16384).contains(&window.tokens.len()) && window.score_from > 0
        && window.score_from < window.tokens.len()), "MiniMax requires bounded text-only causal windows");
    Ok(windows)
}

fn selected_programs(cfg: &MiniMaxM3Config, weights: &CoordinatorWeights,
    manifest: &serde_json::Value, rows: u32, tokens: u32) -> Result<BTreeSet<String>> {
    let (_, msa) = MsaPrograms::resolve(cfg, manifest, 1, rows, tokens)?;
    let (_, producers) = ProducerPrograms::resolve(manifest, rows)?;
    let (_, forward) = ForwardPrograms::resolve(manifest, rows)?;
    let dense = weights.resolve_dense_programs(manifest, rows)?;
    Ok(msa.iter().chain(&producers).chain(&forward).chain(&dense)
        .map(|spec| spec.name.clone()).collect())
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || run(args)).await?
}

fn run(args: GoldenArgs) -> Result<()> {
    ensure!(args.nll && args.prefill_rows >= 256 && args.prefill_rows <= 4096
        && args.prefill_rows % 64 == 0, "MiniMax needs --nll and page-aligned prefill rows 256..=4096");
    let view = args.attention.selected_view.clone().map(Ok)
        .unwrap_or_else(|| args.attention.select(&args.snapshot))?;
    args.attention.validate_view(&view)?;
    let checkpoint = Checkpoint::coordinator_view(&view, false, false)?;
    let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
    cfg.release_geometry()?;
    let windows = reference(&args.golden, args.window.as_deref(), cfg.vocab_size)?;
    let tokens = u32::try_from(windows.iter().map(|window| window.tokens.len() - 1).max().unwrap())?;
    let weights = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &vec![0; cfg.layers])?
        .into_iter().next().context("MiniMax target layout missing")?;
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
    let names = selected_programs(&cfg, &weights, &manifest, args.prefill_rows, tokens)?;
    println!("MiniMax native full golden: reference={} windows={} scored_rows={} programs={}",
        fidelity_dataset::MINIMAX_M3_MANIFEST_SHA256, windows.len(),
        windows.iter().map(|window| window.positions.len()).sum::<usize>(), names.len());
    if args.check_headers { return Ok(()); }
    cuteafd_core::set_expert_geometry(ExpertGeometry::MINIMAX_M3)
        .map_err(|_| anyhow::anyhow!("MiniMax golden needs an isolated process"))?;
    ensure!(args.peers.len() == 4, "MiniMax golden currently qualifies four real Spark ranks");
    ensure!(!args.save_logits.try_exists()? && args.save_logits.parent().is_some_and(Path::is_dir),
        "MiniMax logits output needs an existing parent and a new directory");
    let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    // SAFETY: every program, allocation and Spark terminal owner remains within
    // the library lifetime; terminal ownership retains an Arc during failures.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&args.native_lib)? });
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let runtime = RuntimeInventory::measure(&library, &[args.device], InventoryPhase::PreWeight,
        |_| { programs.load_matching(|name| names.contains(name))?; Ok(()) })?;
    let sample = &runtime.gpus[0];
    let device = Device { library: &library, id: args.device };
    let config = TcpTransportConfig::default();
    let endpoint = SparkEndpoint { ring_depth: None, capacity: args.prefill_rows,
        ingress: SparkIngress::Bf16, partial_reply_bytes: None };
    let rings = spark_rings(config.max_frame_bytes)?;
    let waves = usize::from(args.prefill_waves);
    let link_layout = PreparedSparkLink::endpoint("spark", 0,
        SparkEndpointOwners { ranks: args.peers.len(), geometry: ExpertGeometry::MINIMAX_M3, endpoint, rings },
        waves, true, true, SparkIntakeCandidates::Probed, SparkReplyFormat::Bf16, Lifetime::TARGET)?;
    let contract = admission::contract(&checkpoint, &cfg, &weights, &manifest,
        args.prefill_rows, tokens, waves, &link_layout, &sample.arch)?;
    admission::admit(&contract, sample.free_bytes, sample.total_bytes, host)?;
    let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
    let exports = cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?.exports();
    let executions = cuteafd_loader::plan::spark_executions::admit_family("minimax_m3", &exports,
        None, u64::from(args.prefill_rows), waves as u64)?;
    let model = ModelAdmission::for_clients(ExpertGeometry::MINIMAX_M3, endpoint, rings, 1, executions)?;
    let pending = SparkAdmission::plan_view(&view, ExpertGeometry::MINIMAX_M3, Some(&catalog),
        &SparkWorkingSet { rtx_layers: vec![], spark_layers: (3..cfg.layers).collect(),
            spark_ranks: args.peers.len(), prefill_lanes: 1 }, args.peers.len(), model, None)?.start(&args.peers)?;
    let target = TargetWeights::mount(device, &checkpoint, &weights)?;
    let kernels = Kernels::bind(device, &programs, &manifest, &cfg, &weights, args.prefill_rows, tokens)?;
    let stream = Rc::new(Stream::new(device)?);
    let mut owners = (0..waves).map(|_| UnitOwner::new(&kernels, Rc::clone(&stream), &cfg, &weights,
        args.prefill_rows)).collect::<Result<Vec<_>>>()?;
    let cache = Cache::new(Rc::clone(&stream), &cfg, tokens as usize)?;
    let mut metadata = (0..waves).map(|_| cache.metadata(args.prefill_rows)).collect::<Result<Vec<_>>>()?;
    let head = Head::new(Rc::clone(&stream), &programs, &cfg, args.prefill_rows as usize)?;
    let residual = Allocation::new(device, tokens as usize * 6144 * 4, Tag::workspace("minimax/residual"))?;
    let devices = [args.device];
    let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "minimax",
        Lifetime::TARGET, Category::Workspace, "cold golden");
    target.inventory(&mut inventory, "target")?;
    cache.inventory(&mut inventory, "cache")?;
    head.inventory(&mut inventory, "head")?;
    residual.inventory(&mut inventory, "residual")?;
    for (slot, owner) in owners.iter().enumerate() { owner.inventory(&mut inventory, &format!("unit.{slot}"))?; }
    for (slot, owner) in metadata.iter().enumerate() { owner.inventory(&mut inventory, &format!("metadata.{slot}"))?; }
    let receipts = inventory.receipts.iter().map(|receipt| serde_json::json!({
        "allocation":receipt.allocation, "address":receipt.address })).collect::<Vec<_>>();
    tracing::info!(receipts = %serde_json::to_string(&receipts)?, "MiniMax retained owner inventory");
    pending.wait()?;
    let mut link = device.run(|| SparkLink::prepared(&library, SparkWorld::canonical(args.peers.len())?,
        &args.peers, config, link_layout))?;
    link.enable_terminal_ownership(Arc::clone(&library))?;
    // The link is declared after all compute owners. Every error, including
    // bootstrap, goes through this terminal block before their destructors run.
    let result = (|| {
        let mut shape = SparkSessionShape::new(args.prefill_rows, 6144, 4, ExpertV2Dtype::Bf16)?;
        shape.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        device.run(|| link.bootstrap(&shape))?;
        score(&args, &view, &cfg, &weights, &target, &kernels, &cache, &head, &residual,
            &mut owners, &mut metadata, &mut link)
    })();
    let terminal = link.terminal_quiesce().and_then(|_| link.terminal_drain_copies())
        .and_then(|_| stream.drain()).and_then(|_| link.terminal_release());
    if let Err(error) = terminal {
        tracing::error!(%error, "MiniMax terminal drainage failed; refusing to retire queued compute storage");
        std::process::abort();
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn score<'a>(args: &GoldenArgs, view: &CheckpointView, cfg: &MiniMaxM3Config,
    admitted: &CoordinatorWeights, weights: &TargetWeights<'a>, kernels: &Kernels<'a>,
    cache: &Cache<'a>, head: &Head<'a>, residual: &Allocation<'a>, owners: &mut [UnitOwner<'a>],
    metadata: &mut [super::cache::Metadata<'a>], link: &mut SparkLink<'a>) -> Result<()> {
    use cuteafd_api::openai::probe::{Probe, ProbeSpec};
    let windows = reference(&args.golden, args.window.as_deref(), cfg.vocab_size)?;
    std::fs::create_dir(&args.save_logits)?;
    let provenance = serde_json::json!({"family":"minimax_m3", "snapshot":view.snapshot(),
        "reference_manifest_sha256":fidelity_dataset::MINIMAX_M3_MANIFEST_SHA256,
        "native_lib":args.native_lib,"manifest_sha256":digest(&std::fs::read(&args.manifest)?),
        "prefill_rows":args.prefill_rows,"prefill_waves":args.prefill_waves,
        "expert_ingress":"BF16; native calibrated NVFP4 W4A4 experts", "shared_prefill":true});
    let mut records = Vec::new();
    let mut next_identity = 1;
    for window in windows {
        let count = window.tokens.len() - 1;
        let dump = args.save_logits.join(&window.id);
        let probe = Probe::new(ProbeSpec { cold: true, no_speculation: true,
            prompt_ids: Some(window.tokens.clone()), score_from: Some(window.score_from),
            score_positions: Some(window.positions.iter().map(|position| position.pos).collect()),
            top_k: 32, want: window.want(), dump_rows: Some(dump.clone()), ..Default::default() });
        probe.admitted("minimax_m3", &window.tokens, 0);
        probe.selected_score_path("prefill");
        probe.provenance(provenance.clone());
        for first in (0..count).step_by(args.prefill_rows as usize) {
            let end = (first + args.prefill_rows as usize).min(count);
            let span = crate::shared::prefix::view(residual.buffer, first * 6144 * 4, (end - first) * 6144 * 4)?;
            head.embed(weights, &window.tokens[first..end], span)?;
        }
        let units = sequence_units(count, args.prefill_rows as usize, owners.len(), cfg.layers)?;
        Prefill { cfg, admitted, weights, kernels, cache, owners, metadata, waves: link.exchange()?,
            residual: residual.buffer, placement_version: 17, next_identity: &mut next_identity }.run(&units)?;
        for chunk in units {
            let first = chunk.rows.start;
            let rows = chunk.rows.len();
            let span = crate::shared::prefix::view(residual.buffer, first * 6144 * 4, rows * 6144 * 4)?;
            let logits = head.project(weights, span, rows)?;
            for (row, logits) in logits.chunks_exact(cfg.vocab_size).enumerate() {
                let position = first + row + 1;
                if position >= window.score_from { probe.row(position, logits); }
            }
            probe.scored_prefill_chunk(first + 1, rows);
        }
        let record = probe.record();
        ensure!(record.error.is_none(), "MiniMax row dump failed: {:?}", record.error);
        let mut scored = window.score(&record.rows);
        ensure!(scored.missing == 0 && scored.non_finite == 0 && scored.records.len() == window.positions.len(),
            "MiniMax full forward did not cover all finite scored positions");
        fidelity_rows::score_compact(cfg.vocab_size, &window, &dump, &mut scored)?;
        println!("MiniMax {}: rows={} top1={:.6} NLL={:.8} KL(top1024+tail)={:.8}",
            window.id, scored.positions, scored.top1, scored.nll, scored.kl);
        serde_json::to_writer(std::fs::OpenOptions::new().write(true).create_new(true)
            .open(args.save_logits.join(format!("{}.json", window.id)))?,
            &serde_json::json!({"probe":record,"score":scored}))?;
        records.extend(scored.records);
    }
    let score = Fidelity::from_records(records);
    serde_json::to_writer(std::fs::OpenOptions::new().write(true).create_new(true)
        .open(args.save_logits.join("fidelity.json"))?, &serde_json::json!({
            "schema":"cuteafd.minimax.golden/1","kl_kind":"top1024+tail", "provenance":provenance,
            "score":score,"qualified":score.kl <= 0.03 && score.top1 >= 0.95}))?;
    ensure!(score.kl <= 0.03 && score.top1 >= 0.95,
        "MiniMax qualification failed: KL={} (<=0.03), top1={} (>=0.95)", score.kl, score.top1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_rejects_an_unpinned_manifest_before_loading_any_rows() -> Result<()> {
        let root = tempfile::tempdir()?;
        std::fs::write(root.path().join("manifest.json"), b"{}")?;
        assert!(reference(root.path(), None, 200064).unwrap_err().to_string()
            .contains("immutable publication"));
        Ok(())
    }
    #[test]
    fn golden_uses_shared_estimator_and_terminal_release_before_retirement() {
        let source = include_str!("golden.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("set_expert_geometry(ExpertGeometry::MINIMAX_M3)"));
        assert!(source.find("set_expert_geometry(").unwrap() < source.find("NativeLibrary::load(").unwrap());
        assert!(source.contains("fidelity_rows::score_compact("));
        assert!(source.contains("sequence_units("));
        assert!(source.contains("pending.wait()?"));
        assert!(source.contains("terminal_quiesce().and_then(|_| link.terminal_drain_copies())"));
        assert!(source.contains("stream.drain()).and_then(|_| link.terminal_release())"));
    }
}
