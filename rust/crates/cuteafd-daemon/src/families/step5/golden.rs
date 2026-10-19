//! Target diagnostics. Cache publication is confined to the explicit prefix-state check.
use super::{
    cache::CacheRank,
    chain::TargetChain,
    head::{HeadPlan, TargetHeadOwner},
    prefix::{Frontier, Placement, Step5Prefix},
    transaction::{Staging, Step},
};
use crate::shared::{
    memory::device::{Allocation, Device},
    golden_scores::{select_scored_rows, scored_nll, scored_kl},
    placement::{spark_rings, SparkAdmission},
    spark_intake::SparkDeviceLink,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
use cuteafd_hostcache::copy::StubCopyEngine;
use cuteafd_loader::{
    plan::{checkpoint::Checkpoint, families::step5::Config},
    serving_capacity::step5::Step5CacheLayout,
};
use cuteafd_transport::{
    expert::{DeviceBuild, SparkWorld, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
    worker_selection::ModelAdmission,
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, net::SocketAddr, os::unix::fs::FileExt, path::PathBuf};

const H: usize = 4096;
const FP8_REVISION: &str = "364b3b03fed2477aa0953fe855a2c4c8c48ee3cc";
const BF16_REVISION: &str = "7734b7ffac118d70d8398bfa3eb0fab007eb45b1";

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[arg(long)]
    snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    native_lib: Option<PathBuf>,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    manifest: PathBuf,
    /// Pinned official reference export containing meta.json and windows/ID/layerNN.bin.
    #[arg(long)]
    golden: PathBuf,
    #[arg(long)]
    window: String,
    #[arg(long, value_delimiter = ',')]
    peers: Vec<SocketAddr>,
    /// Last target layer, inclusive; logits require the complete target through layer 91.
    #[arg(long, default_value_t = 87)]
    last_layer: usize,
    /// Cold prefix rows (at most 4096); omitted uses the complete reference window.
    #[arg(long)]
    prefill: Option<usize>,
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// Validate source, reference and routed-layer selection without CUDA.
    #[arg(long)]
    check_headers: bool,
    /// Score complete embedding-to-logits execution against the pinned BF16 reference.
    #[arg(long)]
    nll: bool,
    /// Isolate engine error with native FP8 arithmetic; BF16 remains the fidelity root.
    #[arg(long)]
    fp8_diagnostic_reference: bool,
    /// Compare retained outputs from the same cold target execution.
    #[arg(long, value_delimiter = ',', requires = "nll", conflicts_with = "prefix_state")]
    compare_layers: Vec<usize>,
    /// Compare the post-layer-91 BF16 final-norm plane to an explicit reference export.
    #[arg(long, requires = "nll", conflicts_with = "prefix_state")]
    compare_final_norm: bool,
    /// Compare real cold/shared-prefix canonical state and an identical continuation.
    #[arg(long, requires = "nll")]
    prefix_state: bool,
}

fn selection(last: usize, peers: usize, hosted: &[usize], nll: bool) -> Result<()> {
    ensure!((3..=91).contains(&last), "Step-5 golden supports target layers 3..91");
    ensure!([4, 6].contains(&peers), "Step-5 golden needs canonical TP4 or TP6 real experts");
    let final_routed = last.min(90);
    ensure!(hosted == (3..=final_routed).collect::<Vec<_>>(),
        "Step-5 golden hosted experts must exactly cover layers 3..{final_routed}");
    ensure!(!nll || last == 91, "NLL/KL requires complete target execution through dense layer 91");
    Ok(())
}

fn selected_layers() -> Result<Vec<usize>> {
    std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")
        .context("golden-run must select --expert-layers")?
        .split(',')
        .map(|s| s.parse().context("invalid hosted expert layer"))
        .collect()
}

fn reference_identity(meta: &Value, fp8_diagnostic: bool) -> Result<()> {
    let (checkpoint, revision) = if fp8_diagnostic {
        ("stepfun-ai/Step-5-Preview", FP8_REVISION)
    } else {
        ("stepfun-ai/Step-5-Preview-BF16", BF16_REVISION)
    };
    ensure!(
        meta["schema"] == "cuteafd.fidelity.golden/2"
            && meta["family"] == "step5"
            && meta["checkpoint"] == checkpoint
            && meta["snapshot_identity"]["snapshot_revision"] == revision
            && meta["mtp"] == false,
        "Step-5 golden requires the explicitly selected pinned official target reference"
    );
    if fp8_diagnostic {
        ensure!(meta["diagnostic_only"] == true
            && meta["fidelity_root"] == "stepfun-ai/Step-5-Preview-BF16"
            && meta["expert_arithmetic"] == "official act_quant + block-scaled fp8_gemm"
            && meta["diagnostic_bf16_set_sha256"].as_str().is_some_and(|s|
                s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())),
            "native FP8 diagnostic provenance missing or mismatched");
    } else {
        ensure!(meta["diagnostic_only"] != true, "diagnostic reference cannot replace fidelity root");
    }
    let source = &meta["source_pin"];
    for (field, expected) in [
        ("revision", revision),
        (
            "model_sha256",
            "330a166878910b21e37f65669033a84fcaada3b58d7903fc16ef55c496cb1ea5",
        ),
        (
            "kernel_sha256",
            "2cd01d57d2d135397d2fd36c203684af7564b5e6c6f5d3853b3d35bedeb548be",
        ),
        (
            "vision_encoder_sha256",
            "8943ee555ffd2a32213f6f9f0a7146d1f49566c65c8b7691c37055208cdb6b53",
        ),
    ] {
        ensure!(
            source[field] == expected,
            "Step-5 reference source pin differs: {field}"
        );
    }
    for field in ["tokenizer_sha256", "config_sha256"] {
        ensure!(
            meta["snapshot_identity"][field]
                .as_str()
                .is_some_and(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())),
            "missing reference identity: {field}"
        );
    }
    let proof = &meta["prefix_qualification"];
    ensure!(
        proof["schema"] == "cuteafd.fidelity.prefix/1"
            && proof["family"] == "step5"
            && proof["set_sha256"] == meta["set_sha256"]
            && meta["set_sha256"].is_string()
            && proof["snapshot_identity"] == meta["snapshot_identity"]
            && proof["passed"] == true
            && proof["finite"] == true
            && proof["rows"] == 512
            && proof["different_rows"] == serde_json::json!([])
            && proof["argmax_disagreements"] == 0
            && proof["lengths"] == serde_json::json!([576, 640])
            && proof["score_from"] == 64
            && proof["fixed_rows"] == 128,
        "Step-5 reference prefix-invariance proof missing or mismatched"
    );
    Ok(())
}

struct Reference {
    tokens: Vec<u32>,
    expected: Vec<f32>,
    positions: Vec<usize>,
    hidden: Vec<(usize, Vec<f32>)>,
    final_norm: Option<Vec<f32>>,
}

fn hidden_reference(folder: &std::path::Path, layer: usize, full_rows: usize, rows: usize) -> Result<Vec<f32>> {
    hidden_file(folder.join(format!("layer{layer:02}.bin")), full_rows, rows)
}

fn hidden_file(path: PathBuf, full_rows: usize, rows: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() == full_rows * H * 4, "reference layer shape must cover every input row");
    let values: Vec<_> = bytes[..rows * H * 4].chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite reference hidden payload");
    Ok(values)
}

fn hidden_metrics(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
    ensure!(!actual.is_empty() && actual.len() == expected.len()
        && actual.iter().chain(expected).all(|v| v.is_finite()), "hidden comparison shape/finite mismatch");
    let (mut error, mut norm, mut dot, mut ours) = (0., 0., 0., 0.);
    for (&a, &b) in actual.iter().zip(expected) {
        let (a, b) = (f64::from(a), f64::from(b));
        error += (a - b).powi(2);
        norm += b * b;
        dot += a * b;
        ours += a * a;
    }
    ensure!(norm > 0. && ours > 0., "zero hidden state norm");
    Ok((dot / (norm * ours).sqrt(), (error / norm).sqrt()))
}

fn reference(args: &GoldenArgs, vocab: usize) -> Result<Reference> {
    ensure!(
        !args.window.is_empty()
            && args
                .window
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c)),
        "invalid golden window id"
    );
    let meta: Value = serde_json::from_slice(&std::fs::read(args.golden.join("meta.json"))?)?;
    reference_identity(&meta, args.fp8_diagnostic_reference)?;
    let tokenizer = format!(
        "{:x}",
        Sha256::digest(std::fs::read(args.snapshot.join("tokenizer.json"))?)
    );
    ensure!(
        meta["snapshot_identity"]["tokenizer_sha256"] == tokenizer,
        "reference tokenizer differs from target"
    );
    let folder = args.golden.join("windows").join(&args.window);
    let entry = meta["windows"].as_array().context("golden windows")?
        .iter().find(|w| w["id"] == args.window && w["path"] == format!("windows/{}", args.window))
        .context("window missing from reference inventory")?;
    let bytes = std::fs::read(folder.join("tokens.bin"))?;
    ensure!(
        !bytes.is_empty() && bytes.len() % 4 == 0,
        "invalid golden token extent"
    );
    let mut tokens: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let rows = args.prefill.unwrap_or(tokens.len());
    ensure!(
        (1..=4096).contains(&rows) && rows <= tokens.len(),
        "cold Step-5 golden requires 1..4096 reference rows"
    );
    ensure!(tokens.iter().all(|&t| (t as usize) < vocab), "reference token outside vocabulary");
    ensure!(args.compare_layers.iter().all(|&layer| layer <= args.last_layer)
        && args.compare_layers.windows(2).all(|layers| layers[0] < layers[1])
        && (args.compare_layers.is_empty() || (args.nll && !args.prefix_state)),
        "hidden comparison layers must be increasing, executed, and use cold NLL mode");
    let hidden = args.compare_layers.iter().map(|&layer|
        Ok((layer, hidden_reference(&folder, layer, tokens.len(), rows)?)))
        .collect::<Result<Vec<_>>>()?;
    ensure!(!args.compare_final_norm || (args.nll && !args.prefix_state),
        "final norm comparison requires cold NLL mode");
    let final_norm = args.compare_final_norm.then(||
        hidden_file(folder.join("final_norm.bin"), tokens.len(), rows)).transpose()?;
    if args.fp8_diagnostic_reference && (args.compare_final_norm || args.compare_layers.iter().any(|l| (88..=90).contains(l))) {
        let tail: Value = serde_json::from_slice(&std::fs::read(args.golden.join("tail-diagnostic.json"))?)?;
        let source_sha = format!("{:x}", Sha256::digest(std::fs::read(args.golden.join("meta.json"))?));
        ensure!(tail["schema"] == "cuteafd.step5.tail-diagnostic/1" && tail["diagnostic_only"] == true
            && tail["source_meta_sha256"] == source_sha && tail["original_layer91_byte_exact"] == true,
            "tail continuation provenance missing or mismatched");
        let files = args.compare_layers.iter().filter(|&l| (88..=90).contains(l))
            .map(|l| format!("layer{l:02}.bin"))
            .chain(args.compare_final_norm.then_some("final_norm.bin".into()));
        for file in files {
            let sha = format!("{:x}", Sha256::digest(std::fs::read(folder.join(&file))?));
            ensure!(tail["files"][format!("windows/{}/{file}", args.window)] == sha,
                "tail diagnostic payload digest differs: {file}");
        }
    }
    let (expected, positions) = if args.nll {
        ensure!(entry["vocab"].as_u64() == Some(vocab as u64), "reference vocabulary mismatch");
        let positions = entry["positions"].as_array().context("missing scored positions")?.iter()
            .map(|p| usize::try_from(p.as_u64().context("invalid scored position")?).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!positions.is_empty() && positions.iter().all(|&p| p > 0 && p < tokens.len())
            && positions.windows(2).all(|p| p[0] < p[1]), "invalid scored positions");
        let bytes = std::fs::read(folder.join("logits.bin"))?;
        ensure!(bytes.len() == positions.len().checked_mul(vocab).and_then(|n| n.checked_mul(4))
            .context("reference logit extent overflow")?, "reference scored logit extent");
        let scored = positions.partition_point(|&p| p < rows);
        ensure!(scored > 0, "selected prefix has no scored reference positions");
        (bytes[..scored * vocab * 4].chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>(), positions[..scored].to_vec())
    } else {
        (hidden_reference(&folder, args.last_layer, tokens.len(), rows)?, vec![])
    };
    ensure!(expected.iter().all(|v| v.is_finite()), "nonfinite reference payload");
    tokens.truncate(rows);
    Ok(Reference { tokens, expected, positions, hidden, final_norm })
}

fn embedding_rows(checkpoint: &Checkpoint, tokens: &[u32], vocab: usize) -> Result<Vec<u8>> {
    let tensor = checkpoint
        .tensors
        .iter()
        .find(|t| t.meta.name == "model.embed_tokens.weight")
        .context("target embedding")?;
    ensure!(
        tensor.meta.dtype == cuteafd_core::DType::Bf16
            && tensor.meta.shape == [vocab, H]
            && tokens.iter().all(|&t| (t as usize) < vocab),
        "native embedding shape/dtype/token extent"
    );
    let file = std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?;
    let mut result = Vec::with_capacity(tokens.len() * H * 4);
    let mut row = vec![0; H * 2];
    // One cold golden input, not a serving embedding fallback or expert host hop.
    for &token in tokens {
        file.read_exact_at(
            &mut row,
            tensor.meta.byte_offset + u64::from(token) * (H * 2) as u64,
        )?;
        result.extend(row.chunks_exact(2).flat_map(|b| {
            (u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16).to_le_bytes()
        }));
    }
    Ok(result)
}

fn request(
    layer: usize,
    rows: usize,
    routes: Vec<ExpertProtocolV2RouteEntry>,
    wire: bytes::Bytes,
) -> Result<ExpertProtocolV2Request> {
    let mut request = ExpertProtocolV2Request::new_bytes(
        layer as u64 + 1,
        17,
        layer as u32,
        H as u32,
        ExpertV2Dtype::Bf16,
        (0..rows as u32)
            .map(|row| ExpertProtocolV2RowDescriptor {
                row_id: row as u64,
                source_kind: ExpertV2SourceKind::Prefill,
                source_request_id: 1,
                token_position: row as u64,
                route_offset: row * 8,
                route_count: 8,
            })
            .collect(),
        routes,
        wire,
    )?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

fn link<'a>(
    device: Device<'a>,
    args: &GoldenArgs,
    rows: usize,
    layers: &[usize],
) -> Result<SparkDeviceLink<'a>> {
    let world = SparkWorld::canonical(args.peers.len())?;
    ensure!(
        world.supports_device_exchange(),
        "Step-5 requires native device exchange"
    );
    let packages = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SPARK_PACKAGES")?);
    let metadata: Value = serde_json::from_slice(&std::fs::read(
        packages.join("fp8/fp8-step5/manifest.json"),
    )?)?;
    let layout = &metadata["layouts"][format!("tp{}", args.peers.len())];
    ensure!(
        layout["hidden"] == H
            && layout["experts"] == 352
            && layout["top_k"] == 8
            && layout["intermediate"] == 1536
            && layout["slice"] == 1536 / args.peers.len()
            && layout["tp"] == args.peers.len(),
        "sealed Step-5 FP8 TP geometry"
    );
    let compiled = layout["capacities"]
        .as_array()
        .context("FP8 capacities")?
        .iter()
        .map(|c| c["capacity"].as_u64().context("FP8 capacity integer"))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("empty FP8 capacities")?;
    ensure!(
        compiled >= rows as u64 && compiled <= 4096,
        "FP8 package row capacity"
    );
    let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    let config = TcpTransportConfig::default();
    SparkAdmission::plan(
        &args.snapshot,
        ExpertGeometry::STEP5,
        Some(&catalog),
        &cuteafd_loader::placement::SparkWorkingSet {
            rtx_layers: vec![],
            spark_layers: layers.to_vec(),
            spark_ranks: args.peers.len(),
            prefill_lanes: 1,
        },
        args.peers.len(),
        None,
        ModelAdmission {
            capacity: rows as u32,
            compiled_capacity: compiled as u32,
            endpoints: vec![SparkEndpoint {
                capacity: rows as u32,
                ingress: SparkIngress::Bf16,
            }],
            rings: spark_rings(config.max_frame_bytes)?,
        },
    )?
    .select(&args.peers)?;
    let warm = request(
        3,
        rows,
        (0..rows * 8)
            .map(|i| ExpertProtocolV2RouteEntry {
                row_index: (i / 8) as u32,
                expert_id: (i % 352) as u32,
                gate_weight: 0.,
            })
            .collect(),
        vec![0; rows * H * 2].into(),
    )?;
    let build: DeviceBuild = Box::new(|wave, routes, wire| {
        request(wave.layer as usize, wave.rows as usize, routes, wire)
    });
    SparkDeviceLink::new(
        device.library,
        device.id,
        world,
        &args.peers,
        rows,
        8,
        H * 2,
        H * 2,
        config,
        Some(warm),
        build,
    )
}

const PREFIX_FRONTIERS: [usize; 5] = [63, 64, 65, 575, 576];

fn prefix_state_bytes(layout: &Step5CacheLayout, rows: usize) -> Result<u64> {
    ensure!(rows > 576, "Step-5 prefix state check needs at least 577 tokens");
    let pages = (rows as u64).div_ceil(64).checked_mul(4)
        .and_then(|n| n.checked_add(4)).context("prefix page capacity overflow")?;
    layout.ranks[0].bytes.checked_mul(6)
        .and_then(|n| n.checked_add(layout.geometry.ranks[0].persistent_unit_bytes.checked_mul(pages)?))
        .and_then(|n| n.checked_add(Staging::bytes(layout, 0).ok()? as u64))
        .context("prefix state admission overflow")
}

fn read_buffer(device: Device<'_>, buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.bytes];
    device.run(|| device.library.copy_d2h(&mut bytes, buffer))?;
    Ok(bytes)
}

#[derive(Debug, PartialEq, Eq)]
struct CanonicalState {
    mark: Vec<u8>,
    pages: Vec<Vec<u8>>,
}

fn canonical_state(cache: &RefCell<CacheRank<'_>>, placement: &Placement) -> Result<CanonicalState> {
    let mut owner = cache.borrow_mut();
    // Slot 2 is reserved for inspection, outside the shared pool's two slots.
    owner.capture(placement.sequence, 2, placement.committed as u64)?;
    let device = owner.stream.device;
    let mark = read_buffer(device, owner.mark_view(2)?)?;
    let mut pages = Vec::new();
    for (logical, &physical) in placement.pages.iter().take(placement.committed.div_ceil(64)).enumerate() {
        let live = (placement.committed - logical * 64).min(64);
        for plane in owner.page_views(physical)? {
            let mut bytes = read_buffer(device, plane)?;
            bytes.truncate(bytes.len() / 64 * live);
            pages.push(bytes);
        }
    }
    Ok(CanonicalState { mark, pages })
}

#[allow(clippy::too_many_arguments)]
fn committed_step<'a>(
    checkpoint: &Checkpoint, vocab: usize, tokens: &[u32], chain: &mut TargetChain<'a>,
    cache: &RefCell<CacheRank<'a>>, staging: &Staging<'a>,
    pool: &mut PrefixCache<StubCopyEngine, Frontier>, placement: &mut Placement,
    input: &Allocation<'a>, link: &SparkDeviceLink<'a>, waves: usize,
) -> Result<Vec<u8>> {
    let device = cache.borrow().stream.device;
    device.run(|| device.library.copy_h2d(input.buffer, &embedding_rows(checkpoint, tokens, vocab)?))?;
    link.arm()?;
    link.expect(waves as u64);
    let mut step = Step::begin(vec![cache], vec![staging], pool, placement, tokens.len())?;
    chain.enqueue_step(&mut step, 0, input, Some(link), 0)?;
    cache.borrow().stream.drain()?;
    link.check()?;
    step.publish(tokens.len())?;
    let mut output = chain.output(91)?;
    output.bytes = tokens.len() * H * 4;
    let hidden = read_buffer(device, output)?;
    ensure!(hidden.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
        "nonfinite Step-5 prefix continuation");
    Ok(hidden)
}

#[allow(clippy::too_many_arguments)]
fn check_prefix_state<'a>(
    device: Device<'a>, programs: &'a cuteafd_ffi::programs::Programs<'a>,
    checkpoint: &Checkpoint, cfg: &Config, tokens: &[u32], chain: &mut TargetChain<'a>,
    input: &Allocation<'a>, link: &SparkDeviceLink<'a>, waves: usize, available: usize,
) -> Result<()> {
    let shape = Step5CacheLayout::new(cfg, 1, tokens.len() as u64, 0, tokens.len() as u64)?;
    prefix_state_bytes(&shape, tokens.len())?;
    let pages = tokens.len().div_ceil(64) * 4 + 4;
    for frontier in PREFIX_FRONTIERS {
        let cache = RefCell::new(CacheRank::new(device, shape.clone(), 0, 3, 3, pages as u64, available)?);
        let staging = Staging::new(device, programs, &cache.borrow(), available)?;
        let family = Step5Prefix::new(vec![&cache])?;
        let mut pool = PrefixCache::<StubCopyEngine, Frontier>::new(family.layout(),
            PrefixConfig { entries: 1, mark_slots: 2, keep_logits: false, min_tokens: 1 }, None)?;
        let mut source = pool.admit_cold(&family, frontier, frontier + 1,
            |pages| Placement { pages, sequence: 0, committed: 0 })?.placement;
        committed_step(checkpoint, cfg.vocab, &tokens[..frontier], chain, &cache, &staging,
            &mut pool, &mut source, input, link, waves)?;
        let expected = canonical_state(&cache, &source)?;
        ensure!(pool.capture(&family, SnapshotKind::Prompt, &tokens[..frontier], &source, After::default())?,
            "Step-5 diagnostic snapshot was not retained");
        let mut cold = pool.admit_cold(&family, frontier, frontier + 1,
            |pages| Placement { pages, sequence: 1, committed: 0 })?.placement;
        committed_step(checkpoint, cfg.vocab, &tokens[..frontier], chain, &cache, &staging,
            &mut pool, &mut cold, input, link, waves)?;
        ensure!(expected == canonical_state(&cache, &cold)?,
            "Step-5 independently cold state differs at {frontier}");
        let admitted = pool.admit(&family, &tokens[..frontier + 1], frontier + 1, false,
            |pages| Placement { pages, sequence: 2, committed: 0 })?;
        ensure!(admitted.resume == frontier && admitted.source.as_ref().is_some_and(|s|
            !s.partial && s.frontier == frontier && s.plan.exact()),
            "Step-5 expected deepest exact prefix hit at {frontier}");
        let mut restored = admitted.placement;
        ensure!(expected == canonical_state(&cache, &restored)?,
            "Step-5 restored canonical mark or live CSA rows differ at {frontier}");
        let cold_hidden = committed_step(checkpoint, cfg.vocab, &tokens[frontier..frontier + 1],
            chain, &cache, &staging, &mut pool, &mut cold, input, link, waves)?;
        let restored_hidden = committed_step(checkpoint, cfg.vocab, &tokens[frontier..frontier + 1],
            chain, &cache, &staging, &mut pool, &mut restored, input, link, waves)?;
        ensure!(cold_hidden == restored_hidden, "Step-5 continuation hidden bytes differ at {frontier}");
        ensure!(canonical_state(&cache, &cold)? == canonical_state(&cache, &restored)?,
            "Step-5 continuation canonical state differs at {frontier}");
        ensure!(expected == canonical_state(&cache, &source)?,
            "Step-5 continuation mutated retained source at {frontier}");
        println!("Step-5 exact-prefix diagnostic frontier={frontier} canonical_mark_bytes={} live_csa_planes={} continuation_rows=1 byte_exact=true (no serving/lossless qualification)", expected.mark.len(), expected.pages.len());
        for placement in [&source, &cold, &restored] { pool.release(&family, &placement.pages)?; }
        pool.clear(&family)?;
    }
    Ok(())
}

pub(crate) fn run(args: GoldenArgs) -> Result<()> {
    let layers = selected_layers()?;
    selection(args.last_layer, args.peers.len(), &layers, args.nll)?;
    ensure!(
        args.snapshot.file_name().and_then(|s| s.to_str()) == Some(FP8_REVISION),
        "Step-5 golden requires pinned native FP8 revision"
    );
    let checkpoint = Checkpoint::coordinator(&args.snapshot, false, false)?;
    let cfg = Config::parse(&checkpoint)?;
    let Reference { tokens, expected, positions, hidden, final_norm } = reference(&args, cfg.vocab)?;
    let reference_kind = if args.fp8_diagnostic_reference { "native-FP8 diagnostic" } else { "BF16-root diagnostic" };
    let rows = tokens.len();
    ensure!(!args.prefix_state || (args.nll && args.last_layer == 91),
        "Step-5 prefix state check requires the complete target");
    let layout = Step5CacheLayout::new(&cfg, 1, rows as u64, 0, rows as u64)?;
    if args.check_headers {
        println!("Step-5 headers/reference validated: rows={rows} layers=0..{} TP{}; no CUDA or numerical qualification", args.last_layer, args.peers.len());
        return Ok(());
    }
    cuteafd_core::set_expert_geometry(ExpertGeometry::STEP5)
        .map_err(|_| anyhow::anyhow!("Step-5 golden requires an isolated process"))?;
    // SAFETY: runner supplies the sealed library; every program and device owner drops before it.
    let library = unsafe {
        cuteafd_ffi::NativeLibrary::load(
            args.native_lib.as_ref().context("--native-lib required")?,
        )?
    };
    library.cuda_set_device(args.device)?;
    let device = Device {
        library: &library,
        id: args.device,
    };
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let selected = cuteafd_loader::placement::ProgramSet::from_names(&["step5"], programs.names());
    let pending = crate::shared::inventory::pending_code(
        &library,
        args.device,
        0,
        false,
        "step5",
        "fp8",
        &selected,
    )?;
    let (free, _) = library.cuda_memory_info()?;
    let available = free as u64;
    let pages = (rows as u64).div_ceil(64) + 1;
    let prefix_bytes = if args.prefix_state {
        prefix_state_bytes(&layout, rows)?
    } else { 0 };
    let cache_bytes = layout.ranks[0]
        .bytes
        .checked_add(layout.geometry.ranks[0].persistent_unit_bytes * pages)
        .and_then(|n| n.checked_add(prefix_bytes))
        .context("golden cache bytes")?;
    let staging_bytes = Staging::bytes(&layout, 0)? as u64;
    let input_bytes = (rows * H * 4) as u64;
    let chain_bytes =
        TargetChain::required_bytes(&programs, &checkpoint, args.last_layer + 1, rows)?;
    let head_plan = args.nll.then(|| HeadPlan::prepare(&programs, &checkpoint, rows)).transpose()?;
    let head_bytes = head_plan.as_ref().map_or(0, HeadPlan::bytes);
    // Reserve transport construction, transient library use and a separate 2 GiB headroom.
    let transport = 2u64 << 30;
    let reserve = pending
        .checked_add(transport)
        .and_then(|n| n.checked_add(2u64 << 30))
        .context("golden reserve")?;
    let total = [
        chain_bytes,
        head_bytes,
        cache_bytes,
        staging_bytes,
        input_bytes,
        reserve,
    ]
    .into_iter()
    .try_fold(0u64, |a, b| a.checked_add(b))
    .context("golden admission overflow")?;
    ensure!(total <= available, "Step-5 golden needs {total} bytes including weights/workspaces/cache/code/transport/headroom, available {available}");
    tracing::info!(
        rows,
        last_layer = args.last_layer,
        chain_bytes,
        head_bytes,
        cache_bytes,
        staging_bytes,
        reserve,
        total,
        available,
        "Step-5 cold prefix admission"
    );
    let started = std::time::Instant::now();
    let link = device.run(|| link(device, &args, rows, &layers))?;
    let (remaining, _) = library.cuda_memory_info()?;
    let owners = chain_bytes + head_bytes + cache_bytes + staging_bytes + input_bytes + pending + (2u64 << 30);
    ensure!(
        owners <= remaining as u64,
        "Step-5 golden live transport left insufficient admitted owner storage"
    );
    let cache = RefCell::new(CacheRank::new(device, layout, 0, 1, 0, pages, remaining)?);
    let staging = Staging::new(device, &programs, &cache.borrow(), remaining)?;
    let family = Step5Prefix::new(vec![&cache])?;
    let mut pool = PrefixCache::<StubCopyEngine, Frontier>::new(
        family.layout(),
        PrefixConfig {
            entries: 0,
            mark_slots: 0,
            keep_logits: false,
            min_tokens: 1,
        },
        None,
    )?;
    let mut placement = pool
        .admit_cold(&family, 1, 1, |pages| Placement {
            pages,
            sequence: 0,
            committed: 0,
        })?
        .placement;
    let mut chain = TargetChain::load(
        device,
        &programs,
        &checkpoint,
        args.last_layer + 1,
        rows,
        available,
        &[head_bytes, cache_bytes, staging_bytes, input_bytes, reserve],
    )?;
    let head = head_plan.map(|plan| TargetHeadOwner::load(device, &programs, &checkpoint, plan,
        available, &[chain_bytes, cache_bytes, staging_bytes, input_bytes, reserve])).transpose()?;
    let input = Allocation::new(device, rows * H * 4)?;
    library.copy_h2d(
        input.buffer,
        &embedding_rows(&checkpoint, &tokens, cfg.vocab)?,
    )?;
    tracing::info!(
        seconds = started.elapsed().as_secs_f64(),
        "Step-5 cold prefix readiness"
    );
    if args.prefix_state {
        return check_prefix_state(device, &programs, &checkpoint, &cfg, &tokens,
            &mut chain, &input, &link, layers.len(), remaining);
    }
    link.arm()?;
    link.expect(layers.len() as u64);
    let mut step = Step::begin(
        vec![&cache],
        vec![&staging],
        &mut pool,
        &mut placement,
        rows,
    )?;
    chain.enqueue_step(&mut step, 0, &input, Some(&link), 0)?;
    cache.borrow().stream.drain()?;
    link.check()?;
    drop(step);
    ensure!(
        placement.committed == 0,
        "cold qualification must not publish target state"
    );
    for (layer, expected) in hidden {
        let bytes = read_buffer(device, chain.output(layer)?)?;
        let actual: Vec<_> = bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let (cosine, relative_l2) = hidden_metrics(&actual, &expected)?;
        println!("Step-5 hidden ladder rows={rows} layer={layer} cosine={cosine:.9} relative_l2={relative_l2:.9} reference={reference_kind}");
    }
    if let Some(head) = head {
        let actual = head.logits(chain.output(91)?)?;
        if let Some(expected) = final_norm {
            let (cosine, relative_l2) = hidden_metrics(&head.normalized()?, &expected)?;
            println!("Step-5 hidden ladder rows={rows} layer=final_norm cosine={cosine:.9} relative_l2={relative_l2:.9} reference={reference_kind}");
        }
        let ours = select_scored_rows(&actual, &positions, cfg.vocab)?;
        let targets = positions.iter().map(|&p| tokens[p]).collect::<Vec<_>>();
        let nll = scored_nll(&ours, &targets, cfg.vocab)?;
        let reference_nll = scored_nll(&expected, &targets, cfg.vocab)?;
        let (kl, max_kl) = scored_kl(&expected, &ours, cfg.vocab)?;
        println!("Step-5 complete target rows={rows} scored={} engine_nll={nll:.12} reference_nll={reference_nll:.12} delta_nll={:.12} mean_kl={kl:.12} max_kl={max_kl:.12} (pinned {reference_kind}; no serving/lossless/cache qualification)", targets.len(), nll - reference_nll);
        return Ok(());
    }
    let mut bytes = vec![0; rows * H * 4];
    library.copy_d2h(&mut bytes, chain.output(args.last_layer)?)?;
    let actual: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        actual.iter().all(|v| v.is_finite()),
        "nonfinite Step-5 prefix result"
    );
    let (cosine, relative_l2) = hidden_metrics(&actual, &expected)?;
    println!("Step-5 prefix rows={rows} layer {}: cosine {cosine:.9} relative_l2 {relative_l2:.9} ({reference_kind} only; no NLL/KL/full-forward qualification)", args.last_layer);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_state_admits_three_sequences_three_marks_and_cow_pages() -> Result<()> {
        let layout = super::super::cache::tests::layout_at(640);
        assert_eq!(PREFIX_FRONTIERS, [63, 64, 65, 575, 576]);
        assert!(prefix_state_bytes(&layout, 576).is_err());
        assert_eq!(prefix_state_bytes(&layout, 640)?,
            layout.ranks[0].bytes * 6 + layout.geometry.ranks[0].persistent_unit_bytes * 44
                + Staging::bytes(&layout, 0)? as u64);
        Ok(())
    }
    #[test]
    fn prefix_scope_requires_exact_real_experts_and_refuses_logits_claims() {
        for world in [4, 6] {
            assert!(selection(87, world, &(3..=87).collect::<Vec<_>>(), false).is_ok());
        }
        for last in [0, 2, 92, usize::MAX] {
            assert!(selection(last, 6, &[], false).is_err());
        }
        for last in [88, 89, 90, 91] {
            assert!(selection(last, 6, &(3..=last.min(90)).collect::<Vec<_>>(), last == 91).is_ok());
        }
        assert!(selection(91, 6, &(3..=91).collect::<Vec<_>>(), true).is_err());
        assert!(selection(90, 6, &(3..=90).collect::<Vec<_>>(), true).is_err());
        assert!(selection(3, 6, &[3], false).is_ok());
        assert!(selection(3, 6, &[3, 4], false).is_err());
        assert!(selection(4, 6, &[4, 3], false).is_err());
        assert!(selection(3, 2, &[3], false).is_err());
        assert!(selection(3, 6, &[3], true).is_err());
    }
    fn reference_meta() -> Value {
        let identity = serde_json::json!({"snapshot_revision":BF16_REVISION,
            "tokenizer_sha256":format!("{:x}", Sha256::digest(b"tokenizer")), "config_sha256":"a".repeat(64)});
        serde_json::json!({"schema":"cuteafd.fidelity.golden/2", "family":"step5",
            "checkpoint":"stepfun-ai/Step-5-Preview-BF16", "snapshot_identity":identity,
            "mtp":false, "set_sha256":"b".repeat(64), "windows":[{"id":"code-1","path":"windows/code-1"}],
            "source_pin":{"revision":BF16_REVISION,
                "model_sha256":"330a166878910b21e37f65669033a84fcaada3b58d7903fc16ef55c496cb1ea5",
                "kernel_sha256":"2cd01d57d2d135397d2fd36c203684af7564b5e6c6f5d3853b3d35bedeb548be",
                "vision_encoder_sha256":"8943ee555ffd2a32213f6f9f0a7146d1f49566c65c8b7691c37055208cdb6b53"},
            "prefix_qualification":{"schema":"cuteafd.fidelity.prefix/1", "family":"step5",
                "set_sha256":"b".repeat(64), "snapshot_identity":identity, "passed":true, "finite":true,
                "rows":512, "different_rows":[], "argmax_disagreements":0,
                "lengths":[576,640], "score_from":64, "fixed_rows":128}})
    }
    #[test]
    fn reference_requires_pinned_source_and_prefix_proof() {
        let meta = reference_meta();
        reference_identity(&meta, false).unwrap();
        for pointer in [
            "/source_pin/model_sha256",
            "/snapshot_identity/config_sha256",
            "/prefix_qualification/passed",
            "/prefix_qualification/snapshot_identity/tokenizer_sha256",
            "/prefix_qualification/set_sha256",
            "/mtp",
        ] {
            let mut invalid = meta.clone();
            *invalid.pointer_mut(pointer).unwrap() = Value::Null;
            assert!(reference_identity(&invalid, false).is_err(), "{pointer}");
        }
    }
    #[test]
    fn native_fp8_reference_requires_explicit_diagnostic_provenance() {
        let mut meta = reference_meta();
        assert!(reference_identity(&meta, true).is_err());
        meta["checkpoint"] = "stepfun-ai/Step-5-Preview".into();
        meta["snapshot_identity"]["snapshot_revision"] = FP8_REVISION.into();
        meta["prefix_qualification"]["snapshot_identity"] = meta["snapshot_identity"].clone();
        meta["source_pin"]["revision"] = FP8_REVISION.into();
        meta["diagnostic_only"] = true.into();
        meta["fidelity_root"] = "stepfun-ai/Step-5-Preview-BF16".into();
        meta["expert_arithmetic"] = "official act_quant + block-scaled fp8_gemm".into();
        meta["diagnostic_bf16_set_sha256"] = "c".repeat(64).into();
        reference_identity(&meta, true).unwrap();
        assert!(reference_identity(&meta, false).is_err());
        for field in ["diagnostic_only", "fidelity_root", "expert_arithmetic", "diagnostic_bf16_set_sha256"] {
            let mut bad = meta.clone();
            bad[field] = Value::Null;
            assert!(reference_identity(&bad, true).is_err(), "{field}");
        }
        let mut bad = reference_meta();
        bad["diagnostic_only"] = true.into();
        assert!(reference_identity(&bad, false).is_err());
    }
    #[test]
    fn hidden_comparison_checks_geometry_finite_values_and_norms() -> Result<()> {
        assert_eq!(hidden_metrics(&[1., 2.], &[1., 2.])?, (1., 0.));
        assert_eq!(hidden_metrics(&[2., 4.], &[1., 2.])?, (1., 1.));
        assert!(hidden_metrics(&[], &[]).is_err());
        assert!(hidden_metrics(&[1.], &[1., 2.]).is_err());
        assert!(hidden_metrics(&[f32::NAN], &[1.]).is_err());
        assert!(hidden_metrics(&[1.], &[f32::INFINITY]).is_err());
        assert!(hidden_metrics(&[0.], &[1.]).is_err());
        assert!(hidden_metrics(&[1.], &[0.]).is_err());
        Ok(())
    }
    #[test]
    fn reference_reader_checks_extents_tokenizer_and_finite_rows() -> Result<()> {
        let root = tempfile::tempdir()?;
        let golden = root.path().join("golden");
        let folder = golden.join("windows/code-1");
        std::fs::create_dir_all(&folder)?;
        std::fs::write(root.path().join("tokenizer.json"), b"tokenizer")?;
        std::fs::write(
            golden.join("meta.json"),
            serde_json::to_vec(&reference_meta())?,
        )?;
        std::fs::write(
            folder.join("tokens.bin"),
            [1u32, 2]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let bytes = vec![1f32; 2 * H]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        std::fs::write(folder.join("layer87.bin"), &bytes)?;
        let mut args = GoldenArgs {
            snapshot: root.path().into(),
            native_lib: None,
            manifest: PathBuf::new(),
            golden,
            window: "code-1".into(),
            peers: vec![],
            last_layer: 87,
            prefill: Some(1),
            device: 0,
            check_headers: true,
            nll: false,
            prefix_state: false,
            fp8_diagnostic_reference: false,
            compare_layers: vec![],
            compare_final_norm: false,
        };
        let Reference { tokens, expected: hidden, .. } = reference(&args, 3)?;
        assert_eq!(tokens, [1]);
        assert_eq!(hidden, vec![1f32; H]);
        args.prefill = Some(3);
        assert!(reference(&args, 3).is_err());
        args.prefill = Some(1);
        args.window = "../code-1".into();
        assert!(reference(&args, 3).is_err());
        args.window = "code-1".into();
        std::fs::write(folder.join("layer87.bin"), &bytes[..bytes.len() - 4])?;
        assert!(reference(&args, 3).is_err());
        let mut nonfinite = bytes.clone();
        nonfinite[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(folder.join("layer87.bin"), nonfinite)?;
        assert!(reference(&args, 3).is_err());
        std::fs::write(folder.join("layer87.bin"), bytes)?;
        std::fs::write(root.path().join("tokenizer.json"), b"different")?;
        assert!(reference(&args, 3).is_err());
        Ok(())
    }
    #[test]
    fn full_target_scores_reference_positions_without_a_layer91_hidden_fixture() -> Result<()> {
        let root = tempfile::tempdir()?;
        let golden = root.path().join("golden");
        let folder = golden.join("windows/code-1");
        std::fs::create_dir_all(&folder)?;
        std::fs::write(root.path().join("tokenizer.json"), b"tokenizer")?;
        let mut meta = reference_meta();
        meta["windows"][0]["positions"] = serde_json::json!([1, 3]);
        meta["windows"][0]["vocab"] = 3.into();
        let write_meta = |meta: &Value| -> Result<()> {
            Ok(std::fs::write(golden.join("meta.json"), serde_json::to_vec(meta)?)?)
        };
        write_meta(&meta)?;
        std::fs::write(folder.join("tokens.bin"), [0u32, 1, 0, 2].into_iter()
            .flat_map(u32::to_le_bytes).collect::<Vec<_>>())?;
        let logits = [0f32, 5., 0., 0., 0., 5.].into_iter()
            .flat_map(f32::to_le_bytes).collect::<Vec<_>>();
        std::fs::write(folder.join("logits.bin"), &logits)?;
        let mut args = GoldenArgs { snapshot: root.path().into(), native_lib: None,
            manifest: PathBuf::new(), golden: golden.clone(), window: "code-1".into(),
            peers: vec![], last_layer: 91, prefill: None, device: 0, check_headers: true, nll: true, prefix_state: false,
            fp8_diagnostic_reference: false, compare_layers: vec![], compare_final_norm: false };
        let full = reference(&args, 3)?;
        assert_eq!(full.positions, [1, 3]);
        let targets = full.positions.iter().map(|&p| full.tokens[p]).collect::<Vec<_>>();
        assert_eq!(targets, [1, 2]);
        assert!(scored_nll(&full.expected, &targets, 3)? < 0.02);
        std::fs::write(folder.join("layer03.bin"), vec![1f32; 4 * H].into_iter()
            .flat_map(f32::to_le_bytes).collect::<Vec<_>>())?;
        args.compare_layers = vec![3];
        let ladder = reference(&args, 3)?;
        assert_eq!(ladder.hidden[0], (3, vec![1.; 4 * H]));
        for invalid in [vec![3, 3], vec![4, 3], vec![92], vec![10]] {
            args.compare_layers = invalid;
            assert!(reference(&args, 3).is_err());
        }
        args.compare_layers = vec![3];
        args.prefill = Some(3);
        assert_eq!(reference(&args, 3)?.hidden[0].1.len(), 3 * H);
        args.compare_layers.clear();
        args.prefill = Some(3);
        let partial = reference(&args, 3)?;
        assert_eq!(partial.positions, [1]);
        assert_eq!(partial.expected, [0., 5., 0.]);
        args.prefill = Some(1);
        assert!(reference(&args, 3).is_err());
        args.prefill = None;
        for invalid in [serde_json::json!([0, 3]), serde_json::json!([3, 1]),
            serde_json::json!([1, 1]), serde_json::json!([1, 4]), serde_json::json!([])] {
            let mut bad = meta.clone();
            bad["windows"][0]["positions"] = invalid;
            write_meta(&bad)?;
            assert!(reference(&args, 3).is_err());
        }
        write_meta(&meta)?;
        assert!(reference(&args, 4).is_err());
        std::fs::write(folder.join("logits.bin"), &logits[..logits.len() - 4])?;
        assert!(reference(&args, 3).is_err());
        let mut bad = logits;
        bad[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(folder.join("logits.bin"), bad)?;
        assert!(reference(&args, 3).is_err());
        Ok(())
    }

    #[test]
    fn golden_subcommand_parses_runner_owned_flags() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from([
            "cuteafd",
            "step5-golden",
            "--snapshot",
            "/snapshot",
            "--golden",
            "/reference",
            "--window",
            "code-1",
            "--peers",
            "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4,127.0.0.1:5,127.0.0.1:6",
            "--native-lib",
            "/lib.so",
            "--manifest",
            "/PROGRAMS.json",
            "--last-layer",
            "87",
            "--prefill",
            "64",
            "--check-headers",
        ])
        .unwrap();
        let crate::cli::Commands::Step5Golden(args) = cli.command else {
            panic!("Step-5 golden");
        };
        assert_eq!(args.last_layer, 87);
        assert_eq!(args.peers.len(), 6);
    }
}
