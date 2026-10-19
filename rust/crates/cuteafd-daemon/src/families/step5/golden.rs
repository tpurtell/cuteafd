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
    /// Shared native pipeline qualification with exactly one explicitly admitted lane.
    #[arg(long, conflicts_with = "prefix_state")]
    split_device: Option<i32>,
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
    /// Save every input row's F32 logits and U32 tokens in the shared golden format.
    #[arg(long, requires = "nll", conflicts_with = "prefix_state")]
    save_logits: Option<PathBuf>,
    /// Save selected post-layer F32 rows and the final BF16 norm expanded to F32.
    #[arg(long, requires_all = ["nll", "dump_layers"], conflicts_with = "prefix_state")]
    dump_residuals: Option<PathBuf>,
    /// Increasing target layer IDs to dump; independent of reference hidden fixtures.
    #[arg(long, value_delimiter = ',', requires = "dump_residuals")]
    dump_layers: Vec<usize>,
    /// Compare real cold/shared-prefix canonical state and an identical continuation.
    #[arg(long, requires = "nll")]
    prefix_state: bool,
    /// Worker-free layer-zero boundary diagnostic, without scoring or state publication.
    #[arg(long, conflicts_with_all = ["nll", "prefix_state", "save_logits", "dump_residuals", "compare_final_norm"])]
    dump_layer0: Option<PathBuf>,
}

fn diagnostic_selection(args: &GoldenArgs) -> Result<()> {
    ensure!(args.dump_layer0.is_none() || (args.last_layer == 0 && !args.nll
        && !args.prefix_state && args.peers.is_empty() && args.compare_layers.is_empty()
        && args.save_logits.is_none() && args.dump_residuals.is_none() && !args.compare_final_norm),
        "layer-zero dump requires --last-layer 0, no peers, scoring, or full-target diagnostics");
    ensure!(args.last_layer != 0 || args.dump_layer0.is_some(), "layer zero requires --dump-layer0");
    ensure!(args.split_device.is_none() || args.dump_layer0.is_some() || (args.nll && args.last_layer == 91),
        "shared execution requires full NLL or layer-zero diagnostic");
    let writes = args.save_logits.is_some() || args.dump_residuals.is_some();
    ensure!(!writes || (args.nll && args.last_layer == 91 && !args.prefix_state),
        "Step-5 diagnostic outputs require complete cold NLL mode");
    ensure!(args.dump_residuals.is_some() != args.dump_layers.is_empty()
        && args.dump_layers.iter().all(|&layer| layer <= args.last_layer)
        && args.dump_layers.windows(2).all(|layers| layers[0] < layers[1]),
        "residual dump layers must be nonempty, increasing, executed, and paired with --dump-residuals");
    Ok(())
}

fn write_f32(path: PathBuf, values: &[f32], rows: usize, width: usize) -> Result<()> {
    ensure!(rows > 0 && width > 0 && rows.checked_mul(width) == Some(values.len())
        && values.iter().all(|v| v.is_finite()), "diagnostic F32 shape/finite mismatch");
    std::fs::write(path, values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn save_diagnostics(
    args: &GoldenArgs, tokens: &[u32], positions: &[usize], vocab: usize,
    logits: &[f32], normalized: Option<&[f32]>,
    mut residual: impl FnMut(usize) -> Result<Vec<f32>>,
) -> Result<()> {
    diagnostic_selection(args)?;
    if args.save_logits.is_none() && args.dump_residuals.is_none() { return Ok(()); }
    ensure!(!tokens.is_empty() && tokens.iter().all(|&t| (t as usize) < vocab)
        && !positions.is_empty() && positions.iter().all(|&p| p > 0 && p < tokens.len())
        && positions.windows(2).all(|p| p[0] < p[1]), "diagnostic token/scored positions mismatch");
    let token_bytes = tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    if let Some(dir) = &args.save_logits {
        std::fs::create_dir_all(dir)?;
        write_f32(dir.join("logits.bin"), logits, tokens.len(), vocab)?;
        std::fs::write(dir.join("tokens.bin"), &token_bytes)?;
    }
    if let Some(dir) = &args.dump_residuals {
        let normalized = normalized.context("residual dump requires final normalized head input")?;
        std::fs::create_dir_all(dir)?;
        // Metadata is published last, so a failed dump is never marked complete.
        let meta = dir.join("meta.json");
        if meta.exists() { std::fs::remove_file(&meta)?; }
        for &layer in &args.dump_layers {
            write_f32(dir.join(format!("layer{layer:02}.bin")), &residual(layer)?, tokens.len(), H)?;
        }
        write_f32(dir.join("final_norm.bin"), normalized, tokens.len(), H)?;
        std::fs::write(dir.join("tokens.bin"), token_bytes)?;
        std::fs::write(dir.join("positions.bin"), (0..tokens.len() as u32)
            .flat_map(u32::to_le_bytes).collect::<Vec<_>>())?;
        std::fs::write(meta, serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "cuteafd.step5.residuals/1", "window": args.window,
            "rows": tokens.len(), "width": H, "dtype": "F32", "byte_order": "little",
            "positions": "positions.bin", "position_dtype": "U32", "tokens": "tokens.bin",
            "layers": args.dump_layers, "layer_pattern": "layerNN.bin",
            "final_norm": "final_norm.bin", "final_norm_source_dtype": "BF16",
            "scored_target_positions": positions, "logits_row_target_offset": 1,
            "complete": true
        }))?)?;
    }
    Ok(())
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
    let (expected, positions) = if args.dump_layer0.is_some() {
        (vec![], vec![])
    } else if args.nll {
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

pub(super) fn request(
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

fn expert_admission(args: &GoldenArgs, rows: usize, layers: &[usize], clients: usize) -> Result<(SparkWorld, TcpTransportConfig)> {
    let world = SparkWorld::canonical(args.peers.len())?;
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
        ModelAdmission::for_clients(ExpertGeometry::STEP5,
            SparkEndpoint { capacity: rows as u32, ingress: SparkIngress::Bf16 },
            compiled as u32, spark_rings(config.max_frame_bytes)?, clients)?,
    )?
    .select(&args.peers)?;
    cuteafd_transport::worker_selection::require_peer_clients(&args.peers,
        SparkEndpoint { capacity: rows as u32, ingress: SparkIngress::Bf16 }, clients)?;
    Ok((world, config))
}

fn link<'a>(device: Device<'a>, args: &GoldenArgs, rows: usize, layers: &[usize]) -> Result<SparkDeviceLink<'a>> {
    let (world, config) = expert_admission(args, rows, layers, 1)?;
    ensure!(world.supports_device_exchange(), "Step-5 diagnostic requires native device exchange");
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

pub(super) fn split_ownership(cfg: &Config, rows: usize) -> Result<cuteafd_loader::plan::families::step5::ownership::ExecutionOwnership> {
    use cuteafd_loader::placement::*;
    let modes = (0..cfg.layers).map(|layer| if cfg.full[layer] {
        LayerMode::Whole { gpu: ((layer / 4) % 2) as u8, ffn: FfnMode::Owner }
    } else { LayerMode::HeadSplit }).collect::<Vec<_>>();
    let spec = HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: (H * 4) as u64,
        rows: rows as u64, lanes: 1, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner };
    let placement = cuteafd_loader::placement::Placement { pool_tokens: 0, pool_limits: PoolLimits(vec![]),
        attention_placement: AttentionPlacement::Layers, attention_by_kind: vec![], peer_row_bytes: 0,
        onboard_layers: 0, layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::Spark }).collect(),
        movables: vec![], expert_ranges: vec![], residual: residual_homes(&modes, &spec)?,
        hops: plan_hops(&modes, &spec)?, tp2: None, items: vec![] };
    cuteafd_loader::plan::families::step5::ownership::ExecutionOwnership::new(cfg, 2, LayerExecutionPlan::new(&placement, spec)?)
}

fn shared_diagnostic_verdict(nll: f64, reference_nll: f64, kl: f64, max_kl: f64) -> Result<()> {
    let delta = nll - reference_nll;
    ensure!([nll, reference_nll, delta, kl, max_kl].into_iter().all(f64::is_finite),
        "Step-5 shared diagnostic produced nonfinite scores");
    ensure!(delta <= 0.01 && kl <= 0.03,
        "Step-5 shared diagnostic exceeds stop bar: delta_nll={delta:.12} (<=0.01), mean_kl={kl:.12} (<=0.03)");
    Ok(())
}

pub(super) fn dump_plane(directory: &std::path::Path, name: &str, device: Device<'_>,
    buffer: cuteafd_ffi::CuteafdDeviceBuffer, rows: usize, bf16: bool) -> Result<()> {
    let bytes = rows.checked_mul(H).and_then(|n| n.checked_mul(if bf16 { 2 } else { 4 }))
        .context("layer-zero dump extent overflow")?;
    ensure!(buffer.bytes >= bytes, "layer-zero dump buffer extent");
    let raw = read_buffer(device, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..buffer })?;
    let values = if bf16 {
        raw.chunks_exact(2).map(|v| f32::from_bits((u16::from_le_bytes(v.try_into().unwrap()) as u32) << 16)).collect::<Vec<_>>()
    } else {
        raw.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>()
    };
    write_f32(directory.join(format!("{name}.bin")), &values, rows, H)
}

fn dump_layer0_tail(directory: &std::path::Path, prefix: &str, device: Device<'_>,
    attention: &super::swa::SwaLayer<'_>, ffn: &super::ffn::DenseFfn<'_>, rows: usize) -> Result<()> {
    for (name, (buffer, bf16)) in ["attention_output", "post_attention_residual"].into_iter()
        .zip(attention.diagnostic_attention()).chain(
            ["ffn_input", "ffn_normalized_input", "ffn_delta", "post_layer_residual"].into_iter().zip(ffn.diagnostic_planes())) {
        dump_plane(directory, &format!("{prefix}{name}"), device, buffer, rows, bf16)?;
    }
    Ok(())
}

fn layer0_run(args: &GoldenArgs, checkpoint: &Checkpoint, cfg: &Config, reference: Reference) -> Result<()> {
    use super::execution::Execution;
    let directory = args.dump_layer0.as_ref().context("layer-zero dump directory")?;
    let rows = reference.tokens.len();
    if args.check_headers { println!("Step-5 layer-zero diagnostic headers validated: rows={rows}; no scoring or experts"); return Ok(()); }
    ensure!(args.device >= 0 && args.split_device.is_none_or(|peer| peer >= 0 && peer != args.device),
        "layer-zero device selection");
    std::fs::create_dir_all(directory)?;
    let metadata = directory.join("meta.json");
    if metadata.exists() { std::fs::remove_file(&metadata)?; }
    // SAFETY: sealed runner library outlives every program, stream and allocation.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load(args.native_lib.as_ref().context("--native-lib required")?)? };
    library.cuda_set_device(args.device)?;
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let pages = (rows as u64).div_ceil(64) + 1;
    let config = PrefixConfig { entries: 0, mark_slots: 0, keep_logits: false, min_tokens: 1 };
    let shared = args.split_device.is_some();
    if let Some(peer) = args.split_device {
        let devices = [args.device, peer].map(|id| Device { library: &library, id });
        let ownership = split_ownership(cfg, rows)?;
        let layout = Step5CacheLayout::for_execution(cfg, &ownership, rows as u64, 0, rows as u64)?;
        let mut available = [0; 2];
        let mut reservations = [0; 2];
        let mut cache_bytes = [0; 2];
        let mut stage_bytes = [0; 2];
        for rank in 0..2 {
            available[rank] = devices[rank].run(|| library.cuda_memory_info())?.0 as u64;
            cache_bytes[rank] = layout.ranks[rank].bytes + layout.geometry.ranks[rank].persistent_unit_bytes * pages;
            stage_bytes[rank] = Staging::bytes(&layout, rank)? as u64;
            let hops = cuteafd_loader::placement::hop_buffer_bytes(ownership.execution().hops(), &ownership.execution().spec(), 2)
                .context("layer-zero hop admission")?;
            reservations[rank] = cache_bytes[rank] + stage_bytes[rank] + hops[rank]
                + (9 * rows * H * 4) as u64 + (4u64 << 30);
            ensure!(Execution::required_bytes_count(&programs, checkpoint, &ownership, rank, rows, 1)?
                + reservations[rank] <= available[rank], "layer-zero rank allocation admission");
        }
        let caches = [RefCell::new(CacheRank::new(devices[0], layout.clone(), 0, 1, 0, pages, cache_bytes[0] as usize)?),
            RefCell::new(CacheRank::new(devices[1], layout, 1, 1, 0, pages, cache_bytes[1] as usize)?)];
        let stages = [Staging::new(devices[0], &programs, &caches[0].borrow(), stage_bytes[0] as usize)?,
            Staging::new(devices[1], &programs, &caches[1].borrow(), stage_bytes[1] as usize)?];
        let family = Step5Prefix::new(vec![&caches[0], &caches[1]])?;
        let mut pool = PrefixCache::<StubCopyEngine, Frontier>::new(family.layout(), config, None)?;
        let mut placement = pool.admit_cold(&family, 1, 1, |pages| Placement { pages, sequence: 0, committed: 0 })?.placement;
        let execution = Execution::load_count(&programs, checkpoint, ownership, [&caches[0], &caches[1]],
            [&stages[0], &stages[1]], rows, available, reservations, [None, None], 1)?;
        let input = Allocation::new(devices[0], rows * H * 4)?;
        devices[0].run(|| library.copy_h2d(input.buffer, &embedding_rows(checkpoint, &reference.tokens, cfg.vocab)?))?;
        let mut step = Step::begin(vec![&caches[0], &caches[1]], vec![&stages[0], &stages[1]], &mut pool, &mut placement, rows)?;
        execution.enqueue_layer0(&mut step, &input, directory)?;
        drop(step);
        ensure!(placement.committed == 0, "layer-zero diagnostic must not publish state");
        for rank in 0..2 {
            let (attention, ffn) = execution.diagnostic_layer0(rank)?;
            dump_layer0_tail(directory, &format!("rank{rank}-"), devices[rank], attention, ffn, rows)?;
        }
    } else {
        let device = Device { library: &library, id: args.device };
        let available = library.cuda_memory_info()?.0 as u64;
        let layout = Step5CacheLayout::new(cfg, 1, rows as u64, 0, rows as u64)?;
        let cache_bytes = layout.ranks[0].bytes + layout.geometry.ranks[0].persistent_unit_bytes * pages;
        let staging_bytes = Staging::bytes(&layout, 0)? as u64;
        let reserved = cache_bytes + staging_bytes + (rows * H * 4) as u64 + (4u64 << 30);
        ensure!(TargetChain::required_bytes(&programs, checkpoint, 1, rows)? + reserved <= available,
            "layer-zero old-path allocation admission");
        let cache = RefCell::new(CacheRank::new(device, layout, 0, 1, 0, pages, cache_bytes as usize)?);
        let staging = Staging::new(device, &programs, &cache.borrow(), staging_bytes as usize)?;
        let family = Step5Prefix::new(vec![&cache])?;
        let mut pool = PrefixCache::<StubCopyEngine, Frontier>::new(family.layout(), config, None)?;
        let mut placement = pool.admit_cold(&family, 1, 1, |pages| Placement { pages, sequence: 0, committed: 0 })?.placement;
        let mut chain = TargetChain::load(device, &programs, checkpoint, 1, rows, available, &[reserved])?;
        let input = Allocation::new(device, rows * H * 4)?;
        library.copy_h2d(input.buffer, &embedding_rows(checkpoint, &reference.tokens, cfg.vocab)?)?;
        let mut step = Step::begin(vec![&cache], vec![&staging], &mut pool, &mut placement, rows)?;
        chain.enqueue_step(&mut step, 0, &input, None, 0)?;
        cache.borrow().stream.drain()?;
        drop(step);
        ensure!(placement.committed == 0, "layer-zero diagnostic must not publish state");
        let attention = chain.diagnostic_layer0()?;
        dump_plane(directory, "attention_input", device, attention.diagnostic_norm(), rows, true)?;
        dump_layer0_tail(directory, "", device, attention, attention.diagnostic_dense()?, rows)?;
    }
    std::fs::write(directory.join("positions.bin"), (0..rows as u32).flat_map(u32::to_le_bytes).collect::<Vec<_>>())?;
    std::fs::write(directory.join("tokens.bin"), reference.tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    std::fs::write(metadata, serde_json::to_vec_pretty(&serde_json::json!({
        "schema": "cuteafd.step5.layer0/1", "rows": rows, "width": H, "dtype": "F32",
        "byte_order": "little", "positions": "positions.bin", "position_dtype": "U32",
        "tokens": "tokens.bin", "shared": shared, "window": args.window, "complete": true,
        "bf16_planes_expanded_losslessly": true, "attention_output": "rounded O projection delta",
        "ffn_delta": "rounded dense down projection before residual addition", "diagnostic_only": true
    }))?)?;
    println!("Step-5 layer-zero boundary diagnostic rows={rows} shared={shared}; no scoring or serving qualification");
    Ok(())
}

fn split_run(args: &GoldenArgs, checkpoint: &Checkpoint, cfg: &Config,
    reference: Reference, layers: &[usize]) -> Result<()> {
    use super::execution::Execution;
    use crate::shared::spark_intake::SparkLane;
    use std::sync::Arc;
    let peer_device = args.split_device.context("Step-5 split device")?;
    ensure!(args.device >= 0 && peer_device >= 0 && args.device != peer_device
        && args.nll && args.last_layer == 91 && !args.prefix_state,
        "Step-5 shared golden requires distinct devices and complete one-lane target");
    let rows = reference.tokens.len();
    let ownership = split_ownership(cfg, rows)?;
    let layout = Step5CacheLayout::for_execution(cfg, &ownership, rows as u64, 0, rows as u64)?;
    if args.check_headers { println!("Step-5 shared headers/reference validated: rows={rows} ranks=2 lanes=1; no CUDA or numerical qualification"); return Ok(()); }
    cuteafd_core::set_expert_geometry(ExpertGeometry::STEP5)
        .map_err(|_| anyhow::anyhow!("Step-5 golden requires an isolated process"))?;
    // SAFETY: sealed runner library outlives every allocation, transport and program.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(args.native_lib.as_ref().context("--native-lib required")?)? });
    library.cuda_set_device(args.device)?;
    let devices = [args.device, peer_device].map(|id| Device { library: &library, id });
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let selected = cuteafd_loader::placement::ProgramSet::from_names(&["step5", "step52"], programs.names());
    let head_plan = HeadPlan::for_execution(&programs, checkpoint, rows, &ownership)?;
    let head_rank = ownership.execution().spec().head_gpu as usize;
    let pages = (rows as u64).div_ceil(64) + 1;
    let mut available = [0u64; 2];
    let mut chain_bytes = [0u64; 2];
    let mut cache_bytes = [0u64; 2];
    let mut staging_bytes = [0u64; 2];
    let mut reservations = [0u64; 2];
    for rank in 0..2 {
        available[rank] = devices[rank].run(|| library.cuda_memory_info()).map(|(free, _)| free as u64)?;
        chain_bytes[rank] = Execution::required_bytes(&programs, checkpoint, &ownership, rank, rows)?;
        cache_bytes[rank] = layout.ranks[rank].bytes.checked_add(layout.geometry.ranks[rank].persistent_unit_bytes * pages)
            .context("Step-5 shared cache overflow")?;
        staging_bytes[rank] = Staging::bytes(&layout, rank)? as u64;
        let pending = crate::shared::inventory::pending_code(&library, devices[rank].id, rank, true, "step5", "fp8", &selected)?;
        let hops = cuteafd_loader::placement::hop_buffer_bytes(ownership.execution().hops(), &ownership.execution().spec(), 2)
            .context("Step-5 shared hop charge")?;
        // Eight peer receive slots, hop buffers/control, intake construction,
        // native code and independent headroom are charged before allocations.
        let peer = (8 * rows * H * 4) as u64 + hops[rank] + (2 << 20);
        let input = if rank == 0 { (rows * H * 4) as u64 } else { 0 };
        let head = if rank == head_rank { head_plan.bytes() } else { 0 };
        reservations[rank] = [cache_bytes[rank], staging_bytes[rank], peer, input, head, pending,
            2u64 << 30, 2u64 << 30].into_iter().try_fold(0u64, |n, bytes| n.checked_add(bytes))
            .context("Step-5 shared reservation overflow")?;
        let total = chain_bytes[rank].checked_add(reservations[rank]).context("Step-5 shared admission overflow")?;
        ensure!(total <= available[rank], "Step-5 shared rank {rank} needs {total} bytes, available {}", available[rank]);
        tracing::info!(rank, rows, lanes=1, chain_bytes=chain_bytes[rank], cache_bytes=cache_bytes[rank],
            staging_bytes=staging_bytes[rank], peer, pending, total, available=available[rank], "Step-5 shared golden admission");
    }
    let started = std::time::Instant::now();
    let caches = [RefCell::new(CacheRank::new(devices[0], layout.clone(), 0, 1, 0, pages, cache_bytes[0] as usize)?),
        RefCell::new(CacheRank::new(devices[1], layout, 1, 1, 0, pages, cache_bytes[1] as usize)?)];
    let stages = [Staging::new(devices[0], &programs, &caches[0].borrow(), staging_bytes[0] as usize)?,
        Staging::new(devices[1], &programs, &caches[1].borrow(), staging_bytes[1] as usize)?];
    let mut lanes = [None, None];
    let wave_ranks = ownership.routed_wave_ranks()?;
    let (world, config) = expert_admission(args, rows, layers, wave_ranks.len())?;
    for rank in wave_ranks {
        lanes[rank] = Some(devices[rank].own(|| SparkLane::new(&library, world, args.peers.clone(), rows as u32, config.clone(), H * 2))?);
    }
    let family = Step5Prefix::new(vec![&caches[0], &caches[1]])?;
    let mut pool = PrefixCache::<StubCopyEngine, Frontier>::new(family.layout(), PrefixConfig {
        entries: 0, mark_slots: 0, keep_logits: false, min_tokens: 1 }, None)?;
    let mut placement = pool.admit_cold(&family, 1, 1, |pages| Placement { pages, sequence: 0, committed: 0 })?.placement;
    let execution = Execution::load(&programs, checkpoint, ownership, [&caches[0], &caches[1]],
        [&stages[0], &stages[1]], rows, available, reservations, lanes)?;
    let head = TargetHeadOwner::load(devices[head_rank], &programs, checkpoint, head_plan,
        available[head_rank], &[])?;
    let input = Allocation::new(devices[0], rows * H * 4)?;
    devices[0].run(|| library.copy_h2d(input.buffer, &embedding_rows(checkpoint, &reference.tokens, cfg.vocab)?))?;
    tracing::info!(seconds=started.elapsed().as_secs_f64(), "Step-5 shared one-lane readiness");
    let mut step = Step::begin(vec![&caches[0], &caches[1]], vec![&stages[0], &stages[1]], &mut pool, &mut placement, rows)?;
    execution.enqueue(&mut step, &input, request)?;
    drop(step);
    ensure!(placement.committed == 0, "Step-5 shared cold golden must not publish state");
    for (layer, expected) in reference.hidden {
        let (device, output) = execution.output(layer)?;
        let bytes = read_buffer(device, output)?;
        let actual = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let (cosine, relative_l2) = hidden_metrics(&actual, &expected)?;
        println!("Step-5 shared hidden rows={rows} layer={layer} cosine={cosine:.9} relative_l2={relative_l2:.9}");
    }
    let (_, output) = execution.output(91)?;
    ensure!(output.device_id == devices[head_rank].id, "Step-5 shared final head owner");
    let actual = head.logits(output)?;
    let normalized = (args.dump_residuals.is_some() || reference.final_norm.is_some())
        .then(|| head.normalized()).transpose()?;
    save_diagnostics(args, &reference.tokens, &reference.positions, cfg.vocab,
        &actual, normalized.as_deref(), |layer| {
            let (device, output) = execution.output(layer)?;
            Ok(read_buffer(device, output)?.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
        })?;
    if let Some(expected) = reference.final_norm {
        let (cosine, relative_l2) = hidden_metrics(normalized.as_deref().context("final norm inspection")?, &expected)?;
        println!("Step-5 shared hidden rows={rows} layer=final_norm cosine={cosine:.9} relative_l2={relative_l2:.9}");
    }
    let ours = select_scored_rows(&actual, &reference.positions, cfg.vocab)?;
    let targets = reference.positions.iter().map(|&p| reference.tokens[p]).collect::<Vec<_>>();
    let nll = scored_nll(&ours, &targets, cfg.vocab)?;
    let reference_nll = scored_nll(&reference.expected, &targets, cfg.vocab)?;
    let (kl, max_kl) = scored_kl(&reference.expected, &ours, cfg.vocab)?;
    println!("Step-5 shared complete target rows={rows} lanes=1 scored={} engine_nll={nll:.12} reference_nll={reference_nll:.12} delta_nll={:.12} mean_kl={kl:.12} max_kl={max_kl:.12} (no serving/lossless/cache qualification)", targets.len(), nll-reference_nll);
    // Engine-only stop bars do not measure quantization error against the BF16 root.
    if args.fp8_diagnostic_reference {
        shared_diagnostic_verdict(nll, reference_nll, kl, max_kl)?;
    }
    Ok(())
}

pub(crate) fn run(args: GoldenArgs) -> Result<()> {
    diagnostic_selection(&args)?;
    let layers = if args.dump_layer0.is_some() { vec![] } else { selected_layers()? };
    if args.dump_layer0.is_none() { selection(args.last_layer, args.peers.len(), &layers, args.nll)?; }
    ensure!(
        args.snapshot.file_name().and_then(|s| s.to_str()) == Some(FP8_REVISION),
        "Step-5 golden requires pinned native FP8 revision"
    );
    let checkpoint = Checkpoint::coordinator(&args.snapshot, false, false)?;
    let cfg = Config::parse(&checkpoint)?;
    let reference = reference(&args, cfg.vocab)?;
    if args.dump_layer0.is_some() { return layer0_run(&args, &checkpoint, &cfg, reference); }
    if args.split_device.is_some() { return split_run(&args, &checkpoint, &cfg, reference, &layers); }
    let Reference { tokens, expected, positions, hidden, final_norm } = reference;
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
        let normalized = (args.dump_residuals.is_some() || final_norm.is_some())
            .then(|| head.normalized()).transpose()?;
        save_diagnostics(&args, &tokens, &positions, cfg.vocab, &actual,
            normalized.as_deref(), |layer| {
                Ok(read_buffer(device, chain.output(layer)?)?.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
            })?;
        if let Some(expected) = final_norm {
            let (cosine, relative_l2) = hidden_metrics(normalized.as_deref().context("final norm inspection")?, &expected)?;
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
    fn diagnostic_args(flags: &[&str]) -> Result<GoldenArgs> {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from([
            "cuteafd", "step5-golden", "--snapshot", "/snapshot", "--golden", "/reference",
            "--window", "code-1", "--last-layer", "91",
        ].into_iter().chain(flags.iter().copied()))?;
        let crate::cli::Commands::Step5Golden(args) = cli.command else { unreachable!() };
        Ok(args)
    }

    #[test]
    fn diagnostic_flags_require_complete_cold_execution_and_ordered_layers() -> Result<()> {
        for split in [vec![], vec!["--split-device", "1"]] {
            let mut flags = vec!["--nll", "--save-logits", "/logits", "--dump-residuals", "/residuals",
                "--dump-layers", "0,3,10,30,60,87,91"];
            flags.extend(split);
            let args = diagnostic_args(&flags)?;
            diagnostic_selection(&args)?;
            assert_eq!(args.save_logits.as_deref(), Some(std::path::Path::new("/logits")));
            assert_eq!(args.dump_layers, [0, 3, 10, 30, 60, 87, 91]);
            assert!(args.compare_layers.is_empty());
        }
        diagnostic_selection(&diagnostic_args(&[])?)?;
        diagnostic_selection(&diagnostic_args(&["--nll", "--save-logits", "/logits"])?)?;
        for flags in [
            vec!["--save-logits", "/logits"],
            vec!["--nll", "--prefix-state", "--save-logits", "/logits"],
            vec!["--nll", "--dump-residuals", "/residuals"],
            vec!["--nll", "--dump-layers", "0"],
            vec!["--nll", "--prefix-state", "--dump-residuals", "/residuals", "--dump-layers", "0"],
        ] { assert!(diagnostic_args(&flags).is_err(), "{flags:?}"); }
        for layers in ["0,0", "3,0", "92", "18446744073709551615"] {
            let args = diagnostic_args(&["--nll", "--dump-residuals", "/residuals", "--dump-layers", layers])?;
            assert!(diagnostic_selection(&args).is_err(), "{layers}");
        }
        let mut args = diagnostic_args(&["--nll", "--save-logits", "/logits"])?;
        args.last_layer = 87;
        assert!(diagnostic_selection(&args).is_err());
        args.last_layer = 91;
        args.prefix_state = true;
        assert!(diagnostic_selection(&args).is_err());
        Ok(())
    }

    #[test]
    fn layer_zero_diagnostic_flags_refuse_scoring_peers_and_other_prefixes() -> Result<()> {
        for split in [None, Some(1)] {
            let mut args = diagnostic_args(&["--dump-layer0", "/boundary"])?;
            args.last_layer = 0;
            args.split_device = split;
            diagnostic_selection(&args)?;
            args.last_layer = 1;
            assert!(diagnostic_selection(&args).is_err());
            args.last_layer = 0;
            args.peers.push("127.0.0.1:1".parse()?);
            assert!(diagnostic_selection(&args).is_err());
            args.peers.clear();
            args.nll = true;
            assert!(diagnostic_selection(&args).is_err());
        }
        assert!(diagnostic_args(&["--dump-layer0", "/boundary", "--nll"]).is_err());
        let mut args = diagnostic_args(&[])?;
        args.last_layer = 0;
        assert!(diagnostic_selection(&args).is_err());
        args.last_layer = 91;
        args.split_device = Some(1);
        assert!(diagnostic_selection(&args).is_err());
        Ok(())
    }

    #[test]
    fn diagnostic_files_preserve_full_rows_bits_positions_and_head_input() -> Result<()> {
        let root = tempfile::tempdir()?;
        let mut args = diagnostic_args(&["--nll", "--dump-residuals", "/unused", "--dump-layers", "0,91"])?;
        let logits_dir = root.path().join("logits");
        let residual_dir = root.path().join("residuals");
        args.save_logits = Some(logits_dir.clone());
        args.dump_residuals = Some(residual_dir.clone());
        let tokens = [2u32, 0, 1];
        let positions = [1, 2];
        let logits = [-0.0f32, 1., -2., 3., 4., 5., 6., 7., 8.];
        let norm = vec![0.5f32; tokens.len() * H];
        let mut requested = vec![];
        save_diagnostics(&args, &tokens, &positions, 3, &logits, Some(&norm), |layer| {
            requested.push(layer);
            Ok(vec![layer as f32; tokens.len() * H])
        })?;
        assert_eq!(requested, [0, 91]);
        assert_eq!(std::fs::read(logits_dir.join("tokens.bin"))?,
            tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>());
        assert_eq!(std::fs::read(logits_dir.join("logits.bin"))?,
            logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        assert_eq!(std::fs::read(residual_dir.join("positions.bin"))?,
            [0u32, 1, 2].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>());
        for (file, value) in [("layer00.bin", 0f32), ("layer91.bin", 91.), ("final_norm.bin", 0.5)] {
            assert_eq!(std::fs::read(residual_dir.join(file))?,
                vec![value; tokens.len() * H].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        }
        let meta: Value = serde_json::from_slice(&std::fs::read(residual_dir.join("meta.json"))?)?;
        assert_eq!(meta, serde_json::json!({
            "schema":"cuteafd.step5.residuals/1", "window":"code-1", "rows":3, "width":H,
            "dtype":"F32", "byte_order":"little", "positions":"positions.bin", "position_dtype":"U32",
            "tokens":"tokens.bin", "layers":[0,91], "layer_pattern":"layerNN.bin",
            "final_norm":"final_norm.bin", "final_norm_source_dtype":"BF16",
            "scored_target_positions":[1,2], "logits_row_target_offset":1, "complete":true
        }));
        let scored = select_scored_rows(&logits, &positions, 3)?;
        assert_eq!(scored, logits[..6]);
        // Outputs survive the numerical rejection that follows writing in the shared path.
        assert!(shared_diagnostic_verdict(0.02, 0., 0.01, 0.1).is_err());
        assert!(logits_dir.join("logits.bin").is_file());
        assert!(residual_dir.join("meta.json").is_file());
        Ok(())
    }

    #[test]
    fn diagnostic_writer_rejects_bad_payloads_without_complete_residual_metadata() -> Result<()> {
        let root = tempfile::tempdir()?;
        let mut args = diagnostic_args(&["--nll", "--dump-residuals", "/unused", "--dump-layers", "0"])?;
        args.dump_residuals = Some(root.path().into());
        let norm = vec![1f32; 2 * H];
        for bad in [vec![1.; H], vec![f32::NAN; 2 * H], vec![f32::INFINITY; 2 * H]] {
            std::fs::write(root.path().join("meta.json"), b"stale")?;
            assert!(save_diagnostics(&args, &[0, 1], &[1], 2, &[0.; 4], Some(&norm),
                |_| Ok(bad.clone())).is_err());
            assert!(!root.path().join("meta.json").exists());
        }
        assert!(save_diagnostics(&args, &[0, 1], &[1], 2, &[0.; 4], None,
            |_| Ok(norm.clone())).is_err());
        for positions in [vec![], vec![0], vec![2], vec![1, 1]] {
            assert!(save_diagnostics(&args, &[0, 1], &positions, 2, &[0.; 4], Some(&norm),
                |_| Ok(norm.clone())).is_err());
        }
        args.dump_residuals = None;
        args.dump_layers.clear();
        args.save_logits = Some(root.path().join("logits"));
        for bad in [vec![0.; 3], vec![f32::NAN; 4]] {
            assert!(save_diagnostics(&args, &[0, 1], &[1], 2, &bad, None,
                |_| unreachable!()).is_err());
        }
        Ok(())
    }

    #[test]
    fn shared_diagnostic_enforces_finite_scores_and_inclusive_stop_bars() {
        assert!(shared_diagnostic_verdict(0.01, 0.0, 0.03, 0.7).is_ok());
        assert!(shared_diagnostic_verdict(2.493438096150, 2.492457464922,
            0.021612583211, 0.613037691297).is_ok());
        assert!(shared_diagnostic_verdict(0.01f64.next_up(), 0.0, 0.03, 0.7).is_err());
        assert!(shared_diagnostic_verdict(0.01, 0.0, 0.03f64.next_up(), 0.7).is_err());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for index in 0..4 {
                let mut scores = [2.0, 2.0, 0.02, 0.7];
                scores[index] = value;
                assert!(shared_diagnostic_verdict(scores[0], scores[1], scores[2], scores[3]).is_err());
            }
        }
        assert!(shared_diagnostic_verdict(f64::MAX, -f64::MAX, 0.02, 0.7).is_err());
    }
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
            split_device: None,
            check_headers: true,
            nll: false,
            prefix_state: false,
            fp8_diagnostic_reference: false,
            compare_layers: vec![],
            compare_final_norm: false,
            save_logits: None,
            dump_residuals: None,
            dump_layers: vec![],
            dump_layer0: None,
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
            peers: vec![], last_layer: 91, prefill: None, device: 0, split_device: None, check_headers: true, nll: true, prefix_state: false,
            fp8_diagnostic_reference: false, compare_layers: vec![], compare_final_norm: false,
            save_logits: None, dump_residuals: None, dump_layers: vec![], dump_layer0: None };
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
