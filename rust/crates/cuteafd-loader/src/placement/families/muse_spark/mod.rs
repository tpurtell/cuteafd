//! Muse Spark facts and cost tables: the shared contract planner prices them
//! (`admission`), and both `cuteafd plan --layout` and serving startup submit
//! the same request, so plan == runtime by construction.
//!
//! Coordinator: header-exact per-rank layer weights from the shared
//! [`LayerLayout`] the device packs, paged full-layer KV, per-sequence sliding
//! rings and exact sliding marks, pool first, then routed layers onboarded on
//! the RTXs: TP2 halves on a pair (both GPUs reply BF16 partials over P2P),
//! whole on one RTX (its own partials); the same partial combine norms every
//! holder's partials (ranks 2 or 1). Sparks hold
//! intermediate slices of every expert (TP, as every EXL3 family) and reply
//! per-expert partials (BF16 or NVFP4 by the per-request row cutoff); the
//! coordinator norms, weights and sums them (`replies.rs`).
use crate::families::muse_spark::MuseSparkConfig;
use crate::placement::*;
use crate::serving_capacity::{prefill_waves::PrefillPlan, Exl3Epilogue};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};

mod admission;
pub use admission::*;

pub const PAGE_ROWS: u64 = 64;
/// Prefill transport endpoints: one shared link.
pub const LANES: u64 = 1;
const HEAD_SPLIT: &[LayerMode] = &[LayerMode::HeadSplit];
const WHOLE: &[LayerMode] = &[LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }];
/// Every layer head-split on an RTX pair, or whole on one RTX.
pub const MUSE_SPARK: ExecutorModes = ExecutorModes { family: "muse_spark",
    modes: &[LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }], hops: false,
    attention: AttentionPlacement::Heads };

/// Program namespaces a coordinator layout binds (`muse_` geometry-free and
/// whole-head programs; `muse2_` the head-split halves).
pub fn program_families(gpus: usize) -> &'static [&'static str] {
    if gpus == 2 { &["muse", "muse2"] } else { &["muse"] }
}

/// Native code a rank holds beyond its CUDA context once ready: offline the
/// full allowance (`pending` = allowance), live what the sample has not yet
/// loaded. One group name on both edges.
pub fn code_demand(rank: usize, pending: u64) -> Result<Demand> {
    Ok(Demand::new(u8::try_from(rank)?, Category::Workspace, "Muse future loaded code", pending, Basis::Formula))
}

/// The executor and the layer modes one coordinator count runs.
pub fn executor(gpus: usize) -> Result<(ExecutorModes, &'static [LayerMode])> {
    match gpus {
        1 => Ok((MUSE_SPARK, WHOLE)),
        2 => Ok((MUSE_SPARK, HEAD_SPLIT)),
        other => anyhow::bail!("Muse Spark runs on one RTX or a head-split RTX pair, not {other}"),
    }
}

/// Header-only bundled tower inventory shared by offline and measured admission.
#[derive(Debug, Clone)]
pub struct MediaOptions {
    pub vision: crate::plan::MediaMode,
    pub audio: crate::plan::MediaMode,
    pub replicas: usize,
    pub vision_bytes: (u64, u64),
    pub audio_bytes: (u64, u64),
}

impl MediaOptions {
    pub fn from_checkpoint(checkpoint: &crate::plan::Checkpoint, vision: crate::plan::MediaMode,
        audio: crate::plan::MediaMode, replicas: usize) -> Result<Self> {
        use crate::{media, plan::MediaMode};
        let vision_bytes = if vision == MediaMode::Off
            || checkpoint.config.get("vision_config").is_none_or(serde_json::Value::is_null) { (0, 0) }
        else {
            (media::muse_vision::resident_weights(&checkpoint.config, checkpoint.tensors.iter().map(|t| &t.meta))
                .map_err(anyhow::Error::msg)?,
                media::muse_vision::scratch_bytes(media::muse_vision::MAX_TOKENS).map_err(anyhow::Error::msg)?)
        };
        ensure!(vision_bytes != (0, 0) || matches!(vision, MediaMode::Auto | MediaMode::Off),
            "checkpoint has no bundled Muse vision tower");
        let audio = crate::plan::resolve_audio_config(audio, &checkpoint.snapshot, &checkpoint.config)?;
        let audio_bytes = if audio == MediaMode::Off { (0, 0) } else {
            let plan = media::pcm_audio::PcmAudioPlan::from_snapshot(&checkpoint.snapshot)?;
            (plan.weight_bytes(), media::pcm_audio::scratch_bytes(media::audio::MAX_CLIP_SAMPLES)?)
        };
        Ok(Self { vision, audio, replicas, vision_bytes, audio_bytes })
    }
}

#[derive(Debug, Clone)]
pub struct MediaAdmission {
    pub vision: crate::plan::encoder::EncoderPlacement,
    pub audio: crate::plan::encoder::EncoderPlacement,
    /// Exact native weights and scratch, charged before expert allocation.
    pub spark_items: Vec<Vec<cuteafd_core::memory_layout::Item>>,
}

/// Sliding-ring rows: the shared ring geometry (window + the rollback step).
pub fn ring_rows(cfg: &MuseSparkConfig) -> Result<u64> {
    Ok(crate::families::muse_spark::cache::swa_ring(cfg).map_err(|error| anyhow::anyhow!("{error}"))?.rows)
}

/// Bytes of one retained mark on one rank: each sliding layer's visible window.
pub fn mark_bytes(cfg: &MuseSparkConfig, ranks: usize) -> u64 {
    (cfg.sliding_layers() * cfg.window * cfg.kv_record_bytes() / ranks) as u64
}

/// The step metadata a rank allocates: a bank per in-flight prefill slot and a
/// decode bank whose page table serves `slots` sequences each stepping up to
/// the 64-row rollback step (decode and verify), capped at the unit's rows.
pub fn step_tables(unit_rows: usize, slots: usize, context: u64, waves: usize)
    -> Result<crate::families::muse_spark::lane::StepTables> {
    let decode = (slots * crate::families::muse_spark::cache::SWA_RING_STEP as usize).min(unit_rows);
    crate::families::muse_spark::lane::StepTables::new(unit_rows, decode, context, waves)
}

/// The sliding layers' rotary tables: the module's default RoPE over the head
/// (span 64, theta from the config), extent = the admitted context.
pub fn rope_tables(cfg: &MuseSparkConfig, context: u64) -> Result<crate::plan::rope::Tables> {
    crate::plan::rope::Tables::new(usize::try_from(context)?,
        [crate::plan::rope::Recipe::new(cfg.head_dim, cfg.rope_theta, None)?])
}

/// Per GPU of a TP2 pair: the receive slots of every channel but `Partials`
/// (attention and post-expert cores, the expert-input and combined halves).
pub fn tp2_exchange_bytes(cfg: &MuseSparkConfig, unit_rows: usize) -> Result<usize> {
    use crate::families::muse_spark::tp2::{receive_slots, slot_bytes, Channel};
    Channel::ALL.iter().filter(|&&c| c != Channel::Partials).try_fold(0usize, |sum, &channel| {
        let bytes = slot_bytes(channel, cfg.hidden, cfg.expert_hidden, cfg.topk, unit_rows)? * receive_slots(channel);
        sum.checked_add(bytes).context("Muse TP2 exchange overflow")
    })
}

/// Per GPU under TP2 onboarding: its own BF16 partials and the `Partials`
/// receive slot holding its peer's for one unit (`rows x topk x E` each).
pub fn tp2_partial_bytes(cfg: &MuseSparkConfig, unit_rows: usize) -> usize {
    2 * unit_rows * cfg.topk * crate::families::muse_spark::replies::PartialFormat::Bf16.row_bytes(cfg.expert_hidden)
}

/// The coordinator EXL3 package for onboarded layers, as the image carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalPackage {
    /// Installed; its per-GPU executor workspace (`local_package`).
    Installed(u64),
    /// The image's manifest is known but the package beside it is not (a
    /// broken image): any onboarded layer the placement chooses is refused.
    Missing(String),
    /// No image to inspect (a plan without a manifest): onboarding is planned,
    /// the package's workspace uncharged.
    Unknown,
}

impl LocalPackage {
    pub fn workspace(&self) -> u64 { if let Self::Installed(bytes) = self { *bytes } else { 0 } }

    fn refusal(package: &str) -> String {
        format!("image lacks {package}; rebuild with the Muse coordinator scope")
    }

    /// Refuses a placement that onboards a layer this image cannot run: the
    /// solve kept onboarding in, so automatic placement lands on 0 RTX layers
    /// only when memory says 0, never because the package is missing.
    pub fn check(&self, placed: &Placement) -> Result<()> {
        if let Self::Missing(package) = self {
            ensure!(placed.onboard_layers == 0, "{} (the placement onboards {} RTX expert layers)",
                Self::refusal(package), placed.onboard_layers);
        }
        Ok(())
    }
}

/// A Muse DFlash2 drafter's coordinator charge: its weights, KV rings and
/// workspaces (the shared DFlash2 inventory), and the head scratch its
/// `TargetHead::Launch` owns to run the target's exact head (K6 head into
/// stored-width F32 rows, trimmed, softcapped) over a draft block batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseDrafter {
    pub owned: u64,
    pub weights: ResidentWeights,
    pub scratch: u64,
    pub head_scratch: u64,
    /// The draft batch's input ids and the policy probe's device route ring
    /// (`layers x HEAD_ROWS x topk` ids) on the head GPU.
    pub runtime: u64,
    pub context_slots: usize,
    pub sequences: usize,
    pub block: usize,
    pub head_dim: usize,
}

/// The drafter's own config against the target: hidden, vocabulary, taps,
/// and the head's softcap and output multiplier (the drafter has no head of
/// its own and reads the target's, which applies both; a drafter trained
/// against other values is refused by name). `stored` is the K6 head's
/// stored vocabulary width; `sms` the head GPU's SM count.
pub fn drafter_inventory(cfg: &MuseSparkConfig, config: &serde_json::Value, slots: usize, sequences: usize,
    stored: usize, sms: u64, kv_cache: crate::families::qwen4::Qwen4KvCache) -> Result<MuseDrafter> {
    use crate::families::glm5::draft_representation::{draft_geometry, draft_resident_bytes_with_mode, GlmDraftCapacity,
        GlmDraftRepresentation};
    let (geometry, block) = draft_geometry(config).map_err(|error| anyhow::anyhow!("Muse drafter geometry: {error}"))?;
    let ring = crate::families::glm5::draft_physical::RING;
    ensure!(config["sliding_window"].as_u64() == Some(ring as u64),
        "Muse DFlash2 window {} (the ring holds {ring})", config["sliding_window"]);
    let d = &config["dflash_config"];
    let taps = d["target_layer_ids"].as_array().context("Muse drafter taps")?.iter()
        .map(|tap| tap.as_u64().map(|t| t as usize).context("Muse drafter tap")).collect::<Result<Vec<_>>>()?;
    ensure!(geometry.hidden as usize == cfg.hidden && geometry.vocab as usize == cfg.vocab_size
        && taps.iter().all(|&tap| tap < cfg.layers), "Muse drafter hidden/vocab/taps do not fit the target");
    let close = |key: &str, target: f64| d[key].as_f64().is_some_and(|value| (value - target).abs() <= 1e-9 * target.abs());
    ensure!(close("final_logit_softcapping", cfg.softcap) && close("output_multiplier", cfg.output_multiplier),
        "Muse drafter head softcap/output_multiplier differ from the target's ({}, {}): its logits would not be \
         the target head's", cfg.softcap, cfg.output_multiplier);
    let block = usize::try_from(block)?;
    let capacity = GlmDraftCapacity::new(slots, sequences, block).map_err(|error| anyhow::anyhow!("{error}"))?;
    let residency = draft_resident_bytes_with_mode(config, capacity.context_slots, capacity.max_batch_sequences,
        sms, GlmDraftRepresentation::Bf16Only, 2, kv_cache).map_err(|error| anyhow::anyhow!("Muse drafter: {error}"))?;
    // Stored-width F32 rows of the K6 head and the trimmed F32 rows, per draft row.
    ensure!(stored >= cfg.vocab_size, "Muse head stored width below the vocabulary");
    let rows = capacity.max_batch_sequences * block;
    let head_scratch = u64::try_from(rows * (stored + cfg.vocab_size) * 4)?;
    let probe = (cfg.layers * crate::families::muse_spark::lane::HEAD_ROWS * cfg.topk * 4) as u64;
    let runtime = u64::try_from(rows * 4)? + probe;
    Ok(MuseDrafter { owned: residency.owned, weights: residency.weights, scratch: residency.scratch,
        head_scratch, runtime, context_slots: capacity.context_slots,
        sequences: capacity.max_batch_sequences, block, head_dim: geometry.head_dim as usize })
}

/// The combine's rank tables the coordinator uploads per wave: BF16, wide and other compressed
/// `rank_regions` tables (`MAX_COMBINE_RANKS` i64 addresses each) and their
/// three row maps (`capacity` u32 each).
pub const MAX_COMBINE_RANKS: usize = 8;
pub fn combine_tables_bytes(capacity: usize) -> usize { 3 * MAX_COMBINE_RANKS * 8 + 3 * capacity * 4 }

/// A missing coordinator EXL3 package (onboarding unavailable in this build).
#[derive(Debug, thiserror::Error)]
#[error("coordinator EXL3 package {0} is not installed beside the program manifest")]
pub struct MissingLocalPackage(pub String);

/// The coordinator's installed EXL3 package for onboarded layers (`rtx-tp2` on
/// a pair, `rtx-tp1` on one GPU) and the device workspace its executor
/// allocates per GPU: every capacity specialization's buffers up to the
/// compiled one (the shared `exl3_workspace_bytes`), plus the executor's F32
/// token output (allocated by the shared executor; Muse reads the partial
/// epilogue's planes instead), the partial epilogue's three row maps and the
/// onboarded combine's tables (three 8-rank address tables, three row maps).
pub fn local_package(catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path, gpus: usize, rows: u64)
    -> Result<(std::path::PathBuf, u64)> {
    let exl3 = catalog.exl3().context("Muse onboarding needs EXL3 expert headers")?;
    let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let stem = format!("exl3-muse-k{tiers}");
    let layout = if gpus == 2 { "rtx-tp2" } else { "rtx-tp1" };
    let parent = manifest.parent().context("Muse program manifest parent")?;
    let package = [parent.join("exl3").join(&stem), parent.join("../lib/exl3").join(&stem)].into_iter()
        .map(|p| p.join(layout)).find(|p| p.is_dir())
        .ok_or_else(|| anyhow::Error::new(MissingLocalPackage(format!("{stem}/{layout}"))))?;
    let capacities = cuteafd_core::EXL3_CAPACITIES.map(u64::from);
    let compiled = capacities.into_iter().find(|&c| c >= rows).context("Muse onboarded rows beyond EXL3 capacities")?;
    let shape = catalog.routed_experts();
    let manifests = capacities.into_iter().filter(|&c| c <= compiled).map(|capacity| {
        let directory = package.join(format!("m{capacity}"));
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
            && value["intermediate"].as_u64() == Some(shape.intermediate as u64 / gpus as u64)
            && value["experts"].as_u64() == Some(shape.experts as u64) && value["top_k"].as_u64() == Some(shape.topk as u64)
            && value["capacity"].as_u64() == Some(capacity) && value["output_dtype"].as_str() == Some("fp32")
            && value["bits"] == serde_json::json!(exl3.decoder_tiers()),
            "Muse {layout} EXL3 capacity/geometry/tiers/output mismatch in {}", directory.display());
        Ok(value)
    }).collect::<Result<Vec<_>>>()?;
    // The coordinator's onboarded TP2 lane runs the shared EXL3 lane builder,
    // so this census keeps the weighted sum's owner for it; only the Spark
    // worker (muse_owner_bytes) declares the partial-reply epilogue.
    let workspace = crate::serving_capacity::exl3_workspace_bytes(&manifests, true, Exl3Epilogue::Sum)
        .map_err(|error| anyhow::anyhow!("{error}"))? + compiled * shape.hidden as u64 * 4
        + 3 * rows * 4 + combine_tables_bytes(usize::try_from(rows)?) as u64;
    Ok((package, workspace))
}

/// The coordinator package state both the planner and startup build from:
/// installed (its workspace charged), missing beside a known manifest, or
/// unknown without one; any other inventory error refused.
pub fn local_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, gpus: usize,
    unit_rows: u64) -> Result<LocalPackage> {
    let Some(manifest) = manifest else { return Ok(LocalPackage::Unknown) };
    match local_package(catalog, manifest, gpus, unit_rows) {
        Ok((_, workspace)) => Ok(LocalPackage::Installed(workspace)),
        Err(error) => match error.downcast::<MissingLocalPackage>() {
            Ok(MissingLocalPackage(package)) => Ok(LocalPackage::Missing(package)),
            Err(error) => Err(error),
        },
    }
}

/// The coordinator intake of the Spark link: each rank's plane row width
/// (`replies::intake_row_bytes`, the worst reply mix spread over the unit) and
/// the bytes of every rank's planes over the in-flight waves. The runtime link
/// allocates exactly this (`SparkIntake::with_waves(ranks, unit, row, waves)`).
pub fn intake(cfg: &MuseSparkConfig, prefill: PrefillPlan, cutoff: crate::families::muse_spark::replies::Cutoff,
    sequences: usize, spark_ranks: usize) -> Result<(usize, u64)> {
    let unit = usize::try_from(prefill.unit_rows)?;
    let row = crate::families::muse_spark::replies::intake_row_bytes(cutoff, unit, sequences, cfg.topk, cfg.expert_hidden);
    ensure!(row % 2 == 0, "Muse intake rows are whole BF16 elements");
    Ok((row, crate::serving_capacity::prefill_return_bytes(prefill.waves, prefill.unit_rows, spark_ranks as u64,
        row as u64 / 2)))
}

/// NIC largest RC message (ibv `max_msg_sz`, 1 GiB on the ConnectX-7 ports of
/// raptor and every Spark); a partial-reply frame above it is refused at admission.
pub const NIC_MAX_MESSAGE_BYTES: usize = 1 << 30;

/// Muse's one persistent Spark endpoint (decode, verify and prefill share its
/// QP) and its rings, shared by the planner, admission and the transport:
///
/// - the response frame is the worst partial reply one rank sends at the unit
///   (`replies::max_rank_bytes`: BF16 decode/verify segments and the wide
///   prefill rest, over the admitted sequences), one unchunked SEND;
/// - the ring depth is what can be in flight on the QP: the fixed prefill
///   waves plus the decode lane's one step. Slots are uniform, so each holds
///   the worst frame.
pub fn spark_endpoint(prefill: PrefillPlan, cutoff: crate::families::muse_spark::replies::Cutoff,
    sequences: usize) -> Result<(cuteafd_core::SparkEndpoint, cuteafd_core::SparkRings)> {
    let geometry = cuteafd_core::ExpertGeometry::MUSE_SPARK;
    let unit = usize::try_from(prefill.unit_rows)?;
    let reply = crate::families::muse_spark::replies::max_rank_bytes(cutoff, unit, sequences,
        geometry.topk as usize, geometry.hidden as usize);
    let depth = usize::try_from(prefill.waves)?.checked_add(1).context("Muse ring depth")?;
    let base = crate::plan::layout::spark_ring_geometry(LANES as usize);
    let endpoint = cuteafd_core::SparkEndpoint { ring_depth: Some(depth), capacity: u32::try_from(unit)?,
        ingress: cuteafd_core::SparkIngress::Fp8K32, partial_reply_bytes: Some(u64::try_from(reply)?) };
    let rings = cuteafd_core::SparkRings { max_frame_bytes: base.max_frame_bytes, depth: base.depth,
        slot_bytes: base.slot_bytes, alignment: base.alignment };
    let frame = endpoint.frame_bytes(geometry, rings).context("Muse Spark endpoint frame")?;
    ensure!(frame <= NIC_MAX_MESSAGE_BYTES, "Muse Spark partial reply frame of {frame} bytes exceeds the NIC's \
        {NIC_MAX_MESSAGE_BYTES}-byte RC message limit (lower the unit, the sequences or the BF16 cutoff)");
    Ok((endpoint, rings))
}

/// The coordinator host's Spark transport for Muse's one endpoint, as the
/// link registers it: every rank's rings at the sealed endpoint (`spark_endpoint`),
/// the request egress (one bootstrap-wide FP8 wire per wave slot), and the
/// intake planes' host-mode upper bound (`intake`; GPU landing uses less).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostTransport { pub rings: u64, pub egress: u64, pub intake: u64 }

pub fn host_transport(cfg: &MuseSparkConfig, prefill: PrefillPlan, cutoff: crate::families::muse_spark::replies::Cutoff,
    sequences: usize, spark_ranks: usize) -> Result<HostTransport> {
    let geometry = cuteafd_core::ExpertGeometry::MUSE_SPARK;
    let (endpoint, rings) = spark_endpoint(prefill, cutoff, sequences)?;
    let per_rank = endpoint.ring_bytes(geometry, rings).context("Muse host ring geometry")?;
    let unit = u32::try_from(prefill.unit_rows)?;
    let egress = geometry.compact_wire_bytes(unit, false).context("Muse egress geometry")?.0 as u64;
    let (_, intake) = intake(cfg, prefill, cutoff, sequences, spark_ranks)?;
    Ok(HostTransport {
        rings: (per_rank as u64).checked_mul(spark_ranks as u64).context("Muse host rings overflow")?,
        egress: egress.checked_mul(prefill.waves).context("Muse host egress overflow")?,
        intake })
}

/// Muse's typed Spark owners on the EXL3 ladder.
pub fn spark_executions(waves: u64, unit_rows: u64) -> Result<cuteafd_core::SparkExecutions> {
    let exports = crate::plan::spark_executions::SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest).exports();
    crate::plan::spark_executions::admit_family("muse_spark", &exports, None, unit_rows, waves)
}

/// The pre-placement Spark reserve: the runtime residual plus the typed owners.
pub fn spark_reserve_bytes(world: usize, waves: u64, unit_rows: u64) -> Result<u64> {
    let owners = spark_executions(waves, unit_rows)?.workspace_bytes(|extent| owner_bytes(world, extent))?
        .and_then(|bytes| bytes.total()).context("Muse Spark owner reserve overflow")?;
    cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0)
        .and_then(|bytes| bytes.checked_add(owners)).context("Muse Spark reserve overflow")
}

pub fn owner_bytes(world: usize, extent: cuteafd_core::OwnerExtent) -> Result<cuteafd_core::OwnerBytes> {
    crate::plan::layout::spark_slots::muse_owner_bytes(world, extent)
}
