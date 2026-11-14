//! Qwen automatic KV admission after weights/PLE and exact expert ownership are established.
use super::EngineArgs;
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::qwen4::Qwen4Config;
use cuteafd_loader::serving_capacity::qwen_cache_geometry;

use cuteafd_loader::serving_capacity::qwen_graphs::QwenAdmissionInputs;
use cuteafd_loader::placement::{families::qwen4, Baseline, ResidentWeights, RuntimeSample};

pub(super) fn single_weights(args: &EngineArgs, checkpoint: &cuteafd_loader::plan::Checkpoint,
    cfg: &Qwen4Config, layers: usize, mtp: bool) -> Result<ResidentWeights> {
    use cuteafd_loader::families::qwen4::resident::Qwen4Representation;
    let mut weights = qwen4::single_owner_weights(checkpoint, cfg, layers, mtp,
        Qwen4Representation { fp8_projections: args.fp8_decode, fp8_head: args.mtp_fp8_head },
        args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Host)?;
    if let Some((gpu, media)) = args.planner_media_weights {
        if gpu == args.device { weights = weights.checked_add(media)?; }
        else { media.check(0, crate::shared::memory::budget::resident_weight_limit())?; }
    }
    weights.check(0, crate::shared::memory::budget::resident_weight_limit())?;
    Ok(weights)
}

pub(super) fn loaded_media_weights(devices: &[i32], media: Option<(i32, ResidentWeights)>)
    -> Result<Option<cuteafd_loader::placement::Demand>> {
    let Some((gpu, weights)) = media else { return Ok(None) };
    let Some(rank) = devices.iter().position(|&device| device == gpu) else {
        weights.check(0, crate::shared::memory::budget::resident_weight_limit())?;
        return Ok(None);
    };
    Ok(Some(cuteafd_loader::placement::Demand::new(u8::try_from(rank)?,
        cuteafd_core::memory_layout::Category::Weights, "loaded vision weights", 0,
        cuteafd_core::memory_layout::Basis::Exact).with_resident_weights(weights)))
}

pub(super) fn expert_weights(experts: Option<&super::engine::Experts<'_>>) -> Result<ResidentWeights> {
    use super::engine::{Experts, MtpExperts};
    let mtp = |owner: &MtpExperts<'_>| match owner {
        MtpExperts::Fp8(weights) => weights.resident_bytes(),
        MtpExperts::Exl3(weights) => weights.borrow().resident_weights(),
    };
    Ok(ResidentWeights(match experts {
        Some(Experts::Local(local)) => local.experts.borrow().resident_bytes()
            .checked_add(local.mtp_experts.as_ref().map_or(0, |w| w.resident_bytes()))
            .ok_or_else(|| anyhow::anyhow!("Qwen expert weight overflow"))?,
        Some(Experts::LocalExl3(local)) => local.resident.borrow().as_ref()
            .map_or(0, |(_, weights)| weights.resident_weights()),
        Some(Experts::Spark { mtp: draft, .. }) => draft.as_ref().map_or(0, mtp),
        Some(Experts::SharedOnly) | None => 0,
        Some(Experts::Tp2 { .. }) => anyhow::bail!("Qwen TP2 needs dual admission"),
    } as u64))
}

pub(super) fn spark_model(backend: cuteafd_loader::plan::spark_executions::SparkBackend,
    rows: usize, waves: u64, bf16: bool) -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    let capacity = u32::try_from(rows)?;
    let ingress = if bf16 { cuteafd_core::SparkIngress::Bf16 }
        else { cuteafd_core::SparkIngress::Fp8K32 };
    let executions = cuteafd_loader::plan::spark_executions::admit_family("qwen4", &backend.exports(), None,
        rows as u64, waves)?;
    cuteafd_transport::worker_selection::ModelAdmission::new(
        vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity, ingress, partial_reply_bytes: None }],
        crate::shared::placement::spark_rings(64 << 20)?, executions)
}

/// The shared admission inputs (`serving_capacity::qwen_graphs::qwen_admission`) at serve's arguments.
#[allow(clippy::too_many_arguments)]
pub(super) fn inputs<'a>(args: &EngineArgs, cfg: &'a Qwen4Config, layers: usize, mtp: bool,
    manifest: Option<&'a serde_json::Value>, ple: Option<(u64, bool)>, future_expert_bytes: u64) -> Result<QwenAdmissionInputs<'a>> {
    let geometry = qwen_cache_geometry(cfg, layers, mtp, args.kv_format)?;
    let shape = args.resolved_prefill()?;
    let marks = args.planner_prefix_bytes.unwrap_or(geometry.ranks[0].retained_mark_bytes * qwen4::DEFAULT_MARK_SLOTS);
    Ok(QwenAdmissionInputs { prefill_shape: args.prefill_shape()?, cfg, layers, mtp, kv_format: args.kv_format, manifest, prefill_rows: shape.span_rows,
        slots: args.slots as u64, mark_bytes: marks, ple, future_expert_bytes,
        headroom: cuteafd_loader::placement::reserve::gpu("qwen4") })
}

pub(super) use cuteafd_loader::families::qwen4::StepTableBytes;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct HostWorkspaceBytes {
    pub staging: u64,
    pub uploads: cuteafd_loader::shared::token_table::UploadShape,
    pub logits: u64,
}

impl HostWorkspaceBytes {
    #[allow(clippy::too_many_arguments)]
    pub fn new(cfg: &Qwen4Config, rows: usize, decode: bool, pages: usize, pool_pages: usize,
        units: usize, spark: bool, mtp: bool, ple_ids: bool, logits: usize) -> Result<Self> {
        use anyhow::Context;
        let uploads = cuteafd_loader::families::qwen4::upload_shape(rows, decode, pages, pool_pages,
            cfg.ple_rows(), cfg.layers, units, spark, mtp, ple_ids)?;
        Ok(Self { staging: uploads.capacity_bytes()? as u64, uploads,
            logits: logits.checked_mul(cfg.vocab_size).and_then(|n| n.checked_mul(4))
                .context("Qwen pinned logits overflow")? as u64 })
    }
    pub fn mtp(cfg: &Qwen4Config, rows: usize, pages: usize, pools: usize, calls: usize) -> Result<Self> {
        use anyhow::Context;
        let uploads = cuteafd_loader::families::qwen4::mtp_upload_shape(rows, pages, pools, calls)?;
        Ok(Self { staging: uploads.capacity_bytes()? as u64, uploads,
            logits: cfg.vocab_size.checked_mul(4).context("Qwen MTP pinned logits overflow")? as u64 })
    }

}

/// Admission follows the selected executor, not unused configured peers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExpertSource {
    LocalFp8,
    LocalExl3,
    Spark { ranks: usize },
    None,
}

impl ExpertSource {
    pub(super) fn selected(experts: Option<&super::engine::Experts<'_>>) -> Result<Self> {
        Ok(match experts {
            Some(super::engine::Experts::Local(_)) => Self::LocalFp8,
            Some(super::engine::Experts::LocalExl3(_)) => Self::LocalExl3,
            Some(super::engine::Experts::Spark { transport, .. }) => Self::Spark { ranks: transport.borrow().world_size() },
            Some(super::engine::Experts::SharedOnly) | None => Self::None,
            Some(super::engine::Experts::Tp2 { .. }) => anyhow::bail!("Qwen TP2 needs dual admission"),
        })
    }

    fn code_key(self) -> &'static str {
        match self { Self::LocalFp8 => "fp8", Self::LocalExl3 => "exl3", Self::Spark { .. } | Self::None => "none" }
    }

    pub(super) fn spark_ranks(self) -> usize {
        match self { Self::Spark { ranks } => ranks, _ => 0 }
    }
}

pub(super) struct Exl3Admission {
    pub layers: Vec<usize>,
    pub peak_bytes: usize,
    pub rows: usize,
    pub mtp: bool,
    pub selections: Vec<crate::shared::experts::layer::ExpertLayer>,
    pub weights_budget: usize,
}

impl Exl3Admission {
    pub fn validate_layers(&self, actual: &[usize]) -> Result<()> {
        anyhow::ensure!(actual == self.layers,
            "Qwen EXL3 runtime layers differ from pre-upload admission: {actual:?} != {:?}", self.layers);
        Ok(())
    }
}

/// Solve from the pre-weight inventory. Header/package queries below must not
/// allocate target, table, expert or KV storage.
pub(super) fn preallocate(library: &NativeLibrary, args: &EngineArgs,
    checkpoint: &cuteafd_loader::plan::Checkpoint, cfg: &Qwen4Config,
    catalog: Option<&cuteafd_loader::OfficialV41Catalog>, layers: usize, mtp: bool,
    manifest: Option<&serde_json::Value>, context: cuteafd_loader::placement::planner::Context,
    media: &crate::shared::experts::rtx::dense::Records,
    inventory: &crate::shared::inventory::RuntimeInventory) -> Result<(usize, qwen4::QwenHost, Option<Exl3Admission>)> {
    use anyhow::{ensure, Context};
    use cuteafd_core::memory_layout::{Basis, Category};
    use cuteafd_loader::families::qwen4::{resident::Qwen4Representation, weight_layout};
    use cuteafd_loader::placement::Demand;
    use crate::shared::experts::{layer::ExpertLayer, rtx::push::PushPolicy};
    let rows = args.prefill_rows.max(super::engine::DECODE_ROWS);
    ensure!(args.shared_only || args.expert_window.is_none(),
        "Qwen placement requires resident experts; diagnostic expert paging is not admitted");
    let source = if args.shared_only { ExpertSource::None }
        else if args.local_experts {
            if catalog.context("Qwen local expert headers")?.exl3().is_some() { ExpertSource::LocalExl3 }
            else { ExpertSource::LocalFp8 }
        } else if let Some(peers) = &args.peers { ExpertSource::Spark { ranks: peers.split(',').count() } }
        else { ExpertSource::None };
    let selected = Qwen4Representation { fp8_projections: args.fp8_decode, fp8_head: args.mtp_fp8_head };
    let embedding = if args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu {
        weight_layout::embedding_table_bytes(checkpoint)? } else { 0 };
    let arenas = weight_layout::coordinator_arenas(checkpoint, cfg, &vec![0; layers], 1,
        mtp, selected, embedding)?.into_iter().map(|(_, layout)| layout).collect::<Vec<_>>();
    let resident = cuteafd_loader::families::qwen4::resident::checkpoint_resident_bytes(
        checkpoint, cfg, layers, mtp, selected)?;
    let ple = checkpoint.tensors.iter().find(|t| t.meta.name.contains("ngram_embedding.shard_"))
        .map(|t| (resident.ple_row_bytes, t.meta.dtype == cuteafd_core::DType::F8E4M3));
    let mut future = Vec::new();
    let mut expert_arenas = Vec::new();
    if args.fp8_decode {
        future.push(Demand::new(0, Category::Staging, "FP8 projection load staging",
            cuteafd_loader::shared::dense_read::STEP_BYTES as u64, Basis::Exact));
    }
    let mut window_push = Vec::new();
    let mut exl3 = None;
    if !args.shared_only {
        if let Some(catalog) = catalog {
            if source == ExpertSource::LocalExl3 {
                let backbone = (0..layers).collect::<Vec<_>>();
                let planned = crate::families::deepseek_v4::local::plan(library, &args.native_lib,
                    catalog, usize::from(mtp), &backbone, rows, usize::MAX)?;
                ensure!(planned.layers == backbone, "Qwen EXL3 header plan omits admitted layers");
                let expected = (0..usize::from(mtp)).map(|stage| ExpertLayer::Dspark { stage })
                    .chain(backbone.iter().map(|&layer| ExpertLayer::BackboneFull { layer }))
                    .collect::<Vec<_>>();
                let (selections, weights_budget) = crate::families::deepseek_v4::local::LocalExperts::exl3_selections(
                    &args.native_lib, catalog, usize::from(mtp), &backbone, rows, planned.peak_bytes)?
                    .context("Qwen EXL3 admitted package")?;
                ensure!(selections == expected, "Qwen EXL3 push plan omits admitted layers");
                expert_arenas.push(crate::shared::experts::weight_layout::layout(args.device, "exl3",
                    crate::shared::experts::exl3::regions(catalog, selections.iter().copied())?)?);
                future.push(Demand::new(0, Category::Workspace, "EXL3 execution and load workspace",
                    planned.peak_bytes.checked_sub(planned.resident_weights)
                        .context("Qwen EXL3 peak below resident weights")? as u64, Basis::Exact));
                if PushPolicy::for_snapshot(args.experts_snapshot.as_deref().unwrap_or(&args.snapshot))?
                    == PushPolicy::Required {
                    use crate::shared::experts::rtx::push::{CoordinatorExperts, WholeOwner};
                    window_push.push(("qwen.host.push.exl3", CoordinatorExperts::Whole {
                        owners: vec![WholeOwner { rank: 0, selections: selections.clone(), budget: weights_budget }]
                    }.host_envelope(catalog, 1, &[])?));
                }
                exl3 = Some(Exl3Admission { layers: planned.layers, peak_bytes: planned.peak_bytes,
                    rows, mtp, selections, weights_budget });
            } else if source == ExpertSource::LocalFp8 {
                let tensors = catalog.fp8().context("Qwen native expert headers")?;
                let mut sets = vec![(0..layers).collect::<Vec<_>>()];
                if mtp {
                    if tensors.layer_format(layers)? == tensors.format() { sets[0].push(layers); }
                    else { sets.push(vec![layers]); }
                }
                for ids in sets {
                    let tensors = tensors.for_layer(ids[0])?;
                    let directory = if ids[0] < layers { args.fp8_package.clone() } else { None }
                        .unwrap_or_else(|| crate::shared::experts::fp8::package_directory(&args.native_lib, 1, tensors.format()));
                    // SAFETY: trusted generated package static ABI; no CUDA state is created.
                    let metadata = unsafe { cuteafd_ffi::fp8_moe::Fp8MoeMetadata::read(&directory) }?;
                    crate::shared::experts::fp8::validate_checkpoint_contract(&metadata.info, &tensors,
                        &ids, 1, 0, cuteafd_loader::formats::fp8_experts::Slicing::Padded)?;
                    let layout = crate::shared::experts::weight_layout::layout(args.device, "fp8",
                        cuteafd_loader::serving_capacity::fp8_expert_regions(&tensors, &ids, 1, 0,
                            cuteafd_loader::formats::fp8_experts::Slicing::Padded)?)?;
                    expert_arenas.push(layout);
                    future.push(Demand::new(0, Category::Workspace, "native expert scratch",
                        cuteafd_ffi::fp8_moe::fp8_slot_scratch_bytes(metadata.scratch_layout(), rows)? as u64, Basis::Exact));
                }
            } else if mtp && source.spark_ranks() > 0 {
                future.extend(qwen4::dual_draft_demands(catalog, &args.native_lib, rows as u64)?);
            }
        }
    }
    let policy = PushPolicy::for_snapshot(&args.snapshot)?;
    if let Some(envelope) = policy.plan(|| {
        let staging = dual_dense_staging(checkpoint, cfg, &vec![0; layers], mtp,
            args.fp8_decode, args.mtp_fp8_head, media, [args.device, args.device])?;
        Ok(cuteafd_loader::push_load::PushHostEnvelope::new(vec![0], vec![staging[0]],
            cuteafd_ffi::push_ring::DEFAULT_SLOTS as u64, cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64)?)
    })? { window_push.push(("qwen.host.push.dense", envelope)); }
    let host_embedding = if embedding == 0 { weight_layout::embedding_table_bytes(checkpoint)? as u64 } else { 0 };
    let host = host_owners(args, cfg, layers, mtp, 1, source.spark_ranks(),
        checkpoint.spark_input_is_bf16(), ple.map(|p| p.0), host_embedding, window_push)?;
    let admission = inputs(args, cfg, layers, mtp, manifest, ple, 0)?;
    let sample = inventory.gpus.first().context("Qwen pre-weight sample")?;
    let pending = crate::shared::inventory::pending_code(library, args.device, 0, false, "qwen4", source.code_key(),
        &manifest.map(|m| cuteafd_loader::placement::ProgramSet::from_manifest(m, &["qwen4"]))
            .unwrap_or_else(|| cuteafd_loader::placement::ProgramSet::from_names(&["qwen4"], [])))?;
    let mut inputs = placement_inputs(args, admission, source, (sample.free_bytes, sample.total_bytes), pending, context, host);
    inputs.arenas = arenas;
    inputs.expert_arenas = expert_arenas;
    inputs.preallocation = true;
    inputs.future = future;
    // The optional tower already loaded before this sample, so only declare its
    // resident bytes. Its physical allocation is part of the measured baseline.
    inputs.loaded_weights = args.planner_media_weights.filter(|(gpu, _)| *gpu == args.device)
        .map_or(ResidentWeights::default(), |(_, bytes)| bytes);
    let (placement, solved) = qwen4::admitted(&inputs)?;
    cuteafd_loader::placement::reserve::log_plan(&solved, "qwen4");
    publish_reconciliation(args, &inventory.gpus, &solved, source.spark_ranks())?;
    cuteafd_bench::context::ResolvedSettings::placement(&placement).publish();
    tracing::info!(pool_tokens = placement.pool_tokens,
        ledger = %serde_json::to_string(&cuteafd_loader::plan::SolvedPhaseLedger::from(&solved))?,
        "Qwen shared solver admission before allocation");
    Ok((usize::try_from(placement.pool_tokens)?, inputs.host, exl3))
}

/// Qwen's KV pool from one sample of free memory after weights, the PLE table and resident experts:
/// the shared admission's fixed items, the graph reserve before them, and the largest pool that fits.
pub(super) fn pool_tokens(library: &NativeLibrary, args: &EngineArgs, inputs: &QwenAdmissionInputs<'_>,
    source: ExpertSource, context: cuteafd_loader::placement::planner::Context, loaded_weights: ResidentWeights,
    host: qwen4::QwenHost, devices: &[RuntimeSample]) -> Result<(usize, qwen4::QwenHost)> {
    let pending = crate::shared::inventory::pending_code(library, args.device, 0, false, "qwen4", source.code_key(),
        &inputs.manifest.map(|m| cuteafd_loader::placement::ProgramSet::from_manifest(m, &["qwen4"]))
            .unwrap_or_else(|| cuteafd_loader::placement::ProgramSet::from_names(&["qwen4"], [])))?;
    let current = library.cuda_get_device()?;
    library.cuda_set_device(args.device)?;
    let sample = library.cuda_memory_info();
    library.cuda_set_device(current)?;
    let (available, total) = sample?;
    let admission = QwenAdmissionInputs { prefill_shape: inputs.prefill_shape, cfg: inputs.cfg, layers: inputs.layers, mtp: inputs.mtp, kv_format: inputs.kv_format,
        manifest: inputs.manifest, prefill_rows: inputs.prefill_rows, slots: inputs.slots,
        mark_bytes: inputs.mark_bytes,         ple: inputs.ple, future_expert_bytes: inputs.future_expert_bytes, headroom: inputs.headroom };
    let mut inputs = placement_inputs(args, admission, source,
        (available as u64, total as u64), pending, context, host);
    inputs.loaded_weights = loaded_weights;
    let (placement, solved) = qwen4::admitted(&inputs)?;
    cuteafd_loader::placement::reserve::log_plan(&solved, "qwen4");
    publish_reconciliation(args, devices, &solved, source.spark_ranks())?;
    cuteafd_bench::context::ResolvedSettings::placement(&placement).publish();
    tracing::info!(pool_tokens = placement.pool_tokens,
        ledger = %serde_json::to_string(&cuteafd_loader::plan::SolvedPhaseLedger::from(&solved))?,
        "Qwen shared solver admission before allocation");
    Ok((usize::try_from(placement.pool_tokens)?, inputs.host))
}

/// Publish the admission identity the shared Ready and first-request memory
/// reconciliation reads (`shared::memory_report::reconcile`). The capture
/// returns before sampling without `memory_reconciliation_provenance`, so
/// `/v1/stats` would report no `delta_vs_measured` or `graph_reserve_slack`
/// for the family. Identity only: hashes of the native library and program
/// manifest, the served layout, and one inventory sample per physical GPU in
/// rank order (the capture indexes `devices` by the `rtxN` owner row).
pub(super) fn publish_reconciliation(args: &EngineArgs, devices: &[RuntimeSample],
    solved: &cuteafd_loader::placement::planner::ResolvedPlan, spark_world: usize) -> Result<()> {
    use sha2::{Digest, Sha256};
    let hash = |path: &std::path::Path| -> Result<String> {
        let mut file = std::fs::File::open(path)?;
        let mut digest = Sha256::new();
        std::io::copy(&mut file, &mut digest)?;
        Ok(format!("{:x}", digest.finalize()))
    };
    let layout = format!("snapshot/{}; context/{}; slots/{}; kv/{:?}; prefill/{}x{} rows",
        args.snapshot.display(), args.max_context, args.slots, args.kv_format,
        solved.decisions.prefill.rows, solved.decisions.prefill.waves);
    cuteafd_ffi::memory_ledger::fault::set_inputs(serde_json::json!({
        "memory_reconciliation_provenance": reconciliation_provenance(
            env!("CUTEAFD_BUILD_COMMIT"), env!("CUTEAFD_BUILD_DIRTY"), env!("CUTEAFD_BUILD_TOOLCHAIN"),
            &hash(&args.native_lib)?, &hash(&args.manifest)?, &layout, devices, spark_world,
            &solved.decisions.onboard_layers, solved.decisions.prefill)
    }));
    Ok(())
}

/// Split out so a CPU test checks the fields the shared capture reads.
fn reconciliation_provenance(source_commit: &str, source_dirty: &str, toolchain: &str,
    native_sha256: &str, programs_sha256: &str, layout: &str, devices: &[RuntimeSample],
    spark_world: usize, onboard_layers: &[Vec<usize>],
    prefill: cuteafd_loader::placement::planner::PrefillShape) -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1, "family": "qwen4", "source_commit": source_commit,
        "source_dirty": source_dirty, "toolchain": toolchain,
        "native_sha256": native_sha256, "programs_sha256": programs_sha256,
        "layout": layout, "devices": devices, "spark_world": spark_world,
        "onboard_layers": onboard_layers, "prefill": prefill,
        "graph_basis": "per-device bounded cache; Basis::Bytes from each shape's traced capture bytes (#1627)"
    })
}

fn placement_inputs<'a>(args: &EngineArgs, admission: QwenAdmissionInputs<'a>, source: ExpertSource,
    (available, total): (u64, u64), pending: u64, context: cuteafd_loader::placement::planner::Context,
    host: qwen4::QwenHost) -> qwen4::QwenInputs<'a> {
    qwen4::QwenInputs {
        experts: None, onboard: cuteafd_loader::placement::Onboard::Auto,        arenas: vec![], expert_arenas: vec![], preallocation: false, future: vec![],
        blas_head: crate::shared::target_head::TargetHead::configured().blas_head(!args.mtp_fp8_head),
        loaded_weights: Default::default(),
        resident_weight_limit: Some(crate::shared::memory::budget::resident_weight_limit()), spark_reply_format: cuteafd_core::SparkReplyFormat::resolve(cuteafd_core::spark_reply::requested().prefill, false), context, admission, capacity_bytes: total,
        baseline: Baseline::Measured { free_bytes: available }, pending_code_bytes: pending,
        max_context: args.max_context as u64, requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        spark_ranks: source.spark_ranks(),
        host,
        // The launcher placed the tower; it loaded before this solve and its
        // device bytes are in the measured baseline.
        encoders: Default::default() }
}

/// Live coordinator host owners for one route, from the engine arguments and
/// the same formulas the offline plan uses, against one host sample that
/// credits this process's own anonymous memory (every host owner it already
/// holds is charged once by the recipe). `push` names the admitted push
/// receive envelopes that will be prepared after the solve.
pub(super) fn host_owners(args: &EngineArgs, cfg: &Qwen4Config, layers: usize, mtp: bool, owners: usize,
    spark_ranks: usize, spark_bf16: bool, ple_row_bytes: Option<u64>, host_embedding: u64,
    push: Vec<(&'static str, cuteafd_loader::push_load::PushHostEnvelope)>)
    -> Result<qwen4::QwenHost> {
    host_owners_with_memory(args, cfg, layers, mtp, owners, spark_ranks, spark_bf16, ple_row_bytes,
        host_embedding, push, cuteafd_loader::formats::mapped_preload::host_memory_before_process()?)
}

fn host_owners_with_memory(args: &EngineArgs, cfg: &Qwen4Config, layers: usize, mtp: bool, owners: usize,
    spark_ranks: usize, spark_bf16: bool, ple_row_bytes: Option<u64>, host_embedding: u64,
    push: Vec<(&'static str, cuteafd_loader::push_load::PushHostEnvelope)>, memory: (u64, u64))
    -> Result<qwen4::QwenHost> {
    use qwen4::host::{HostMemory, PleHost, QwenHost, UploadInputs, mtp_pinned_logits_bytes, pinned_logits_bytes, scoring_logits_bytes};
    let (total, available) = memory;
    let shape = args.resolved_prefill()?;
    let unit_rows = shape.unit_rows.max(super::engine::DECODE_ROWS as u64);
    let waves = shape.waves.max(1);
    let mapped = cfg.ple_layers.first().is_some_and(|&layer| layer < layers);
    let ple = match (mapped, ple_row_bytes) {
        (true, Some(row_bytes)) => Some(PleHost::new(cfg, unit_rows, waves, row_bytes)?),
        (true, None) => anyhow::bail!("Qwen mapped PLE layer without table headers"),
        (false, _) => None,
    };
    let (bounds, mtp_upload) = UploadInputs { prefill_rows: usize::try_from(unit_rows)?,
        waves: usize::try_from(waves)?, spark: spark_ranks > 0, mtp: mtp && cfg.mtp_layers > 0, ple_ids: !mapped }
        .bounds(cfg)?;
    let (prefix_quota, prefix_chunk) = match &args.planner_host_prefix {
        Some((prefix, concurrency)) if prefix.prefix_cache_entries != 0 && prefix.host_cache_bytes.enabled() =>
            qwen4::host::prefix_quota(qwen4::host::prefix_geometry(cfg, layers, mtp, args.kv_format)?,
                match prefix.host_cache_bytes { crate::shared::prefix::HostBudget::Bytes(bytes) => Some(bytes),
                    crate::shared::prefix::HostBudget::Auto => None },
                args.max_context as u64, *concurrency as u64, total)?,
        _ => (0, 0),
    };
    let spark = if spark_ranks > 0 {
        let intake_rows = if waves == 1 { unit_rows } else { shape.units().slot_rows() };
        let geometry = cuteafd_core::ExpertGeometry { hidden: u32::try_from(cfg.hidden)?,
            experts: u32::try_from(cfg.experts)?, topk: u32::try_from(cfg.topk)?,
            intermediate: u32::try_from(cfg.moe_intermediate)?, layers: u32::try_from(cfg.layers)? };
        cuteafd_loader::serving_capacity::spark_host_transport(geometry,
            cuteafd_loader::plan::layout::spark_endpoints("qwen4", 1), spark_ranks, false, true,
            spark_bf16, unit_rows, intake_rows, waves, None)?
    } else { [0; 3] };
    Ok(QwenHost {
        memory: HostMemory { total, available },
        ple, uploads: vec![bounds; owners], mtp_upload,
        pinned_logits: pinned_logits_bytes(owners, cfg.vocab_size as u64)?,
        mtp_pinned_logits: mtp_pinned_logits_bytes(args.prefill_shape()?.is_some(), mtp && cfg.mtp_layers > 0, cfg.vocab_size as u64)?,
        scoring_logits: scoring_logits_bytes(shape.span_rows, cfg.vocab_size as u64)?,
        prefix_quota, prefix_chunk,
        loaded_media_cache: args.planner_media_cache_bytes,
        host_embedding,
        table_reactor: cuteafd_loader::formats::mapped_table::table_reactor_workspace_bytes(usize::from(mapped)) as u64,
        preloaded_tables: 0, spark, push,
    })
}

/// Static rank packages, selected before either device allocates experts. The
/// loader manifest plan and backend DSO ABI must agree on widths and bytes.
pub(super) struct DualExpertPlan {
    pub packages: [std::path::PathBuf; 2],
    pub costs: Vec<Option<cuteafd_loader::placement::ExpertCost>>,
    pub workspace: [u64; 2],
    pub transport: Vec<cuteafd_loader::placement::Demand>,
    pub wire: bool,
}

pub(super) fn dual_experts(args: &EngineArgs, catalog: &cuteafd_loader::OfficialV41Catalog,
    layers: usize) -> Result<DualExpertPlan> {
    use anyhow::{ensure, Context};
    use crate::shared::experts::rtx::{exl3::Exl3Tp2, fp8moe::Fp8MoeTp2};
    use cuteafd_loader::formats::fp8_experts::Slicing;
    let rows = args.prefill_rows.max(super::engine::DECODE_ROWS);
    let costs = qwen4::dual_expert_costs(catalog, layers)?;
    let wire = catalog.exl3().is_some();
    let (packages, backend_workspace) = if let Some(exl3) = catalog.exl3() {
        let name = crate::shared::experts::exl3::package_name("qwen4", exl3.decoder_tiers());
        let root = args.native_lib.parent().unwrap_or(std::path::Path::new(".")).join("exl3").join(name);
        let packages = [0, 1].map(|rank| root.join(format!("rtx-tp2-rank{rank}")));
        let workspace = [Exl3Tp2::workspace_bytes_for(&packages[0], catalog.routed_experts().hidden, rows)?,
            Exl3Tp2::workspace_bytes_for(&packages[1], catalog.routed_experts().hidden, rows)?];
        (packages, workspace)
    } else {
        let tensors = catalog.fp8().context("Qwen dual experts need EXL3/FP8/NVFP4")?;
        let directory = args.fp8_package.clone().unwrap_or_else(||
            crate::shared::experts::fp8::package_directory(&args.native_lib, 2, tensors.format()));
        let plans = [Fp8MoeTp2::plan(tensors, &directory, &[tensors.shape().first_layer], 0, rows)?,
            Fp8MoeTp2::plan(tensors, &directory, &[tensors.shape().first_layer], 1, rows)?];
        for rank in 0..2 {
            ensure!(plans[rank].slicing == Slicing::Blocks(128),
                "Qwen dual experts require complete-H128 exact rank packages; padded TP2 is unsupported");
            for cost in costs.iter().flatten() {
                ensure!(cost.half[rank].resident == plans[rank].resident_layer_bytes as u64
                    && cost.half[rank].staging == plans[rank].staging_bytes as u64,
                    "Qwen TP2 header residency disagrees with backend rank {rank}");
            }
        }
        let workspace = plans.each_ref().map(|plan| plan.workspace_bytes);
        (plans.map(|plan| plan.directory), workspace)
    };
    let workspace = qwen4::dual_expert_workspace(catalog,
        packages.each_ref().map(|p| p.as_path()), rows as u64)?;
    for rank in 0..2 {
        ensure!(workspace[rank] == backend_workspace[rank] as u64,
            "Qwen TP2 manifest workspace disagrees with backend rank {rank}");
    }
    let shape = catalog.routed_experts();
    let transport = cuteafd_loader::serving_capacity::OwnerTp2Rows::new(shape.hidden as u64, shape.topk as u64, rows as u64, wire)?.demands()?;
    Ok(DualExpertPlan { packages, costs, workspace, transport, wire })
}

/// Checkpoint-native whole-layer storage plus entry-owner operands, planned
/// before allocations. PLE follows its layer; draft experts remain separate.
pub(super) struct DualWeightPlan {
    pub resident: cuteafd_loader::families::qwen4::resident::Qwen4CheckpointResident,
    /// The arenas `Qwen4Loader::with_arenas` allocates at this cut, from the
    /// same headers, owners and representation: charged once each by the solver.
    pub arenas: Vec<(usize, cuteafd_loader::serving_capacity::WeightArenaLayout)>,
    pub fixed: Vec<cuteafd_loader::placement::Demand>,
    pub layer_extra: Vec<Vec<cuteafd_loader::placement::LayerExtraDemand>>,
}

pub(super) fn dual_weights(args: &EngineArgs, checkpoint: &cuteafd_loader::plan::Checkpoint,
    cfg: &Qwen4Config, layers: usize, mtp: bool) -> Result<DualWeightPlan> {
    use cuteafd_loader::families::qwen4::{resident::{checkpoint_resident_bytes, Qwen4Representation},
        weight_layout::{dual_arenas, embedding_table_bytes}};
    let selected = Qwen4Representation { fp8_projections: args.fp8_decode, fp8_head: args.mtp_fp8_head };
    let resident = checkpoint_resident_bytes(checkpoint, cfg, layers, mtp, selected)?;
    let embedding = if args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu {
        embedding_table_bytes(checkpoint)? } else { 0 };
    let arenas = dual_arenas(checkpoint, cfg, &qwen4::dual_owners(layers), mtp, selected, embedding)?;
    let fixed = qwen4::dual_projection_load_staging(selected);
    let layer_extra = qwen4::dual_ple_demands(cfg, layers, &resident)?;
    Ok(DualWeightPlan { resident, arenas, fixed, layer_extra })
}

/// Header-only pinned staging of each dual owner's coordinator records: the
/// staged source rectangles the loader records on that GPU (FP8 projection
/// quantization in 128-row windows, the FP8 head and MTP on owner 0, BF16 to
/// FP32 widening), split by the same window rules, bounded by the same
/// `HostStaging::for_steps` the receiving rank applies, plus its job table.
/// The executor validates the recorded ranks against this before offers.
pub(super) fn dual_dense_staging(checkpoint: &cuteafd_loader::plan::Checkpoint, cfg: &Qwen4Config,
    owners: &[usize], mtp: bool, fp8_decode: bool, fp8_head: bool,
    media: &crate::shared::experts::rtx::dense::Records, devices: [i32; 2]) -> Result<[u64; 2]> {
    use anyhow::{ensure, Context};
    use cuteafd_core::DType;
    use crate::shared::experts::rtx::dense::{windows, windows_with_tail, HostStaging, Rect};
    use cuteafd_loader::families::qwen4::Qwen4Attention;
    use super::weights::PREFIX;
    let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .with_context(|| format!("Qwen checkpoint lacks {name}"));
    let rect = |name: &str| -> Result<Rect> {
        let t = tensor(name)?;
        let total = if t.meta.shape.len() >= 2 { t.meta.shape[0] } else { 1 };
        let row_bytes = usize::try_from(t.meta.byte_length)? / total;
        Ok(Rect { name: name.into(), path: checkpoint.snapshot.join(&t.shard), base: t.meta.byte_offset,
            row_bytes, rows: 0..total, cols: 0..row_bytes })
    };
    let mut steps: [Vec<Vec<Rect>>; 2] = [Vec::new(), Vec::new()];
    let mut tables = [0usize; 2];
    let widen = |name: &str, out: &mut Vec<Vec<Rect>>| -> Result<()> {
        if tensor(name)?.meta.dtype == DType::Bf16 {
            for (parts, _) in windows(&[rect(name)?], 1)? { out.push(parts); }
        }
        Ok(())
    };
    let layer = |p: &str, attention: Qwen4Attention, ple: bool, out: &mut Vec<Vec<Rect>>| -> Result<()> {
        let projections: Vec<Vec<String>> = match attention {
            Qwen4Attention::Gdn => {
                let a = |n: &str| format!("{p}.linear_attn.{n}");
                for name in [a("conv1d.weight"), a("A_log"), a("dt_bias")] { widen(&name, out)?; }
                vec![["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"].map(|n| a(&format!("{n}.weight"))).to_vec(),
                    vec![a("out_proj.weight")]]
            }
            Qwen4Attention::Full => {
                let a = |n: &str| format!("{p}.self_attn.{n}");
                vec![vec![a("q_proj.weight"), a("k_proj.weight"), a("v_proj.weight"), a("indexer.index_qk_proj.weight")],
                    vec![a("o_proj.weight")]]
            }
        };
        if fp8_decode {
            for names in projections {
                let rects = names.iter().map(|name| rect(name)).collect::<Result<Vec<_>>>()?;
                for (parts, _) in windows_with_tail(&rects, 128)? { out.push(parts); }
            }
        }
        if ple {
            widen(&format!("{p}.ple.conv1d.weight"), out)?;
            if cuteafd_loader::families::qwen4::weight_layout::ple_scale_dtype(checkpoint, p)?.is_some() {
                widen(&format!("{p}.ple.ple_embedding.ngram_embedding.weight_scale"), out)?;
            }
        }
        Ok(())
    };
    for (index, &owner) in owners.iter().enumerate() {
        ensure!(owner < 2, "Qwen dual owner outside two GPUs");
        layer(&format!("{PREFIX}layers.{index}"), cfg.attention[index], cfg.ple_layers.contains(&index), &mut steps[owner])?;
    }
    // The head and MTP load under owner 0 (model_placed enters ranks[0]).
    if fp8_head {
        for (parts, window) in windows(&[rect("lm_head.weight")?], 1)? {
            tables[0] = tables[0].max(crate::shared::fp8_load_quant::quant_table_bytes(window.len(),
                crate::families::glm5_flash::fp8::Layout::Row128)?);
            steps[0].push(parts);
        }
    }
    if mtp { layer("mtp.layers.0", Qwen4Attention::Full, false, &mut steps[0])?; }
    // A coordinator tower's own recorded (header-derived) widen steps join its GPU.
    ensure!(media.finishes.is_empty() && media.peer_copies.is_empty(), "Qwen media records only operands and steps");
    for step in &media.steps {
        let rank = devices.iter().position(|&device| device == step.device)
            .context("Qwen media step outside the dual group")?;
        steps[rank].push(step.rects.clone());
    }
    let mut out = [0u64; 2];
    for rank in 0..2 {
        out[rank] = u64::try_from(HostStaging::for_steps(steps[rank].iter().map(Vec::as_slice), tables[rank])?.bytes()?)?;
    }
    Ok(out)
}

/// The dual coordinator push group's receive envelope (two GPUs, dense only).
pub(super) fn dual_push_envelope(dense: [u64; 2]) -> Result<cuteafd_loader::push_load::PushHostEnvelope> {
    Ok(cuteafd_loader::push_load::PushHostEnvelope::new(vec![0, 0], dense.to_vec(),
        cuteafd_ffi::push_ring::DEFAULT_SLOTS as u64, cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64)?)
}

#[cfg(test)]
mod tests {
    use cuteafd_loader::serving_capacity::qwen_graphs::*;

    #[test]
    fn exl3_runtime_requires_the_complete_preupload_admitted_window() -> anyhow::Result<()> {
        use crate::shared::experts::layer::ExpertLayer;
        let layers = (0..48).collect::<Vec<_>>();
        let selections = std::iter::once(ExpertLayer::Dspark { stage: 0 })
            .chain(layers.iter().map(|&layer| ExpertLayer::BackboneFull { layer })).collect();
        let admitted = super::Exl3Admission { layers: layers.clone(), peak_bytes: 70 << 30,
            rows: 4096, mtp: true, selections, weights_budget: 69 << 30 };
        let runtime_layers = admitted.selections.iter().filter_map(|selection| match selection {
            ExpertLayer::BackboneFull { layer } => Some(*layer), _ => None,
        }).collect::<Vec<_>>();
        admitted.validate_layers(&runtime_layers)?;
        assert_eq!(runtime_layers, layers);
        assert!(admitted.validate_layers(&runtime_layers[..39]).is_err(),
            "a post-KV free-memory sample must not silently shrink the resident window");
        assert!(admitted.validate_layers(&runtime_layers[1..]).is_err());
        assert_eq!(admitted.peak_bytes, 70 << 30);
        assert_eq!(admitted.weights_budget, 69 << 30);
        Ok(())
    }

    #[test]
    fn local_media_weight_owner_follows_sparse_and_reversed_cuda_ordinals() -> anyhow::Result<()> {
        use cuteafd_loader::placement::ResidentWeights;
        for devices in [[7, 2], [2, 7]] {
            for rank in 0..2 {
                let demand = super::loaded_media_weights(&devices, Some((devices[rank], ResidentWeights(123))))?.unwrap();
                assert_eq!(demand.gpu as usize, rank);
                assert_eq!(demand.bytes, 0, "already-loaded tower storage remains in the baseline");
                assert_eq!(demand.resident_weights, ResidentWeights(123));
            }
            assert!(super::loaded_media_weights(&devices, None)?.is_none());
            assert!(super::loaded_media_weights(&devices, Some((9, ResidentWeights(123))))?.is_none());
            assert!(super::loaded_media_weights(&devices, Some((9, ResidentWeights(u64::MAX)))).is_err());
        }
        Ok(())
    }

    /// The shared Ready/first-request capture returns before sampling when
    /// admission never registered `memory_reconciliation_provenance`, so the
    /// family's `/v1/stats` would have no `delta_vs_measured` or
    /// `graph_reserve_slack`. Register exactly what admission publishes and
    /// read the reconciled device back.
    #[test]
    fn reconciliation_registers_the_identity_the_shared_capture_reads() -> anyhow::Result<()> {
        let sample = |device: i32| cuteafd_loader::placement::RuntimeSample { device,
            total_bytes: 96 << 30, free_bytes: 80 << 30, context_bytes: 1 << 20, module_bytes: 0,
            sms: 188, arch: "sm_120".into(), driver: "580.65".into() };
        let provenance = super::reconciliation_provenance("commit", "false", "toolchain", "native",
            "programs", "layout", &[sample(0), sample(1)], 2, &[vec![0], vec![1]],
            cuteafd_loader::placement::planner::PrefillShape { rows: 2048, waves: 1 });
        assert_eq!(provenance["family"], "qwen4");
        let devices = provenance["devices"].as_array().expect("devices array");
        assert_eq!(devices.len(), 2, "one inventory sample per physical GPU, in rank order");
        assert_eq!(devices[1]["device"].as_i64(), Some(1));
        assert_eq!(provenance["spark_world"], 2);
        assert_eq!(provenance["prefill"]["rows"], 2048);
        let owners = cuteafd_core::memory_policy::DeviceBreakdown { device: "rtx0".into(),
            cap_bytes: 96 << 30, weights_bytes: 64 << 30, workspace_bytes: 8 << 30, kv_bytes: 8 << 30,
            other_bytes: 0, reserve_overridden: false,
            reserve: vec![cuteafd_core::memory_policy::ReserveComponent {
                name: "graph reserve".into(), bytes: 212 << 20 }] };
        use crate::shared::memory_report::reconcile;
        cuteafd_ffi::memory_ledger::fault::set_inputs(serde_json::json!({
            "memory_reconciliation_provenance": provenance, "device.rtx0": owners }));
        reconcile::capture("ready");
        let snapshot = reconcile::snapshot();
        let row = snapshot["ready"]["devices"].as_array().expect("reconciled rows").iter()
            .find(|row| row["device"].as_str() == Some("rtx0")).expect("admitted rtx0 row");
        assert_eq!(row["graph_reserve_bytes"].as_u64(), Some(212 << 20));
        assert!(row["graph_reserve_bytes"].as_u64().is_some_and(|bytes| bytes > 0),
            "the canonical graph reserve row must be charged");
        assert_eq!(row["provenance"]["family"], "qwen4");
        assert!(row.get("delta_vs_measured").is_some());
        assert!(row.get("graph_reserve_slack").is_some());
        Ok(())
    }

    #[test]
    fn host_tables_match_full_destination_extents() -> anyhow::Result<()> {
        for rows in [1usize, 64, 512, 2048, 2752, 4096] {
            for decode in [false, true] {
                let tables = super::StepTableBytes::new(rows, decode, 32768, 8192, 9)?;
                assert_eq!(tables.positions, rows * 8);
                assert_eq!(tables.rope_positions, rows * 12);
                assert_eq!(tables.page_table, if decode { rows } else { 1 } * 32768 * 4);
                assert_eq!(tables.pool_table, if decode { rows } else { 1 } * 8192 * 4);
                assert_eq!(tables.ple_ids, rows * 9 * 8);
            }
        }
        assert!(super::StepTableBytes::new(usize::MAX, false, 1, 1, 1).is_err());
        Ok(())
    }

    #[test]
    fn public_planner_retained_tables_equal_runtime_at_forced_32gib() -> anyhow::Result<()> {
        use cuteafd_loader::plan::{plan, testing, ExpertPlacement, PlanOptions};
        use cuteafd_loader::plan::layout::LayoutOptions;
        use cuteafd_loader::serving_capacity::qwen_graphs::{qwen_pool_pages, QWEN_UNIT_PAGES};
        for mapped_ple in [false, true] {
            let dir = tempfile::tempdir()?;
            let mut config = testing::qwen4_config(4);
            if mapped_ple {
                config["text_config"]["ple_layer_ids"] = serde_json::json!([2]);
                config["text_config"]["ple_embed_dim"] = 2560.into();
                config["text_config"]["ngram_size"] = 3.into();
                config["text_config"]["heads_per_ngram"] = 8.into();
            }
            let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(&config)?;
            let mut tensors = testing::qwen4_coordinator(&config, false);
            if mapped_ple {
                tensors.push(testing::t("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    "F8_E4M3", &[100, 160]));
                tensors.push(testing::t("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale", "BF16", &[1]));
            }
            testing::write_snapshot(dir.path(), &config, &tensors, None);
            // Exercise explicit serial and pipelined metadata owners.
            for (rows, waves) in [(4096, 1), (2048, 2), (4096, 2)] {
                let report = plan(dir.path(), &PlanOptions {
                    placement: ExpertPlacement::Sparks { ranks: 4 },
                    layout: Some(LayoutOptions { rtx_bytes: vec![32 << 30], prefill_rows: rows,
                        prefill_waves: waves, prefill_lanes: waves, context_tokens: 32768, prefix_slots: Some(0),
                        host_cache_bytes: Some(0), host_memory: Some((128 << 30, 128 << 30)),
                        ..Default::default() }), ..Default::default()
                })?;
                assert!(report.solver_placement.is_some(), "{}", cuteafd_loader::plan::render(&report));
                assert_eq!((report.prefill_unit_rows, report.prefill_waves, report.prefill_span_rows),
                    (Some(rows), Some(waves), Some(rows * waves)));
                let pool = usize::try_from(report.memory_layout.as_ref().unwrap().pool_tokens)?;
                let pages = qwen_pool_pages(pool).unwrap().max(QWEN_UNIT_PAGES);
                let runtime = super::HostWorkspaceBytes::new(&cfg, rows as usize, false, pages, pages / QWEN_UNIT_PAGES,
                    if waves > 1 { waves as usize } else { 0 }, true, false, !mapped_ple, 1)?;
                let expected = cuteafd_loader::families::qwen4::upload_shape(rows as usize, false,
                    pages, pages / QWEN_UNIT_PAGES, cfg.ple_rows(), cfg.layers,
                    if waves > 1 { waves as usize } else { 0 }, true, false, !mapped_ple)?;
                assert_eq!(runtime.staging as usize, expected.capacity_bytes()?);
                let decode = super::HostWorkspaceBytes::new(&cfg, super::super::engine::DECODE_ROWS, true,
                    pages, pages / QWEN_UNIT_PAGES, 0, true, false, !mapped_ple, 64)?;
                let planned = report.host_ledger.as_ref().expect("host admission ledger").items.iter()
                    .filter(|item| item.group == "staging/tables-and-sync").map(|item| item.bytes).sum::<u64>();
                // The recipe admits a conservative page-linear bound: it covers
                // the runtime's exact arenas at this pool, within the bound's
                // fixed term plus 1% (estimate, then exact).
                let exact = runtime.staging + decode.staging;
                let bounds = cuteafd_loader::placement::families::qwen4::host::UploadInputs {
                    prefill_rows: rows as usize, waves: waves as usize, spark: true, mtp: false, ple_ids: !mapped_ple }
                    .bounds(&cfg)?.0;
                assert!(planned >= exact, "host source ledger must cover the lazy runtime owners: {planned} < {exact}");
                assert!(planned - exact <= bounds[0].at_first + bounds[1].at_first + exact / 100,
                    "upload bound too loose: planned {planned}, exact {exact}");
                let shape = (waves > 1).then_some(cuteafd_loader::serving_capacity::PrefillShape {
                    unit_rows: rows, waves,
                });
                let runtime = qwen_admission(&QwenAdmissionInputs {
                    cfg: &cfg, layers: 4, mtp: false,
                    kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache::default(),
                    manifest: None, prefill_rows: rows * waves, prefill_shape: shape,
                    slots: 8, mark_bytes: 0,
                    ple: mapped_ple.then_some((160, true)), future_expert_bytes: 0, headroom: 0,
                })?;
                for (_, name, bytes) in runtime.items.iter().filter(|item| item.1.ends_with("step")) {
                    let planned = report.memory_layout.as_ref().unwrap().devices[0].items.iter()
                        .filter(|item| item.group == *name).map(|item| item.bytes).sum::<u64>();
                    assert_eq!(planned, *bytes, "{rows}x{waves}, {name}, mapped PLE {mapped_ple}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn host_workspace_charges_all_lazy_consumers_for_serial_and_spark_owners() -> anyhow::Result<()> {
        use super::HostWorkspaceBytes;
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(
            &cuteafd_loader::plan::testing::qwen4_config(48))?;
        let (pages, pools) = (32768usize, 8192usize);
        for rows in [64usize, 512, 4096] {
            for decode in [false, true] {
                for spark in [false, true] {
                    let host = HostWorkspaceBytes::new(&cfg, rows, decode, pages, pools,
                        if !decode && spark { 3 } else { 0 }, spark, true, false, if decode { 64 } else { 1 })?;
                    let expected = cuteafd_loader::families::qwen4::upload_shape(rows, decode, pages, pools,
                        cfg.ple_rows(), cfg.layers, if !decode && spark { 3 } else { 0 }, spark, true, false)?;
                    assert_eq!(host.staging as usize, expected.capacity_bytes()?);
                    assert_eq!(host.uploads, expected);
                    assert_eq!(host.logits as usize, (if decode { 64 } else { 1 }) * cfg.vocab_size * 4);
                }
            }
        }
        let full = HostWorkspaceBytes::new(&cfg, 4096, false, pages, pools, 3, true, true, false, 8192)?;
        assert_eq!(full.logits, 8192 * cfg.vocab_size as u64 * 4);
        assert!(HostWorkspaceBytes::new(&cfg, usize::MAX, false, pages, pools,
            3, true, true, false, 1).is_err());
        Ok(())
    }


    fn planned(family: &str, backend: cuteafd_loader::plan::spark_executions::SparkBackend, decode: Option<u32>,
        unit: u64, waves: u64) -> cuteafd_core::SparkExecutions {
        let options = cuteafd_loader::plan::layout::LayoutOptions { prefill_unit_rows: Some(unit),
            prefill_waves: waves, ..Default::default() };
        cuteafd_loader::plan::layout::planned_spark_executions(family, backend, decode, &options).unwrap()
    }

    #[test]
    fn planner_and_serve_admit_identical_spark_owners() -> anyhow::Result<()> {
        use cuteafd_loader::plan::spark_executions::SparkBackend;
        for backend in [SparkBackend::Fp8, SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest)] {
            for (waves, unit) in [(1u64, 4096usize), (2, 2048), (2, 4096)] {
                assert_eq!(super::spark_model(backend, unit, waves, false)?.executions,
                    planned("qwen4", backend, None, unit as u64, waves));
            }
        }
        Ok(())
    }

    #[test]
    fn spark_contract_matches_single_link_capacity_and_checkpoint_ingress() -> anyhow::Result<()> {
        use cuteafd_loader::plan::spark_executions::SparkBackend;
        for (rows, waves) in [(64, 1), (2048, 2), (4096, 2)] {
            for bf16 in [false, true] {
                let model = super::spark_model(SparkBackend::Fp8, rows, waves, bf16)?;
                model.validate(cuteafd_core::ExpertGeometry::QWEN4)?;
                let compiled = cuteafd_core::covering_export(&SparkBackend::Fp8.exports(), rows as u32).unwrap();
                assert_eq!((model.capacity(), model.compiled_capacity()), (rows.max(64) as u32, compiled));
                assert_eq!((model.executions.prefill.unit_rows, model.executions.prefill.buffered_units),
                    (rows as u32, waves as u8));
                assert_eq!(model.endpoints.len(), 1);
                assert_eq!(model.endpoints[0].capacity, rows as u32);
                assert_eq!(model.endpoints[0].ingress, if bf16 { cuteafd_core::SparkIngress::Bf16 }
                    else { cuteafd_core::SparkIngress::Fp8K32 });
            }
        }
        Ok(())
    }

    #[test]
    fn dual_exl3_packages_match_shared_manifest_plan_before_cuda() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::testing;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let dir = tempfile::tempdir()?;
        let (tensors, metadata) = testing::qwen4_exl3(1, 4);
        testing::write_snapshot(dir.path(), &testing::qwen4_config(1), &tensors, None);
        testing::write_quantize_config(dir.path(), &metadata);
        let catalog = cuteafd_loader::read_expert_catalog(dir.path())?;
        let mut args = Cli::try_parse_from(["serve", "--snapshot", dir.path().to_str().unwrap(),
            "--native-lib", "/absent/native.so", "--prefill-rows", "16"])?.engine;
        args.native_lib = dir.path().join("libcuteafd_native.so");
        // Qwen admits at least the 64-row decode shape, hence m80 as well.
        let package = crate::shared::experts::exl3::package_name("qwen4", catalog.exl3().unwrap().decoder_tiers());
        for (rank, width) in [384, 256].into_iter().enumerate() {
            for capacity in [1, 16, 80] {
                let path = dir.path().join("exl3").join(&package)
                    .join(format!("rtx-tp2-rank{rank}/m{capacity}"));
                std::fs::create_dir_all(&path)?;
                let manifest = serde_json::json!({"hidden": 2560, "intermediate": width, "experts": 512,
                    "top_k": 10, "capacity": capacity, "decode_capacity": true, "output_dtype": "fp32",
                    "tile": [64, 128, 64, 128], "trellis_lut": {"bytes": 32},
                    "buffers": {"scratch": {"allocation": "scratch", "bytes": capacity * width,
                        "dtype": "f16", "zero_on_create": false},
                        "state": {"allocation": "state", "bytes": 16, "zero_on_create": true}}});
                std::fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&manifest)?)?;
            }
        }
        let plan = super::dual_experts(&args, &catalog, 1)?;
        assert!(plan.wire);
        assert_eq!(plan.costs.len(), 1);
        assert!(plan.costs[0].unwrap().half[0].resident > plan.costs[0].unwrap().half[1].resident);
        assert_eq!(plan.workspace, [80 * 384 + 3 * (32 + 16) + 97 * 2560 * 2 + 64 * 2560 * 4,
            80 * 256 + 3 * (32 + 16) + 97 * 2560 * 2 + 64 * 2560 * 4]);
        std::fs::remove_file(plan.packages[1].join("m80/v41_exl3.json"))?;
        assert!(super::dual_experts(&args, &catalog, 1).is_err());
        Ok(())
    }

    #[test]
    fn large_ple_default_stays_mapped_and_charges_only_rows_and_scale() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::families::qwen4::{resident::Qwen4CheckpointResident, Qwen4Config, Qwen4KvCache};
        use cuteafd_loader::placement::{families::qwen4, Baseline, GpuBudget, Onboard};
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let argv = ["serve", "--snapshot", "/absent/checkpoint", "--native-lib", "/absent/native.so"];
        Cli::try_parse_from(argv)?;
        let mut config = cuteafd_loader::plan::testing::qwen4_config(48);
        config["text_config"]["ple_layer_ids"] = serde_json::json!([24]);
        config["text_config"]["ple_embed_dim"] = serde_json::json!(2560);
        config["text_config"]["ngram_size"] = serde_json::json!(3);
        config["text_config"]["heads_per_ngram"] = serde_json::json!(8);
        let cfg = Qwen4Config::from_hf(&config)?;
        // The BF16 table is about 95.4 GiB; headers alone must not promote it.
        let resident = Qwen4CheckpointResident {
            ple_table_bytes: (954 * (1u64 << 30)).div_ceil(10), ple_row_bytes: 160 * 2,
            ..Default::default()
        };
        let extra = qwen4::dual_ple_demands(&cfg, 48, &resident)?;
        // The loader's own arenas from a header-only fixture at the fixed cut.
        let dir = tempfile::tempdir()?;
        let mut tensors = cuteafd_loader::plan::testing::qwen4_coordinator(&config, false);
        tensors.push(cuteafd_loader::plan::testing::t(
            "model.language_model.layers.23.ple.ple_embedding.ngram_embedding.shard_0.weight", "BF16", &[100, 160]));
        cuteafd_loader::plan::testing::write_snapshot(dir.path(), &config, &tensors, None);
        let arenas = cuteafd_loader::families::qwen4::weight_layout::dual_arenas(
            &cuteafd_loader::plan::Checkpoint::open(dir.path())?, &cfg, &qwen4::dual_owners(48), false,
            Default::default(), 0)?;
        let experts = vec![None; 48];
        // Even enough capacity for the full table must retain mapped residency.
        for capacity in [96u64 << 30, 192u64 << 30] {
            let onboard = Onboard::Auto;
            {
                let mut inputs = qwen4::QwenDualInputs {
        experts: None,        blas_head: cuteafd_loader::placement::planner::serving::BlasHead::Native,
        resident_weight_limit: None, context: cuteafd_loader::placement::planner::Context::pinned(131_072),
                    owner_manifests: [None; 2],
                    admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: 48, mtp: false,
                        kv_format: Qwen4KvCache::Bf16, manifest: None, prefill_rows: 4096,
                        slots: 16, mark_bytes: 0,
                        ple: Some((resident.ple_row_bytes, false)), future_expert_bytes: 0, headroom: 3 << 30 },
                    gpus: [GpuBudget { capacity_bytes: capacity, reserve_bytes: 3 << 30,
                        baseline: Baseline::Planned { context_bytes: 0, loaded_bytes: 0 } }; 2],
                    arenas: &arenas, layer_experts: &experts, layer_extra: &extra,
                    fixed: Vec::new(), tp2_workspace: [0; 2], pending_code_bytes: [0; 2], mark_slots: 18,
                    max_context: 131072, requested_pool: Some(2 << 20), spark_ranks: 0, onboard,
                    host: qwen4::QwenHost::ample(&cfg, 2, 4096)?, encoders: Default::default(),
                };
                let mapped = qwen4::dual_placement(&inputs)?;
                assert!(mapped.movables.is_empty());
                inputs.admission.ple = None;
                let direct = qwen4::dual_placement(&inputs)?;
                for (group, rows) in [("decode step", 64), ("prefill step", 4096)] {
                    for gpu in 0..2 {
                        let bytes = |p: &cuteafd_loader::placement::Placement| p.items[gpu].iter()
                            .find(|item| item.group == group).unwrap().bytes;
                        assert_eq!(bytes(&mapped) - bytes(&direct),
                            rows * cfg.ple_rows() as u64 * (8 + resident.ple_row_bytes));
                    }
                }
                assert_eq!(mapped.pool_tokens, 2 << 20);
                assert_eq!(mapped.items.iter().flatten().filter(|i| i.category == Category::Tables)
                    .map(|i| (i.group.as_str(), i.bytes)).collect::<Vec<_>>(), Vec::<(&str, u64)>::new());
                assert!(!mapped.items.iter().flatten().any(|i| i.bytes == resident.ple_table_bytes));
            }
        }
        for (option, value) in [("--table-placement", "mapped"), ("--ple", "mapped"), ("--table-cache-mib", "0"),
            ("--table-threads", "16"), ("--table-prefetch", "true")] {
            assert!(Cli::try_parse_from(argv.into_iter().chain([option, value])).is_err());
        }
        Ok(())
    }

    #[test]
    fn dual_weight_headers_match_selected_operands_before_cuda() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::testing;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let dir = tempfile::tempdir()?;
        let mut config = testing::qwen4_config(2);
        config["text_config"]["ple_layer_ids"] = serde_json::json!([2]);
        config["text_config"]["ple_embed_dim"] = serde_json::json!(2560);
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(&config)?;
        // The loader's real coordinator source tensors, plus the mapped PLE table shard.
        let mut tensors = testing::qwen4_coordinator(&config, false);
        tensors.push(testing::t("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
            "F8_E4M3", &[100, 160]));
        tensors.push(testing::t("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale", "BF16", &[1]));
        testing::write_snapshot(dir.path(), &config, &tensors, None);
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(dir.path())?;
        let mut args = Cli::try_parse_from(["serve", "--snapshot", dir.path().to_str().unwrap(),
            "--native-lib", "/absent/native.so"])?.engine;
        for projections in [false, true] {
            for host in [false, true] {
                args.fp8_decode = projections;
                args.token_io.embed_placement = if host { crate::shared::token_io::EmbedPlacement::Host }
                    else { crate::shared::token_io::EmbedPlacement::Gpu };
                let plan = super::dual_weights(&args, &checkpoint, &cfg, 2, false)?;
                assert_eq!(plan.resident.layer_bytes.len(), 2);
                assert_eq!(plan.resident.target_bytes, plan.resident.entry_bytes
                    + plan.resident.layer_bytes.iter().sum::<u64>());
                let staging = if projections { cuteafd_loader::shared::dense_read::STEP_BYTES as u64 } else { 0 };
                // Owner0's head, mixer and embedding are arena views now: only staging stays fixed.
                assert_eq!(plan.fixed.iter().map(|d| d.bytes).sum::<u64>(), 2 * staging);
                // The arenas hold exactly the selected payload plus their own padding.
                let payload = plan.resident.target_bytes + plan.resident.head_bytes
                    + if host { 0 } else { plan.resident.embedding_bytes };
                assert_eq!(plan.arenas.iter().map(|(_, l)| l.payload_bytes() as u64).sum::<u64>(), payload);
                assert_eq!(plan.arenas.iter().map(|(_, l)| l.bytes() as u64).sum::<u64>(),
                    payload + plan.arenas.iter().map(|(_, l)| l.padding_bytes() as u64).sum::<u64>());
                assert_eq!(plan.arenas.iter().map(|(r, l)| (*r, l.id().to_string())).collect::<Vec<_>>(),
                    vec![(0, "qwen4/target/rank0".into()), (1, "qwen4/target/rank1".into())]);
                let load = plan.fixed.iter().filter(|d| d.category == cuteafd_core::memory_layout::Category::Staging)
                    .collect::<Vec<_>>();
                assert_eq!(load.len(), if projections { 2 } else { 0 });
                for (gpu, demand) in load.iter().enumerate() {
                    assert_eq!((demand.gpu, demand.bytes), (gpu as u8, staging));
                    assert_eq!(demand.group, "FP8 projection load staging");
                }
                assert!(plan.fixed.iter().filter(|d| d.category != cuteafd_core::memory_layout::Category::Staging)
                    .all(|d| d.gpu == 0));
                let plan = super::dual_weights(&args, &checkpoint, &cfg, 2, false)?;
                assert_eq!(plan.layer_extra.len(), 2);
                assert!(plan.layer_extra.iter().all(Vec::is_empty));
                let scales = plan.arenas.iter().flat_map(|(owner, arena)| arena.views().iter()
                    .filter(|(name, _)| name.ends_with(".ple.scale"))
                    .map(move |(name, view)| (*owner, arena.id(), name.as_str(), view.bytes(), view.offset())))
                    .collect::<Vec<_>>();
                assert_eq!(scales.len(), 1);
                let (owner, arena, name, bytes, offset) = scales[0];
                assert_eq!((owner, arena, name, bytes), (1, "qwen4/target/rank1",
                    "model.language_model.layers.1.ple.scale", 256));
                assert_eq!(offset % 256, 0);
            }
        }
        Ok(())
    }

    #[test]
    fn public_dual_planner_equals_runtime_before_allocation() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::{plan, testing, ExpertPlacement, PlanOptions};
        use cuteafd_loader::plan::layout::LayoutOptions;
        use cuteafd_loader::placement::{families::qwen4, Baseline, GpuBudget, Onboard};
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)] engine: super::super::EngineArgs,
            #[command(flatten)] prefix: crate::shared::prefix::PrefixArgs,
        }
        for mtp in [false, true] {
        let dir = tempfile::tempdir()?;
        let (mut tensors, mut metadata) = testing::qwen4_exl3(4, 4);
        let mut config = testing::qwen4_config(4);
        if mtp { config["text_config"]["mtp_num_hidden_layers"] = 1.into(); }
        // The loader's real coordinator source tensors (and, with MTP, the
        // draft layer's coordinator set); routed experts stay EXL3 below.
        tensors.extend(testing::qwen4_coordinator(&config, mtp));
        if mtp {
            let (draft_tensors, draft_metadata) = testing::qwen4_exl3(1, 4);
            for (name, dtype, shape) in draft_tensors {
                tensors.push((name.replace("model.language_model.layers.0", "mtp.layers.0"), dtype, shape));
            }
            for (name, storage) in draft_metadata["tensor_storage"].as_object().unwrap() {
                let name = name.replace("model.language_model.layers.0", "mtp.layers.0");
                let mut storage = storage.clone();
                let stored = storage["stored_tensors"].as_object().unwrap().iter().map(|(key, value)|
                    (key.replace("model.language_model.layers.0", "mtp.layers.0"), value.clone())).collect();
                storage["stored_tensors"] = serde_json::Value::Object(stored);
                metadata["tensor_storage"][name] = storage;
            }
        }
        testing::write_snapshot(dir.path(), &config, &tensors, None);
        testing::write_quantize_config(dir.path(), &metadata);
        let share = dir.path().join("share");
        std::fs::create_dir_all(&share)?;
        let manifest_path = share.join("PROGRAMS.json");
        let manifest = serde_json::json!({"programs": [{"name": "qwen4_sparse_gqa_kv_fp8_m64",
            "scratch_bytes_at_capacity": {"scratch": 8192}, "geometry": {
                "attention_plan_capacity_rows": 64, "attention_plan_scratch": [
                    {"rows_cap": 1, "scratch": 4096}, {"rows_cap": 8, "scratch": 2048},
                    {"rows_cap": 64, "scratch": 1024}]}}]});
        let mut manifest = manifest;
        let recipe = manifest["programs"][0].clone();
        manifest["programs"][0]["card_plans"] = serde_json::json!([
            {"sm_count": 170, "scratch_bytes_at_capacity": {"scratch": 16384},
                "geometry": {"attention_plan_capacity_rows": 64, "attention_plan_scratch": [
                    {"rows_cap": 1, "scratch": 8192}, {"rows_cap": 8, "scratch": 4096},
                    {"rows_cap": 64, "scratch": 2048}]}},
            {"sm_count": 188, "scratch_bytes_at_capacity": recipe["scratch_bytes_at_capacity"],
                "geometry": recipe["geometry"]}]);
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
        let lib = dir.path().join("lib/libcuteafd_native.so");
        let catalog = cuteafd_loader::read_expert_catalog(dir.path())?;
        let packages = qwen4::dual_packages(&catalog, &lib)?;
        for (rank, width) in [384, 256].into_iter().enumerate() {
            for capacity in [1, 16, 80] {
                let path = packages[rank].join(format!("m{capacity}"));
                std::fs::create_dir_all(&path)?;
                let m = serde_json::json!({"hidden": 2560, "intermediate": width, "experts": 512,
                    "top_k": 10, "capacity": capacity, "decode_capacity": true, "output_dtype": "fp32",
                    "tile": [64, 128, 64, 128], "trellis_lut": {"bytes": 32},
                    "buffers": {"scratch": {"allocation": "scratch", "bytes": capacity * width,
                        "dtype": "f16", "zero_on_create": false}}});
                std::fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&m)?)?;
            }
        }
        if mtp {
            let draft_package = packages[0].parent().unwrap().join("rtx-tp1");
            for capacity in [1, 16, 80] {
                let path = draft_package.join(format!("m{capacity}"));
                std::fs::create_dir_all(&path)?;
                let m = serde_json::json!({"hidden": 2560, "intermediate": 640, "experts": 512,
                    "top_k": 10, "capacity": capacity, "decode_capacity": true, "tile": [64, 128, 64, 128],
                    "trellis_lut": {"bytes": 32},
                    "buffers": {"scratch": {"allocation": "scratch", "bytes": capacity * 640,
                        "dtype": "f16", "zero_on_create": false}}});
                std::fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&m)?)?;
            }
        }
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(dir.path())?;
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::read(dir.path())?;
        let cli = Cli::try_parse_from(["serve", "--snapshot", dir.path().to_str().unwrap(),
            "--native-lib", lib.to_str().unwrap(), "--local-experts", "--split-device", "1",
            "--prefill-rows", "64", "--max-context-tokens", "131072", "--pool-tokens", "0",
            "--host-cache-bytes", "auto"])?;
        let mut args = cli.engine;
        let mut prefix = cli.prefix;
        prefix.resolve_entries(8);
        args.planner_host_prefix = Some((prefix, 8));
        args.mtp = if mtp { 3 } else { 0 };
        args.planner_prefix_bytes = Some(0);
        for (host, fp8_projections, fp8_head, attention_rows, capacities) in [
            (false, false, true, None, [101_973_491_712; 2]),
            (true, false, true, None, [32 << 30, 101_973_491_712]),
            (false, true, true, None, [101_973_491_712, 32 << 30]),
            (false, false, false, None, [101_973_491_712; 2]),
            (false, false, true, Some(64), [101_973_491_712; 2]),
        ] {
            args.attention_plan_rows = attention_rows;
            let reduction = super::super::attention_plan(&args, mtp)?;
            let manifests = qwen4::dual_program_manifests(&manifest,
                capacities.map(|bytes| cuteafd_loader::placement::ArchContext::for_device("sm_120", bytes).sms), reduction)?;
            args.fp8_decode = fp8_projections;
            args.mtp_fp8_head = fp8_head;
            args.token_io.embed_placement = if host { crate::shared::token_io::EmbedPlacement::Host }
                else { crate::shared::token_io::EmbedPlacement::Gpu };
            let report = plan(dir.path(), &PlanOptions { placement: ExpertPlacement::Local,
                layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: capacities.to_vec(),
                    context_tokens: 131072, prefill_rows: 64, concurrency: 8, state_slots: Some(8),
                    prefix_slots: Some(0), native_mtp_layers: args.mtp, host_embedding: host,
                    qwen_attention_plan_rows: attention_rows,
                    qwen_representation: cuteafd_loader::families::qwen4::resident::Qwen4Representation { fp8_projections, fp8_head },
                    workspace_manifest: Some(manifest_path.clone()), ..Default::default() }), ..Default::default() })?;
            assert!(report.placement_supported, "{:?}", report.hints);
            assert_eq!(report.attention_placement, cuteafd_loader::placement::AttentionPlacement::Layers);
            let layout = report.memory_layout.unwrap();
            let weights = super::dual_weights(&args, &checkpoint, &cfg, 4, mtp)?;
            let experts = super::dual_experts(&args, &catalog, 4)?;
            let mut fixed = weights.fixed;
            fixed.extend(experts.transport);
            if mtp { fixed.extend(qwen4::dual_draft_demands(&catalog, &lib, args.prefill_rows as u64)?); }
            let admission = super::inputs(&args, &cfg, 4, mtp, Some(&manifests[0]), None, 0)?;
            let host_embedding = if host {
                u64::try_from(cuteafd_loader::families::qwen4::weight_layout::embedding_table_bytes(&checkpoint)?)?
            } else { 0 };
            let runtime = qwen4::dual_placement(&qwen4::QwenDualInputs {
        experts: None,        blas_head: crate::shared::target_head::TargetHead::configured().blas_head(!args.mtp_fp8_head),
        resident_weight_limit: None, context: cuteafd_loader::placement::planner::Context::pinned(131_072), owner_manifests: manifests.each_ref().map(Some), admission,
                gpus: [0, 1].map(|rank| GpuBudget { capacity_bytes: capacities[rank],
                    reserve_bytes: 3 << 30, baseline: Baseline::Measured { free_bytes: capacities[rank]
                        - layout.devices[rank].items.iter().filter(|item| item.group == "context+modules")
                            .map(|item| item.bytes).sum::<u64>() } }),
                arenas: &weights.arenas, layer_experts: &experts.costs,
                layer_extra: &weights.layer_extra, fixed, tp2_workspace: experts.workspace,
                pending_code_bytes: [0; 2], mark_slots: 0, max_context: 131072, requested_pool: None,
                spark_ranks: 0, onboard: Onboard::Auto,
                host: super::host_owners_with_memory(&args, &cfg, 4, mtp, 2, 0, false, None,
                    host_embedding, Vec::new(), (128 << 30, 128 << 30))?, encoders: Default::default() })?;
            assert_eq!(runtime.pool_tokens, if capacities.iter().any(|&bytes| bytes <= 32 << 30) { 1 << 20 } else { 2 << 20 });
            assert_eq!(runtime.pool_tokens, layout.pool_tokens);
            assert_eq!(runtime.tp2.unwrap().layers, 4);
            assert!(layout.notes.contains(&format!("placement: {}", runtime.summary())));
            for rank in 0..2 {
                // TP2 samples free memory after the resident upload: the plan's
                // `context+modules` bytes are what the runtime sees already
                // occupied, and it charges them as one named owner at exactly
                // those bytes. Every other runtime item is a plan item.
                let sampled = layout.devices[rank].items.iter().filter(|item| item.group == "context+modules")
                    .map(|item| item.bytes).sum::<u64>();
                let planned = layout.devices[rank].items.iter().filter(|item| item.group != "context+modules")
                    .cloned().collect::<Vec<_>>();
                let mut actual = runtime.items[rank].clone();
                let index = actual.iter().position(|item| item.group.starts_with("sampled baseline: Qwen context and selected modules"))
                    .unwrap_or_else(|| panic!("rank {rank}: no measured-occupancy owner in {actual:?}"));
                let owner = actual.remove(index);
                assert_eq!(owner.bytes, sampled,
                    "rank {rank}: the owner charges the sampled bytes, not a second estimate");
                assert_eq!(actual, planned, "rank {rank}, MTP={mtp}, attention={attention_rows:?}");
            }
        }
        std::fs::remove_file(packages[1].join("m80/v41_exl3.json"))?;
        assert!(super::dual_experts(&args, &catalog, 4).is_err());
        }
        Ok(())
    }

    #[test]
    fn planner_equals_runtime_qwen4() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::{plan, testing, ExpertPlacement, PlanOptions};
        use cuteafd_loader::plan::layout::LayoutOptions;
        use cuteafd_loader::families::qwen4::Qwen4KvCache;
        use cuteafd_loader::placement::{families::qwen4, Baseline, ResidentWeights};
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let snapshot = tempfile::tempdir()?;
        // One full GDN/GQA cycle, with the same native target and expert owners
        // the live single-GPU executor retains before pool admission.
        let config = testing::qwen4_config(4);
        let (mut tensors, metadata) = testing::qwen4_exl3(4, 4);
        tensors.extend(testing::qwen4_coordinator(&config, false));
        testing::write_snapshot(snapshot.path(), &config, &tensors, None);
        testing::write_quantize_config(snapshot.path(), &metadata);
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::read(snapshot.path())?;
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(snapshot.path())?;
        let catalog = cuteafd_loader::read_expert_catalog(snapshot.path())?;
        for (capacity, kv_format) in [32_u64 << 30, 101_973_491_712].into_iter().flat_map(|capacity|
            [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8].map(|format| (capacity, format))) {
            for (pool, concurrency) in [None, Some(32768)].into_iter()
                .flat_map(|pool| [1, 16, 17, 32, 64].map(|c| (pool, c))) {
                let context = pool.unwrap_or(131072);
                let report = plan(snapshot.path(), &PlanOptions {
                    placement: ExpertPlacement::Local,
                    layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![capacity], context_tokens: context,
                        pool_tokens: pool, qwen_kv: kv_format, concurrency, state_slots: Some(concurrency), prefix_slots: Some(0),
                        ..Default::default() }), ..Default::default()
                })?;
                assert!(report.placement_supported, "{}", cuteafd_loader::plan::render(&report));
                let layout = report.memory_layout.unwrap();
                let device = &layout.devices[0];
                assert!(device.items.iter().any(|i| i.group == "target incl PLE projections" && i.bytes > 0));
                let baseline = device.items.iter().filter(|i| !matches!(i.group.as_str(), "records" | "state" |
                    "marks" | "startup graphs" | "decode step" | "prefill step"))
                    .map(|i| i.bytes).sum::<u64>();
                let mut args = Cli::try_parse_from(["serve", "--snapshot", snapshot.path().to_str().unwrap(),
                    "--native-lib", "/nonexistent/lib.so", "--max-context-tokens", &context.to_string(),
                    "--slots", &concurrency.to_string(),
                    "--pool-tokens", &pool.unwrap_or(0).to_string(), "--kv-cache", &kv_format.to_string()])?.engine;
                args.planner_prefix_bytes = Some(0);
                let loaded_weights = super::single_weights(&args, &checkpoint, &cfg, cfg.layers, false)?
                    .checked_add(qwen4::single_expert_weights(&catalog, cfg.layers, true, false)?)?;
                let inputs = super::inputs(&args, &cfg, cfg.layers, false, None, None, 0)?;
                let planned = qwen4::QwenInputs {
        experts: None, onboard: cuteafd_loader::placement::Onboard::Auto,                    arenas: vec![], expert_arenas: vec![], preallocation: false, future: vec![],
        blas_head: crate::shared::target_head::TargetHead::configured().blas_head(!args.mtp_fp8_head),
                    loaded_weights, resident_weight_limit: Some(ResidentWeights::DEFAULT_LIMIT),
                    spark_reply_format: cuteafd_core::SparkReplyFormat::Bf16, context: cuteafd_loader::placement::planner::Context::pinned(context), admission: inputs, capacity_bytes: capacity,
                    baseline: Baseline::Measured { free_bytes: capacity - baseline }, pending_code_bytes: 0,
                    max_context: context, requested_pool: pool, spark_ranks: 0,

                    host: qwen4::QwenHost::ample(&cfg, 1, 4096)?, encoders: Default::default() };
                // The graph reserve is one owner at the resolved capture shape,
                // charged once before KV sizing.
                let placement = qwen4::placement(&planned)?;
                assert_eq!(placement.pool_tokens, layout.pool_tokens);
                assert!(device.items.iter().any(|item| item.group == "graph reserve: qwen4.graphs.0"),
                    "the plan charges the graph reserve");
                assert!(!device.items.iter().any(|item| item.group == "startup graphs"),
                    "the declared startup-graph owner is gone");
                // The single-flavor sample is taken after the resident upload,
                // so the whole measured occupancy is one named owner at the
                // bytes the plan names for them; host embeddings are not mapped
                // in this fixture (`QwenHost::ample` sets none). Every other
                // runtime item is the plan's own item.
                let mut measured = Vec::new();
                for item in &placement.items[0] {
                    if item.group.starts_with("sampled baseline: Qwen context and selected modules") {
                        measured.push(item);
                    } else {
                        assert!(device.items.contains(item), "runtime item absent in plan: {item:?}");
                    }
                }
                assert_eq!(measured.len(), 1, "the sampled occupancy is one named owner");
                assert_eq!(measured[0].bytes, baseline,
                    "the owner charges the sampled bytes, not a second estimate");
            }
        }
        Ok(())
    }

    #[test]
    fn selected_local_source_ignores_configured_peers_for_admission() -> anyhow::Result<()> {
        use super::{placement_inputs, ExpertSource};
        use clap::Parser;
        use cuteafd_loader::placement::families::qwen4;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(
            &cuteafd_loader::plan::testing::qwen4_config(48))?;
        let local = Cli::try_parse_from(["serve", "--snapshot", "/absent/checkpoint", "--native-lib", "/absent/native.so",
            "--local-experts", "--max-context-tokens", "32768"])?.engine;
        let mut with_peers = local.clone();
        with_peers.peers = Some("127.0.0.1:1234,127.0.0.1:1235".into());
        let input = |args, source, pending| -> anyhow::Result<_> {
            Ok(placement_inputs(args, super::inputs(args, &cfg, 48, false, None, None, 0)?,
                source, (96 << 30, 96 << 30), pending, cuteafd_loader::placement::planner::Context::pinned(args.max_context as u64), qwen4::QwenHost::ample(&cfg, 1, 4096)?))
        };
        for (source, key) in [(ExpertSource::LocalFp8, "fp8"), (ExpertSource::LocalExl3, "exl3")] {
            assert_eq!(source.code_key(), key);
            assert_eq!(source.spark_ranks(), 0);
            let pending = cuteafd_loader::placement::inventory::loaded_code_allowance(
                "qwen4", source.code_key(), false, 0,
                cuteafd_loader::placement::ArchContext::for_device("sm_120", 96 << 30), None).pending(0, 0);
            let plain = input(&local, source, pending)?;
            let peers = input(&with_peers, source, pending)?;
            assert_eq!(peers.pending_code_bytes, pending);
            assert_eq!(peers.spark_ranks, 0);
            assert_eq!(with_peers.resolved_prefill()?.waves, 1);
            assert_eq!(with_peers.resolved_prefill()?, local.resolved_prefill()?);
            // Configured peers never reach admission; the selected source does.
            assert_eq!(qwen4::placement(&plain)?, qwen4::placement(&peers)?);
        }
        let spark = ExpertSource::Spark { ranks: 4 };
        assert_eq!(spark.code_key(), "none");
        // The selected transport's actual world, not the configured two peers.
        assert_eq!(input(&with_peers, spark, 0)?.spark_ranks, 4);
        // Runtime's measured baseline already includes the selected transport,
        // so the header-only Spark return planes are not charged a second time.
        let spark_inputs = input(&with_peers, spark, 0)?;
        assert!(matches!(spark_inputs.baseline, cuteafd_loader::placement::Baseline::Measured { .. }));
        assert!(qwen4::placement(&spark_inputs)?.items[0].iter().all(|item| item.group != "Spark intake planes"));
        assert!(qwen4::placement(&input(&local, ExpertSource::LocalExl3, 0)?)?.items[0].iter()
            .all(|item| item.group != "Spark intake planes"));
        assert_eq!(ExpertSource::selected(None)?, ExpertSource::None);
        let shared = super::super::engine::Experts::SharedOnly;
        assert_eq!(ExpertSource::selected(Some(&shared))?, ExpertSource::None);
        Ok(())
    }

    #[test]
    fn spark_layouts_admit_the_separate_mtp_logits_landing() -> anyhow::Result<()> {
        use clap::Parser;
        use super::super::engine::validate_admitted_host;
        use super::HostWorkspaceBytes;
        #[derive(Parser)]
        struct Cli { #[command(flatten)] engine: super::super::EngineArgs }
        let mut cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(
            &cuteafd_loader::plan::testing::qwen4_config(4))?;
        cfg.mtp_layers = 1;
        // Two requested RTX cards with Sparks still use one attention owner.
        for requested_gpus in [1, 2] {
            for sparks in [0, 1, 4] {
                for mtp in [false, true] {
                    let mut args = Cli::parse_from(["test", "--snapshot", ".", "--native-lib", "native.so"]).engine;
                    args.prefill_rows = if sparks > 0 { 2752 } else { 4096 };
                    args.prefill_waves = if sparks > 0 { 3 } else { 1 };
                    args.local_experts = sparks == 0;
                    args.peers = (sparks > 0).then(|| (0..sparks).map(|rank| format!("host{rank}:9000")).collect::<Vec<_>>().join(","));
                    let owners = if sparks == 0 { requested_gpus } else { 1 };
                    let host = super::host_owners_with_memory(&args, &cfg, 4, mtp,
                        owners, sparks, false, None, 0, Vec::new(), (1 << 40, 1 << 40))?;
                    let expected = if sparks > 0 && mtp { cfg.vocab_size as u64 * 4 } else { 0 };
                    assert_eq!(host.mtp_pinned_logits, expected);
                    assert_eq!(host.pinned_logits, owners as u64 * 65 * cfg.vocab_size as u64 * 4);
                    if expected > 0 {
                        let exact = HostWorkspaceBytes::mtp(&cfg, 2752, 16, 1, 3)?;
                        let tables = [("qwen4/owner-0/mtp-prefill/pinned-logits".into(), exact.logits)];
                        validate_admitted_host(&host, &tables, 0, 1)?;
                        let over = [(tables[0].0.clone(), exact.logits + 1)];
                        assert!(validate_admitted_host(&host, &over, 0, 1).is_err());
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn live_host_owners_must_fit_what_the_recipe_admitted() -> anyhow::Result<()> {
        use super::super::engine::validate_admitted_host;
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(&cuteafd_loader::plan::testing::qwen4_config(4))?;
        let mut host = cuteafd_loader::placement::families::qwen4::QwenHost::ample(&cfg, 1, 4096)?;
        host.pinned_logits = 1000;
        host.prefix_quota = 1 << 30;
        let units = 2048;
        let [prefill, decode] = host.uploads[0];
        let exact = [("qwen4/owner-0/prefill/staging".to_string(), prefill.at(units).unwrap()),
            ("qwen4/owner-0/decode/staging".to_string(), decode.at(units).unwrap()),
            ("qwen4/owner-0/decode/pinned-logits".to_string(), 600),
            ("qwen4/owner-0/prefill/pinned-logits".to_string(), 400)];
        validate_admitted_host(&host, &exact, 1 << 30, units)?;
        let mut over = exact.to_vec();
        over[0].1 += 1;
        assert!(validate_admitted_host(&host, &over, 0, units).unwrap_err().to_string().contains("prefill/staging"));
        let mut logits = exact.to_vec();
        logits[2].1 += 1;
        assert!(validate_admitted_host(&host, &logits, 0, units).unwrap_err().to_string().contains("pinned logits"));
        assert!(validate_admitted_host(&host, &exact, (1 << 30) + 1, units).unwrap_err().to_string().contains("host prefix"));
        // A second owner or an unadmitted PLE owner refuses rather than passing silently.
        let peer = [("qwen4/owner-1/decode/staging".to_string(), 1)];
        assert!(validate_admitted_host(&host, &peer, 0, units).is_err());
        let ple = [("mapped-table/row-cache".to_string(), 1)];
        assert!(validate_admitted_host(&host, &ple, 0, units).is_err());
        Ok(())
    }

}

#[cfg(test)]
mod dual_push_tests {
    use super::*;
    use anyhow::Context;

    /// The header-only dual staging charge equals each GPU's recorded
    /// `HostStaging` (the bound the receiving rank enforces) on the installed
    /// checkpoint, for every projection/head representation: a loader change
    /// that adds, moves or reshapes a staged source fails here, not at serving.
    #[test]
    #[ignore = "requires CUTEAFD_PUSH_QWEN4 (installed Qwen checkpoint) and an NVMe fixture directory"]
    fn installed_dual_staging_charge_equals_recorded_ranks() -> anyhow::Result<()> {
        use crate::shared::{experts::rtx::dense::{DenseLoad, RankLayout}, memory::device::Device};
        let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_QWEN4").map(std::path::PathBuf::from) else { return Ok(()) };
        let root = std::path::PathBuf::from(std::env::var_os("HOME").context("HOME")?)
            .join(".cache/cuteafd/builds/qwen-dual-push-fixture");
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_in(&root)?;
        let library = fixture.load()?;
        fixture.extent_only(&library)?;
        fixture.quiet_events(&library, true)?;
        for device in 0..2 { fixture.configure_memory(&library, device, 96usize << 30)?; }
        let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&snapshot, true, true)?;
        let cfg = Qwen4Config::read(&snapshot)?;
        let owners = qwen4::dual_owners(cfg.layers);
        // The bundled tower's contribution (header-only, on GPU1 as the smoke places it).
        let spec = crate::shared::vision::TowerSpec::qwen(&snapshot, cuteafd_loader::media::QWEN_MAX_IMAGE_TOKENS)?;
        let tower = crate::shared::vision::push::tests::vision_records(1, &spec)?;
        anyhow::ensure!(!tower.steps.is_empty(), "the Qwen tower widens BF16 vectors");
        let mut arms = 0;
        for (fp8_decode, fp8_head) in [(false, false), (true, false), (false, true), (true, true)] {
            let load = std::rc::Rc::new(DenseLoad::push(&library));
            let loader = super::super::weights::Qwen4Loader { library: &library, checkpoint: &checkpoint, fp8_decode,
                fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax, stream: std::ptr::null_mut(), dense: load.clone(), arenas: None };
            let ranks: Vec<_> = (0..2).map(|id| (Device { library: &library, id }, std::ptr::null_mut())).collect();
            let model = loader.model_placed(&cfg, &owners, &ranks, true, fp8_head)?;
            let mut records = load.sink.as_ref().context("push sink")?.take_all();
            records.extend(tower.clone());
            let charged = dual_dense_staging(&checkpoint, &cfg, &owners, true, fp8_decode, fp8_head,
                &tower, [0, 1])?;
            for rank in 0..2 {
                let layout = RankLayout::new(records.operands.clone(), records.steps.clone(), rank as i32)?;
                let actual = layout.host_staging.bytes()? as u64;
                eprintln!("fp8_decode={fp8_decode} fp8_head={fp8_head} GPU{rank}: {} operands, {} steps, \
                    recorded staging {actual} B, charged {} B", layout.operands.len(), layout.steps.len(), charged[rank]);
                anyhow::ensure!(actual == charged[rank],
                    "GPU{rank} recorded staging {actual} B differs from the header charge {} B", charged[rank]);
            }
            let envelope = dual_push_envelope(charged)?;
            let actual = crate::shared::experts::rtx::push::push_host_envelope(&[None, None],
                &[0, 1].map(|rank| RankLayout::new(records.operands.clone(), records.steps.clone(), rank)
                    .map(|layout| layout.host_staging)).into_iter().collect::<anyhow::Result<Vec<_>>>()?)?;
            anyhow::ensure!(envelope == actual, "dual envelope differs from the recorded ranks");
            drop(model);
            arms += 1;
        }
        anyhow::ensure!(arms == 4, "every representation arm ran");
        Ok(())
    }

}
