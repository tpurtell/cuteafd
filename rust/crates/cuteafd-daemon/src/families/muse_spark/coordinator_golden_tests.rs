//! Muse Spark's max push group from the installed checkpoint on the CPU
//! native fixture (CUTEAFD_PUSH_MUSE): the six Spark ranks' EXL3 K4/K5
//! balanced TP6 residencies and the two coordinator ranks' dense records for
//! every layer and the token embedding. Placement is admitted from arm A's
//! measured pre-weight inventory through the real startup solve, so the dump
//! is the group Muse max actually submits: same layer homes, same zero
//! onboarded RTX expert layers at the default pool, six Sparks, TP6.
//!
//! CUTEAFD_PUSH_PLAN_DUMP names the `{plan, file_sizes, arenas}` JSON that
//! SparkNest replays offline; the `rank-bytes.json` file beside it splits each
//! Spark rank's per-layer residency by projection and tensor kind.
use crate::shared::memory::device::Device;
use anyhow::{Context, Result};

/// The installed Muse Spark checkpoint, when this dump is requested.
pub(crate) fn snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("CUTEAFD_PUSH_MUSE").map(std::path::PathBuf::from)
}

/// The whole Muse Spark max push group's plan from the installed headers, for
/// offline replay of SparkNest's grant schedule: six Spark ranks' EXL3 TP6
/// balanced residencies of layers 0..62, then the two coordinator GPU ranks'
/// dense records (all layers' attention/MLP operands and the token
/// embedding), recorded through the native fixture with no GPU.
///
/// CUTEAFD_PUSH_MUSE_SHARE is the image's share directory (`PROGRAMS.json`,
/// with `../lib/exl3/exl3-muse-k45` beside it); CUTEAFD_PUSH_MUSE_DRAFT
/// optionally adds the DFlash2 drafter's charge, as arm A served it.
#[test]
#[ignore = "requires CUTEAFD_PUSH_MUSE, CUTEAFD_PUSH_MUSE_SHARE, CUTEAFD_PUSH_PLAN_DUMP and an NVMe fixture directory"]
fn dump_max_group_plan_from_installed_headers() -> Result<()> {
    use crate::families::muse_spark::weights::{load_rank, GlobalWeights};
    use crate::shared::experts::rtx::{dense, push::CoordinatorPush};
    use crate::shared::memory::push_exl3::{validate_offer, RankPlan};
    use crate::shared::placement::SparkAdmission;
    use crate::shared::token_io::{EmbedSource, TokenEmbedding};
    use cuteafd_loader::families::muse_spark::{core::CoreInventory, MuseSparkConfig};
    use cuteafd_loader::placement::families::muse_spark as shared;
    use cuteafd_loader::placement::{ExpertHome, Onboard};
    use cuteafd_loader::plan::MediaMode;
    use cuteafd_transport::worker_selection::push::Offer;

    let Some(snapshot) = snapshot() else { return Ok(()) };
    let share = std::path::PathBuf::from(std::env::var_os("CUTEAFD_PUSH_MUSE_SHARE")
        .context("set CUTEAFD_PUSH_MUSE_SHARE to the image's share directory")?);
    let manifest_path = share.join("PROGRAMS.json");
    let out = std::path::PathBuf::from(std::env::var_os("CUTEAFD_PUSH_PLAN_DUMP")
        .context("set CUTEAFD_PUSH_PLAN_DUMP")?);
    if let Some(parent) = out.parent() { std::fs::create_dir_all(parent)?; }
    const SPARKS: usize = 6;
    const GPUS: usize = 2;
    const SEQUENCES: usize = 16;
    /// Arm A's admitted pool and logged per-Spark-rank residency (incast-a).
    const ARM_A_POOL: u64 = 2_097_152;
    const ARM_A_SPARK_RANK: u64 = 67_612_180_480;
    /// Arm A's coordinator dense records: (class2 bytes, operands, class3
    /// bytes, steps, largest step, finishes).
    const ARM_A_DENSE: (usize, usize, usize, usize, usize, usize) =
        (12_696_238_848, 2485, 11_387_666_432, 11_300, 2_425_344, 0);

    cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::MUSE_SPARK)
        .map_err(|_| anyhow::anyhow!("Muse Spark plan dump needs an isolated process"))?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&snapshot, false, false)?;
    let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
    cfg.release_geometry()?;
    let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)
        .with_context(|| format!("program manifest {}", manifest_path.display()))?)?;
    // Serving's own argument defaults (no prefill overrides, 16 sequence slots,
    // the shared 2048 MiB mark rule with no host retention).
    let prefill = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(SPARKS,
        cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::muse_spark::MUSE_SPARK),
        cuteafd_loader::serving_capacity::prefill_waves::PrefillOverrides {
            unit_rows: None, waves: None, span_rows: None, reserve_rows: None })?;
    let mark = crate::shared::prefix::MarkRule { lanes: SEQUENCES,
        entries: cuteafd_core::concurrency::prefix_entries(SEQUENCES as u64) as usize,
        budget_bytes: 2048 << 20, host: false };
    let drafter = match std::env::var_os("CUTEAFD_PUSH_MUSE_DRAFT") {
        Some(path) => {
            let path = std::path::PathBuf::from(path);
            let config: serde_json::Value = serde_json::from_slice(&std::fs::read(path.join("config.json"))
                .with_context(|| format!("drafter config {}", path.display()))?)?;
            let head = core.get_weight(cuteafd_loader::families::muse_spark::core::HEAD)
                .map(|head| head.stored_outputs).context("Muse drafter needs the K6 target head")?;
            let sms = std::env::var("CUTEAFD_PUSH_SMS").ok()
                .map(|sms| sms.parse::<u64>().context("CUTEAFD_PUSH_SMS")).transpose()?.unwrap_or(188);
            Some(shared::drafter_inventory(&cfg, &config, SEQUENCES, SEQUENCES, head, sms,
                cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8)?)
        }
        None => None,
    };
    // Arm A's live pre-weight inventory: what startup measured before any
    // target weights were resident, so the solve reproduces its placement.
    let measured = vec![
        super::startup::Measured { total_bytes: 101_973_491_712, free_bytes: 101_387_075_584,
            untracked_bytes: 586_416_128 },
        super::startup::Measured { total_bytes: 101_970_345_984, free_bytes: 101_375_541_248,
            untracked_bytes: 594_804_736 },
    ];
    let (built, placement) = super::startup::admit(super::startup::Options { cfg: &cfg, core: &core,
        catalog: &catalog, manifest: &manifest, measured, headroom_bytes: 2 << 30, spark_ranks: SPARKS,
        prefill, full_prefill_logits: false, slots: SEQUENCES as u64,
        mark_slots: mark.slots(shared::mark_bytes(&cfg, 1) as usize) as u64, max_context: None,
        pool_tokens: None, onboard: Onboard::Auto,
        spark_budget_bytes: cuteafd_loader::plan::DEFAULT_SPARK_BUDGET_BYTES,
        partial_bf16_max_rows: cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
        partial_reply_formats: Default::default(),
        local: shared::local_workspace(&catalog, Some(&manifest_path), GPUS, prefill.unit_rows)?,
        drafter, media: Some(super::media::inventory(&snapshot, MediaMode::Off, MediaMode::Off, 1)?) })?;
    let cutoff = cuteafd_loader::families::muse_spark::replies::Cutoff::with_wide(
        cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
        cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
        cuteafd_loader::families::muse_spark::replies::WideFormat::Nvfp4)?;
    let spark = super::admission::spark_plan(&placement, prefill, cutoff, SEQUENCES, SPARKS)?;
    let admission = SparkAdmission::plan(&snapshot, cuteafd_core::ExpertGeometry::MUSE_SPARK,
        Some(&catalog), &spark.working, SPARKS, None, spark.model.clone())?;
    anyhow::ensure!(admission.coordinator_pushes(), "Muse max planned without a coordinator push group");
    let (layers, assignments) = admission.admitted();
    eprintln!("placement: {}; pool {} tokens; onboard {} RTX expert layers; Spark layers {:?}..={:?}",
        placement.summary(), placement.pool_tokens, placement.onboard_layers,
        layers.first(), layers.last());
    // Serving's own expression: with nothing onboarded there is no
    // CoordinatorExperts payload (and no installed package to read).
    let onboarded = placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
        matches!(assignment.experts, ExpertHome::RtxTp2 | ExpertHome::RtxWhole { .. }).then_some(layer))
        .collect::<Vec<_>>();
    let package = if onboarded.is_empty() { None } else {
        Some(shared::local_package(&catalog, &manifest_path, GPUS, prefill.unit_rows)?.0) };
    let experts = package.as_ref().map(|package| super::onboard::push_plan(&cfg, &placement, GPUS,
        &onboarded, usize::try_from(prefill.unit_rows)?, package)).transpose()?;
    let (mut push, streams, coordinator) = CoordinatorPush::prepare(&snapshot, Some(&catalog), GPUS, experts)?;
    let mut plans = Vec::with_capacity(SPARKS);
    for rank in 0..SPARKS {
        plans.push(RankPlan::new(&catalog, &layers, SPARKS, rank,
            usize::try_from(assignments[rank].resident_bytes)?)?);
    }
    // The coordinator's dense records, as serving records them before membership.
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
    let library = fixture.load()?;
    for device in 0..GPUS as i32 { fixture.configure_memory(&library, device, 512 << 30)?; }
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let name = "model.language_model.embed_tokens.weight";
    let tensor = checkpoint.tensors.iter().find(|tensor| tensor.meta.name == name)
        .with_context(|| format!("Muse checkpoint lacks {name}"))?;
    let source = EmbedSource::new(&checkpoint.snapshot, &tensor.shard, &tensor.meta, cfg.hidden)?;
    let ((embedding, weights), records) = dense::recording(|| {
        devices[0].run(|| TokenEmbedding::load(&library, source.clone(),
            crate::shared::memory::EmbedPlacement::Gpu, || {
                let mut loaded = Vec::with_capacity(GPUS);
                for (rank, device) in devices.iter().enumerate() {
                    loaded.push(device.run(|| Ok::<_, anyhow::Error>((
                        load_rank(*device, &cfg, &checkpoint, &built.layouts, rank)?,
                        GlobalWeights::load(*device, &checkpoint, &built.global, rank, GPUS)?)))?);
                }
                Ok(loaded)
            }))
    })?;
    let summary = records.summary();
    eprintln!("coordinator records: class2 {} B in {} operands; class3 {} B in {} steps \
        (largest {} B); {} finishes", summary.0, summary.1, summary.2, summary.3, summary.4, summary.5);
    let (dense, _arenas) = dense::plan_records(&snapshot, records, &devices, SPARKS)?;
    let (root, host) = cuteafd_loader::push_load::namespace(&snapshot)?;
    let mut arenas: Vec<u64> = plans.iter().map(|plan| plan.layout.logical_bytes() as u64).collect();
    arenas.extend(dense.iter().map(|plan| plan.base + plan.host_bytes));
    // The coordinator ranks' offers as the group's driver receives them; the
    // group's own append places the dense entries at those arenas.
    let offers: Vec<Offer> = dense.iter().enumerate().map(|(gpu, plan)| Offer { rank: SPARKS + gpu,
        host: host.clone(), arena_bytes: plan.base + plan.host_bytes, placement: Vec::new() }).collect();
    push.install_dense(dense);
    let mut builder = cuteafd_loader::push_load::builder::Builder::new(&root, arenas);
    // The Spark half of push_exl3::prepare: every worker-reported offer must
    // equal its admitted balanced residency, then the rank plan appends.
    for (rank, plan) in plans.iter().enumerate() {
        let offer = Offer { rank, host: host.clone(), arena_bytes: plan.layout.logical_bytes() as u64,
            placement: plan.placement.clone() };
        validate_offer(&offer, rank, &assignments[rank], &plan.placement,
            plan.layout.logical_bytes() as u64)?;
        plan.append(&mut builder, &catalog, u32::try_from(rank)?)?;
    }
    (coordinator.append)(&mut builder, Some(&catalog), &offers)?;
    let checked = builder.finish()?;
    anyhow::ensure!(checked.arenas.len() == SPARKS + GPUS, "eight ranks");
    for rank in 0..SPARKS + GPUS {
        let entries = checked.plan.entries.iter().filter(|entry| entry.rank == rank as u32).count();
        eprintln!("rank {rank}: {} B in {entries} entries", checked.totals.rank_bytes[rank]);
    }
    let file = std::io::BufWriter::new(std::fs::File::create(&out)?);
    serde_json::to_writer(file, &serde_json::json!({"plan": checked.plan, "file_sizes": checked.file_sizes,
        "arenas": checked.arenas}))?;
    eprintln!("wrote {} ({} files, {} patterns, {} entries)", out.display(), checked.plan.files.len(),
        checked.plan.patterns.len(), checked.plan.entries.len());

    // Per-rank residency split, from the same RankPlan the group appends:
    // trellis payloads and the suh/svh scale tensors of each projection, plus
    // the fixed descriptor/unit-map/scales buffers.
    let sidecar = out.with_extension("rank-bytes.json");
    let mut ranks = Vec::with_capacity(SPARKS);
    let mut resident_totals = Vec::with_capacity(SPARKS);
    for (rank, plan) in plans.iter().enumerate() {
        let mut per_layer = Vec::with_capacity(plan.residencies.len());
        // [gate, up, down] x [trellis, suh, svh], whole-rank destination bytes.
        let mut totals = [[0u64; 3]; 3];
        let (mut arena_sum, mut buffer_sum) = (0u64, 0u64);
        let mut widths = std::collections::BTreeMap::<usize, usize>::new();
        for (index, residency) in plan.residencies.iter().enumerate() {
            let (_, arena) = residency.device_arena_layout()?;
            let buffers: usize = residency.buffers.iter().map(|buffer| buffer.bytes).sum();
            let mut kinds = [[0u64; 3]; 3];
            for load in &residency.loads {
                let projection = if load.tensor.contains(".gate_proj.") { 0 }
                    else if load.tensor.contains(".up_proj.") { 1 }
                    else if load.tensor.contains(".down_proj.") { 2 }
                    else { anyhow::bail!("unexpected Muse EXL3 tensor {}", load.tensor) };
                let kind = if load.tensor.ends_with(".trellis") { 0 }
                    else if load.tensor.ends_with(".suh") { 1 }
                    else if load.tensor.ends_with(".svh") { 2 }
                    else { anyhow::bail!("unexpected Muse EXL3 tensor {}", load.tensor) };
                // The whole residency: every destination tier gets its own copy.
                let placed = (load.bytes * load.destinations.len()) as u64;
                kinds[projection][kind] += placed;
                totals[projection][kind] += placed;
            }
            *widths.entry(residency.intermediate).or_default() += 1;
            arena_sum += arena as u64;
            buffer_sum += buffers as u64;
            per_layer.push(serde_json::json!({
                "layer": layers[index],
                "intermediate_start": residency.intermediate_start,
                "intermediate": residency.intermediate,
                "tiers": residency.tiers,
                "projection_counts": residency.projection_counts,
                "arena_bytes": arena,
                "buffer_bytes": buffers,
                "page_and_alignment_bytes": arena as u64 - buffers as u64,
                "trellis_bytes": {"gate": kinds[0][0], "up": kinds[1][0], "down": kinds[2][0]},
                "suh_bytes": {"gate": kinds[0][1], "up": kinds[1][1], "down": kinds[2][1]},
                "svh_bytes": {"gate": kinds[0][2], "up": kinds[1][2], "down": kinds[2][2]},
            }));
        }
        let placed: u64 = totals.iter().flatten().sum();
        resident_totals.push(assignments[rank].resident_bytes);
        eprintln!("Spark rank {rank}: admitted {} B, padded arena {} B, logical {} B, widths {widths:?}",
            assignments[rank].resident_bytes, plan.layout.bytes(), plan.layout.logical_bytes());
        ranks.push(serde_json::json!({
            "rank": rank,
            "layers": layers.len(),
            "assignment_resident_bytes": assignments[rank].resident_bytes,
            "per_layer_arena_sum": arena_sum,
            "buffer_sum": buffer_sum,
            "validation_bytes": plan.layout.logical_bytes() - plan.layout.bytes(),
            "padded_arena_bytes": plan.layout.bytes(),
            "logical_bytes": plan.layout.logical_bytes(),
            "intermediate_widths": widths,
            "trellis_bytes": {"gate": totals[0][0], "up": totals[1][0], "down": totals[2][0]},
            "suh_bytes": {"gate": totals[0][1], "up": totals[1][1], "down": totals[2][1]},
            "svh_bytes": {"gate": totals[0][2], "up": totals[1][2], "down": totals[2][2]},
            "scales_bytes": totals.iter().map(|kinds| kinds[1] + kinds[2]).sum::<u64>(),
            "fixed_bytes": buffer_sum - placed,
            "page_and_alignment_bytes": arena_sum - buffer_sum,
            "per_layer": per_layer,
        }));
    }
    let sidecar_file = std::io::BufWriter::new(std::fs::File::create(&sidecar)?);
    serde_json::to_writer(sidecar_file, &serde_json::json!({
        "model": "wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-Core-K6",
        "layout": "2 RTX + 6 Sparks, EXL3 K4/K5 TP6, 0 onboarded RTX expert layers",
        "world": SPARKS + GPUS,
        "sparks": SPARKS,
        "spark_layers": layers,
        "arms": {"spark_rank_resident_bytes": ARM_A_SPARK_RANK, "dense": {
            "class2_bytes": ARM_A_DENSE.0, "operands": ARM_A_DENSE.1,
            "class3_bytes": ARM_A_DENSE.2, "steps": ARM_A_DENSE.3,
            "largest_step_bytes": ARM_A_DENSE.4, "finishes": ARM_A_DENSE.5}},
        "ranks": ranks,
    }))?;
    eprintln!("wrote {} (per-rank residency split)", sidecar.display());

    // Sanity against arm A (incast-a): the same placement, the same Spark
    // resident bytes and the same coordinator dense records. Report rather
    // than adjust: a mismatch means this dump is not arm A's group.
    anyhow::ensure!(placement.pool_tokens == ARM_A_POOL && placement.onboard_layers == 0
        && layers.len() == cfg.layers,
        "placement differs from arm A: pool {} (want {}), onboard {} (want 0), layers {} (want {})",
        placement.pool_tokens, ARM_A_POOL, placement.onboard_layers, layers.len(), cfg.layers);
    anyhow::ensure!(resident_totals.iter().all(|&bytes| bytes == ARM_A_SPARK_RANK),
        "Spark rank residency differs from arm A's {} B: {resident_totals:?}", ARM_A_SPARK_RANK);
    anyhow::ensure!(summary == ARM_A_DENSE, "coordinator dense records {summary:?} differ from arm A's {ARM_A_DENSE:?}");
    let _ = (&embedding, &weights, &streams, &host);
    Ok(())
}
