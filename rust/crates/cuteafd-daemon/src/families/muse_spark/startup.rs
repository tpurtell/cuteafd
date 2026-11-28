//! Muse Spark startup: manifest preflight, then measured admission over the
//! same contract facts the planner builds
//! (`cuteafd_loader::placement::families::muse_spark`). Admission precedes
//! every retained operand; nothing here allocates.
use super::programs;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{families::muse_spark::{core::CoreInventory, MuseSparkConfig},
    placement::{families::muse_spark as shared, Onboard}};
use std::path::Path;

/// Every program a coordinator layout binds is in the image's manifest, before
/// any module load or device query.
pub(crate) fn preflight(manifest: &Path, gpus: usize) -> Result<serde_json::Value> {
    preflight_with_format(manifest, gpus, Default::default())
}

pub(crate) fn preflight_with_format(manifest: &Path, gpus: usize,
    format: cuteafd_loader::families::muse_spark::cache::MuseKvCache) -> Result<serde_json::Value> {
    let inventory: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest)
        .with_context(|| format!("Muse program manifest {}", manifest.display()))?)?;
    let names = inventory["programs"].as_array().context("Muse PROGRAMS.json has no program list")?
        .iter().filter_map(|program| program["name"].as_str()).collect::<std::collections::BTreeSet<_>>();
    let mut required = programs::required_with_format(gpus, format)?.into_iter().map(|(name, _)| name).collect::<Vec<_>>();
    // The shared rotary-table builder (sliding layers, span 64).
    required.push(programs::ROPE_TABLES.into());
    let missing = required.into_iter().filter(|name| !names.contains(name.as_str())).collect::<Vec<_>>();
    ensure!(missing.is_empty(), "Muse Spark image lacks coordinator programs {missing:?}");
    Ok(inventory)
}

/// Startup's measured view of each coordinator GPU, sampled before weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Measured {
    pub total_bytes: u64,
    pub free_bytes: u64,
    /// Code already resident at the sample (CUDA used minus tracked allocations).
    pub untracked_bytes: u64,
}

pub(crate) struct Options<'a> {
    pub kv_cache: cuteafd_loader::families::muse_spark::cache::MuseKvCache,
    pub cfg: &'a MuseSparkConfig,
    pub core: &'a CoreInventory,
    pub projections: Vec<cuteafd_loader::families::muse_spark::weight_layout::RankProjections>,
    pub catalog: &'a cuteafd_loader::OfficialV41Catalog,
    pub manifest: &'a serde_json::Value,
    pub measured: Vec<Measured>,
    pub headroom_bytes: u64,
    pub spark_ranks: usize,
    pub prefill: cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan,
    pub slots: u64,
    pub decode_rows: u32,
    /// Device mark slots from the shared prefix rule (`PrefixArgs::mark_rule`).
    pub mark_slots: u64,
    pub max_context: Option<u64>,
    pub pool_tokens: Option<u64>,
    pub onboard: Onboard,
    pub spark_rank_bytes: Vec<u64>,
    pub partial_bf16_max_rows: usize,
    pub partial_reply_formats: cuteafd_core::spark_reply::SparkReplyFormats,
    /// The coordinator EXL3 package state (`shared::local_workspace`).
    pub local: shared::LocalPackage,
    /// The DFlash2 drafter's charge (`shared::drafter_inventory`), when one serves.
    pub drafter: Option<shared::MuseDrafter>,
    pub media: Option<shared::MediaOptions>,
    pub host: shared::HostInput,
    /// `--decode-graphs`: off charges no graph reserve.
    pub decode_graphs: bool,
}

/// Measured admission: the same facts the planner states, with live baselines
/// and the code the sample has not yet loaded. Same solve, same executor and
/// package checks.
pub(crate) fn admit(options: Options<'_>) -> Result<shared::AdmittedMuse> {
    let gpus = options.measured.len();
    shared::executor(gpus)?;
    ensure!(options.projections.len() == gpus, "Muse native projection rank count differs from samples");
    let programs = shared::program_set(options.manifest, gpus, options.kv_cache);
    let mut fixed = Vec::with_capacity(gpus);
    let mut budgets = Vec::with_capacity(gpus + options.spark_ranks);
    for (rank, sample) in options.measured.iter().enumerate() {
        let mut programs = programs.clone();
        options.projections[rank].include_native_programs(&mut programs);
        let arch = cuteafd_loader::placement::ArchContext::for_device("sm_120", sample.total_bytes);
        let code = cuteafd_loader::placement::inventory::loaded_code_allowance("muse", "exl3", gpus == 2,
            rank as u8, arch, Some(&programs));
        fixed.push(shared::code_demand(rank, code.pending(sample.untracked_bytes, arch.context_bytes))?);
        let capacity = cuteafd_ffi::coordinator_gpu_budget().map_or(sample.total_bytes, |b| b.0.min(sample.total_bytes));
        budgets.push(shared::measured_budget(rank as u8, sample.total_bytes, capacity, options.headroom_bytes, sample.free_bytes));
    }
    ensure!(options.spark_rank_bytes.len() >= options.spark_ranks, "missing Muse Spark capacity");
    budgets.extend((0..options.spark_ranks).map(|rank| shared::spark_budget(rank as u8, options.spark_rank_bytes[rank])));
    let context = options.max_context.unwrap_or(options.cfg.max_position as u64);
    let capacities = options.measured.iter().map(|sample| sample.total_bytes).collect::<Vec<_>>();
    let cache = cuteafd_loader::families::muse_spark::cache::MuseCacheLayout::with_format(options.cfg,
        gpus, options.kv_cache)?;
    let mut facts = cuteafd_loader::placement::PoolFacts::new(cuteafd_loader::placement::planner::Context::resolve(context, options.max_context),
        options.pool_tokens, shared::PAGE_ROWS);
    facts.memory = cache.geometry.ranks.iter().zip(&capacities).map(|(rank, &available_bytes)|
        cuteafd_loader::placement::PoolMemory { available_bytes, bytes_per_unit: rank.persistent_unit_bytes }).collect();
    let pool = cuteafd_loader::placement::prepare_pool(facts)?;
    let media = options.media.unwrap_or(shared::MediaOptions { vision: cuteafd_loader::plan::MediaMode::Off,
        audio: cuteafd_loader::plan::MediaMode::Off, replicas: 1, vision_bytes: (0, 0), audio_bytes: (0, 0) });
    let prepared = shared::ProductionInput { kv_cache: options.kv_cache, cfg: options.cfg, core: options.core, projections: options.projections,
        catalog: options.catalog,
        budgets, host: options.host, prefill: options.prefill,
        bf16_target_head: crate::shared::target_head::TargetHead::configured().blas_head(true), slots: options.slots,
        decode_rows: options.decode_rows, mark_slots: options.mark_slots, context, pool,
        partial_bf16_max_rows: options.partial_bf16_max_rows, partial_reply_formats: options.partial_reply_formats,
        local: options.local, drafter: options.drafter, fixed, media, decode_graphs: options.decode_graphs }.prepare()?;
    let request = shared::production_request(cuteafd_loader::placement::planner::Context::resolve(context, options.max_context), options.pool_tokens, options.onboard,
        cuteafd_loader::placement::planner::PrefillShape { rows: options.prefill.unit_rows, waves: options.prefill.waves },
        cuteafd_loader::placement::planner::Operation::Serving,
        Some(crate::shared::memory::budget::resident_weight_limit()), options.partial_reply_formats);
    let admitted = prepared.admit(request)?;
    cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "muse_spark");
    cuteafd_bench::context::ResolvedSettings::placement(&admitted.placement).publish();
    cuteafd_bench::context::ResolvedSettings::prefill(options.prefill)
        .value("decode-rows", options.decode_rows)
        .value("max-context-tokens", admitted.solved.request.context.served(admitted.solved.pool_tokens))
        .publish();
    Ok(admitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::{layout::LayoutOptions, testing, ExpertPlacement, PlanOptions};

    fn without_sampled_baseline(mut placement: cuteafd_loader::placement::Placement,
        occupied: u64) -> cuteafd_loader::placement::Placement {
        for items in &mut placement.items {
            let owners = items.iter().filter(|item|
                cuteafd_loader::placement::sampled_baseline::is_source(&item.group)).collect::<Vec<_>>();
            assert_eq!(owners.len(), usize::from(occupied > 0));
            assert_eq!(owners.iter().map(|item| item.bytes).sum::<u64>(), occupied);
            items.retain(|item| !cuteafd_loader::placement::sampled_baseline::is_source(&item.group));
        }
        placement
    }

    fn manifest(gpus: usize) -> serde_json::Value {
        let programs = programs::required(gpus).unwrap().into_iter().map(|(name, _)| name)
            .chain([programs::ROPE_TABLES.to_string()])
            .map(|name| serde_json::json!({"name": name, "capacity_rows": 4096})).collect::<Vec<_>>();
        serde_json::json!({"schema": 1, "programs": programs})
    }

    #[test]
    fn preflight_names_every_missing_program() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("PROGRAMS.json");
        for gpus in [1, 2] {
            std::fs::write(&path, serde_json::to_vec(&manifest(gpus))?)?;
            preflight(&path, gpus)?;
        }
        let mut partial = manifest(2);
        partial["programs"].as_array_mut().unwrap().retain(|p| p["name"] != "muse2_attention_swa_decode");
        std::fs::write(&path, serde_json::to_vec(&partial)?)?;
        let error = preflight(&path, 2).unwrap_err().to_string();
        assert!(error.contains("muse2_attention_swa_decode"), "{error}");
        // A pair image carries only the head-split programs; one RTX needs the whole-head set.
        assert!(preflight(&path, 1).unwrap_err().to_string().contains("muse_attention_swa_decode"));
        assert!(preflight(&dir.path().join("absent.json"), 1).is_err());
        assert!(preflight(&path, 3).is_err());
        Ok(())
    }

    /// Plan == serve admission for every override serve-muse takes: the
    /// partial-reply cutoff and wide format, the drafter (none, RadixArk d128,
    /// NVIDIA d64) and its ring format. The coordinator ledgers match item by
    /// item; the Spark rings and the coordinator intake are what the serve
    /// path seals (`admission::spark_plan`, `shared::intake`); the host
    /// transport is `shared::host_transport` at the serve cutoff, nonzero.
    #[test]
    fn plan_equals_serve_admission_for_every_override() -> Result<()> {
        use cuteafd_loader::families::muse_spark::replies::{Cutoff, WideFormat, DEFAULT_BF16_MAX_ROWS};
        use cuteafd_loader::families::qwen4::Qwen4KvCache;
        use cuteafd_core::{spark_reply::SparkReplyFormats, SparkReplyFormat::{Bf16, Fp8, Nvfp4}};
        let dir = tempfile::tempdir()?;
        testing::write_muse_spark_snapshot(dir.path(), &(0..62).collect::<Vec<_>>());
        let input = testing::BudgetedPlanInput::open(dir.path())?;
        let checkpoint = input.checkpoint();
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let core = CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        let drafters = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/muse_spark/fixtures/drafters");
        let (gpus, sparks, total) = (2usize, 6usize, cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES);
        let image = dir.path().join("image");
        let path = testing::write_muse_exl3_capacities(&image);
        std::fs::write(&path, serde_json::to_vec(&manifest(gpus))?)?;
        let mut cases = [(DEFAULT_BF16_MAX_ROWS, WideFormat::Nvfp4, None, Qwen4KvCache::Fp8),
            (128, WideFormat::Nvfp4, None, Qwen4KvCache::Fp8), (512, WideFormat::Fp8, None, Qwen4KvCache::Fp8),
            (DEFAULT_BF16_MAX_ROWS, WideFormat::Fp8, Some("radixark.json"), Qwen4KvCache::Fp8),
            (DEFAULT_BF16_MAX_ROWS, WideFormat::Nvfp4, Some("radixark.json"), Qwen4KvCache::Bf16),
            (128, WideFormat::Nvfp4, Some("nvidia.json"), Qwen4KvCache::Fp8)]
            .into_iter().map(|(rows, wide, draft, kv)| (rows, SparkReplyFormats {
                prefill: Some(match wide { WideFormat::Bf16 => Bf16, WideFormat::Fp8 => Fp8, WideFormat::Nvfp4 => Nvfp4 }),
                decode: None }, draft, kv)).collect::<Vec<_>>();
        cases.extend([
            (DEFAULT_BF16_MAX_ROWS, SparkReplyFormats { prefill: Some(Bf16), decode: Some(Fp8) }, None, Qwen4KvCache::Fp8),
            (DEFAULT_BF16_MAX_ROWS, SparkReplyFormats { prefill: Some(Fp8), decode: Some(Nvfp4) }, None, Qwen4KvCache::Fp8),
            (DEFAULT_BF16_MAX_ROWS, SparkReplyFormats { prefill: None, decode: Some(Bf16) }, None, Qwen4KvCache::Fp8),
        ]);
        let mut host_rings = std::collections::BTreeSet::new();
        let cases = cases.into_iter().flat_map(|(cutoff, formats, draft, kv)| {
            let mut cases = vec![(cutoff, formats, draft, kv, false)];
            if draft.is_some() { cases.push((cutoff, formats, draft, kv, true)); }
            cases
        });
        let cases = cases.flat_map(|case| [cuteafd_loader::families::muse_spark::cache::MuseKvCache::Bf16,
            cuteafd_loader::families::muse_spark::cache::MuseKvCache::Fp8].into_iter().map(move |format| (case, format)));
        for ((cutoff_rows, formats, drafter, draft_kv, tp2), target_kv) in cases {
            let label = format!("cutoff {cutoff_rows} {formats:?} drafter {drafter:?} {draft_kv:?} TP2={tp2} target={target_kv:?}");
            let snapshot = drafter.map(|name| -> Result<_> {
                let dir = tempfile::tempdir()?;
                let config: serde_json::Value = serde_json::from_slice(&std::fs::read(drafters.join(name))?)?;
                std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&config)?)?;
                let geometry = crate::shared::speculation::dflash::DflashConfig::read(dir.path())?;
                let tensors = geometry.tensor_shapes().into_iter().map(|(name, shape)|
                    testing::t(name, "BF16", &shape)).collect::<Vec<_>>();
                testing::write_safetensors(&dir.path().join("model.safetensors"), &tensors);
                Ok(dir)
            }).transpose()?;
            let opts = LayoutOptions { context_tokens: 786432, host_memory: Some(cuteafd_loader::plan::testing::SimulatedHost { host_ram: (256 << 30, 256 << 30), ..cuteafd_loader::plan::testing::SimulatedHost::raptor() }.host_ram), rtx_bytes: vec![total; gpus],
                head_split: true, tp2_dflash: tp2, muse_kv: target_kv, host_cache_bytes: Some(0), workspace_manifest: Some(path.clone()),
                muse_partial_bf16_max_rows: cutoff_rows, spark_reply_formats: formats, draft_kv_cache: draft_kv,
                muse_drafter_snapshot: snapshot.as_ref().map(|d| d.path().to_path_buf()), ..Default::default() };
            let report = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(sparks),
                layout: Some(opts.clone()), ..Default::default() })?;
            let planned = report.solver_placement.clone().unwrap_or_else(|| panic!("{label}: {:?}", report.hints));
            let prefill = cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan {
                unit_rows: report.prefill_unit_rows.context("resolved unit rows")?,
                waves: report.prefill_waves.context("resolved waves")?,
                span_rows: report.prefill_span_rows.context("resolved span rows")?,
            };
            if cutoff_rows == DEFAULT_BF16_MAX_ROWS && formats.prefill == Some(Nvfp4) && formats.decode.is_none() && drafter.is_none() {
                for rank in 0..2 {
                    let items = &report.solver_placement.as_ref().unwrap().items[rank];
                    let owners = cuteafd_loader::placement::reserve::breakdown(format!("gpu{rank}"),
                        cuteafd_core::memory_policy::DeviceKind::Gpu, total, 0, items);
                    let graph = owners.reserve.iter().filter(|row| row.name == "graph reserve").collect::<Vec<_>>();
                    // The untraced bootstrap is empty; an explicit policy may charge a
                    // nonzero reserve, but never a second family-named owner or tail.
                    let policy = cuteafd_loader::serving_capacity::muse_graphs::policy(2, report.onboard_layers.unwrap_or(0), rank, true)?;
                    assert_eq!(graph.iter().map(|row| row.bytes).sum::<u64>(), policy.bytes);
                    assert!(graph.len() <= 1);
                    assert!(!owners.reserve.iter().any(|row| row.name != "graph reserve" && row.name.contains("graph")));
                }
                let placement = report.solver_placement.as_ref().unwrap();
                let charged = placement.items[1].iter().find(|item| item.group == "Muse head logits").unwrap().bytes;
                let (stored, trimmed) = cuteafd_loader::families::muse_spark::lane::head_planes(202112, 202048)?;
                assert_eq!(charged, (stored + trimmed) as u64);
                assert!(placement.items[0].iter().all(|item| item.group != "Muse head logits"), "head rank only");
                let placement = report.solver_placement.clone().context("placed")?;
                let folded_cutoff = Cutoff::with_formats(cutoff_rows, DEFAULT_BF16_MAX_ROWS, formats)?;
                let plan = super::super::admission::spark_plan(&placement, prefill, folded_cutoff, 16, 6, 64)?;
                assert_eq!(plan.working.spark_layers.len() + placement.onboard_layers, 62);
                assert!(plan.working.spark_layers.iter().all(|layer| !plan.working.rtx_layers.contains(layer)));
                let endpoint = plan.model.endpoints[0];
                let reply = endpoint.partial_reply_bytes.context("partial endpoint")? as usize;
                assert!(plan.max_frame_bytes >= reply && plan.model.rings.max_frame_bytes == plan.max_frame_bytes);
                let spark = report.memory_layout.as_ref().context("layout")?.devices.iter()
                    .find(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Spark).context("spark")?;
                let rings = spark.items.iter().find(|item| item.group == "rdma rings").context("rings")?.bytes;
                assert_eq!(rings, plan.model.ring_bytes(cuteafd_core::ExpertGeometry::MUSE_SPARK)? as u64);
                // Step metadata: the runtime's StepStorage extent is the charged item.
                let charged = placement.items[0].iter().find(|item| item.group == "Muse step metadata").context("charge")?.bytes;
                let tables = shared::step_tables(prefill.unit_rows as usize, 16, opts.context_tokens, prefill.waves as usize)?;
                assert_eq!(charged, tables.bytes() as u64);
                // The coordinator intake the link allocates (ranks x waves x unit x row) is the charge.
                let (row, planes) = shared::intake(&cuteafd_loader::families::muse_spark::MuseSparkConfig::from_hf(
                    &testing::muse_spark_config())?, prefill, folded_cutoff, 16, 6)?;
                assert_eq!(planes, 6 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(prefill.span_rows * row as u64)?);
                let charged = placement.items[0].iter().find(|item| item.group == "Muse Spark partial reply planes")
                    .context("intake charge")?.bytes;
                assert_eq!(charged, planes);
            }
            let descriptor = cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::muse_spark::MUSE_SPARK);
            let prefill = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(sparks, descriptor, Default::default())?;
            // serve-muse's own derivations from the same flags.
            let cutoff = Cutoff::with_formats(cutoff_rows, DEFAULT_BF16_MAX_ROWS, opts.spark_reply_formats)?;
            let served_drafter = snapshot.as_ref().map(|d| -> Result<_> {
                let config: serde_json::Value = serde_json::from_slice(&std::fs::read(d.path().join("config.json"))?)?;
                let sms = cuteafd_loader::placement::ArchContext::for_device("sm_120", total).sms;
                let draft = shared::drafter_inventory(&cfg, &config, 16, 16, 202_112, u64::from(sms), draft_kv)?;
                if tp2 { draft.with_tp2(d.path(), &[u64::from(sms); 2], draft_kv) }
                else { Ok(draft) }
            }).transpose()?;
            let arch = cuteafd_loader::placement::ArchContext::for_device("sm_120", total);
            let live = crate::commands::bench::assert_resolved_family_seam("muse_spark", &[], || {
                crate::commands::bench::fixture_prefix().resolve_entries(16);
                admit(Options { decode_graphs: true, kv_cache: target_kv, host: shared::HostInput { total: 256 << 30, available: 256 << 30,
                reserve: 8 << 30, prefix_bytes: Some(0), prefix_entries: opts.mimo_prefix_entries,
                media_bytes: opts.media_cache_bytes, device_sampling: true }, cfg: &cfg, core: &core, catalog: input.expert_catalog()?,
                manifest: &preflight(&path, gpus)?,
                projections: cuteafd_loader::families::muse_spark::weight_layout::projections(&cfg, &core,
                    gpus, prefill.unit_rows as usize)?,
                measured: vec![Measured { total_bytes: total, free_bytes: total - arch.context_bytes,
                    untracked_bytes: arch.context_bytes }; gpus],

                headroom_bytes: cuteafd_loader::placement::reserve::gpu("muse_spark"), spark_ranks: sparks, prefill, slots: 16, decode_rows: 64,
                mark_slots: cuteafd_core::prefix::mark_slots_with_host(16, opts.mimo_prefix_entries,
                    shared::mark_bytes_with_format(&cfg, 1, target_kv), opts.mimo_prefix_mark_bytes, false),
                max_context: Some(opts.context_tokens), pool_tokens: None, onboard: Onboard::Auto,

                spark_rank_bytes: cuteafd_loader::serving_capacity::spark_memory::serving_bytes(),
                partial_bf16_max_rows: cutoff_rows, partial_reply_formats: formats,
                local: shared::local_workspace(input.expert_catalog()?, Some(&path), gpus, prefill.unit_rows)?,
                drafter: served_drafter, media: None })
            })?;
            assert_eq!(report.phase_ledger.as_ref().context("solved ledger")?.pool_policy, live.solved.pool_policy);
            let draft_report = super::super::serve::drafter_report(&live, &[6, 4])?;
            assert_eq!(draft_report.is_some(), drafter.is_some(), "{label}");
            if let Some(layout) = draft_report {
                use crate::shared::draft::report::{Kind, Placement, Rank};
                assert_eq!((layout.kind, layout.placement, layout.lead), (Kind::Dflash2,
                    Some(if tp2 { Placement::Split } else { Placement::Whole }), Some(Rank::CudaRank(4))), "{label}");
                assert_eq!(layout.ranks.len(), if tp2 { 2 } else { 1 });
            }

            let live = without_sampled_baseline(live.placement, arch.context_bytes);
            // Coordinator ledgers, item by item (then the whole placement).
            for gpu in 0..gpus {
                assert_eq!(live.items[gpu], planned.items[gpu], "{label}: GPU {gpu} ledger");
            }
            assert_eq!(live, planned, "{label}");
            // The Spark side: the planner's rings are the sealed endpoint's.
            let spark = super::super::admission::spark_plan(&live, prefill, cutoff, 16, sparks, 64)?;
            let layout = report.memory_layout.as_ref().context("layout")?;
            for device in layout.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Spark) {
                let rings = device.items.iter().find(|item| item.group == "rdma rings").context("rings")?.bytes;
                assert_eq!(rings, spark.model.ring_bytes(cuteafd_core::ExpertGeometry::MUSE_SPARK)? as u64, "{label}");
            }
            // The coordinator host transport: what the link registers at this cutoff.
            let host = shared::host_transport(&cfg, prefill, cutoff, 16, sparks)?;
            let ledger = report.host_ledger.as_ref().context("host ledger")?;
            let bytes = |group: &str| ledger.items.iter().filter(|item| item.group == group).map(|item| item.bytes).sum::<u64>();
            assert_eq!((bytes("transport/rdma-rings"), bytes("transport/egress"), bytes("spark-intake")),
                (host.rings, host.egress, host.intake), "{label}");
            assert!(host.rings > 0 && host.egress > 0 && host.intake > 0, "{label}");
            host_rings.insert((cutoff_rows, format!("{formats:?}"), host.rings));
        }
        // The overrides move the transport (they are not ignored by either side).
        assert!(host_rings.iter().map(|(_, _, rings)| rings).collect::<std::collections::BTreeSet<_>>().len() > 1);
        Ok(())
    }

    /// The planner and measured startup build one request: on a sample whose
    /// free bytes are the planner's context-only baseline and with no code yet
    /// resident, both solve to the identical placement.
    #[test]
    fn planner_equals_measured_startup_before_allocation() -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_muse_vision_snapshot(dir.path(), &(0..62).collect::<Vec<_>>());
        let input = testing::BudgetedPlanInput::open(dir.path())?;
        let checkpoint = input.checkpoint();
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let core = CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        let mut refused = 0;
        let mut reduced_pools = 0;
        for (gpus, sparks) in [(2, 4), (1, 4)] {
            // A complete image: the program manifest with the coordinator EXL3 package beside it.
            let image = dir.path().join(format!("image-{gpus}"));
            let path = testing::write_muse_exl3_capacities(&image);
            std::fs::write(&path, serde_json::to_vec(&manifest(gpus))?)?;
            let total = cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES;
            let opts = LayoutOptions { context_tokens: 634304, host_memory: Some(cuteafd_loader::plan::testing::SimulatedHost { host_ram: (256 << 30, 256 << 30), ..cuteafd_loader::plan::testing::SimulatedHost::raptor() }.host_ram), rtx_bytes: vec![total; gpus],
                head_split: gpus == 2, host_cache_bytes: Some(0), workspace_manifest: Some(path.clone()),
                ..Default::default() };
            let vision = cuteafd_loader::plan::MediaMode::Spark(None);
            let report = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(sparks), vision,
                layout: Some(opts.clone()), ..Default::default() })?;
            let planned = report.solver_placement.clone()
                .unwrap_or_else(|| panic!("{gpus} RTX + {sparks}: {:?}", report.hints));
            let descriptor = cuteafd_loader::plan::Family::prefill_limits(
                &cuteafd_loader::plan::families::muse_spark::MUSE_SPARK);
            let prefill = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(sparks, descriptor,
                Default::default())?;
            assert_eq!((report.prefill_unit_rows, report.prefill_waves), (Some(prefill.unit_rows), Some(prefill.waves)));
            let arch = cuteafd_loader::placement::ArchContext::for_device("sm_120", total);
            let inventory = preflight(&path, gpus)?;
            let live = admit(Options { decode_graphs: true, kv_cache: Default::default(), host: shared::HostInput { total: 256 << 30, available: 256 << 30,
                reserve: 8 << 30, prefix_bytes: Some(0), prefix_entries: opts.mimo_prefix_entries,
                media_bytes: opts.media_cache_bytes, device_sampling: true }, cfg: &cfg, core: &core, catalog: input.expert_catalog()?,
                manifest: &inventory,
                projections: cuteafd_loader::families::muse_spark::weight_layout::projections(&cfg, &core,
                    gpus, prefill.unit_rows as usize)?,
                measured: vec![Measured { total_bytes: total, free_bytes: total - arch.context_bytes,
                    untracked_bytes: arch.context_bytes }; gpus],

                headroom_bytes: cuteafd_loader::placement::reserve::gpu("muse_spark"), spark_ranks: sparks, prefill, slots: 16, decode_rows: 64,
                mark_slots: cuteafd_core::prefix::mark_slots_with_host(16, opts.mimo_prefix_entries,
                    shared::mark_bytes(&cfg, 1), opts.mimo_prefix_mark_bytes, opts.host_cache_bytes != Some(0)),
                max_context: Some(opts.context_tokens),
                pool_tokens: None, onboard: Onboard::Auto,

                spark_rank_bytes: cuteafd_loader::serving_capacity::spark_memory::serving_bytes(),
                partial_bf16_max_rows: cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
                partial_reply_formats: Default::default(),
                local: shared::local_workspace(input.expert_catalog()?, Some(&path), gpus, prefill.unit_rows)?,
                drafter: None, media: Some(shared::MediaOptions::from_checkpoint(checkpoint,
                    vision, cuteafd_loader::plan::MediaMode::Off, 1)?) })?;
            assert_eq!(report.phase_ledger.as_ref().context("solved ledger")?.pool_policy, live.solved.pool_policy);
            let live = without_sampled_baseline(live.placement, arch.context_bytes);
            assert_eq!(live, planned, "{gpus} RTX + {sparks} Sparks");
            for gpu in 0..gpus {
                let output = live.items[gpu].iter().find(|item| item.group == "Muse prefill logits");
                assert!(output.is_none(), "scoring reuses the bounded head output");
            }
            let encoder = report.encoder.as_ref().context("vision placement")?;
            assert_eq!(encoder.admitted_bytes(), 4_860_356_960, "native vision ledger");
            assert_eq!(encoder.shortfall, 0, "explicit Spark media remains admitted");
            assert!(live.pool_tokens >= opts.context_tokens);
            if live.pool_tokens < cfg.max_position as u64 {
                reduced_pools += 1;
                assert_eq!(gpus, 1, "single-RTX cases retain media with a smaller pool");
            }
            let layout = report.memory_layout.as_ref().context("vision layout")?;
            let native: u64 = layout.devices.iter().flat_map(|device| &device.items)
                .filter(|item| item.group == cuteafd_loader::media::still_tower::vision_weight_arena_label() || item.group == "vision scratch")
                .map(|item| item.bytes).sum();
            assert_eq!(native, encoder.admitted_bytes());
            assert!(live.pool_tokens > 0);
            // The same image without its package: startup refuses whenever the placement
            // onboards, never serving the layers from the Sparks instead.
            if live.onboard_layers > 0 {
                std::fs::remove_dir_all(image.join("share/exl3"))?;
                let local = shared::local_workspace(input.expert_catalog()?, Some(&path), gpus, prefill.unit_rows)?;
                assert!(matches!(local, shared::LocalPackage::Missing(_)));
                let error = admit(Options { decode_graphs: true, kv_cache: Default::default(), host: shared::HostInput { total: 256 << 30, available: 256 << 30,
                    reserve: 8 << 30, prefix_bytes: Some(0), prefix_entries: opts.mimo_prefix_entries,
                    media_bytes: opts.media_cache_bytes, device_sampling: true }, cfg: &cfg, core: &core, catalog: input.expert_catalog()?,
                    manifest: &inventory,
                projections: cuteafd_loader::families::muse_spark::weight_layout::projections(&cfg, &core,
                    gpus, prefill.unit_rows as usize)?,
                    measured: vec![Measured { total_bytes: total, free_bytes: total - arch.context_bytes,
                        untracked_bytes: arch.context_bytes }; gpus],

                    headroom_bytes: cuteafd_loader::placement::reserve::gpu("muse_spark"), spark_ranks: sparks, prefill, slots: 16, decode_rows: 64,
                    mark_slots: cuteafd_core::prefix::mark_slots_with_host(16, opts.mimo_prefix_entries,
                        shared::mark_bytes(&cfg, 1), opts.mimo_prefix_mark_bytes, opts.host_cache_bytes != Some(0)),
                    max_context: Some(opts.context_tokens), pool_tokens: None, onboard: Onboard::Auto,

                spark_rank_bytes: cuteafd_loader::serving_capacity::spark_memory::serving_bytes(),
                    partial_bf16_max_rows: cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
                    partial_reply_formats: Default::default(), local, drafter: None,
                    media: Some(shared::MediaOptions::from_checkpoint(checkpoint,
                        vision, cuteafd_loader::plan::MediaMode::Off, 1)?)
                }).err().expect("missing package refuses").to_string();
                assert!(error.contains("rebuild with the Muse coordinator scope"), "{error}");
                refused += 1;
            }
        }
        assert!(refused > 0, "no layout onboarded, so the missing-package refusal went unexercised");
        assert!(reduced_pools > 0, "no reduced-pool media case was exercised");
        Ok(())
    }

    #[test]
    fn mixed_rtx_media_ledgers_equal_measured_startup() -> Result<()> {
        use cuteafd_loader::plan::MediaMode;
        media_ledger_case(MediaMode::Rtx(Some(0)), MediaMode::Rtx(Some(1)), 1)
    }

    #[test]
    fn replicated_spark_media_ledgers_equal_measured_startup() -> Result<()> {
        use cuteafd_loader::plan::MediaMode;
        media_ledger_case(MediaMode::Spark(Some(0)), MediaMode::Spark(Some(1)), 2)
    }

    #[test]
    fn audio_only_ledger_equals_measured_startup() -> Result<()> {
        use cuteafd_loader::plan::MediaMode;
        media_ledger_case(MediaMode::Off, MediaMode::Rtx(Some(0)), 1)
    }

    #[test]
    fn disabled_media_has_no_native_or_host_reservation() -> Result<()> {
        use cuteafd_loader::plan::MediaMode;
        media_ledger_case(MediaMode::Off, MediaMode::Off, 1)
    }

    fn media_ledger_case(vision: cuteafd_loader::plan::MediaMode, audio: cuteafd_loader::plan::MediaMode,
        replicas: usize) -> Result<()> {
        use cuteafd_core::memory_layout::DeviceKind;
        use cuteafd_loader::plan::{encoder::EncoderKind, MediaMode};
        let dir = tempfile::tempdir()?;
        testing::write_muse_media_snapshot(dir.path(), &(0..62).collect::<Vec<_>>());
        let input = testing::BudgetedPlanInput::open(dir.path())?;
        let checkpoint = input.checkpoint();
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let core = CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        let (gpus, sparks) = (2, 6);
        let total = cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES;
        let path = testing::write_muse_exl3_capacities(&dir.path().join("image"));
        std::fs::write(&path, serde_json::to_vec(&manifest(gpus))?)?;
        let opts = LayoutOptions { context_tokens: 786432, host_memory: Some(cuteafd_loader::plan::testing::SimulatedHost { host_ram: (256 << 30, 256 << 30), ..cuteafd_loader::plan::testing::SimulatedHost::raptor() }.host_ram), rtx_bytes: vec![total; gpus],
            head_split: true, host_cache_bytes: Some(0), workspace_manifest: Some(path.clone()),
            vision_replicas: replicas, media_cache_bytes: Some(123_456_789), ..Default::default() };
        let report = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(sparks), vision, audio,
            layout: Some(opts.clone()), ..Default::default() })?;
        assert!(report.fits, "{:?}", report.hints);
        let planned = report.solver_placement.as_ref().context("media solver placement")?;
        let prefill = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(sparks,
            cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::muse_spark::MUSE_SPARK),
            Default::default())?;
        let arch = cuteafd_loader::placement::ArchContext::for_device("sm_120", total);
        let built = admit(Options { decode_graphs: true, kv_cache: Default::default(), host: shared::HostInput { total: 256 << 30, available: 256 << 30,
                reserve: 8 << 30, prefix_bytes: Some(0), prefix_entries: opts.mimo_prefix_entries,
                media_bytes: opts.media_cache_bytes, device_sampling: true }, cfg: &cfg, core: &core, catalog: input.expert_catalog()?,
            manifest: &preflight(&path, gpus)?,
                projections: cuteafd_loader::families::muse_spark::weight_layout::projections(&cfg, &core,
                    gpus, prefill.unit_rows as usize)?,
            measured: vec![Measured { total_bytes: total, free_bytes: total - arch.context_bytes,
                untracked_bytes: arch.context_bytes }; gpus],

            headroom_bytes: cuteafd_loader::placement::reserve::gpu("muse_spark"), spark_ranks: sparks, prefill, slots: 16, decode_rows: 64,
            mark_slots: cuteafd_core::prefix::mark_slots_with_host(16, opts.mimo_prefix_entries,
                shared::mark_bytes(&cfg, 1), opts.mimo_prefix_mark_bytes, false),
            max_context: Some(opts.context_tokens), pool_tokens: None, onboard: Onboard::Auto,

                spark_rank_bytes: cuteafd_loader::serving_capacity::spark_memory::serving_bytes(),
            partial_bf16_max_rows: opts.muse_partial_bf16_max_rows, partial_reply_formats: opts.spark_reply_formats,
            local: shared::local_workspace(input.expert_catalog()?, Some(&path), gpus, prefill.unit_rows)?,
            drafter: None, media: Some(shared::MediaOptions::from_checkpoint(checkpoint, vision, audio, replicas)?) })?;
        let live = without_sampled_baseline(built.placement.clone(), arch.context_bytes);
        assert_eq!(&live, planned);
        let planned_phases = report.phase_ledger.as_ref().context("admitted phase ledger")?;
        assert_eq!(planned_phases.pool_policy, built.solved.pool_policy);
        assert_eq!(planned_phases.budgets.len(), built.solved.budgets.len());
        for (planned, live) in planned_phases.budgets.iter().zip(&built.solved.budgets) {
            assert_eq!((&planned.id, &planned.locations, planned.physical_bytes,
                planned.logical_bytes, planned.headroom),
                (&live.id, &live.locations, live.physical_bytes,
                    live.logical_bytes, live.headroom));
        }
        for (planned, live) in planned_phases.phases.iter().zip(&built.solved.phases) {
            assert_eq!((&planned.budget, planned.phase, planned.owned),
                (&live.budget, live.phase, live.owned));
            // Sampled context moves from unavailable capacity to a named opaque owner.
            assert_eq!(planned.available - planned.owned - planned.opaque,
                live.available - live.owned - live.opaque);
        }
        assert_eq!(report.host_budget_bytes, built.host.prefix_bytes);
        for device in report.memory_layout.as_ref().context("admitted devices")?.devices.iter()
            .filter(|device| device.kind == DeviceKind::Spark) {
            assert_eq!(device.capacity_bytes, cuteafd_loader::serving_capacity::spark_memory::serving_bytes()[device.index as usize]);
            assert_eq!(device.worker_budget_bytes, device.capacity_bytes);
        }
        let admitted = &built.media;
        assert_eq!(Some(&admitted.vision), report.encoder.as_ref());
        assert_eq!(Some(&admitted.audio), report.audio_encoder.as_ref());
        let layout = report.memory_layout.as_ref().context("media memory layout")?;
        for (name, mode, encoder, bytes, copies) in [
            ("vision", vision, &admitted.vision, 4_860_356_960, replicas),
            ("audio", audio, &admitted.audio, 246_586_368, 1),
        ] {
            let native = layout.devices.iter().flat_map(|device| &device.items)
                .filter(|item| item.group == if name == "vision" { cuteafd_loader::media::still_tower::vision_weight_arena_label() } else { "audio tower" } || item.group == format!("{name} scratch"))
                .map(|item| item.bytes).sum::<u64>();
            if mode == MediaMode::Off { assert_eq!(native, 0); continue; }
            assert_eq!(encoder.admitted_bytes(), bytes);
            let expected_copies = if matches!(encoder.kind, EncoderKind::Spark { .. }) { copies } else { 1 };
            assert_eq!(native, bytes * expected_copies as u64);
        }
        for device in layout.devices.iter().filter(|device| device.kind == DeviceKind::Spark) {
            let expected = &admitted.spark_items[device.index as usize];
            for item in expected { assert!(device.items.contains(item)); }
            let native = device.items.iter().filter(|item| item.group == cuteafd_loader::media::still_tower::vision_weight_arena_label()
                || matches!(item.group.as_str(), "vision scratch" | "audio tower" | "audio scratch"))
                .map(|item| item.bytes).sum::<u64>();
            assert_eq!(native, expected.iter().map(|item| item.bytes).sum::<u64>());
            assert!(!device.items.iter().any(|item| item.group == "media admission guard"));
        }
        let host = report.host_ledger.as_ref().context("media host ledger")?;
        let cache = if vision == MediaMode::Off && audio == MediaMode::Off { 0 } else { 123_456_789 };
        assert_eq!(host.items.iter().filter(|item| item.group == "media/embedding-cache")
            .map(|item| item.bytes).sum::<u64>(), cache);
        Ok(())
    }

    #[test]
    fn three_coordinators_refuse_before_any_device_query() {
        assert!(shared::executor(3).is_err());
        assert!(shared::executor(0).is_err());
    }

    #[test]
    fn decode_rows_128_plan_equals_measured_startup() -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_muse_spark_snapshot(dir.path(), &(0..62).collect::<Vec<_>>());
        let input = testing::BudgetedPlanInput::open(dir.path())?;
        let checkpoint = input.checkpoint();
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let core = CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        let (gpus, sparks, total) = (2usize, 6usize, cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES);
        let path = testing::write_muse_exl3_capacities(&dir.path().join("image"));
        std::fs::write(&path, serde_json::to_vec(&manifest(gpus))?)?;
        let opts = LayoutOptions {
            context_tokens: 786432,
            host_memory: Some(cuteafd_loader::plan::testing::SimulatedHost {
                host_ram: (256 << 30, 256 << 30), ..cuteafd_loader::plan::testing::SimulatedHost::raptor()
            }.host_ram),
            rtx_bytes: vec![total; gpus],
            head_split: true,
            host_cache_bytes: Some(0),
            workspace_manifest: Some(path.clone()),
            decode_rows: 128,
            ..Default::default()
        };
        let report = input.plan(&PlanOptions {
            placement: ExpertPlacement::from_spark_ranks(sparks),
            decode_rows: Some(128),
            layout: Some(opts.clone()),
            ..Default::default()
        })?;
        assert!(report.fits, "{:?}", report.hints);
        assert_eq!(report.decode_rows, 128);
        let planned = report.solver_placement.clone().context("planned 128")?;
        let prefill = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(
            sparks,
            cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::muse_spark::MUSE_SPARK),
            Default::default(),
        )?;
        let arch = cuteafd_loader::placement::ArchContext::for_device("sm_120", total);
        let live = admit(Options { decode_graphs: true,
            kv_cache: Default::default(),
            host: shared::HostInput {
                total: 256 << 30, available: 256 << 30, reserve: 8 << 30,
                prefix_bytes: Some(0), prefix_entries: opts.mimo_prefix_entries,
                media_bytes: opts.media_cache_bytes, device_sampling: true,
            },
            cfg: &cfg, core: &core, catalog: input.expert_catalog()?,
            manifest: &preflight(&path, gpus)?,
            projections: cuteafd_loader::families::muse_spark::weight_layout::projections(
                &cfg, &core, gpus, prefill.unit_rows as usize)?,
            measured: vec![Measured {
                total_bytes: total, free_bytes: total - arch.context_bytes, untracked_bytes: arch.context_bytes,
            }; gpus],
            headroom_bytes: cuteafd_loader::placement::reserve::gpu("muse_spark"),
            spark_ranks: sparks, prefill, slots: 16, decode_rows: 128,
            mark_slots: cuteafd_core::prefix::mark_slots_with_host(
                16, opts.mimo_prefix_entries, shared::mark_bytes(&cfg, 1), opts.mimo_prefix_mark_bytes, false),
            max_context: Some(opts.context_tokens), pool_tokens: None, onboard: Onboard::Auto,
            spark_rank_bytes: cuteafd_loader::serving_capacity::spark_memory::serving_bytes(),
            partial_bf16_max_rows: opts.muse_partial_bf16_max_rows,
            partial_reply_formats: opts.spark_reply_formats,
            local: shared::local_workspace(input.expert_catalog()?, Some(&path), gpus, prefill.unit_rows)?,
            drafter: None, media: None,
        })?;
        let live = without_sampled_baseline(live.placement, arch.context_bytes);
        assert_eq!(live, planned);
        let head_logits = live.items[1].iter().find(|item| item.group == "Muse head logits").unwrap().bytes;
        let (stored, trimmed) = cuteafd_loader::families::muse_spark::lane::head_planes_for_rows(128, 202112, 202048)?;
        assert_eq!(head_logits, (stored + trimmed) as u64);
        let cutoff = cuteafd_loader::families::muse_spark::replies::Cutoff::new(
            opts.muse_partial_bf16_max_rows, cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS)?;
        let spark = super::super::admission::spark_plan(&live, prefill, cutoff, 16, sparks, 128)?;
        assert_eq!(spark.model.executions.decode[0].capacity, 128);
        assert_eq!(spark.model.executions.decode[0].compiled_capacity, 256);
        Ok(())
    }
}
