//! Muse Spark startup: manifest preflight, then measured admission through the
//! same request builder the planner uses
//! (`cuteafd_loader::placement::families::muse_spark`). Admission precedes
//! every retained operand; nothing here allocates.
use super::programs;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{families::muse_spark::{core::CoreInventory, MuseSparkConfig},
    placement::{families::muse_spark as shared, Baseline, GpuBudget, Onboard, Placement, ProgramSet}};
use std::path::Path;

/// Every program a coordinator layout binds is in the image's manifest, before
/// any module load or device query.
pub(crate) fn preflight(manifest: &Path, gpus: usize) -> Result<serde_json::Value> {
    let inventory: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest)
        .with_context(|| format!("Muse program manifest {}", manifest.display()))?)?;
    let names = inventory["programs"].as_array().context("Muse PROGRAMS.json has no program list")?
        .iter().filter_map(|program| program["name"].as_str()).collect::<std::collections::BTreeSet<_>>();
    let mut required = programs::required(gpus)?.into_iter().map(|(name, _)| name).collect::<Vec<_>>();
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
    pub cfg: &'a MuseSparkConfig,
    pub core: &'a CoreInventory,
    pub catalog: &'a cuteafd_loader::OfficialV41Catalog,
    pub manifest: &'a serde_json::Value,
    pub measured: Vec<Measured>,
    pub headroom_bytes: u64,
    pub spark_ranks: usize,
    pub prefill: cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan,
    pub slots: u64,
    /// Device mark slots from the shared prefix rule (`PrefixArgs::mark_rule`).
    pub mark_slots: u64,
    pub max_context: Option<u64>,
    pub pool_tokens: Option<u64>,
    pub onboard: Onboard,
    pub spark_budget_bytes: u64,
    pub partial_bf16_max_rows: usize,
    pub partial_wide_format: cuteafd_loader::families::muse_spark::replies::WideFormat,
    /// The coordinator EXL3 package state (`shared::local_workspace`).
    pub local: shared::LocalPackage,
    /// The DFlash2 drafter's charge (`shared::drafter_inventory`), when one serves.
    pub drafter: Option<shared::MuseDrafter>,
    pub media: Option<shared::MediaOptions>,
}

/// Measured admission: the shared request with live baselines and the code the
/// sample has not yet loaded. Same solver, same executor check as the planner.
pub(crate) fn admit(options: Options<'_>) -> Result<(shared::MuseRequest, Placement)> {
    let gpus = options.measured.len();
    shared::executor(gpus)?;
    let programs = ProgramSet::from_manifest(options.manifest, shared::program_families(gpus));
    let mut fixed = Vec::with_capacity(gpus);
    let budgets = options.measured.iter().enumerate().map(|(rank, sample)| {
        let arch = cuteafd_loader::placement::ArchContext::for_device("sm_120", sample.total_bytes);
        let code = cuteafd_loader::placement::inventory::loaded_code_allowance("muse", "exl3", gpus == 2,
            rank as u8, arch, Some(&programs));
        fixed.push(shared::code_demand(rank, code.pending(sample.untracked_bytes, arch.context_bytes))?);
        let capacity = cuteafd_ffi::coordinator_gpu_budget().map_or(sample.total_bytes, |b| b.0.min(sample.total_bytes));
        Ok(GpuBudget { capacity_bytes: capacity, headroom_bytes: options.headroom_bytes,
            baseline: Baseline::Measured { free_bytes: sample.free_bytes } })
    }).collect::<Result<Vec<_>>>()?;
    let context = options.max_context.unwrap_or(options.cfg.max_position as u64);
    let mut built = shared::request(shared::MuseInputs { cfg: options.cfg, core: options.core, catalog: options.catalog,
        gpus: budgets, spark_ranks: options.spark_ranks, prefill: options.prefill, slots: options.slots,
        mark_slots: options.mark_slots, context,
        automatic_context: options.max_context.is_none(), requested_pool: options.pool_tokens,
        onboard: options.onboard, spark_budget_bytes: options.spark_budget_bytes,
        partial_bf16_max_rows: options.partial_bf16_max_rows, partial_wide_format: options.partial_wide_format,
        local: options.local, drafter: options.drafter, fixed, startup: Vec::new() })?;
    if let Some(media) = &options.media { built.admit_media(media)?; }
    let placed = cuteafd_loader::placement::solve(&built.request)?;
    built.local.check(&placed)?;
    built.request.executor.check(&placed)?;
    built.check_media_workers(&placed)?;
    Ok((built, placed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::{layout::LayoutOptions, testing, ExpertPlacement, PlanOptions};

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
        let cases = [(DEFAULT_BF16_MAX_ROWS, WideFormat::Nvfp4, None, Qwen4KvCache::Fp8),
            (128, WideFormat::Nvfp4, None, Qwen4KvCache::Fp8), (512, WideFormat::Fp8, None, Qwen4KvCache::Fp8),
            (DEFAULT_BF16_MAX_ROWS, WideFormat::Fp8, Some("radixark.json"), Qwen4KvCache::Fp8),
            (DEFAULT_BF16_MAX_ROWS, WideFormat::Nvfp4, Some("radixark.json"), Qwen4KvCache::Bf16),
            (128, WideFormat::Nvfp4, Some("nvidia.json"), Qwen4KvCache::Fp8)];
        let mut host_rings = std::collections::BTreeSet::new();
        for (cutoff_rows, wide, drafter, draft_kv) in cases {
            let label = format!("cutoff {cutoff_rows} {wide:?} drafter {drafter:?} {draft_kv:?}");
            let snapshot = drafter.map(|name| -> Result<_> {
                let dir = tempfile::tempdir()?;
                std::fs::copy(drafters.join(name), dir.path().join("config.json"))?;
                Ok(dir)
            }).transpose()?;
            let opts = LayoutOptions { host_memory: Some((256 << 30, 256 << 30)), rtx_bytes: vec![total; gpus],
                head_split: true, host_cache_bytes: Some(0), workspace_manifest: Some(path.clone()),
                muse_partial_bf16_max_rows: cutoff_rows, muse_partial_wide_format: wide, draft_kv_cache: draft_kv,
                muse_drafter_snapshot: snapshot.as_ref().map(|d| d.path().to_path_buf()), ..Default::default() };
            let report = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(sparks),
                layout: Some(opts.clone()), ..Default::default() })?;
            let planned = report.solver_placement.clone().unwrap_or_else(|| panic!("{label}: {:?}", report.hints));
            let descriptor = cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::muse_spark::MUSE_SPARK);
            let prefill = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(sparks, descriptor, Default::default())?;
            // serve-muse's own derivations from the same flags.
            let cutoff = Cutoff::with_wide(cutoff_rows, DEFAULT_BF16_MAX_ROWS, wide)?;
            let served_drafter = snapshot.as_ref().map(|d| -> Result<_> {
                let config: serde_json::Value = serde_json::from_slice(&std::fs::read(d.path().join("config.json"))?)?;
                let sms = cuteafd_loader::placement::ArchContext::for_device("sm_120", total).sms;
                shared::drafter_inventory(&cfg, &config, 16, 16, 202_112, u64::from(sms), draft_kv)
            }).transpose()?;
            let arch = cuteafd_loader::placement::ArchContext::for_device("sm_120", total);
            let (_, live) = admit(Options { cfg: &cfg, core: &core, catalog: input.expert_catalog()?,
                manifest: &preflight(&path, gpus)?,
                measured: vec![Measured { total_bytes: total, free_bytes: total - arch.context_bytes,
                    untracked_bytes: arch.context_bytes }; gpus],
                headroom_bytes: opts.headroom_bytes, spark_ranks: sparks, prefill, slots: 16,
                mark_slots: cuteafd_core::prefix::mark_slots_with_host(16, opts.mimo_prefix_entries,
                    shared::mark_bytes(&cfg, 1), opts.mimo_prefix_mark_bytes, false),
                max_context: None, pool_tokens: None, onboard: Onboard::Auto,
                spark_budget_bytes: cuteafd_loader::plan::DEFAULT_SPARK_BUDGET_BYTES,
                partial_bf16_max_rows: cutoff_rows, partial_wide_format: wide,
                local: shared::local_workspace(input.expert_catalog()?, Some(&path), gpus, prefill.unit_rows)?,
                drafter: served_drafter, media: None })?;
            // Coordinator ledgers, item by item (then the whole placement).
            for gpu in 0..gpus {
                assert_eq!(live.items[gpu], planned.items[gpu], "{label}: GPU {gpu} ledger");
            }
            assert_eq!(live, planned, "{label}");
            // The Spark side: the planner's rings are the sealed endpoint's.
            let spark = super::super::admission::spark_plan(&live, prefill, cutoff, 16, sparks)?;
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
            host_rings.insert((cutoff_rows, format!("{wide:?}"), host.rings));
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
        for (gpus, sparks) in [(2, 6), (2, 4), (1, 6)] {
            // A complete image: the program manifest with the coordinator EXL3 package beside it.
            let image = dir.path().join(format!("image-{gpus}"));
            let path = testing::write_muse_exl3_capacities(&image);
            std::fs::write(&path, serde_json::to_vec(&manifest(gpus))?)?;
            let total = cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES;
            let opts = LayoutOptions { host_memory: Some((256 << 30, 256 << 30)), rtx_bytes: vec![total; gpus],
                head_split: gpus == 2, host_cache_bytes: Some(0), workspace_manifest: Some(path.clone()),
                ..Default::default() };
            let report = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(sparks),
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
            let (_, live) = admit(Options { cfg: &cfg, core: &core, catalog: input.expert_catalog()?,
                manifest: &inventory,
                measured: vec![Measured { total_bytes: total, free_bytes: total - arch.context_bytes,
                    untracked_bytes: arch.context_bytes }; gpus],
                headroom_bytes: opts.headroom_bytes, spark_ranks: sparks, prefill, slots: 16,
                mark_slots: cuteafd_core::prefix::mark_slots_with_host(16, opts.mimo_prefix_entries,
                    shared::mark_bytes(&cfg, 1), opts.mimo_prefix_mark_bytes, opts.host_cache_bytes != Some(0)),
                max_context: None,
                pool_tokens: None, onboard: Onboard::Auto,
                spark_budget_bytes: cuteafd_loader::plan::DEFAULT_SPARK_BUDGET_BYTES,
                partial_bf16_max_rows: cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
                partial_wide_format: Default::default(),
                local: shared::local_workspace(input.expert_catalog()?, Some(&path), gpus, prefill.unit_rows)?,
                drafter: None, media: Some(shared::MediaOptions::from_checkpoint(checkpoint,
                    cuteafd_loader::plan::MediaMode::Auto, cuteafd_loader::plan::MediaMode::Off, 1)?) })?;
            assert_eq!(live, planned, "{gpus} RTX + {sparks} Sparks");
            let encoder = report.encoder.as_ref().context("vision placement")?;
            assert_eq!(encoder.admitted_bytes(), 4_864_551_264, "native vision ledger");
            let layout = report.memory_layout.as_ref().context("vision layout")?;
            let native: u64 = layout.devices.iter().flat_map(|device| &device.items)
                .filter(|item| item.group == "vision tower" || item.group == "vision scratch")
                .map(|item| item.bytes).sum();
            assert_eq!(native, encoder.admitted_bytes());
            assert!(live.pool_tokens > 0);
            // The same image without its package: startup refuses whenever the placement
            // onboards, never serving the layers from the Sparks instead.
            if live.onboard_layers > 0 {
                std::fs::remove_dir_all(image.join("share/exl3"))?;
                let local = shared::local_workspace(input.expert_catalog()?, Some(&path), gpus, prefill.unit_rows)?;
                assert!(matches!(local, shared::LocalPackage::Missing(_)));
                let error = admit(Options { cfg: &cfg, core: &core, catalog: input.expert_catalog()?,
                    manifest: &inventory,
                    measured: vec![Measured { total_bytes: total, free_bytes: total - arch.context_bytes,
                        untracked_bytes: arch.context_bytes }; gpus],
                    headroom_bytes: opts.headroom_bytes, spark_ranks: sparks, prefill, slots: 16,
                    mark_slots: cuteafd_core::prefix::mark_slots_with_host(16, opts.mimo_prefix_entries,
                        shared::mark_bytes(&cfg, 1), opts.mimo_prefix_mark_bytes, opts.host_cache_bytes != Some(0)),
                    max_context: None, pool_tokens: None, onboard: Onboard::Auto,
                    spark_budget_bytes: cuteafd_loader::plan::DEFAULT_SPARK_BUDGET_BYTES,
                    partial_bf16_max_rows: cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
                    partial_wide_format: Default::default(), local, drafter: None, media: None }).err().expect("missing package refuses").to_string();
                assert!(error.contains("rebuild with the Muse coordinator scope"), "{error}");
                refused += 1;
            }
        }
        assert!(refused > 0, "no layout onboarded, so the missing-package refusal went unexercised");
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
        let opts = LayoutOptions { host_memory: Some((256 << 30, 256 << 30)), rtx_bytes: vec![total; gpus],
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
        let (built, live) = admit(Options { cfg: &cfg, core: &core, catalog: input.expert_catalog()?,
            manifest: &preflight(&path, gpus)?,
            measured: vec![Measured { total_bytes: total, free_bytes: total - arch.context_bytes,
                untracked_bytes: arch.context_bytes }; gpus],
            headroom_bytes: opts.headroom_bytes, spark_ranks: sparks, prefill, slots: 16,
            mark_slots: cuteafd_core::prefix::mark_slots_with_host(16, opts.mimo_prefix_entries,
                shared::mark_bytes(&cfg, 1), opts.mimo_prefix_mark_bytes, false),
            max_context: None, pool_tokens: None, onboard: Onboard::Auto,
            spark_budget_bytes: cuteafd_loader::plan::DEFAULT_SPARK_BUDGET_BYTES,
            partial_bf16_max_rows: opts.muse_partial_bf16_max_rows, partial_wide_format: opts.muse_partial_wide_format,
            local: shared::local_workspace(input.expert_catalog()?, Some(&path), gpus, prefill.unit_rows)?,
            drafter: None, media: Some(shared::MediaOptions::from_checkpoint(checkpoint, vision, audio, replicas)?) })?;
        assert_eq!(&live, planned);
        let admitted = built.media.context("runtime media")?;
        assert_eq!(Some(&admitted.vision), report.encoder.as_ref());
        assert_eq!(Some(&admitted.audio), report.audio_encoder.as_ref());
        let layout = report.memory_layout.as_ref().context("media memory layout")?;
        for (name, mode, encoder, bytes, copies) in [
            ("vision", vision, &admitted.vision, 4_864_551_264, replicas),
            ("audio", audio, &admitted.audio, 246_586_368, 1),
        ] {
            let native = layout.devices.iter().flat_map(|device| &device.items)
                .filter(|item| item.group == format!("{name} tower") || item.group == format!("{name} scratch"))
                .map(|item| item.bytes).sum::<u64>();
            if mode == MediaMode::Off { assert_eq!(native, 0); continue; }
            assert_eq!(encoder.admitted_bytes(), bytes);
            let expected_copies = if matches!(encoder.kind, EncoderKind::Spark { .. }) { copies } else { 1 };
            assert_eq!(native, bytes * expected_copies as u64);
        }
        for device in layout.devices.iter().filter(|device| device.kind == DeviceKind::Spark) {
            let expected = &admitted.spark_items[device.index as usize];
            for item in expected { assert!(device.items.contains(item)); }
            let guards = device.items.iter().filter(|item| item.group == "media admission guard").count();
            assert_eq!(guards, usize::from(!expected.is_empty()));
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
}
