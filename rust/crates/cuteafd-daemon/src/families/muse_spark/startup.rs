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
    let missing = programs::required(gpus)?.into_iter().map(|(name, _)| name)
        .filter(|name| !names.contains(name.as_str())).collect::<Vec<_>>();
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
    shared::placement(shared::MuseInputs { cfg: options.cfg, core: options.core, catalog: options.catalog,
        gpus: budgets, spark_ranks: options.spark_ranks, prefill: options.prefill, slots: options.slots,
        mark_slots: options.mark_slots, context,
        automatic_context: options.max_context.is_none(), requested_pool: options.pool_tokens,
        onboard: options.onboard, spark_budget_bytes: options.spark_budget_bytes,
        partial_bf16_max_rows: options.partial_bf16_max_rows, partial_wide_format: options.partial_wide_format,
        fixed, startup: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::{self, layout::LayoutOptions, testing, ExpertPlacement, PlanOptions};

    fn manifest(gpus: usize) -> serde_json::Value {
        let programs = programs::required(gpus).unwrap().into_iter()
            .map(|(name, _)| serde_json::json!({"name": name, "capacity_rows": 4096})).collect::<Vec<_>>();
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

    /// The planner and measured startup build one request: on a sample whose
    /// free bytes are the planner's context-only baseline and with no code yet
    /// resident, both solve to the identical placement.
    #[test]
    fn planner_equals_measured_startup_before_allocation() -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_muse_spark_snapshot(dir.path(), &(0..62).collect::<Vec<_>>());
        let input = testing::BudgetedPlanInput::open(dir.path())?;
        let checkpoint = input.checkpoint();
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        let core = CoreInventory::from_checkpoint(&cfg, checkpoint)?;
        for (gpus, sparks) in [(2, 6), (2, 4), (1, 6)] {
            let path = dir.path().join(format!("PROGRAMS-{gpus}.json"));
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
                partial_wide_format: Default::default() })?;
            assert_eq!(live, planned, "{gpus} RTX + {sparks} Sparks");
            assert!(live.pool_tokens > 0);
        }
        Ok(())
    }

    #[test]
    fn three_coordinators_refuse_before_any_device_query() {
        assert!(shared::executor(3).is_err());
        assert!(shared::executor(0).is_err());
    }
}
