//! Native export validation and measured-inventory edge of shared Step-5 admission.
use super::execution::Execution;
pub(super) use admission::RuntimeInputs;
use anyhow::{ensure, Result};
use cuteafd_ffi::programs::Programs;
use cuteafd_loader::plan::{checkpoint::Checkpoint, families::step5::admission};

pub(super) fn prefix_marks(prefix: &crate::shared::prefix::PrefixArgs, sequences: usize) -> admission::MarkRule {
    let rule = prefix.mark_rule(sequences);
    admission::MarkRule { lanes: rule.lanes, entries: rule.entries,
        budget_bytes: rule.budget_bytes, host: rule.host }
}

pub(super) fn admit<'a>(
    programs: &'a Programs<'a>,
    checkpoint: &'a Checkpoint,
    inputs: &RuntimeInputs,
) -> Result<admission::Admission> {
    admit_mtp(programs, checkpoint, inputs, 0)
}

pub(super) fn admit_mtp<'a>(
    programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
    inputs: &RuntimeInputs, depth: usize,
) -> Result<admission::Admission> {
    let contract = std::rc::Rc::new(cuteafd_loader::plan::families::step5::coordinator::CoordinatorContract::from_checkpoint(checkpoint)?);
    let cfg = cuteafd_loader::plan::families::step5::Config::parse(checkpoint)?;
    let specs = programs.names().map(|name| programs.spec(name)).collect::<Result<Vec<_>>>()?;
    let manifest = cuteafd_loader::placement::inventory::ProgramsManifest::from_specs(specs.into_iter())?;
    let selection = std::rc::Rc::new(cuteafd_loader::plan::families::step5::native::NativeSelection::resolve(
        &contract, &cfg, inputs.inventory.gpus.len(), depth, inputs.unit_rows(), Some(&manifest))?);
    Ok(admit_mtp_with_contract(programs, checkpoint, inputs, depth, contract, selection)?.0)
}

pub(super) fn admit_mtp_with_contract<'a>(programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
    inputs: &RuntimeInputs, depth: usize,
    contract: std::rc::Rc<cuteafd_loader::plan::families::step5::coordinator::CoordinatorContract>,
    selection: std::rc::Rc<cuteafd_loader::plan::families::step5::native::NativeSelection>,
) -> Result<(admission::Admission, super::execution::NativePlan)> {
    measured(inputs)?;
    let admitted = admission::admit_mtp_with_contract(checkpoint, inputs, depth, &contract)?;
    let plan = super::execution::NativePlan::new(checkpoint, &admitted.ownership, contract)?
        .with_selection(checkpoint, depth, inputs.unit_rows(), selection)?;
    if depth > 0 {
        for stage in 0..depth {
            super::mtp::InputPlan::prepare(programs, checkpoint, stage, inputs.unit_rows(), &plan.contract)?;
        }
    }
    let ownership = &admitted.ownership;
    for rank in 0..ownership.ranks() {
        let native =
            Execution::serving_admission_with_plan(programs, checkpoint, ownership, rank, inputs.unit_rows(), inputs.rows,
                inputs.logit_rows(), &plan)?;
        let shared = cuteafd_loader::plan::families::step5::native::rank_admission_for_layout(
            &plan.contract, &cuteafd_loader::plan::families::step5::Config::parse(checkpoint)?, &plan.layouts[rank],
            ownership,
            rank,
            inputs.unit_rows(),
            inputs.rows,
            inputs.logit_rows(),
        )?;
        ensure!(
            native.immutable == shared.immutable && native.workspace == shared.workspace,
            "Step-5 native allocation owners differ from shared admission"
        );
    }
    Ok((admitted, plan))
}

fn measured(inputs: &RuntimeInputs) -> Result<()> {
    ensure!(
        inputs.inventory.gpus.iter().all(|g| matches!(
            g.baseline,
            cuteafd_loader::placement::Baseline::Measured { .. }
        )),
        "Step-5 runtime requires a pre-weight measured baseline"
    );
    Ok(())
}

fn checkpoint_admission(
    checkpoint: &Checkpoint,
    inputs: &RuntimeInputs,
) -> Result<admission::Admission> {
    measured(inputs)?;
    let admitted = admission::admit(checkpoint, inputs)?;
    cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "step5");
    Ok(admitted)
}
#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::{
        placement::*,
        plan::{self, layout::LayoutOptions, testing, ExpertPlacement, PlanOptions},
    };

    #[test]
    fn production_reconciliation_registers_tp2_and_spark_owner_rows() -> Result<()> {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "families::step5::admission::tests::public_max_card_planner_equals_measured_daemon_before_allocation", "--nocapture"])
            .env("CUTEAFD_TEST_STEP5_RECONCILIATION", "1").status()?;
        ensure!(status.success(), "isolated Step-5 production publication regression failed");
        Ok(())
    }

    #[test]
    fn public_max_card_planner_equals_measured_daemon_before_allocation() -> Result<()> {
        planner_equals_measured_daemon(false)
    }

    #[test]
    fn public_nvfp4_min_card_planner_equals_measured_daemon_before_allocation() -> Result<()> {
        planner_equals_measured_daemon(true)
    }

    fn planner_equals_measured_daemon(nvfp4: bool) -> Result<()> {
        let ranks = if nvfp4 { 1 } else { 2 };
        let sparks = if nvfp4 { 4 } else { 6 };
        let package = if nvfp4 { "fp8-step5-nvfp4a4" } else { "fp8-step5" };
        let dir = tempfile::tempdir()?;
        let source: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/plan/families/step5/fixtures/official-fp8.json"
        )))?;
        let tensors = source["tensors"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, value)| {
                let dtype = match value[0].as_str().unwrap() {
                    "BF16" => "BF16",
                    "F32" => "F32",
                    "F8_E4M3" => "F8_E4M3",
                    _ => panic!("fixture dtype"),
                };
                testing::t(
                    name,
                    dtype,
                    &value[1]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|n| n.as_u64().unwrap() as usize)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let (config, tensors) = if nvfp4 {
            let config = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
                "/../cuteafd-loader/src/plan/families/step5/fixtures/official-nvfp4-config.json")))?;
            let mut native = Vec::new();
            for (name, dtype, shape) in tensors {
                if name.ends_with("weight_scale_inv") { continue; }
                if dtype == "F8_E4M3" {
                    let stem = name.strip_suffix(".weight").unwrap();
                    let [e, n, k]: [usize; 3] = shape.try_into().unwrap();
                    native.push(testing::t(&name, "U8", &[e, n, k / 2]));
                    native.push(testing::t(format!("{stem}.weight_scale"), "F8_E4M3", &[e, n, k / 16]));
                    native.push(testing::t(format!("{stem}.weight_scale_2"), "F32", &[e]));
                    native.push(testing::t(format!("{stem}.input_scale"), "F32", &[e]));
                } else { native.push((name, dtype, shape)); }
            }
            (config, native)
        } else { (source["config"].clone(), tensors) };
        testing::write_snapshot(dir.path(), &config, &tensors, None);
        // Export metadata only: no fabricated executor, library or CUDA pointers.
        let ops = [
            "hidden_norm",
            "final_norm",
            "residual_bf16",
            "residual_f32",
            "swa_attention",
            "csa_attention",
            "region_compress",
            "region_scores",
            "region_selection",
            "region_commit",
            "sliding_commit",
            "attn_q",
            "attn_kv",
            "attn_gate",
            "attn_out",
            "router",
            "dense_gate_up",
            "dense_down",
            "shared_gate_up",
            "shared_down",
        ];
        let entries = ["step5", "step52"]
            .iter()
            .flat_map(|prefix| {
                ops.iter().map(move |op|
            serde_json::json!({"name": format!("{prefix}_{op}"), "capacity_rows": 4096}))
            })
            .collect::<Vec<_>>();
        let manifest =
            serde_json::json!({"capacities": {"max_context": 1048576}, "programs": entries});
        let path = dir.path().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
        // The coordinator fp8moe packages a two-RTX FP8 card onboards TP2 through.
        for name in ["fp8-step5", "fp8-step5-nvfp4a4"] {
            let root = dir.path().join("fp8").join(name);
            for tp in ["tp1", "tp2"] { std::fs::create_dir_all(root.join(tp))?; }
            let capacities = |scale: u64| [1u64, 16, 80, 256, 1024, 4096].map(|capacity|
                serde_json::json!({"capacity": capacity, "scratch_bytes": capacity * scale}));
            std::fs::write(root.join("manifest.json"), serde_json::to_vec(&serde_json::json!({"layouts": {
                "tp1": {"abi_version":6,"projection":"bf16","activation":"silu_bf16_product","swiglu_limit":0.0,"swiglu_alpha":1.0,"swiglu_beta":0.0,"weights":"fp8","capacities": capacities(65536)}, "tp2": {"abi_version":6,"projection":"bf16","activation":"silu_bf16_product","swiglu_limit":0.0,"swiglu_alpha":1.0,"swiglu_beta":0.0,"weights":"fp8","capacities": capacities(32768)}}}))?)?;
        }
        let catalog = cuteafd_loader::read_expert_catalog(dir.path())?;
        let prefixes: &[&str] = if ranks == 2 { &["step5", "step52"] } else { &["step5"] };
        let programs = ProgramSet::from_manifest(&manifest, prefixes);
        let checkpoint = Checkpoint::coordinator(dir.path(), false, false)?;
        #[derive(clap::Parser)]
        struct PrefixCli {
            #[command(flatten)]
            prefix: crate::shared::prefix::PrefixArgs,
        }
        // 1M tokens of BF16 KV do not fit beside nvfp4 min's weights on one
        // 96 GiB card; FP8 KV does. The contract refuses with the feasible
        // context instead of clamping the pool and serving a smaller one.
        let full_context_fits = !nvfp4;
        for ((context, pool, launchable, host, entries), kv_format) in [
            (262_144, Some(262_144), true, None, 20),
            (1_048_576, None, full_context_fits, None, 20),
            (1_048_576, Some(262_144), false, None, 20),
            (1_048_576, None, full_context_fits, Some(0), 20),
            (1_048_576, None, full_context_fits, None, 0),
        ].into_iter().flat_map(|case| [cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
            cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8]
            .into_iter().map(move |format| (case, format))) {
            let opts = LayoutOptions { host_memory: Some((128 << 30, 128 << 30)),
                step5_kv: kv_format,
                rtx_bytes: vec![96 << 30; ranks],
                pool_tokens: pool,
                context_tokens: context,
                workspace_manifest: Some(path.clone()),
                host_cache_bytes: host,
                mimo_prefix_entries: entries,
                ..Default::default()
            };
            let report = plan::plan(
                dir.path(),
                &PlanOptions {
                    placement: ExpertPlacement::Sparks { ranks: sparks },
                    draft_ownership: Some(plan::spec::DraftOwnership::None),
                    layout: Some(opts.clone()),
                    ..Default::default()
                },
            )?;
            let launchable = launchable
                || (context > 262_144 && pool.is_none() && kv_format == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8);
            let retention_pressure = host == Some(0) && entries > 0;
            if !retention_pressure {
                assert_eq!(report.wip_launchable(), launchable,
                    "nvfp4={nvfp4} context={context} pool={pool:?} host={host:?} entries={entries}: {:?}", report.hints);
            }
            assert_eq!(report.executable(), report.wip_launchable());
            // A rejected allocation still reports the selected topology shape;
            // capacity admission never retries a smaller unit or fewer waves.
            let descriptor = cuteafd_loader::plan::Family::prefill_limits(
                &cuteafd_loader::plan::families::step5::STEP5);
            let daemon = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(
                sparks, descriptor, Default::default())?;
            let (waves, rows, unit) = (daemon.waves as usize, daemon.span_rows as usize, daemon.unit_rows as usize);
            assert_eq!(report.prefill_unit_rows, Some(daemon.unit_rows));
            assert_eq!(report.prefill_waves, Some(daemon.waves));
            assert_eq!(report.prefill_span_rows, Some(daemon.span_rows));
            let mut code = Vec::new();
            let gpus = opts
                .rtx_bytes
                .iter()
                .enumerate()
                .map(|(rank, &capacity_bytes)| {
                    let arch = ArchContext::coordinator(capacity_bytes, None);
                    code.push(
                        inventory::loaded_code_allowance(
                            "step5",
                            package,
                            ranks == 2,
                            rank as u8,
                            &arch,
                            Some(&programs),
                        )
                        .bytes,
                    );
                    GpuBudget {
                        capacity_bytes,
                        reserve_bytes: cuteafd_loader::placement::reserve::gpu("step5"),
                        baseline: Baseline::Measured {
                            free_bytes: capacity_bytes - arch.context_bytes,
                        },
                    }
                })
                .collect();
            let contract = cuteafd_loader::plan::families::step5::coordinator::CoordinatorContract::from_checkpoint(&checkpoint)?;
            let cfg = cuteafd_loader::plan::families::step5::Config::parse(&checkpoint)?;
            let mut inputs = RuntimeInputs { graph_layout: None, decode_graphs: false,
                bf16_target_head: opts.bf16_target_head,
                host: admission::HostInputs::from_contract(&contract, &cfg, ranks,
                    opts.host_memory.unwrap(), host, 8 << 30)?, resident_weight_limit: Some(crate::shared::memory::budget::resident_weight_limit()),
                spark_reply_format: cuteafd_core::SparkReplyFormat::Bf16,
                kv_format,
                inventory: Inventory {
                    gpus,
                    spark_ranks: sparks,
                    peer_access: ranks == 2,
                },
                rows,
                waves,
                unit,
                sequences: 16,
                context, serving_context: cuteafd_loader::placement::planner::Context::pinned(context),
                requested_pool: pool,
                marks: prefix_marks(&<PrefixCli as clap::Parser>::parse_from([
                    "step5", "--host-cache-bytes", &host.map_or_else(|| "auto".into(), |n| n.to_string()),
                    "--prefix-cache-entries", &entries.to_string(),
                    "--prefix-cache-mark-mib", &(opts.mimo_prefix_mark_bytes >> 20).to_string(),
                ]).prefix, 16),
                peer_diagnostics: false,
                pending_code: code,
                // As serve-step5 builds them: pool-first RTX onboarding.
                onboard: cuteafd_loader::plan::families::step5::admission::OnboardInputs::new(&catalog,
                    Some(&path), unit, ranks, waves, cuteafd_loader::placement::Onboard::Auto)?,
            };
            let live = checkpoint_admission(&checkpoint, &inputs);
            if !report.wip_launchable() {
                let error = live.err().expect("both edges must refuse the unsupported allocation");
                let refusal = error.to_string();
                if !retention_pressure {
                    // An explicit pool below the context names the floor; an
                    // automatic one names the context it could not fit.
                    let expected = if pool.is_some_and(|tokens| tokens < context) {
                        "compiled/max-context floor of 1048576"
                    } else {
                        "selected context 1048576 cannot fit"
                    };
                    assert!(refusal.contains(expected), "{refusal}");
                }
                // Live names the sampled occupancy as an owner; offline deducts
                // the same bytes from its budget. Normalize both capacity sides,
                // preserving the exact shortfall, phase, owners and context limit.
                let mut comparable = refusal.clone();
                // The device-only retention edge appends guidance and erases the
                // typed refusal, so preserve that full text while changing only
                // the two numeric fields in the coordinator capacity clause.
                for (rank, gpu) in inputs.inventory.gpus.iter().enumerate() {
                    let marker = format!("memory budget 'rtx{rank}' during ");
                    let Some((_, capacity)) = refusal.split_once(&marker) else { continue };
                    let (_, values) = capacity.split_once(" needs ").unwrap();
                    let (required, values) = values.split_once(" bytes, has ").unwrap();
                    let (available, _) = values.split_once(" bytes available").unwrap();
                    let Baseline::Measured { free_bytes } = gpu.baseline else { unreachable!() };
                    let occupied = gpu.capacity_bytes - free_bytes;
                    let normalized_required = required.parse::<u64>()?.checked_sub(occupied).unwrap();
                    let normalized_available = available.parse::<u64>()?.checked_sub(occupied).unwrap();
                    comparable = comparable.replacen(
                        &format!("needs {required} bytes, has {available} bytes available"),
                        &format!("needs {normalized_required} bytes, has {normalized_available} bytes available"), 1);
                }
                assert!(report.hints.iter().any(|hint| hint.what.contains(&comparable)),
                    "live refusal {refusal}; normalized {comparable}; offline hints {:?}", report.hints);
                continue;
            }
            let live = live?;
            if !nvfp4 && std::env::var_os("CUTEAFD_TEST_STEP5_RECONCILIATION").is_some() {
                // Isolated child: do not let the fixture helper's publication mask
                // missing owner registration in the production serve path.
                cuteafd_ffi::memory_ledger::fault::set_inputs(serde_json::json!({
                    "device.rtx0": null, "device.rtx1": null,
                    "device.spark0": null, "device.spark1": null, "device.spark2": null,
                    "device.spark3": null, "device.spark4": null, "device.spark5": null
                }));
                use clap::Parser;
                let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-step5", "--snapshot", "/m",
                    "--native-lib", path.to_str().unwrap(), "--manifest", path.to_str().unwrap(),
                    "--peers", "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4,127.0.0.1:5,127.0.0.1:6"])?;
                let crate::cli::Commands::ServeStep5(args) = cli.command else { unreachable!() };
                super::super::serve::publish_reconciliation(&args, &[1, 3], &live.solved)?;
                let registered = cuteafd_ffi::memory_ledger::fault::inputs();
                assert_eq!(registered["memory_reconciliation_provenance"]["devices"],
                    serde_json::json!([{"device": 1}, {"device": 3}]));
                assert_eq!(registered["memory_reconciliation_provenance"]["spark_world"], 6);
                for id in ["rtx0", "rtx1", "spark0", "spark1", "spark2", "spark3", "spark4", "spark5"] {
                    let row: cuteafd_core::memory_policy::DeviceBreakdown =
                        serde_json::from_value(registered[format!("device.{id}")].clone())?;
                    assert_eq!(row.device, id);
                    assert!(row.cap_bytes > 0);
                }
            }
            let mark_bytes = live.cache.ranks.iter().map(|r| r.bytes).sum::<u64>();
            let expected_slots = if entries == 0 { 0 } else if host != Some(0) { 2 } else {
                cuteafd_core::prefix::mark_slots_for(16, entries, mark_bytes, opts.mimo_prefix_mark_bytes) as usize
            };
            assert_eq!(live.marks, expected_slots);
            // Offline context/code forecasts remain budget baselines; measured
            // occupancy is instead a named owner in live plans. All other owners
            // and placement decisions must still match exactly.
            let mut comparable = live.placement.clone();
            for (items, gpu) in comparable.items.iter_mut().zip(&inputs.inventory.gpus) {
                let available_before = gpu.available();
                let future = super::super::serve::future_allocation_bytes(items, gpu)?;
                let Baseline::Measured { free_bytes } = gpu.baseline else { unreachable!() };
                let owners = items.iter().filter(|item|
                    cuteafd_loader::placement::sampled_baseline::is_source(&item.group)).collect::<Vec<_>>();
                assert_eq!(owners.len(), usize::from(gpu.capacity_bytes > free_bytes));
                assert_eq!(owners.iter().map(|item| item.bytes).sum::<u64>(), gpu.capacity_bytes - free_bytes);
                items.retain(|item| !cuteafd_loader::placement::sampled_baseline::is_source(&item.group));
                assert_eq!(future, items.iter().map(|item| item.bytes).sum::<u64>());
                assert_eq!(gpu.available(), available_before, "conversion does not change CUDA-available bytes");
                ensure!(future <= available_before, "future allocations exceed sampled free budget");
            }
            assert_eq!(report.solver_placement.as_ref(), Some(&comparable));
            assert!(live.cache.context >= 262_144 && live.cache.context <= 1_048_576);
            for (rank, reservation) in live.reservations.iter().enumerate() {
                assert_eq!(reservation.active as u64, live.cache.ranks[rank].bytes * 16);
                assert_eq!(
                    reservation.marks as u64,
                    live.cache.ranks[rank].bytes * live.marks as u64
                );
            }
            inputs.inventory.gpus[0].baseline = Baseline::Planned {
                context_bytes: 0,
                loaded_bytes: 0,
            };
            let error = checkpoint_admission(&checkpoint, &inputs)
                .err()
                .expect("live requires measured baseline");
            assert!(error.to_string().contains("measured baseline"));
        }
        Ok(())
    }

    #[test]
    fn shared_native_allocation_constants_match_ffi_owners() -> Result<()> {
        assert_eq!(
            std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>(),
            48
        );
        for rows in [1, 64, 256, 4096] {
            assert_eq!(
                cuteafd_loader::plan::families::step5::native::head_planes(rows)?[3],
                cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE
            );
        }
        Ok(())
    }
}
