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
    measured(inputs)?;
    let admitted = if depth == 0 { checkpoint_admission(checkpoint, inputs)? }
        else { admission::admit_mtp(checkpoint, inputs, depth)? };
    if depth > 0 {
        for stage in 0..depth {
            super::mtp::InputPlan::prepare(programs, checkpoint, stage, inputs.rows)?;
        }
    }
    let ownership = &admitted.ownership;
    for rank in 0..ownership.ranks() {
        let native =
            Execution::serving_admission(programs, checkpoint, ownership, rank, inputs.rows)?;
        let shared = cuteafd_loader::plan::families::step5::native::rank_admission(
            checkpoint,
            ownership,
            rank,
            inputs.rows,
        )?;
        ensure!(
            native.immutable == shared.immutable && native.workspace == shared.workspace,
            "Step-5 native allocation owners differ from shared admission"
        );
    }
    Ok(admitted)
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
    admission::admit(checkpoint, inputs)
}
#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::{
        placement::*,
        plan::{self, layout::LayoutOptions, testing, ExpertPlacement, PlanOptions},
    };

    #[test]
    fn public_max_card_planner_equals_measured_daemon_before_allocation() -> Result<()> {
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
        testing::write_snapshot(dir.path(), &source["config"], &tensors, None);
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
        let programs = ProgramSet::from_manifest(&manifest, &["step5", "step52"]);
        let checkpoint = Checkpoint::coordinator(dir.path(), false, false)?;
        #[derive(clap::Parser)]
        struct PrefixCli {
            #[command(flatten)]
            prefix: crate::shared::prefix::PrefixArgs,
        }
        for (context, pool, launchable, host, entries) in [
            (262_144, Some(262_144), true, None, 20),
            (1_048_576, None, true, None, 20),
            (1_048_576, Some(262_144), false, None, 20),
            (1_048_576, None, true, Some(0), 20),
            (1_048_576, None, true, Some(1 << 30), 20),
            (1_048_576, None, true, None, 0),
            (1_048_576, None, true, Some(0), 0),
        ] {
            let opts = LayoutOptions {
                rtx_bytes: vec![96 << 30; 2],
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
                    placement: ExpertPlacement::Sparks { ranks: 6 },
                    draft_ownership: Some(plan::spec::DraftOwnership::None),
                    layout: Some(opts.clone()),
                    ..Default::default()
                },
            )?;
            assert_eq!(report.wip_launchable(), launchable, "{:?}", report.hints);
            assert!(!report.executable());
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
                            "step5:fp8",
                            true,
                            rank as u8,
                            &arch,
                            Some(&programs),
                        )
                        .bytes,
                    );
                    GpuBudget {
                        capacity_bytes,
                        headroom_bytes: opts.headroom_bytes,
                        baseline: Baseline::Measured {
                            free_bytes: capacity_bytes - arch.context_bytes,
                        },
                    }
                })
                .collect();
            let mut inputs = RuntimeInputs {
                inventory: Inventory {
                    gpus,
                    spark_ranks: 6,
                    peer_access: true,
                },
                rows: 256,
                sequences: 16,
                lanes: 2,
                context,
                requested_pool: pool,
                marks: prefix_marks(&<PrefixCli as clap::Parser>::parse_from([
                    "step5", "--host-cache-bytes", &host.map_or_else(|| "auto".into(), |n| n.to_string()),
                    "--prefix-cache-entries", &entries.to_string(),
                    "--prefix-cache-mark-mib", &(opts.mimo_prefix_mark_bytes >> 20).to_string(),
                ]).prefix, 16),
                peer_diagnostics: false,
                pending_code: code,
                startup: vec![],
            };
            let live = checkpoint_admission(&checkpoint, &inputs);
            if !launchable {
                let error = live.err().expect("explicit pool below requested context");
                let refusal = error.to_string();
                assert!(refusal.contains("compiled/max-context floor of 1048576"));
                assert!(report.hints.iter().any(|hint| hint.what.contains(&refusal)));
                continue;
            }
            let live = live?;
            let mark_bytes = live.cache.ranks.iter().map(|r| r.bytes).sum::<u64>();
            let expected_slots = if entries == 0 { 0 } else if host != Some(0) { 2 } else {
                cuteafd_core::prefix::mark_slots_for(16, entries, mark_bytes, opts.mimo_prefix_mark_bytes) as usize
            };
            assert_eq!(live.marks, expected_slots);
            assert_eq!(report.solver_placement.as_ref(), Some(&live.placement));
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
