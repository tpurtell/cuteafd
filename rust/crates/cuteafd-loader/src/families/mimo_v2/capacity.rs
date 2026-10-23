//! Compose MiMo's actual loaded representations and reusable workspace
//! description with its rank-partitioned caches before allocating any of them.
//! CUDA module, graph, transport, expert and external-drafter reservations are
//! supplied by the runtime, where their selected implementations are known.
use super::resident::{MimoResidentLayout, MimoResidentOptions};
use super::{MimoKvFormats, MimoV2Config, MimoWorkspaceLayout, MimoWorkspaceOptions};
use crate::plan::checkpoint::Checkpoint;
use crate::serving_capacity::{mimo_cache_geometry, CacheGeometryError};
use cuteafd_core::serving_capacity::{
    CapacityProfile, ContextLimits, DeviceCosts, MemoryReservation,
};

#[derive(Debug, Clone)]
pub struct MimoRankRuntime {
    pub device: u32,
    /// Every workspace that can remain live together: prefill lanes and decode.
    /// Native scratch comes from the selected manifest, not a guessed headroom.
    pub workspaces: Vec<(String, MimoWorkspaceOptions)>,
    /// Native modules, captures, peer/Spark receive planes, expert stores and
    /// an optional external drafter. The caller must reserve the selected path.
    pub additional: Vec<MemoryReservation>,
    /// Reservations live while target weights load, before target KV exists
    /// (for example loaded CUDA modules). Drafters loaded after target KV
    /// belong in `additional` instead, unless they also load in this phase.
    pub loading_additional: Vec<MemoryReservation>,
}

#[derive(Debug, Clone)]
pub struct MimoCapacityOptions {
    pub checkpoint_max_context_tokens: u64,
    /// The exact context that startup will consume (no silent downgrading).
    pub max_context_tokens: u64,
    pub rings: u64,
    pub mark_slots: u64,
    pub kv_cache: MimoKvFormats,
    /// Logical rank order, independent of physical device id order.
    pub ranks: Vec<MimoRankRuntime>,
    pub host_prefix_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct MimoCapacityProfiles {
    /// Fixed-only admission before any target weight allocation.
    pub loading: Vec<DeviceCosts>,
    /// Actual target storage after KV/peer attachment, before drafter packing,
    /// marks, workspaces, routed experts and Spark transport are allocated.
    /// `pool_unit_bytes` contains only the persistent target record pools.
    /// The runtime adds the selected drafter packing costs and live peer/module
    /// storage to admit that later loading phase separately from steady use.
    pub post_target_kv: Vec<DeviceCosts>,
    /// Persistent admission, including the selected logical KV pool.
    pub steady: CapacityProfile,
}

fn product(label: &'static str, dims: &[u64]) -> Result<u64, CacheGeometryError> {
    dims.iter()
        .try_fold(1u64, |n, &dim| n.checked_mul(dim))
        .ok_or(CacheGeometryError::Overflow(label))
}

fn add(label: &'static str, left: u64, right: u64) -> Result<u64, CacheGeometryError> {
    left.checked_add(right)
        .ok_or(CacheGeometryError::Overflow(label))
}

fn reserve(costs: &mut Vec<MemoryReservation>, name: &str, bytes: u64) {
    if bytes > 0 {
        costs.push(MemoryReservation {
            name: name.into(),
            bytes,
        });
    }
}

/// This builder intentionally does not invent missing runtime reservations.
/// The engine obtains native/module/draft/transport costs before it calls this
/// function, then hands the resulting profile to the shared capacity resolver.
pub fn mimo_capacity_profiles(
    checkpoint: &Checkpoint,
    cfg: &MimoV2Config,
    resident_options: MimoResidentOptions,
    options: &MimoCapacityOptions,
) -> Result<MimoCapacityProfiles, CacheGeometryError> {
    if options.max_context_tokens == 0
        || options.checkpoint_max_context_tokens == 0
        || options.max_context_tokens > options.checkpoint_max_context_tokens
        || options.rings == 0
        || options.ranks.len() != resident_options.coordinator_ranks
    {
        return Err(CacheGeometryError::Unsupported {
            family: "mimo_v2",
            what: "capacity context, rings or physical rank mapping",
        });
    }
    let resident = MimoResidentLayout::new(checkpoint, cfg, &resident_options)?;
    let cache = mimo_cache_geometry(
        cfg,
        resident_options.layers,
        resident_options.coordinator_ranks,
        options.kv_cache,
        resident_options.native_mtp_layers,
    )?;
    let before_kv_state = if resident_options.native_mtp_layers > 0 {
        // MTP stage rings/hidden history and fixed buffers are allocated in
        // with_engine before target KV, interleaved with MTP weight loads.
        // Derive their exact increment from the shared cache description.
        let target = mimo_cache_geometry(
            cfg,
            resident_options.layers,
            resident_options.coordinator_ranks,
            options.kv_cache,
            0,
        )?;
        let active = cache.ranks[0]
            .active_state_per_sequence_bytes
            .checked_sub(target.ranks[0].active_state_per_sequence_bytes)
            .ok_or(CacheGeometryError::Overflow("MiMo pre-KV MTP active state"))?;
        let fixed = cache.ranks[0]
            .fixed_state_bytes
            .checked_sub(target.ranks[0].fixed_state_bytes)
            .ok_or(CacheGeometryError::Overflow("MiMo pre-KV MTP fixed state"))?;
        add(
            "MiMo pre-KV MTP state",
            product("MiMo pre-KV MTP rings", &[options.rings, active])?,
            fixed,
        )?
    } else {
        0
    };
    let mark_bytes = cache
        .ranks
        .iter()
        .try_fold(0u64, |n, rank| n.checked_add(rank.retained_mark_bytes))
        .ok_or(CacheGeometryError::Overflow("MiMo aggregate exact mark"))?;
    let mut devices = Vec::with_capacity(options.ranks.len());
    let mut loading = Vec::with_capacity(options.ranks.len());
    let mut post_target_kv = Vec::with_capacity(options.ranks.len());
    for (rank, runtime) in options.ranks.iter().enumerate() {
        if options.ranks[..rank]
            .iter()
            .any(|r| r.device == runtime.device)
            || runtime.workspaces.is_empty()
        {
            return Err(CacheGeometryError::Unsupported {
                family: "mimo_v2",
                what: "duplicate physical device or missing live workspace",
            });
        }
        let storage = cache.ranks[rank];
        let mut loading_costs = resident.ranks[rank].clone();
        if rank == 0 {
            reserve(
                &mut loading_costs,
                "loading.mtp_state_before_target_kv",
                before_kv_state,
            );
        }
        // Each rank's loader drains its source peak before target KV. This
        // constrains startup without reducing the steady KV/expert budget.
        reserve(
            &mut loading_costs,
            "loading.temporary",
            resident.loading_temporary_rank_bytes[rank],
        );
        loading_costs.extend(runtime.loading_additional.iter().cloned());
        loading_costs
            .iter()
            .try_fold(0u64, |n, reservation| n.checked_add(reservation.bytes))
            .ok_or(CacheGeometryError::Overflow(
                "MiMo loading phase reservations",
            ))?;
        loading.push(DeviceCosts {
            device: runtime.device,
            reservations: loading_costs,
            pool_unit_bytes: 0,
        });
        let mut costs = resident.ranks[rank].clone();
        reserve(
            &mut costs,
            "state.active_rings",
            product(
                "MiMo active rings",
                &[options.rings, storage.active_state_per_sequence_bytes],
            )?,
        );
        reserve(&mut costs, "state.fixed", storage.fixed_state_bytes);
        post_target_kv.push(DeviceCosts {
            device: runtime.device,
            reservations: costs.clone(),
            pool_unit_bytes: storage.persistent_unit_bytes,
        });
        if options.mark_slots > 0 && mark_bytes > 0 {
            reserve(
                &mut costs,
                "prefix.exact_mark_arena",
                product(
                    "MiMo retained marks",
                    &[options.mark_slots, storage.retained_mark_bytes],
                )?
                .max(256),
            );
        }
        reserve(
            &mut costs,
            "state.speculative_replay",
            storage.speculative_replay_bytes,
        );
        let mut unit_bytes = add(
            "MiMo persistent pool metadata",
            storage.persistent_unit_bytes,
            storage.pool_metadata_unit_bytes,
        )?;
        let mut kv_shadow = super::workspace::MimoPrefillKvShadowPlan::default();
        for (name, workspace) in &runtime.workspaces {
            if workspace.max_context != options.max_context_tokens
                || workspace.lead != (rank == 0)
                || workspace.kv_cache != options.kv_cache
            {
                return Err(CacheGeometryError::Unsupported {
                    family: "mimo_v2",
                    what: "workspace context, rank or KV format disagrees with capacity",
                });
            }
            let layout = MimoWorkspaceLayout::new(
                cfg,
                MimoWorkspaceOptions {
                    pool_pages: 0,
                    ..*workspace
                },
            )?;
            costs.extend(kv_shadow.workspace_reservations(name, &layout, *workspace)?);
            // The allocator uses max(256, pages * table_rows * 4). Reserving
            // the fixed minimum plus its linear term is safe at every pool
            // size, overcounting by at most 256 bytes per workspace.
            let table_rows = if workspace.decode { workspace.rows } else { workspace.waves };
            unit_bytes = add(
                "MiMo workspace pool tables",
                unit_bytes,
                product("MiMo workspace page indices", &[table_rows, 4])?,
            )?;
        }
        costs.extend(kv_shadow.reservation());
        costs.extend(runtime.additional.iter().cloned());
        costs
            .iter()
            .try_fold(0u64, |n, reservation| n.checked_add(reservation.bytes))
            .ok_or(CacheGeometryError::Overflow("MiMo full fixed reservations"))?;
        devices.push(DeviceCosts {
            device: runtime.device,
            reservations: costs,
            pool_unit_bytes: unit_bytes,
        });
    }
    Ok(MimoCapacityProfiles {
        loading,
        post_target_kv,
        steady: CapacityProfile {
            // MiMo attention consumes dynamic key/page extents; it does not index
            // the GLM/Qwen MAX_CONTEXT-sized exported attention map.
            context: ContextLimits {
                checkpoint_max_tokens: options.checkpoint_max_context_tokens,
                compiled_index_max_tokens: None,
            },
            pool_unit_rows: cache.logical_unit_rows,
            devices,
            host_prefix_bytes: options.host_prefix_bytes,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::super::{MimoAttentionWorkspace, MimoPrefillOutput};
    use super::*;
    use crate::plan::testing::{mimo_flash_config, mimo_flash_tensors, write_snapshot};

    fn weights(ranks: usize) -> MimoResidentOptions {
        MimoResidentOptions {
            layers: 2,
            coordinator_ranks: ranks,
            checkpoint_tp: 1,
            native_mtp_layers: 0,
            gpu_embedding: true,
            head_format: super::super::projection::MimoProjectionRepresentation::Bf16,
            output_formats: (0..2)
                .map(|layer| {
                    (
                        format!("model.layers.{layer}.self_attn.o_proj.weight"),
                        super::super::projection::MimoProjectionRepresentation::Bf16,
                    )
                })
                .collect(),
        }
    }

    fn options(ranks: usize) -> MimoCapacityOptions {
        MimoCapacityOptions {
            checkpoint_max_context_tokens: 1 << 20,
            max_context_tokens: 32768,
            rings: 20,
            mark_slots: 42,
            kv_cache: MimoKvFormats::default(),
            host_prefix_bytes: 64 << 30,
            ranks: (0..ranks)
                .map(|rank| MimoRankRuntime {
                    // Rank0 need not be physical device0.
                    device: (1 - rank) as u32,
                    additional: vec![MemoryReservation {
                        name: "native.modules".into(),
                        bytes: 1 << 20,
                    }],
                    loading_additional: vec![MemoryReservation {
                        name: "native.modules".into(),
                        bytes: 1 << 20,
                    }],
                    workspaces: [("prefill", false, 4096), ("decode", true, 64)]
                        .into_iter()
                        .map(|(name, decode, rows)| {
                            (
                                name.into(),
                                MimoWorkspaceOptions {
                                    rows,
                                    waves: 1,
                                    decode,
                                    lead: rank == 0,
                                    with_head: rank == 0,
                                    spark: rank == 0,
                                    expert_input: super::super::MimoExpertInput::Fp8,
                                    max_context: 32768,
                                    pool_pages: 0,
                                    kv_cache: MimoKvFormats::default(),
                                    attention: MimoAttentionWorkspace::Global,
                                    prefill_output: MimoPrefillOutput::AllRows,
                                    native_scratch_bytes: 16 << 20,
                                    head_workspace_bytes: 32 << 20,
                                },
                            )
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    fn options_with_swa(ranks: usize, sliding: super::super::MimoKvCache) -> MimoCapacityOptions {
        let mut options = options(ranks);
        options.kv_cache.sliding = sliding;
        for rank in &mut options.ranks {
            for (_, workspace) in &mut rank.workspaces {
                workspace.kv_cache = options.kv_cache;
            }
        }
        options
    }

    #[test]
    fn rank_mapping_keeps_lead_weights_and_partitions_only_kv_heads() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(
            dir.path(),
            &mimo_flash_config(),
            &mimo_flash_tensors(),
            Some(1),
        );
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        for (sliding, record) in [(super::super::MimoKvCache::Bf16, 5120),
            (super::super::MimoKvCache::Int8, 2880), (super::super::MimoKvCache::Fp8, 2720)] {
            let options = options_with_swa(2, sliding);
            let phases = mimo_capacity_profiles(&checkpoint, &cfg, weights(2), &options).unwrap();
            let profile = &phases.steady;
            assert_eq!(
                profile.devices.iter().map(|d| d.device).collect::<Vec<_>>(),
                [1, 0]
            );
            assert_eq!(profile.pool_unit_rows, 64);
            // The one full-attention layer has720B/row per rank. All65 page
            // index rows (64decode+1prefill) are owned independently per rank.
            assert_eq!(profile.devices[0].pool_unit_bytes, 64 * 720 + 65 * 4);
            assert_eq!(
                profile.devices[0].pool_unit_bytes,
                profile.devices[1].pool_unit_bytes
            );
            assert!(profile.devices[0]
                .reservations
                .iter()
                .any(|r| r.name == "lm_head.bf16"));
            assert!(profile.devices[1]
                .reservations
                .iter()
                .all(|r| r.name != "lm_head.bf16"));
            assert_eq!(profile.host_prefix_bytes, 64 << 30);
            let arena = |rank: usize| {
                profile.devices[rank]
                    .reservations
                    .iter()
                    .find(|r| r.name == "prefix.exact_mark_arena")
                    .unwrap()
                    .bytes
            };
            assert_eq!(arena(0), 42 * 128 * record / 2);
            assert_eq!(arena(0), arena(1));
            assert!(profile
                .devices
                .iter()
                .all(|d| d.reservations.iter().all(|r| r.name != "loading.temporary")));
            assert!(phases.loading.iter().all(|d| d.pool_unit_bytes == 0));
            let loading_extra = phases.loading[0]
                .reservations
                .iter()
                .find(|r| r.name == "loading.temporary")
                .unwrap();
            assert_eq!(loading_extra.bytes, 4096 * 8192 * 2);
            for phase in &phases.post_target_kv {
                assert_eq!(phase.pool_unit_bytes, 64 * 720);
                assert!(phase
                    .reservations
                    .iter()
                    .any(|cost| cost.name == "state.active_rings"));
                assert!(phase
                    .reservations
                    .iter()
                    .all(|cost| cost.name != "context.rope_tables"));
                assert!(!phase
                    .reservations
                    .iter()
                    .any(|cost| cost.name.starts_with("prefix.")
                        || cost.name.starts_with("prefill.")
                        || cost.name.starts_with("decode.")
                        || cost.name.starts_with("loading.")
                        || cost.name == "native.modules"));
            }
        }
    }

    #[test]
    fn serving_last_row_contract_removes_only_the_lead_prefill_logits() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(
            dir.path(),
            &mimo_flash_config(),
            &mimo_flash_tensors(),
            Some(1),
        );
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        let mut diagnostic = options(2);
        for rank in &mut diagnostic.ranks {
            let mut first_lane = rank.workspaces[0].1;
            first_lane.with_head = false;
            rank.workspaces
                .push(("prefill_first_lane".into(), first_lane));
        }
        let mut serving = diagnostic.clone();
        for rank in &mut serving.ranks {
            for (_, workspace) in &mut rank.workspaces {
                workspace.prefill_output = MimoPrefillOutput::LastRow;
            }
        }
        let all = mimo_capacity_profiles(&checkpoint, &cfg, weights(2), &diagnostic).unwrap();
        let last = mimo_capacity_profiles(&checkpoint, &cfg, weights(2), &serving).unwrap();
        assert_eq!(all.loading, last.loading);
        assert_eq!(all.post_target_kv, last.post_target_kv);
        assert_eq!(all.steady.host_prefix_bytes, last.steady.host_prefix_bytes);
        for rank in 0..2 {
            let before = &all.steady.devices[rank];
            let after = &last.steady.devices[rank];
            assert_eq!(before.pool_unit_bytes, after.pool_unit_bytes);
            let mut changed = 0;
            for (a, b) in before.reservations.iter().zip(&after.reservations) {
                assert_eq!(a.name, b.name);
                if a.bytes != b.bytes {
                    changed += 1;
                    assert_eq!(rank, 0);
                    assert_eq!(a.name, "prefill.logits");
                    assert_eq!(a.bytes - b.bytes, (16384 - 1) * cfg.vocab_size as u64 * 4);
                    assert_eq!(b.bytes, cfg.vocab_size as u64 * 4);
                }
            }
            assert_eq!(changed, usize::from(rank == 0));
        }
    }

    #[test]
    fn pool_affine_reservation_covers_minimum_allocations_at_every_page_count() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(
            dir.path(),
            &mimo_flash_config(),
            &mimo_flash_tensors(),
            Some(1),
        );
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        let options = options(1);
        let profile = mimo_capacity_profiles(&checkpoint, &cfg, weights(1), &options)
            .unwrap()
            .steady;
        let fixed_workspace: u64 = profile.devices[0]
            .reservations
            .iter()
            .filter(|r| {
                r.name.starts_with("prefill.")
                    || r.name.starts_with("decode.")
                    || r.name == "state.prefill_kv_wide"
            })
            .map(|r| r.bytes)
            .sum();
        for pages in [0, 1, 2, 7, 64, 4096] {
            let actual_workspaces: u64 = options.ranks[0]
                .workspaces
                .iter()
                .map(|(_, shape)| {
                    MimoWorkspaceLayout::new(
                        &cfg,
                        MimoWorkspaceOptions {
                            pool_pages: pages,
                            ..*shape
                        },
                    )
                    .unwrap()
                    .device_bytes()
                    .unwrap()
                })
                .sum();
            let records = pages * 64 * 1440;
            let predicted = fixed_workspace + pages * profile.devices[0].pool_unit_bytes;
            assert!(predicted >= actual_workspaces + records, "pages={pages}");
            assert!(
                predicted - actual_workspaces - records <= 2 * 256,
                "pages={pages}"
            );
        }
    }

    #[test]
    fn mismatched_physical_or_workspace_inputs_do_not_reach_the_resolver() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(
            dir.path(),
            &mimo_flash_config(),
            &mimo_flash_tensors(),
            Some(1),
        );
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        let mut options = options(2);
        options.ranks[1].device = options.ranks[0].device;
        assert!(mimo_capacity_profiles(&checkpoint, &cfg, weights(2), &options).is_err());
        options.ranks[1].device = 0;
        options.ranks[1].workspaces[0].1.max_context = 1 << 20;
        assert!(mimo_capacity_profiles(&checkpoint, &cfg, weights(2), &options).is_err());
    }

    #[test]
    fn native_mtp_state_is_admitted_while_weights_load_before_target_kv() {
        use crate::plan::testing::t;
        let mut tensors = mimo_flash_tensors();
        let stage: Vec<_> = tensors
            .iter()
            .filter_map(|(name, dtype, shape)| {
                let suffix = name.strip_prefix("model.layers.1.")?;
                (suffix.starts_with("self_attn.") || suffix.ends_with("layernorm.weight")).then(
                    || {
                        (
                            format!(
                                "model.mtp.layers.0.{}",
                                suffix.replace("post_attention_layernorm", "pre_mlp_layernorm")
                            ),
                            *dtype,
                            shape.clone(),
                        )
                    },
                )
            })
            .chain(tensors.iter().filter_map(|(name, dtype, shape)| {
                let suffix = name.strip_prefix("model.layers.0.mlp.")?;
                Some((
                    format!("model.mtp.layers.0.mlp.{suffix}"),
                    *dtype,
                    shape.clone(),
                ))
            }))
            .collect();
        tensors.extend(stage);
        tensors.push(t(
            "model.mtp.layers.0.eh_proj.weight",
            "BF16",
            &[4096, 8192],
        ));
        for norm in ["enorm", "hnorm", "final_layernorm"] {
            tensors.push(t(
                format!("model.mtp.layers.0.{norm}.weight"),
                "BF16",
                &[4096],
            ));
        }
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &mimo_flash_config(), &tensors, Some(1));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        for (sliding, record) in [(super::super::MimoKvCache::Bf16, 5120),
            (super::super::MimoKvCache::Int8, 2880), (super::super::MimoKvCache::Fp8, 2720)] {
            let profiles = mimo_capacity_profiles(
                &checkpoint,
                &cfg,
                MimoResidentOptions {
                    native_mtp_layers: 1,
                    output_formats: weights(2)
                        .output_formats
                        .into_iter()
                        .chain(std::iter::once((
                            "model.mtp.layers.0.self_attn.o_proj.weight".into(),
                            super::super::projection::MimoProjectionRepresentation::Bf16,
                        )))
                        .collect(),
                    ..weights(2)
                },
                &options_with_swa(2, sliding),
            )
            .unwrap();
            let name = "loading.mtp_state_before_target_kv";
            let state = profiles.loading[0]
                .reservations
                .iter()
                .find(|r| r.name == name)
                .unwrap();
            // One unsplit SWA stage, a 256-row hidden history, and the five
            // fixed hidden buffers plus stage ids/index. All are lead-only.
            assert_eq!(
                state.bytes,
                20 * (256 * record + 256 * 4096 * 2) + 12 * 64 * 4096 + 2 * 64 * 4 + 256
            );
            assert!(profiles.loading[1]
                .reservations
                .iter()
                .all(|r| r.name != name));
            assert!(profiles
                .steady
                .devices
                .iter()
                .all(|d| d.reservations.iter().all(|r| r.name != name)));
        }
    }
}
