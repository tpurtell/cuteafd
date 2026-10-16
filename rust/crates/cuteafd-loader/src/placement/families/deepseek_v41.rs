//! V4.1 CED ownership stays fixed; pool and local experts use shared admission.
use super::super::*;

/// Reserved lazy graph envelope; small cards charge their measured 2 GiB bank
/// through the admission sample instead.
pub fn graph_reserve(total: u64, ranks: usize) -> u64 {
    if total <= 32 << 30 { 0 } else if ranks == 1 { 3264 << 20 } else { 1472 << 20 }
}

pub fn runtime_reserve(total: u64, ranks: usize, automatic: bool) -> u64 {
    if total <= 32 << 30 { 0 } else if automatic { 3 << 30 }
    else if ranks == 1 { 2 << 30 } else { 800 << 20 }
}

/// V4.1 requires its full admitted context; other Spark families may clamp it.
pub fn validate_pool_floor(pool: &PoolPolicy, placement: &Placement) -> Result<(), PlacementError> {
    if placement.pool_tokens < pool.floor {
        return Err(PlacementError::BelowFloor {
            pool: placement.pool_tokens,
            floor: pool.floor,
            layers: placement.onboard_layers,
            short: pool.floor - placement.pool_tokens,
            limits: placement.pool_limits.clone(),
        });
    }
    Ok(())
}

/// Exact expert allocation extents, supplied either by linked native info or
/// the corresponding published AOT manifest. No device-specific estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeWorkspace {
    pub scratch_bytes: u64,
    pub kernel_intermediate: u64,
}

impl NativeWorkspace {
    pub fn from_manifest(
        manifest: &serde_json::Value,
        role: &str,
        rows: u64,
        hidden: u64,
        intermediate: u64,
        experts: u64,
        topk: u64,
    ) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        ensure!(
            manifest["role"].as_str() == Some(role),
            "V4.1 expert manifest role mismatch"
        );
        for (key, expected) in [
            ("hidden", hidden),
            ("intermediate", intermediate),
            ("experts", experts),
            ("topk", topk),
        ] {
            ensure!(
                manifest["geometry"][key].as_u64() == Some(expected),
                "V4.1 expert manifest geometry mismatch: {key}"
            );
        }
        let capacities = expert_capacities(rows, role == "rtx_backbone")?;
        let variants = manifest["variants"]
            .as_array()
            .context("missing expert capacity variants")?;
        let mut scratch_bytes = 0;
        for capacity in capacities {
            let matching: Vec<_> = variants
                .iter()
                .filter(|v| v.get("requested_rows").unwrap_or(&v["capacity_rows"]).as_u64() == Some(capacity))
                .collect();
            ensure!(
                matching.len() == 1,
                "missing or duplicate V4.1 expert capacity {capacity}"
            );
            scratch_bytes = scratch_bytes.max(
                matching[0]["core_scratch_nbytes"]
                    .as_u64()
                    .context("missing exact V4.1 expert scratch extent")?,
            );
        }
        let kernel_intermediate = manifest["geometry"]["kernel_intermediate"]
            .as_u64()
            .context("missing expert resident width")?;
        ensure!(
            kernel_intermediate >= intermediate && kernel_intermediate % 32 == 0,
            "unsupported V4.1 expert resident width"
        );
        Ok(Self {
            scratch_bytes,
            kernel_intermediate,
        })
    }

    /// One lane, matching LocalExpertWave or OwnerPair respectively. The
    /// owner path retains route planes even when a kernel can sum tokens.
    pub fn lane_bytes(
        self,
        rows: u64,
        hidden: u64,
        topk: u64,
        dual: bool,
        nvfp4: bool,
    ) -> anyhow::Result<u64> {
        use anyhow::Context;
        let columns = if dual { topk.checked_mul(if nvfp4 { 2 } else { 4 })
            .context("V4.1 expert route extent overflow")? } else { 2 };
        let output = rows
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(columns))
            .context("V4.1 expert output overflow")?;
        let owner = if dual {
            owner_workspace(rows, hidden, topk)?
        } else {
            0
        };
        self.scratch_bytes
            .checked_add(output)
            .and_then(|n| n.checked_add(owner))
            .context("V4.1 expert lane overflow")
    }
}

/// Enumerate retained scratch specializations independently of numerical
/// dispatch: MXFP4/EXL3 use admitted reduction plans; NVFP4 keeps FirstCapacity.
pub fn expert_capacities(rows: u64, local: bool) -> anyhow::Result<Vec<u64>> {
    use anyhow::{ensure, Context};
    ensure!((1..=4096).contains(&rows), "invalid V4.1 expert capacity");
    let capacities = [1, 16, 80, 256, 1024, 4096];
    let compiled = capacities
        .into_iter()
        .find(|&n| n >= rows)
        .context("missing expert capacity")?;
    if local {
        ensure!(
            rows == compiled,
            "local expert rows must be a compiled capacity"
        );
        Ok([1, 16, 80]
            .into_iter()
            .filter(|&n| n < rows)
            .chain([rows])
            .collect())
    } else {
        Ok(capacities.into_iter().filter(|&n| n <= compiled).collect())
    }
}

/// EXL3 needs a canonical wide decode export even for narrow lane admission.
/// Keep the existing explicit prefill state set for local owners.
pub fn exl3_capacities(rows: u64, local: bool) -> anyhow::Result<Vec<u64>> {
    expert_capacities(if rows > 1 && rows < 80 { 80 } else { rows }, local)
}

/// The native packer's four resident planes, independent of staging format.
/// NVFP4 scales remain the checkpoint's calibrated K16 planes and vectors.
pub fn resident_planes(hidden: u64, stored: u64, nvfp4: bool) -> anyhow::Result<[u64; 4]> {
    anyhow::ensure!(
        hidden > 0 && hidden % 128 == 0 && stored > 0 && stored % 32 == 0,
        "invalid expert packed geometry"
    );
    let values = hidden
        .checked_mul(stored)
        .ok_or_else(|| anyhow::anyhow!("expert planes overflow"))?;
    if nvfp4 {
        let scale = |n: u64, k: u64| n.div_ceil(128).checked_mul(128)
            .and_then(|n| n.checked_mul((k / 16).div_ceil(4)))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("expert scale plane overflow"));
        Ok([
            values,
            scale(stored.checked_mul(2).ok_or_else(|| anyhow::anyhow!("expert gate width overflow"))?, hidden)?,
            values / 2,
            scale(hidden, stored)?,
        ])
    } else {
        Ok([values, values / 16, values / 2, values / 32])
    }
}

pub fn resident_bytes(hidden: u64, stored: u64, experts: u64, nvfp4: bool) -> anyhow::Result<u64> {
    use anyhow::Context;
    let planes = resident_planes(hidden, stored, nvfp4)?;
    planes
        .into_iter()
        .try_fold(if nvfp4 { 16u64 } else { 0 }, |sum, bytes| {
            sum.checked_add(bytes)
                .context("expert resident extent overflow")
        })?
        .checked_mul(experts)
        .context("expert resident count overflow")
}

/// Header-derived checkpoint staging and compressed residency, paired with
/// the compiled resident width rather than an assumed padding policy.
pub fn expert_cost(
    catalog: &crate::OfficialV41Catalog,
    layer: usize,
    world: usize,
    rank: usize,
    stored: u64,
) -> anyhow::Result<Bytes2> {
    use anyhow::ensure;
    ensure!(
        matches!(world, 1 | 2) && rank < world && layer < 40,
        "invalid V4.1 RTX expert selection"
    );
    if let Some(exl3) = catalog.exl3() {
        return Ok(Bytes2 {
            resident: exl3
                .residency(crate::V41Exl3Layer::Backbone(layer), world, rank)?
                .device_arena_layout()?
                .1 as u64,
            staging: 0,
        });
    }
    let selection = if world == 1 {
        crate::V41ExpertSelection::BackboneFull { layer, expert: 0 }
    } else {
        crate::V41ExpertSelection::BackboneTp2 {
            layer,
            expert: 0,
            rank,
        }
    };
    let nvfp4 = catalog.nvfp4().is_some();
    let staging = if nvfp4 {
        catalog.nvfp4_expert_staging(selection)?.staging_bytes()
    } else {
        catalog.expert_staging(selection)?.staging_bytes()
    };
    let shape = catalog.routed_experts();
    Ok(Bytes2 {
        resident: resident_bytes(shape.hidden as u64, stored, shape.experts as u64, nvfp4)?,
        staging: staging as u64,
    })
}

/// Exact conditional expert owners: immutable checkpoint residency and two
/// independent serving lanes. Fixed target lanes stay in the baseline.
pub fn expert_inventory(
    catalog: &crate::OfficialV41Catalog,
    programs: &std::path::Path,
    rows: u64,
    gpus: usize,
) -> anyhow::Result<(Vec<ExpertCost>, u64)> {
    use anyhow::{ensure, Context};
    ensure!(matches!(gpus, 1 | 2), "V4.1 supports one or two RTX owners");
    let shape = catalog.routed_experts();
    ensure!(
        shape.layers == 40 && shape.hidden == 5120,
        "unsupported V4.1 expert geometry"
    );
    let parent = programs
        .parent()
        .context("program manifest has no parent")?;
    let local = gpus == 1;
    let role = if local { "rtx_backbone" } else { "rtx_tp2" };
    let capacity = if local {
        cuteafd_core::coordinator_programs::v41_aot_rows(u32::try_from(rows)?)
            .context("invalid V4.1 prefill rows")? as u64
    } else {
        rows
    };
    let (stored, routed) = if let Some(exl3) = catalog.exl3() {
        let tiers: String = exl3.decoder_tiers().iter().map(usize::to_string).collect();
        let layout = if local { "rtx-tp1" } else { "rtx-tp2" };
        let stem = format!("exl3-k{tiers}");
        let package = [
            parent.join("exl3").join(&stem).join(layout),
            parent.join("../lib/exl3").join(&stem).join(layout),
            parent.join("exl3").join(layout),
        ]
        .into_iter()
        .find(|p| p.is_dir())
        .context("missing V4.1 EXL3 expert workspace package")?;
        let manifests = exl3_capacities(capacity, local)?
            .into_iter()
            .map(|c| {
                let path = package.join(format!("m{c}/v41_exl3.json"));
                let value: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
                )?;
                ensure!(
                    value["capacity"].as_u64() == Some(c)
                        && value["hidden"].as_u64() == Some(shape.hidden as u64)
                        && value["experts"].as_u64() == Some(shape.experts as u64)
                        && value["intermediate"].as_u64() == Some((shape.intermediate / gpus) as u64)
                        && value["top_k"].as_u64() == Some(shape.topk as u64),
                    "V4.1 EXL3 workspace geometry mismatch"
                );
                Ok(value)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let scratch = crate::serving_capacity::exl3_workspace_bytes(&manifests, true)?;
        let output = capacity * shape.hidden as u64 * if local { 2 } else { 4 };
        (
            0,
            scratch
                + output
                + if local {
                    0
                } else {
                    owner_workspace(capacity, shape.hidden as u64, shape.topk as u64)?
                },
        )
    } else {
        let format = if catalog.nvfp4().is_some() {
            "nvfp4"
        } else {
            "mxfp4"
        };
        let path = parent
            .join("V41_EXPERT_WORKSPACES")
            .join(format!("{format}_{role}.json"));
        let value =
            serde_json::from_slice(&std::fs::read(&path).with_context(|| {
                format!("missing exact workspace manifest {}", path.display())
            })?)?;
        let workspace = NativeWorkspace::from_manifest(
            &value,
            role,
            capacity,
            shape.hidden as u64,
            (shape.intermediate / gpus) as u64,
            shape.experts as u64,
            shape.topk as u64,
        )?;
        (
            workspace.kernel_intermediate,
            workspace.lane_bytes(
                capacity,
                shape.hidden as u64,
                shape.topk as u64,
                !local,
                catalog.nvfp4().is_some(),
            )?,
        )
    };
    let lane = if local {
        routed
    } else {
        let fp8: serde_json::Value =
            serde_json::from_slice(&std::fs::read(parent.join("V41_FP8_AOT.json"))?)?;
        routed
            .checked_add(shared_workspace(&fp8, rows)?)
            .and_then(|n| n.checked_add(rows * (10240 + 5280 + 24 + 24 + 10240)))
            .context("V4.1 FFN workspace overflow")?
    };
    let experts = (0..40)
        .map(|layer| {
            let mut cost = ExpertCost {
                whole: Bytes2::default(),
                half: [Bytes2::default(); 2],
                tp2: !local,
                spark_ok: true,
            };
            if local {
                cost.whole = expert_cost(catalog, layer, 1, 0, stored)?;
            } else {
                for rank in 0..2 {
                    cost.half[rank] = expert_cost(catalog, layer, 2, rank, stored)?;
                }
            }
            Ok(cost)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((
        experts,
        lane.checked_mul(2).context("V4.1 serving lanes overflow")?,
    ))
}

pub fn owner_workspace(rows: u64, hidden: u64, topk: u64) -> anyhow::Result<u64> {
    let columns = topk.checked_mul(4).and_then(|n| n.checked_add(2))
        .ok_or_else(|| anyhow::anyhow!("V4.1 owner route extent overflow"))?;
    rows.checked_mul(hidden)
        .and_then(|n| n.checked_mul(columns))
        .ok_or_else(|| anyhow::anyhow!("V4.1 owner workspace overflow"))
}

/// The two TP2 FP8 projection plans keep each capacity's aligned arena, while
/// routed experts share a maximum-sized scratch arena across capacities.
pub fn shared_workspace(manifest: &serde_json::Value, rows: u64) -> anyhow::Result<u64> {
    use anyhow::{ensure, Context};
    let compiled = expert_capacities(rows, false)?
        .into_iter()
        .last()
        .context("missing FP8 capacity")?;
    let capacities: Vec<_> = [1, 16, 80]
        .into_iter()
        .filter(|&n| n < compiled)
        .chain([compiled])
        .collect();
    let variants = manifest["variants"]
        .as_array()
        .context("missing V4.1 FP8 variants")?;
    let mut scratch = 0u64;
    for (k, n) in [(5120, 1152), (1152, 5120)] {
        for &capacity in &capacities {
            let matching: Vec<_> = variants
                .iter()
                .filter(|v| {
                    v["capacity"].as_u64() == Some(capacity)
                        && v["input_dim"].as_u64() == Some(k)
                        && v["output_dim"].as_u64() == Some(n)
                })
                .collect();
            ensure!(
                matching.len() == 1,
                "missing or duplicate V4.1 shared projection {k}x{n} m{capacity}"
            );
            let bytes = matching[0]["scratch_bytes"]
                .as_u64()
                .context("missing FP8 scratch extent")?;
            scratch = scratch
                .checked_add(bytes.checked_add(255).context("FP8 alignment overflow")? & !255)
                .context("FP8 capacity arenas overflow")?;
        }
    }
    scratch
        .checked_add(16 + rows * ((3 * 1152 + 5120) * 2 + 5120 * 4))
        .context("shared FFN workspace overflow")
}

/// Install exact cache ownership, including source tables bounded by the
/// compiled 4096-row extent and private active/retained COW tail groups.
/// Snapshot arenas are separate fixed owners; replicas hold KV, never keys.
pub fn cache_demands(request: &mut PlacementRequest, config: &serde_json::Value,
    slots: u64, retained_turns: u64, replicated: bool) -> anyhow::Result<u64> {
    use anyhow::{ensure, Context};
    use cuteafd_core::memory_layout::{Basis, Category};
    let ranks = request.inventory.gpus.len();
    let geometry = crate::serving_capacity::deepseek_v41_cache_geometry(config, ranks)?;
    ensure!((1..=16).contains(&slots) && retained_turns <= 128 && (!replicated || ranks == 2),
        "invalid V4.1 cache admission shape");
    let tails = slots + 2 * retained_turns;
    let mut units = geometry.ranks.iter().map(|r| r.persistent_unit_bytes).collect::<Vec<_>>();
    if replicated {
        units[0] += 2 * 256 * 288;
        units[1] += 3 * 256 * 288;
    }
    // At one group each source table is still linear. Subtract its entries
    // from the canonical allocator total to isolate fixed windows and carry.
    let first = crate::serving_capacity::deepseek_v41_cache_bytes(config, ranks, slots, 1, replicated)?;
    let mut tables = Vec::new();
    let sources = if ranks == 1 { vec![(3, 1)] } else { vec![(3, 0), (0, 1)] };
    for (gpu, &(c2, c1)) in sources.iter().enumerate() {
        let replica = if replicated { if gpu == 0 { (0, 1) } else { (3, 0) } } else { (0, 0) };
        for (count, ratio) in [(c2 + replica.0, 1u64), (c1 + replica.1, 2)] {
            if count == 0 { continue; }
            tables.push(BoundedDemand {
                gpu: gpu as u8, category: Category::Kv, group: "source page tables".into(),
                bytes_per_unit: count * slots * ratio * 4,
                offset_units: tails, cap_units: 4096 / ratio, basis: Basis::Formula,
            });
        }
        let entries = tables.iter().filter(|d| usize::from(d.gpu) == gpu)
            .map(|d| d.bytes_per_unit).sum::<u64>();
        let fixed = first[gpu].checked_sub(units[gpu]).and_then(|n| n.checked_sub(entries))
            .context("V4.1 fixed cache extent underflow")?;
        request.fixed.push(Demand::new(gpu as u8, Category::Kv, "window state, table headers and compressor carry", fixed, Basis::Formula));
        request.fixed.push(Demand::new(gpu as u8, Category::Kv, "active and retained COW tails",
            units[gpu].checked_mul(tails).context("V4.1 tail extent overflow")?, Basis::Formula));
    }
    request.pool_overhead = units;
    request.pool_bounded = tables;
    request.pool.ceiling = request.pool.ceiling.min((131_072 - tails) * 512);
    Ok(tails)
}

/// Growth beyond the calibrated 48-row serving lanes. Keep verifier storage
/// independent of prefill rows and neural width; both target lanes retain it.
pub fn verify_workspace_growth(gpus: usize, rank: usize) -> u64 {
    assert!((1..=2).contains(&gpus) && rank < gpus);
    let extra_rows = cuteafd_core::V41_VERIFY_ROWS.saturating_sub(48) as u64;
    let sampler_row = 64 + 64 + 2 * 129280u64.div_ceil(32) * 4 + 2048 * 4 + 256 * (4 + 8) + 32;
    let normalization_row = 40960 + 16 + 10240 + 10240;
    let head_row = if gpus == 1 {
        normalization_row + 129280 * 4 + 8 + sampler_row
    } else {
        10240
            + 64640 * 4
            + if rank == 0 {
                8
            } else {
                24 + normalization_row + 129280 * 4 + sampler_row
            }
    };
    let sparse_row = 10 * 64 * 514 * 4 + 120;
    2 * extra_rows * (head_row + sparse_row)
}

/// Both target lanes reuse block inputs as their hop destinations. The full
/// mHC residual and incoming pre coefficients move, without an extra receive slot.
pub fn execution_hops(ranks: usize) -> HopSpec {
    HopSpec {
        storage: HopStorage::DirectInput,
        row_bytes: 4 * 5120 * 2 + 4 * 4,
        rows: 4096,
        lanes: 2,
        entry_gpu: 0,
        head_gpu: u8::from(ranks == 2),
        terminal: TerminalHome::AttentionOwner,
    }
}

/// All weights, lanes, graphs and prefix state are charged in the inventory's
/// baseline. Pool units include private tails supplied by the caller as fixed.
pub fn request(
    inventory: Inventory,
    pool: PoolPolicy,
    unit_bytes: Vec<u64>,
    experts: Vec<ExpertCost>,
    workspace: u64,
    onboard: Onboard,
) -> PlacementRequest {
    let gpus = inventory.gpus.len();
    assert!(experts.len() <= 40, "V4.1 has exactly 40 backbone layers");
    PlacementRequest {
        inventory,
        attention_placement: Some(AttentionPlacement::Heads),
        context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0,
        pool,
        pool_bounded: Vec::new(),
        pool_overhead: unit_bytes,
        fixed: vec![],
        movables: vec![],
        layers: (0..40)
            .map(|layer| LayerDemand {
                kind: AttentionClass::Csa,
                weights: ModeBytes::default(),
                kv_unit: ModeBytes::default().into(),
                fixed_bytes: ModeBytes::default(),
                extra: vec![],
                context_indexer: false,
                colocate: None,
                experts: experts.get(layer).cloned(),
                modes: vec![LayerMode::Whole {
                    gpu: u8::from(gpus == 2 && layer >= 20),
                    ffn: FfnMode::Owner,
                }],
            })
            .collect(),
        expert_workspace: workspace,
        tp2_workspace: [workspace; 2],
        onboard,
        minimum_expert_layers: 0,
        expert_gpus: gpus,
        policy: LayerPolicy::default(),
        hops: execution_hops(gpus),
        executor: super::DEEPSEEK_V41,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ced_ownership_and_tp2_loading_peak_are_preserved() {
        let inventory = Inventory {
            gpus: vec![
                GpuBudget {
                    capacity_bytes: 8 << 30,
                    headroom_bytes: 0,
                    baseline: Baseline::Measured {
                        free_bytes: 8 << 30
                    }
                };
                2
            ],
            spark_ranks: 4,
            peer_access: true,
        };
        let pool = PoolPolicy::resolve(&[8 << 30; 2], 512, Some(512), 512, false);
        let cost = ExpertCost {
            whole: Bytes2 {
                resident: 200 << 20,
                staging: 32 << 20,
            },
            half: [Bytes2 {
                resident: 100 << 20,
                staging: 16 << 20,
            }; 2],
            tp2: true,
            spark_ok: true,
        };
        let placement = solve(&request(
            inventory,
            pool,
            vec![1024; 2],
            vec![cost; 40],
            64 << 20,
            Onboard::Layers(3),
        ))
        .unwrap();
        assert_eq!(placement.onboard_layers, 3);
        for (layer, assignment) in placement.layers.iter().enumerate() {
            assert_eq!(
                assignment.mode,
                LayerMode::Whole {
                    gpu: u8::from(layer >= 20),
                    ffn: FfnMode::Owner
                }
            );
            assert_eq!(
                assignment.experts,
                if layer < 3 {
                    ExpertHome::RtxTp2
                } else {
                    ExpertHome::Spark
                }
            );
        }
        assert_eq!(placement.tp2.unwrap().peak_bytes, [380 << 20; 2]);
        let plan = LayerExecutionPlan::new(&placement, execution_hops(2)).unwrap();
        assert_eq!(plan.hops(), &[Hop { at: HopPoint::BeforeLayer(20), from: 0, to: 1,
            kind: HopKind::Boundary, storage: HopStorage::DirectInput, row_bytes: 40976 }]);
        assert_eq!(hop_buffer_bytes(plan.hops(), &plan.spec(), 2), Some(vec![0, 0]));
        assert_eq!(plan.homes()[40], ResidualHome::Owned(1));
        assert!(!placement.items.iter().flatten().any(|item| item.group == "residual hops"));
        assert!(placement.summary().contains("BeforeLayer(20) rtx0->rtx1 40976 B/row"));
        assert!(placement.summary().contains("no extra receive storage"));
        let json = serde_json::to_value(&placement).unwrap();
        assert_eq!(json["hops"][0]["storage"], "direct_input");
        assert_eq!(json["hops"][0]["row_bytes"], 40976);
        let mut wrong_storage = placement.clone();
        wrong_storage.hops[0].storage = HopStorage::ReceiveSlots;
        assert_eq!(LayerExecutionPlan::new(&wrong_storage, execution_hops(2)), Err(LayerPlanError::Hops));
    }

    #[test]
    fn ced_demands_cover_every_layer_with_partial_or_no_expert_inventory() {
        let cost = ExpertCost {
            whole: Bytes2 {
                resident: 1024,
                staging: 256,
            },
            half: [Bytes2 {
                resident: 512,
                staging: 128,
            }; 2],
            tp2: true,
            spark_ok: true,
        };
        for gpus in [1, 2] {
            for prefix in [0, 1, 3, 19, 20, 21, 40] {
                let request = request(
                    Inventory {
                        gpus: vec![
                            GpuBudget {
                                capacity_bytes: 8 << 30,
                                headroom_bytes: 0,
                                baseline: Baseline::Measured {
                                    free_bytes: 8 << 30
                                },
                            };
                            gpus
                        ],
                        spark_ranks: 4,
                        peer_access: gpus == 2,
                    },
                    PoolPolicy::resolve(&vec![8 << 30; gpus], 512, Some(512), 512, false),
                    vec![1024; gpus],
                    vec![cost; prefix],
                    64,
                    Onboard::Layers(0),
                );
                assert_eq!(request.layers.len(), 40);
                for (layer, demand) in request.layers.iter().enumerate() {
                    assert_eq!(demand.experts.is_some(), layer < prefix);
                    assert_eq!(
                        demand.modes,
                        vec![LayerMode::Whole {
                            gpu: u8::from(gpus == 2 && layer >= 20),
                            ffn: FfnMode::Owner,
                        }]
                    );
                }
                let placement = solve(&request).unwrap();
                assert_eq!(placement.layers.len(), 40);
                assert_eq!(placement.residual.len(), 41);
            }
        }
    }
    fn config() -> serde_json::Value {
        serde_json::from_str(include_str!("../../families/deepseek_v41/official-v41-config.json")).unwrap()
    }

    #[test]
    fn exact_cache_demands_match_allocator_across_table_caps_and_replicas() {
        use cuteafd_core::memory_layout::Category;
        for (ranks, replicated) in [(1, false), (2, false), (2, true)] {
            for (slots, turns) in [(1, 0), (16, 20), (3, 128)] {
                let tails = slots + 2 * turns;
                for physical in [tails + slots, 2047, 2048, 2049, 4095, 4096, 4097, 131072] {
                    let logical = physical - tails;
                    let mut req = request(Inventory { gpus: vec![GpuBudget { capacity_bytes: 1 << 40,
                        headroom_bytes: 0, baseline: Baseline::Measured { free_bytes: 1 << 40 } }; ranks],
                        spark_ranks: 4, peer_access: ranks == 2 },
                        PoolPolicy { requested: Some(logical * 512), target: logical * 512,
                            floor: logical * 512, ceiling: u64::MAX, unit_rows: 512 },
                        vec![0; ranks], vec![], 0, Onboard::Layers(0));
                    assert_eq!(cache_demands(&mut req, &config(), slots, turns, replicated).unwrap(), tails);
                    let placement = solve(&req).unwrap();
                    let exact = crate::serving_capacity::deepseek_v41_cache_bytes(&config(), ranks, slots, physical, replicated).unwrap();
                    for gpu in 0..ranks {
                        assert_eq!(placement.items[gpu].iter().filter(|i| i.category == Category::Kv)
                            .map(|i| i.bytes).sum::<u64>(), exact[gpu], "ranks={ranks} replicas={replicated} slots={slots} groups={physical} gpu={gpu}");
                    }
                }
            }
        }
    }

    #[test]
    fn native_manifest_demands_use_exact_retained_variants_and_resident_width() {
        let mut manifest = serde_json::json!({"role":"rtx_tp2", "geometry": {
            "hidden":5120,"intermediate":1152,"kernel_intermediate":1280,"experts":256,"topk":6},
            "variants": ([1,16,80,256,1024,4096].map(|c| serde_json::json!({
                "capacity_rows":c,"core_scratch_nbytes":if c == 16 { 99999 } else { c * 7 }})))});
        let demand = NativeWorkspace::from_manifest(&manifest,"rtx_tp2", 81,5120,1152,256,6).unwrap();
        assert_eq!(demand, NativeWorkspace { scratch_bytes:99999, kernel_intermediate:1280 });
        assert_eq!(demand.lane_bytes(81,5120,6,true,false).unwrap(),99999+81*5120*(24+26));
        assert_eq!(demand.lane_bytes(81,5120,6,true,true).unwrap(),99999+81*5120*(12+26));
        assert_eq!(demand.lane_bytes(81,5120,6,false,false).unwrap(),99999+81*5120*2);
        manifest["variants"][2]["requested_rows"] = 80.into();
        manifest["variants"][2]["capacity_rows"] = 96.into();
        assert_eq!(NativeWorkspace::from_manifest(&manifest,"rtx_tp2",81,5120,1152,256,6).unwrap(), demand);
        manifest["variants"][2]["requested_rows"] = 79.into();
        assert!(NativeWorkspace::from_manifest(&manifest,"rtx_tp2",81,5120,1152,256,6).is_err());
        manifest["variants"][2]["requested_rows"] = 80.into();
        let duplicate = manifest["variants"][2].clone();
        manifest["variants"].as_array_mut().unwrap().push(duplicate);
        assert!(NativeWorkspace::from_manifest(&manifest,"rtx_tp2",81,5120,1152,256,6).is_err());
        assert!(NativeWorkspace::from_manifest(&manifest,"rtx_backbone",256,5120,1152,256,6).is_err());
        assert_eq!(exl3_capacities(4096,true).unwrap(), [1,16,80,4096]);
        assert_eq!(exl3_capacities(16,true).unwrap(), [1,16,80]);
        assert_eq!(exl3_capacities(16,false).unwrap(), [1,16,80]);
        assert_eq!(exl3_capacities(1,true).unwrap(), [1]);
        assert!(exl3_capacities(0,true).is_err());
        assert!(exl3_capacities(4097,false).is_err());
        assert_eq!(expert_capacities(4096,true).unwrap(), [1,16,80,4096]);
        assert_eq!(expert_capacities(81,false).unwrap(), [1,16,80,256]);
        assert!(expert_capacities(81,true).is_err());
        assert!(expert_capacities(4097,false).is_err());
        assert!(owner_workspace(1,1,u64::MAX).is_err());
        assert!(resident_planes(128,u64::MAX-31,true).is_err());
        assert_eq!(resident_bytes(5120,576,256,true).unwrap(),
            resident_planes(5120,576,true).unwrap().iter().sum::<u64>()*256+256*16);
    }

    #[test]
    fn shared_fp8_capacity_scratch_is_a_sum_not_a_maximum() {
        let variants = [(5120,1152),(1152,5120)].into_iter().flat_map(|(k,n)|
            [1,16,80,256,1024,4096].map(|c| serde_json::json!({"capacity":c,"input_dim":k,
                "output_dim":n,"scratch_bytes":c+1}))).collect::<Vec<_>>();
        let manifest = serde_json::json!({"variants":variants});
        assert_eq!(shared_workspace(&manifest,257).unwrap(),
            2*(256+256+256+1280)+16+257*((3*1152+5120)*2+5120*4));
        let mut bad = manifest.clone();
        bad["variants"].as_array_mut().unwrap().remove(0);
        assert!(shared_workspace(&bad,257).is_err());
    }

    #[test]
    fn equivalent_baselines_produce_identical_deepseek_v41_placement() {
        // Bounded synthetic allocations and exact published-workspace payloads:
        // no HF_HOME, checkpoint shards, live GPU, or exponential layer sweep.
        let config = config();
        for ranks in [1,2] {
            for onboard in [Onboard::Auto,Onboard::Layers(3)] {
                let loaded = (20u64 << 30) + 12345;
                let capacity = 96u64 << 30;
                let expert = ExpertCost { whole: Bytes2 { resident:1<<30,staging:3<<20 },
                    half:[Bytes2 {resident:512<<20,staging:2<<20};2],tp2:ranks==2,spark_ok:true };
                let build = |baseline| {
                    let mut req = request(Inventory { gpus:vec![GpuBudget {capacity_bytes:capacity,
                        headroom_bytes:2<<30,baseline};ranks], spark_ranks:4,peer_access:ranks==2 },
                        PoolPolicy::resolve(&vec![capacity;ranks],1<<20,None,512,false),
                        vec![0;ranks],vec![expert;40], 1234567,onboard);
                    req.pool.ceiling=req.pool.target;
                    cache_demands(&mut req,&config,16,20,false).unwrap();
                    req
                };
                let planner = solve(&build(Baseline::Planned {context_bytes:12345,loaded_bytes:loaded-12345})).unwrap();
                let runtime = solve(&build(Baseline::Measured {free_bytes:capacity-loaded})).unwrap();
                assert_eq!(planner,runtime);
                assert_eq!(planner.pool_tokens,2<<20);
                assert_eq!(planner.layers.len(),40);
            }
        }
    }

}
