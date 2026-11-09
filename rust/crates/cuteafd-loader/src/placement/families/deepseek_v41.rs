//! V4.1 CED ownership stays fixed; pool and local experts use shared admission.
use super::super::*;
pub mod admission;
pub mod owners;
pub mod resident;
pub mod weight_layout;

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

    /// One lane, matching LocalExpertWave or OwnerPair (`OwnerPair::
    /// device_bytes_for` with `OwnerPolicy::native`): the TP2 owner keeps
    /// BF16 route planes on NVFP4 and FP32 token sums on MXFP4 (the shipped
    /// TP2 reducer sums routes; startup cross-checks the exact arena).
    pub fn lane_bytes(
        self,
        rows: u64,
        hidden: u64,
        topk: u64,
        dual: bool,
        nvfp4: bool,
    ) -> anyhow::Result<u64> {
        use anyhow::Context;
        let columns = if dual { if nvfp4 { topk.checked_mul(2).context("V4.1 expert route extent overflow")? } else { 4 } }
            else { 2 };
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
    let capacities = cuteafd_core::V41_TARGET_CAPACITIES;
    let compiled = capacities
        .into_iter()
        .find(|&n| n >= rows)
        .context("missing expert capacity")?;
    if local {
        ensure!(
            rows == compiled,
            "local expert rows must be a compiled capacity"
        );
        Ok(cuteafd_core::V41_DECODE_CAPACITIES.map(|n| n as u64)
            .into_iter()
            .filter(|&n| n < rows)
            .chain([rows])
            .collect())
    } else {
        Ok(capacities.into_iter().filter(|&n| n <= compiled).collect())
    }
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
    use anyhow::{ensure, Context};
    ensure!(
        matches!(world, 1 | 2) && rank < world && layer < 40,
        "invalid V4.1 RTX expert selection"
    );
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
    let resident = if !nvfp4 && catalog.mxfp4_scale_codes() {
        crate::formats::scale_codes::expert_bytes(shape.experts as u64, shape.hidden as u64,
            shape.intermediate as u64 / world as u64).context("coded expert residency overflow")?
    } else {
        resident_bytes(shape.hidden as u64, stored, shape.experts as u64, nvfp4)?
    };
    Ok(Bytes2 { resident, staging: staging as u64 })
}

/// Exact conditional expert owners: immutable checkpoint residency and two
/// independent serving lanes. Fixed target lanes stay in the baseline.
/// Expert costs per layer and the RTX expert workspace for `lanes` target
/// passes (two decode lanes plus any further prefill slot), each with its own
/// local or TP2 expert wave.
/// Per-layer expert costs, and the coordinator's expert workspaces: the
/// prefill slots' (one per slot on one RTX at the `rows` unit; one shared TP2
/// arena on two, planes at the `reserve`) and the two decode lanes' at
/// `decode_rows`.
pub fn expert_inventory(
    catalog: &crate::OfficialV41Catalog,
    programs: &std::path::Path,
    rows: u64,
    gpus: usize,
    sms: u32,
    slots: usize,
    reserve: u64,
    decode_rows: u64,
) -> anyhow::Result<(Vec<ExpertCost>, u64)> {
    use anyhow::{ensure, Context};
    ensure!(matches!(gpus, 1 | 2), "V4.1 supports one or two RTX owners");
    ensure!((1..=4).contains(&slots) && rows <= reserve, "V4.1 runs one to four prefill slots in the reserve");
    // Slot arenas run live rows on the covering AOT capacity; decode lanes
    // at their compiled width.
    let slot_capacity = if gpus == 1 {
        u64::from(cuteafd_core::coordinator_programs::v41_aot_rows(u32::try_from(rows)?)
            .context("invalid V4.1 prefill rows")?)
    } else { rows };
    let (experts, slot) = expert_workspace(catalog, programs, slot_capacity, reserve, gpus, sms)?;
    let (_, decode) = expert_workspace(catalog, programs, decode_rows, decode_rows, gpus, sms)?;
    let slots = if gpus == 1 { slots as u64 } else { 1 };
    Ok((experts, slot.checked_mul(slots).and_then(|n| n.checked_add(2 * decode))
        .context("V4.1 expert workspaces overflow")?))
}

/// Per-layer costs, and one expert workspace at `rows` (TP2: planes at `reserve`).
fn expert_workspace(
    catalog: &crate::OfficialV41Catalog,
    programs: &std::path::Path,
    rows: u64,
    reserve: u64,
    gpus: usize,
    sms: u32,
) -> anyhow::Result<(Vec<ExpertCost>, u64)> {
    use anyhow::{ensure, Context};
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
    // A compiled capacity on one RTX (the caller covers live rows); TP2 rows.
    let capacity = rows;
    let (stored, routed) = {
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
        // Shared TP2 (its sums a plane at the reserve) and the peer inputs
        // and final output planes at the reserve (WP7 `tp2_ffn::Storage`).
        routed
            .checked_add(shared_workspace(&fp8, rows, sms)? - rows * 10240 + reserve * 10240)
            .and_then(|n| n.checked_add(reserve * (10240 + 5280 + 24 + 24 + 10240)))
            .context("V4.1 FFN workspace overflow")?
    };
    let experts = (0..40)
        .map(|layer| {
            let mut cost = ExpertCost {
                whole: Bytes2::default(),
                half: [Bytes2::default(); 2],
                tp2: !local,
                whole_supported: true, spark_ok: true,
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
    Ok((experts, lane))
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
pub fn shared_workspace(manifest: &serde_json::Value, rows: u64, sms: u32) -> anyhow::Result<u64> {
    use anyhow::Context;
    let programs = ProgramSet::v41_fp8(manifest, &[(5120, 1152), (1152, 5120)], rows, sms)?;
    programs.scratch.into_iter().try_fold(0u64, |sum, (_, bytes)| {
        sum.checked_add(bytes).context("FP8 capacity arenas overflow")
    })?.checked_add(16 + rows * ((3 * 1152 + 5120) * 2 + 5120 * 4))
        .context("shared FFN workspace overflow")
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
/// What the V4.1 dSpark drafter retains, measured from its own plan before
/// any payload is read. Runtime owners come from `owners::NativePrograms::dspark`
/// (request lanes, main context, windows) on the default native BF16 path;
/// EXL3 arenas and TP2 expert halves are sized from their packages.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DraftInventory {
    /// Auxiliary transformer weights plus the routed experts this rank owns
    /// beside the head (whole TP1 experts, or the GPU1 TP2 half).
    pub resident: u64,
    /// Request lanes, main contexts and windows on each rank (index = GPU).
    pub runtime: Vec<Demand>,
    /// Prefix draft marks (`DRAFT_MARK_BYTES` per retained mark slot).
    pub marks: u64,
    /// `--tp2-dspark-experts`: GPU0's checkpoint expert halves; runtime is separate.
    pub peer_half: u64,
    /// Largest transient load staging on the home and on the peer half; the
    /// caller folds it into the full dSpark load-phase peak.
    pub staging: [u64; 2],
    /// Routed-expert bytes resident while that staging is live, [home, GPU0
    /// TP2 half]: experts load first, serially, before anything else.
    pub loading_experts: [u64; 2],
    /// The loaders' exact device load bound per GPU [home, GPU0 TP2 half]:
    /// max(experts + their staging, all retained weights). Runtime owners
    /// (lanes, windows, main contexts) are not part of loading.
    pub loader_peak: [u64; 2],
    /// The independently retired dense arena on the home GPU
    /// (`weight_layout::draft_layout`). Its extent is part of `resident`;
    /// admission charges it as one arena plus its tail and the remainder of
    /// `resident` as routed experts.
    pub dense: Option<crate::serving_capacity::WeightArenaLayout>,
}

impl DraftInventory {
    /// The full dSpark load phase above a pre-draft sample: experts plus their
    /// transient staging on each GPU that loads them (`request.startup`).
    pub fn load_phase(&self, ranks: usize) -> Result<Vec<Demand>, PlacementError> {
        if !(1..=2).contains(&ranks) { return Err(PlacementError::Inventory("V4.1 dSpark draft ranks")); }
        let home = (ranks - 1) as u8;
        let peak = |gpu: usize| self.loading_experts[gpu].checked_add(self.staging[gpu])
            .map(|bytes| bytes.max(self.loader_peak[gpu]))
            .ok_or(PlacementError::Overflow("V4.1 dSpark load phase"));
        let mut phases = vec![Demand::new(home, Category::Staging, "dSpark load phase", peak(0)?, Basis::Exact)];
        if self.peer_half > 0 {
            phases.push(Demand::new(0, Category::Staging, "dSpark TP2 load phase", peak(1)?, Basis::Exact));
        }
        phases.retain(|d| d.bytes > 0);
        Ok(phases)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::planner::{Lifetime, Location, Operation, Phase, PrefillShape, Refusal};
    use admission::{production_request, AdmittedV41, ProductionInput};

    fn config() -> serde_json::Value {
        serde_json::from_str(include_str!("../../families/deepseek_v41/official-v41-config.json")).unwrap()
    }

    /// Synthetic routed-layer costs; `arena` owns the exact expected peak.
    fn cost(ranks: usize) -> ExpertCost {
        ExpertCost {
            whole: Bytes2 { resident: 200 << 20, staging: 32 << 20 },
            half: [Bytes2 { resident: 100 << 20, staging: 16 << 20 }; 2],
            tp2: ranks == 2,
            whole_supported: true,
            spark_ok: true,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn admit(config: &serde_json::Value, ranks: usize, owners: Vec<admission::Owner>,
        draft: Option<DraftInventory>, experts: Vec<ExpertCost>, onboard: Onboard,
        pool: Option<u64>, context: u64, concurrency: u64, turns: u64, replicated: bool,
        free: u64,
    ) -> Result<AdmittedV41, Refusal> {
        let total = 96 << 30;
        let budgets = admission::live_budgets(&vec![(total, total, free); ranks], &vec![64 << 30; 4])?;
        let input = ProductionInput {
            blas: Vec::new(),
            expert_usage: false,
            config, gpus: ranks, context, maximum_pool_tokens: pool.unwrap_or(2 << 20),
            prefill: PrefillShape { rows: 128, waves: 1 }, concurrency, retained_turns: turns,
            replicated, spark_ranks: 4, onboard_minimum: 0, onboard_counts: None,
            experts, workspace: 0, owners, weight_arenas: Vec::new(), startup: Vec::new(), allowances: Vec::new(),
            draft, encoders: Vec::new(), media_cache_bytes: 0,
            resident_weight_limit: u64::MAX, spark_reply_formats: Default::default(),
            operation: Operation::Serving, budgets,
        };
        let request = production_request(context, pool, onboard, PrefillShape { rows: 128, waves: 1 },
            Operation::Serving, u64::MAX, Default::default());
        input.prepare()?.admit(request)
    }

    #[test]
    fn expert_usage_is_admitted_once_per_canonical_ced_router() -> anyhow::Result<()> {
        use crate::placement::planner::{ComponentKind, Recipe};
        let config = config();
        for ranks in [1, 2] {
            for enabled in [false, true] {
                let prepared = ProductionInput {
                    blas: Vec::new(),
                    expert_usage: enabled, config: &config, gpus: ranks, context: 512,
                    maximum_pool_tokens: 4096, prefill: PrefillShape { rows: 128, waves: 1 },
                    concurrency: 1, retained_turns: 0, replicated: false, spark_ranks: 4,
                    onboard_minimum: 0, onboard_counts: None, experts: vec![cost(ranks); 40],
                    workspace: 0, owners: Vec::new(), weight_arenas: Vec::new(), startup: Vec::new(), allowances: Vec::new(),
                    draft: None, encoders: Vec::new(), media_cache_bytes: 0,
                    resident_weight_limit: u64::MAX, spark_reply_formats: Default::default(),
                    operation: Operation::Serving,
                    budgets: admission::production_budgets(&vec![(96 << 30, 96 << 30); ranks],
                        0, Some((128 << 30, 128 << 30)), 0, &vec![64 << 30; 4])?,
                }.prepare()?;
                let request = production_request(512, Some(4096), Onboard::Layers(0),
                    PrefillShape { rows: 128, waves: 1 }, Operation::Serving, u64::MAX, Default::default());
                let contract = prepared.recipe.prepare(&request)?;
                let mut found = std::collections::BTreeSet::new();
                for component in &contract.components {
                    let ComponentKind::Allocation { template, .. } = &component.kind else { continue; };
                    if !template.id.starts_with("expert-usage.") { continue; }
                    assert!(found.insert(template.id.clone()));
                    assert_eq!(component.bytes.at(1)?, component.bytes.at(contract.maximum_pages)?);
                    if template.id == "expert-usage.history" {
                        assert_eq!(template.location, Location::HOST);
                        assert_eq!(template.bytes, prepared.recipe.expert_usage.as_ref().unwrap()
                            .host_bytes(ranks).unwrap() as u64);
                    } else {
                        assert_eq!(template.bytes, 40 * 384 * 2 * 8);
                        let gpu = template.id.split('.').nth(1).unwrap().trim_start_matches("gpu").parse::<u8>()?;
                        assert!(usize::from(gpu) < ranks);
                        assert_eq!(template.location, if template.id.ends_with("pinned") {
                            Location::PINNED
                        } else { Location::coordinator(gpu) });
                    }
                }
                assert_eq!(found.len(), if enabled { ranks * 3 + 1 } else { 0 });
                let admitted = prepared.admit(request)?;
                assert_eq!(admitted.expert_usage.is_some(), enabled);
            }
        }
        Ok(())
    }

    #[test]
    fn expert_usage_geometry_comes_from_checkpoint_and_refuses_invalid_headers() {
        let mut config = config();
        let geometry = admission::expert_usage_geometry(&config).unwrap();
        assert_eq!((geometry.layers, geometry.experts, geometry.topk), (40, 384, 6));
        assert_eq!(geometry.routed_layers, (0..40).collect::<Vec<_>>());
        assert_eq!(admission::expert_usage_geometry(&config["text_config"]).unwrap(), geometry);
        for (key, value) in [("n_routed_experts", 0), ("num_experts_per_tok", 385), ("num_hidden_layers", 39)] {
            let old = config["text_config"][key].take();
            config["text_config"][key] = serde_json::json!(value);
            assert!(admission::expert_usage_geometry(&config).is_err(), "{key}");
            config["text_config"][key] = old;
        }
    }

    #[test]
    fn all_onboarded_experts_allow_a_budgeted_spark_tower() {
        use crate::plan::encoder::{EncoderKind, EncoderPlacement};
        let config = config();
        let input = ProductionInput {
            blas: Vec::new(),
            expert_usage: false, config: &config, gpus: 2, context: 32768, maximum_pool_tokens: 32768,
            prefill: PrefillShape { rows: 128, waves: 1 }, concurrency: 1, retained_turns: 0,
            replicated: false, spark_ranks: 1, onboard_minimum: 0, onboard_counts: None,
            experts: vec![cost(2); 40], workspace: 0, owners: vec![], weight_arenas: Vec::new(), startup: vec![], allowances: vec![],
            draft: None, encoders: vec![EncoderPlacement { kind: EncoderKind::Spark { rank: 0 },
                weights: 123456, scratch: 654321, replicas: vec![], reason: "tower only".into(), shortfall: 0 }],
            media_cache_bytes: 0, resident_weight_limit: u64::MAX, spark_reply_formats: Default::default(),
            operation: Operation::Serving,
            budgets: admission::live_budgets(&[(96 << 30, 96 << 30, 96 << 30); 2], &[64 << 30]).unwrap(),
        };
        let request = production_request(32768, Some(32768), Onboard::Layers(40),
            PrefillShape { rows: 128, waves: 1 }, Operation::Serving, u64::MAX, Default::default());
        let mut prepared = input.prepare().unwrap();
        prepared.recipe.vision_arena = Some(crate::serving_capacity::WeightArenaLayout::new(
            "v41/vision", [("tower".into(), 123456)]).unwrap());
        let admitted = prepared.admit(request).unwrap();
        assert!(admitted.spark.spark_layers.is_empty());
        let owners = admitted.solved.facts.allocations.iter().filter(|owner| owner.id.starts_with("v41.encoder.")).collect::<Vec<_>>();
        assert_eq!(owners.iter().map(|owner| owner.bytes).sum::<u64>(), 123456 + 654321);
        assert!(owners.iter().all(|owner| owner.location.device == crate::placement::planner::Device::Spark(0)));
    }

    #[test]
    fn vision_arenas_preserve_choices_payload_and_physical_device_binding() -> anyhow::Result<()> {
        use crate::plan::encoder::{EncoderKind, EncoderPlacement};
        use crate::placement::planner::{Activation, ComponentKind, CoordinatorDevices, Device,
            Domain, Recipe, allocation_granularity_forecast};
        let config = config();
        // Deliberately include alignment padding so Managed B cannot masquerade as P.
        let arena = crate::serving_capacity::WeightArenaLayout::new("v41/vision",
            [("first".into(), (1 << 20) + 1), ("second".into(), 2)])?;
        let payload = arena.payload_bytes() as u64;
        let extent = arena.bytes() as u64;
        assert!(extent > payload);
        for physical in [&[1][..], &[1, 0][..]] {
            let devices = CoordinatorDevices::new(physical)?;
            let ranks = physical.len();
            let mut kinds = vec![EncoderKind::Rtx { gpu: 0 }, EncoderKind::Spark { rank: 0 }, EncoderKind::Off];
            if ranks == 2 { kinds.insert(1, EncoderKind::Rtx { gpu: 1 }); }
            for first in 0..kinds.len() {
                let mut ordered = kinds.clone();
                ordered.rotate_left(first);
                let encoders = ordered.iter().cloned().map(|kind| EncoderPlacement {
                    replicas: if matches!(kind, EncoderKind::Spark { .. }) { vec![1] } else { vec![] },
                    kind, weights: payload, scratch: 333, reason: "arena proof".into(), shortfall: 0,
                }).collect();
                let mut prepared = ProductionInput {
                    blas: Vec::new(),
                    expert_usage: false, config: &config, gpus: ranks, context: 512, maximum_pool_tokens: 512,
                    prefill: PrefillShape { rows: 128, waves: 1 }, concurrency: 1, retained_turns: 0,
                    replicated: false, spark_ranks: 4, onboard_minimum: 0, onboard_counts: None,
                    experts: vec![cost(ranks); 40], workspace: 0, owners: vec![], weight_arenas: vec![],
                    startup: vec![], allowances: vec![], draft: None, encoders, media_cache_bytes: 0,
                    resident_weight_limit: u64::MAX, spark_reply_formats: Default::default(),
                    operation: Operation::Serving,
                    budgets: admission::live_budgets(&vec![(96 << 30, 96 << 30, 96 << 30); ranks],
                        &[64 << 30; 4])?,
                }.prepare()?;
                prepared.recipe.vision_arena = Some(arena.clone());
                let request = production_request(512, Some(512), Onboard::Layers(0),
                    PrefillShape { rows: 128, waves: 1 }, Operation::Serving, u64::MAX, Default::default());
                let contract = prepared.recipe.prepare(&request)?;
                for component in &contract.components {
                    let id = match &component.kind {
                        ComponentKind::Allocation { template, .. } => &template.id,
                        ComponentKind::ResidentWeight(claim) => &claim.id,
                        ComponentKind::AllocationGranularity(tail) => &tail.id,
                        _ => continue,
                    };
                    if let Some(suffix) = id.strip_prefix("v41.encoder.") {
                        let option: usize = suffix.split('.').next().unwrap().parse()?;
                        let clause = &component.activation.clauses[0];
                        assert_eq!(clause.len(), 1);
                        assert_eq!(component.activation, Activation::choice(clause[0].variable, option));
                    }
                }
                let admitted = prepared.admit(request)?;
                assert_eq!(admitted.solved.decisions.encoder, Some(0));
                let handles = admitted.solved.facts.allocations.iter().filter(|owner|
                    owner.group.starts_with("cuBLAS handles:")
                        && owner.lifetime.contains(Phase::Ready, admitted.solved.facts.phase_order))
                    .collect::<Vec<_>>();
                // Spark media is covered by its existing measured overhead (#1568).
                assert_eq!(handles.len(), usize::from(matches!(ordered[0], EncoderKind::Rtx { .. })));
                assert!(handles.iter().all(|owner| matches!(owner.location.device, Device::Coordinator(_))));
                let mapped = devices.remap_plan_weight_arenas(&admitted.solved)?;
                let roots = mapped.facts.allocations.iter().filter(|owner|
                    owner.id.starts_with("v41.encoder.") && owner.id.ends_with(".weights")).collect::<Vec<_>>();
                let expected_copies = match ordered[0] {
                    EncoderKind::Off => 0, EncoderKind::Spark { .. } => 2, _ => 1,
                };
                assert_eq!(roots.len(), expected_copies);
                for (copy, root) in roots.iter().enumerate() {
                    assert_eq!(root.id, format!("v41.encoder.0.{copy}.weights"));
                    assert_eq!((root.bytes, root.group.as_str()), (extent, "weight arena"));
                    assert_eq!(root.lifetime, Lifetime::RESIDENT);
                    assert!(root.view.is_none());
                    let claim = mapped.facts.resident_weights.iter().find(|claim| claim.id == root.id).unwrap();
                    assert_eq!(claim.weights.0, payload);
                    let tails = mapped.facts.allocation_granularity.iter().filter(|tail|
                        tail.backings.iter().any(|backing| backing.backing == root.id)).collect::<Vec<_>>();
                    match root.location.device {
                        Device::Coordinator(rank) => {
                            assert_eq!(devices.location(physical[usize::from(rank)])?, root.location);
                            assert_eq!(tails.len(), 1);
                            assert_eq!(tails[0].id, format!("{}.tail", root.id));
                            assert_eq!(tails[0].requests(), [extent]);
                            assert_eq!(tails[0].bytes()?, allocation_granularity_forecast(&[extent])?);
                        }
                        Device::Spark(rank) => {
                            assert_eq!(usize::from(rank), copy);
                            assert_eq!(root.location.domain, Domain::Managed);
                            assert!(tails.is_empty());
                        }
                        _ => panic!("unexpected tower location"),
                    }
                    let scratch = mapped.facts.allocations.iter().find(|owner|
                        owner.id == format!("v41.encoder.0.{copy}.scratch")).unwrap();
                    assert_eq!((scratch.location, scratch.bytes), (root.location, 333));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn graph_reserve_is_one_post_ready_owner_per_card_of_any_size() {
        // Card size no longer changes the graph budget: the measured arena is
        // one family number, and the small-card fixed-shape set is a capture
        // policy, not a second envelope.
        for ranks in [1, 2] {
            for gpu in 0..ranks {
                let owner = admission::graph_allowance(gpu).unwrap();
                assert_eq!(owner.location, Location::coordinator(gpu as u8));
                assert_eq!(owner.lifetime, Lifetime { first: Phase::Ready, last: Phase::Growth });
                assert_eq!(owner.source, format!("graph reserve: v41.graphs.{gpu}"));
                assert_eq!(owner.bytes, 1_048_576_000);
                assert_eq!(owner.basis, Basis::Calibrated);
            }
        }
    }

    #[test]
    fn ced_ownership_and_tp2_loading_peak_are_preserved() -> anyhow::Result<()> {
        let config = config();
        let costs = vec![cost(2); 40];
        let admitted = admit(&config, 2, Vec::new(), None, costs.clone(), Onboard::Layers(3),
            Some(512), 512, 1, 0, false, 96 << 30)?;
        let placement = &admitted.placement;
        assert_eq!(placement.attention_placement, AttentionPlacement::Layers);
        assert_eq!(placement.onboard_layers, 3);
        for (layer, assignment) in placement.layers.iter().enumerate() {
            assert_eq!(assignment.mode, LayerMode::Whole { gpu: u8::from(layer >= 20), ffn: FfnMode::Owner });
            assert_eq!(assignment.experts, if layer < 3 { ExpertHome::RtxTp2 } else { ExpertHome::Spark });
        }
        let (peak, _) = admission::arena(3, 2, &costs, 0)?;
        assert_eq!(placement.tp2.expect("TP2 range").peak_bytes, peak);
        let plan = LayerExecutionPlan::new(placement, execution_hops(2))?;
        assert_eq!(plan.hops(), &[Hop { at: HopPoint::BeforeLayer(20), from: 0, to: 1,
            kind: HopKind::Boundary, storage: HopStorage::DirectInput, row_bytes: 40976 }]);
        assert_eq!(hop_buffer_bytes(plan.hops(), &plan.spec(), 2), Some(vec![0, 0]));
        assert_eq!(plan.homes()[40], ResidualHome::Owned(1));
        assert!(!placement.items.iter().flatten().any(|item| item.group == "residual hops"));
        let mut wrong_storage = placement.clone();
        wrong_storage.hops[0].storage = HopStorage::ReceiveSlots;
        assert_eq!(LayerExecutionPlan::new(&wrong_storage, execution_hops(2)), Err(LayerPlanError::Hops));
        Ok(())
    }

    #[test]
    fn exact_cache_owners_match_allocator_across_table_caps_and_replicas() -> anyhow::Result<()> {
        use cuteafd_core::memory_layout::Category;
        let config = config();
        for (ranks, replicated) in [(1, false), (2, false), (2, true)] {
            for (slots, turns) in [(1, 0), (16, 20), (17, 20), (32, 20), (64, 80), (3, 128), (64, 128)] {
                let tails = slots + 2 * turns;
                for physical in [tails + slots, 2047, 2048, 2049, 4095, 4096, 4097, 131072] {
                    let logical = physical - tails;
                    let admitted = admit(&config, ranks, Vec::new(), None, vec![cost(ranks); 40],
                        Onboard::Layers(0), Some(logical * 512), logical * 512, slots, turns, replicated, 96 << 30)?;
                    let exact = crate::serving_capacity::deepseek_v41_cache_bytes(
                        &config, ranks, slots, physical, replicated)?;
                    for gpu in 0..ranks {
                        assert_eq!(admitted.placement.items[gpu].iter()
                            .filter(|i| i.category == Category::Kv).map(|i| i.bytes).sum::<u64>(),
                            exact[gpu], "ranks={ranks} replicas={replicated} slots={slots} groups={physical} gpu={gpu}");
                    }
                }
            }
        }
        Ok(())
    }

    /// V4.1 never clamps below its compiled context: an automatic pool that
    /// cannot hold the context is a typed shortfall, not a smaller plan.
    #[test]
    fn an_automatic_pool_below_the_context_is_refused_not_clamped() -> anyhow::Result<()> {
        let config = config();
        // Include 100MiB release padding so the remaining 64MiB still tests
        // the context shortfall, rather than exhausting the reserve itself.
        let padding = cuteafd_core::memory_policy::MemoryPolicy::current().release_padding();
        let cards = vec![(96u64 << 30, 96u64 << 30, (64 << 20) + padding); 2];
        let budgets = admission::live_budgets(&cards, &vec![1 << 30; 4])?;
        let input = ProductionInput {
            blas: Vec::new(),
            expert_usage: false,
            config: &config, gpus: 2, context: 262_144, maximum_pool_tokens: 2 << 20,
            prefill: PrefillShape { rows: 128, waves: 1 }, concurrency: 1, retained_turns: 0,
            replicated: false, spark_ranks: 4, onboard_minimum: 0, onboard_counts: None,
            experts: vec![cost(2); 40], workspace: 0, owners: Vec::new(), weight_arenas: Vec::new(), startup: Vec::new(),
            allowances: Vec::new(), draft: None, encoders: Vec::new(), media_cache_bytes: 0,
            resident_weight_limit: u64::MAX, spark_reply_formats: Default::default(),
            operation: Operation::Serving, budgets,
        };
        let request = production_request(262_144, None, Onboard::Layers(0),
            PrefillShape { rows: 128, waves: 1 }, Operation::Serving, u64::MAX, Default::default());
        assert!(matches!(input.prepare()?.admit(request),
            Err(Refusal::FullContextShortfall { context: 262_144, .. })));
        Ok(())
    }

    #[test]
    fn exact_cache_rejects_slots_outside_compiled_request_bank() {
        for slots in [0, cuteafd_core::V41_MAX_REQUESTS as u64 + 1] {
            assert!(crate::serving_capacity::deepseek_v41_cache_bytes(&config(), 1, slots, 2048, false).is_err());
        }
    }

    #[test]
    fn native_manifest_demands_use_exact_retained_variants_and_resident_width() {
        let mut manifest = serde_json::json!({"role":"rtx_tp2", "geometry": {
            "hidden":5120,"intermediate":1152,"kernel_intermediate":1280,"experts":256,"topk":6},
            "variants": ([1,16,64,256,1024,4096].map(|c| serde_json::json!({
                "capacity_rows":c,"core_scratch_nbytes":if c == 16 { 99999 } else { c * 7 }})))});
        let demand = NativeWorkspace::from_manifest(&manifest,"rtx_tp2", 81,5120,1152,256,6).unwrap();
        assert_eq!(demand, NativeWorkspace { scratch_bytes:99999, kernel_intermediate:1280 });
        assert_eq!(demand.lane_bytes(81,5120,6,true,false).unwrap(),99999+81*5120*(4+26));
        assert_eq!(demand.lane_bytes(81,5120,6,true,true).unwrap(),99999+81*5120*(12+26));
        assert_eq!(demand.lane_bytes(81,5120,6,false,false).unwrap(),99999+81*5120*2);
        manifest["variants"][2]["requested_rows"] = 64.into();
        manifest["variants"][2]["capacity_rows"] = 96.into();
        assert_eq!(NativeWorkspace::from_manifest(&manifest,"rtx_tp2",81,5120,1152,256,6).unwrap(), demand);
        manifest["variants"][2]["requested_rows"] = 79.into();
        assert!(NativeWorkspace::from_manifest(&manifest,"rtx_tp2",81,5120,1152,256,6).is_err());
        manifest["variants"][2]["requested_rows"] = 64.into();
        let duplicate = manifest["variants"][2].clone();
        manifest["variants"].as_array_mut().unwrap().push(duplicate);
        assert!(NativeWorkspace::from_manifest(&manifest,"rtx_tp2",81,5120,1152,256,6).is_err());
        assert!(NativeWorkspace::from_manifest(&manifest,"rtx_backbone",256,5120,1152,256,6).is_err());
        assert_eq!(expert_capacities(4096,true).unwrap(), [1,16,64,4096]);
        assert_eq!(expert_capacities(81,false).unwrap(), [1,16,64,256]);
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
            [1,16,64,256,1024,4096].map(|c| serde_json::json!({"capacity":c,"input_dim":k,
                "output_dim":n,"label":format!("shared_{k}_{n}_m{c}"),"scratch_bytes":c+1}))).collect::<Vec<_>>();
        let manifest = serde_json::json!({"variants":variants});
        assert_eq!(shared_workspace(&manifest,257,188).unwrap(),
            2*(256+256+256+1280)+16+257*((3*1152+5120)*2+5120*4));
        let mut bad = manifest.clone();
        bad["variants"].as_array_mut().unwrap().remove(0);
        assert!(shared_workspace(&bad,257,188).is_err());
    }

    #[test]
    fn the_drafter_charges_both_gpus_before_the_pool() -> anyhow::Result<()> {
        let config = config();
        let inventory = DraftInventory { resident: 6 << 30,
            runtime: vec![Demand::new(1, Category::Drafter, "dSpark request lanes", 1 << 30, Basis::Formula)],
            marks: 0, peer_half: 2 << 30, staging: [1 << 30, 1 << 29],
            loading_experts: [4 << 30, 2 << 30], loader_peak: [0; 2], dense: None };
        let free = 8 << 30;
        let base = admit(&config, 2, Vec::new(), None, vec![cost(2); 40], Onboard::Layers(4),
            None, 512, 1, 0, false, free)?;
        let with = admit(&config, 2, Vec::new(), Some(inventory), vec![cost(2); 40],
            Onboard::Layers(4), None, 512, 1, 0, false, free)?;
        assert_eq!(crate::placement::draft::home(&with.placement), Some(1));
        assert!(with.placement.pool_tokens < base.placement.pool_tokens,
            "the drafter's bytes come out of the pool, not after it");
        Ok(())
    }

    #[test]
    fn dspark_load_phase_is_experts_plus_staging_or_the_loader_check() {
        let mut d = DraftInventory { staging: [1 << 30, 1 << 29], loading_experts: [6 << 30, 2 << 30],
            peer_half: 2 << 30, ..Default::default() };
        let phase = |d: &DraftInventory| d.load_phase(2).unwrap().iter().map(|p| (p.gpu, p.bytes)).collect::<Vec<_>>();
        assert_eq!(phase(&d), [(1, 7 << 30), (0, (2 << 30) + (1 << 29))]);
        d.loader_peak = [9 << 30, 0];
        assert_eq!(phase(&d), [(1, 9 << 30), (0, (2 << 30) + (1 << 29))]);
        d.peer_half = 0;
        assert_eq!(d.load_phase(1).unwrap()[0].gpu, 0);
    }
}
