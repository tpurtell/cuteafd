//! Native V4.1 allocation owners, shared with serving admission. CED layer
//! ownership is fixed; the solver chooses only KV and the local expert prefix.
pub use crate::placement::families::deepseek_v41::resident::*;
use super::*;
use anyhow::{ensure, Context, Result};
use crate::placement::{self, families::deepseek_v41 as v41, ArchContext, Baseline,
    Demand, GpuBudget, Inventory, Onboard, PoolPolicy, SparkWorkingSet};
use v41::DraftInventory;

// Conservative combined code/ready-graph envelope; pure LoadedCode is unqualified.
// Neither generic PROGRAMS.json module counts nor other families' graph costs
// describe the native executor's loaded code.
const NATIVE_CODE_ESTIMATE: u64 = 2 * GIB;
const SPARK: SparkCharges = SparkCharges {
    host_bytes: Some(9 * GIB), wave_bytes: 21 * GIB / 10, basis: Basis::Estimated,
};

pub(crate) fn layout(report: &mut PlanReport, model: &dyn crate::plan::FamilyModel,
    input: &crate::plan::PlanInput, options: &LayoutOptions) -> Result<MemoryLayout> {
    let checkpoint = input.checkpoint();
    let shape = Shape::resolve(report, model, options, ShapePolicy {
        context: ContextPolicy::Checkpoint, prefill_rows: Some(2048), all_native_stages: false,
    });
    let ranks = options.rtx_bytes.len();
    ensure!((1..=2).contains(&ranks), "V4.1 requires one or two coordinator GPUs");
    ensure!((1..=16).contains(&shape.concurrency), "V4.1 concurrency must be 1..16");
    let rows = u64::from(cuteafd_core::coordinator_programs::v41_live_rows(
        u32::try_from(shape.prefill_rows)?).context("invalid V4.1 prefill rows")?);
    let marks = options.prefix_slots.unwrap_or(42);
    ensure!(marks == 0 || (marks >= 2 && marks % 2 == 0 && marks <= 258),
        "V4.1 prefix slots must be zero or 2 * retained turns + 2 (at most 258)");
    let turns = marks.saturating_sub(2) / 2;
    let dspark = shape.native_layers > 0;
    ensure!(!dspark || shape.native_layers == 3, "V4.1 dSpark retains all three stages");
    let manifest = options.workspace_manifest.as_deref()
        .unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let directory = manifest.parent().context("V4.1 manifest has no parent")?;
    let fp8: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("V41_FP8_AOT.json"))
        .context("missing V4.1 native FP8 allocation manifest")?)?;
    let sms = options.physical_sms.unwrap_or_else(|| ArchContext::for_device("sm_120", options.rtx_bytes[0]).sms);
    let programs = v41::owners::NativePrograms::new(&fp8, sms)?;
    let mode = Fp8Head::configured();
    let fp8_head = mode.target();
    let window_batch = ranks == 1 && std::env::var("CUTEAFD_WINDOW_BATCH").map_or(true, |v| v != "0");
    let target = programs.target(rows, ranks, fp8_head, window_batch)?;
    let weights = resident_weights(checkpoint, ranks, false)
        .context("unsupported V4.1 native backbone residency: check non-expert tensor formats and geometry")?;
    let mut devices = options.rtx_bytes.iter().enumerate().map(|(rank, &bytes)| DeviceLayout {
        kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: bytes,
        items: weights[rank].clone(), kv_tokens: 0,
    }).collect::<Vec<_>>();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {} tokens (checkpoint {})", shape.context_tokens, shape.checkpoint_context),
        "Native LoadedCode: Estimated 2 GiB per GPU beyond the measured architecture context; combined startup census residuals fit this bound. Pure LoadedCode and ready parity are not qualified.".into(),
        "Spark host, wave residual and native runtime: Estimated pending qualified worker inventory; explicit rank I/O remains formula-derived.".into(),
        "Lazy graph/runtime growth is reserved by shared V4.1 solver demands, not included in the ready residency claim.".into()];
    mapped_table_note(report, &mut notes);
    if options.host_embedding || (options.rtx_bytes[0] <= 32 * GIB && !options.force_gpu_embedding) {
        place_host_embedding(report, checkpoint, &mut devices, &mut waste, &mut notes);
    }
    let fixed = devices.iter().flat_map(|d| d.items.iter().map(|i|
        Demand::new(d.index as u8, i.category, i.group.clone(), i.bytes, i.basis))).collect::<Vec<_>>();
    let mut startup = Vec::new();
    if ranks == 1 {
        if let Some(head) = checkpoint.tensors.iter().find(|t| t.meta.name == "head.weight") {
            let source = head.meta.byte_length;
            let packing = mode.load_bytes(source);
            startup.push(v41::owners::vocabulary_load_phase(&fixed, &target, packing)?);
        }
    }
    let spark_ranks = report.placement.spark_ranks();
    let transport = if spark_ranks > 0 { 2 * v41::owners::transport_bytes(rows, spark_ranks)? } else { 0 };
    let mut fixed = fixed;
    if options.drafter_bytes > 0 {
        fixed.push(Demand::new(0, Category::Drafter, "external drafter weights",
            options.drafter_bytes, Basis::Exact));
    }
    let mut owners = v41::owners::TargetOwners::new(fixed, target, ranks, transport, marks)?;
    let mut fixed = owners.fixed.clone();
    let catalog = input.expert_catalog()?;
    let (experts, workspace) = v41::expert_inventory(catalog, manifest, rows, ranks, sms)?;
    let draft = if dspark {
        ensure!(catalog.native_dspark_experts(), "V4.1 compressed dSpark owner manifest is not supported by this adapter");
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("V41_EXPERT_AOT.json"))
            .context("missing V4.1 dSpark coordinator allocation manifest")?)?;
        let resident = resident_weights(checkpoint, ranks, true).context("unsupported V4.1 dSpark resident format")?;
        let bytes = resident.iter().flatten().filter(|i| i.category == Category::Drafter).map(|i| i.bytes).sum();
        let staging = catalog.expert_staging(crate::V41ExpertSelection::Dspark { stage: 0, expert: 0 })?.staging_bytes() as u64;
        Some(DraftInventory { resident: bytes, runtime: programs.dspark(&manifest, rows, shape.concurrency, 5, ranks,
            mode.draft())?, marks: marks * 3 * 128 * 528,
            staging: [staging, 0], loading_experts: [3 * native_expert_bytes(128, 5120, 2304), 0],
            loader_peak: [bytes, 0], ..Default::default() })
    } else { None };
    let automatic = options.pool_tokens == Some(0);
    let mut pool = PoolPolicy::resolve(&options.rtx_bytes, shape.context_tokens, options.pool_tokens, 512, spark_ranks == 0)
        .with_automatic_context(options.context_tokens == 0);
    if pool.requested.is_none() {
        pool.floor = pool.floor.max(shape.context_tokens.div_ceil(512) * 512 + shape.concurrency * 512);
        pool.target = pool.target.min(options.target_pool_tokens).max(pool.floor);
    }
    pool.ceiling = pool.requested.unwrap_or(pool.target).div_ceil(512) * 512;
    let mut inventory = Inventory { gpus: Vec::new(), spark_ranks, peer_access: ranks == 2 };
    for (gpu, &total) in options.rtx_bytes.iter().enumerate() {
        let arch = ArchContext::coordinator(total, None);
        let ceiling = if total <= 32 * GIB {
            cuteafd_core::serving_capacity::admission_ceiling(total, 97,
                cuteafd_core::serving_capacity::small_card_headroom_bytes(total))?
        } else if automatic { (u128::from(total) * 97 / 100) as u64 } else { total };
        let context = arch.context_bytes + arch.cublas_bytes;
        inventory.gpus.push(GpuBudget { capacity_bytes: total, headroom_bytes: total - ceiling,
            baseline: Baseline::Planned { context_bytes: context, loaded_bytes: NATIVE_CODE_ESTIMATE } });
        devices[gpu].capacity_bytes = ceiling;
        devices[gpu].items.clear();
        devices[gpu].items.push(Item::new(Category::Runtime, "CUDA context and cuBLAS", "SM120 architecture inventory", context, Basis::Calibrated));
        devices[gpu].items.push(Item::new(Category::Runtime, "native LoadedCode estimate", "unqualified native code/ready graph residency", NATIVE_CODE_ESTIMATE, Basis::Estimated));
        fixed.extend(v41::growth_demands(total, ranks, automatic, gpu as u8));
    }
    let onboard = if spark_ranks == 0 { Onboard::Layers(40) } else {
        options.onboard.or(options.local_expert_layers.map(Onboard::Layers)).unwrap_or(Onboard::Auto)
    };
    let mut request = v41::request(inventory, pool, vec![0; ranks], experts, workspace, onboard);
    request.fixed = fixed;
    request.startup = startup.clone();
    if let Some(draft) = &draft { request.startup.extend(owners.draft_load_phases(draft)?); }
    request.minimum_expert_layers = usize::from(ranks == 2 && !matches!(onboard, Onboard::Layers(0)));
    if let Some(draft) = &draft { request.movables.push(v41::drafter(draft, ranks)?); }
    v41::cache_demands(&mut request, &checkpoint.config, shape.concurrency, turns, false)?;
    let mut placement = solve_with_weight_limit(&request, draft.as_ref(), options.weight_budget_bytes)?;
    let mut working = SparkWorkingSet { rtx_layers: (0..placement.onboard_layers).collect(),
        spark_layers: (placement.onboard_layers..40).collect(), spark_ranks, prefill_lanes: 1 };
    let mut sparks = spark_devices(report, model, checkpoint, options, SPARK, &working, 2, &mut waste, &mut notes);
    update_sparks(catalog, &working, &mut sparks)?;
    // A mandatory encoder displaces an optional expert prefix, not the pool.
    // Resolve its policy against only compulsory experts, then admit its owners.
    let mut encoder_request = request.clone();
    if request.inventory.spark_ranks > 0 && request.onboard.layers(40).is_none() {
        encoder_request.onboard = Onboard::Layers(request.minimum_expert_layers);
        encoder_request.pool.pin_selected(placement.pool_tokens);
    }
    let encoder_budget = placement::solve(&encoder_request)?;
    let mut encoder_devices = devices.clone();
    for (d, items) in encoder_devices.iter_mut().zip(&encoder_budget.items) { d.items.extend(items.iter().cloned()); }
    resolve_encoder(report, &v41_towers(report), &mut encoder_devices, &mut sparks, &vec![0; ranks], options, &mut notes);
    for device in &encoder_devices {
        for item in device.items.iter().filter(|i| i.group.starts_with("vision ")) {
            let demand = Demand::new(device.index as u8, item.category, item.group.clone(), item.bytes, item.basis);
            owners.add_vision(demand.clone())?;
            request.fixed.push(demand);
        }
    }
    if request.fixed.len() > request.fixed.iter().filter(|d| !d.group.starts_with("vision ")).count() {
        if let Some(draft) = &draft {
            request.startup = startup.clone();
            request.startup.extend(owners.draft_load_phases(draft)?);
        }
        placement = solve_with_weight_limit(&request, draft.as_ref(), options.weight_budget_bytes)?;
    }
    v41::validate_pool_floor(&request.pool, &placement)?;
    working.rtx_layers = (0..placement.onboard_layers).collect();
    working.spark_layers = (placement.onboard_layers..40).collect();
    if working.spark_layers.is_empty() { sparks.clear(); }
    update_sparks(catalog, &working, &mut sparks)?;
    charge_spark_devices(report, model, checkpoint, options, SPARK, &working,
        spark_endpoints("deepseek_v41", working.prefill_lanes), &mut sparks, &mut notes,
        |d| format!("Spark rank {} I/O admission", d.index));
    for spark in &mut sparks {
        for item in &mut spark.items {
            if item.group == "expert waves" || item.group == "vision CUDA overhead" { item.basis = Basis::Estimated; }
        }
    }
    for (device, items) in devices.iter_mut().zip(&placement.items) {
        device.items.extend(items.iter().cloned());
        device.kv_tokens = placement.pool_tokens;
    }
    report.attention_placement = placement.attention_placement;
    notes.push(format!("attention placement {}", placement.attention_placement));
    notes.push(format!("placement: {}", placement.summary()));
    notes.push(format!("{} routed backbone layers resident on RTX", placement.onboard_layers));
    let pool_tokens = placement.pool_tokens;
    report.solver_placement = Some(placement);
    devices.extend(sparks);
    Ok(MemoryLayout { devices, pool_tokens, waste, notes })
}

fn solve_with_weight_limit(request: &placement::PlacementRequest, draft: Option<&DraftInventory>,
    limit: u64) -> Result<placement::Placement> {
    let mut placement = placement::solve(request)?;
    loop {
        let fits = retained_weight_bytes(request, &placement, draft)?.into_iter().all(|bytes| bytes <= limit);
        if fits { return Ok(placement); }
        ensure!(request.inventory.spark_ranks > 0 && request.onboard.layers(40).is_none()
            && placement.onboard_layers > request.minimum_expert_layers,
            "V4.1 coordinator weight budget exceeded; mandatory owners require more than {limit} bytes");
        let mut trial = request.clone();
        trial.onboard = Onboard::Layers(placement.onboard_layers - 1);
        // Keep pool-first's admitted pool while reducing only the optional prefix.
        if matches!(request.onboard, Onboard::Auto) {
            trial.pool.pin_selected(placement.pool_tokens);
        }
        placement = placement::solve(&trial)?;
    }
}

fn retained_weight_bytes(request: &placement::PlacementRequest, placed: &placement::Placement,
    draft: Option<&DraftInventory>) -> Result<Vec<u64>> {
    let ranks = request.inventory.gpus.len();
    (0..ranks).map(|gpu| {
        let mut bytes = request.fixed.iter().filter(|d| usize::from(d.gpu) == gpu
            && (matches!(d.category, Category::Weights | Category::Embedding | Category::Tables)
                || d.group == "external drafter weights"))
            .try_fold(0u64, |sum, d| sum.checked_add(d.bytes).context("V4.1 weight budget overflow"))?;
        if gpu == ranks - 1 {
            bytes = bytes.checked_add(draft.map_or(0, |d| d.resident)).context("V4.1 draft weight budget overflow")?;
        }
        for (layer, assigned) in request.layers.iter().zip(&placed.layers) {
            if let Some(cost) = layer.experts {
                let resident = match assigned.experts {
                    placement::ExpertHome::RtxTp2 => cost.half[gpu].resident,
                    placement::ExpertHome::RtxWhole { gpu: owner } if usize::from(owner) == gpu => cost.whole.resident,
                    _ => 0,
                };
                bytes = bytes.checked_add(resident).context("V4.1 expert weight budget overflow")?;
            }
        }
        Ok(bytes)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_ready_census_residuals_fit_the_native_estimate() {
        // Raw ready untracked samples from the min/max native startup census.
        // Removing the calibrated context/cuBLAS leaves a combined code/graph
        // residual, NOT pure LoadedCode or a 64 MiB ready-parity qualification.
        let arch = ArchContext::coordinator(96 * GIB, None);
        assert_eq!(arch.context_bytes + arch.cublas_bytes, 657_719_296);
        for (card, untracked, residual, slack) in [
            ("min GPU0", 2_149_527_512, 1_491_808_216, 655_675_432),
            ("max GPU0", 1_771_348_088, 1_113_628_792, 1_033_854_856),
            ("max GPU1", 2_010_851_632, 1_353_132_336, 794_351_312),
        ] {
            let combined = untracked - arch.context_bytes - arch.cublas_bytes;
            assert_eq!(combined, residual, "{card}");
            assert!(combined <= NATIVE_CODE_ESTIMATE, "{card}");
            assert_eq!(NATIVE_CODE_ESTIMATE - combined, slack, "{card}");
        }
    }

    #[test]
    fn retained_weight_ceiling_limits_optional_experts_not_their_arena() -> Result<()> {
        let inventory = Inventory { gpus: vec![GpuBudget { capacity_bytes: 96 * GIB,
            headroom_bytes: 0, baseline: Baseline::Planned { context_bytes: 0, loaded_bytes: 0 } }],
            spark_ranks: 4, peer_access: false };
        let expert = placement::ExpertCost { whole: placement::Bytes2 { resident: 4 * GIB, staging: 8 * GIB },
            half: Default::default(), tp2: false, spark_ok: true };
        let mut request = v41::request(inventory, PoolPolicy::resolve(&[96 * GIB], 512, None, 512, false),
            vec![512 * 1024], vec![expert; 40], 16 * GIB, Onboard::Auto);
        request.fixed.push(Demand::new(0, Category::Weights, "backbone", 2 * GIB, Basis::Exact));
        let unconstrained = placement::solve(&request)?;
        for policy in [Onboard::Auto, Onboard::ExpertsFirst { pool_floor: 262_144 }] {
            request.onboard = policy;
            let placed = solve_with_weight_limit(&request, None, 10 * GIB)?;
            assert_eq!(placed.onboard_layers, 2);
            assert_eq!(retained_weight_bytes(&request, &placed, None)?, vec![10 * GIB]);
            assert!(placed.items[0].iter().filter(|i| i.category == Category::Experts)
                .map(|i| i.bytes).sum::<u64>() > 10 * GIB);
            assert!(placed.pool_tokens >= 262_144);
            if matches!(policy, Onboard::Auto) { assert_eq!(placed.pool_tokens, unconstrained.pool_tokens); }
        }
        request.onboard = Onboard::Layers(3);
        assert!(solve_with_weight_limit(&request, None, 10 * GIB).is_err());
        request.onboard = Onboard::Auto;
        request.minimum_expert_layers = 1;
        assert!(solve_with_weight_limit(&request, None, 5 * GIB).is_err());
        Ok(())
    }
}

fn update_sparks(catalog: &crate::OfficialV41Catalog, working: &SparkWorkingSet, devices: &mut [DeviceLayout]) -> Result<()> {
    let shape = catalog.routed_experts();
    for device in devices {
        let bytes = if let Some(exl3) = catalog.exl3() {
            working.spark_layers.iter().try_fold(0u64, |sum, &layer| -> Result<u64> {
                sum.checked_add(exl3.residency(crate::V41Exl3Layer::Backbone(layer), working.spark_ranks,
                    device.index as usize)?.device_arena_layout()?.1 as u64).context("V4.1 Spark EXL3 residency overflow")
            })?
        } else {
            ensure!(working.spark_ranks > 0 && shape.intermediate % working.spark_ranks == 0,
                "V4.1 Spark expert width is not divisible by rank count");
            let width = shape.intermediate as u64 / working.spark_ranks as u64;
            let per_layer = if catalog.nvfp4().is_some() { nvfp4_expert_bytes(shape.experts as u64, shape.hidden as u64, width) }
                else { native_expert_bytes(shape.experts as u64, shape.hidden as u64, width) };
            per_layer.checked_mul(working.spark_layers.len() as u64).context("V4.1 Spark residency overflow")?
        };
        if let Some(item) = device.items.iter_mut().find(|i| i.group == "routed_expert") { item.bytes = bytes; item.basis = Basis::Formula; }
    }
    Ok(())
}
