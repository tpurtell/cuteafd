//! Deterministic placement in TJ's order: layer modes, fixed demands,
//! mandatory movables, the KV pool, then contiguous RTX expert ranges.
use super::*;
use cuteafd_core::memory_layout::{Basis, Category, Item};

/// Places `request`. Pure arithmetic over the request: the planner and the
/// runtime get the same answer for the same inputs.
pub fn solve(request: &PlacementRequest) -> Result<Placement, PlacementError> {
    request.executor.check_attention(request.attention_placement, request.inventory.gpus.len(), request.inventory.peer_access)?;
    let mut best = solve_once(request, &[]);
    if request.attention_placement.is_some() || request.inventory.gpus.len() != 2 || !request.inventory.peer_access
        || best.as_ref().is_ok_and(|p| p.pool_tokens >= request.pool.target) { return best; }
    // Step 7: largest replicated-KV saving first. Only auto may flip; only
    // modes the executor actually runs are candidates.
    let mut kinds = std::collections::BTreeMap::<AttentionClass, u64>::new();
    for layer in &request.layers {
        if layer.kv_unit.unit_bytes_context.is_some() || layer_whole_mode(request, layer, 0).is_some()
            && layer_whole_mode(request, layer, 1).is_some() {
            let total = |bytes: [u64; 2]| bytes[0].checked_add(bytes[1]).ok_or(PlacementError::Overflow("mode savings"));
            let owned = layer.kv_unit.unit_bytes_context.map(total).transpose()?.unwrap_or(layer.kv_unit.unit_bytes_whole);
            let saving = total(layer.kv_unit.unit_bytes_split)?.saturating_sub(owned);
            let entry = kinds.entry(layer.kind).or_default();
            *entry = entry.checked_add(saving).ok_or(PlacementError::Overflow("mode savings"))?;
        }
    }
    let mut kinds = kinds.into_iter().collect::<Vec<_>>();
    kinds.sort_by_key(|&(kind, saving)| (std::cmp::Reverse(saving), kind));
    let mut flips = Vec::new();
    for (kind, _) in kinds {
        let prior_flips = flips.clone();
        let mut kind_best = best.as_ref().map(|p| p.pool_tokens).unwrap_or(0);
        let mut selected_flips = prior_flips.clone();
        for mode in [AttentionPlacement::Context, AttentionPlacement::Layers] {
            if mode == AttentionPlacement::Context && !request.layers.iter().any(|l| l.kind == kind && l.kv_unit.unit_bytes_context.is_some()) { continue; }
            if mode == AttentionPlacement::Layers && !request.layers.iter().any(|l| l.kind == kind
                && layer_whole_mode(request, l, 0).is_some() && layer_whole_mode(request, l, 1).is_some()) { continue; }
            if request.executor.check_attention(Some(mode), 2, true).is_err() { continue; }
            flips = prior_flips.clone();
            flips.retain(|(k, _)| *k != kind);
            flips.push((kind, mode));
            let trial = solve_once(request, &flips);
            if trial.as_ref().is_ok_and(|p| p.pool_tokens >= request.pool.target) { return prune_flips(request, &flips); }
            if let Ok(p) = &trial {
                if p.pool_tokens >= kind_best { kind_best = p.pool_tokens; selected_flips = flips.clone(); }
            }
            if trial.as_ref().is_ok_and(|p| best.as_ref().map_or(true, |b| p.pool_tokens >= b.pool_tokens)) { best = trial; }
        }
        flips = selected_flips;
    }
    if let Ok(placement) = &best {
        let chosen = placement.attention_by_kind.iter().copied().filter(|(_, mode)| *mode != request.executor.attention_default()).collect::<Vec<_>>();
        if !chosen.is_empty() { return prune_flips(request, &chosen); }
    }
    best
}

fn prune_flips(request: &PlacementRequest, flips: &[(AttentionClass, AttentionPlacement)]) -> Result<Placement, PlacementError> {
    let mut flips = flips.to_vec();
    let mut placement = solve_once(request, &flips)?;
    loop {
        let mut removed = false;
        for i in 0..flips.len() {
            let mut trial = flips.clone();
            trial.remove(i);
            if let Ok(candidate) = solve_once(request, &trial) {
                // Removing context can also recover an odd pool unit that its
                // even-unit constraint rounded away.
                if candidate.pool_tokens >= placement.pool_tokens {
                    flips = trial;
                    placement = candidate;
                    removed = true;
                    break;
                }
            }
        }
        if !removed { return Ok(placement); }
    }
}

fn solve_once(request: &PlacementRequest, flips: &[(AttentionClass, AttentionPlacement)]) -> Result<Placement, PlacementError> {
    use PlacementError::Overflow;
    let gpus = request.inventory.gpus.len();
    if !(1..=2).contains(&gpus) { return Err(PlacementError::Inventory("one or two coordinator GPUs")); }
    if request.layers_first_gpu > 1 { return Err(PlacementError::Inventory("layers first GPU must be 0 or 1")); }
    if request.pool.unit_rows == 0 { return Err(PlacementError::Inventory("zero pool unit rows")); }
    if request.pool_overhead.len() != gpus { return Err(PlacementError::Inventory("pool overhead per GPU")); }
    if request.fixed.iter().any(|d| usize::from(d.gpu) >= gpus)
        || request.movables.iter().any(|m| m.allowed.is_empty() || m.allowed.iter().any(|&g| usize::from(g) >= gpus)) {
        return Err(PlacementError::Inventory("demand on an absent GPU"));
    }
    for movable in &request.movables {
        let mut keys = Vec::new();
        for conditional in &movable.conditional {
            if usize::from(conditional.placement_gpu) >= gpus || keys.contains(&conditional.placement_gpu)
                || conditional.demands.iter().any(|d| usize::from(d.gpu) >= gpus) {
                return Err(PlacementError::Inventory("invalid movable conditional demand"));
            }
            keys.push(conditional.placement_gpu);
        }
    }
    let split = gpus == 2 && request.inventory.peer_access;
    let tp2 = split && request.layers.iter().any(|l| l.experts.is_some_and(|c| c.tp2));

    // 1. Explicit attention placement is strict. Auto retains the existing
    // policy until a qualified memory lever is needed.
    let executor = &request.executor;
    let layer_selected = |layer: &LayerDemand| flips.iter().find(|(kind, _)| *kind == layer.kind).map(|(_, m)| *m)
        .or(request.attention_placement).unwrap_or_else(|| executor.attention_default()) == AttentionPlacement::Layers
        && layer_whole_mode(request, layer, 0).is_some() && layer_whole_mode(request, layer, 1).is_some();
    let switch = if request.layers.iter().any(&layer_selected) { layer_switch(request, &layer_selected)? } else { 0 };
    let modes = request.layers.iter().enumerate().map(|(index, layer)| {
        let selected = flips.iter().find(|(kind, _)| *kind == layer.kind).map(|(_, m)| *m)
            .or(request.attention_placement)
            .or_else(|| (executor.attention_default() != AttentionPlacement::Heads).then(|| executor.attention_default()));
        let desired = match selected {
            Some(AttentionPlacement::Context) if layer.kv_unit.unit_bytes_context.is_some() => Some(LayerMode::ContextSplit),
            Some(AttentionPlacement::Context) => Some(LayerMode::HeadSplit),
            Some(AttentionPlacement::Layers) if layer_selected(layer) => {
                let owner = if index < switch { request.layers_first_gpu } else { 1 - request.layers_first_gpu };
                layer_whole_mode(request, layer, owner)
            }
            Some(AttentionPlacement::Layers) => Some(LayerMode::HeadSplit),
            // Heads is the established layout, including V4.1's ranges.
            Some(AttentionPlacement::Heads) => None,
            _ => None,
        };
        if let Some(mode) = desired {
            if !layer.modes.contains(&mode) || !executable(mode, gpus, split) || !executor.runs(mode) {
                return Err(PlacementError::UnsupportedMode { family: executor.family, layer: index, mode });
            }
            return Ok(mode);
        }
        let wanted = request.policy.preference(layer.kind);
        wanted.iter().copied().chain(layer.modes.iter().copied())
            .find(|mode| layer.modes.contains(mode) && executable(*mode, gpus, split) && executor.runs(*mode))
            .ok_or_else(|| PlacementError::NoMode { layer: index, allowed: layer.modes.clone() })
    }).collect::<Result<Vec<_>, _>>()?;
    // Residual homes and the hops the modes imply (one GPU never hops).
    let (residual, hops) = residual_plan(&modes, &request.hops, gpus);
    if let Some(hop) = hops.iter().find(|h| h.at != HopPoint::Entry).filter(|_| !executor.hops) {
        return Err(PlacementError::UnsupportedHop { family: executor.family, hop: *hop });
    }

    // 2. Fixed demands, including mode-dependent layer weights.
    let mut items: Vec<Vec<Item>> = vec![Vec::new(); gpus];
    let mut used = vec![0u64; gpus];
    let charge = |items: &mut Vec<Vec<Item>>, used: &mut Vec<u64>, gpu: usize, item: Item| -> Result<(), PlacementError> {
        used[gpu] = used[gpu].checked_add(item.bytes).ok_or(Overflow("fixed demands"))?;
        items[gpu].push(item);
        Ok(())
    };
    let context = modes.contains(&LayerMode::ContextSplit);
    let context_demands = if context { request.context_buffers.demands()? } else { Vec::new() };
    for demand in request.fixed.iter().chain(&context_demands) {
        charge(&mut items, &mut used, usize::from(demand.gpu),
            Item::new(demand.category, demand.group.clone(), "", demand.bytes, demand.basis))?;
    }
    let mut layer_weights = vec![0u64; gpus];
    let mut unit_bytes = request.pool_overhead.clone();
    let mut layer_fixed = vec![0u64; gpus];
    for (layer, mode) in request.layers.iter().zip(&modes) {
        for gpu in 0..gpus {
            let (weights, kv) = per_gpu(layer, *mode, gpu);
            let state = match mode {
                LayerMode::HeadSplit | LayerMode::ContextSplit => layer.fixed_bytes.split[gpu],
                LayerMode::Whole { gpu: owner, .. } if usize::from(*owner) == gpu => layer.fixed_bytes.whole,
                LayerMode::Whole { .. } => 0,
            };
            layer_fixed[gpu] = layer_fixed[gpu].checked_add(state).ok_or(Overflow("layer state"))?;
            layer_weights[gpu] = layer_weights[gpu].checked_add(weights).ok_or(Overflow("layer weights"))?;
            unit_bytes[gpu] = unit_bytes[gpu].checked_add(kv).ok_or(Overflow("pool unit"))?;
        }
    }
    for (gpu, &bytes) in layer_weights.iter().enumerate() {
        if bytes > 0 { charge(&mut items, &mut used, gpu, Item::new(Category::Weights, "layers", "", bytes, Basis::Exact))?; }
    }
    for (gpu, &bytes) in layer_fixed.iter().enumerate() {
        if bytes > 0 { charge(&mut items, &mut used, gpu, Item::new(Category::Kv, "layer state and marks", "", bytes, Basis::Formula))?; }
    }
    // Hop receive buffers are fixed demands of the modes (charged before the pool).
    let hop_bytes = hop_buffer_bytes(&hops, &request.hops, gpus).ok_or(Overflow("hop buffers"))?;
    for (gpu, &bytes) in hop_bytes.iter().enumerate() {
        if bytes > 0 { charge(&mut items, &mut used, gpu, Item::new(Category::Transport, "residual hops", "", bytes, Basis::Formula))?; }
    }
    let available = request.inventory.gpus.iter().map(GpuBudget::available).collect::<Vec<_>>();

    // 3. Mandatory movables, largest first. Conditional reservations participate
    // in candidate choice; ordinary movables retain the most-free-GPU policy.
    let mut order: Vec<usize> = (0..request.movables.len()).collect();
    let totals = request.movables.iter().map(|m| m.parts.iter().try_fold(0u64, |n, p| n.checked_add(p.resident))
        .ok_or(Overflow("movable"))).collect::<Result<Vec<_>, _>>()?;
    order.sort_by_key(|&i| std::cmp::Reverse(totals[i]));
    let mut arenas: Vec<Arena> = (0..gpus).map(|_| Arena::default()).collect();
    let mut movables = Vec::new();
    let mut selected_fixed = request.fixed.clone();
    for index in order {
        let movable = &request.movables[index];
        if !movable.conditional.is_empty() {
            let (gpu, candidate_used, candidate_arenas, demands) = conditional_movable(
                request, movable, &available, &used, &arenas, &unit_bytes, context)?;
            used = candidate_used;
            arenas = candidate_arenas;
            if !movable.expert_arena {
                items[gpu].push(Item::new(Category::Drafter, format!("{:?}", movable.id), "",
                    totals[index], Basis::Exact));
            }
            selected_fixed.extend(demands.iter().cloned());
            for demand in demands {
                items[usize::from(demand.gpu)].push(Item::new(demand.category, demand.group, "", demand.bytes, demand.basis));
            }
            movables.push((movable.id, gpu as u8));
            continue;
        }
        let gpu = usize::from(*movable.allowed.iter()
            .max_by_key(|&&g| (available[usize::from(g)].saturating_sub(used[usize::from(g)] + arenas[usize::from(g)].peak), std::cmp::Reverse(g)))
            .expect("allowed is non-empty"));
        if movable.expert_arena {
            let arena = &mut arenas[gpu];
            arena.open(request.expert_workspace);
            for part in &movable.parts { arena.add(*part)?; }
        } else {
            let bytes = movable.parts.iter().try_fold(0u64, |n, p| n.checked_add(p.resident)).ok_or(Overflow("movable"))?;
            charge(&mut items, &mut used, gpu, Item::new(Category::Drafter, format!("{:?}", movable.id), "", bytes, Basis::Exact))?;
        }
        movables.push((movable.id, gpu as u8));
    }
    movables.sort_by_key(|(id, _)| *id as u8);

    // 4/5. The KV pool and RTX expert layers, in the order the onboard fixes.
    let routed = count_moe(request);
    // Spark-free only means something when there are routed layers to hold.
    let spark_free = request.inventory.spark_ranks == 0 && routed > 0;
    let first_moe = request.layers.iter().position(|l| l.experts.is_some()).unwrap_or(request.layers.len());
    for gpu in 0..gpus {
        if available[gpu].checked_sub(used[gpu] + arenas[gpu].peak).is_none() {
            return Err(PlacementError::Mandatory { gpu: gpu as u8, what: describe(&selected_fixed, gpu, &movables) });
        }
    }
    let fixed_layers = match request.onboard.layers(routed) {
        None if spark_free => Some(routed),
        other => other,
    };
    if spark_free && fixed_layers != Some(routed) {
        return Err(PlacementError::SparkFree { layers: routed, placed: fixed_layers.unwrap_or(0) });
    }
    // Experts first: the most layers whose fixed-onboard pool still meets the
    // floor (pool output is monotone in the layer count, so bisect).
    let fixed_layers = match (fixed_layers, request.onboard) {
        (None, Onboard::ExpertsFirst { pool_floor }) => {
            // An explicit pool is the floor the layers must leave (and then the pool).
            let floor = request.pool.requested.unwrap_or(pool_floor).max(request.pool.floor);
            let fits = |n: usize| -> bool {
                let mut trial = request.clone();
                trial.onboard = Onboard::Layers(n);
                trial.pool.requested = None;
                trial.pool.floor = floor;
                solve_once(&trial, flips).is_ok()
            };
            let (mut lo, mut hi) = (0usize, routed);
            if !fits(0) {
                // With Sparks an automatic pool may fall short of the context: no RTX layers and
                // the largest pool above v2's floor, the serving context clamped to it (v2 served
                // a 31.8 GiB V4 Flash card at 905K tokens).
                let fallback = request.pool.requested.is_none() && !spark_free && pool_floor < floor && {
                    let mut trial = request.clone();
                    trial.onboard = Onboard::Layers(0);
                    trial.pool.floor = pool_floor;
                    solve_once(&trial, flips).is_ok()
                };
                if !fallback { return Err(match request.pool.requested {
                    Some(requested) => PlacementError::PoolDoesNotFit { requested, fit: 0 },
                    None => PlacementError::Mandatory { gpu: 0,
                        what: format!("a {floor}-token pool beside {}", describe(&selected_fixed, 0, &movables)) },
                }); }
                let mut trial = request.clone();
                trial.onboard = Onboard::Layers(0);
                trial.pool.floor = pool_floor;
                return solve_once(&trial, flips);
            }
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if fits(mid) { lo = mid } else { hi = mid - 1 }
            }
            Some(lo)
        }
        (fixed, _) => fixed,
    };
    let experts_first = matches!(request.onboard, Onboard::ExpertsFirst { .. });
    let (units, ranges, homes, tp2_range) = match fixed_layers {
        // Pool first: reserve the target, then fill each GPU's arena in order.
        None => {
            let fit = pool_fit(&available, &used, &arenas, &unit_bytes);
            let units = pool_units(request, if context { fit / 2 * 2 } else { fit })?;
            let used = with_pool(&used, &unit_bytes, units)?;
            let caps = [if request.expert_gpus > 0 { usize::MAX } else { 0 }, if request.expert_gpus > 1 { usize::MAX } else { 0 }];
            let (ranges, homes, _, tp2_range) = place_experts(request, &available, &used, &mut arenas, first_moe,
                request.layers.len(), &caps)?;
            (units, ranges, homes, tp2_range)
        }
        // Fixed onboard: exactly `n` routed layers (the contiguous prefix the
        // executors run), split over the GPUs where the pool is largest; the
        // pool is the output.
        Some(n) => {
            let end = nth_moe_end(request, first_moe, n);
            let mut best: Option<(u64, Vec<ExpertRange>, Vec<ExpertHome>, Option<Tp2Range>)> = None;
            let splits: Vec<usize> = if tp2 || gpus == 1 || request.expert_gpus < 2 { vec![n] } else { (0..=n).rev().collect() };
            let mut most = 0;
            for on_first in splits {
                let mut trial = arenas.clone();
                let caps = [if request.expert_gpus > 0 { on_first } else { 0 }, if gpus == 2 { n - on_first } else { 0 }];
                let (ranges, homes, next, tp2_range) = place_experts(request, &available, &used, &mut trial, first_moe, end, &caps)?;
                let placed = homes.iter().filter(|h| matches!(h, ExpertHome::RtxWhole { .. } | ExpertHome::RtxTp2)).count();
                most = most.max(placed);
                if next < end || placed < n { continue; }
                let fit = pool_fit(&available, &used, &trial, &unit_bytes);
                if best.as_ref().is_none_or(|(units, ..)| fit > *units) { best = Some((fit, ranges, homes, tp2_range)); }
            }
            let Some((fit, ranges, homes, tp2_range)) = best else {
                return Err(if spark_free { PlacementError::SparkFree { layers: routed, placed: most } }
                    else { PlacementError::ExpertLayers { requested: n, placed: most } });
            };
            let units = match request.pool.requested {
                Some(requested) => pool_units(request, fit).map_err(|_| PlacementError::PoolDoesNotFit { requested,
                    fit: fit.saturating_mul(request.pool.unit_rows) })?,
                // Experts first keeps v2's automatic ceiling: the target.
                None if experts_first => fit.min(request.pool.wanted_units()),
                None => fit.min(request.pool.ceiling_units()),
            };
            let tokens = units.saturating_mul(request.pool.unit_rows);
            if tokens < request.pool.floor {
                return Err(PlacementError::BelowFloor { pool: tokens, floor: request.pool.floor, layers: n,
                    short: request.pool.floor - tokens });
            }
            (units, ranges, homes, tp2_range)
        }
    };
    if units == 0 {
        return Err(PlacementError::Mandatory { gpu: 0, what: format!("one KV unit beside {}", describe(&selected_fixed, 0, &movables)) });
    }
    let units = if context {
        if request.pool.requested.is_some() && units % 2 != 0 {
            return Err(PlacementError::Inventory("context pool must contain an even number of logical units"));
        }
        units / 2 * 2
    } else { units };
    if context && (units == 0 || units.saturating_mul(request.pool.unit_rows) < request.pool.floor) {
        let pool = units.saturating_mul(request.pool.unit_rows);
        return Err(PlacementError::BelowFloor { pool, floor: request.pool.floor, layers: 0,
            short: request.pool.floor.saturating_sub(pool) });
    }
    let onboard_layers = homes.iter().filter(|h| matches!(h, ExpertHome::RtxWhole { .. } | ExpertHome::RtxTp2)).count();

    // Charged items: pool records and expert arenas.
    let pool_tokens = units.checked_mul(request.pool.unit_rows).ok_or(Overflow("pool tokens"))?;
    for gpu in 0..gpus {
        if unit_bytes[gpu] > 0 {
            items[gpu].push(Item::new(Category::Kv, "records", "", unit_bytes[gpu] * units, Basis::Formula));
        }
        if let Some(range) = tp2_range {
            items[gpu].push(Item::new(Category::Experts,
                format!("TP2 expert layer halves {}..{}", range.first, range.first + range.layers),
                "tp2", range.peak_bytes[gpu], Basis::Formula));
        }
        let range = ranges[gpu];
        if range.peak_bytes > 0 {
            items[gpu].push(Item::new(Category::Experts,
                format!("resident routed layers {}..{}", range.first, range.first + range.layers),
                "local", range.peak_bytes, Basis::Formula));
        }
    }
    let homes = request.layers.iter().zip(homes)
        .map(|(layer, home)| if layer.experts.is_none() { ExpertHome::Dense } else { home }).collect::<Vec<_>>();
    Ok(Placement {
        pool_tokens,
        attention_placement: if context { AttentionPlacement::Context }
            else if request.attention_placement == Some(AttentionPlacement::Layers) || flips.iter().any(|(_, m)| *m == AttentionPlacement::Layers) {
                AttentionPlacement::Layers
            } else { AttentionPlacement::Heads },
        attention_by_kind: request.layers.iter().map(|layer| {
            let mode = flips.iter().find(|(kind, _)| *kind == layer.kind).map(|(_, mode)| *mode)
                .or(request.attention_placement).unwrap_or_else(|| executor.attention_default());
            (layer.kind, mode)
        }).collect::<std::collections::BTreeMap<_, _>>().into_iter().collect(),
        peer_row_bytes: if context {
            request.layers.iter().zip(&modes).filter(|(_, m)| **m == LayerMode::ContextSplit).try_fold(0u64, |sum, (layer, _)|
                request.context_buffers.query_row_bytes.checked_add(request.context_buffers.partial_row_bytes)
                    .and_then(|n| n.checked_add(if layer.context_indexer { request.context_buffers.candidate_row_bytes } else { 0 }))
                    .and_then(|n| n.checked_add(sum))).ok_or(Overflow("context peer rows"))?
        } else { 0 },
        onboard_layers,
        layers: modes.into_iter().zip(homes).map(|(mode, experts)| LayerAssignment { mode, experts }).collect(),
        movables,
        expert_ranges: ranges,
        residual,
        hops,
        tp2: tp2_range,
        items,
    })
}

fn layer_whole_mode(request: &PlacementRequest, layer: &LayerDemand, gpu: u8) -> Option<LayerMode> {
    [FfnMode::Split, FfnMode::Owner].into_iter().map(|ffn| LayerMode::Whole { gpu, ffn })
        .find(|mode| layer.modes.contains(mode) && request.executor.runs(*mode))
}

/// One ownership boundary, never inside an indexer/colocate group. Compare
/// persistent bytes at the requested pool, then the pool left by fixed state.
fn layer_switch(request: &PlacementRequest, selected: &impl Fn(&LayerDemand) -> bool) -> Result<usize, PlacementError> {
    let mut best = None;
    for k in 0..=request.layers.len() {
        if request.layers.iter().enumerate().any(|(i, layer)| selected(layer) && layer.colocate.is_some_and(|group|
            request.layers.iter().enumerate().any(|(j, other)| selected(other) && other.colocate == Some(group) && (i < k) != (j < k)))) { continue; }
        let mut kv = [0u64; 2];
        let mut fixed = [0u64; 2];
        for (i, layer) in request.layers.iter().enumerate().filter(|(_, l)| selected(l)) {
            let gpu = usize::from(if i < k { request.layers_first_gpu } else { 1 - request.layers_first_gpu });
            kv[gpu] = kv[gpu].checked_add(layer.kv_unit.unit_bytes_whole).ok_or(PlacementError::Overflow("owned KV bytes"))?;
            fixed[gpu] = fixed[gpu].checked_add(layer.fixed_bytes.whole).ok_or(PlacementError::Overflow("owned state"))?;
        }
        let mut owned = [0u64; 2];
        for gpu in 0..2 { owned[gpu] = kv[gpu].checked_mul(request.pool.wanted_units()).and_then(|b| b.checked_add(fixed[gpu]))
            .ok_or(PlacementError::Overflow("owned pool bytes"))?; }
        let mut pool_bytes = request.pool_overhead.clone();
        let mut base = [0u64; 2];
        for demand in &request.fixed {
            let g = usize::from(demand.gpu);
            base[g] = base[g].checked_add(demand.bytes).ok_or(PlacementError::Overflow("ownership fixed"))?;
        }
        for (i, layer) in request.layers.iter().enumerate() {
            for g in 0..request.inventory.gpus.len() {
                let owner = usize::from(if i < k { request.layers_first_gpu } else { 1 - request.layers_first_gpu });
                let (bytes, state, weights) = if selected(layer) {
                    if g == owner { (layer.kv_unit.unit_bytes_whole, layer.fixed_bytes.whole, layer.weights.whole) } else { (0, 0, 0) }
                } else { (layer.kv_unit.unit_bytes_split[g], layer.fixed_bytes.split[g], layer.weights.split[g]) };
                pool_bytes[g] = pool_bytes[g].checked_add(bytes).ok_or(PlacementError::Overflow("ownership pool"))?;
                base[g] = base[g].checked_add(state).and_then(|n| n.checked_add(weights)).ok_or(PlacementError::Overflow("ownership fixed"))?;
            }
        }
        let fit = (0..request.inventory.gpus.len()).filter(|&g| pool_bytes[g] > 0).map(|g|
            request.inventory.gpus[g].available().saturating_sub(base[g]) / pool_bytes[g]).min().unwrap_or(u64::MAX);
        let score = (owned[0].max(owned[1]), std::cmp::Reverse(fit), k);
        if best.as_ref().is_none_or(|(old, _)| score < *old) { best = Some((score, k)); }
    }
    best.map(|(_, k)| k).ok_or(PlacementError::Inventory("no legal layer ownership boundary"))
}

/// The residual's home at every layer boundary (`[0]`: the embedding's GPU,
/// `[i + 1]`: after layer `i`) and the hops between them.
fn residual_plan(modes: &[LayerMode], spec: &HopSpec, gpus: usize) -> (Vec<ResidualHome>, Vec<Hop>) {
    let entry = if gpus == 2 { spec.entry_gpu } else { 0 };
    let mut homes = Vec::with_capacity(modes.len() + 1);
    homes.push(ResidualHome::Owned(entry));
    for &mode in modes {
        let home = *homes.last().expect("seeded");
        homes.push(home.transition(mode).home);
    }
    let hops = if gpus == 2 { plan_hops(modes, spec) } else { Vec::new() };
    (homes, hops)
}

/// Try cross-device reservations before selecting a movable's home. Ordinary
/// movables retain their historical most-free-device policy.
#[allow(clippy::type_complexity)]
fn conditional_movable(request: &PlacementRequest, movable: &Movable, available: &[u64],
    used: &[u64], arenas: &[Arena], unit_bytes: &[u64], context: bool)
    -> Result<(usize, Vec<u64>, Vec<Arena>, Vec<Demand>), PlacementError> {
    let resident = movable.parts.iter().try_fold(0u64, |n, p| n.checked_add(p.resident))
        .ok_or(PlacementError::Overflow("movable"))?;
    let mut best: Option<((u64, u64, std::cmp::Reverse<u8>), usize, Vec<u64>, Vec<Arena>, Vec<Demand>)> = None;
    let mut refusal = None;
    for &home in &movable.allowed {
        let gpu = usize::from(home);
        let mut trial_used = used.to_vec();
        let mut trial_arenas = arenas.to_vec();
        if movable.expert_arena {
            trial_arenas[gpu].open(request.expert_workspace);
            for &part in &movable.parts { trial_arenas[gpu].add(part)?; }
        } else {
            trial_used[gpu] = trial_used[gpu].checked_add(resident)
                .ok_or(PlacementError::Overflow("movable"))?;
        }
        let demands = movable.conditional.iter().find(|d| d.placement_gpu == home)
            .map_or_else(Vec::new, |d| d.demands.clone());
        for demand in &demands {
            let charged = usize::from(demand.gpu);
            trial_used[charged] = trial_used[charged].checked_add(demand.bytes)
                .ok_or(PlacementError::Overflow("movable conditional demand"))?;
        }
        let totals = trial_used.iter().zip(&trial_arenas).map(|(&bytes, arena)|
            bytes.checked_add(arena.peak).ok_or(PlacementError::Overflow("movable candidate peak")))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(charged) = (0..available.len()).find(|&g| totals[g] > available[g]) {
            refusal.get_or_insert_with(|| PlacementError::Mandatory { gpu: charged as u8,
                what: format!("{:?}, {}", movable.id, describe(&demands, charged, &[])) });
            continue;
        }
        let fit = pool_fit(available, &trial_used, &trial_arenas, unit_bytes);
        let fit = if context { fit / 2 * 2 } else { fit };
        let required = request.pool.requested.map_or(request.pool.floor, |n| n.max(request.pool.floor))
            .div_ceil(request.pool.unit_rows);
        if fit < required {
            refusal.get_or_insert_with(|| match request.pool.requested {
                Some(requested) => PlacementError::PoolDoesNotFit { requested,
                    fit: fit.saturating_mul(request.pool.unit_rows) },
                None => PlacementError::BelowFloor { pool: fit.saturating_mul(request.pool.unit_rows),
                    floor: request.pool.floor, layers: 0,
                    short: request.pool.floor.saturating_sub(fit.saturating_mul(request.pool.unit_rows)) },
            });
            continue;
        }
        let score = (fit, available[gpu] - totals[gpu], std::cmp::Reverse(home));
        if best.as_ref().is_none_or(|(previous, ..)| score > *previous) {
            best = Some((score, gpu, trial_used, trial_arenas, demands));
        }
    }
    best.map(|(_, gpu, used, arenas, demands)| (gpu, used, arenas, demands))
        .ok_or_else(|| refusal.unwrap_or(PlacementError::Inventory("no movable candidate")))
}

/// Whole units every KV-owning GPU holds beside what is charged and its arena.
fn pool_fit(available: &[u64], used: &[u64], arenas: &[Arena], unit_bytes: &[u64]) -> u64 {
    (0..available.len()).filter(|&gpu| unit_bytes[gpu] > 0)
        .map(|gpu| available[gpu].saturating_sub(used[gpu] + arenas[gpu].peak) / unit_bytes[gpu])
        .min().unwrap_or(u64::MAX)
}

fn with_pool(used: &[u64], unit_bytes: &[u64], units: u64) -> Result<Vec<u64>, PlacementError> {
    used.iter().zip(unit_bytes).map(|(&used, &unit)| unit.checked_mul(units).and_then(|b| b.checked_add(used))
        .ok_or(PlacementError::Overflow("pool bytes"))).collect()
}

/// Automatic pools stop at the target; an explicit pool is strict.
fn pool_units(request: &PlacementRequest, fit: u64) -> Result<u64, PlacementError> {
    let wanted = request.pool.wanted_units();
    match request.pool.requested {
        Some(requested) if wanted > fit => Err(PlacementError::PoolDoesNotFit { requested,
            fit: fit.saturating_mul(request.pool.unit_rows) }),
        Some(_) => Ok(wanted),
        None => Ok(wanted.min(fit)),
    }
}

fn count_moe(request: &PlacementRequest) -> usize {
    request.layers.iter().filter(|l| l.experts.is_some()).count()
}

/// The layer index just past the `n`th routed layer from `first`.
fn nth_moe_end(request: &PlacementRequest, first: usize, n: usize) -> usize {
    let mut seen = 0;
    for (index, layer) in request.layers.iter().enumerate().skip(first) {
        if seen == n { return index; }
        if layer.experts.is_some() { seen += 1; }
    }
    request.layers.len()
}

/// One contiguous RTX prefix from the first MoE layer up to `limit`: GPU0's
/// range, then GPU1's, each in its own arena and holding at most `caps[gpu]`
/// routed layers. Returns the ranges, homes and the first layer not placed.
fn place_experts(request: &PlacementRequest, available: &[u64], used: &[u64], arenas: &mut [Arena],
    first_moe: usize, limit: usize, caps: &[usize]) -> Result<(Vec<ExpertRange>, Vec<ExpertHome>, usize, Option<Tp2Range>), PlacementError> {
    if available.len() == 2 && request.inventory.peer_access
        && request.layers.iter().any(|l| l.experts.is_some_and(|c| c.tp2)) {
        return place_tp2(request, available, used, arenas, first_moe, limit, caps[0]);
    }
    let mut next = first_moe;
    let mut ranges = Vec::new();
    let mut homes = vec![ExpertHome::Spark; request.layers.len()];
    for (gpu, arena) in arenas.iter_mut().enumerate() {
        let budget = available[gpu].saturating_sub(used[gpu]);
        let first = next;
        let mut trial = arena.clone();
        let mut placed = 0;
        while next < limit && placed < caps[gpu] {
            let Some(cost) = request.layers[next].experts else { next += 1; continue };
            let mut candidate = trial.clone();
            candidate.open(request.expert_workspace);
            candidate.add(cost.whole)?;
            if candidate.peak > budget { break; }
            trial = candidate;
            homes[next] = ExpertHome::RtxWhole { gpu: gpu as u8 };
            next += 1;
            placed += 1;
        }
        *arena = trial;
        ranges.push(ExpertRange { first, layers: next - first, peak_bytes: arena.peak });
    }
    Ok((ranges, homes, next, None))
}

/// Whether the inventory can run `mode`: a head split, a split FFN or any
/// layer on GPU1 needs two GPUs with peer access.
fn executable(mode: LayerMode, gpus: usize, split: bool) -> bool {
    match mode {
        LayerMode::HeadSplit | LayerMode::ContextSplit | LayerMode::Whole { ffn: FfnMode::Split, .. } => split,
        LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } => true,
        LayerMode::Whole { gpu, .. } => usize::from(gpu) < gpus && split,
    }
}

/// A layer's weights and pool-unit bytes on `gpu` under `mode`.
fn per_gpu(layer: &LayerDemand, mode: LayerMode, gpu: usize) -> (u64, u64) {
    match mode {
        LayerMode::HeadSplit => (layer.weights.split[gpu], layer.kv_unit.unit_bytes_split[gpu]),
        LayerMode::ContextSplit => (layer.weights.split[gpu], layer.kv_unit.unit_bytes_context.expect("context layer")[gpu]),
        LayerMode::Whole { gpu: owner, .. } if usize::from(owner) == gpu => (layer.weights.whole, layer.kv_unit.unit_bytes_whole),
        LayerMode::Whole { .. } => (0, 0),
    }
}

fn describe(fixed: &[Demand], gpu: usize, movables: &[(MovableId, u8)]) -> String {
    let mut names: Vec<String> = fixed.iter().filter(|d| usize::from(d.gpu) == gpu).map(|d| d.group.clone()).collect();
    names.extend(movables.iter().filter(|(_, g)| usize::from(*g) == gpu).map(|(id, _)| format!("{id:?}")));
    names.join(", ")
}

/// One GPU's local expert arena: the workspace once, then parts loaded in
/// order, each with its transient staging on top of what is resident.
#[derive(Debug, Clone, Default)]
struct Arena {
    opened: bool,
    resident: u64,
    peak: u64,
    staging: u64,
}

impl Arena {
    fn open(&mut self, workspace: u64) {
        if !self.opened { self.opened = true; self.resident = workspace; self.peak = workspace; }
    }
    fn add(&mut self, part: Bytes2) -> Result<(), PlacementError> {
        let loading = self.resident.checked_add(part.resident).and_then(|n| n.checked_add(part.staging))
            .ok_or(PlacementError::Overflow("expert arena peak"))?;
        self.peak = self.peak.max(loading);
        self.resident = self.resident.checked_add(part.resident).ok_or(PlacementError::Overflow("expert arena"))?;
        Ok(())
    }
}

/// TP1 movable arenas remain alive beside distinct TP2 backbone arenas. Both
/// peaks are reserved; fixed onboard never searches a whole-layer GPU split.
fn place_tp2(request: &PlacementRequest, available: &[u64], used: &[u64], tp1: &mut [Arena],
    first: usize, limit: usize, cap: usize)
    -> Result<(Vec<ExpertRange>, Vec<ExpertHome>, usize, Option<Tp2Range>), PlacementError> {
    let ranges = tp1.iter().map(|a| ExpertRange { first, layers: 0, peak_bytes: a.peak }).collect();
    let mut halves = [Arena::default(), Arena::default()];
    let mut homes = vec![ExpertHome::Spark; request.layers.len()];
    let (mut next, mut placed) = (first, 0);
    while next < limit && placed < cap {
        let Some(cost) = request.layers[next].experts else { next += 1; continue };
        if !cost.tp2 { break; }
        let mut candidate = halves.clone();
        for gpu in 0..2 {
            candidate[gpu].open(request.tp2_workspace[gpu]);
            candidate[gpu].add(cost.half[gpu])?;
            candidate[gpu].staging = candidate[gpu].staging.max(cost.half[gpu].staging);
            candidate[gpu].peak = candidate[gpu].resident.checked_add(candidate[gpu].staging)
                .ok_or(PlacementError::Overflow("TP2 resident plus largest staging"))?;
        }
        if (0..2).any(|g| candidate[g].peak > available[g].saturating_sub(used[g]).saturating_sub(tp1[g].peak)) { break; }
        halves = candidate;
        homes[next] = ExpertHome::RtxTp2;
        next += 1;
        placed += 1;
    }
    let range = (placed > 0).then_some(Tp2Range { first, layers: next - first,
        peak_bytes: [halves[0].peak, halves[1].peak] });
    for gpu in 0..2 {
        tp1[gpu].peak = tp1[gpu].peak.checked_add(halves[gpu].peak)
            .ok_or(PlacementError::Overflow("TP1 and TP2 arena peaks"))?;
    }
    Ok((ranges, homes, next, range))
}
