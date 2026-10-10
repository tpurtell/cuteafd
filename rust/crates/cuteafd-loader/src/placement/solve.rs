//! Deterministic placement in TJ's order: layer modes, fixed demands,
//! mandatory movables, the KV pool, then contiguous RTX expert ranges.
use super::*;
use cuteafd_core::memory_layout::{Basis, Category, Item};

/// Places `request`. Pure arithmetic over the request: the planner and the
/// runtime get the same answer for the same inputs.
pub fn solve(request: &PlacementRequest) -> Result<Placement, PlacementError> {
    use PlacementError::Overflow;
    let gpus = request.inventory.gpus.len();
    if !(1..=2).contains(&gpus) { return Err(PlacementError::Inventory("one or two coordinator GPUs")); }
    if request.pool.unit_rows == 0 { return Err(PlacementError::Inventory("zero pool unit rows")); }
    if request.pool_overhead.len() != gpus { return Err(PlacementError::Inventory("pool overhead per GPU")); }
    if request.fixed.iter().any(|d| usize::from(d.gpu) >= gpus)
        || request.movables.iter().any(|m| m.allowed.is_empty() || m.allowed.iter().any(|&g| usize::from(g) >= gpus)) {
        return Err(PlacementError::Inventory("demand on an absent GPU"));
    }
    let split = gpus == 2 && request.inventory.peer_access;

    // 1. Layer modes: the policy's first mode this build executes.
    let modes = request.layers.iter().map(|layer| {
        let wanted = request.policy.preference(layer.kind);
        wanted.iter().copied().chain(layer.modes.iter().copied())
            .find(|mode| layer.modes.contains(mode) && executable(*mode, gpus, split))
            .unwrap_or(LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner })
    }).collect::<Vec<_>>();

    // 2. Fixed demands, including mode-dependent layer weights.
    let mut items: Vec<Vec<Item>> = vec![Vec::new(); gpus];
    let mut used = vec![0u64; gpus];
    let charge = |items: &mut Vec<Vec<Item>>, used: &mut Vec<u64>, gpu: usize, item: Item| -> Result<(), PlacementError> {
        used[gpu] = used[gpu].checked_add(item.bytes).ok_or(Overflow("fixed demands"))?;
        items[gpu].push(item);
        Ok(())
    };
    for demand in &request.fixed {
        charge(&mut items, &mut used, usize::from(demand.gpu),
            Item::new(demand.category, demand.group.clone(), "", demand.bytes, demand.basis))?;
    }
    let mut layer_weights = vec![0u64; gpus];
    let mut unit_bytes = request.pool_overhead.clone();
    for (layer, mode) in request.layers.iter().zip(&modes) {
        for gpu in 0..gpus {
            let (weights, kv) = per_gpu(layer, *mode, gpu);
            layer_weights[gpu] = layer_weights[gpu].checked_add(weights).ok_or(Overflow("layer weights"))?;
            unit_bytes[gpu] = unit_bytes[gpu].checked_add(kv).ok_or(Overflow("pool unit"))?;
        }
    }
    for (gpu, &bytes) in layer_weights.iter().enumerate() {
        if bytes > 0 { charge(&mut items, &mut used, gpu, Item::new(Category::Weights, "layers", "", bytes, Basis::Exact))?; }
    }
    let available = request.inventory.gpus.iter().map(GpuBudget::available).collect::<Vec<_>>();

    // 3. Mandatory movables on the allowed GPU with the most room, largest first.
    let mut order: Vec<usize> = (0..request.movables.len()).collect();
    let total = |m: &Movable| m.parts.iter().map(|p| p.resident).sum::<u64>();
    order.sort_by_key(|&i| std::cmp::Reverse(total(&request.movables[i])));
    let mut arenas: Vec<Arena> = (0..gpus).map(|_| Arena::default()).collect();
    let mut movables = Vec::new();
    for index in order {
        let movable = &request.movables[index];
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
            return Err(PlacementError::Mandatory { gpu: gpu as u8, what: describe(&request.fixed, gpu, &movables) });
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
                solve(&trial).is_ok()
            };
            let (mut lo, mut hi) = (0usize, routed);
            if !fits(0) { return Err(match request.pool.requested {
                Some(requested) => PlacementError::PoolDoesNotFit { requested, fit: 0 },
                None => PlacementError::Mandatory { gpu: 0,
                    what: format!("a {floor}-token pool beside {}", describe(&request.fixed, 0, &movables)) },
            }); }
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if fits(mid) { lo = mid } else { hi = mid - 1 }
            }
            Some(lo)
        }
        (fixed, _) => fixed,
    };
    let experts_first = matches!(request.onboard, Onboard::ExpertsFirst { .. });
    let (units, ranges, homes) = match fixed_layers {
        // Pool first: reserve the target, then fill each GPU's arena in order.
        None => {
            let fit = pool_fit(&available, &used, &arenas, &unit_bytes);
            let units = pool_units(request, fit)?;
            let used = with_pool(&used, &unit_bytes, units)?;
            let caps = [usize::MAX, if request.expert_gpus > 1 { usize::MAX } else { 0 }];
            let (ranges, homes, _) = place_experts(request, &available, &used, &mut arenas, first_moe,
                request.layers.len(), &caps)?;
            (units, ranges, homes)
        }
        // Fixed onboard: exactly `n` routed layers (the contiguous prefix the
        // executors run), split over the GPUs where the pool is largest; the
        // pool is the output.
        Some(n) => {
            let end = nth_moe_end(request, first_moe, n);
            let mut best: Option<(u64, Vec<ExpertRange>, Vec<ExpertHome>, Vec<Arena>)> = None;
            let splits: Vec<usize> = if gpus == 1 || request.expert_gpus < 2 { vec![n] } else { (0..=n).rev().collect() };
            let mut most = 0;
            for on_first in splits {
                let mut trial = arenas.clone();
                let caps = [on_first, if gpus == 2 { n - on_first } else { 0 }];
                let (ranges, homes, next) = place_experts(request, &available, &used, &mut trial, first_moe, end, &caps)?;
                let placed = homes.iter().filter(|h| matches!(h, ExpertHome::RtxWhole { .. })).count();
                most = most.max(placed);
                if next < end || placed < n { continue; }
                let fit = pool_fit(&available, &used, &trial, &unit_bytes);
                if best.as_ref().is_none_or(|(units, ..)| fit > *units) { best = Some((fit, ranges, homes, trial)); }
            }
            let Some((fit, ranges, homes, trial)) = best else {
                return Err(if spark_free { PlacementError::SparkFree { layers: routed, placed: most } }
                    else { PlacementError::ExpertLayers { requested: n, placed: most } });
            };
            drop(trial);
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
            (units, ranges, homes)
        }
    };
    if units == 0 {
        return Err(PlacementError::Mandatory { gpu: 0, what: format!("one KV unit beside {}", describe(&request.fixed, 0, &movables)) });
    }
    let onboard_layers = homes.iter().filter(|h| matches!(h, ExpertHome::RtxWhole { .. } | ExpertHome::RtxTp2)).count();

    // Charged items: pool records and expert arenas.
    let pool_tokens = units.checked_mul(request.pool.unit_rows).ok_or(Overflow("pool tokens"))?;
    for gpu in 0..gpus {
        if unit_bytes[gpu] > 0 {
            items[gpu].push(Item::new(Category::Kv, "records", "", unit_bytes[gpu] * units, Basis::Formula));
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
        onboard_layers,
        layers: modes.into_iter().zip(homes).map(|(mode, experts)| LayerAssignment { mode, experts }).collect(),
        movables,
        expert_ranges: ranges,
        items,
    })
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
    first_moe: usize, limit: usize, caps: &[usize]) -> Result<(Vec<ExpertRange>, Vec<ExpertHome>, usize), PlacementError> {
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
    Ok((ranges, homes, next))
}

fn executable(mode: LayerMode, gpus: usize, split: bool) -> bool {
    match mode {
        LayerMode::HeadSplit => split,
        LayerMode::Whole { gpu, .. } => usize::from(gpu) < gpus,
    }
}

/// A layer's weights and pool-unit bytes on `gpu` under `mode`.
fn per_gpu(layer: &LayerDemand, mode: LayerMode, gpu: usize) -> (u64, u64) {
    match mode {
        LayerMode::HeadSplit => (layer.weights.split[gpu], layer.kv_unit.split[gpu]),
        LayerMode::Whole { gpu: owner, .. } if usize::from(owner) == gpu => (layer.weights.whole, layer.kv_unit.whole),
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
