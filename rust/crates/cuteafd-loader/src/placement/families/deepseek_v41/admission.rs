//! V4.1 physical component composition and the shared plan/serve assembler.
//!
//! Offline `cuteafd plan --layout` and live serving admission prepare these
//! facts once and admit a single typed request. Loading provenance never
//! changes an owner's charge or its activation; the executor consumes the
//! admitted owners and never samples free memory again.
use super::{DraftInventory, ExpertCost};
use crate::placement::{AttentionPlacement, Demand, FfnMode, ItemLifetime, LayerMode, MovableId,
    ResidentWeights, planner::*};
use cuteafd_core::memory_layout::{Basis, Category, MemoryKind};
use cuteafd_core::spark_reply::SparkReplyFormats;
use std::collections::{BTreeMap, BTreeSet};

/// A condition names a domain value, never a solver variable index. Conditions
/// in one case are conjunctive; membership in a count interval is disjunctive.
#[derive(Debug, Clone)]
pub enum When {
    Prefill(PrefillShape),
    Lanes(u64),
    Onboard { first: usize, last: usize },
}

#[derive(Debug, Clone)]
pub struct Case<T> {
    pub when: Vec<When>,
    pub value: T,
}

/// One physical owner as the loader/executor allocates it. `bytes` is the
/// allocation charge, `resident` the retained-weight ledger; a deferred owner
/// is reserved after the target KV and never touches the load phases.
#[derive(Debug, Clone)]
pub struct Owner {
    pub location: Location,
    pub category: Category,
    pub group: String,
    pub bytes: u64,
    pub resident: u64,
    pub deferred: bool,
    /// Already inside the availability sample: a sequential load peak stated
    /// above that sample excludes it, so it is not subtracted from the peak.
    pub pre_sample: bool,
    pub lifetime: Lifetime,
    /// An immutable weight arena: `bytes` and `resident` are its physical
    /// extent B, charged once with its single SM120 tail (no per-tensor payload).
    pub arena: Option<(crate::serving_capacity::WeightArenaLayout, MemoryKind)>,
}

impl Owner {
    /// Retained weights: the allocation and its resident-weight ledger agree.
    pub fn weights(gpu: u8, category: Category, group: impl Into<String>, bytes: u64) -> Self {
        Self { resident: bytes, ..Self::charged(gpu, category, group, bytes) }
    }

    /// One preplanned weight arena on `gpu`, live from Load for the whole run.
    pub fn arena(gpu: u8, layout: crate::serving_capacity::WeightArenaLayout, kind: MemoryKind) -> Self {
        let bytes = layout.bytes() as u64;
        Self { arena: Some((layout, kind)), ..Self::weights(gpu, kind.category(), "weight arena", bytes) }
    }

    /// A transient allocation that is not retained checkpoint weights.
    pub fn charged(gpu: u8, category: Category, group: impl Into<String>, bytes: u64) -> Self {
        Self {
            location: Location::coordinator(gpu),
            category,
            group: group.into(),
            bytes,
            resident: 0,
            deferred: false,
            pre_sample: false,
            lifetime: Lifetime::RESIDENT,
            arena: None,
        }
    }

    /// A component of the runtime base the availability sample already holds
    /// (CUDA context, module images).
    pub fn pre_sample(mut self) -> Self {
        self.pre_sample = true;
        self
    }

    /// An owner already resident in the serving sample: ledger only.
    pub fn loaded(gpu: u8, category: Category, group: impl Into<String>, resident: u64) -> Self {
        Self { resident, ..Self::charged(gpu, category, group, 0) }
    }

    pub fn host(category: Category, group: impl Into<String>, bytes: u64) -> Self {
        Self { location: Location::HOST, ..Self::charged(0, category, group, bytes) }
    }

    pub fn at(mut self, location: Location) -> Self {
        self.location = location;
        self
    }

    pub fn with_lifetime(mut self, lifetime: Lifetime) -> Self {
        self.lifetime = lifetime;
        self
    }

    /// A reserve for bytes a later stage allocates; no phase before ready pays it.
    pub fn deferred(mut self) -> Self {
        self.deferred = true;
        self.lifetime = Lifetime { first: Phase::Ready, last: Phase::Growth };
        self
    }

    /// The legacy shared ledger's `Demand` as an exact owner. A post-ready
    /// reserve is a deferred owner of the same physical bytes.
    pub fn from_demand(demand: &Demand) -> Self {
        Self {
            location: Location::coordinator(demand.gpu),
            category: demand.category,
            group: demand.group.clone(),
            bytes: demand.bytes,
            resident: demand.resident_weights.0,
            deferred: demand.lifetime == ItemLifetime::PostReadyReserve,
            pre_sample: false,
            lifetime: match demand.lifetime {
                ItemLifetime::ReadyResident => Lifetime::RESIDENT,
                ItemLifetime::PostReadyReserve => Lifetime { first: Phase::Ready, last: Phase::Growth },
            },
            arena: None,
        }
    }
}

/// An opaque declared reserve: bytes this engine budgets for a documented
/// envelope it cannot name as one allocator receipt. Never a second ledger.
#[derive(Debug, Clone)]
pub struct Opaque {
    pub location: Location,
    pub bytes: u64,
    pub basis: Basis,
    pub lifetime: Lifetime,
    pub source: String,
}

/// The bounded graph arena, one owner per coordinator GPU. Admission charges
/// the whole configured reserve once before KV sizing; the daemon's device
/// cache reports its measured high-water against it. `GPU_GRAPH_RESERVE`
/// overrides it per device and never splits across ranks. Nothing here is live
/// at Ready, so it prices only the post-ready phases.
pub fn graph_allowance(rank: usize) -> Result<Opaque, Refusal> {
    let (base, extra) = crate::placement::graph_reserve::v41_default();
    let resolved = crate::placement::graph_reserve::current(rank, false, base, extra)
        .map_err(Refusal::InvalidFact)?;
    Ok(Opaque {
        location: Location::coordinator(rank as u8),
        bytes: resolved.bytes,
        // A configured budget, not a census estimate: it is never "assumed".
        basis: Basis::Calibrated,
        lifetime: Lifetime { first: Phase::Ready, last: Phase::Growth },
        source: crate::placement::graph_reserve::source(&format!("v41.graphs.{rank}")),
    })
}

/// The runtime base the pre-weight availability sample already holds, named
/// component by component instead of taken as the leftover of a free sample:
/// the CUDA context on an otherwise empty device and the images of the
/// family's selected native modules.
///
/// cuBLAS handles, BLAS device initialization and driver page tables are
/// already named owners of their own (`planner::serving::blas_components` and
/// the plan's page-table mapping), so they are not repeated here. The preload
/// window loads the selected program set; program storage that a later first
/// use resolves (per-shape warm-up) is outside the prediction, reported by
/// `runtime_base_check` at Ready and covered after Ready by the runtime growth
/// reserve.
pub const CONTEXT_OWNER: &str = "CUDA context";
pub const MODULE_OWNER: &str = "module images";
pub const RUNTIME_OWNERS: [&str; 2] = [CONTEXT_OWNER, MODULE_OWNER];

/// The two predicted runtime-base owners of one coordinator GPU. The context
/// figure is the calibrated `ArchContext` probe; the module figure is the
/// measured free-memory drop across the pre-weight preload window
/// (`RuntimeInventory::measure`, `InventoryPhase::PreWeight`) — the modules the
/// launch actually resolved on that device, never a program-count estimate and
/// never a free-memory remainder. The offline layout's generic estimate
/// (`ProgramSet::module_bytes`, calibrated on MiMo) prices `plan --layout` only;
/// on V4.1 it understates the measured set, so serve charges the measurement.
pub fn runtime_base(gpu: u8, arch: &crate::placement::ArchContext, module_bytes: u64) -> Vec<Owner> {
    [(CONTEXT_OWNER, arch.context_bytes), (MODULE_OWNER, module_bytes)]
        .into_iter()
        .filter(|&(_, bytes)| bytes > 0)
        .map(|(group, bytes)| Owner::charged(gpu, Category::Runtime, group, bytes).pre_sample())
        .collect()
}

/// Whether a whole-GPU reserve (`GPU_RESERVE`) is in force. It replaces the
/// family's named runtime owners in `apply_reserve_override`, so the predicted
/// runtime base cannot be charged beside the budget in that mode: the measured
/// pre-weight sample stays the budget's sampled baseline instead.
pub fn whole_gpu_reserve() -> bool {
    cuteafd_core::memory_policy::MemoryPolicy::current()
        .reserve_override(cuteafd_core::memory_policy::DeviceKind::Gpu)
        .is_some()
}

/// The predicted runtime base and the device's measured occupancy at Ready:
/// `(predicted, measured)`. `measured` is what the device holds once every
/// other Ready-resident owner (weights, workspaces, KV, transports, BLAS
/// handles, page tables and arena tails) is priced, so
/// `measured - predicted` is the prediction-only residual this plan carries.
/// Post-Ready reserves are not occupancy and are not subtracted.
pub fn runtime_base_check(facts: &Facts, gpu: u8, used: u64)
    -> Result<(u64, u64), Refusal> {
    let location = Location::coordinator(gpu);
    let (mut predicted, mut others) = (0u64, 0u64);
    for allocation in &facts.allocations {
        if allocation.location != location || allocation.view.is_some()
            || !allocation.lifetime.contains(Phase::Ready, facts.phase_order) {
            continue;
        }
        let bucket = if RUNTIME_OWNERS.contains(&allocation.group.as_str()) { &mut predicted } else { &mut others };
        *bucket = bucket.checked_add(allocation.bytes).ok_or(Refusal::Overflow("V4.1 runtime base check"))?;
    }
    for mapping in facts.driver_page_tables.iter().filter(|mapping| mapping.location == location) {
        others = others.checked_add(mapping.bytes()).ok_or(Refusal::Overflow("V4.1 page-table check"))?;
    }
    for owner in facts.allocation_granularity.iter().filter(|owner| owner.location == location) {
        others = others.checked_add(owner.bytes()?).ok_or(Refusal::Overflow("V4.1 arena-tail check"))?;
    }
    Ok((predicted, used.saturating_sub(others)))
}

/// The Ready-resident device owners no allocator ledger row can see: the two
/// predicted runtime-base owners, the SM120 cuBLAS/BLAS library owners that
/// `serving::blas_occupancy` charges as occupancy, and the driver's page
/// tables and arena tails. `runtime_base_check` prices exactly this set as
/// `others`, so a live measurement of the device subtracts the same bytes
/// before it calls the rest unattributed (#1634).
///
/// KV is excluded: the pre-pool measurement runs before the cache exists, so
/// its owners are planned but not yet on the device. Every other Ready-active
/// owner has been created by then (the predicted base before the weights, the
/// loading owners as they load).
pub fn untracked_ready_owners(facts: &Facts, gpu: u8) -> Result<u64, Refusal> {
    let location = Location::coordinator(gpu);
    let mut total = 0u64;
    for allocation in &facts.allocations {
        if allocation.location != location || allocation.view.is_some()
            || allocation.category == Category::Kv
            || !allocation.lifetime.contains(Phase::Ready, facts.phase_order)
            || !(RUNTIME_OWNERS.contains(&allocation.group.as_str())
                || allocation.group.starts_with(serving::CUBLAS_HANDLE_OWNER)
                || allocation.group.starts_with(serving::BLAS_INITIALIZATION_OWNER))
        {
            continue;
        }
        total = total.checked_add(allocation.bytes).ok_or(Refusal::Overflow("V4.1 untracked owners"))?;
    }
    for mapping in facts.driver_page_tables.iter().filter(|mapping| mapping.location == location) {
        total = total.checked_add(mapping.bytes()).ok_or(Refusal::Overflow("V4.1 untracked page tables"))?;
    }
    for owner in facts.allocation_granularity.iter().filter(|owner| owner.location == location) {
        total = total.checked_add(owner.bytes()?).ok_or(Refusal::Overflow("V4.1 untracked arena tail"))?;
    }
    Ok(total)
}

/// The charged components of one GPU's predicted runtime base, named: the
/// calibrated context and the measured module extent, exactly the allocations
/// `runtime_base_check` sums into `predicted`. A Ready report prints these
/// beside the measured occupancy, so a residual is always read against what
/// was actually predicted.
pub fn runtime_base_components(facts: &Facts, gpu: u8) -> RuntimeBaseComponents {
    let location = Location::coordinator(gpu);
    let mut components = RuntimeBaseComponents::default();
    for allocation in &facts.allocations {
        if allocation.location != location || allocation.view.is_some()
            || !allocation.lifetime.contains(Phase::Ready, facts.phase_order) {
            continue;
        }
        let slot = match allocation.group.as_str() {
            CONTEXT_OWNER => &mut components.context_bytes,
            MODULE_OWNER => &mut components.module_bytes,
            _ => continue,
        };
        *slot = slot.saturating_add(allocation.bytes);
    }
    components
}

/// One GPU's predicted runtime base, component by component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeBaseComponents {
    pub context_bytes: u64,
    pub module_bytes: u64,
}

impl RuntimeBaseComponents {
    pub fn total(&self) -> u64 { self.context_bytes.saturating_add(self.module_bytes) }
}

fn byte_component(template: Allocation, curve: ByteCurve, maximum: u64) -> Result<Component, Refusal> {
    let mut component = Component::allocation(template, curve.clone(), false, maximum);
    component.byte_extent(curve)?;
    Ok(component)
}

fn owner_components(prefix: &str, owners: &[Owner], maximum: u64) -> Result<Vec<Component>, Refusal> {
    let mut out = Vec::new();
    for (index, owner) in owners.iter().enumerate() {
        if let Some((layout, kind)) = &owner.arena {
            // Root B, its resident claim and one separate tail; the shared
            // helper replaces any payload, padding or per-tensor forecast.
            if owner.bytes != layout.bytes() as u64 || owner.resident != owner.bytes || owner.deferred {
                return Err(invalid("weight arena owner differs from its layout"));
            }
            // The resident-weight budget keeps counting logical payload P, not
            // the physical extent; the root and tail carry the physical charge.
            let mut components = weight_arena_components(layout, owner.location, owner.lifetime, maximum, *kind)?;
            components.retain(|component| !matches!(component.kind, ComponentKind::ResidentWeight(_)));
            let payload = u64::try_from(layout.payload_bytes()).map_err(|_| Refusal::Overflow("V4.1 arena payload"))?;
            let root = Allocation::buffer(layout.id(), owner.location, owner.bytes, owner.lifetime,
                kind.category(), "weight arena");
            out.extend(components);
            out.push(Component::resident_weight(ResidentWeight::from_allocation(&root, ResidentWeights(payload)),
                ByteCurve::fixed(payload, maximum)));
            continue;
        }
        let id = format!("v41.{prefix}.{index}");
        if owner.bytes > 0 {
            let allocation = Allocation::buffer(
                id.clone(),
                owner.location,
                owner.bytes,
                owner.lifetime,
                owner.category,
                owner.group.clone(),
            );
            let component = Component::allocation(
                allocation,
                ByteCurve::fixed(owner.bytes, maximum),
                owner.deferred,
                maximum,
            );
            out.push(component);
        }
        if owner.resident > 0 {
            let backing = Allocation::buffer(
                id,
                owner.location,
                owner.resident,
                owner.lifetime,
                owner.category,
                owner.group.clone(),
            );
            out.push(Component::resident_weight(
                ResidentWeight::from_allocation(&backing, ResidentWeights(owner.resident)),
                ByteCurve::fixed(owner.resident, maximum),
            ));
        }
    }
    Ok(out)
}

fn opaque_components(opaques: &[Opaque], maximum: u64) -> Vec<Component> {
    opaques
        .iter()
        .filter(|opaque| opaque.bytes > 0)
        .map(|opaque| Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(opaque.bytes, maximum),
            kind: ComponentKind::Allowance(Allowance {
                location: opaque.location,
                bytes: opaque.bytes,
                lifetime: opaque.lifetime,
                source: opaque.source.clone(),
                assumed: matches!(opaque.basis, Basis::Estimated),
            }),
        })
        .collect()
}

/// Expert arena residency for one ordered prefix: the prefill slots' shared
/// workspace, each loaded layer's resident planes and the largest staging.
/// TP2 names the same layers on both GPUs but counts each routed layer once.
pub fn arena(
    count: usize,
    gpus: usize,
    experts: &[ExpertCost],
    workspace: u64,
) -> Result<([u64; 2], [u64; 2]), Refusal> {
    let mut peak = [0u64; 2];
    let mut resident = [0u64; 2];
    if count == 0 {
        return Ok((peak, resident));
    }
    if count > experts.len() || !(1..=2).contains(&gpus) {
        return Err(invalid("expert prefix exceeds the routed inventory"));
    }
    if gpus == 1 {
        let mut used = workspace;
        let mut staging = 0;
        for cost in &experts[..count] {
            used = used
                .checked_add(cost.whole.resident)
                .ok_or(Refusal::Overflow("V4.1 expert arena"))?;
            staging = staging.max(cost.whole.staging);
            peak[0] = peak[0].max(
                used.checked_add(staging)
                    .ok_or(Refusal::Overflow("V4.1 expert arena peak"))?,
            );
            resident[0] = used - workspace;
        }
    } else {
        for gpu in 0..2 {
            let mut used = workspace;
            let mut staging = 0;
            for cost in &experts[..count] {
                used = used
                    .checked_add(cost.half[gpu].resident)
                    .ok_or(Refusal::Overflow("V4.1 TP2 expert arena"))?;
                staging = staging.max(cost.half[gpu].staging);
                resident[gpu] = used - workspace;
            }
            peak[gpu] = used
                .checked_add(staging)
                .ok_or(Refusal::Overflow("V4.1 TP2 expert arena peak"))?;
        }
    }
    Ok((peak, resident))
}

/// Exact routed-expert cache ownership: private COW tails, one record unit per
/// pool page, source tables bounded by the compiled 4096-row extent, and the
/// fixed window/header/carry extent. Tails are charged once, so the allocator's
/// `groups = pages + tails` total is reproduced exactly.
pub fn cache_owners(
    config: &serde_json::Value,
    ranks: usize,
    slots: u64,
    tails: u64,
    replicated: bool,
    maximum: u64,
) -> Result<Vec<Vec<Component>>, Refusal> {
    let geometry = crate::serving_capacity::deepseek_v41_cache_geometry(config, ranks)
        .map_err(|error| invalid(&error.to_string()))?;
    if tails > 4096 || (replicated && ranks != 2) {
        return Err(invalid("invalid V4.1 cache admission shape"));
    }
    let mut units = geometry
        .ranks
        .iter()
        .map(|rank| rank.persistent_unit_bytes)
        .collect::<Vec<_>>();
    if replicated {
        units[0] += 2 * 256 * 288;
        units[1] += 3 * 256 * 288;
    }
    // At one group each source table is still linear; subtracting its entries
    // from the allocator total isolates fixed windows and compressor carry.
    let first = crate::serving_capacity::deepseek_v41_cache_bytes(config, ranks, slots, 1, replicated)
        .map_err(|error| invalid(&error.to_string()))?;
    let mut out = vec![Vec::new(); ranks];
    let sources = if ranks == 1 { vec![(3u64, 1u64)] } else { vec![(3, 0), (0, 1)] };
    for (gpu, &(c2, c1)) in sources.iter().enumerate() {
        let replica = if replicated {
            if gpu == 0 { (0, 1) } else { (3, 0) }
        } else {
            (0, 0)
        };
        let mut entries = 0u64;
        for (count, ratio) in [(c2 + replica.0, 1u64), (c1 + replica.1, 2)] {
            if count == 0 {
                continue;
            }
            let per_unit = count
                .checked_mul(slots)
                .and_then(|n| n.checked_mul(ratio))
                .and_then(|n| n.checked_mul(4))
                .ok_or(Refusal::Overflow("V4.1 source page table"))?;
            entries = entries
                .checked_add(per_unit)
                .ok_or(Refusal::Overflow("V4.1 source page table"))?;
            let cap = (4096 / ratio).min(maximum.saturating_add(tails));
            let owner = Allocation::buffer(
                format!("v41.kv.tables.{gpu}.{count}x{ratio}"),
                Location::coordinator(gpu as u8),
                per_unit,
                Lifetime::TARGET,
                Category::Kv,
                "source page tables",
            );
            let curve = bounded_curve(per_unit, tails, cap.max(1), maximum)?;
            out[gpu].push(byte_component(owner, curve, maximum)?);
        }
        let fixed = first[gpu]
            .checked_sub(units[gpu])
            .and_then(|n| n.checked_sub(entries))
            .ok_or_else(|| invalid("V4.1 fixed cache extent underflow"))?;
        let records = Allocation::buffer(
            format!("v41.kv.records.{gpu}"),
            Location::coordinator(gpu as u8),
            units[gpu],
            Lifetime::TARGET,
            Category::Kv,
            "records",
        );
        out[gpu].push(byte_component(
            records,
            ByteCurve::linear(units[gpu], units[gpu], maximum),
            maximum,
        )?);
        let fixed_owner = Allocation::buffer(
            format!("v41.kv.fixed.{gpu}"),
            Location::coordinator(gpu as u8),
            fixed,
            Lifetime::TARGET,
            Category::Kv,
            "window state, table headers and compressor carry",
        );
        out[gpu].push(byte_component(
            fixed_owner,
            ByteCurve::fixed(fixed, maximum),
            maximum,
        )?);
        let tail = units[gpu]
            .checked_mul(tails)
            .ok_or(Refusal::Overflow("V4.1 tail extent"))?;
        if tail > 0 {
            let owner = Allocation::buffer(
                format!("v41.kv.tails.{gpu}"),
                Location::coordinator(gpu as u8),
                tail,
                Lifetime::TARGET,
                Category::Kv,
                "active and retained COW tails",
            );
            out[gpu].push(byte_component(
                owner,
                ByteCurve::fixed(tail, maximum),
                maximum,
            )?);
        }
    }
    Ok(out)
}

/// One coordinator GPU's measured pre-pool device state that no allocator
/// ledger row and no named owner prices (#1634).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasuredOccupancy {
    pub gpu: u8,
    /// The driver's 2 MiB backing beyond the live requests' own bytes
    /// (`memory_ledger::Granularity::tail`).
    pub backing: u64,
    /// Device bytes beyond both the ledger and every named owner.
    pub unattributed: u64,
}

/// The pool after adopting a measurement: its page count, the allocator's
/// group total (`pages + tails`) and each GPU's exact KV bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdoptedPool {
    pub pages: u64,
    pub groups: u64,
    pub bytes: Vec<u64>,
}

/// The two owners a measurement is charged as. Names, not ids: the Ready
/// report and the plan ledger print them, so the bytes are never opaque.
pub const BACKING_OWNER: &str = "SM120 allocation backing";
pub const UNATTRIBUTED_OWNER: &str = "unattributed runtime, measured";

/// Adopt the pre-pool device measurement into a solved plan (#1634).
///
/// The solve sizes the KV pool from a sample taken before the family loads its
/// weights, so the driver backing that weight, workspace and exchange
/// allocations add, and the device state no owner names, are not in it. Both
/// are measurable between the solve and the pool's creation: the ledger counts
/// every request it made and its own rounding, the plan counts every owner it
/// named, and the device holds the rest. This charges them as Ready-activated
/// occupancy owners — the same activation the memory is live on — and then
/// re-fits the pool to the largest page count whose exact facts still fit
/// every budget, using the same `evaluate` the solver used. A boot with room
/// to spare keeps its pool; a budget-bound one gives up KV instead of
/// discovering the bytes at Ready.
pub fn adopt_measured(
    config: &serde_json::Value,
    gpus: usize,
    slots: u64,
    retained_turns: u64,
    replicated: bool,
    maximum_pool_tokens: u64,
    plan: &mut ResolvedPlan,
    measured: &[MeasuredOccupancy],
) -> Result<AdoptedPool, Refusal> {
    let tails = slots
        .checked_add(retained_turns.checked_mul(2).ok_or(Refusal::Overflow("V4.1 retained turns"))?)
        .ok_or(Refusal::Overflow("V4.1 private tails"))?;
    let maximum = maximum_pool_tokens
        .div_ceil(plan.pool_page_rows)
        .min(131_072u64.saturating_sub(tails));
    // Zero is a measurement too: an owner with no bytes is not a charge.
    for occupant in measured {
        let location = Location::coordinator(occupant.gpu);
        for (index, (group, bytes)) in [(BACKING_OWNER, occupant.backing),
            (UNATTRIBUTED_OWNER, occupant.unattributed)].into_iter().enumerate() {
            if bytes == 0 { continue; }
            plan.facts.allocations.push(Allocation::buffer(
                format!("v41.measured.{}.{index}", occupant.gpu),
                location, bytes, Lifetime { first: Phase::Ready, last: Phase::Growth },
                Category::Runtime, group));
        }
    }
    let owners = cache_owners(config, gpus, slots, tails, replicated, maximum)?;
    refit_pool(plan, &owners, tails)
}

/// Charge `owners` (one KV component list per GPU) in place of the solved
/// pool's facts and return the largest page count that still fits every
/// budget. Only the KV components depend on the page count, so the rest of the
/// plan is re-priced untouched.
fn refit_pool(plan: &mut ResolvedPlan, owners: &[Vec<Component>], tails: u64) -> Result<AdoptedPool, Refusal> {
    let page_rows = plan.pool_page_rows;
    let solved = plan.pool_tokens.div_ceil(page_rows);
    let floor = if matches!(plan.request.pool, Pin::Exact(_)) {
        solved
    } else {
        plan.request.context.required().div_ceil(page_rows).max(1)
    }.min(solved);
    let (mut low, mut high) = (floor, solved);
    let mut best: Option<AdoptedPool> = None;
    while low <= high {
        let pages = low + (high - low) / 2;
        let bytes = install_pool_facts(plan, owners, pages)?;
        match evaluate(&plan.facts, &plan.budgets) {
            Ok(_) => {
                best = Some(AdoptedPool { pages, groups: pages.saturating_add(tails), bytes });
                low = pages.saturating_add(1);
            }
            Err(Refusal::Capacity { .. }) if pages > 0 => high = pages - 1,
            Err(_) => break,
        }
    }
    let adopted = match best {
        Some(best) => best,
        None => {
            // No probe fit. The floor is the smallest legal pool: adopt it
            // when it fits (a degenerate solved pool can leave the search
            // empty) and otherwise report why it does not.
            let bytes = install_pool_facts(plan, owners, floor)?;
            evaluate(&plan.facts, &plan.budgets)?;
            AdoptedPool { pages: floor, groups: floor.saturating_add(tails), bytes }
        }
    };
    // The search's last probe may be the rejected one just above `best`; the
    // plan's facts must end at the adopted page count, not at that probe.
    let bytes = install_pool_facts(plan, owners, adopted.pages)?;
    debug_assert_eq!(bytes, adopted.bytes);
    plan.pool_tokens = adopted.pages.checked_mul(page_rows).ok_or(Refusal::Overflow("pool tokens"))?;
    plan.phases = evaluate(&plan.facts, &plan.budgets)?;
    Ok(adopted)
}

/// Replace every coordinator KV owner with the component list at `pages` and
/// return each GPU's exact total. The components are the ones `prepare` priced
/// the solve with, so the pool's facts and the runtime's allocator agree.
fn install_pool_facts(plan: &mut ResolvedPlan, owners: &[Vec<Component>], pages: u64)
    -> Result<Vec<u64>, Refusal> {
    let mut totals = Vec::with_capacity(owners.len());
    for (gpu, components) in owners.iter().enumerate() {
        let location = Location::coordinator(gpu as u8);
        plan.facts.allocations.retain(|owner| !(owner.location == location && owner.category == Category::Kv));
        let mut total = 0u64;
        for component in components {
            let ComponentKind::Allocation { template, shape, offset, deferred } = &component.kind else {
                return Err(Refusal::InvalidFact("V4.1 KV component is not an allocation".into()));
            };
            if template.location != location || template.category != Category::Kv
                || template.lifetime != Lifetime::TARGET || *deferred {
                return Err(Refusal::InvalidFact(format!(
                    "V4.1 KV owner {} differs from its cache geometry", template.id)));
            }
            let mut allocation = template.clone();
            allocation.bytes = component.bytes.at(pages)?;
            allocation.shape = shape.iter().map(|curve| curve.at(pages)).collect::<Result<_, _>>()?;
            if let (Some(view), Some(offset)) = (&mut allocation.view, offset) {
                view.offset = offset.at(pages)?;
            }
            total = total.checked_add(allocation.bytes).ok_or(Refusal::Overflow("V4.1 KV pool"))?;
            plan.facts.allocations.push(allocation);
        }
        totals.push(total);
    }
    Ok(totals)
}

/// `min(pages + tails, cap) * per_page`, contiguous over the whole pool.
fn bounded_curve(per_page: u64, tails: u64, cap: u64, maximum: u64) -> Result<ByteCurve, Refusal> {
    let linear = |pages: u64| -> Result<u64, Refusal> {
        pages
            .checked_add(tails)
            .unwrap_or(u64::MAX)
            .min(cap)
            .checked_mul(per_page)
            .ok_or(Refusal::Overflow("V4.1 bounded table"))
    };
    let knee = cap.saturating_sub(tails);
    if knee == 0 {
        return Ok(ByteCurve(vec![BytePiece { granularity: 1,
            first: 1,
            last: maximum,
            bytes_at_first: linear(1)?,
            bytes_per_page: 0, backings: 1,
        }]));
    }
    if knee >= maximum {
        return Ok(ByteCurve::linear(linear(1)?, per_page, maximum));
    }
    Ok(ByteCurve(vec![
        BytePiece { granularity: 1, first: 1, last: knee, bytes_at_first: linear(1)?, bytes_per_page: per_page, backings: 1, },
        BytePiece { granularity: 1,
            first: knee + 1,
            last: maximum,
            bytes_at_first: cap
                .checked_mul(per_page)
                .ok_or(Refusal::Overflow("V4.1 bounded table"))?,
            bytes_per_page: 0, backings: 1,
        },
    ]))
}

fn invalid(detail: &str) -> Refusal {
    Refusal::InvalidFact(format!("V4.1 component contract: {detail}"))
}

/// Header-derived host owners shared by offline planning and the live executor.
#[derive(Debug, Clone)]
pub struct HostFacts {
    pub owners: Vec<Owner>,
    pub estimates: Vec<Opaque>,
    pub prefix_bytes: u64,
    pub mapped: crate::formats::mapped_preload::MappedPlacement,
    pub push: crate::push_load::PushHostEnvelope,
}

impl HostFacts {
    pub fn mapped_placement(headers: &[&crate::SafetensorsTensorMetadata],
        shape: crate::EngramStagingShape, host_budget: u64, preload_cap: u64)
        -> anyhow::Result<crate::formats::mapped_preload::MappedPlacement> {
        use anyhow::{Context, ensure};
        let table_headers = headers.iter().copied().filter(|h| h.name.contains(".engram.embed.")
            && h.shape.len() == 2).collect::<Vec<_>>();
        let first = table_headers.first().context("V4.1 host contract requires Engram headers")?;
        let stem = first.name.split_once(".engram.embed.").unwrap().0;
        let row_bytes = if first.name.contains(".engram.embed.strided_") { first.shape[1] } else {
            table_headers.iter().filter(|h| h.name.starts_with(&format!("{stem}.engram.embed.")))
                .map(|h| h.shape[1]).sum::<usize>()
        };
        ensure!(row_bytes > 0, "V4.1 Engram row width is zero");
        let scales = table_headers.iter().filter(|h| h.name.ends_with(".scale")).count();
        let scale_bytes = table_headers.iter().filter(|h| h.name.ends_with(".scale")).map(|h| h.byte_length).sum();
        let table_bytes = table_headers.iter().map(|h| h.byte_length).sum();
        Ok(crate::formats::mapped_preload::MappedPlacement::resolve(table_headers.len(), scales,
            scale_bytes, preload_cap, shape.preload_budget(host_budget, row_bytes))
            .admit_tables(table_bytes))
    }

    pub fn prepare(config: &serde_json::Value, headers: &[&crate::SafetensorsTensorMetadata],
        prefill: PrefillShape, gpus: usize, sparks: usize, context: u64, concurrency: u64,
        draft: bool, prefix: Option<u64>, prefix_enabled: bool, chunk: u64,
        host: (u64, u64), preload_cap: u64, preload_budget: Option<u64>, full_logits: bool) -> anyhow::Result<Self> {
        use anyhow::{Context, ensure};
        let shape = crate::EngramStagingShape::new(prefill.rows as usize, prefill.waves as usize)?;
        let mapped = Self::mapped_placement(headers, shape, preload_budget.unwrap_or_else(|| host.1.saturating_sub(host.0 / 10)), preload_cap)?;
        let row_bytes = headers.iter().find(|h| h.name.contains(".engram.embed."))
            .map(|h| if h.name.contains("strided_") { h.shape[1] } else {
                headers.iter().filter(|other| other.name.split_once(".engram.embed.").map(|v| v.0)
                    == h.name.split_once(".engram.embed.").map(|v| v.0) && other.shape.len() == 2)
                    .map(|other| other.shape[1]).sum()
            }).context("V4.1 host contract requires Engram headers")?;
        let cache = crate::serving_capacity::deepseek_v41_cache_geometry(config, gpus)?;
        let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: cache.logical_unit_rows,
            page_bytes: cache.ranks.iter().map(|r| r.persistent_unit_bytes).sum(),
            mark_bytes: cache.ranks.iter().map(|r| r.retained_mark_bytes).sum::<u64>()
                + if draft { 3 * 128 * 528 } else { 0 }, draft_bytes: 0 };
        ensure!(chunk > 0, "V4.1 host cache chunk must be positive");
        let prefix_bytes = if prefix_enabled { geometry.resolve(prefix, context, concurrency, host.0, chunk)
            .context("invalid V4.1 host prefix quota")? } else { 0 };
        let mut owners = Vec::new();
        let mut add = |name: &str, bytes: u64, pinned: bool, category| {
            if bytes > 0 { owners.push(Owner::host(category, name, bytes)
                .at(if pinned { Location::PINNED } else { Location::HOST })); }
        };
        add("v41/engram-prefill", shape.host_resident_bytes() as u64, true, Category::Staging);
        add("v41/engram-decode", shape.decode_host_resident_bytes() as u64, true, Category::Staging);
        add("v41/engram-prefill-worker", shape.host_workspace_bytes() as u64, false, Category::Workspace);
        add("v41/engram-decode-worker", shape.decode_host_workspace_bytes() as u64, false, Category::Workspace);
        add("v41/engram-row-cache", shape.cache_bytes(row_bytes) as u64, false, Category::Workspace);
        add("mapped-tables/preloaded (coordinator local)", mapped.preloaded_bytes, false, Category::Tables);
        add("workspace/table-read-reactor", crate::formats::mapped_table::table_reactor_workspace_bytes(mapped.mapped) as u64,
            false, Category::Workspace);
        add("host prefix cache", prefix_bytes / chunk.max(1) * chunk, true, Category::Prefix);
        let rows = prefill.rows * prefill.waves;
        let vocab = config.get("text_config").unwrap_or(config)["vocab_size"].as_u64().context("missing V4.1 vocab")?;
        add("scoring/full-logits", cuteafd_core::memory_layout::host_logits_bytes(rows.max(256), rows, vocab)
            .context("host logits overflow")?, false, Category::Workspace);
        // Estimated upper bounds from the live wire/row metadata allocators, not a measured census.
        let mut estimates = Vec::new();
        let mut estimate = |name: &str, bytes: u64, pinned: bool| {
            estimates.push(Opaque { location: if pinned { Location::PINNED } else { Location::HOST }, bytes,
                basis: Basis::Estimated, lifetime: Lifetime::RESIDENT, source: format!("{name} (estimated allocator upper bound)") });
        };
        let geometry = cuteafd_core::ExpertGeometry::DEEPSEEK_V41;
        if sparks > 0 {
            let rings = cuteafd_core::SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
            let mut ring_bytes = 0;
            for capacity in [prefill.rows.min(256), prefill.rows.min(256), prefill.rows] {
                let endpoint = cuteafd_core::SparkEndpoint { ring_depth: None, capacity: capacity as u32,
                    ingress: cuteafd_core::SparkIngress::Fp8K32, partial_reply_bytes: None };
                ring_bytes += endpoint.ring_bytes(geometry, rings).context("host transport overflow")? as u64 * sparks as u64;
            }
            add("transport/rdma-rings", ring_bytes, true, Category::Transport);
            let request = geometry.compact_wire_bytes(prefill.rows as u32, false).context("host egress overflow")?.0 as u64;
            add("transport/egress", request * (prefill.waves + 2), true, Category::Transport);
            estimate("spark-intake", rows * 5120 * 4 * sparks as u64 * 3, true);
        }
        estimate("workspace/router", (prefill.rows + 64) * (5120 * 2 + 8 * 8), false);
        estimate("staging/tables-and-sync", (prefill.rows + 64) * 5120 * 8 + (2 << 20), true);
        // Estimated ring-frontier bound: one maximum dense step per receive
        // slot, not twice the checkpoint. prepare_admitted checks real usage.
        let dense = 128 * crate::shared::dense_read::STEP_BYTES as u64;
        let expert = headers.iter().filter(|h| h.name.contains(".ffn.experts.") && h.name.ends_with(".scale"))
            .map(|h| h.byte_length).sum::<u64>();
        let push = crate::push_load::PushHostEnvelope::new(vec![expert.saturating_add(1 << 20); gpus], vec![dense; gpus], 128, 1 << 20)?;
        for allocation in push.allocations("v41.push")? {
            if allocation.id.ends_with(".dense") {
                estimates.push(Opaque { location: allocation.location, bytes: allocation.bytes,
                    basis: Basis::Estimated, lifetime: allocation.lifetime,
                    source: "V4.1 dense push staging (estimated: 128 slots x 32 MiB maximum step)".into() });
            } else {
                owners.push(Owner::host(allocation.category, allocation.group, allocation.bytes)
                    .at(allocation.location).with_lifetime(allocation.lifetime));
            }
        }
        let _ = full_logits; // The executor retains a full-span scoring plane in both modes.
        Ok(Self { owners, estimates, prefix_bytes, mapped, push })
    }
}

impl AdmittedV41 {
    /// A projection of the admitted shared host budget, never another admission.
    pub fn host_ledger(&self, quota: u64) -> Result<cuteafd_core::memory_layout::HostLayout, Refusal> {
        use cuteafd_core::memory_layout::{HostLayout, Item, ItemLifetime};
        let budget = self.solved.budgets.iter().find(|b| b.locations.contains(&Location::HOST))
            .ok_or_else(|| invalid("missing host budget"))?;
        let available = budget.available()?;
        let peak = self.solved.phases.iter().filter(|p| p.budget == budget.id)
            .map(|p| p.owned + p.opaque).max().unwrap_or(0);
        let mut items = self.solved.items(Location::HOST)?;
        items.extend(self.solved.items(Location::PINNED)?);
        items.retain(|item| item.group != "host prefix cache");
        let ready = items.iter().map(|i| i.bytes).sum::<u64>() + quota;
        if peak > ready { items.push(Item::new(Category::Staging, "load phase above retained host owners", "", peak - ready,
            Basis::Exact).with_lifetime(ItemLifetime::ReadyResident)); }
        Ok(HostLayout { total_bytes: budget.physical_bytes, available_bytes: available,
            reserve_bytes: budget.headroom, host_budget_bytes: quota,
            resident_bytes: peak.saturating_sub(quota), deferred_bytes: 0,
            headroom_bytes: available.saturating_sub(peak), shortfall_bytes: peak.saturating_sub(available), items })
    }
}

/// Physical capacity and the operator ceiling are independent inputs. The
/// offline baseline is a forecast: the recipe charges every owner itself.
pub fn production_budgets(
    cards: &[(u64, u64)],
    headroom: u64,
    host: Option<(u64, u64)>,
    host_headroom: u64,
    spark_bytes: &[u64],
) -> Result<Vec<Budget>, Refusal> {
    if !(1..=2).contains(&cards.len()) || cards.iter().any(|&(physical, logical)| logical > physical) {
        return Err(invalid("invalid production physical capacities"));
    }
    let forecast = || Baseline::Forecast {
        occupied: 0,
        source: "physical capacity; all V4.1 owners charged in the recipe".into(),
        assumed: false,
    };
    let mut budgets = cards
        .iter()
        .enumerate()
        .map(|(rank, &(physical, logical))| Budget {
            id: format!("rtx{rank}"),
            locations: vec![Location::coordinator(rank as u8)],
            physical_bytes: physical,
            logical_bytes: logical,
            baseline: forecast(),
            headroom,
        })
        .collect::<Vec<_>>();
    if let Some(host) = host {
        if host.1 > host.0 || host.0 == 0 {
            return Err(invalid("invalid host physical capacity"));
        }
        budgets.push(Budget {
            id: "coordinator host RAM".into(),
            locations: vec![Location::HOST, Location::PINNED],
            physical_bytes: host.0,
            logical_bytes: host.0,
            baseline: Baseline::Sampled {
                free: host.1,
                phase: Phase::Load,
                source: "host MemAvailable normalized to external occupancy before whole-process recipe".into(),
            },
            headroom: host_headroom,
        });
    }
    budgets.extend(spark_bytes.iter().enumerate().map(|(rank, &bytes)| Budget {
        id: format!("spark{rank}"),
        locations: vec![Location {
            device: Device::Spark(rank as u8),
            domain: Domain::Managed,
        }],
        physical_bytes: bytes,
        logical_bytes: bytes,
        baseline: forecast(),
        headroom: 0,
    }));
    Ok(budgets)
}

/// Live admission: each card's measured free bytes witness the occupied part of
/// its physical capacity, and the operator ceiling stays a separate logical cap.
pub fn live_budgets(
    cards: &[(u64, u64, u64)],
    spark_bytes: &[u64],
) -> Result<Vec<Budget>, Refusal> {
    live(cards, spark_bytes, None)
}

/// Live admission for a caller that charges the device's pre-weight occupancy
/// itself: every card's cap starts whole and its sampled free bytes are not a
/// baseline, because the predicted runtime base (`runtime_base`) is beside it
/// as named owners. Never used with a whole-GPU reserve, which replaces those
/// owners and therefore needs the sample (`live_budgets`).
pub fn live_budgets_charged(
    cards: &[(u64, u64, u64)],
    spark_bytes: &[u64],
) -> Result<Vec<Budget>, Refusal> {
    live(cards, spark_bytes, Some(
        "full cap; the CUDA context and selected module images are charged as predicted owners",
    ))
}

fn live(
    cards: &[(u64, u64, u64)],
    spark_bytes: &[u64],
    charged: Option<&str>,
) -> Result<Vec<Budget>, Refusal> {
    if !(1..=2).contains(&cards.len())
        || cards.iter().any(|&(total, ceiling, free)| total == 0 || ceiling > total || free > total)
    {
        return Err(invalid("invalid live memory samples"));
    }
    let mut budgets = cards
        .iter()
        .enumerate()
        .map(|(rank, &(total, ceiling, free))| Budget {
            id: format!("rtx{rank}"),
            locations: vec![Location::coordinator(rank as u8)],
            physical_bytes: total,
            logical_bytes: ceiling,
            baseline: match charged {
                Some(source) => Baseline::Forecast { occupied: 0, source: source.into(), assumed: false },
                None => Baseline::Sampled {
                    free,
                    phase: Phase::Load,
                    source: "measured free bytes before the whole-process recipe".into(),
                },
            },
            headroom: crate::placement::reserve::gpu("deepseek_v41"),
        })
        .collect::<Vec<_>>();
    budgets.extend(spark_bytes.iter().enumerate().map(|(rank, &bytes)| Budget {
        id: format!("spark{rank}"),
        locations: vec![Location {
            device: Device::Spark(rank as u8),
            domain: Domain::Managed,
        }],
        physical_bytes: bytes,
        logical_bytes: bytes,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "Spark physical capacity; native expert charges applied at placement".into(),
            assumed: false,
        },
        headroom: 0,
    }));
    Ok(budgets)
}

pub fn production_request(
    context: impl Into<crate::placement::planner::Context>,
    pool: Option<u64>,
    onboard: crate::placement::Onboard,
    prefill: PrefillShape,
    operation: Operation,
    resident_weight_limit: u64,
    spark_reply_formats: SparkReplyFormats,
) -> Request {
    Request {
        context: context.into(),
        pool: crate::placement::PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: match onboard {
            crate::placement::Onboard::Auto => Pin::Auto,
            crate::placement::Onboard::Layers(n) => Pin::Exact(n as u64),
        },
        prefill: Pin::Exact(prefill),
        operation,
        resident_weight_limit: Some(ResidentWeights(resident_weight_limit)),
        spark_reply_formats,
    }
}

/// Physical subrecipes prepared from headers and program tables. This builder
/// selects neither the pool nor an expert prefix; the shared planner does.
pub struct V41Recipe<'a> {
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub config: &'a serde_json::Value,
    pub context: u64,
    pub page_tokens: u64,
    pub maximum_pages: u64,
    /// The executor's fixed CED ownership, unchanged by this port.
    pub attention: Vec<LayerMode>,
    pub gpus: usize,
    pub spark_ranks: usize,
    pub prefill: PrefillShape,
    pub counts: Vec<usize>,
    /// Routed layers 0..40 in canonical order (the exact executor prefix).
    pub layers: Vec<usize>,
    pub arena_peak: Vec<[u64; 2]>,
    pub arena_resident: Vec<[u64; 2]>,
    pub encoders: Vec<crate::plan::encoder::EncoderPlacement>,
    /// The bundled tower's exact arena (`weight_layout::vision_layout_from_headers`),
    /// charged wherever an encoder option places it (Managed has no SM120 tail).
    pub vision_arena: Option<crate::serving_capacity::WeightArenaLayout>,
    pub media_cache_bytes: u64,
    pub concurrency: u64,
    pub tails: u64,
    pub replicated: bool,
    pub draft: Option<DraftInventory>,
    pub blas: Vec<serving::BlasOwners>,
    /// Every always-active physical owner except KV and the routed-expert arena.
    pub owners: Vec<Case<Vec<Component>>>,
    pub opaques: Vec<Opaque>,
}

impl V41Recipe<'_> {
    fn destination(&self) -> Vec<u8> {
        (0..self.gpus as u8).collect()
    }

    /// Mechanical executor output from the admitted decisions.
    pub fn execution(
        &self,
        solved: &ResolvedPlan,
    ) -> Result<(crate::placement::Placement, crate::placement::SparkWorkingSet), Refusal> {
        use crate::placement::{AttentionClass, ExpertHome, ExpertRange, LayerAssignment, Placement,
            PoolLimits, Tp2Range, plan_hops, residual_homes};
        let modes = &solved.decisions.attention;
        if !(1..=2).contains(&self.gpus)
            || *modes != self.attention
            || self.attention.len() != self.layers.len()
            || solved.decisions.onboard_layers.len() != self.gpus
            || solved.decisions.prefill != self.prefill
            || solved.decisions.lanes != 1
        {
            return Err(invalid("resolved execution differs from prepared geometry"));
        }
        let attention = if self.gpus == 1 {
            AttentionPlacement::Heads
        } else {
            AttentionPlacement::Layers
        };
        let mut local = Vec::new();
        let mut remote = Vec::new();
        let mut local_by_rank = vec![Vec::new(); self.gpus];
        let mut ranks = None;
        let mut assignments = Vec::with_capacity(modes.len());
        for (layer, &mode) in modes.iter().enumerate() {
            let experts = match solved.decisions.expert_homes.get(&layer) {
                None => return Err(invalid("resolved routed layer has no expert home")),
                Some(ExpertDestination::Coordinator { gpus }) => {
                    if *gpus != self.destination() {
                        return Err(invalid("resolved local home differs from prepared destination"));
                    }
                    let home = match gpus.as_slice() {
                        [0, 1] if self.gpus == 2 => ExpertHome::RtxTp2,
                        [gpu] if usize::from(*gpu) < self.gpus => ExpertHome::RtxWhole { gpu: *gpu },
                        _ => return Err(invalid("executor cannot bind resolved local expert home")),
                    };
                    for &gpu in gpus {
                        local_by_rank[usize::from(gpu)].push(layer);
                    }
                    local.push(layer);
                    home
                }
                Some(ExpertDestination::Sparks { ranks: selected }) => {
                    if selected.is_empty()
                        || selected.len() > 8
                        || selected
                            .iter()
                            .enumerate()
                            .any(|(rank, &value)| usize::from(value) != rank)
                    {
                        return Err(invalid(
                            "V4.1 requires a Spark rank prefix starting at logical rank 0",
                        ));
                    }
                    if selected.len() != self.spark_ranks
                        || ranks.as_ref().is_some_and(|ranks| ranks != selected)
                    {
                        return Err(invalid(
                            "resolved Spark layers do not share the prepared rank set",
                        ));
                    }
                    ranks = Some(selected.clone());
                    remote.push(layer);
                    ExpertHome::Spark
                }
            };
            assignments.push(LayerAssignment { mode, experts });
        }
        if solved
            .decisions
            .expert_homes
            .keys()
            .any(|&layer| layer >= modes.len())
            || local_by_rank != solved.decisions.onboard_layers
            || !self.counts.contains(&local.len())
        {
            return Err(invalid("resolved onboarding is not the declared ordered prefix"));
        }
        let count = local.len();
        let peak = self.arena_peak[count];
        let whole = local
            .iter()
            .any(|&layer| matches!(assignments[layer].experts, ExpertHome::RtxWhole { .. }));
        let tp2 = if count > 0 && !whole {
            Some(Tp2Range { first: 0, layers: count, peak_bytes: peak })
        } else {
            None
        };
        let ranges = (0..self.gpus)
            .map(|rank| {
                if tp2.is_some() || count == 0 {
                    ExpertRange { first: 0, layers: 0, peak_bytes: 0 }
                } else {
                    ExpertRange { first: 0, layers: count, peak_bytes: peak[rank] }
                }
            })
            .collect();
        let spec = super::execution_hops(self.gpus);
        let residual = residual_homes(modes, &spec).map_err(|error| invalid(&error.to_string()))?;
        let hops = if self.gpus == 2 {
            plan_hops(modes, &spec).map_err(|error| invalid(&error.to_string()))?
        } else {
            Vec::new()
        };
        let movables = match self.draft {
            Some(_) => vec![(
                MovableId::Drafter,
                u8::try_from(self.gpus - 1).map_err(|_| invalid("draft home"))?,
            )],
            None => Vec::new(),
        };
        let placement = Placement {
            pool_tokens: solved.pool_tokens,
            pool_limits: PoolLimits(vec![]),
            attention_placement: attention,
            attention_by_kind: vec![(AttentionClass::Csa, attention)],
            peer_row_bytes: 0,
            onboard_layers: count,
            max_pool_tokens_no_onboard: None,
            layers: assignments,
            movables,
            expert_ranges: ranges,
            residual,
            hops,
            tp2,
            items: (0..self.gpus)
                .map(|rank| solved.items(Location::coordinator(rank as u8)))
                .collect::<Result<_, _>>()?,
        };
        super::super::DEEPSEEK_V41
            .check(&placement)
            .map_err(|error| invalid(&error.to_string()))?;
        let working = crate::placement::SparkWorkingSet {
            rtx_layers: local,
            spark_layers: remote,
            spark_ranks: ranks.map_or(0, |ranks| ranks.len()),
            prefill_lanes: solved.decisions.prefill.waves,
        };
        Ok((placement, working))
    }
}

pub struct ProductionInput<'a> {
    pub expert_usage: bool,
    pub config: &'a serde_json::Value,
    pub gpus: usize,
    /// Smallest pool the layout admits, in page tokens.
    pub context: u64,
    /// Largest indexable pool request, in page tokens.
    pub maximum_pool_tokens: u64,
    pub prefill: PrefillShape,
    /// Concurrent sequences whose private source tables and tails are charged.
    pub concurrency: u64,
    pub retained_turns: u64,
    pub replicated: bool,
    pub spark_ranks: usize,
    /// Automatic admission keeps at least this many routed layers local.
    pub onboard_minimum: usize,
    pub onboard_counts: Option<Vec<usize>>,
    pub experts: Vec<ExpertCost>,
    pub workspace: u64,
    /// Every retained physical owner except KV and the routed-expert arena.
    pub owners: Vec<Owner>,
    /// Weight arenas (target ranks, serving vision tower). Each replaces the one
    /// `owners::arena_demand` owner naming its ID, which carries the same extent
    /// B, with the shared root, a logical-payload claim and the tail.
    pub weight_arenas: Vec<crate::serving_capacity::WeightArenaLayout>,
    /// Sequential load peaks in absolute bytes above the availability sample.
    /// Each is charged as the part above the owners already live at load, and
    /// the peaks never coexist.
    pub startup: Vec<Owner>,
    /// Post-ready reserves that are not allocator receipts.
    pub allowances: Vec<Opaque>,
    pub draft: Option<DraftInventory>,
    /// Future native borrowers above the sample, including a local media thread.
    pub blas: Vec<serving::BlasOwners>,
    pub encoders: Vec<crate::plan::encoder::EncoderPlacement>,
    pub media_cache_bytes: u64,
    pub resident_weight_limit: u64,
    pub spark_reply_formats: SparkReplyFormats,
    pub operation: Operation,
    pub budgets: Vec<Budget>,
}

pub struct PreparedV41<'a> {
    pub recipe: V41Recipe<'a>,
    budgets: Vec<Budget>,
    /// Compulsory local routed layers; a smaller solved prefix is refused.
    minimum: usize,
}

pub struct AdmittedV41 {
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub placement: crate::placement::Placement,
    pub spark: crate::placement::SparkWorkingSet,
    pub solved: ResolvedPlan,
}

/// The backbone router emits full top-k on its attention owner, not on TP peers.
pub fn expert_usage_geometry(config: &serde_json::Value) -> Result<cuteafd_core::expert_usage::Geometry, Refusal> {
    let config = config.get("text_config").unwrap_or(config);
    let field = |key| config.get(key).and_then(serde_json::Value::as_u64)
        .and_then(|n| usize::try_from(n).ok()).ok_or_else(|| invalid("missing expert usage geometry"));
    let layers = field("num_hidden_layers")?;
    let geometry = cuteafd_core::expert_usage::Geometry {
        layers, experts: field("n_routed_experts")?, topk: field("num_experts_per_tok")?,
        routed_layers: (0..layers).collect(),
    };
    if !geometry.validate() || layers != 40 { return Err(invalid("invalid V4.1 expert usage geometry")); }
    Ok(geometry)
}

impl<'a> ProductionInput<'a> {
    pub fn prepare(mut self) -> Result<PreparedV41<'a>, Refusal> {
        self.owners.extend(super::owners::packed_head_load_owners(&self.weight_arenas)
            .map_err(|error| invalid(&error.to_string()))?);
        let routed = self.experts.len();
        if !(1..=2).contains(&self.gpus)
            || routed != 40
            || self.context == 0
            || self.maximum_pool_tokens < self.context
            || self.spark_ranks > 8
            || !(1..=cuteafd_core::V41_MAX_REQUESTS as u64).contains(&self.concurrency)
            || self.retained_turns > 128
            || self
                .budgets
                .iter()
                .filter(|budget| {
                    budget
                        .locations
                        .iter()
                        .any(|location| matches!(location.device, Device::Coordinator(_)))
                })
                .count()
                != self.gpus
            || self.replicated && self.gpus != 2
            || self.onboard_minimum > routed
        {
            return Err(invalid("incomplete production geometry"));
        }
        let tails = self
            .concurrency
            .checked_add(
                2u64.checked_mul(self.retained_turns)
                    .ok_or(Refusal::Overflow("V4.1 retained turns"))?,
            )
            .ok_or(Refusal::Overflow("V4.1 private tails"))?;
        let maximum_pages = self
            .maximum_pool_tokens
            .div_ceil(512)
            .min(131_072u64.saturating_sub(tails));
        if maximum_pages == 0 || self.context.div_ceil(512) > maximum_pages {
            return Err(Refusal::PoolBelowContext {
                tokens: self.maximum_pool_tokens,
                context: self.context,
            });
        }
        let attention = (0..routed)
            .map(|layer| LayerMode::Whole {
                gpu: u8::from(self.gpus == 2 && layer >= 20),
                ffn: FfnMode::Owner,
            })
            .collect::<Vec<_>>();
        let counts = match &self.onboard_counts {
            Some(counts) => counts.clone(),
            None if self.spark_ranks == 0 => vec![routed],
            None => (self.onboard_minimum..=routed).collect(),
        };
        if counts.is_empty()
            || counts.windows(2).any(|pair| pair[0] >= pair[1])
            || counts.iter().any(|&count| count > routed || count < self.onboard_minimum)
        {
            return Err(invalid("invalid local expert prefix profile"));
        }
        let mut arenas = Vec::with_capacity(routed + 1);
        for count in 0..=routed {
            arenas.push(arena(count, self.gpus, &self.experts, self.workspace)?);
        }
        let expert_usage = self.expert_usage.then(|| expert_usage_geometry(self.config)).transpose()?;
        let mut recipe = V41Recipe {
            expert_usage,
            config: self.config,
            context: self.context,
            page_tokens: 512,
            maximum_pages,
            attention,
            gpus: self.gpus,
            spark_ranks: self.spark_ranks,
            prefill: self.prefill,
            counts,
            layers: (0..routed).collect(),
            arena_peak: arenas.iter().map(|(peak, _)| *peak).collect(),
            arena_resident: arenas.iter().map(|(_, resident)| *resident).collect(),
            encoders: self.encoders,
            vision_arena: None,
            media_cache_bytes: self.media_cache_bytes,
            concurrency: self.concurrency,
            tails,
            replicated: self.replicated,
            draft: self.draft,
            blas: self.blas,
            owners: Vec::new(),
            opaques: self.allowances,
        };
        // Every arena replaces its one same-ID, same-extent owner exactly once.
        let mut arenas = BTreeMap::new();
        for layout in &self.weight_arenas {
            if arenas.insert(layout.id().to_string(), Some(layout)).is_some() {
                return Err(invalid("duplicate weight arena layout"));
            }
        }
        let mut owners = Vec::with_capacity(self.owners.len());
        for owner in &self.owners {
            let Some(id) = owner.group.strip_prefix(super::owners::WEIGHT_ARENA_PREFIX) else {
                owners.push(owner.clone());
                continue;
            };
            if owner.arena.is_some() || !matches!(owner.location.device, Device::Coordinator(_)) {
                return Err(invalid("weight arena owner is not an unbound coordinator owner"));
            }
            let layout = arenas.get_mut(id).and_then(Option::take)
                .ok_or_else(|| invalid("weight arena owner has no layout or appears twice"))?;
            if layout.bytes() as u64 != owner.bytes || owner.resident != owner.bytes {
                return Err(invalid("weight arena owner differs from its layout"));
            }
            owners.push(Owner { arena: Some((layout.clone(), MemoryKind::Weights)), ..owner.clone() });
        }
        if arenas.values().any(Option::is_some) { return Err(invalid("weight arena layout without an owner")); }
        recipe.owners.push(Case {
            when: Vec::new(),
            value: owner_components("owner", &owners, maximum_pages)?,
        });
        let draft_demands = recipe.draft.as_ref().map(|draft| draft_owners(draft, self.gpus))
            .transpose()?;
        if let Some(owners) = &draft_demands {
            recipe.owners.push(Case {
                when: Vec::new(),
                value: owner_components("draft", owners, maximum_pages)?,
            });
        }
        // Sequential load peaks are absolute above the sample; charge each as
        // the part not already covered by the owners that are live at load,
        // including the drafter's retained weights the peak already contains.
        // Owners the sample itself already holds (the predicted runtime base)
        // are not in those peaks.
        let mut base = vec![0u64; self.gpus];
        for owner in self.owners.iter().chain(draft_demands.iter().flatten()) {
            if let Device::Coordinator(rank) = owner.location.device {
                if !owner.deferred && !owner.pre_sample && owner.lifetime.first == Phase::Load {
                    base[usize::from(rank)] = base[usize::from(rank)]
                        .checked_add(owner.bytes)
                        .ok_or(Refusal::Overflow("V4.1 load base"))?;
                }
            }
        }
        let mut startup = vec![Vec::new(); self.gpus];
        for (index, phase) in self.startup.iter().enumerate() {
            let Device::Coordinator(rank) = phase.location.device else {
                return Err(invalid("startup peaks are device owners"));
            };
            let delta = phase.bytes.saturating_sub(base[usize::from(rank)]);
            if delta == 0 {
                continue;
            }
            startup[usize::from(rank)].push(Allocation::buffer(
                format!("v41.startup.{rank}.{index}"),
                phase.location,
                delta,
                Lifetime { first: Phase::Load, last: Phase::Load },
                phase.category,
                phase.group.clone(),
            ));
        }
        for members in startup.into_iter().filter(|members| !members.is_empty()) {
            let peak = members.iter().map(|member| member.bytes).max().unwrap_or(0);
            recipe.owners.push(Case {
                when: Vec::new(),
                value: vec![Component {
                    activation: Activation::always(),
                    bytes: ByteCurve::fixed(peak, maximum_pages),
                    kind: ComponentKind::Sequential(Sequential {
                        id: members[0].id.clone(),
                        members,
                    }),
                }],
            });
        }
        let minimum = self.onboard_minimum;
        Ok(PreparedV41 { recipe, budgets: self.budgets, minimum })
    }
}

/// dSpark drafter owners: retained weights and request lanes at the head, the
/// optional peer expert half on GPU0, and marks; load staging is not retained.
pub fn draft_owners(draft: &DraftInventory, gpus: usize) -> Result<Vec<Owner>, Refusal> {
    if !(1..=2).contains(&gpus)
        || draft.runtime.iter().any(|owner| usize::from(owner.gpu) >= gpus)
    {
        return Err(invalid("dSpark owner is outside the topology"));
    }
    let home = (gpus - 1) as u8;
    let mut owners = Vec::new();
    let mut experts = draft.resident;
    if let Some(dense) = &draft.dense {
        // The dense arena is part of `resident`; the rest is routed experts.
        experts = experts.checked_sub(dense.bytes() as u64)
            .ok_or_else(|| invalid("dSpark dense arena exceeds its retained weights"))?;
        owners.push(Owner::arena(home, dense.clone(), MemoryKind::DrafterWeights));
    }
    if experts > 0 { owners.push(Owner::weights(home, Category::Drafter, "dSpark experts", experts)); }
    if draft.marks > 0 {
        owners.push(Owner::charged(home, Category::Prefix, "dSpark marks", draft.marks));
    }
    if draft.peer_half > 0 {
        owners.push(Owner::weights(
            0,
            Category::Drafter,
            "dSpark TP2 expert half",
            draft.peer_half,
        ));
    }
    owners.extend(draft.runtime.iter().map(Owner::from_demand));
    Ok(owners)
}

impl PreparedV41<'_> {
    pub fn admit(self, request: Request) -> Result<AdmittedV41, Refusal> {
        let solved = crate::placement::planner::solve(request, self.budgets, &self.recipe)?;
        let count = solved
            .decisions
            .onboard_layers
            .iter()
            .flatten()
            .collect::<BTreeSet<_>>()
            .len();
        if count < self.minimum {
            return Err(invalid("mandatory V4.1 expert prefix does not fit"));
        }
        let (placement, spark) = self.recipe.execution(&solved)?;
        Ok(AdmittedV41 { expert_usage: self.recipe.expert_usage, placement, spark, solved })
    }
}

struct Domains<'a> {
    prefill: (usize, &'a [PrefillShape]),
    lanes: (usize, &'a [u64]),
    onboard: (usize, &'a [usize]),
}

impl Domains<'_> {
    fn activation(&self, conditions: &[When]) -> Result<Activation, Refusal> {
        let mut domains = BTreeMap::<usize, BTreeSet<usize>>::new();
        for condition in conditions {
            let (variable, options): (usize, Vec<usize>) = match condition {
                When::Prefill(shape) => (
                    self.prefill.0,
                    self.prefill
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(index, value)| (value == shape).then_some(index))
                        .collect(),
                ),
                When::Lanes(lanes) => (
                    self.lanes.0,
                    self.lanes
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(index, value)| (value == lanes).then_some(index))
                        .collect(),
                ),
                When::Onboard { first, last } => {
                    let (first, last) = (*first, *last);
                    if first > last {
                        return Err(invalid("invalid onboarding interval"));
                    }
                    (
                        self.onboard.0,
                        self.onboard
                            .1
                            .iter()
                            .enumerate()
                            .filter_map(|(index, &count)| {
                                (first <= count && count <= last).then_some(index)
                            })
                            .collect(),
                    )
                }
            };
            if options.is_empty() {
                return Err(invalid("condition has no domain value"));
            }
            let options: BTreeSet<_> = options.into_iter().collect();
            if let Some(existing) = domains.get_mut(&variable) {
                existing.retain(|option| options.contains(option));
                if existing.is_empty() {
                    return Err(invalid("contradictory component conditions"));
                }
            } else {
                domains.insert(variable, options);
            }
        }
        let mut clauses = vec![Vec::new()];
        for (variable, options) in domains {
            clauses = clauses
                .into_iter()
                .flat_map(|clause| {
                    options.iter().map(move |&option| {
                        let mut clause = clause.clone();
                        clause.push(Choice { variable, option });
                        clause
                    })
                })
                .collect();
        }
        Ok(Activation { clauses })
    }

    fn append(
        &self,
        contract: &mut Contract,
        conditions: &[When],
        components: Vec<Component>,
    ) -> Result<(), Refusal> {
        let activation = self.activation(conditions)?;
        for mut component in components {
            // Subrecipes cannot smuggle private variable indexes into the family contract.
            if component.activation != Activation::always() {
                return Err(invalid("subrecipe supplied a private activation"));
            }
            component.activation = activation.clone();
            contract.components.push(component);
        }
        Ok(())
    }
}

impl Recipe for V41Recipe<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if !(1..=2).contains(&self.gpus)
            || self.context == 0
            || request.context.maximum != self.context
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
            || self.page_tokens != 512
            || self.attention.len() != 40
            || self.counts.is_empty()
            || self.layers != (0..40).collect::<Vec<_>>()
            || self.tails > 4096
            || self.counts.iter().any(|&count| count > self.layers.len())
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        self.page_tokens
            .checked_mul(self.maximum_pages)
            .ok_or(Refusal::Overflow("V4.1 pool extent"))?;
        let mut contract = Contract {
            floor_fallback: None,
            resident_weight_limit: request.resident_weight_limit,
            phase_order: PhaseOrder::DRAFT_THEN_TARGET,
            spark_reply_formats: (self.spark_ranks > 0).then(|| SparkReplyFormats {
                prefill: Some(
                    request
                        .spark_reply_formats
                        .prefill(cuteafd_core::ExpertGeometry::DEEPSEEK_V41.partial_replies()),
                ),
                decode: Some(request.spark_reply_formats.decode()),
            }),
            page_rows: self.page_tokens,
            pool_page_multiple: 1,
            maximum_pages: self.maximum_pages,
            gpus: self.gpus,
            variables: Vec::new(),
            requires: Vec::new(),
            components: Vec::new(),
        };
        for (layer, &mode) in self.attention.iter().enumerate() {
            contract
                .variables
                .push(Variable::Attention { layer, values: vec![mode] });
        }
        if self.draft.is_some() { contract.variables.push(Variable::DraftHome(vec![(self.gpus - 1) as u8])); }
        let prefill = contract.variables.len();
        contract.variables.push(Variable::Prefill(vec![self.prefill]));
        let lanes = contract.variables.len();
        contract.variables.push(Variable::Lanes(vec![1]));
        let onboard = contract.variables.len();
        contract.variables.push(Variable::Onboard {
            layers: self.layers.clone(),
            destination: (0..self.gpus as u8).collect(),
            counts: self.counts.clone(),
        });
        if self.spark_ranks > 0 {
            contract.variables.push(Variable::ExpertHome {
                layers: self.layers.clone(),
                rank_sets: vec![(0..self.spark_ranks as u8).collect()],
            });
        }
        let domains = Domains {
            prefill: (prefill, std::slice::from_ref(&self.prefill)),
            lanes: (lanes, &[1]),
            onboard: (onboard, &self.counts),
        };
        for case in &self.owners {
            domains.append(&mut contract, &case.when, case.value.clone())?;
        }
        if let Some(geometry) = &self.expert_usage {
            contract.components.extend(crate::placement::expert_usage::components(
                geometry, &(0..self.gpus as u8).collect::<Vec<_>>(), self.maximum_pages)?);
        }
        // Opaque reserves are unconditional and price only the post-ready phases.
        contract
            .components
            .extend(opaque_components(&self.opaques, self.maximum_pages));
        // Routed-expert arena residency per ordered prefix.
        for (option, &count) in self.counts.iter().enumerate() {
            let mut components = Vec::new();
            for rank in 0..self.gpus {
                let peak = self.arena_peak[count][rank];
                let resident = self.arena_resident[count][rank];
                let id = format!("v41.experts.{option}.{rank}");
                if peak > 0 {
                    let allocation = Allocation::buffer(
                        id.clone(),
                        Location::coordinator(rank as u8),
                        peak,
                        Lifetime::RESIDENT,
                        Category::Experts,
                        "routed experts",
                    );
                    components.push(Component::allocation(
                        allocation,
                        ByteCurve::fixed(peak, self.maximum_pages),
                        false,
                        self.maximum_pages,
                    ));
                }
                if resident > 0 {
                    let backing = Allocation::buffer(
                        id,
                        Location::coordinator(rank as u8),
                        resident,
                        Lifetime::RESIDENT,
                        Category::Experts,
                        "routed experts",
                    );
                    components.push(Component::resident_weight(
                        ResidentWeight::from_allocation(&backing, ResidentWeights(resident)),
                        ByteCurve::fixed(resident, self.maximum_pages),
                    ));
                }
            }
            if components.is_empty() {
                continue;
            }
            contract.components.extend(components.into_iter().map(|mut component| {
                component.activation = Activation::choice(onboard, option);
                component
            }));
        }
        let mut blas = self.blas.clone();
        // Encoder candidates are finite; the shared solve picks the first that
        // fits every phase and never runs again to choose a topology.
        if !self.encoders.is_empty() {
            use crate::plan::encoder::EncoderKind;
            let variable = contract.variables.len();
            contract
                .variables
                .push(Variable::Encoder((0..self.encoders.len()).collect()));
            for (option, encoder) in self.encoders.iter().enumerate() {
                let when = Choice { variable, option };
                let locations = match &encoder.kind {
                    EncoderKind::Off => Vec::new(),
                    EncoderKind::Rtx { gpu } if *gpu < self.gpus => {
                        vec![Location::coordinator(*gpu as u8)]
                    }
                    EncoderKind::Spark { .. } => {
                        encoder
                            .spark_ranks()
                            .into_iter()
                            .map(|rank| Location {
                                device: Device::Spark(rank as u8),
                                domain: Domain::Managed,
                            })
                            .collect()
                    }
                    _ => return Err(invalid("encoder device absent from the executable topology")),
                };
                let mut components = Vec::new();
                for (copy, location) in locations.iter().copied().enumerate() {
                    // Spark media BLAS is already in V41_SPARK_CUDA_OVERHEAD_BYTES (#1568).
                    if matches!(location.device, Device::Coordinator(_)) {
                        blas.push(serving::BlasOwners { name: format!("V4.1 vision candidate {option} copy {copy}"),
                            process: "coordinator".into(), thread: "v41/blas/media".into(), location,
                            gemm: true, lt: false, lifetime: Lifetime::RESIDENT,
                            activation: Activation::choice(variable, option) });
                    }
                    for (part, bytes, category) in [
                        ("weights", encoder.weights, Category::Weights),
                        ("scratch", encoder.scratch, Category::Workspace),
                    ] {
                        if bytes == 0 {
                            continue;
                        }
                        // Every tower owns B, budgeted at logical payload P. Only
                        // coordinator device arenas have the SM120 allocation tail.
                        if part == "weights" {
                            let layout = self.vision_arena.as_ref()
                                .ok_or_else(|| invalid("vision tower lacks its arena layout"))?;
                            let payload = u64::try_from(layout.payload_bytes())
                                .map_err(|_| Refusal::Overflow("V4.1 vision payload"))?;
                            if payload != bytes {
                                return Err(invalid("vision arena payload differs from the tower source"));
                            }
                            let id = format!("v41.encoder.{option}.{copy}.weights");
                            let extent = u64::try_from(layout.bytes()).map_err(|_| Refusal::Overflow("V4.1 vision arena"))?;
                            let root = Allocation::buffer(&id, location, extent, Lifetime::RESIDENT,
                                MemoryKind::Weights.category(), "weight arena");
                            if matches!(location.device, Device::Coordinator(_)) {
                                let mut arena = weight_arena_extent_components(&id, extent, location,
                                    Lifetime::RESIDENT, self.maximum_pages, cuteafd_core::memory_layout::MemoryKind::Weights)?;
                                arena.retain(|component| !matches!(component.kind, ComponentKind::ResidentWeight(_)));
                                components.extend(arena);
                            } else {
                                components.push(Component::allocation(root.clone(),
                                    ByteCurve::fixed(extent, self.maximum_pages), false, self.maximum_pages));
                            }
                            components.push(Component::resident_weight(
                                ResidentWeight::from_allocation(&root, ResidentWeights(payload)),
                                ByteCurve::fixed(payload, self.maximum_pages),
                            ));
                            continue;
                        }
                        let owner = Allocation::buffer(
                            format!("v41.encoder.{option}.{copy}.{part}"),
                            location,
                            bytes,
                            Lifetime::RESIDENT,
                            category,
                            match part {
                                "weights" => "vision tower".to_string(),
                                _ => "vision scratch".to_string(),
                            },
                        );
                        if part == "weights" {
                            components.push(Component::resident_weight(
                                ResidentWeight::from_allocation(&owner, ResidentWeights(bytes)),
                                ByteCurve::fixed(bytes, self.maximum_pages),
                            ));
                        }
                        components.push(Component::allocation(
                            owner,
                            ByteCurve::fixed(bytes, self.maximum_pages),
                            false,
                            self.maximum_pages,
                        ));
                    }
                }
                if !locations.is_empty() && self.media_cache_bytes > 0 {
                    let owner = Allocation::buffer(
                        "v41.media.cache",
                        Location::HOST,
                        self.media_cache_bytes,
                        Lifetime::TARGET,
                        Category::Embedding,
                        "media embedding cache",
                    );
                    components.push(Component::allocation(
                        owner,
                        ByteCurve::fixed(self.media_cache_bytes, self.maximum_pages),
                        false,
                        self.maximum_pages,
                    ));
                }
                for mut component in components {
                    component.activation = Activation::choice(variable, option);
                    contract.components.push(component);
                }
            }
        }
        for (arch, coordinator) in [("sm_120", true), ("sm_121", false)] {
            let borrowers = blas.iter().filter(|owner| matches!(owner.location.device,
                Device::Coordinator(_)) == coordinator).cloned().collect::<Vec<_>>();
            let components = serving::blas_components(arch, &borrowers,
                self.maximum_pages, contract.phase_order)?;
            // The RTX coordinator's cuBLAS handles and shared BLAS
            // initialization exist by Ready, so charge them as occupancy: the
            // Ready check compares named owners against the device, and a
            // reserve it cannot see leaves that memory unattributed (#1635).
            let components = if arch == "sm_120" { serving::blas_occupancy(components) }
                else { components };
            contract.components.extend(components);
        }
        for components in cache_owners(
            self.config,
            self.gpus,
            self.concurrency,
            self.tails,
            self.replicated,
            self.maximum_pages,
        )? {
            contract.components.extend(components);
        }
        Ok(contract)
    }
}

#[cfg(test)]
mod explicit_budget_tests {
    #[test]
    fn graph_reserve_prices_only_the_named_arena() {
        let owner = super::graph_allowance(0).unwrap();
        assert_eq!(owner.bytes, 1_048_576_000);
        assert_eq!(owner.source, "graph reserve: v41.graphs.0");
        // The old lazy envelope and runtime growth owner are gone, not renamed.
        assert!(!owner.source.contains("runtime"));
    }
}

#[cfg(test)]
mod runtime_base_tests {
    use super::*;
    use crate::placement::planner::{DriverPageTable, DriverPageTableModel, PhaseOrder, View};

    fn facts(allocations: Vec<Allocation>) -> Facts {
        Facts {
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            resident_weights: Vec::new(),
            allocations,
            deferred: Vec::new(),
            sequential: Vec::new(),
            reservations: Vec::new(),
            allowances: Vec::new(),
            driver_page_tables: Vec::new(),
            allocation_granularity: Vec::new(),
        }
    }

    fn ready(id: &str, gpu: u8, bytes: u64, category: Category, group: &str) -> Allocation {
        Allocation::buffer(id, Location::coordinator(gpu), bytes, Lifetime::RESIDENT, category, group)
    }

    #[test]
    fn runtime_base_is_the_calibrated_context_plus_the_measured_modules() {
        let arch = crate::placement::ArchContext::coordinator(
            crate::placement::inventory::PRO_TOTAL_BYTES, None);
        let context_only = runtime_base(0, &arch, 0);
        assert_eq!(context_only.len(), 1, "zero module bytes are not a charge");
        assert_eq!((context_only[0].group.as_str(), context_only[0].bytes),
            (CONTEXT_OWNER, arch.context_bytes));
        let owners = runtime_base(1, &arch, 40 << 20);
        let named = owners.iter().map(|owner| (owner.group.as_str(), owner.bytes)).collect::<Vec<_>>();
        assert_eq!(named, [(CONTEXT_OWNER, arch.context_bytes), (MODULE_OWNER, 40 << 20)]);
        // Both parts are already inside the pre-weight sample: never load-phase.
        assert!(owners.iter().all(|owner| owner.pre_sample && owner.resident == 0
            && owner.location == Location::coordinator(1) && owner.category == Category::Runtime));
    }

    #[test]
    fn charged_budgets_start_from_the_whole_cap_and_keep_the_operator_reserve() {
        let cards = [(96u64 << 30, 96 << 30, 40 << 30)];
        let sampled = live_budgets(&cards, &[]).unwrap();
        let charged = live_budgets_charged(&cards, &[]).unwrap();
        assert!(matches!(sampled[0].baseline,
            Baseline::Sampled { free, phase: Phase::Load, .. } if free == 40 << 30));
        assert!(matches!(charged[0].baseline,
            Baseline::Forecast { occupied: 0, assumed: false, .. }));
        assert_eq!(charged[0].physical_bytes, sampled[0].physical_bytes);
        assert_eq!(charged[0].headroom, sampled[0].headroom);
    }

    #[test]
    fn ready_check_prices_every_other_ready_owner_before_measuring_the_base() {
        let arch = crate::placement::ArchContext::coordinator(
            crate::placement::inventory::PRO_TOTAL_BYTES, None);
        let mut allocations = vec![
            ready("v41.arena", 0, 6 << 30, Category::Weights, "weight arena"),
            // A view lives inside its arena, never a second allocation.
            Allocation { view: Some(View { backing: "v41.arena".into(), offset: 0 }),
                ..ready("v41.view", 0, 1 << 20, Category::Weights, "weight arena") },
        ];
        allocations.push(ready("v41.context", 0, arch.context_bytes, Category::Runtime, CONTEXT_OWNER));
        allocations.push(ready("v41.modules", 0, 40 << 20, Category::Runtime, MODULE_OWNER));
        let mut facts = facts(allocations);
        // A post-ready reserve is headroom the operator keeps free, never occupancy
        // at Ready: it rides in facts.reservations, which the check ignores.
        facts.reservations.push(Reservation {
            id: "v41.reserve".into(),
            location: Location::coordinator(0),
            bytes: 1 << 30,
            lifetime: Lifetime { first: Phase::Ready, last: Phase::Growth },
            source: "lazy graphs".into(),
        });
        facts.driver_page_tables.push(DriverPageTable {
            model: DriverPageTableModel::Sm120Portable,
            location: Location::coordinator(0),
            lifetime: Lifetime::RESIDENT,
            backing: "v41.arena".into(),
            mapped_bytes: 6 << 30,
        });
        let tables = DriverPageTableModel::Sm120Portable.bytes(6 << 30);
        let used = (6 << 30) + arch.context_bytes + (40 << 20) + tables + 1234;
        let (predicted, measured) = runtime_base_check(&facts, 0, used).unwrap();
        assert_eq!(predicted, arch.context_bytes + (40 << 20));
        assert_eq!(measured, arch.context_bytes + (40 << 20) + 1234);
        // The Ready report names exactly the owners the check sums, so its
        // components add up to the prediction it is compared against.
        let components = runtime_base_components(&facts, 0);
        assert_eq!(components, RuntimeBaseComponents {
            context_bytes: arch.context_bytes, module_bytes: 40 << 20 });
        assert_eq!(components.total(), predicted);
        assert_eq!(runtime_base_components(&facts, 1), RuntimeBaseComponents::default(),
            "another GPU's owners are never reported as this one's");
    }
}

#[cfg(test)]
mod measured_adoption_tests {
    use super::*;
    use crate::placement::planner::{DriverPageTable, DriverPageTableModel, PhaseOrder};

    fn facts(allocations: Vec<Allocation>) -> Facts {
        Facts {
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            resident_weights: Vec::new(),
            allocations,
            deferred: Vec::new(),
            sequential: Vec::new(),
            reservations: Vec::new(),
            allowances: Vec::new(),
            driver_page_tables: Vec::new(),
            allocation_granularity: Vec::new(),
        }
    }

    fn ready(id: &str, bytes: u64, category: Category, group: &str) -> Allocation {
        Allocation::buffer(id, Location::coordinator(0), bytes, Lifetime::RESIDENT, category, group)
    }

    #[test]
    fn untracked_owners_are_every_ready_owner_the_ledger_cannot_see() {
        let arch = crate::placement::ArchContext::coordinator(
            crate::placement::inventory::PRO_TOTAL_BYTES, None);
        let mut facts = facts(vec![
            ready("v41.arena", 6 << 30, Category::Weights, "weight arena"),
            ready("v41.context", arch.context_bytes, Category::Runtime, CONTEXT_OWNER),
            ready("v41.modules", 40 << 20, Category::Runtime, MODULE_OWNER),
            ready("v41.blas", 64 << 20, Category::Runtime, serving::CUBLAS_HANDLE_OWNER),
            ready("v41.blas.init", 2 << 20, Category::Runtime, serving::BLAS_INITIALIZATION_OWNER),
            // Planned for Ready but not created until the pool is (the
            // measurement runs before it), so it must not be subtracted.
            ready("v41.kv.records.0", 40 << 30, Category::Kv, "records"),
        ]);
        facts.driver_page_tables.push(DriverPageTable {
            model: DriverPageTableModel::Sm120Portable,
            location: Location::coordinator(0),
            lifetime: Lifetime::RESIDENT,
            backing: "v41.arena".into(),
            mapped_bytes: 6 << 30,
        });
        let tables = DriverPageTableModel::Sm120Portable.bytes(6 << 30);
        assert_eq!(untracked_ready_owners(&facts, 0).unwrap(),
            arch.context_bytes + (40 << 20) + (64 << 20) + (2 << 20) + tables);
        assert_eq!(untracked_ready_owners(&facts, 1).unwrap(), 0,
            "another GPU's owners are never subtracted from this one's device");
    }

    /// One linear 1 MiB owner per page, priced at `solved` pages inside a
    /// budget that also holds a fixed 4 MiB weight arena.
    fn plan(solved: u64, budget_bytes: u64) -> (ResolvedPlan, Vec<Vec<Component>>, u64) {
        const UNIT: u64 = 1 << 20;
        let maximum = 16;
        let component = Component::allocation(
            Allocation::buffer("v41.kv.records.0", Location::coordinator(0), UNIT,
                Lifetime::TARGET, Category::Kv, "records"),
            ByteCurve::linear(UNIT, UNIT, maximum), false, maximum);
        let mut facts = facts(vec![
            ready("v41.arena", 4 * UNIT, Category::Weights, "weight arena"),
        ]);
        facts.allocations.push(Allocation::buffer("v41.kv.records.0", Location::coordinator(0),
            UNIT * solved, Lifetime::TARGET, Category::Kv, "records"));
        let budgets = vec![Budget {
            id: "gpu".into(),
            locations: vec![Location::coordinator(0)],
            physical_bytes: budget_bytes,
            logical_bytes: budget_bytes,
            baseline: Baseline::Forecast { occupied: 0, source: "test".into(), assumed: false },
            headroom: 0,
        }];
        let plan = ResolvedPlan {
            encoder_diagnostics: Vec::new(),
            request: production_request(1024u64, None, crate::placement::Onboard::Auto,
                PrefillShape { rows: 1, waves: 1 }, Operation::Serving, 0,
                SparkReplyFormats::default()),
            decisions: Decisions {
                attention: Vec::new(), onboard_layers: vec![Vec::new()],
                expert_homes: BTreeMap::new(), embedding: None, draft_home: None,
                head_home: None, intake_home: None, encoder: None,
                prefill: PrefillShape { rows: 1, waves: 1 }, lanes: 1,
                spark_reply_formats: None,
            },
            pool_tokens: solved * 512,
            pool_page_rows: 512,
            budgets,
            facts,
            phases: Vec::new(),
        };
        (plan, vec![vec![component]], UNIT)
    }

    #[test]
    fn a_measured_charge_shrinks_a_budget_bound_pool_and_keeps_a_free_one() {
        // 16 MiB available, 4 MiB fixed and 12 MiB of KV: the solved pool fills it.
        let (mut bound, owners, unit) = plan(12, 16 * (1 << 20));
        let adopted = refit_pool(&mut bound, &owners, 0).unwrap();
        assert_eq!(adopted.pages, 12, "an uncharged pool is not re-fitted");
        assert_eq!(adopted.groups, 12);
        assert_eq!(adopted.bytes, vec![12 * unit]);

        // A 3 MiB charge only fits if the pool gives up three pages.
        let (mut bound, owners, unit) = plan(12, 16 * (1 << 20));
        bound.facts.allocations.push(Allocation::buffer("v41.measured.0.0",
            Location::coordinator(0), 2 * unit, Lifetime { first: Phase::Ready, last: Phase::Growth },
            Category::Runtime, BACKING_OWNER));
        bound.facts.allocations.push(Allocation::buffer("v41.measured.0.1",
            Location::coordinator(0), unit, Lifetime { first: Phase::Ready, last: Phase::Growth },
            Category::Runtime, UNATTRIBUTED_OWNER));
        let adopted = refit_pool(&mut bound, &owners, 0).unwrap();
        assert_eq!((adopted.pages, adopted.groups, adopted.bytes), (9, 9, vec![9 * unit]));
        assert_eq!(bound.pool_tokens, 9 * 512);
        // The pool's facts are the components at the adopted page count, so the
        // runtime allocator and the plan still charge the same bytes.
        let kv = bound.facts.allocations.iter()
            .filter(|owner| owner.category == Category::Kv).map(|owner| owner.bytes).sum::<u64>();
        assert_eq!(kv, 9 * unit);
        assert!(bound.phases.iter().all(|row| row.owned + row.opaque <= row.available),
            "{:?}", bound.phases);

        // A search whose final probe is the rejected page count still ends its
        // facts at the adopted count (15 MiB: 8 pages fit, 9 do not).
        let (mut tight, owners, unit) = plan(12, 15 * (1 << 20));
        tight.facts.allocations.push(Allocation::buffer("v41.measured.0.0",
            Location::coordinator(0), 2 * unit, Lifetime { first: Phase::Ready, last: Phase::Growth },
            Category::Runtime, BACKING_OWNER));
        tight.facts.allocations.push(Allocation::buffer("v41.measured.0.1",
            Location::coordinator(0), unit, Lifetime { first: Phase::Ready, last: Phase::Growth },
            Category::Runtime, UNATTRIBUTED_OWNER));
        let adopted = refit_pool(&mut tight, &owners, 0).unwrap();
        assert_eq!((adopted.pages, adopted.bytes), (8, vec![8 * unit]));
        let kv = tight.facts.allocations.iter()
            .filter(|owner| owner.category == Category::Kv).map(|owner| owner.bytes).sum::<u64>();
        assert_eq!(kv, 8 * unit, "facts end at the adopted page count, not the last probe");

        // A pool with room to spare keeps every page.
        let (mut slack, owners, _) = plan(12, 24 * (1 << 20));
        assert_eq!(refit_pool(&mut slack, &owners, 0).unwrap().pages, 12);
    }
}
