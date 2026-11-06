//! Pure, phase-aware allocation planning. Recipes describe one legal candidate;
//! policy, exact pins and allocation reconciliation belong here, not in families.
use super::{LayerMode, ResidentWeights};
use cuteafd_core::{spark_reply::SparkReplyFormats, memory_layout::{Basis, Category, Item}};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Load,
    PostTargetKv,
    DraftLoad,
    Warmup,
    Ready,
    Growth,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Load => "checkpoint loading",
            Self::PostTargetKv => "target KV allocation",
            Self::DraftLoad => "draft loading",
            Self::Warmup => "decode warm-up",
            Self::Ready => "ready",
            Self::Growth => "post-ready growth",
        })
    }
}

/// A family's real allocation chronology, not the declaration order of Phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "[Phase; 6]", into = "[Phase; 6]")]
pub struct PhaseOrder([Phase; 6]);
impl PhaseOrder {
    pub const TARGET_THEN_DRAFT: Self = Self([
        Phase::Load, Phase::PostTargetKv, Phase::DraftLoad,
        Phase::Warmup, Phase::Ready, Phase::Growth,
    ]);
    pub const DRAFT_THEN_TARGET: Self = Self([
        Phase::Load, Phase::DraftLoad, Phase::PostTargetKv,
        Phase::Warmup, Phase::Ready, Phase::Growth,
    ]);

    pub fn phases(&self) -> &[Phase; 6] { &self.0 }

    pub fn at_or_before(self, first: Phase, last: Phase) -> bool {
        self.index(first) <= self.index(last)
    }

    fn index(self, phase: Phase) -> usize {
        self.0.iter().position(|&at| at == phase).expect("validated phase order")
    }
}
impl TryFrom<[Phase; 6]> for PhaseOrder {
    type Error = Refusal;
    fn try_from(phases: [Phase; 6]) -> Result<Self, Refusal> {
        if phases != Self::TARGET_THEN_DRAFT.0 && phases != Self::DRAFT_THEN_TARGET.0 {
            return Err(Refusal::InvalidFact("invalid startup phase order".into()));
        }
        Ok(Self(phases))
    }
}
impl From<PhaseOrder> for [Phase; 6] {
    fn from(order: PhaseOrder) -> Self { order.0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lifetime {
    pub first: Phase,
    pub last: Phase,
}
impl Lifetime {
    pub const RESIDENT: Self = Self {
        first: Phase::Load,
        last: Phase::Growth,
    };
    pub const TARGET: Self = Self {
        first: Phase::PostTargetKv,
        last: Phase::Growth,
    };
    pub fn contains(self, phase: Phase, order: PhaseOrder) -> bool {
        order.at_or_before(self.first, phase) && order.at_or_before(phase, self.last)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Device {
    Coordinator(u8),
    Spark(u8),
    Host,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Domain {
    Device,
    Host,
    Pinned,
    Managed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Location {
    pub device: Device,
    pub domain: Domain,
}
impl Location {
    pub fn coordinator(index: u8) -> Self {
        Self {
            device: Device::Coordinator(index),
            domain: Domain::Device,
        }
    }
    pub const HOST: Self = Self {
        device: Device::Host,
        domain: Domain::Host,
    };
    pub const PINNED: Self = Self {
        device: Device::Host,
        domain: Domain::Pinned,
    };
}

/// Logical coordinator ranks in launch order, independent of CUDA ordinals.
#[derive(Debug, Clone, Copy)]
pub struct CoordinatorDevices<'a> {
    physical: &'a [i32],
}
impl<'a> CoordinatorDevices<'a> {
    pub fn new(physical: &'a [i32]) -> Result<Self, Refusal> {
        if !(1..=2).contains(&physical.len())
            || physical.iter().any(|&device| device < 0)
            || physical.iter().collect::<BTreeSet<_>>().len() != physical.len()
        {
            return Err(Refusal::InvalidFact("distinct physical devices required for coordinator ranks".into()));
        }
        Ok(Self { physical })
    }

    pub fn logical_rank(self, physical: i32) -> Result<u8, Refusal> {
        self.physical.iter().position(|&device| device == physical)
            .map(|rank| rank as u8)
            .ok_or_else(|| Refusal::InvalidFact(format!("physical device {physical} is outside coordinator ranks")))
    }

    pub fn physical_device(self, rank: u8) -> Result<i32, Refusal> {
        self.physical.get(usize::from(rank)).copied()
            .ok_or_else(|| Refusal::InvalidFact(format!("unknown coordinator rank {rank}")))
    }

    pub fn location(self, physical: i32) -> Result<Location, Refusal> {
        Ok(Location::coordinator(self.logical_rank(physical)?))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    Serving,
    Scoring,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pin<T> {
    Auto,
    Exact(T),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillShape {
    pub rows: u64,
    pub waves: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub context: u64,
    pub pool: Pin<u64>,
    pub onboard: Pin<u64>,
    pub prefill: Pin<PrefillShape>,
    pub operation: Operation,
    /// Retained checkpoint weights per coordinator, independent of memory budgets.
    pub resident_weight_limit: Option<ResidentWeights>,
    /// Each explicit phase format must match the immutable family contract.
    pub spark_reply_formats: SparkReplyFormats,
}

/// The physical witness stays independent of a logical memory ceiling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Baseline {
    Forecast {
        occupied: u64,
        source: String,
        assumed: bool,
    },
    Sampled {
        free: u64,
        phase: Phase,
        source: String,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub id: String,
    /// Domains sharing physical RAM (e.g. GB10 host, pinned and device) appear
    /// in the same budget. A location must belong to exactly one budget.
    pub locations: Vec<Location>,
    pub physical_bytes: u64,
    pub logical_bytes: u64,
    pub baseline: Baseline,
    pub headroom: u64,
}
impl Budget {
    pub fn available(&self) -> Result<u64, Refusal> {
        let occupied = match &self.baseline {
            Baseline::Forecast { occupied, .. } => *occupied,
            Baseline::Sampled { free, .. } => {
                self.physical_bytes.checked_sub(*free).ok_or_else(|| {
                    Refusal::InvalidFact(format!("{} sample exceeds physical capacity", self.id))
                })?
            }
        };
        self.physical_bytes
            .min(self.logical_bytes)
            .checked_sub(occupied)
            .and_then(|n| n.checked_sub(self.headroom))
            .ok_or_else(|| Refusal::Capacity {
                budget: self.id.clone(),
                phase: Phase::Load,
                required: occupied.saturating_add(self.headroom),
                available: self.physical_bytes.min(self.logical_bytes),
                owners: Vec::new(),
            })
    }
}

/// Physical expert-weight backing; loading source does not affect payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightArenaTopology {
    PerLayer,
    SharedRank,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct View {
    pub backing: String,
    pub offset: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allocation {
    pub id: String,
    pub location: Location,
    pub bytes: u64,
    pub alignment: u64,
    /// Raw allocations are byte arrays; tensor views may give a native shape.
    pub shape: Vec<u64>,
    pub dtype: String,
    pub lifetime: Lifetime,
    pub slot: Option<u32>,
    pub view: Option<View>,
    pub category: Category,
    pub group: String,
}
impl Allocation {
    pub fn buffer(
        id: impl Into<String>,
        location: Location,
        bytes: u64,
        lifetime: Lifetime,
        category: Category,
        group: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            location,
            bytes,
            alignment: 1,
            shape: vec![bytes],
            dtype: "bytes".into(),
            lifetime,
            slot: None,
            view: None,
            category,
            group: group.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allowance {
    pub location: Location,
    pub bytes: u64,
    pub lifetime: Lifetime,
    pub source: String,
    pub assumed: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingHome {
    Device(u8),
    HostPinned,
}
impl EmbeddingHome {
    pub fn location(self) -> Location {
        match self {
            Self::Device(gpu) => Location::coordinator(gpu),
            Self::HostPinned => Location::PINNED,
        }
    }
}

/// Routed layers only. Local homes follow an ordered onboarding profile;
/// remote homes name the actual worker ranks, not an unspecified Spark pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertDestination {
    Coordinator { gpus: Vec<u8> },
    Sparks { ranks: Vec<u8> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decisions {
    pub attention: Vec<LayerMode>,
    pub onboard_layers: Vec<Vec<usize>>,
    pub expert_homes: BTreeMap<usize, ExpertDestination>,
    pub embedding: Option<EmbeddingHome>,
    pub draft_home: Option<u8>,
    pub encoder: Option<usize>,
    pub prefill: PrefillShape,
    pub lanes: u64,
    /// Resolved at preparation; None means this contract has no Spark link.
    pub spark_reply_formats: Option<SparkReplyFormats>,
}
impl Decisions {
    fn onboard(&self) -> u64 {
        self.onboard_layers
            .iter()
            .flatten()
            .collect::<BTreeSet<_>>()
            .len() as u64
    }
}
/// Exact short-lived owners that never coexist. The phase peak is their
/// maximum, but reconciliation still requires every allocation/free edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sequential {
    pub id: String,
    pub members: Vec<Allocation>,
}
impl Sequential {
    pub fn peak(&self) -> u64 {
        self.members.iter().map(|a| a.bytes).max().unwrap_or(0)
    }

    fn validate(&self, order: PhaseOrder) -> Result<(), Refusal> {
        let Some(first) = self.members.first() else {
            return Err(Refusal::InvalidFact(
                "empty sequential allocation group".into(),
            ));
        };
        let mut ids = BTreeSet::new();
        if self.id.is_empty()
            || first.lifetime.first != first.lifetime.last
            || order.at_or_before(Phase::Ready, first.lifetime.last)
            || self.members.iter().any(|a| {
                a.id.is_empty()
                    || !ids.insert(a.id.as_str())
                    || a.bytes == 0
                    || a.alignment == 0
                    || !a.alignment.is_power_of_two()
                    || a.location != first.location
                    || a.lifetime != first.lifetime
                    || a.view.is_some()
            })
        {
            return Err(Refusal::InvalidFact(format!(
                "invalid sequential group {}",
                self.id
            )));
        }
        Ok(())
    }
}

/// A delivery-order-dependent load pool, not a fictitious physical backing.
/// Forecasts charge the proven bound; startup records every actual allocation
/// and requires its independently replayed peak <= bound. All other owner
/// extents remain exact. This exception must never become a total-byte equality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reservation {
    pub id: String,
    pub location: Location,
    pub bytes: u64,
    pub lifetime: Lifetime,
    pub source: String,
}

/// One retained physical weight owner, not an allocation or a baseline delta.
/// Aliases use their backing owner's fact; already-loaded owners still appear.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidentWeight {
    pub id: String,
    pub location: Location,
    pub weights: ResidentWeights,
}
impl ResidentWeight {
    pub fn from_allocation(allocation: &Allocation, weights: ResidentWeights) -> Self {
        Self { id: allocation.id.clone(), location: allocation.location, weights }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Facts {
    pub phase_order: PhaseOrder,
    pub resident_weights: Vec<ResidentWeight>,
    pub allocations: Vec<Allocation>,
    /// Reserved by a later owner port; never evidence that it was allocated.
    pub deferred: Vec<Allocation>,
    pub sequential: Vec<Sequential>,
    pub reservations: Vec<Reservation>,
    pub allowances: Vec<Allowance>,
}

mod contract;
mod search;
pub mod serving;
pub use contract::{
    Activation, ByteCurve, BytePiece, Choice, Component, ComponentKind, Contract, Requires,
    Variable,
};
pub use search::solve;

/// Prepare immutable geometry/cost terms once. A family unable to describe its
/// demand with finite piecewise curves refuses preparation, never searches it.
pub trait Recipe {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseBudget {
    pub budget: String,
    pub phase: Phase,
    pub owned: u64,
    pub opaque: u64,
    pub available: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPlan {
    /// Rejected earlier encoder choices at the admitted pool and onboarding.
    pub encoder_diagnostics: Vec<(String, u64)>,
    pub request: Request,
    pub decisions: Decisions,
    pub pool_tokens: u64,
    pub budgets: Vec<Budget>,
    pub facts: Facts,
    pub phases: Vec<PhaseBudget>,
}
impl ResolvedPlan {
    pub fn items(&self, location: Location) -> Result<Vec<Item>, Refusal> {
        let mut groups = BTreeMap::new();
        for allocation in &self.facts.allocations {
            if allocation.location != location
                || allocation.view.is_some()
                || !allocation.lifetime.contains(Phase::Ready, self.facts.phase_order)
            {
                continue;
            }
            let total = groups
                .entry((allocation.group.clone(), allocation.category))
                .or_insert(0u64);
            *total = total
                .checked_add(allocation.bytes)
                .ok_or(Refusal::Overflow("layout items"))?;
        }
        let mut items: Vec<_> = groups
            .into_iter()
            .map(|((group, category), bytes)| Item::new(category, group, "", bytes, Basis::Exact))
            .collect();
        items.extend(self.facts.allowances.iter().filter(|allowance|
            allowance.location == location
                && allowance.bytes != 0
                && allowance.lifetime.contains(Phase::Ready, self.facts.phase_order))
            .map(|allowance| Item::new(Category::Runtime, &allowance.source, "", allowance.bytes,
                if allowance.assumed { Basis::Estimated } else { Basis::Calibrated })));
        Ok(items)
    }
    pub fn remaining(&self, budget: &str, phase: Phase) -> Option<u64> {
        self.phases
            .iter()
            .find(|row| row.budget == budget && row.phase == phase)
            .map(|row| row.available - row.owned - row.opaque)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Refusal {
    #[error("invalid planner fact: {0}")]
    InvalidFact(String),
    #[error("invalid exact pin: {0}")]
    InvalidPin(&'static str),
    #[error("pool pin {tokens} is not aligned to {page_rows} rows")]
    MisalignedPool { tokens: u64, page_rows: u64 },
    #[error("pool {tokens} cannot hold selected context {context}; pass a lower explicit context no greater than {tokens} tokens")]
    PoolBelowContext { tokens: u64, context: u64 },
    #[error("selected context {context} cannot fit: maximum feasible pool is {tokens} tokens; limiting constraint: {limiting}; pass a lower explicit context no greater than {tokens} tokens (or add memory if zero)")]
    FullContextShortfall { context: u64, tokens: u64, limiting: Box<Refusal> },
    #[error("no supported candidate satisfies the exact pins")]
    PinConflict,
    #[error("no supported candidate")]
    Unsupported,
    #[error("missing selected export {export} for {geometry} at context {context}: {detail}")]
    MissingExport {
        export: String,
        geometry: String,
        context: u64,
        detail: String,
    },
    #[error("memory budget '{budget}' during {phase} needs {required} bytes, has {available} bytes available; owners {owners:?}")]
    Capacity {
        budget: String,
        phase: Phase,
        required: u64,
        available: u64,
        owners: Vec<String>,
    },
    #[error("rtx{gpu} resident weights need {required} bytes, over the {limit} byte weight budget")]
    ResidentWeightLimit { gpu: u8, required: u64, limit: u64 },
    #[error("planner arithmetic overflow: {0}")]
    Overflow(&'static str),
    #[error("allocation reconciliation: {0:?}")]
    Reconciliation(Vec<Mismatch>),
}

fn validate(facts: &Facts, budgets: &[Budget]) -> Result<(), Refusal> {
    let mut locations = BTreeSet::new();
    let mut names = BTreeSet::new();
    for budget in budgets {
        if budget.id.is_empty()
            || budget.locations.is_empty()
            || !names.insert(&budget.id)
            || match &budget.baseline {
                Baseline::Forecast { source, .. } => source.is_empty(),
                Baseline::Sampled { phase, source, .. } => {
                    *phase != Phase::Load || source.is_empty()
                }
            }
        {
            return Err(Refusal::InvalidFact(
                "empty or duplicate memory budget".into(),
            ));
        }
        budget.available()?;
        for location in &budget.locations {
            if !locations.insert(*location) {
                return Err(Refusal::InvalidFact(
                    "location charged to multiple budgets".into(),
                ));
            }
        }
    }
    validate_owners(facts)?;
    if facts
        .allocations
        .iter()
        .chain(&facts.deferred)
        .chain(facts.sequential.iter().flat_map(|g| &g.members))
        .any(|a| !locations.contains(&a.location))
        || facts
            .allowances
            .iter()
            .any(|a| !locations.contains(&a.location))
        || facts.resident_weights.iter().any(|w| !locations.contains(&w.location))
        || facts.reservations.iter().any(|r| !locations.contains(&r.location))
    {
        return Err(Refusal::InvalidFact(
            "allocation outside memory budgets".into(),
        ));
    }
    Ok(())
}

fn validate_owners(facts: &Facts) -> Result<(), Refusal> {
    let order = facts.phase_order;
    let mut groups = BTreeSet::new();
    for group in &facts.sequential {
        group.validate(order)?;
        if !groups.insert(group.id.as_str()) {
            return Err(Refusal::InvalidFact(format!(
                "duplicate sequential group {}",
                group.id
            )));
        }
    }
    let mut ids = BTreeMap::new();
    for allocation in facts
        .allocations
        .iter()
        .chain(&facts.deferred)
        .chain(facts.sequential.iter().flat_map(|g| &g.members))
    {
        if allocation.id.is_empty()
            || ids.insert(allocation.id.clone(), allocation).is_some()
            || allocation.bytes == 0
            || allocation.alignment == 0
            || !allocation.alignment.is_power_of_two()
            || !order.at_or_before(allocation.lifetime.first, allocation.lifetime.last)
        {
            return Err(Refusal::InvalidFact(format!(
                "invalid allocation {}",
                allocation.id
            )));
        }
    }
    for allocation in facts.allocations.iter().chain(&facts.deferred) {
        if let Some(view) = &allocation.view {
            let backing = ids
                .get(&view.backing)
                .ok_or_else(|| Refusal::InvalidFact(format!("missing backing {}", view.backing)))?;
            if facts
                .sequential
                .iter()
                .flat_map(|g| &g.members)
                .any(|a| a.id == view.backing)
                || backing.view.is_some()
                || backing.location != allocation.location
                || view
                    .offset
                    .checked_add(allocation.bytes)
                    .is_none_or(|end| end > backing.bytes)
                || !order.at_or_before(backing.lifetime.first, allocation.lifetime.first)
                || !order.at_or_before(allocation.lifetime.last, backing.lifetime.last)
            {
                return Err(Refusal::InvalidFact(format!(
                    "invalid backing view {}",
                    allocation.id
                )));
            }
        }
    }
    let mut weight_ids = BTreeSet::new();
    for owner in &facts.resident_weights {
        if owner.id.is_empty() || !weight_ids.insert(&owner.id)
            || !matches!(owner.location,
                Location { device: Device::Coordinator(_) | Device::Spark(_), domain: Domain::Device | Domain::Managed }
                | Location { device: Device::Host, domain: Domain::Host | Domain::Pinned })
            || ids.get(&owner.id).is_some_and(|allocation|
                allocation.view.is_some() || allocation.location != owner.location
                    || owner.weights.0 > allocation.bytes)
            || facts.sequential.iter().flat_map(|group| &group.members).any(|a| a.id == owner.id)
        {
            return Err(Refusal::InvalidFact(format!("invalid resident weight {}", owner.id)));
        }
    }
    let mut reservation_ids = BTreeSet::new();
    for reservation in &facts.reservations {
        if reservation.id.is_empty() || !reservation_ids.insert(&reservation.id)
            || ids.contains_key(&reservation.id) || groups.contains(reservation.id.as_str())
            || reservation.bytes == 0 || reservation.source.is_empty()
            || reservation.lifetime.first != reservation.lifetime.last
            || order.at_or_before(Phase::Ready, reservation.lifetime.last)
            || reservation.location.domain != Domain::Pinned
        {
            return Err(Refusal::InvalidFact(format!("invalid load pool reservation {}", reservation.id)));
        }
    }
    for allowance in &facts.allowances {
        if allowance.source.is_empty() || !order.at_or_before(allowance.lifetime.first, allowance.lifetime.last) {
            return Err(Refusal::InvalidFact("invalid opaque allowance".into()));
        }
    }
    Ok(())
}

fn resident_weight_totals(facts: &Facts, gpus: usize) -> Result<Vec<ResidentWeights>, Refusal> {
    let mut totals = vec![ResidentWeights::default(); gpus];
    for owner in &facts.resident_weights {
        if let Device::Coordinator(gpu) = owner.location.device {
            let total = totals.get_mut(usize::from(gpu))
                .ok_or_else(|| Refusal::InvalidFact(format!("unknown resident weight coordinator {gpu}")))?;
            total.0 = total.0.checked_add(owner.weights.0)
                .ok_or(Refusal::Overflow("resident weights"))?;
        }
    }
    Ok(totals)
}

pub fn evaluate(facts: &Facts, budgets: &[Budget]) -> Result<Vec<PhaseBudget>, Refusal> {
    validate(facts, budgets)?;
    let mut rows = Vec::new();
    for budget in budgets {
        let available = budget.available()?;
        for &phase in facts.phase_order.phases() {
            let owners: Vec<_> = facts
                .allocations
                .iter()
                .chain(&facts.deferred)
                .filter(|a| {
                    a.view.is_none()
                        && a.lifetime.contains(phase, facts.phase_order)
                        && budget.locations.contains(&a.location)
                })
                .collect();
            let mut owned = owners.iter().try_fold(0u64, |n, a| {
                n.checked_add(a.bytes)
                    .ok_or(Refusal::Overflow("phase owners"))
            })?;
            for group in &facts.sequential {
                let member = &group.members[0];
                if member.lifetime.contains(phase, facts.phase_order) && budget.locations.contains(&member.location) {
                    owned = owned
                        .checked_add(group.peak())
                        .ok_or(Refusal::Overflow("sequential phase peak"))?;
                }
            }
            for reservation in &facts.reservations {
                if reservation.lifetime.contains(phase, facts.phase_order)
                    && budget.locations.contains(&reservation.location)
                {
                    owned = owned.checked_add(reservation.bytes)
                        .ok_or(Refusal::Overflow("phase reservations"))?;
                }
            }
            let opaque = facts
                .allowances
                .iter()
                .filter(|a| a.lifetime.contains(phase, facts.phase_order) && budget.locations.contains(&a.location))
                .try_fold(0u64, |n, a| {
                    n.checked_add(a.bytes)
                        .ok_or(Refusal::Overflow("phase allowances"))
                })?;
            let required = owned
                .checked_add(opaque)
                .ok_or(Refusal::Overflow("phase peak"))?;
            if required > available {
                return Err(Refusal::Capacity {
                    budget: budget.id.clone(),
                    phase,
                    required,
                    available,
                    owners: owners
                        .iter()
                        .map(|a| a.id.clone())
                        .chain(
                            facts
                                .sequential
                                .iter()
                                .filter(|g| {
                                    g.members[0].lifetime.contains(phase, facts.phase_order)
                                        && budget.locations.contains(&g.members[0].location)
                                })
                                .map(|g| g.id.clone()),
                        )
                        .chain(facts.reservations.iter()
                            .filter(|r| r.lifetime.contains(phase, facts.phase_order)
                                && budget.locations.contains(&r.location))
                            .map(|r| r.id.clone()))
                        .collect(),
                });
            }
            rows.push(PhaseBudget {
                budget: budget.id.clone(),
                phase,
                owned,
                opaque,
                available,
            });
        }
    }
    Ok(rows)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mismatch {
    Missing(String),
    Extra(String),
    Duplicate(String),
    Contract(String),
    Address(String),
    Lifetime(String),
}
#[derive(Debug, Clone)]
pub struct Receipt {
    pub allocation: Allocation,
    pub address: u64,
}
#[derive(Debug, Clone)]
pub struct Observation {
    pub receipt: Receipt,
    pub created: Phase,
    pub freed: Option<Phase>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservationReceipt {
    pub reservation: Reservation,
    pub peak: u64,
}

#[derive(Debug, Clone)]
pub enum OwnerEvent {
    Allocated { receipt: Receipt, phase: Phase },
    Freed { receipt: Receipt, phase: Phase },
    /// Actual physical allocation/free edges inside one bounded load pool.
    /// These are not charged again alongside the pool's reservation.
    Reserved { reservation: String, receipt: Receipt, phase: Phase, allocate: bool },
}

/// Startup observations carry actual creation/free phases, not merely the
/// planned lifetime copied onto a live owner's receipt. Sequential sources
/// require the ordered physical event stream instead.
pub fn reconcile_history(
    plan: &ResolvedPlan,
    phase: Phase,
    observations: &[Observation],
) -> Result<(), Refusal> {
    validate(&plan.facts, &plan.budgets)?;
    reconcile_observations(&plan.facts, phase, observations, false)
}

/// Replay every physical owner in allocation order, including freed packing
/// sources. Views are not allocations and are supplied separately.
pub fn reconcile_events(
    plan: &ResolvedPlan,
    phase: Phase,
    events: &[OwnerEvent],
    views: &[Observation],
) -> Result<(), Refusal> {
    reconcile_startup(plan, phase, events, views, &[])
}

/// Exact owner replay plus the sole bounded-pool exception: the actual replayed
/// peak must equal its independent census and stay at or below the reservation.
pub fn reconcile_startup(
    plan: &ResolvedPlan,
    phase: Phase,
    events: &[OwnerEvent],
    views: &[Observation],
    reservations: &[ReservationReceipt],
) -> Result<(), Refusal> {
    validate(&plan.facts, &plan.budgets)?;
    reconcile_owner_events(&plan.facts, phase, events, views, reservations)
}

/// Temporary owner-only bridge for the legacy Kimi MLA drafter. This proves
/// ownership/history, not capacity or Ready admission. Full Kimi migration
/// deletes this entry point and uses `reconcile_events` on its resolved plan.
pub fn reconcile_legacy_mla_events(
    facts: &Facts,
    phase: Phase,
    events: &[OwnerEvent],
) -> Result<(), Refusal> {
    if !facts.phase_order.at_or_before(phase, Phase::Warmup) || !facts.deferred.is_empty()
        || !facts.allowances.is_empty() || !facts.reservations.is_empty() {
        return Err(Refusal::InvalidFact(
            "legacy MLA bridge only checks startup owners".into(),
        ));
    }
    reconcile_owner_events(facts, phase, events, &[], &[])
}

fn reconcile_owner_events(
    facts: &Facts,
    phase: Phase,
    events: &[OwnerEvent],
    views: &[Observation],
    reservations: &[ReservationReceipt],
) -> Result<(), Refusal> {
    validate_owners(facts)?;
    let mut expected = facts
        .allocations
        .iter()
        .filter(|a| a.view.is_none())
        .map(|a| (a.id.as_str(), (None, a)))
        .collect::<BTreeMap<_, _>>();
    expected.extend(facts.sequential.iter().enumerate().flat_map(|(group, g)| {
        g.members
            .iter()
            .map(move |a| (a.id.as_str(), (Some(group), a)))
    }));
    let mut errors = Vec::new();
    let mut seen = BTreeSet::new();
    let mut active = BTreeMap::<&str, &Receipt>::new();
    let mut groups = BTreeMap::<usize, &str>::new();
    let mut observations = BTreeMap::<&str, Observation>::new();
    let mut previous = Phase::Load;
    let mut held = BTreeMap::<&str, u64>::new();
    let mut peaks = BTreeMap::<&str, u64>::new();
    let mut reserved_live = BTreeMap::<&str, (&str, &Receipt)>::new();
    for event in events {
        let (receipt, at, allocate, reservation) = match event {
            OwnerEvent::Allocated { receipt, phase } => (receipt, *phase, true, None),
            OwnerEvent::Freed { receipt, phase } => (receipt, *phase, false, None),
            OwnerEvent::Reserved { reservation, receipt, phase, allocate } =>
                (receipt, *phase, *allocate, Some(reservation.as_str())),
        };
        let id = receipt.allocation.id.as_str();
        if !facts.phase_order.at_or_before(previous, at) || !facts.phase_order.at_or_before(at, phase) {
            errors.push(Mismatch::Lifetime(id.into()));
        }
        previous = at;
        let reserved = reservation.and_then(|id| facts.reservations.iter().find(|r| r.id == id));
        let (group, spec) = if let Some(reservation) = reservation {
            let Some(bound) = reserved else {
                errors.push(Mismatch::Extra(reservation.into()));
                continue;
            };
            let a = &receipt.allocation;
            if expected.contains_key(id) || facts.deferred.iter().any(|a| a.id == id)
                || facts.reservations.iter().any(|r| r.id == id)
                || a.id.is_empty() || a.bytes == 0 || a.view.is_some() || a.slot.is_some()
                || a.location != bound.location || a.lifetime != bound.lifetime
                || a.alignment == 0 || !a.alignment.is_power_of_two()
                || a.dtype != "bytes" || a.shape != [a.bytes]
            {
                errors.push(Mismatch::Contract(id.into()));
            }
            if allocate {
                if held.get(reservation).copied().unwrap_or(0) > bound.bytes.saturating_sub(a.bytes)
                    || a.bytes > bound.bytes
                {
                    errors.push(Mismatch::Contract(reservation.into()));
                }
                let current = held.entry(reservation).or_default();
                *current = current.checked_add(a.bytes).ok_or(Refusal::Overflow("reserved pool held"))?;
                let peak = peaks.entry(reservation).or_default();
                *peak = (*peak).max(*current);
                if reserved_live.insert(id, (reservation, receipt)).is_some() {
                    errors.push(Mismatch::Duplicate(id.into()));
                }
            } else {
                match reserved_live.remove(id) {
                    Some((owner, live)) if owner == reservation && live.allocation == *a
                        && live.address == receipt.address => {
                        let current = held.entry(reservation).or_default();
                        *current = current.checked_sub(a.bytes).ok_or(Refusal::Overflow("reserved pool free"))?;
                    }
                    _ => errors.push(Mismatch::Lifetime(id.into())),
                }
            }
            (None, a)
        } else {
            let Some(&(group, spec)) = expected.get(id) else {
                errors.push(Mismatch::Extra(id.into()));
                continue;
            };
            (group, spec)
        };
        if receipt.allocation != *spec {
            errors.push(Mismatch::Contract(id.into()));
        }
        if at
            != if allocate {
                spec.lifetime.first
            } else {
                spec.lifetime.last
            }
        {
            errors.push(Mismatch::Lifetime(id.into()));
        }
        if allocate {
            if !seen.insert(id) {
                errors.push(Mismatch::Duplicate(id.into()));
            }
            let end = receipt.address.checked_add(receipt.allocation.bytes);
            if receipt.address == 0 || spec.alignment == 0 || receipt.address % spec.alignment != 0 || end.is_none() {
                errors.push(Mismatch::Address(id.into()));
            }
            for live in active.values() {
                if live.allocation.location == receipt.allocation.location
                    && end
                        .zip(live.address.checked_add(live.allocation.bytes))
                        .is_none_or(|(end, live_end)| {
                            live.address < end && receipt.address < live_end
                        })
                {
                    errors.push(Mismatch::Address(format!("{} / {id}", live.allocation.id)));
                }
            }
            if let Some(group) = group {
                if groups.insert(group, id).is_some() {
                    errors.push(Mismatch::Lifetime(facts.sequential[group].id.clone()));
                }
            } else if reservation.is_none() {
                observations.entry(id).or_insert_with(|| Observation {
                    receipt: receipt.clone(),
                    created: at,
                    freed: None,
                });
            }
            if active.insert(id, receipt).is_some() {
                errors.push(Mismatch::Duplicate(id.into()));
            }
        } else {
            match active.remove(id) {
                Some(live)
                    if live.allocation == receipt.allocation && live.address == receipt.address => {
                }
                _ => errors.push(Mismatch::Lifetime(id.into())),
            }
            if let Some(group) = group {
                if groups.remove(&group) != Some(id) {
                    errors.push(Mismatch::Lifetime(id.into()));
                }
            } else if let Some(observation) = observations.get_mut(id) {
                observation.freed = Some(at);
            }
        }
    }
    let mut reported = BTreeSet::new();
    for actual in reservations {
        let id = &actual.reservation.id;
        if !reported.insert(id.as_str()) { errors.push(Mismatch::Duplicate(id.clone())); }
        match facts.reservations.iter().find(|r| &r.id == id) {
            Some(spec) if *spec == actual.reservation
                && facts.phase_order.at_or_before(spec.lifetime.first, phase)
                && actual.peak <= spec.bytes
                && actual.peak == peaks.get(id.as_str()).copied().unwrap_or(0) => {}
            Some(_) => errors.push(Mismatch::Contract(id.clone())),
            None => errors.push(Mismatch::Extra(id.clone())),
        }
    }
    for spec in &facts.reservations {
        if facts.phase_order.at_or_before(spec.lifetime.first, phase) && !reported.contains(spec.id.as_str()) {
            errors.push(Mismatch::Missing(spec.id.clone()));
        }
    }
    for (id, (_, receipt)) in reserved_live {
        if !facts.phase_order.at_or_before(phase, receipt.allocation.lifetime.last) {
            errors.push(Mismatch::Lifetime(id.into()));
        }
    }
    for group in &facts.sequential {
        for spec in &group.members {
            if facts.phase_order.at_or_before(spec.lifetime.first, phase) && !seen.contains(spec.id.as_str()) {
                errors.push(Mismatch::Missing(spec.id.clone()));
            }
            if active.contains_key(spec.id.as_str()) {
                errors.push(Mismatch::Lifetime(spec.id.clone()));
            }
        }
    }
    for view in views {
        if view.receipt.allocation.view.is_none() {
            errors.push(Mismatch::Contract(view.receipt.allocation.id.clone()));
        }
    }
    if !errors.is_empty() {
        return Err(Refusal::Reconciliation(errors));
    }
    let observations = observations
        .into_values()
        .chain(views.iter().cloned())
        .collect::<Vec<_>>();
    reconcile_observations(facts, phase, &observations, true)
}

fn reconcile_observations(
    facts: &Facts,
    phase: Phase,
    observations: &[Observation],
    ordered_proved: bool,
) -> Result<(), Refusal> {
    validate_owners(facts)?;
    let mut seen = BTreeSet::new();
    let mut errors = Vec::new();
    for observation in observations {
        let id = &observation.receipt.allocation.id;
        if !seen.insert(id.as_str()) {
            errors.push(Mismatch::Duplicate(id.clone()));
        }
        match facts.allocations.iter().find(|a| &a.id == id) {
            None => errors.push(Mismatch::Extra(id.clone())),
            Some(spec) => {
                if *spec != observation.receipt.allocation {
                    errors.push(Mismatch::Contract(id.clone()));
                }
                if observation.created != spec.lifetime.first
                    || !facts.phase_order.at_or_before(observation.created, phase)
                    || observation.freed.is_some_and(|freed| {
                        !facts.phase_order.at_or_before(observation.created, freed) || !facts.phase_order.at_or_before(freed, phase) || freed != spec.lifetime.last
                    })
                    || (!facts.phase_order.at_or_before(phase, spec.lifetime.last) && observation.freed.is_none())
                {
                    errors.push(Mismatch::Lifetime(id.clone()));
                }
            }
        }
    }
    for spec in &facts.allocations {
        if facts.phase_order.at_or_before(spec.lifetime.first, phase) && !seen.contains(spec.id.as_str()) {
            errors.push(Mismatch::Missing(spec.id.clone()));
        }
    }
    if !ordered_proved {
        for group in &facts.sequential {
            for spec in &group.members {
                if facts.phase_order.at_or_before(spec.lifetime.first, phase) {
                    errors.push(Mismatch::Missing(spec.id.clone()));
                }
            }
        }
    }
    if !errors.is_empty() {
        return Err(Refusal::Reconciliation(errors));
    }
    for &at in facts.phase_order.phases().iter().take_while(|&&at| facts.phase_order.at_or_before(at, phase)) {
        let receipts = observations
            .iter()
            .filter(|o| facts.phase_order.at_or_before(o.created, at) && o.freed.is_none_or(|freed| facts.phase_order.at_or_before(at, freed)))
            .map(|o| o.receipt.clone())
            .collect::<Vec<_>>();
        reconcile_snapshot(facts, at, &receipts, ordered_proved)?;
    }
    Ok(())
}

/// Observed device use since the before-weights sample must stay within owned
/// allocations plus the explicit opaque bound, never consume reserved headroom.
pub fn reconcile_usage(
    plan: &ResolvedPlan,
    budget: &str,
    phase: Phase,
    consumed: u64,
) -> Result<(), Refusal> {
    let row = plan
        .phases
        .iter()
        .find(|row| row.budget == budget && row.phase == phase)
        .ok_or_else(|| Refusal::InvalidFact("missing usage budget/phase".into()))?;
    let bound = row
        .owned
        .checked_add(row.opaque)
        .ok_or(Refusal::Overflow("usage bound"))?;
    if consumed > bound {
        return Err(Refusal::Capacity {
            budget: budget.into(),
            phase,
            required: consumed,
            available: bound,
            owners: vec!["observed owned plus opaque usage".into()],
        });
    }
    Ok(())
}

/// Receipts describe actual owners. A view address must equal backing+offset;
/// overlapping physical owners and overlapping independent slot views fail.
pub fn reconcile(plan: &ResolvedPlan, phase: Phase, receipts: &[Receipt]) -> Result<(), Refusal> {
    validate(&plan.facts, &plan.budgets)?;
    reconcile_snapshot(&plan.facts, phase, receipts, false)
}

fn reconcile_snapshot(
    facts: &Facts,
    phase: Phase,
    receipts: &[Receipt],
    ordered_proved: bool,
) -> Result<(), Refusal> {
    validate_owners(facts)?;
    let expected: BTreeMap<_, _> = facts
        .allocations
        .iter()
        .filter(|a| a.lifetime.contains(phase, facts.phase_order))
        .map(|a| (a.id.as_str(), a))
        .collect();
    let mut actual = BTreeMap::new();
    let mut errors = Vec::new();
    if facts.phase_order.at_or_before(Phase::Ready, phase) {
        errors.extend(
            facts
                .deferred
                .iter()
                .map(|a| Mismatch::Missing(a.id.clone())),
        );
    }
    if !ordered_proved {
        errors.extend(facts.reservations.iter()
            .filter(|r| facts.phase_order.at_or_before(r.lifetime.first, phase))
            .map(|r| Mismatch::Missing(r.id.clone())));
        errors.extend(
            facts
                .sequential
                .iter()
                .flat_map(|g| &g.members)
                .filter(|a| facts.phase_order.at_or_before(a.lifetime.first, phase))
                .map(|a| Mismatch::Missing(a.id.clone())),
        );
    }
    for receipt in receipts {
        let id = receipt.allocation.id.as_str();
        if actual.insert(id, receipt).is_some() {
            errors.push(Mismatch::Duplicate(id.into()));
        }
        match expected.get(id) {
            None => errors.push(Mismatch::Extra(id.into())),
            Some(spec) if **spec != receipt.allocation => {
                errors.push(Mismatch::Contract(id.into()))
            }
            _ => (),
        }
        if receipt.address == 0
            || receipt
                .address
                .checked_add(receipt.allocation.bytes)
                .is_none()
            || receipt.allocation.alignment == 0
            || receipt.address % receipt.allocation.alignment.max(1) != 0
        {
            errors.push(Mismatch::Address(id.into()));
        }
    }
    for id in expected.keys() {
        if !actual.contains_key(id) {
            errors.push(Mismatch::Missing((*id).into()));
        }
    }
    for receipt in actual.values() {
        let a = &receipt.allocation;
        if let Some(view) = &a.view {
            if actual.get(view.backing.as_str()).is_none_or(|backing| {
                backing.address.checked_add(view.offset) != Some(receipt.address)
            }) {
                errors.push(Mismatch::Address(a.id.clone()));
            }
        }
    }
    let values: Vec<_> = actual.values().collect();
    for (i, a) in values.iter().enumerate() {
        for b in &values[i + 1..] {
            let (left, right) = (&a.allocation, &b.allocation);
            if left.location != right.location {
                continue;
            }
            let overlap = a
                .address
                .checked_add(left.bytes)
                .zip(b.address.checked_add(right.bytes))
                .is_none_or(|(end_a, end_b)| a.address < end_b && b.address < end_a);
            // Ordered replay already proves physical overlap at each edge;
            // inclusive phase snapshots cannot distinguish same-phase reuse.
            let distinct_backings = left.view.is_none() && right.view.is_none();
            let distinct_slots =
                left.view
                    .as_ref()
                    .zip(right.view.as_ref())
                    .is_some_and(|(a, b)| {
                        a.backing == b.backing
                            && left.slot != right.slot
                            && left.slot.is_some()
                            && right.slot.is_some()
                    });
            if overlap && ((distinct_backings && !ordered_proved) || distinct_slots) {
                errors.push(Mismatch::Address(format!("{} / {}", left.id, right.id)));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(Refusal::Reconciliation(errors))
    }
}

#[cfg(test)]
mod tests;
