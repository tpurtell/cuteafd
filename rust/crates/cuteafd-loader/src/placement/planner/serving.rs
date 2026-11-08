//! Physical serving owners shared by family recipes. Quotas and selector
//! placement are resolved inputs; this module never chooses admission policy.
use super::*;

fn startup_graph_source(id: &str) -> String { format!("startup graphs: {id}") }

/// Executable residency is opaque driver storage, not a ledger allocation.
/// Estimates have no variance margin and are replaced after startup capture.
pub fn startup_graph_component(id: &str, location: Location, rank: crate::placement::GraphRank,
    maximum: u64) -> Result<Component, Refusal> {
    if id.is_empty() || maximum == 0 || rank.margin != 0 {
        return Err(Refusal::InvalidFact("invalid startup graph owner".into()));
    }
    Ok(Component { activation: Activation::always(), bytes: ByteCurve::fixed(rank.bytes, maximum),
        kind: ComponentKind::Allowance(Allowance { location, bytes: rank.bytes,
            lifetime: Lifetime { first: Phase::Warmup, last: Phase::Growth },
            source: startup_graph_source(id), assumed: rank.basis != Basis::Calibrated }) })
}

/// Reevaluate the selected facts, never select a different pool or placement.
/// A capture may exceed its estimate whenever the allocation itself succeeded:
/// the graphs are physically resident, so the measured rows are installed and
/// the budget gap is reported, never turned into a refusal (#1605). Only a
/// facts mismatch still refuses.
pub fn reconcile_startup_graphs(plan: &mut ResolvedPlan, prefix: &str,
    estimated: &crate::placement::GraphSet, declared: &[String],
    captured: &[crate::placement::inventory::CapturedGraphRank])
    -> Result<crate::placement::GraphSet, Refusal> {
    let measured = estimated.captured(declared, captured).map_err(Refusal::InvalidFact)?;
    let mut facts = plan.facts.clone();
    for (rank, graph) in measured.ranks.iter().enumerate() {
        let source = startup_graph_source(&format!("{prefix}.{rank}"));
        let matches: Vec<_> = facts.allowances.iter_mut().filter(|owner| owner.source == source).collect();
        let [owner] = matches.try_into().map_err(|_: Vec<_>|
            Refusal::InvalidFact("missing or duplicate startup graph owner".into()))?;
        if owner.location != Location::coordinator(rank as u8)
            || owner.lifetime != (Lifetime { first: Phase::Warmup, last: Phase::Growth }) {
            return Err(Refusal::InvalidFact("startup graph owner location or lifetime differs".into()));
        }
        owner.bytes = graph.bytes;
        owner.assumed = false;
    }
    let assessment = assess(&facts, &plan.budgets)?;
    plan.facts = facts;
    plan.phases = assessment.rows;
    for refusal in &assessment.over_budget {
        let Refusal::Capacity { budget, phase, required, available, owners } = refusal else {
            // `assess` reports capacity vetoes only; anything else is a defect.
            return Err(refusal.clone());
        };
        tracing::warn!(budget = %budget, %phase, required, available, owners = ?owners,
            "startup graph capture succeeded above its admitted estimate ({prefix}); the captured graphs are \
             resident and serving continues, but this budget has no room for the excess: free device memory \
             with GPU_MEMORY_CAP or GPU_RESERVE, or measure this layout so its row replaces the estimate");
    }
    Ok(measured)
}

/// Backend of the checkpoint's target vocabulary projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlasHead { Native, Gemm, TensorLt }

impl BlasHead {
    pub fn bf16(self) -> bool { self != Self::Native }
    pub fn gemm(self) -> bool { self == Self::Gemm }
    pub fn lt(self) -> bool { self == Self::TensorLt }
}

/// A borrower of a thread/device BLAS pool. Private workspaces stay allocations.
#[derive(Debug, Clone)]
pub struct BlasOwners {
    pub name: String,
    /// Retained libraries are process-local; a remote media worker is not the coordinator.
    pub process: String,
    /// Stable execution-thread role, e.g. `glmf/blas/serve` or `glmf/blas/media`.
    pub thread: String,
    pub location: Location,
    pub gemm: bool,
    pub lt: bool,
    pub lifetime: Lifetime,
    pub activation: Activation,
}

/// SM120/driver 13020: 64 MiB/cuBLAS handle, shared 2 MiB initialization;
/// first SGEMM and representative Lt plans add zero (explicit 4 MiB scratch).
/// SM121/driver 13020, LAZY: four warmed handles average 67,508,224 B;
/// destroying the warm handle leaves 158,674,944 B. First Lt planning beside
/// BLAS adds 201,461,760 B, later creates/plans zero. These are separate
/// process-local owners, not a padded per-handle price. Probe: 2026-11-07,
/// BF16 M16/K4096/N4096. SM121 Lt-only processes need a separate qualification.
pub fn blas_components(arch: &str, owners: &[BlasOwners], maximum: u64,
    order: PhaseOrder) -> Result<Vec<Component>, Refusal> {
    let per_handle = match arch {
        "sm_120" => 64 << 20,
        "sm_121" => 67_508_224,
        _ => return Err(Refusal::InvalidFact(format!("unmeasured BLAS owner architecture {arch}"))),
    };
    if maximum == 0 { return Err(Refusal::InvalidFact("empty BLAS owner pool domain".into())); }
    let mut names = BTreeSet::new();
    let mut out = Vec::new();
    for owner in owners {
        let device_matches = match arch {
            "sm_120" => matches!(owner.location, Location { device: Device::Coordinator(_), domain: Domain::Device }),
            _ => matches!(owner.location, Location { device: Device::Spark(_), domain: Domain::Managed }),
        };
        if owner.name.is_empty() || owner.process.is_empty() || owner.thread.is_empty()
            || !names.insert((owner.location, owner.process.as_str(), owner.thread.as_str(), owner.name.as_str()))
            || !device_matches || !order.at_or_before(owner.lifetime.first, owner.lifetime.last)
        {
            return Err(Refusal::InvalidFact("invalid retained BLAS owner".into()));
        }
        if arch == "sm_121" && owner.lt && !owner.gemm {
            return Err(Refusal::InvalidFact("SM121 Lt-only process lacks a qualified first-use owner (#1463)".into()));
        }
    }
    fn union(clauses: impl IntoIterator<Item = Vec<Choice>>) -> BTreeSet<Vec<Choice>> {
        let all: BTreeSet<_> = clauses.into_iter().collect();
        all.iter().filter(|clause| !all.iter().any(|other| other.len() < clause.len()
            && other.iter().all(|choice| clause.contains(choice)))).cloned().collect()
    }
    fn intersect(left: &BTreeSet<Vec<Choice>>, right: &BTreeSet<Vec<Choice>>) -> BTreeSet<Vec<Choice>> {
        union(left.iter().flat_map(|left| right.iter().filter_map(move |right| {
            let mut choices = BTreeMap::new();
            left.iter().chain(right).all(|choice| choices.insert(choice.variable, choice.option)
                .is_none_or(|old| old == choice.option)).then(|| choices.into_iter()
                    .map(|(variable, option)| Choice { variable, option }).collect())
        })))
    }
    for (location, process, thread) in owners.iter().map(|owner|
        (owner.location, owner.process.as_str(), owner.thread.as_str())).collect::<BTreeSet<_>>() {
        let pool: Vec<_> = owners.iter().filter(|owner| owner.location == location
            && owner.process == process && owner.thread == thread).collect();
        let mut retained = BTreeSet::new();
        let mut previous = None;
        for &phase in order.phases() {
            let gemm = union(pool.iter().filter(|owner| owner.gemm && owner.lifetime.contains(phase, order))
                .flat_map(|owner| owner.activation.clauses.iter().cloned()));
            // The native weak pool dies when its final borrower retires. Carry
            // its GEMM backend only through borrowers spanning both phases, not
            // across a gap followed by a new Lt-only pool incarnation.
            let continuity = union(pool.iter().filter(|owner| (owner.gemm || owner.lt)
                && owner.lifetime.contains(phase, order)
                && previous.is_some_and(|before| owner.lifetime.contains(before, order)))
                .flat_map(|owner| owner.activation.clauses.iter().cloned()));
            retained = union(gemm.into_iter().chain(intersect(&retained, &continuity)));
            previous = Some(phase);
            if !retained.is_empty() {
                out.push(Component { activation: Activation { clauses: retained.iter().cloned().collect() },
                    bytes: ByteCurve::fixed(per_handle, maximum),
                    kind: ComponentKind::Allowance(Allowance { location, bytes: per_handle,
                        lifetime: Lifetime { first: phase, last: phase }, assumed: false,
                        source: format!("cuBLAS handles: {thread} (1 x {per_handle} B, {arch} driver13020, 2026-11-07, process {process})") }) });
            }
        }
    }
    let initialization = if arch == "sm_120" {
        vec![("BLAS device initialization (cuBLAS/cuBLASLt shared)", 2 << 20, false)]
    } else {
        vec![("cuBLAS first use", 158_674_944, false), ("cuBLASLt first plan", 201_461_760, true)]
    };
    for (location, process) in owners.iter().map(|owner| (owner.location, owner.process.as_str()))
        .collect::<BTreeSet<_>>() {
        for &(source, bytes, lt_only) in &initialization {
            for &phase in order.phases() {
                // Library residency survives handle destruction. Union selected
                // triggers per process/context; independent workers pay separately.
                let clauses = owners.iter().filter(|owner| owner.location == location && owner.process == process
                    && if lt_only { owner.lt } else { owner.gemm || owner.lt })
                    .filter(|owner| order.at_or_before(owner.lifetime.first, phase))
                    .flat_map(|owner| owner.activation.clauses.iter().cloned()).collect::<BTreeSet<_>>();
                if !clauses.is_empty() {
                    out.push(Component { activation: Activation { clauses: clauses.into_iter().collect() },
                        bytes: ByteCurve::fixed(bytes, maximum),
                        kind: ComponentKind::Allowance(Allowance { location, bytes,
                            lifetime: Lifetime { first: phase, last: phase }, assumed: false,
                            source: format!("{source} ({arch} driver13020, LAZY, 2026-11-07, process {process})") }) });
                }
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
pub struct SelectorOwners {
    pub location: Location,
    pub capacity: usize,
    pub vocab: usize,
    pub device_sampling: bool,
}

impl SelectorOwners {
    pub fn allocations(self, prefix: &str, lifetime: Lifetime) -> Result<Vec<Allocation>, Refusal> {
        if prefix.is_empty() || self.location.domain != Domain::Device
            || self.location.device == Device::Host || self.capacity == 0 || self.vocab == 0
        {
            return Err(Refusal::InvalidFact("invalid token selector geometry or identity".into()));
        }
        let bytes = self.capacity.checked_mul(12)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(Refusal::Overflow("token selector output"))?;
        let mut owners = vec![
            Allocation::buffer(format!("{prefix}.output"), self.location, bytes, lifetime,
                Category::Workspace, "token selector"),
            Allocation::buffer(format!("{prefix}.landing"), Location::PINNED, bytes, lifetime,
                Category::Staging, "token selector"),
        ];
        if self.device_sampling {
            // The retained sampler handles at most 128 rows per wave, even
            // when a selector's independent greedy output plane is larger.
            for owner in cuteafd_core::target_sampler_allocations(self.capacity.min(128), self.vocab)
                .map_err(|e| Refusal::InvalidFact(e.into()))?
            {
                owners.push(Allocation::buffer(format!("{prefix}.sampler.{}", owner.id),
                    if owner.pinned { Location::PINNED } else { self.location },
                    u64::try_from(owner.bytes).map_err(|_| Refusal::Overflow("target sampler owner"))?,
                    lifetime, Category::Workspace, "target sampler"));
            }
        }
        Ok(owners)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HostPrefixOwners {
    pub quota_bytes: u64,
    pub chunk_bytes: u64,
}

impl HostPrefixOwners {
    pub fn allocations(self, prefix: &str, lifetime: Lifetime) -> Result<Vec<Allocation>, Refusal> {
        if prefix.is_empty() {
            return Err(Refusal::InvalidFact("empty host prefix owner identity".into()));
        }
        if self.quota_bytes == 0 { return Ok(vec![]); }
        if self.chunk_bytes == 0 || self.chunk_bytes > self.quota_bytes
            || usize::try_from(self.chunk_bytes).is_err()
            || self.quota_bytes / self.chunk_bytes > u32::MAX as u64
        {
            return Err(Refusal::InvalidFact("invalid host prefix chunks".into()));
        }
        // SlabPool retains only whole chunks; the explicit quota remainder
        // is neither an allocation nor a view into another owner.
        Ok((0..self.quota_bytes / self.chunk_bytes).map(|index|
            Allocation::buffer(format!("{prefix}.{index}"), Location::PINNED, self.chunk_bytes,
                lifetime, Category::Prefix, "retained host prefix chunk")).collect())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ServingOwners {
    pub selector: SelectorOwners,
    pub prefix: HostPrefixOwners,
}

impl ServingOwners {
    pub const LIFETIME: Lifetime = Lifetime { first: Phase::Warmup, last: Phase::Growth };

    pub fn allocations(self) -> Result<Vec<Allocation>, Refusal> {
        let mut owners = self.selector.allocations("selector", Self::LIFETIME)?;
        owners.extend(self.prefix.allocations("host.prefix", Self::LIFETIME)?);
        Ok(owners)
    }

    pub fn components(self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 {
            return Err(Refusal::InvalidFact("serving owners require a positive pool domain".into()));
        }
        Ok(self.allocations()?.into_iter().map(|owner| {
            let bytes = ByteCurve::fixed(owner.bytes, maximum_pages);
            Component::allocation(owner, bytes, false, maximum_pages)
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blas_borrowers_union_by_thread_device_and_backend_lifetime() {
        let order = PhaseOrder::TARGET_THEN_DRAFT;
        let target = BlasOwners { name: "target Lt".into(), process: "coordinator".into(),
            thread: "family/blas/serve".into(), location: Location::coordinator(0),
            gemm: false, lt: true, lifetime: Lifetime::TARGET, activation: Activation::always() };
        let draft = BlasOwners { name: "draft GEMM".into(), gemm: true, lt: false,
            lifetime: Lifetime { first: Phase::Warmup, last: Phase::Ready },
            activation: Activation::choice(0, 1), ..target.clone() };
        let media = BlasOwners { name: "vision".into(), thread: "family/blas/media".into(),
            gemm: true, lt: false, lifetime: Lifetime::RESIDENT, ..target.clone() };
        let audio = BlasOwners { name: "audio".into(), ..media.clone() };
        let owners = [target.clone(), draft, media, audio];
        let empty = Facts { phase_order: order, allocations: vec![], deferred: vec![],
            sequential: vec![], reservations: vec![], allowances: vec![],
            resident_weights: vec![], allocation_granularity: vec![], driver_page_tables: vec![] };
        let mut contract = Contract::fixed(64, 1, 1, empty);
        contract.variables = vec![Variable::Encoder(vec![0, 1]),
            Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]), Variable::Lanes(vec![1])];
        contract.components = blas_components("sm_120", &owners, 1, order).unwrap();
        for selected in [0, 1] {
            let facts = contract.facts(&[selected, 0, 0], 1).unwrap();
            for &phase in order.phases() {
                let active = facts.allowances.iter().filter(|owner| owner.lifetime.contains(phase, order)).collect::<Vec<_>>();
                let draft_started = selected == 1 && order.at_or_before(Phase::Warmup, phase);
                assert_eq!(active.iter().map(|owner| owner.bytes).sum::<u64>(),
                    (1 + u64::from(draft_started)) * (64 << 20) + (2 << 20));
                assert_eq!(active.iter().filter(|owner| owner.source.starts_with("BLAS device initialization")).count(), 1);
            }
            assert!(facts.allocations.is_empty(), "private workspace allocations stay separate");
        }
        assert!(blas_components("sm_90", &owners, 1, order).is_err());
        assert!(blas_components("sm_121", &owners, 1, order).is_err());
        assert!(blas_components("sm_120", &[target.clone(), target], 1, order).is_err());
    }

    #[test]
    fn blas_residency_and_portable_page_tables_charge_disjoint_owners() {
        let order = PhaseOrder::TARGET_THEN_DRAFT;
        let owners = (0..2).map(|gpu| BlasOwners {
            name: format!("target {gpu}"), process: "coordinator".into(),
            thread: "family/blas/serve".into(), location: Location::coordinator(gpu),
            gemm: true, lt: true, lifetime: Lifetime::RESIDENT,
            activation: Activation::always(),
        }).collect::<Vec<_>>();
        let mut pinned = Allocation::buffer("staging", Location::PINNED, 4096,
            Lifetime::RESIDENT, Category::Staging, "host staging");
        let mut allocations = vec![pinned.clone()];
        pinned.id = "staging.alias".into();
        pinned.view = Some(View { backing: "staging".into(), offset: 0 });
        allocations.push(pinned);
        for gpu in 0..2 {
            allocations.push(Allocation::buffer(&format!("blas.scratch.{gpu}"),
                Location::coordinator(gpu), 4 << 20, Lifetime::RESIDENT,
                Category::Workspace, "private BLAS scratch"));
        }
        let facts = Facts { phase_order: order, allocations, deferred: vec![],
            sequential: vec![], reservations: vec![], allowances: vec![],
            resident_weights: vec![], allocation_granularity: vec![], driver_page_tables: vec![] };
        let mut contract = Contract::fixed(64, 1, 2, facts);
        contract.variables = vec![Variable::Prefill(vec![PrefillShape { rows: 64, waves: 1 }]),
            Variable::Lanes(vec![1])];
        contract.components.extend(blas_components("sm_120", &owners, 1, order).unwrap());
        contract.add_driver_page_tables();
        contract.add_driver_page_tables();
        let facts = contract.facts(&[0, 0], 1).unwrap();
        for gpu in 0..2 {
            let location = Location::coordinator(gpu);
            let mappings = facts.driver_page_tables.iter().filter(|owner| owner.location == location)
                .collect::<Vec<_>>();
            assert_eq!(mappings.len(), 1, "only the physical pinned backing registers");
            assert_eq!(mappings[0].backing, "staging");
            assert_eq!(mappings[0].bytes(), 8);
            for &phase in order.phases() {
                assert_eq!(facts.allowances.iter().filter(|owner| owner.location == location
                    && owner.lifetime.contains(phase, order)).map(|owner| owner.bytes).sum::<u64>(),
                    66 << 20, "opaque BLAS residency does not include portable mappings");
            }
        }
    }

    #[test]
    fn blas_backend_retention_does_not_cross_a_retired_pool_or_an_unselected_borrower() {
        let order = PhaseOrder::TARGET_THEN_DRAFT;
        let first = BlasOwners { name: "load GEMM".into(), process: "coordinator".into(),
            thread: "family/blas/serve".into(), location: Location::coordinator(0),
            gemm: true, lt: false, lifetime: Lifetime { first: Phase::Load, last: Phase::Load },
            activation: Activation::always() };
        let bridge = BlasOwners { name: "conditional bridge".into(), gemm: false, lt: true,
            lifetime: Lifetime { first: Phase::Load, last: Phase::Warmup },
            activation: Activation::choice(0, 1), ..first.clone() };
        let second = BlasOwners { name: "target Lt".into(), gemm: false, lt: true,
            lifetime: Lifetime { first: Phase::Warmup, last: Phase::Growth }, ..first.clone() };
        let components = blas_components("sm_120", &[first, bridge, second], 1, order).unwrap();
        for selected in [0, 1] {
            for &phase in order.phases() {
                let handles = components.iter().filter(|component| component.activation.active(&[selected]))
                    .filter_map(|component| match &component.kind {
                        ComponentKind::Allowance(owner) if owner.source.starts_with("cuBLAS handles:")
                            && owner.lifetime.contains(phase, order) => Some(owner.bytes), _ => None,
                    }).sum::<u64>();
                assert_eq!(handles, if selected == 1 || phase == Phase::Load { 64 << 20 } else { 0 });
            }
        }
    }

    #[test]
    fn spark_blas_prices_distinct_threads_but_initializes_each_process_once() {
        let order = PhaseOrder::DRAFT_THEN_TARGET;
        let location = Location { device: Device::Spark(0), domain: Domain::Managed };
        let target = BlasOwners { name: "target".into(), thread: "family/blas/serve".into(),
            process: "worker".into(), location, gemm: true, lt: false,
            lifetime: Lifetime::RESIDENT, activation: Activation::always() };
        let draft = BlasOwners { name: "draft".into(), lt: true, ..target.clone() };
        let media = BlasOwners { name: "media".into(), thread: "family/blas/media".into(),
            process: "media worker".into(), ..target.clone() };
        let owners = [target.clone(), draft, media];
        let components = blas_components("sm_121", &owners, 1, order).unwrap();
        for &phase in order.phases() {
            let active = components.iter().filter_map(|component| {
                let ComponentKind::Allowance(owner) = &component.kind else { panic!("opaque BLAS") };
                owner.lifetime.contains(phase, order).then_some(owner)
            }).collect::<Vec<_>>();
            assert_eq!(active.iter().map(|owner| owner.bytes).sum::<u64>(),
                2 * (67_508_224 + 158_674_944) + 201_461_760);
            assert!(active.iter().all(|owner| !owner.assumed));
        }
        assert!(blas_components("sm_121", &[BlasOwners { lt: true, gemm: false, ..target.clone() }], 1, order).is_err());
        assert!(blas_components("sm_121", &[BlasOwners { thread: String::new(), ..target }], 1, order).is_err());
    }

    #[test]
    fn selector_geometry_preserves_domains_capacity_and_independent_sampler_owners() {
        for capacity in [1, 64, 128, 256] {
            for vocab in [1, 33, 154880] {
                for device_sampling in [false, true] {
                    let spec = SelectorOwners { location: Location::coordinator(1), capacity, vocab, device_sampling };
                    let owners = spec.allocations("target", Lifetime::RESIDENT).unwrap();
                    assert_eq!(owners.len(), if device_sampling {24} else {2});
                    assert_eq!(owners[0].bytes, capacity as u64 * 12);
                    assert_eq!(owners[0].location, Location::coordinator(1));
                    assert_eq!(owners[1].location, Location::PINNED);
                    assert!(owners.iter().all(|a| a.view.is_none() && a.lifetime == Lifetime::RESIDENT));
                    let sampler = cuteafd_core::target_sampler_allocations(capacity.min(128), vocab).unwrap();
                    for expected in sampler.into_iter().filter(|_| device_sampling) {
                        let actual = owners.iter().find(|a| a.id == format!("target.sampler.{}", expected.id)).unwrap();
                        assert_eq!(actual.bytes, expected.bytes as u64);
                        assert_eq!(actual.location, if expected.pinned {Location::PINNED} else {spec.location});
                    }
                }
            }
        }
        let base = SelectorOwners {location: Location::coordinator(0), capacity: 64, vocab: 33, device_sampling: true};
        for bad in [SelectorOwners {capacity: 0, ..base}, SelectorOwners {vocab: 0, ..base},
            SelectorOwners {capacity: usize::MAX, ..base}, SelectorOwners {vocab: usize::MAX, ..base},
            SelectorOwners {location: Location::PINNED, ..base},
            SelectorOwners {location: Location {device: Device::Host, domain: Domain::Device}, ..base}]
        {
            assert!(bad.allocations("selector", Lifetime::RESIDENT).is_err());
        }
        assert!(base.allocations("", Lifetime::RESIDENT).is_err());
    }

    #[test]
    fn prefix_charges_only_whole_retained_chunks_and_composes_without_policy() {
        let prefix = HostPrefixOwners {quota_bytes: 2 * 4096 + 17, chunk_bytes: 4096};
        let owners = prefix.allocations("host.prefix", ServingOwners::LIFETIME).unwrap();
        assert_eq!(owners.len(), 2);
        assert_eq!(owners.iter().map(|a| a.bytes).sum::<u64>(), 8192);
        assert_eq!(owners[1].id, "host.prefix.1");
        assert!(owners.iter().all(|a| a.location == Location::PINNED && a.view.is_none()));
        for chunk_bytes in [0, 4096] {
            assert!(HostPrefixOwners {quota_bytes: 0, chunk_bytes}
                .allocations("host.prefix", ServingOwners::LIFETIME).unwrap().is_empty());
        }
        for bad in [HostPrefixOwners {quota_bytes: 1, chunk_bytes: 0},
            HostPrefixOwners {quota_bytes: 1, chunk_bytes: 2},
            HostPrefixOwners {quota_bytes: u64::MAX, chunk_bytes: 1}]
        {
            assert!(bad.allocations("host.prefix", ServingOwners::LIFETIME).is_err());
        }
        assert!(prefix.allocations("", ServingOwners::LIFETIME).is_err());
        let serving = ServingOwners {selector: SelectorOwners {location: Location::coordinator(0),
            capacity: 64, vocab: 33, device_sampling: true}, prefix};
        let exact = serving.allocations().unwrap();
        assert_eq!(exact.len(), 26);
        let contract = Contract { floor_fallback: None,resident_weight_limit: None, spark_reply_formats: None, phase_order: crate::placement::planner::PhaseOrder::TARGET_THEN_DRAFT,page_rows: 64, pool_page_multiple: 1, maximum_pages: 16, gpus: 1,
            variables: vec![Variable::Prefill(vec![PrefillShape {rows: 64, waves: 1}]), Variable::Lanes(vec![1])],
            requires: vec![], components: serving.components(16).unwrap()};
        for pages in [1, 8, 16] {
            assert_eq!(contract.facts(&[0, 0], pages).unwrap().allocations, exact);
        }
        assert!(serving.components(0).is_err());
    }
}
