//! Direct cudaMalloc backing on SM120, CUDA 13.2 LAZY. Requested payload stays
//! with its allocation; this opaque owner charges only additional backing pages.
use super::*;

pub const ALLOCATION_GRANULARITY_GROUP: &str = "allocation granularity";
pub const SM120_ALLOCATION_GRANULARITY_BYTES: u64 = 2 << 20;
const PAGE: u64 = SM120_ALLOCATION_GRANULARITY_BYTES;
const SMALL: u64 = 1 << 20;

/// SM120, driver/runtime 13020, LAZY: stream slots grow in retained 2 MiB
/// slabs at live-stream counts 2, 6, 10, ... . Destroyed streams reuse slots;
/// count the context's peak live streams, never cumulative creations. Qualified
/// on both raptor GPUs, 2026-11-08 (#1511); not qualified for GB10/SM121.
pub fn cuda_stream_slab_bytes(peak_live: u64) -> Result<u64, Refusal> {
    let slabs = peak_live / 4 + u64::from(peak_live % 4 >= 2);
    slabs.checked_mul(PAGE).ok_or(Refusal::Overflow("CUDA stream slabs"))
}

/// Explicit physical requests behind one logical owner, not its tensor views.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GranularityBacking {
    pub backing: String,
    pub requests: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GranularityEvidence {
    /// Large allocations round independently; small requests share driver blocks.
    /// This is not a packing guarantee (e.g. repeated 576 KiB requests fragment).
    Forecast,
}

/// Qualified only for discrete SM120 direct cudaMalloc, driver/runtime 13020,
/// CUDA_MODULE_LOADING=LAZY. Never apply this physical model to GB10/SM121.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationGranularity {
    pub id: String,
    pub location: Location,
    pub lifetime: Lifetime,
    pub backings: Vec<GranularityBacking>,
    pub evidence: GranularityEvidence,
}

fn sum_requests(requests: impl IntoIterator<Item = u64>) -> Result<u64, Refusal> {
    requests.into_iter().try_fold(0u64, |sum, bytes| {
        if bytes == 0 { return Err(Refusal::InvalidFact("empty cudaMalloc request".into())); }
        sum.checked_add(bytes).ok_or(Refusal::Overflow("cudaMalloc requests"))
    })
}

fn round(bytes: u64) -> Result<u64, Refusal> {
    bytes.div_ceil(PAGE).checked_mul(PAGE).ok_or(Refusal::Overflow("cudaMalloc backing"))
}

/// Sizes-only forecast, calibrated against Kimi's 3,935 core requests:
/// 1,244,796,544 B forecast vs 1,242,699,392 B physical slack (+0.16876%).
/// #1454, 2026-11-08. No multiplier, residual or safety margin.
pub fn allocation_granularity_forecast(requests: &[u64]) -> Result<u64, Refusal> {
    let requested = sum_requests(requests.iter().copied())?;
    let (mut large, mut small) = (0u64, 0u64);
    for &bytes in requests {
        if bytes > SMALL {
            large = large.checked_add(round(bytes)?).ok_or(Refusal::Overflow("large cudaMalloc backing"))?;
        } else {
            small = small.checked_add(bytes).ok_or(Refusal::Overflow("small cudaMalloc backing"))?;
        }
    }
    large.checked_add(round(small)?).and_then(|bytes| bytes.checked_sub(requested))
        .ok_or(Refusal::Overflow("cudaMalloc granularity"))
}

/// Physical slack of the independently rounded backings in `requests`: one
/// SM120 block round-up per request above 1 MiB. Sub-MiB requests share driver
/// blocks across owners, so their shared remainder is attributable to no single
/// owner and stays outside this charge.
pub fn independent_backing_slack(requests: impl IntoIterator<Item = u64>) -> Result<u64, Refusal> {
    let large = requests.into_iter().filter(|&bytes| bytes > SMALL).collect::<Vec<_>>();
    allocation_granularity_forecast(&large)
}

impl AllocationGranularity {
    pub fn requests(&self) -> Vec<u64> {
        self.backings.iter().flat_map(|owner| owner.requests.iter().copied()).collect()
    }

    pub fn bytes(&self) -> Result<u64, Refusal> {
        if self.id.is_empty() || self.backings.is_empty()
            || !matches!(self.location, Location { device: Device::Coordinator(_), domain: Domain::Device }) {
            return Err(Refusal::InvalidFact("SM120 cudaMalloc granularity requires a named coordinator owner".into()));
        }
        allocation_granularity_forecast(&self.requests())
    }

    pub fn basis(&self) -> Basis { Basis::Estimated }

    pub fn source(&self) -> String {
        format!("{} (SM120 direct cudaMalloc, driver/runtime13020 LAZY, #1454 2026-11-08, sizes-only forecast)", self.id)
    }

    pub(super) fn validate_backings(&self, facts: &Facts, seen: &mut BTreeSet<String>) -> Result<(), Refusal> {
        self.bytes()?;
        for backing in &self.backings {
            let owner = facts.allocations.iter().chain(&facts.deferred).find(|owner| owner.id == backing.backing);
            if backing.backing.is_empty() || !seen.insert(backing.backing.clone())
                || !owner.is_some_and(|owner| owner.view.is_none() && owner.location == self.location
                    && owner.lifetime == self.lifetime && sum_requests(backing.requests.iter().copied()) == Ok(owner.bytes)) {
                return Err(Refusal::InvalidFact(format!("invalid cudaMalloc granularity backing {}", backing.backing)));
            }
        }
        Ok(())
    }
}

/// Fixed request inventory, activated with the same finite placement profile as
/// its payload. A logical aggregate alone is never sufficient to call this.
pub fn allocation_granularity_component(owner: AllocationGranularity, maximum: u64) -> Result<Component, Refusal> {
    let bytes = owner.bytes()?;
    Ok(Component { activation: Activation::always(), bytes: ByteCurve::fixed(bytes, maximum),
        kind: ComponentKind::AllocationGranularity(owner) })
}

/// One immutable SM120 arena request, its resident-weight claim and its single
/// separately named backing tail. Replaces payload/padding/old request forecasts;
/// callers must not retain those charges or apply this model to Spark storage.
/// Apply a finite-profile activation to every returned component together.
pub fn weight_arena_components(
    layout: &crate::serving_capacity::WeightArenaLayout,
    location: Location,
    lifetime: Lifetime,
    maximum: u64,
    kind: MemoryKind,
) -> Result<Vec<Component>, Refusal> {
    let bytes = u64::try_from(layout.bytes()).map_err(|_| Refusal::Overflow("weight arena extent"))?;
    weight_arena_extent_components(layout.id(), bytes, location, lifetime, maximum, kind)
}

/// Extent-only counterpart for facts carrying the shared recipe's exact ID/B.
/// `bytes` includes inter-view padding, never the driver tail or a payload sum.
pub fn weight_arena_extent_components(
    id: &str,
    bytes: u64,
    location: Location,
    lifetime: Lifetime,
    maximum: u64,
    kind: MemoryKind,
) -> Result<Vec<Component>, Refusal> {
    if kind.behaviour() != Behaviour::Immutable || id.is_empty() || bytes == 0 || maximum == 0
        || !matches!(location, Location { device: Device::Coordinator(_), domain: Domain::Device }) {
        return Err(Refusal::InvalidFact("weight arena requires a named positive SM120 device extent and page domain".into()));
    }
    let root = Allocation::buffer(id, location, bytes, lifetime, kind.category(), "weight arena");
    let curve = ByteCurve::fixed(bytes, maximum);
    let resident = ResidentWeight::from_allocation(&root, ResidentWeights(bytes));
    let tail = allocation_granularity_component(AllocationGranularity {
        id: format!("{id}.tail"), location, lifetime,
        backings: vec![GranularityBacking { backing: id.into(), requests: vec![bytes] }],
        evidence: GranularityEvidence::Forecast,
    }, maximum)?;
    Ok(vec![Component::allocation(root, curve.clone(), false, maximum),
        Component::resident_weight(resident, curve), tail])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_slabs_follow_peak_live_slots_not_cumulative_creations() -> Result<(), Refusal> {
        for peak in 0..=34 {
            assert_eq!(cuda_stream_slab_bytes(peak)?, (peak + 2) / 4 * PAGE);
        }
        // Two retained cache/lane streams plus sequential rotary streams.
        assert_eq!(cuda_stream_slab_bytes(3)?, PAGE);
        // Dense load plus four copies coexist with cache/lane streams.
        assert_eq!(cuda_stream_slab_bytes(7)?, 2 * PAGE);
        assert!(cuda_stream_slab_bytes(u64::MAX).is_err());
        Ok(())
    }

    #[test]
    fn arena_extent_replaces_payload_and_prices_one_physical_request() -> Result<(), Refusal> {
        let layout = crate::serving_capacity::WeightArenaLayout::new("target.rank0",
            [("a".into(), (3 << 20) + 1), ("b".into(), 17)])
            .map_err(|e| Refusal::InvalidFact(e.to_string()))?;
        assert_eq!(layout.padding_bytes(), 255);
        let lifetime = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
        let components = weight_arena_components(&layout, Location::coordinator(0), lifetime, 32, cuteafd_core::memory_layout::MemoryKind::Weights)?;
        assert_eq!(components.len(), 3);
        let ComponentKind::Allocation { template: root, deferred, .. } = &components[0].kind else { panic!("missing root") };
        assert_eq!(root.id, layout.id());
        assert_eq!(root.bytes, layout.bytes() as u64);
        assert_eq!(root.lifetime, lifetime);
        assert_eq!(root.group, "weight arena");
        assert!(root.view.is_none() && !deferred);
        let ComponentKind::ResidentWeight(resident) = &components[1].kind else { panic!("missing residency") };
        assert_eq!(resident.id, root.id);
        assert_eq!(resident.weights, ResidentWeights(root.bytes));
        let ComponentKind::AllocationGranularity(tail) = &components[2].kind else { panic!("missing tail") };
        assert_eq!(tail.id, "target.rank0.tail");
        assert_eq!(tail.lifetime, lifetime);
        assert_eq!(tail.requests(), [root.bytes]);
        assert_eq!(tail.backings[0].backing, root.id);
        assert_eq!(root.bytes + tail.bytes()?, 4 << 20);
        assert_ne!(root.bytes, layout.payload_bytes() as u64);
        assert!(weight_arena_extent_components("", 1, Location::coordinator(0), lifetime, 32, cuteafd_core::memory_layout::MemoryKind::Weights).is_err());
        assert!(weight_arena_extent_components("x", 0, Location::coordinator(0), lifetime, 32, cuteafd_core::memory_layout::MemoryKind::Weights).is_err());
        assert!(weight_arena_extent_components("x", 1, Location::coordinator(0), lifetime, 0, cuteafd_core::memory_layout::MemoryKind::Weights).is_err());
        assert!(weight_arena_extent_components("x", 1, Location::PINNED, lifetime, 32, cuteafd_core::memory_layout::MemoryKind::Weights).is_err());
        assert!(weight_arena_extent_components("x", 1, Location { device: Device::Spark(0), domain: Domain::Device }, lifetime, 32, cuteafd_core::memory_layout::MemoryKind::Weights).is_err());
        Ok(())
    }

    #[test]
    fn arena_root_kind_is_immutable_and_never_guessed_from_id() -> Result<(), Refusal> {
        for kind in MemoryKind::ALL {
            let components = weight_arena_extent_components("neutral", 3 << 20,
                Location::coordinator(0), Lifetime::RESIDENT, 1, kind);
            if kind.behaviour() == Behaviour::Immutable {
                let components = components?;
                let ComponentKind::Allocation { template, .. } = &components[0].kind else { panic!("root") };
                assert_eq!(template.category, kind.category());
                let ComponentKind::AllocationGranularity(tail) = &components[2].kind else { panic!("tail") };
                assert_eq!(tail.bytes()?, 1 << 20);
                assert_eq!(tail.backings[0].backing, template.id);
            } else {
                assert!(components.is_err(), "nonimmutable {kind:?} accepted by weight arena");
            }
        }
        Ok(())
    }

    #[test]
    fn retained_kimi_requests_match_physical_slack_within_one_block() {
        let histogram = [(384,69),(512,69),(1024,24),(1280,24),(3072,72),(3584,92),(7168,276),
            (12288,276),(14336,1087),(24576,393),(28672,186),(36864,24),(49152,69),(589824,69),
            (3145728,69),(3211264,69),(3440640,24),(8257536,24),(12582912,48),(12845056,92),
            (19267584,184),(21233664,24),(33030144,276),(66060288,393),(484442112,1),(968884224,1)];
        let requests = histogram.into_iter().flat_map(|(bytes, count)| std::iter::repeat_n(bytes, count)).collect::<Vec<_>>();
        assert_eq!(requests.len(), 3935);
        assert_eq!(sum_requests(requests.iter().copied()).unwrap(), 43_172_882_816);
        let forecast = allocation_granularity_forecast(&requests).unwrap();
        assert_eq!(forecast, 1_244_796_544);
        assert_eq!(forecast - 1_242_699_392, PAGE);
        assert!(forecast - 1_242_699_392 < 1_242_699_392 / 100);
    }

    #[test]
    fn small_requests_share_blocks_and_large_requests_round_independently() {
        assert_eq!(allocation_granularity_forecast(&[]).unwrap(), 0);
        for (requests, physical) in [(vec![SMALL; 2], PAGE), (vec![SMALL + 1; 2], 2 * PAGE),
            (vec![63 << 20, 14 << 10, 24 << 10], 66 << 20)] {
            assert_eq!(allocation_granularity_forecast(&requests).unwrap() + requests.iter().sum::<u64>(), physical);
        }
        assert!(allocation_granularity_forecast(&[0]).is_err());
        assert!(allocation_granularity_forecast(&[u64::MAX]).is_err());
    }
}
