//! Physical serving owners shared by family recipes. Quotas and selector
//! placement are resolved inputs; this module never chooses admission policy.
use super::*;

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
        let contract = Contract {resident_weight_limit: None, spark_reply_format: None, phase_order: crate::placement::planner::PhaseOrder::TARGET_THEN_DRAFT,page_rows: 64, maximum_pages: 16, gpus: 1,
            variables: vec![Variable::Prefill(vec![PrefillShape {rows: 64, waves: 1}]), Variable::Lanes(vec![1])],
            requires: vec![], components: serving.components(16).unwrap()};
        for pages in [1, 8, 16] {
            assert_eq!(contract.facts(&[0, 0], pages).unwrap().allocations, exact);
        }
        assert!(serving.components(0).is_err());
    }
}
