//! Actual allocation/free chronology, joined with independently retained owners.
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::memory_ledger::{self, AllocationAudit, AllocationEvent, AllocationRecord, Space};
use cuteafd_loader::placement::planner::{
    CoordinatorDevices, Domain, Location, Observation, OwnerEvent, Phase, PhaseOrder, Receipt,
};
use std::collections::BTreeMap;

pub(crate) struct StartupHistory {
    audit: AllocationAudit,
    phases: Vec<(Phase, usize)>,
}
impl StartupHistory {
    pub(crate) fn start() -> Self {
        Self {
            audit: AllocationAudit::start(),
            phases: vec![(Phase::Load, 0)],
        }
    }

    pub(crate) fn phase(&self) -> Phase {
        self.phases.last().expect("initial Load phase").0
    }

    /// Called at the real phase boundary, before that phase allocates anything.
    pub(crate) fn begin(&mut self, phase: Phase) -> Result<()> {
        ensure!(
            !self.phases.iter().any(|&(seen, _)| seen == phase),
            "startup phase already recorded: {phase:?}"
        );
        self.phases.push((phase, self.audit.events().len()));
        Ok(())
    }

    pub(crate) fn owners(
        &self,
        order: PhaseOrder,
        through: Phase,
        receipts: &[Receipt],
        devices: CoordinatorDevices<'_>,
        mut transient: impl FnMut(&AllocationRecord) -> Result<Option<Receipt>>,
    ) -> Result<(Vec<OwnerEvent>, Vec<Observation>)> {
        observed_owners(
            &self.audit.events(),
            &self.phases,
            order,
            through,
            receipts,
            devices,
            &memory_ledger::allocations(),
            &mut transient,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn observed_owners(
    audit: &[AllocationEvent],
    phases: &[(Phase, usize)],
    order: PhaseOrder,
    through: Phase,
    receipts: &[Receipt],
    devices: CoordinatorDevices<'_>,
    census: &[AllocationRecord],
    transient: &mut impl FnMut(&AllocationRecord) -> Result<Option<Receipt>>,
) -> Result<(Vec<OwnerEvent>, Vec<Observation>)> {
    let expected = order
        .phases()
        .iter()
        .copied()
        .take_while(|&phase| order.at_or_before(phase, through))
        .collect::<Vec<_>>();
    ensure!(
        phases.iter().map(|&(phase, _)| phase).eq(expected),
        "observed startup phases differ from planner family order"
    );
    ensure!(
        phases.first().is_some_and(|&(_, index)| index == 0)
            && phases.windows(2).all(|pair| pair[0].1 <= pair[1].1)
            && phases
                .last()
                .is_some_and(|&(_, index)| index <= audit.len()),
        "invalid observed startup boundaries"
    );
    let key = |a: &AllocationRecord| (a.key.space, a.key.device, a.address);
    let mut live = BTreeMap::new();
    let mut events = Vec::new();
    let mut phase_index = 0;
    for (index, event) in audit.iter().enumerate() {
        while phase_index + 1 < phases.len() && phases[phase_index + 1].1 <= index {
            phase_index += 1;
        }
        let at = phases[phase_index].0;
        match event {
            AllocationEvent::Allocated(a) => {
                let location = match a.key.space {
                    Space::Device => devices.location(a.key.device)?,
                    Space::Host => Location::HOST,
                    Space::Pinned => Location::PINNED,
                    Space::Managed => Location {
                        device: devices.location(a.key.device)?.device,
                        domain: Domain::Managed,
                    },
                };
                let receipt = match transient(a)? {
                    Some(receipt) => receipt,
                    None => receipts
                        .iter()
                        .find(|r| {
                            r.allocation.view.is_none()
                                && r.address == a.address as u64
                                && r.allocation.location == location
                        })
                        .context("audited allocation absent from independent owner receipts")?
                        .clone(),
                };
                ensure!(
                    receipt.address == a.address as u64
                        && receipt.allocation.location == location
                        && receipt.allocation.bytes == a.bytes as u64,
                    "allocation census differs from object receipt {}",
                    receipt.allocation.id
                );
                events.push(OwnerEvent::Allocated {
                    receipt: receipt.clone(),
                    phase: at,
                });
                ensure!(
                    live.insert(key(a), (*a, receipt)).is_none(),
                    "startup allocation reused a live pointer"
                );
            }
            AllocationEvent::Freed(a) => {
                let (allocated, receipt) = live
                    .remove(&key(a))
                    .context("free without audited allocation")?;
                ensure!(
                    allocated.key == a.key && allocated.bytes == a.bytes,
                    "free differs from audited owner"
                );
                events.push(OwnerEvent::Freed { receipt, phase: at });
            }
        }
    }
    let current = census
        .iter()
        .map(|a| (key(a), a))
        .collect::<BTreeMap<_, _>>();
    for (k, (owner, _)) in &live {
        ensure!(
            current
                .get(k)
                .is_some_and(|a| a.key == owner.key && a.bytes == owner.bytes),
            "retained owner absent from live census"
        );
    }
    let views = receipts
        .iter()
        .filter(|r| r.allocation.view.is_some())
        .map(|r| Observation {
            receipt: r.clone(),
            created: r.allocation.lifetime.first,
            freed: None,
        })
        .collect();
    Ok((events, views))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::memory_layout::Category;
    use cuteafd_ffi::memory_ledger::Key;
    use cuteafd_loader::placement::planner::{Allocation, Lifetime};

    #[test]
    fn observed_history_uses_each_familys_order_and_independent_physical_census() -> Result<()> {
        let record = |address, bytes, device| AllocationRecord {
            address,
            bytes,
            key: Key {
                space: Space::Device,
                device,
                scope: "test",
                owner: None,
                tensor: None,
                format: None,
            },
        };
        for order in [PhaseOrder::TARGET_THEN_DRAFT, PhaseOrder::DRAFT_THEN_TARGET] {
            let records = [record(4096, 100, 7), record(8192, 200, 3)];
            let receipts = records
                .iter()
                .zip([Phase::PostTargetKv, Phase::DraftLoad])
                .enumerate()
                .map(|(rank, (owner, first))| Receipt {
                    allocation: Allocation::buffer(
                        format!("owner.{rank}"),
                        Location::coordinator(rank as u8),
                        owner.bytes as u64,
                        Lifetime {
                            first,
                            last: Phase::Growth,
                        },
                        Category::Workspace,
                        "test",
                    ),
                    address: owner.address as u64,
                })
                .collect::<Vec<_>>();
            let mut audit = Vec::new();
            let mut phases = Vec::new();
            for &phase in order.phases().iter().take(5) {
                phases.push((phase, audit.len()));
                for (owner, receipt) in records.iter().zip(&receipts) {
                    if receipt.allocation.lifetime.first == phase {
                        audit.push(AllocationEvent::Allocated(*owner));
                    }
                }
            }
            let check = |audit: &[AllocationEvent],
                         phases: &[(Phase, usize)],
                         expected,
                         owners: &[Receipt],
                         census: &[AllocationRecord]| {
                observed_owners(
                    audit,
                    phases,
                    expected,
                    Phase::Ready,
                    owners,
                    CoordinatorDevices::new(&[7, 3])?,
                    census,
                    &mut |_| Ok(None),
                )
            };
            let (events, views) = check(&audit, &phases, order, &receipts, &records)?;
            assert!(views.is_empty());
            assert_eq!(events.len(), 2);
            for (event, &phase) in events.iter().zip(&order.phases()[1..3]) {
                assert!(matches!(event, OwnerEvent::Allocated { phase: at, .. } if *at == phase));
            }
            let other = if order == PhaseOrder::TARGET_THEN_DRAFT {
                PhaseOrder::DRAFT_THEN_TARGET
            } else {
                PhaseOrder::TARGET_THEN_DRAFT
            };
            assert!(check(&audit, &phases, other, &receipts, &records).is_err());
            assert!(check(&audit, &phases[..4], order, &receipts, &records).is_err());
            let mut invalid = phases.clone();
            invalid[2].1 = audit.len() + 1;
            assert!(check(&audit, &invalid, order, &receipts, &records).is_err());
            assert!(check(&audit, &phases, order, &receipts[..1], &records).is_err());
            assert!(check(&audit, &phases, order, &receipts, &records[..1]).is_err());
            let mut wrong = receipts.clone();
            wrong[0].allocation.bytes += 1;
            assert!(check(&audit, &phases, order, &wrong, &records).is_err());
            wrong = receipts.clone();
            wrong[0].allocation.location = Location::coordinator(1);
            assert!(check(&audit, &phases, order, &wrong, &records).is_err());
            let mut extra = audit.clone();
            extra.push(AllocationEvent::Allocated(record(12288, 33, 7)));
            assert!(check(&extra, &phases, order, &receipts, &records).is_err());
            let mut unknown = audit.clone();
            if let AllocationEvent::Allocated(owner) = &mut unknown[0] {
                owner.key.device = 9;
            }
            assert!(check(&unknown, &phases, order, &receipts, &records).is_err());
        }
        Ok(())
    }

    #[test]
    fn recorder_observes_phase_boundaries_without_a_planned_order() -> Result<()> {
        for order in [PhaseOrder::TARGET_THEN_DRAFT, PhaseOrder::DRAFT_THEN_TARGET] {
            let mut history = StartupHistory::start();
            for &phase in &order.phases()[1..5] {
                history.begin(phase)?;
            }
            assert!(history.begin(Phase::Ready).is_err());
            assert_eq!(history.phase(), Phase::Ready);
            assert!(history.phases.iter().map(|&(phase, _)| phase)
                .eq(order.phases()[..5].iter().copied()));
        }
        Ok(())
    }
}
