//! Actual allocation/free chronology, joined with independently retained owners.
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::memory_ledger::{self, AllocationAudit, AllocationEvent, AllocationRecord, Space};
use cuteafd_loader::placement::planner::{
    Allocation, CoordinatorDevices, Domain, Location, Observation, OwnerEvent, Phase, PhaseOrder,
    Receipt, ReservationReceipt,
};
use cuteafd_core::memory_layout::Category;
use std::{collections::BTreeMap, sync::{Arc, Mutex}};

/// The bound is planned; the peak comes from the load pool's own held-byte census.
/// Audit replay independently checks that peak, then requires peak <= bound.
pub(crate) struct ReservationCensus {
    spec: cuteafd_loader::placement::planner::Reservation,
    peak: Arc<Mutex<Option<u64>>>,
}
impl ReservationCensus {
    pub(crate) fn spec(&self) -> &cuteafd_loader::placement::planner::Reservation { &self.spec }

    pub(crate) fn complete(self, peak: u64) -> Result<()> {
        ensure!(peak <= self.spec.bytes, "load pool exceeded reservation {}", self.spec.id);
        let mut observed = self.peak.lock().unwrap_or_else(|e| e.into_inner());
        ensure!(observed.is_none(), "load pool census already completed: {}", self.spec.id);
        *observed = Some(peak);
        Ok(())
    }

    fn receipt(&self) -> Result<ReservationReceipt> {
        Ok(ReservationReceipt { reservation: self.spec.clone(),
            peak: self.peak.lock().unwrap_or_else(|e| e.into_inner())
                .context("load pool has no completed peak census")? })
    }
}

pub(crate) struct StartupHistory {
    audit: AllocationAudit,
    phases: Vec<(Phase, usize)>,
    reservations: Vec<ReservationCensus>,
}
impl StartupHistory {
    pub(crate) fn start() -> Self {
        Self {
            audit: AllocationAudit::start(),
            phases: vec![(Phase::Load, 0)],
            reservations: Vec::new(),
        }
    }

    pub(crate) fn reserve(&mut self, spec: cuteafd_loader::placement::planner::Reservation) -> Result<ReservationCensus> {
        ensure!(spec.location == Location::PINNED && spec.bytes > 0 && !spec.id.is_empty()
            && !spec.source.is_empty() && spec.lifetime.first == self.phase()
            && spec.lifetime.last == self.phase()
            && !matches!(self.phase(), Phase::Ready | Phase::Growth), "invalid startup pool reservation");
        ensure!(!self.reservations.iter().any(|r| r.spec.id == spec.id), "duplicate startup pool reservation");
        let census = ReservationCensus { spec, peak: Arc::new(Mutex::new(None)) };
        self.reservations.push(ReservationCensus { spec: census.spec.clone(), peak: Arc::clone(&census.peak) });
        Ok(census)
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
    ) -> Result<(Vec<OwnerEvent>, Vec<Observation>, Vec<ReservationReceipt>)> {
        let reservations = self.reservations.iter()
            .filter(|r| order.at_or_before(r.spec.lifetime.first, through))
            .map(ReservationCensus::receipt).collect::<Result<Vec<_>>>()?;
        let (events, views) = observed_owners(
            &self.audit.events(),
            &self.phases,
            order,
            through,
            receipts,
            devices,
            &memory_ledger::allocations(),
            &reservations,
            &mut transient,
        )?;
        Ok((events, views, reservations))
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
    reservations: &[ReservationReceipt],
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
                let reserved = reservations.iter().find(|r| a.key.owner == Some(r.reservation.id.as_str()));
                let transient = transient(a)?;
                let retained = receipts.iter().find(|r| r.allocation.view.is_none()
                    && r.address == a.address as u64 && r.allocation.location == location);
                let receipt = if let Some(pool) = reserved {
                    ensure!(a.key.space == Space::Pinned && a.key.scope == "staging/coordinator-dense"
                        && location == pool.reservation.location && at == pool.reservation.lifetime.first
                        && transient.is_none() && retained.is_none(), "reservation cannot absorb an exact owner");
                    Receipt { allocation: Allocation::buffer(format!("{}.buffer.{index}", pool.reservation.id),
                        location, a.bytes as u64, pool.reservation.lifetime, Category::Staging,
                        "delivery-order-dependent load pool"), address: a.address as u64 }
                } else {
                    transient.or_else(|| retained.cloned())
                        .context("audited allocation absent from independent owner receipts")?
                };
                ensure!(
                    receipt.address == a.address as u64
                        && receipt.allocation.location == location
                        && receipt.allocation.bytes == a.bytes as u64,
                    "allocation census differs from object receipt {}",
                    receipt.allocation.id
                );
                let reservation = reserved.map(|r| r.reservation.id.clone());
                events.push(if let Some(reservation) = &reservation {
                    OwnerEvent::Reserved { reservation: reservation.clone(), receipt: receipt.clone(), phase: at, allocate: true }
                } else { OwnerEvent::Allocated { receipt: receipt.clone(), phase: at } });
                ensure!(
                    live.insert(key(a), (*a, receipt, reservation)).is_none(),
                    "startup allocation reused a live pointer"
                );
            }
            AllocationEvent::Freed(a) => {
                let (allocated, receipt, reservation) = live
                    .remove(&key(a))
                    .context("free without audited allocation")?;
                ensure!(
                    allocated.key == a.key && allocated.bytes == a.bytes,
                    "free differs from audited owner"
                );
                events.push(if let Some(reservation) = reservation {
                    OwnerEvent::Reserved { reservation, receipt, phase: at, allocate: false }
                } else { OwnerEvent::Freed { receipt, phase: at } });
            }
        }
    }
    let current = census
        .iter()
        .map(|a| (key(a), a))
        .collect::<BTreeMap<_, _>>();
    for (k, (owner, _, _)) in &live {
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
                    &[],
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
    fn pool_history_requires_distinct_completed_censuses_and_exact_tagged_audit_edges() -> Result<()> {
        use cuteafd_loader::placement::planner::Reservation;
        let spec = Reservation { id: "dense.0".into(), location: Location::PINNED, bytes: 300,
            lifetime: Lifetime { first: Phase::Load, last: Phase::Load }, source: "checked steps".into() };
        let mut history = StartupHistory::start();
        let census = history.reserve(spec.clone())?;
        assert!(history.reserve(spec.clone()).is_err());
        assert!(history.reservations[0].receipt().is_err());
        std::thread::spawn(move || census.complete(200)).join().expect("census thread")?;
        assert_eq!(history.reservations[0].receipt()?, ReservationReceipt { reservation: spec.clone(), peak: 200 });
        let mut other = spec.clone(); other.id = "coded-scales.0".into();
        assert!(history.reserve(other)?.complete(301).is_err());
        let record = AllocationRecord { address: 4096, bytes: 200, key: Key { space: Space::Pinned,
            device: -1, scope: "staging/coordinator-dense", owner: Some("dense.0"), tensor: None, format: None } };
        let audit = [AllocationEvent::Allocated(record), AllocationEvent::Freed(record)];
        let phases = PhaseOrder::DRAFT_THEN_TARGET.phases().iter().take(5)
            .map(|&phase| (phase, if phase == Phase::Load { 0 } else { audit.len() })).collect::<Vec<_>>();
        let reports = [ReservationReceipt { reservation: spec, peak: 200 }];
        let check = |audit: &[AllocationEvent], receipts: &[Receipt], reports: &[ReservationReceipt]| {
            observed_owners(audit, &phases, PhaseOrder::DRAFT_THEN_TARGET, Phase::Ready, receipts,
                CoordinatorDevices::new(&[7, 3])?, &[], reports, &mut |_| Ok(None))
        };
        let (events, views) = check(&audit, &[], &reports)?;
        assert!(views.is_empty());
        assert!(matches!(&events[0], OwnerEvent::Reserved { reservation, allocate: true, .. } if reservation == "dense.0"));
        assert!(matches!(&events[1], OwnerEvent::Reserved { reservation, allocate: false, .. } if reservation == "dense.0"));
        assert!(check(&audit, &[], &[]).is_err());
        let exact = Receipt { allocation: Allocation::buffer("exact", Location::PINNED, 200,
            Lifetime::RESIDENT, Category::Staging, "exact"), address: 4096 };
        assert!(check(&audit, &[exact], &reports).is_err(), "pool tags cannot hide independently retained exact owners");
        for mutation in 0..5 {
            let mut wrong = audit;
            if let AllocationEvent::Allocated(a) = &mut wrong[0] {
                match mutation {
                    0 => a.key.owner = Some("unregistered-pool"),
                    1 => a.key.scope = "staging/fp8-load-quant",
                    2 => a.key.space = Space::Host,
                    3 => a.bytes += 1,
                    _ => a.address += 16,
                }
            }
            assert!(check(&wrong, &[], &reports).is_err(), "mutation {mutation}");
        }
        history.begin(Phase::DraftLoad)?;
        let mut wrong = reports[0].reservation.clone(); wrong.id = "late".into();
        assert!(history.reserve(wrong).is_err());
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
