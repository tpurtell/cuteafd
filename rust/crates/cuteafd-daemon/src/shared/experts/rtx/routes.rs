//! Replicated routers and the route-identity check.
//!
//! Under the head split both GPUs hold the same post-attention hidden state,
//! so each runs its own router replica. [`RouteIdentity::check`] compares the
//! two ranks' ids and weights byte for byte once at startup (after graph
//! warm-up, on a fixed prefill plus one decode step). On a mismatch it logs
//! the first layer and row and the engine falls back to
//! [`RouteSource::Broadcast`] (rank 0 pushes ids + weights each MoE layer).
//! `CUTEAFD_ROUTE_CHECK=N` re-checks every N steps (off by default).

/// Where rank 1's routes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteSource {
    /// Rank 1 runs its own router replica.
    Replicated,
    /// Rank 0 pushes ids + weights (`rows * topk * 8` bytes) per MoE layer.
    Broadcast,
}

/// One MoE layer's routes from both ranks, as host bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouteCheck {
    pub layer: usize,
    pub rows: usize,
    pub topk: usize,
    /// `[rows, topk]` U32 ids then `[rows, topk]` FP32 weights, per rank.
    pub ranks: [Vec<u8>; 2],
}

/// The result of comparing replicated routes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RouteIdentity {
    pub layers_checked: usize,
    pub rows_checked: usize,
    /// First `(layer, row)` whose ids or weights differ.
    pub first_mismatch: Option<(usize, usize)>,
}

impl RouteIdentity {
    /// Compares every layer's two ranks; stops at the first differing row.
    pub(crate) fn check(layers: &[RouteCheck]) -> Self {
        let mut result = Self::default();
        for check in layers {
            result.layers_checked += 1;
            result.rows_checked += check.rows;
            let row_bytes = check.topk * 4;
            let ids = check.rows * row_bytes;
            for row in 0..check.rows {
                let span = |bytes: &[u8]| -> (Vec<u8>, Vec<u8>) {
                    (bytes[row * row_bytes..(row + 1) * row_bytes].to_vec(),
                     bytes[ids + row * row_bytes..ids + (row + 1) * row_bytes].to_vec())
                };
                if span(&check.ranks[0]) != span(&check.ranks[1]) {
                    result.first_mismatch = Some((check.layer, row));
                    return result;
                }
            }
        }
        result
    }

    /// The route source the engine should use after this check.
    pub(crate) fn source(&self) -> RouteSource {
        if self.first_mismatch.is_some() { RouteSource::Broadcast } else { RouteSource::Replicated }
    }
}

/// Retained diagnostic buffers for live prefill slots. Allocate max-row buffers
/// for every slot before any peer waits are queued; begin/finish never replace
/// those pinned allocations. The `finish` callback drains both producer streams,
/// then trims each buffer to the unit's actual rows before returning RouteChecks.
/// Always call `finish_span`, including before propagating an executor error.
/// Any unfinished slot rejects release, regardless of strictness. Retain this
/// owner until the submission guard drains or aborts all device work; only a
/// successful `finish` drain permits a slot to retire during normal execution.
#[derive(Debug)]
pub(crate) struct SlotRouteChecks<T> {
    slots: Vec<(T, Option<CheckedUnit>)>,
    strict: bool,
}

#[derive(Debug, Clone, Copy)]
struct CheckedUnit {
    unit: usize,
    rows: usize,
    enabled: bool,
}

impl<T> SlotRouteChecks<T> {
    /// Explicit opt-in diagnostics use `strict`; startup checks keep broadcast fallback.
    pub(crate) fn new(storage: Vec<T>, strict: bool) -> anyhow::Result<Self> {
        anyhow::ensure!(!storage.is_empty(), "route diagnostics need at least one slot");
        Ok(Self { slots: storage.into_iter().map(|storage| (storage, None)).collect(), strict })
    }

    /// Census retained storage without activating or completing any diagnostic unit.
    pub(crate) fn retained_storage(&self) -> impl ExactSizeIterator<Item = &T> {
        self.slots.iter().map(|(storage, _)| storage)
    }

    pub(crate) fn begin(&mut self, slot: usize, unit: usize, rows: usize, enabled: bool) -> anyhow::Result<()> {
        let (_, active) = self.slots.get_mut(slot).ok_or_else(|| anyhow::anyhow!("route diagnostic slot {slot}"))?;
        anyhow::ensure!(rows > 0 && active.is_none(), "route diagnostic slot {slot} still live or empty unit {unit}");
        *active = Some(CheckedUnit { unit, rows, enabled });
        Ok(())
    }

    /// Call at span end, including error unwinding. On error, retain this owner
    /// through submission teardown: non-strict comparison is not a drain proof.
    pub(crate) fn finish_span(&self) -> anyhow::Result<()> {
        if let Some((slot, (_, Some(active)))) = self.slots.iter().enumerate().find(|(_, (_, active))| active.is_some()) {
            anyhow::bail!("unfinished route diagnostics in unit {}, slot {slot}", active.unit);
        }
        Ok(())
    }

    /// A disabled unit never writes diagnostics left in a recycled slot.
    pub(crate) fn storage(&mut self, slot: usize, unit: usize) -> anyhow::Result<Option<&mut T>> {
        let (storage, active) = self.slots.get_mut(slot).ok_or_else(|| anyhow::anyhow!("route diagnostic slot {slot}"))?;
        let active = active.as_ref().ok_or_else(|| anyhow::anyhow!("route diagnostic slot {slot} is idle"))?;
        anyhow::ensure!(active.unit == unit, "route diagnostic slot {slot}: expected unit {}, got {unit}", active.unit);
        Ok(active.enabled.then_some(storage))
    }

    pub(crate) fn finish(&mut self, slot: usize, unit: usize,
        read_after_drain: impl FnOnce(&mut T, usize) -> anyhow::Result<Vec<RouteCheck>>)
        -> anyhow::Result<Option<RouteIdentity>> {
        let (storage, active) = self.slots.get_mut(slot).ok_or_else(|| anyhow::anyhow!("route diagnostic slot {slot}"))?;
        let checked = *active.as_ref().ok_or_else(|| anyhow::anyhow!("route diagnostic slot {slot} is idle"))?;
        anyhow::ensure!(checked.unit == unit,
            "route diagnostic slot {slot}: expected unit {}, got {unit}", checked.unit);
        if !checked.enabled { *active = None; return Ok(None); }
        // On a failed drain keep the slot live: its buffers must not be reused.
        let checks = read_after_drain(storage, checked.rows)?;
        anyhow::ensure!(!checks.is_empty(), "unit {unit} route diagnostics checked no layers");
        for check in &checks {
            let bytes = check.rows.checked_mul(check.topk).and_then(|n| n.checked_mul(8));
            anyhow::ensure!(check.rows == checked.rows && check.topk > 0
                && check.ranks.iter().all(|rank| Some(rank.len()) == bytes),
                "unit {unit} route diagnostic shape differs at layer {}", check.layer);
        }
        let identity = RouteIdentity::check(&checks);
        *active = None;
        if self.strict {
            if let Some((layer, row)) = identity.first_mismatch {
                anyhow::bail!("route mismatch in unit {unit}, slot {slot}, layer {layer}, row {row}");
            }
        }
        Ok(Some(identity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(layer: usize, rows: usize, a: Vec<u8>, b: Vec<u8>) -> RouteCheck {
        RouteCheck { layer, rows, topk: 2, ranks: [a, b] }
    }

    struct MockPair {
        checks: SlotRouteChecks<Vec<RouteCheck>>,
        verified: Vec<usize>,
        corrupt: Option<usize>,
    }

    impl MockPair {
        fn run(&mut self) -> anyhow::Result<()> {
            use crate::shared::prefill_pipeline::{prefill_schedule, sequence_units, FfnRoute, Step};
            let units = sequence_units(4096 * 8, 4096, 2, 4)?;
            assert!(units.len() > 2);
            let outcome = (|| {
                for step in prefill_schedule(&[FfnRoute::Local; 4], &units, 2)? {
                    match step {
                        Step::Begin { unit } => self.checks.begin(units[unit].slot, unit, units[unit].rows.len(), true)?,
                        Step::Local { layer, unit } => {
                            let rows = units[unit].rows.len();
                            let bytes = vec![0; rows * 2 * 8];
                            let mut peer = bytes.clone();
                            if self.corrupt == Some(unit) && layer == 3 { peer[rows * 8 + 2 * 8] ^= 1; }
                            self.checks.storage(units[unit].slot, unit)?.unwrap().push(check(layer, rows, bytes, peer));
                        }
                        Step::End { unit } => {
                            let identity = self.checks.finish(units[unit].slot, unit, |storage, _| Ok(std::mem::take(storage)))?.unwrap();
                            assert_eq!(identity.layers_checked, 4);
                            self.verified.push(unit);
                        }
                        _ => {}
                    }
                }
                Ok(())
            })();
            let complete = self.checks.finish_span();
            outcome.and(complete)
        }
    }

    #[test]
    fn recycled_slots_check_every_unit_and_later_corruption_fails() {
        let mut pair = MockPair { checks: SlotRouteChecks::new(vec![Vec::new(), Vec::new()], true).unwrap(),
            verified: Vec::new(), corrupt: None };
        pair.run().unwrap();
        assert_eq!(pair.verified, (0..8).collect::<Vec<_>>());
        pair.verified.clear();
        pair.corrupt = Some(5);
        let error = pair.run().unwrap_err().to_string();
        assert!(error.contains("unit 5, slot 1, layer 3, row 2"), "{error}");
        assert_eq!(pair.verified, (0..5).collect::<Vec<_>>());
    }

    #[test]
    fn live_slot_guards_and_failed_drains_require_span_completion() {
        let mut checks = SlotRouteChecks::new(vec![vec![check(1, 1, vec![0; 16], vec![0; 16])]], true).unwrap();
        checks.begin(0, 7, 1, true).unwrap();
        assert!(checks.begin(0, 8, 1, true).is_err());
        assert!(checks.storage(0, 8).is_err());
        assert!(checks.finish(0, 8, |_, _| unreachable!()).is_err());
        assert!(checks.finish(0, 7, |_, _| anyhow::bail!("drain failed")).is_err());
        assert!(checks.finish_span().unwrap_err().to_string().contains("unit 7, slot 0"));
        checks.finish(0, 7, |storage, _| Ok(storage.clone())).unwrap();
        checks.finish_span().unwrap();
        assert!(checks.finish(0, 7, |_, _| unreachable!()).is_err());
    }

    #[test]
    fn non_strict_executor_errors_retain_storage_until_drain() {
        use std::{cell::Cell, rc::Rc};
        struct PinnedOwner(Rc<Cell<usize>>);
        impl Drop for PinnedOwner {
            fn drop(&mut self) { self.0.set(self.0.get() + 1); }
        }
        let drops = Rc::new(Cell::new(0));
        let mut owner = Some(SlotRouteChecks::new(vec![PinnedOwner(drops.clone())], false).unwrap());
        let checks = owner.as_mut().unwrap();
        checks.begin(0, 5, 1, true).unwrap();
        assert!(checks.finish(0, 5, |_, _| anyhow::bail!("producer failed before drain")).is_err());
        // The adapter may release only after a successful span check.
        if checks.finish_span().is_ok() { owner = None; }
        assert!(owner.is_some());
        assert_eq!(drops.get(), 0);
        assert!(owner.as_mut().unwrap().begin(0, 6, 1, true).is_err());
        let drained = Cell::new(false);
        owner.as_mut().unwrap().finish(0, 5, |_, _| {
            drained.set(true);
            Ok(vec![check(1, 1, vec![0; 16], vec![0; 16])])
        }).unwrap();
        assert!(drained.get());
        owner.as_ref().unwrap().finish_span().unwrap();
        drop(owner.take());
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn disabled_checks_skip_reads_and_normal_mismatches_keep_broadcast() {
        let mut checks = SlotRouteChecks::new(vec![vec![check(1, 1, vec![0; 16], vec![1; 16])]], false).unwrap();
        checks.begin(0, 0, 1, false).unwrap();
        assert!(checks.storage(0, 0).unwrap().is_none());
        assert!(checks.finish(0, 0, |_, _| unreachable!()).unwrap().is_none());
        checks.begin(0, 1, 1, true).unwrap();
        let identity = checks.finish(0, 1, |storage, _| Ok(storage.clone())).unwrap().unwrap();
        assert_eq!(identity.source(), RouteSource::Broadcast);
        checks.finish_span().unwrap();
    }

    #[test]
    fn invalid_route_shapes_fail_without_releasing_the_slot() {
        let mut checks = SlotRouteChecks::new(vec![()], true).unwrap();
        checks.begin(0, 0, 1, true).unwrap();
        assert!(checks.finish(0, 0, |_, _| Ok(vec![check(1, 1, vec![], vec![])])).is_err());
        assert!(checks.finish_span().is_err());
        assert!(checks.storage(1, 0).is_err());
        assert!(SlotRouteChecks::<()>::new(vec![], true).is_err());
    }

    #[test]
    fn identical_routes_stay_replicated_and_a_flip_names_layer_and_row() {
        let rows = 3;
        let bytes: Vec<u8> = (0..rows * 2 * 8).map(|i| i as u8).collect();
        let same = RouteIdentity::check(&[check(4, rows, bytes.clone(), bytes.clone())]);
        assert_eq!((same.layers_checked, same.rows_checked, same.first_mismatch), (1, 3, None));
        assert_eq!(same.source(), RouteSource::Replicated);
        let mut weight = bytes.clone();
        // Row 2's first weight (after the ids section).
        weight[rows * 8 + 2 * 8] ^= 1;
        let differ = RouteIdentity::check(&[check(4, rows, bytes.clone(), bytes.clone()), check(5, rows, bytes, weight)]);
        assert_eq!(differ.first_mismatch, Some((5, 2)));
        assert_eq!(differ.source(), RouteSource::Broadcast);
    }
}
