//! One drafter Movable for every family (#65, #106).
//!
//! A family describes its drafter once, from the checkpoint and the serving
//! batch: the bytes it owns wherever it lives, the bytes that follow the home
//! it is given (SM-sized scratch, home-local rings), and, when the family has
//! a device-driven bridge to run it away from the target head, the bridge's
//! own staging and exchange on both GPUs. [`DrafterDemand::movable`] turns that
//! into the solver's [`Movable`]; the solver picks the home (pool first) and
//! the runtime reads it back with [`home`]. Planner and runtime build the same
//! demand, so the ledger names the drafter identically on both sides.
use super::{Bytes2, Demand, Movable, MovableDemands, MovableId, Placement, PlacementError};

/// Controls a device-driven draft bridge owns on each GPU: four zero-payload
/// receive slots (256 B each), the control block and the abort word. No
/// duplicate payload destination is allocated.
pub const BRIDGE_EXCHANGE_BYTES: u64 = BRIDGE_SLOTS * 256 + 256 + 256;
/// Receive slots of a draft bridge: context, input rows, hidden rows, logits.
pub const BRIDGE_SLOTS: u64 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrafterDemand {
    /// `Drafter` for owned weights; `DsparkExperts` for stage experts loaded
    /// into the home's TP1 expert arena.
    pub id: MovableId,
    /// Loaded in order wherever the drafter lives; staging is transient.
    /// Charged as one `Drafter` (or arena) item on the home.
    pub parts: Vec<Bytes2>,
    /// Named allocations that follow the home (their `gpu` is rewritten to
    /// each candidate home), for families whose ledger itemizes the drafter.
    pub owned: Vec<Demand>,
    /// Shares the home's local expert arena (workspace and load peak).
    pub expert_arena: bool,
    /// The GPU whose target head, embedding and taps the drafter reads.
    pub target_gpu: u8,
    /// Demands that exist only on the chosen home, keyed by that home.
    pub per_home: Vec<MovableDemands>,
    /// Target-side staging and exchange slots of the device-driven bridge
    /// that runs the drafter on the other GPU. `None` keeps it beside the target.
    pub bridge: Option<Vec<Demand>>,
}

impl DrafterDemand {
    /// A drafter that always lives beside the target head.
    pub fn local(id: MovableId, parts: Vec<Bytes2>, target_gpu: u8) -> Self {
        Self { id, parts, owned: Vec::new(), expert_arena: id == MovableId::DsparkExperts, target_gpu,
            per_home: Vec::new(), bridge: None }
    }

    /// A drafter beside the target head whose ledger itemizes every allocation.
    pub fn itemized(owned: Vec<Demand>, target_gpu: u8) -> Self {
        Self { owned, ..Self::local(MovableId::Drafter, Vec::new(), target_gpu) }
    }

    /// The solver's view on `gpus` coordinator GPUs. Candidate homes are the
    /// target GPU, plus the other GPU when a bridge exists. Each home's
    /// conditional demands are its bridge (remote homes only), the owned
    /// demands placed on it, then its own per-home demands; a home with none
    /// carries no entry.
    pub fn movable(&self, gpus: usize) -> Result<Movable, PlacementError> {
        let target = usize::from(self.target_gpu);
        if !(1..=2).contains(&gpus) || target >= gpus {
            return Err(PlacementError::Inventory("drafter target on an absent GPU"));
        }
        if self.bridge.is_some() && gpus != 2 {
            return Err(PlacementError::Inventory("drafter bridge needs two GPUs"));
        }
        let mut keys = Vec::new();
        for home in &self.per_home {
            if usize::from(home.placement_gpu) >= gpus || keys.contains(&home.placement_gpu)
                || home.demands.iter().any(|d| usize::from(d.gpu) >= gpus) {
                return Err(PlacementError::Inventory("drafter per-home demand"));
            }
            keys.push(home.placement_gpu);
        }
        if self.bridge.iter().flatten().any(|d| usize::from(d.gpu) >= gpus) {
            return Err(PlacementError::Inventory("drafter bridge demand on an absent GPU"));
        }
        let remote = (self.bridge.is_some()).then(|| 1 - self.target_gpu);
        let conditional = (0..gpus as u8).filter_map(|home| {
            let mut demands = if Some(home) == remote { self.bridge.clone().unwrap_or_default() } else { Vec::new() };
            if home == self.target_gpu || Some(home) == remote {
                demands.extend(self.owned.iter().cloned().map(|d| d.on(home)));
            }
            if let Some(own) = self.per_home.iter().find(|d| d.placement_gpu == home) {
                demands.extend(own.demands.iter().cloned());
            }
            (!demands.is_empty()).then_some(MovableDemands { placement_gpu: home, demands })
        }).collect();
        let mut allowed: Vec<u8> = std::iter::once(self.target_gpu).chain(remote).collect();
        allowed.sort_unstable();
        Ok(Movable { id: self.id, parts: self.parts.clone(), allowed, expert_arena: self.expert_arena, conditional })
    }
}

/// Test oracle: the same request with every movable charged as fixed demands
/// on its first allowed GPU (how families charged drafters before #65).
#[cfg(test)]
pub(crate) fn as_fixed(mut request: super::PlacementRequest) -> super::PlacementRequest {
    use cuteafd_core::memory_layout::{Basis, Category};
    for movable in std::mem::take(&mut request.movables) {
        let home = movable.allowed[0];
        let total: u64 = movable.parts.iter().map(|p| p.resident).sum();
        if total > 0 {
            request.fixed.push(Demand::new(home, Category::Drafter, format!("{:?}", movable.id), total, Basis::Exact));
        }
        if let Some(own) = movable.conditional.into_iter().find(|c| c.placement_gpu == home) {
            request.fixed.extend(own.demands);
        }
    }
    request
}

/// Placement equality with [`as_fixed`]: everything but the movable list and
/// the per-GPU item order, which the drafter's own items move to the end.
#[cfg(test)]
pub(crate) fn assert_same_as_fixed(request: &super::PlacementRequest) {
    let moved = super::solve(request).unwrap();
    let fixed = super::solve(&as_fixed(request.clone())).unwrap();
    assert_eq!((moved.pool_tokens, &moved.layers, moved.tp2, &moved.expert_ranges, moved.onboard_layers),
        (fixed.pool_tokens, &fixed.layers, fixed.tp2, &fixed.expert_ranges, fixed.onboard_layers));
    for (a, b) in moved.items.iter().zip(&fixed.items) {
        let key = |items: &[cuteafd_core::memory_layout::Item]| {
            let mut v: Vec<_> = items.iter().map(|i| (i.category, i.group.clone(), i.bytes)).collect();
            v.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
            v
        };
        assert_eq!(key(a), key(b));
    }
    assert!(moved.movables.iter().all(|(_, gpu)| request.movables.iter().any(|m| m.allowed[0] == *gpu)));
}

/// The GPU the solver gave the drafter (`Drafter` or `DsparkExperts`), if any.
pub fn home(placement: &Placement) -> Option<u8> {
    placement.movables.iter().find(|(id, _)| matches!(id, MovableId::Drafter | MovableId::DsparkExperts))
        .map(|(_, gpu)| *gpu)
}

/// Refuse a placement whose drafter home this executor cannot run.
pub fn require_home(placement: &Placement, family: &'static str, executable: &[u8]) -> Result<Option<u8>, PlacementError> {
    match home(placement) {
        Some(gpu) if !executable.contains(&gpu) =>
            Err(PlacementError::DrafterHome { family, gpu }),
        home => Ok(home),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::memory_layout::{Basis, Category};

    fn scratch(gpu: u8, bytes: u64) -> MovableDemands {
        MovableDemands { placement_gpu: gpu, demands: vec![Demand::new(gpu, Category::Workspace, "scratch", bytes, Basis::Formula)] }
    }

    #[test]
    fn local_drafter_is_pinned_to_the_target_and_dormant_homes_stay_keyed() {
        let mut d = DrafterDemand::local(MovableId::Drafter, vec![Bytes2 { resident: 10, staging: 0 }], 0);
        d.per_home = vec![scratch(0, 3), scratch(1, 4)];
        let m = d.movable(2).unwrap();
        assert_eq!(m.allowed, [0]);
        assert_eq!(m.conditional.iter().map(|c| c.placement_gpu).collect::<Vec<_>>(), [0, 1]);
        assert!(!m.expert_arena);
        let v4 = DrafterDemand::local(MovableId::DsparkExperts, vec![Bytes2 { resident: 5, staging: 2 }], 0).movable(2).unwrap();
        assert_eq!((v4.allowed, v4.expert_arena, v4.conditional.len()), (vec![0], true, 0));
    }

    #[test]
    fn bridge_opens_the_other_gpu_and_precedes_its_scratch() {
        let mut d = DrafterDemand::local(MovableId::Drafter, vec![Bytes2 { resident: 10, staging: 0 }], 0);
        d.per_home = vec![scratch(1, 4)];
        d.bridge = Some(vec![Demand::new(0, Category::Drafter, "bridge", 7, Basis::Formula)]);
        let m = d.movable(2).unwrap();
        assert_eq!(m.allowed, [0, 1]);
        assert_eq!(m.conditional.len(), 1);
        assert_eq!(m.conditional[0].demands.iter().map(|d| (d.gpu, d.bytes)).collect::<Vec<_>>(), [(0, 7), (1, 4)]);
        // A head on GPU1 (V4.1) bridges to GPU0.
        d.target_gpu = 1;
        assert_eq!(d.movable(2).unwrap().allowed, [0, 1]);
        assert!(d.movable(1).is_err());
    }

    #[test]
    fn itemized_demands_follow_every_candidate_home_and_keep_their_names() {
        let mut d = DrafterDemand::itemized(vec![Demand::new(0, Category::Drafter, "draft.weights", 11, Basis::Exact),
            Demand::new(0, Category::Workspace, "draft.scratch", 2, Basis::Formula)], 0);
        let m = d.movable(2).unwrap();
        assert_eq!((m.allowed.clone(), m.parts.len()), (vec![0], 0));
        assert_eq!(m.conditional.len(), 1);
        assert_eq!(m.conditional[0].demands.iter().map(|d| (d.gpu, d.group.as_str())).collect::<Vec<_>>(),
            [(0, "draft.weights"), (0, "draft.scratch")]);
        d.bridge = Some(vec![Demand::new(0, Category::Transport, "bridge", 1, Basis::Formula)]);
        let m = d.movable(2).unwrap();
        assert_eq!(m.conditional[1].demands.iter().map(|d| (d.gpu, d.group.as_str())).collect::<Vec<_>>(),
            [(0, "bridge"), (1, "draft.weights"), (1, "draft.scratch")]);
    }

    #[test]
    fn invalid_inventory_is_refused() {
        let mut d = DrafterDemand::local(MovableId::Drafter, vec![], 1);
        assert!(d.movable(1).is_err());
        d.target_gpu = 0;
        d.per_home = vec![scratch(0, 1), scratch(0, 1)];
        assert!(d.movable(2).is_err());
    }

    /// The pre-shared GLM Flash construction (#94), kept verbatim as the
    /// migration oracle: the shared builder must produce the same Movable.
    fn glmf_legacy(bytes: u64, scratch: &[u64], bridge: Option<&Vec<Demand>>) -> Movable {
        let conditional = scratch.iter().enumerate().filter_map(|(rank, &bytes)| {
            let mut demands = if rank == 1 { bridge.cloned().unwrap_or_default() } else { vec![] };
            if bytes > 0 { demands.push(Demand::new(rank as u8, Category::Workspace,
                "DFlash linear/attention scratch", bytes, Basis::Formula)); }
            (!demands.is_empty()).then_some(MovableDemands { placement_gpu: rank as u8, demands })
        }).collect();
        Movable { id: MovableId::Drafter, parts: vec![Bytes2 { resident: bytes, staging: 0 }],
            allowed: if bridge.is_some() { vec![0, 1] } else { vec![0] }, expert_arena: false, conditional }
    }

    #[test]
    fn shared_builder_reproduces_glm_flash_and_v4_movables() {
        let bridge = vec![Demand::new(0, Category::Drafter, "DFlash bridge taps", 9, Basis::Formula),
            Demand::new(0, Category::Transport, "DFlash bridge exchange", BRIDGE_EXCHANGE_BYTES, Basis::Formula),
            Demand::new(1, Category::Transport, "DFlash bridge exchange", BRIDGE_EXCHANGE_BYTES, Basis::Formula)];
        for scratch in [vec![0u64], vec![5], vec![0, 0], vec![5, 0], vec![0, 6], vec![5, 6]] {
            for with_bridge in [false, true] {
                if with_bridge && scratch.len() != 2 { continue; }
                let mut d = DrafterDemand::local(MovableId::Drafter, vec![Bytes2 { resident: 100, staging: 0 }], 0);
                d.per_home = scratch.iter().enumerate().filter(|(_, &b)| b > 0).map(|(rank, &b)| MovableDemands {
                    placement_gpu: rank as u8, demands: vec![Demand::new(rank as u8, Category::Workspace,
                        "DFlash linear/attention scratch", b, Basis::Formula)] }).collect();
                d.bridge = with_bridge.then(|| bridge.clone());
                assert_eq!(d.movable(scratch.len()).unwrap(), glmf_legacy(100, &scratch, d.bridge.as_ref()),
                    "scratch {scratch:?} bridge {with_bridge}");
            }
        }
        let parts = vec![Bytes2 { resident: 7, staging: 3 }, Bytes2 { resident: 8, staging: 4 }];
        for gpus in [1, 2] {
            assert_eq!(DrafterDemand::local(MovableId::DsparkExperts, parts.clone(), 0).movable(gpus).unwrap(),
                Movable { id: MovableId::DsparkExperts, allowed: vec![0], expert_arena: true, conditional: vec![], parts: parts.clone() });
        }
    }
}
