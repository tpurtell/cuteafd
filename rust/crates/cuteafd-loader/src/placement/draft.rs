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
use super::{Bytes2, Demand, Movable, MovableDemands, MovableId, MovablePart, Placement, PlacementError, ResidentWeights};

/// Controls a device-driven draft bridge owns on each GPU: four zero-payload
/// receive slots (256 B each), the control block and the abort word. No
/// duplicate payload destination is allocated.
pub const BRIDGE_EXCHANGE_BYTES: u64 = BRIDGE_SLOTS * BRIDGE_SLOT_BYTES as u64 + 256 + 256;
/// Receive slots of a draft bridge: context, input rows, hidden rows, logits.
pub const BRIDGE_SLOTS: u64 = 4;
/// Minimal receive allocation for each control-only slot; payloads go directly
/// to their existing owners, but the exchange still needs valid receive slots.
pub const BRIDGE_SLOT_BYTES: usize = 256;
pub const BRIDGE_SLOT_SIZES: [usize; BRIDGE_SLOTS as usize] = [BRIDGE_SLOT_BYTES; BRIDGE_SLOTS as usize];

/// Rank-1 drafter bridge with embedding/tap staging on rank 0. The head's
/// owning rank is explicit: a rank-1 head reuses the drafter's hidden rows,
/// logits and head scratch instead of allocating duplicate rank-0 storage.
/// `rows` is the full admitted draft batch, not a head launch chunk.
pub fn remote_dflash_bridge_demands(hidden: u64, vocab: u64, taps: u64, rows: u64, head_rank: u8)
    -> Result<MovableDemands, PlacementError> {
    use cuteafd_core::memory_layout::{Basis, Category};
    if hidden == 0 || vocab == 0 || taps == 0 || rows == 0 || rows > i32::MAX as u64 || head_rank > 1 {
        return Err(PlacementError::Inventory("remote DFlash bridge geometry or head owner"));
    }
    let mul = |a: u64, b: u64| a.checked_mul(b)
        .ok_or(PlacementError::Overflow("remote DFlash bridge"));
    let hidden_rows = mul(mul(rows, hidden)?, 2)?.max(256);
    let mut bytes = vec![
        ("DFlash bridge taps", mul(mul(mul(2048, taps)?, hidden)?, 2)?.max(256)),
        ("DFlash bridge input ids", mul(rows, 4)?.max(256)),
        ("DFlash bridge gathered input", hidden_rows),
    ];
    if head_rank == 0 {
        bytes.extend([
            ("DFlash bridge normalized hidden", hidden_rows),
            ("DFlash bridge logits", mul(mul(rows, vocab)?, 4)?.max(256)),
            ("DFlash bridge head scratch", 4 << 20),
        ]);
    }
    let mut demands: Vec<_> = bytes.into_iter().map(|(group, bytes)|
        Demand::new(0, Category::Drafter, group, bytes, Basis::Formula)).collect();
    for gpu in 0..2 {
        demands.push(Demand::new(gpu, Category::Transport, "DFlash bridge exchange",
            BRIDGE_EXCHANGE_BYTES, Basis::Formula));
    }
    for gpu in 0..2 {
        demands.iter().filter(|d| d.gpu == gpu).try_fold(0u64, |sum, d|
            sum.checked_add(d.bytes).ok_or(PlacementError::Overflow("remote DFlash bridge")))?;
    }
    Ok(MovableDemands { placement_gpu: 1, demands })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrafterDemand {
    /// `Drafter` for owned weights; `DsparkExperts` for stage experts loaded
    /// into the home's TP1 expert arena.
    pub id: MovableId,
    /// Loaded in order wherever the drafter lives; staging is transient.
    /// Charged as one `Drafter` (or arena) item on the home.
    pub parts: Vec<MovablePart>,
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
    pub fn local(id: MovableId, parts: Vec<MovablePart>, target_gpu: u8) -> Self {
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

impl crate::formats::dflash::MlaDraftCacheLayout {
    /// Both offline and measured admission must consume these exact charges
    /// before allocating the cache. This is only the cache portion of the
    /// drafter inventory, not weights, activations or native scratch.
    pub fn demands(&self, gpu: u8) -> Result<Vec<Demand>, PlacementError> {
        use cuteafd_core::memory_layout::{Basis, Category};
        let retained = self.cache_bytes.checked_sub(self.margin_bytes)
            .ok_or(PlacementError::Inventory("MLA draft cache margin exceeds physical cache"))?;
        Ok(vec![
            Demand::new(gpu, Category::Drafter, "MLA DFlash retained latent cache",
                retained, Basis::Formula),
            Demand::new(gpu, Category::Drafter, "MLA DFlash per-slot private block and page padding",
                self.margin_bytes, Basis::Formula),
            Demand::new(gpu, Category::Drafter, "MLA DFlash virtual page tables",
                self.page_table_bytes, Basis::Formula),
        ])
    }
}

impl crate::formats::dflash::MlaDraftOwner {
    /// Translate a shared drafter descriptor once; live receipts still require
    /// an independent retained-field census rather than projecting this list.
    pub fn allocation(&self, id: String, gpu: u8, lifetime: super::planner::Lifetime,
        group: &str) -> super::planner::Allocation {
        use crate::formats::dflash::{MlaDraftDomain, MlaDraftDtype};
        use cuteafd_core::memory_layout::Category;
        use super::planner::{Allocation, Location};
        let mut allocation = Allocation::buffer(id, match self.domain {
            MlaDraftDomain::Device => Location::coordinator(gpu),
            MlaDraftDomain::HostPinned => Location::PINNED,
        }, self.bytes as u64, lifetime, Category::Drafter, group);
        allocation.alignment = self.alignment as u64;
        allocation.dtype = match self.dtype {
            MlaDraftDtype::Bytes => "bytes", MlaDraftDtype::Bf16 => "bf16", MlaDraftDtype::F32 => "f32",
            MlaDraftDtype::I32 => "i32", MlaDraftDtype::U32 => "u32", MlaDraftDtype::I64 => "i64",
        }.into();
        allocation.shape = self.shape.iter().map(|&n| n as u64).collect();
        allocation
    }
}

impl crate::families::glm5::draft_representation::GlmDraftRuntimeLayout {
    /// GQA owners join the family's resolved contract; this is not a separate
    /// capacity admission or a Ready-authorizing legacy bridge.
    pub fn gqa_facts(&self, gpu: u8, sms: u64, mode: u8, kv_cache: crate::families::qwen4::Qwen4KvCache)
        -> Result<super::planner::Facts, crate::families::glm5::draft_representation::GlmDraftStorageError> {
        use super::planner::{Facts, Lifetime, Phase, Sequential};
        let mut allocations = self.device_owners(sms, mode, kv_cache)?.iter().map(|owner| {
            let load = owner.id.starts_with("weight.") || owner.id.starts_with("layer.")
                || ["taps", "fused", "fused_norm", "context_positions", "context_slots", "context_kv", "fp8_scratch"]
                    .contains(&owner.id.as_str());
            owner.allocation(format!("draft.gqa.{}", owner.id), gpu,
                Lifetime { first: if load { Phase::Load } else { Phase::Warmup }, last: Phase::Growth },
                "GQA DFlash owner")
        }).collect::<Vec<_>>();
        allocations.extend(self.pinned_owners().iter().filter(|owner| !owner.on_use).map(|owner|
            owner.allocation(format!("draft.gqa.{}", owner.id), gpu,
                Lifetime { first: Phase::Warmup, last: Phase::Growth }, "GQA DFlash owner")));
        let members: Vec<_> = self.packing_sources().iter().map(|owner| owner.allocation(
            format!("draft.gqa.{}", owner.id), gpu,
            Lifetime { first: Phase::Load, last: Phase::Load }, "BF16 packing source")).collect();
        Ok(Facts { phase_order: crate::placement::planner::PhaseOrder::TARGET_THEN_DRAFT, allocations, deferred: vec![], reservations: vec![], allowances: vec![],
            sequential: if members.is_empty() { vec![] } else {
                vec![Sequential { id: "draft.gqa.packing".into(), members }]
            } })
    }
}

impl crate::formats::dflash::MlaDraftRuntimeLayout {
    /// Temporary forecast for the legacy Kimi owner-only bridge. The family
    /// admits the shared synchronous upload store and diagnostic on-use stores;
    /// neither is a private drafter owner. Full Kimi migration deletes this API.
    pub fn legacy_mla_facts(&self, gpu: u8) -> super::planner::Facts {
        use crate::formats::dflash::MlaDraftLifetime;
        use super::planner::{Facts, Lifetime, Phase, Sequential};
        let allocations = self.owners().iter().filter(|o| !o.on_use).map(|owner| {
            let first = match owner.lifetime {
                MlaDraftLifetime::Runtime => Phase::Load,
                MlaDraftLifetime::Library => Phase::Warmup,
            };
            owner.allocation(format!("draft.mla.{}", owner.id), gpu,
                Lifetime { first, last: Phase::Growth }, "MLA DFlash owner")
        }).collect();
        let members: Vec<_> = self.packing_sources().iter().map(|owner| owner.allocation(
            format!("draft.mla.{}", owner.id), gpu,
            Lifetime { first: Phase::Load, last: Phase::Load }, "BF16 packing source")).collect();
        Facts { phase_order: crate::placement::planner::PhaseOrder::TARGET_THEN_DRAFT, allocations, deferred: vec![], reservations: vec![], allowances: vec![],
            sequential: if members.is_empty() { vec![] } else {
                vec![Sequential { id: "draft.mla.packing".into(), members }]
            } }
    }
}

/// Test oracle: the same request with every movable charged as fixed demands
/// on its first allowed GPU (how families charged drafters before #65).
#[cfg(test)]
pub(crate) fn as_fixed(mut request: super::PlacementRequest) -> super::PlacementRequest {
    use cuteafd_core::memory_layout::{Basis, Category};
    for movable in std::mem::take(&mut request.movables) {
        let home = movable.allowed[0];
        let total: u64 = movable.parts.iter().map(|p| p.memory.resident).sum();
        let weights = movable.parts.iter().try_fold(ResidentWeights::default(), |sum, p| sum.checked_add(p.resident_weights)).unwrap();
        if total > 0 || weights.0 > 0 {
            request.fixed.push(Demand::new(home, Category::Drafter, format!("{:?}", movable.id), total, Basis::Exact)
                .with_resident_weights(weights));
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

    // Frozen pre-extraction GLM Flash formula: compare every demand, not just totals.
    fn legacy_glm_flash_bridge(hidden: u64, vocab: u64, taps: u64, rows: u64)
        -> Result<MovableDemands, PlacementError> {
        if hidden == 0 || vocab == 0 || taps == 0 || rows == 0 || rows > i32::MAX as u64 {
            return Err(PlacementError::Inventory("GLM Flash remote DFlash bridge geometry"));
        }
        let mul = |a: u64, b: u64| a.checked_mul(b)
            .ok_or(PlacementError::Overflow("GLM Flash remote DFlash bridge"));
        let hidden_rows = mul(mul(rows, hidden)?, 2)?.max(256);
        let bytes = [
            ("DFlash bridge taps", mul(mul(mul(2048, taps)?, hidden)?, 2)?.max(256)),
            ("DFlash bridge input ids", mul(rows, 4)?.max(256)),
            ("DFlash bridge gathered input", hidden_rows),
            ("DFlash bridge normalized hidden", hidden_rows),
            ("DFlash bridge logits", mul(mul(rows, vocab)?, 4)?.max(256)),
            ("DFlash bridge head scratch", 4 << 20),
        ];
        let mut demands: Vec<_> = bytes.into_iter().map(|(group, bytes)|
            Demand::new(0, Category::Drafter, group, bytes, Basis::Formula)).collect();
        for gpu in 0..2 {
            demands.push(Demand::new(gpu, Category::Transport, "DFlash bridge exchange",
                1536, Basis::Formula));
        }
        for gpu in 0..2 {
            demands.iter().filter(|d| d.gpu == gpu).try_fold(0u64, |sum, d|
                sum.checked_add(d.bytes).ok_or(PlacementError::Overflow("GLM Flash remote DFlash bridge")))?;
        }
        Ok(MovableDemands { placement_gpu: 1, demands })
    }

    #[test]
    fn bridge_formula_reproduces_legacy_glm_flash_allocations_exactly() {
        for (hidden, vocab, taps) in [(4096, 154_880, 5), (4096, 154_880, 6)] {
            for rows in [1, 8, 16, 32, 64, 128] {
                assert_eq!(remote_dflash_bridge_demands(hidden, vocab, taps, rows, 0).unwrap(),
                    legacy_glm_flash_bridge(hidden, vocab, taps, rows).unwrap());
            }
        }
        assert_eq!(remote_dflash_bridge_demands(1, 1, 1, 1, 0).unwrap(),
            legacy_glm_flash_bridge(1, 1, 1, 1).unwrap());
    }

    #[test]
    fn draft_local_head_charges_only_new_bridge_allocations() {
        let (hidden, vocab, taps, rows) = (7168, 163_840, 5, 64);
        let bridge = remote_dflash_bridge_demands(hidden, vocab, taps, rows, 1).unwrap();
        assert_eq!(bridge.placement_gpu, 1);
        assert_eq!(bridge.demands, vec![
            Demand::new(0, Category::Drafter, "DFlash bridge taps", 146_800_640, Basis::Formula),
            Demand::new(0, Category::Drafter, "DFlash bridge input ids", 256, Basis::Formula),
            Demand::new(0, Category::Drafter, "DFlash bridge gathered input", 917_504, Basis::Formula),
            Demand::new(0, Category::Transport, "DFlash bridge exchange", 1536, Basis::Formula),
            Demand::new(1, Category::Transport, "DFlash bridge exchange", 1536, Basis::Formula),
        ]);
        let head0 = remote_dflash_bridge_demands(hidden, vocab, taps, rows, 0).unwrap();
        let total = |d: &MovableDemands| d.demands.iter().map(|d| d.bytes).sum::<u64>();
        assert_eq!(total(&head0) - total(&bridge), rows * hidden * 2 + rows * vocab * 4 + (4 << 20));
        // No logits payload is allocated by this bridge when the head is local.
        assert!(remote_dflash_bridge_demands(1, u64::MAX, 1, 1, 1).is_ok());
    }

    #[test]
    fn shared_bridge_rejects_invalid_owner_and_geometry_or_allocation_overflow() {
        for head in [0, 1] {
            for (h, v, t, r) in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1),
                (1, 1, 1, 0), (1, 1, 1, i32::MAX as u64 + 1)] {
                assert!(matches!(remote_dflash_bridge_demands(h, v, t, r, head), Err(PlacementError::Inventory(_))));
            }
            for (h, t, r) in [(u64::MAX, 1, 1), (1, u64::MAX, 1), (u64::MAX / 4096, 1, 1)] {
                assert!(matches!(remote_dflash_bridge_demands(h, 1, t, r, head), Err(PlacementError::Overflow(_))));
            }
        }
        for head in [2, u8::MAX] {
            assert!(matches!(remote_dflash_bridge_demands(1, 1, 1, 1, head), Err(PlacementError::Inventory(_))));
        }
    }

    fn scratch(gpu: u8, bytes: u64) -> MovableDemands {
        MovableDemands { placement_gpu: gpu, demands: vec![Demand::new(gpu, Category::Workspace, "scratch", bytes, Basis::Formula)] }
    }

    #[test]
    fn local_drafter_is_pinned_to_the_target_and_dormant_homes_stay_keyed() {
        let mut d = DrafterDemand::local(MovableId::Drafter, vec![MovablePart::new(Bytes2 { resident: 10, staging: 0 }, ResidentWeights(10))], 0);
        d.per_home = vec![scratch(0, 3), scratch(1, 4)];
        let m = d.movable(2).unwrap();
        assert_eq!(m.allowed, [0]);
        assert_eq!(m.conditional.iter().map(|c| c.placement_gpu).collect::<Vec<_>>(), [0, 1]);
        assert!(!m.expert_arena);
        let v4 = DrafterDemand::local(MovableId::DsparkExperts, vec![MovablePart::new(Bytes2 { resident: 5, staging: 2 }, ResidentWeights(5))], 0).movable(2).unwrap();
        assert_eq!((v4.allowed, v4.expert_arena, v4.conditional.len()), (vec![0], true, 0));
    }

    #[test]
    fn bridge_opens_the_other_gpu_and_precedes_its_scratch() {
        let mut d = DrafterDemand::local(MovableId::Drafter, vec![MovablePart::new(Bytes2 { resident: 10, staging: 0 }, ResidentWeights(10))], 0);
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
        Movable { id: MovableId::Drafter, parts: vec![MovablePart::new(Bytes2 { resident: bytes, staging: 0 }, ResidentWeights(bytes))],
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
                let mut d = DrafterDemand::local(MovableId::Drafter, vec![MovablePart::new(Bytes2 { resident: 100, staging: 0 }, ResidentWeights(100))], 0);
                d.per_home = scratch.iter().enumerate().filter(|(_, &b)| b > 0).map(|(rank, &b)| MovableDemands {
                    placement_gpu: rank as u8, demands: vec![Demand::new(rank as u8, Category::Workspace,
                        "DFlash linear/attention scratch", b, Basis::Formula)] }).collect();
                d.bridge = with_bridge.then(|| bridge.clone());
                assert_eq!(d.movable(scratch.len()).unwrap(), glmf_legacy(100, &scratch, d.bridge.as_ref()),
                    "scratch {scratch:?} bridge {with_bridge}");
            }
        }
        let parts = vec![MovablePart::new(Bytes2 { resident: 7, staging: 3 }, ResidentWeights(7)), MovablePart::new(Bytes2 { resident: 8, staging: 4 }, ResidentWeights(8))];
        for gpus in [1, 2] {
            assert_eq!(DrafterDemand::local(MovableId::DsparkExperts, parts.clone(), 0).movable(gpus).unwrap(),
                Movable { id: MovableId::DsparkExperts, allowed: vec![0], expert_arena: true, conditional: vec![], parts: parts.clone() });
        }
    }
}
