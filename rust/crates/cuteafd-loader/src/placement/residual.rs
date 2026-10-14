//! Per-layer ownership (PLAN "v3 placement: design", section 3): where the
//! residual lives at each layer boundary, and the hops that move it when the
//! owner changes. The solver derives [`Hop`]s from the chosen [`LayerMode`]s
//! and charges their persistent transport storage as fixed demands; executors follow the
//! same list (the daemon's hop primitive, `shared::peer_split::HopLink`).
use super::{FfnMode, LayerMode, Placement};
use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Which GPU holds the residual (the mHC streams, or `[T,H]`) between layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "home", content = "gpu")]
pub enum ResidualHome {
    /// Both GPUs hold bitwise-identical copies (after a head split or a
    /// split FFN's all-reduce).
    Replicated,
    /// Only this GPU holds it.
    Owned(u8),
}

/// Where in the layer stack a hop runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "at", content = "layer")]
pub enum HopPoint {
    /// Before layer 0: the embedding rows from the GPU that gathered them.
    /// Lands in the destination's step input (both streams drain at a step
    /// start), so it charges no receive buffer.
    Entry,
    /// Before this layer's attention (an ownership boundary or a broadcast
    /// into a head split).
    BeforeLayer(usize),
    /// After this layer's attention, before its split FFN: the owner's FFN
    /// input and residual out to the peer.
    AfterAttention(usize),
    /// After the last layer, into the head's GPU.
    Exit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HopKind {
    /// The owner copies to the peer and keeps its own (the residual becomes
    /// Replicated).
    Broadcast,
    /// Ownership moves from one GPU to the other.
    Boundary,
}

/// One residual move between the two GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hop {
    pub at: HopPoint,
    pub from: u8,
    pub to: u8,
    pub kind: HopKind,
    /// Bytes per token row: `[H]` BF16, or `[4,H]` BF16 + `[4]` FP32
    /// pre-coefficients for mHC families.
    pub row_bytes: u64,
}

impl Hop {
    /// Whether this hop needs a receive buffer on `to` (every hop but the
    /// entry, which lands in the step input).
    pub fn charged(&self) -> bool {
        self.at != HopPoint::Entry
    }
}

/// Explicit final-layer output contract, before any exit hop to the head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalHome {
    /// Split FFN postprocessing produces both residual copies.
    Replicated,
    /// Split FFN postprocessing runs only on the head GPU.
    Head,
    /// Owner FFN postprocessing leaves the residual on the last attention owner.
    AttentionOwner,
}

/// What the family's executor moves per hop; part of the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HopSpec {
    /// Bytes per row at entry, ownership boundaries and exit.
    pub row_bytes: u64,
    /// Complete FFN handoff width; None uses the residual width.
    pub after_attention_row_bytes: Option<u64>,
    /// The family packs into one persistent sender buffer per lane/rank.
    pub packed_sender: bool,
    /// Rows a hop carries at most (the larger of prefill and decode rows).
    pub rows: u64,
    /// Lanes that can have a hop in flight at once (prefill lanes).
    pub lanes: u64,
    /// The GPU the embedding rows land on before layer 0.
    pub entry_gpu: u8,
    /// The GPU the head runs on after the last layer.
    pub head_gpu: u8,
    /// Required even when no hop storage is needed; no inferred terminal default.
    pub terminal: TerminalHome,
}

impl HopSpec {
    pub fn row_bytes_at(&self, point: HopPoint) -> u64 {
        if matches!(point, HopPoint::AfterAttention(_)) {
            self.after_attention_row_bytes.unwrap_or(self.row_bytes)
        } else { self.row_bytes }
    }

    fn terminal_home(&self, mode: LayerMode) -> Result<ResidualHome, LayerPlanError> {
        match (self.terminal, mode) {
            (TerminalHome::AttentionOwner, LayerMode::Whole { gpu, ffn: FfnMode::Owner }) => Ok(ResidualHome::Owned(gpu)),
            (TerminalHome::Head, LayerMode::Whole { gpu, ffn: FfnMode::Owner }) if gpu == self.head_gpu => Ok(ResidualHome::Owned(gpu)),
            (TerminalHome::Replicated, LayerMode::HeadSplit | LayerMode::ContextSplit
                | LayerMode::Whole { ffn: FfnMode::Split, .. }) => Ok(ResidualHome::Replicated),
            (TerminalHome::Head, LayerMode::HeadSplit | LayerMode::ContextSplit
                | LayerMode::Whole { ffn: FfnMode::Split, .. }) => Ok(ResidualHome::Owned(self.head_gpu)),
            _ => Err(LayerPlanError::Terminal { contract: self.terminal, mode }),
        }
    }
}

/// The same boundary homes are validated by admission and the runtime binding.
pub fn residual_homes(modes: &[LayerMode], spec: &HopSpec) -> Result<Vec<ResidualHome>, LayerPlanError> {
    if spec.entry_gpu >= 2 || spec.head_gpu >= 2 { return Err(LayerPlanError::Geometry); }
    let mut homes = vec![ResidualHome::Owned(spec.entry_gpu)];
    for (layer, &mode) in modes.iter().enumerate() {
        if let LayerMode::Whole { gpu, .. } = mode {
            if gpu >= 2 { return Err(LayerPlanError::Owner { layer, gpu }); }
        }
        let home = if layer + 1 == modes.len() { spec.terminal_home(mode)? }
            else { homes.last().expect("seeded").transition(mode).home };
        homes.push(home);
    }
    Ok(homes)
}

/// Receive slots per lane on a destination GPU: hops alternate between two
/// buffers by parity, as the FFN exchange slots do, so at most two are live.
pub const HOP_SLOTS: u64 = 2;

/// What crossing into a layer of `next` mode costs from `self`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transition {
    /// A hop before the layer's attention (`from`, `to`, kind).
    pub before: Option<(u8, u8, HopKind)>,
    /// A broadcast after the layer's attention (a split FFN's input).
    pub after_attention: Option<(u8, u8)>,
    /// The residual's home after the layer.
    pub home: ResidualHome,
}

impl ResidualHome {
    /// The state machine of PLAN section 3 for two GPUs:
    ///
    /// | from \ next  | HeadSplit         | Whole{g, Split}           | Whole{g, Owner} |
    /// |--------------|-------------------|---------------------------|-----------------|
    /// | Replicated   | none (all-reduce) | broadcast g after attn    | none            |
    /// | Owned(g)     | broadcast g       | broadcast g after attn    | none            |
    /// | Owned(other) | broadcast other   | boundary, then broadcast  | boundary        |
    ///
    /// A head split and a split FFN leave the residual Replicated; an owner
    /// FFN leaves it Owned(g). A `Whole{g, Split}` layer always broadcasts
    /// after attention: the peer's FFN half needs the input (the PLAN table
    /// lists only the inbound hop).
    pub fn transition(self, next: LayerMode) -> Transition {
        match next {
            LayerMode::HeadSplit | LayerMode::ContextSplit => Transition {
                before: match self {
                    Self::Replicated => None,
                    Self::Owned(g) => Some((g, 1 - g.min(1), HopKind::Broadcast)),
                },
                after_attention: None,
                home: Self::Replicated,
            },
            LayerMode::Whole { gpu, ffn } => {
                let before = match self {
                    Self::Owned(owner) if owner != gpu => Some((owner, gpu, HopKind::Boundary)),
                    _ => None,
                };
                match ffn {
                    FfnMode::Split => Transition { before, after_attention: Some((gpu, 1 - gpu.min(1))),
                        home: Self::Replicated },
                    FfnMode::Owner => Transition { before, after_attention: None, home: Self::Owned(gpu) },
                }
            }
        }
    }

    /// Whether `gpu` holds the residual.
    pub fn holds(self, gpu: u8) -> bool {
        match self {
            Self::Replicated => true,
            Self::Owned(owner) => owner == gpu,
        }
    }
}

/// Every hop of a stack of `modes` (in layer order), from the embedding on
/// `spec.entry_gpu` to the head on `spec.head_gpu`. One GPU never hops.
pub fn plan_hops(modes: &[LayerMode], spec: &HopSpec) -> Result<Vec<Hop>, LayerPlanError> {
    let homes = residual_homes(modes, spec)?;
    let hop = |at, (from, to, kind)| Hop { at, from, to, kind, row_bytes: spec.row_bytes_at(at) };
    let mut hops = Vec::new();
    let mut home = ResidualHome::Owned(spec.entry_gpu);
    for (layer, &mode) in modes.iter().enumerate() {
        let step = home.transition(mode);
        if let Some(before) = step.before {
            hops.push(hop(if layer == 0 { HopPoint::Entry } else { HopPoint::BeforeLayer(layer) }, before));
        }
        if let Some((from, to)) = step.after_attention {
            hops.push(hop(HopPoint::AfterAttention(layer), (from, to, HopKind::Broadcast)));
        }
        home = homes[layer + 1];
    }
    if !home.holds(spec.head_gpu) {
        if let ResidualHome::Owned(owner) = home {
            hops.push(hop(HopPoint::Exit, (owner, spec.head_gpu, HopKind::Boundary)));
        }
    }
    Ok(hops)
}

/// Invalid runtime binding of the solver's ownership decision.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LayerPlanError {
    #[error("layer execution needs a nonempty stack and logical GPUs 0/1")]
    Geometry,
    #[error("layer {layer} has invalid owner {gpu}")]
    Owner { layer: usize, gpu: u8 },
    #[error("terminal residual contract {contract:?} is incompatible with final mode {mode:?}")]
    Terminal { contract: TerminalHome, mode: LayerMode },
    #[error("runtime residual homes differ from the planner")]
    Residual,
    #[error("runtime residual hops differ from the planner")]
    Hops,
    #[error("layer {layer} attention state is bound to {actual:?}, expected {expected:?}")]
    StateOwners { layer: usize, actual: Vec<u8>, expected: Vec<u8> },
    #[error("layer {0} is past the execution stack")]
    Layer(usize),
}

/// Immutable contract shared by execution, graph keys and prefix snapshots.
/// The spec comes from the same family request that charged hop storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerExecutionPlan {
    modes: Vec<LayerMode>,
    hops: Vec<Hop>,
    homes: Vec<ResidualHome>,
    spec: HopSpec,
}

impl LayerExecutionPlan {
    pub fn new(placement: &Placement, spec: HopSpec) -> Result<Self, LayerPlanError> {
        if placement.layers.is_empty() || spec.entry_gpu >= 2 || spec.head_gpu >= 2
            || spec.rows == 0 || spec.lanes == 0 {
            return Err(LayerPlanError::Geometry);
        }
        let modes: Vec<_> = placement.layers.iter().map(|layer| layer.mode).collect();
        for (layer, &mode) in modes.iter().enumerate() {
            if let LayerMode::Whole { gpu, .. } = mode {
                if gpu >= 2 { return Err(LayerPlanError::Owner { layer, gpu }); }
            }
        }
        let homes = residual_homes(&modes, &spec)?;
        if placement.residual != homes { return Err(LayerPlanError::Residual); }
        let hops = plan_hops(&modes, &spec)?;
        if placement.hops != hops || hops.iter().any(|hop| hop.row_bytes == 0 || hop.row_bytes % 16 != 0) {
            return Err(LayerPlanError::Hops);
        }
        Ok(Self { modes, hops, homes, spec })
    }

    pub fn modes(&self) -> &[LayerMode] { &self.modes }
    pub fn hops(&self) -> &[Hop] { &self.hops }
    pub fn homes(&self) -> &[ResidualHome] { &self.homes }
    pub fn spec(&self) -> HopSpec { self.spec }

    pub fn mode(&self, layer: usize) -> Result<LayerMode, LayerPlanError> {
        self.modes.get(layer).copied().ok_or(LayerPlanError::Layer(layer))
    }

    /// Whole attention's weights, KV, indexer and private state exist once,
    /// independent of whether its FFN is split. Split modes retain both ranks;
    /// context page parity is still handled by the context adapter.
    pub fn attention_owners(&self, layer: usize) -> Result<Vec<u8>, LayerPlanError> {
        Ok(match self.mode(layer)? {
            LayerMode::Whole { gpu, .. } => vec![gpu],
            LayerMode::HeadSplit | LayerMode::ContextSplit => vec![0, 1],
        })
    }

    pub fn validate_state_owners(&self, layer: usize, actual: &[u8]) -> Result<(), LayerPlanError> {
        let expected = self.attention_owners(layer)?;
        let mut actual = actual.to_vec();
        actual.sort_unstable();
        if actual != expected { return Err(LayerPlanError::StateOwners { layer, actual, expected }); }
        Ok(())
    }
}

/// Persistent transport allocations, shared by admission and HopLink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HopRankGeometry {
    /// Each slot is the maximum width of hops that reuse its parity index.
    pub receive_slot_bytes: Vec<u64>,
    pub sender_lane_bytes: u64,
    pub control_bytes: u64,
    pub lanes: u64,
}

impl HopRankGeometry {
    pub fn receive_bytes(&self) -> Option<u64> {
        self.receive_slot_bytes.iter().try_fold(0u64, |n, bytes| n.checked_add(*bytes))?
            .checked_mul(self.lanes)
    }

    pub fn fixed_bytes(&self) -> Option<u64> {
        self.receive_bytes()?.checked_add(self.sender_lane_bytes.checked_mul(self.lanes)?)?
            .checked_add(self.control_bytes)
    }
}

pub fn hop_storage_geometry(hops: &[Hop], spec: &HopSpec, gpus: usize) -> Option<Vec<HopRankGeometry>> {
    if hops.iter().any(|hop| usize::from(hop.from) >= gpus || usize::from(hop.to) >= gpus
        || hop.from == hop.to || hop.row_bytes != spec.row_bytes_at(hop.at)
        || hop.row_bytes % 16 != 0) { return None; }
    // Zero extents remain the planner's explicit no-storage contract. The
    // executable LayerExecutionPlan and HopLink require positive geometry.
    let active = spec.rows > 0 && spec.lanes > 0 && hops.iter().any(|hop| hop.charged() && hop.row_bytes > 0);
    let control_bytes = if active {
        spec.lanes.checked_mul(HOP_SLOTS)?.checked_add(1)?.checked_mul(16)?.max(256)
    } else { 0 };
    (0..gpus).map(|gpu| {
        let incoming: Vec<_> = hops.iter().filter(|hop| active && hop.charged() && usize::from(hop.to) == gpu).collect();
        let mut slots = vec![0; incoming.len().min(HOP_SLOTS as usize)];
        for (ordinal, hop) in incoming.iter().enumerate() {
            let slot = ordinal % slots.len();
            slots[slot] = slots[slot].max(spec.rows.checked_mul(hop.row_bytes)?.max(256));
        }
        let sender = if active && spec.packed_sender {
            hops.iter().filter(|hop| hop.charged() && usize::from(hop.from) == gpu)
                .map(|hop| hop.row_bytes).max().unwrap_or(0)
        } else { 0 };
        Some(HopRankGeometry { receive_slot_bytes: slots,
            sender_lane_bytes: if sender > 0 { spec.rows.checked_mul(sender)?.max(256) } else { 0 },
            control_bytes, lanes: spec.lanes })
    }).collect()
}

/// Receive bytes only; useful for validating landed payload storage.
pub fn hop_buffer_bytes(hops: &[Hop], spec: &HopSpec, gpus: usize) -> Option<Vec<u64>> {
    hop_storage_geometry(hops, spec, gpus)?.iter().map(HopRankGeometry::receive_bytes).collect()
}

/// Full solver charge: receive slots, persistent packing senders and controls.
pub fn hop_storage_bytes(hops: &[Hop], spec: &HopSpec, gpus: usize) -> Option<Vec<u64>> {
    hop_storage_geometry(hops, spec, gpus)?.iter().map(HopRankGeometry::fixed_bytes).collect()
}
