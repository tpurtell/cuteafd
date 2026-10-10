//! Per-layer ownership (PLAN "v3 placement: design", section 3): where the
//! residual lives at each layer boundary, and the hops that move it when the
//! owner changes. The solver derives [`Hop`]s from the chosen [`LayerMode`]s
//! and charges their receive buffers as fixed demands; executors follow the
//! same list (the daemon's hop primitive, `shared::peer_split::HopLink`).
use super::{FfnMode, LayerMode};
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

/// What the family's executor moves per hop; part of the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HopSpec {
    /// Bytes per row of one hop (0: the family has no hops to charge).
    pub row_bytes: u64,
    /// Rows a hop carries at most (the larger of prefill and decode rows).
    pub rows: u64,
    /// Lanes that can have a hop in flight at once (prefill lanes).
    pub lanes: u64,
    /// The GPU the embedding rows land on before layer 0.
    pub entry_gpu: u8,
    /// The GPU the head runs on after the last layer.
    pub head_gpu: u8,
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
            LayerMode::HeadSplit => Transition {
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
pub fn plan_hops(modes: &[LayerMode], spec: &HopSpec) -> Vec<Hop> {
    let hop = |at, (from, to, kind)| Hop { at, from, to, kind, row_bytes: spec.row_bytes };
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
        home = step.home;
    }
    if !home.holds(spec.head_gpu) {
        if let ResidualHome::Owned(owner) = home {
            hops.push(hop(HopPoint::Exit, (owner, spec.head_gpu, HopKind::Boundary)));
        }
    }
    hops
}

/// Receive-buffer bytes `hops` need on each of `gpus` GPUs: `lanes` x
/// [`HOP_SLOTS`] (or fewer, when fewer hops land there) x `rows` x the row
/// bytes, for every charged hop into that GPU.
pub fn hop_buffer_bytes(hops: &[Hop], spec: &HopSpec, gpus: usize) -> Option<Vec<u64>> {
    (0..gpus).map(|gpu| {
        let into = hops.iter().filter(|h| h.charged() && usize::from(h.to) == gpu).count() as u64;
        let slots = into.min(HOP_SLOTS);
        spec.lanes.checked_mul(slots)?.checked_mul(spec.rows)?.checked_mul(spec.row_bytes)
    }).collect()
}
